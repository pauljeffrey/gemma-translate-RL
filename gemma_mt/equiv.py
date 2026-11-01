# SPDX-License-Identifier: AGPL-3.0-or-later
"""Equivalence helpers for comparing attention / loss implementations of the SAME weights on the SAME packs (D-35, D-36).

Used by tests/test_varlen_attention.py, tests/test_fused_loss.py and scripts/varlen_equivalence.py. Everything here
works on torch tensors that are already on the device."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from gemma_mt import attention as attn


def run_path_kw(model, batch, **kw):
    """`run_path` with attention kernel options (flex_attention)."""
    return run_path(model, batch, **kw)


def run_path(model, batch: Dict[str, Any], *, impl: str, mode: str, autocast: Optional[torch.dtype] = None,
             logits: bool = False, backward: bool = True, chunk: int = 512, loss_impl: str = "chunked",
             kernel_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One forward (+ backward) of `model` on `batch` with attention path (impl, mode).

    Returns {"nll", "n", "logp" [n] fp32 (target-token log-probs in row-major order of the labelled positions),
    "logits" [tokens, V] fp32 for the non-padding tokens (tiny models only) or None, "grads" {name: tensor} (the loss is
    the SUM of -logp), "gnorm"}. The batch's labels decide which positions are scored; padded and flattened batches of the
    same packs therefore give lists in the same order."""
    from gemma_mt.losses import final_hidden, sft_selected_logp

    if impl == attn.VARLEN:
        attn.register_varlen_attention()
    if hasattr(model, "set_attn_implementation"):  # transformers >= 5
        model.set_attn_implementation(impl)
    else:
        model.config._attn_implementation = impl
    sw = getattr(model.config, "sliding_window", None)
    kw = attn.model_inputs(batch, attn_implementation=impl, mask_mode=mode, sliding_window=sw, kernel_options=kernel_options)
    dev = batch["input_ids"].device.type
    model.zero_grad(set_to_none=True)
    ctx = torch.autocast(dev, dtype=autocast) if autocast is not None else torch.autocast(dev, enabled=False)
    out: Dict[str, Any] = {"logits": None}
    with ctx:
        if logits:
            lg = model(**kw).logits.float()
            valid = (batch["segment_ids"] > 0) if mode != "flash" else torch.ones_like(batch["input_ids"], dtype=torch.bool)
            out["logits"] = lg[valid]
        logp, sel = sft_selected_logp(model, kw, batch["labels"], chunk, loss_impl=loss_impl)
    out["logp"] = logp.detach().float()
    nll = -logp.sum()
    out["nll"], out["n"] = float(nll.detach()), int(sel.sum())
    out["grads"], out["gnorm"] = {}, float("nan")
    if backward:
        nll.backward()
        sq = 0.0
        out["grads"] = {}
        for k, p in model.named_parameters():
            if p.grad is not None:
                sq += float(torch.linalg.vector_norm(p.grad.detach().float())) ** 2
                out["grads"][k] = p.grad.detach().float().cpu()          # host copy: a 262k-vocab embedding gradient is 0.7 GB
        out["gnorm"] = sq ** 0.5
    return out


def grad_diff(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Global relative L2 difference and the worst per-tensor max-abs difference relative to that tensor's max-abs."""
    num = den = 0.0
    worst = 0.0
    for k in a:
        d = (a[k] - b[k].to(a[k].device)).double()
        num += float((d ** 2).sum())
        den += float((a[k].double() ** 2).sum())
        worst = max(worst, float(d.abs().max() / a[k].double().abs().max().clamp_min(1e-30)))
    return {"rel_l2": (num / max(den, 1e-300)) ** 0.5, "worst_tensor_rel_max": worst}


def compare(ref: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, float]:
    """Differences of `other` from `ref`: logits max/mean abs (if both have them), per-token log-prob max/mean abs,
    relative total-loss difference, gradient global relative L2 and worst per-tensor relative max."""
    d: Dict[str, float] = {}
    if ref["logits"] is not None and other["logits"] is not None:
        e = (ref["logits"] - other["logits"]).abs()
        d["logits_max_abs"], d["logits_mean_abs"] = float(e.max().detach()), float(e.mean().detach())
    e = (ref["logp"] - other["logp"]).abs()
    d["logp_max_abs"], d["logp_mean_abs"] = float(e.max().detach()), float(e.mean().detach())
    d["loss_rel"] = abs(ref["nll"] - other["nll"]) / max(abs(ref["nll"]), 1e-12)
    if ref["grads"] and other["grads"]:
        d.update(grad_diff(ref["grads"], other["grads"]))
        d["gnorm_rel"] = abs(ref["gnorm"] - other["gnorm"]) / max(ref["gnorm"], 1e-30)
    return d
