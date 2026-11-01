# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage 1: supervised fine-tuning (full fine-tune) of Gemma 3 270M IT on packed English -> African translation pairs.

    python -m gemma_mt.train_sft --config configs/sft.yaml                      # single GPU
    torchrun --nproc_per_node=2 -m gemma_mt.train_sft --config configs/sft.yaml # several GPUs (DDP)
    python -m gemma_mt.train_sft --smoke                                        # CPU, tiny random model, seconds

Loss (the one formula, see DECISIONS.md D-10):

    L = ( sum over every labelled token of every micro-batch of every rank in the optimizer step of  -log p )
        / ( number of labelled tokens in that whole step )

i.e. a TOKEN-weighted mean over the accumulated batch, not a mean of micro-batch means. Micro-batches contain
different numbers of target tokens (packing), and a mean of means would over-weight tokens of sparse batches.
DDP averages gradients over ranks, so each rank multiplies its summed loss by `world / N_global`.

Exact resume: `last/trainer_state.pt` holds model, optimizer, step, RNG states, best/patience bookkeeping. The data
order is a pure function of (seed, step) (gemma_mt.data.packed_dataset.batch_indices), so nothing else is needed.

Schedule file (`schedule_file: <file.npy>`, 2026-10-28, pre-registered clean rerun arms A0 / S1 / S2): instead of its own sampler the trainer
takes the packs of every optimizer step from an int64 array of shape [steps, packs_per_step] (row = one optimizer step, in draw order; a flat
1-D array whose length is a multiple of packs_per_step is also accepted). packs_per_step must equal micro_batch_size x grad_accum_steps x world
(row-major reshape [grad_accum, world, micro_batch] as for the internal samplers). The file length IS the run length: total steps, warm-up
fraction and the cosine LR follow it. A sidecar `<file>.json` (written by data_pipeline/sft_clean/chain/sampler.py) pins the schedule's sha256 and
the packed dataset it was drawn from; any mismatch, any index outside the pack set, or a missing sidecar stops the run before the model is
loaded. sha256, length, per-language draws and token counts are printed and saved to `<out_dir>/schedule_manifest.json`.

Embeddings-only warm-up (`embed_only_steps`, for the S2 arm with new tokens): during the first N steps of the schedule only the (tied)
embedding matrix is updated (optionally only rows >= `embed_only_min_row`); the steps are part of the schedule, not extra steps.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from gemma_mt import attention as attn
from gemma_mt import format as fmt
from gemma_mt.data.collate import PackedCollator
from gemma_mt.data.packed_dataset import PackedDataset, balanced_schedule, batch_indices, steps_per_epoch, unimax_schedule
from gemma_mt.languages import ALL_ISOS
from gemma_mt.utils import (Dist, MetricsLogger, dataclass_from_dict, get_rng_state, load_yaml, lr_at,
                            parse_overrides, set_rng_state, set_seed)


@dataclass
class SFTConfig:
    model_name: str = "google/gemma-3-270m-it"
    tokenizer_name: Optional[str] = None            # default: model_name
    data_dir: str = "data/packed"
    out_dir: str = "runs/sft"
    attn_implementation: str = "sdpa"               # sdpa | eager | flash_attention_2 | varlen (torch FlashAttention varlen kernel, D-35) | flex_attention
    mask_mode: str = "auto"                         # auto | explicit | position_ids | flash   (auto: explicit for sdpa/eager, position_ids for flex_attention, flash = flattened [1, N] stream + cu_seqlens for varlen/flash_attention_2)
    flex_kernel_options: Optional[Dict[str, int]] = None   # flex_attention only: Triton tile options, e.g. {BLOCK_M: 32, BLOCK_N: 32, ...} (head_dim 256 does not fit sm_120 shared memory otherwise)
    loss_impl: str = "chunked"                      # chunked (reference) | fused (Triton log-sum-exp + fp32 weight-grad accumulation, D-36)
    param_dtype: str = "float32"                    # master weights
    autocast_dtype: str = "bfloat16"                # "none" disables autocast
    micro_batch_size: int = 8                       # packs per forward per rank
    grad_accum_steps: int = 8
    epochs: float = 3.0
    max_steps: int = 0                              # >0 caps the run
    learning_rate: float = 5.0e-5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 0                           # 0 -> warmup_ratio * total_steps
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_max_packs: int = 0                         # 0 = the whole validation split
    early_stopping_patience: int = 5                # evaluations without improvement; 0 disables
    early_stopping_min_delta: float = 0.0
    min_epochs: float = 1.0                         # early stopping cannot fire before this many epochs
    early_stop_action: str = "stop"                 # stop | propose: "propose" never ends the run by itself; when patience is exhausted it
                                                    # writes <out_dir>/STOP_PROPOSED (step, best, bad evals) and keeps training until the owner decides
    save_every: int = 500
    log_every: int = 10
    gradient_checkpointing: bool = False
    loss_chunk_tokens: int = 512
    seed: int = 1234
    wandb_project: Optional[str] = None
    run_name: Optional[str] = None
    device: Optional[str] = None
    resume: bool = True
    save_optimizer: bool = True                     # False: the `last` checkpoint holds only weights + step + RNG + stopper bookkeeping (~1.1 GB instead of ~2.7 GB with Muon)
    resume_optimizer: bool = True                   # False: a resume loads the weights/step/RNG but starts a FRESH optimizer (also automatic when the checkpoint has none)
    optimizer_rewarmup_steps: int = 20              # after a resume WITHOUT optimizer state the LR ramps linearly over this many steps (0 = no ramp)
    stop_after_steps: int = 0                       # >0: end the run (with a checkpoint) after this many steps,
                                                    # WITHOUT changing the LR schedule (time-boxed runs, resume tests)
    # language-stratified sampling (needs data built with prepare_sft --pack-by-language). None = the old uniform pack permutation.
    language_alpha: Optional[float] = None          # p_lang ~ packs ** alpha: 1.0 natural frequency, 0.5 square root, 0.0 uniform
    min_language_share: float = 0.002               # floor on every language's share of the draws (each language shows up in every interval)
    max_language_epochs: float = 4.0                # a language used this many times is retired from the mixture (0 = no cap)
    val_metric: str = "macro"                       # macro: mean of the per-language losses (stopping + best checkpoint) | micro: token-weighted
    min_val_tokens_macro: int = 2000                # a language needs this many validation target tokens to enter the macro mean
    # Hub: push the current weights every `hub_push_every` steps (0 = at every evaluation) and tag it step-<N>
    hub_repo_id: Optional[str] = None               # None = never push (owner must confirm the repo name first)
    hub_push_every: int = 0
    hub_private: bool = True
    quick_eval_every: int = 0                       # >0: every this many steps evaluate a FIXED small validation subset (quick_eval_packs_per_language
                                                    # packs of every language) and log/print the per-language loss; never used for stopping
    quick_eval_packs_per_language: int = 1
    hub_push_best_only: bool = False                # push ONLY when the monitored validation loss improved (the Hub model is always the best one)
    keep_prev_checkpoint: bool = False              # also keep the PREVIOUS resumable checkpoint (<out_dir>/last_prev/, a hard link; no best/ or final/ copies)
    keep_single_checkpoint: bool = False            # local disk: only `last/` (resumable); no best/ or final/ copies, the staged push copy is deleted after upload
    # optimizer: adamw | muon (Muon on the 2-D decoder matrices, AdamW on the embedding / norms; one learning rate, update RMS matched to AdamW)
    optimizer: str = "adamw"
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    # start a NEW phase from the weights of an earlier run's checkpoint (model only: fresh optimizer, fresh LR schedule, step counter continues)
    init_from: Optional[str] = None                 # path to <run>/last/trainer_state.pt
    # UniMax token budget (replaces language_alpha sampling and `epochs`): per-language draws = clip(lambda * w, min_epochs * n, max_epochs * n)
    unimax_budget_tokens: float = 0.0               # >0 enables; total steps of this phase = budget / (packs per step * seq_len)
    unimax_min_epochs: float = 1.0                  # every pack of every language is used at least this many times
    unimax_max_epochs: float = 2.0
    unimax_priority_languages: List[str] = field(default_factory=list)
    unimax_priority_weight: float = 1.0             # water-level multiplier of the priority languages
    # packs the previous phase already used come LAST in each language's first pass. The previous phase's schedule is a pure function of its
    # sampling parameters and seed, so it is regenerated here (steps < the init step) instead of being stored.
    unimax_unseen_first: bool = False
    prior_language_alpha: float = 0.5
    prior_min_language_share: float = 0.002
    prior_max_language_epochs: float = 4.0
    prior_total_steps: int = 0                      # total_steps of the previous phase (its schedule length)
    # ---- D-58 (short science-SFT stage on top of an RL-trained model): drift control + extra validation sets. All off by default (the earlier runs are unchanged).
    l2sp_coef: float = 0.0                          # >0: L2-SP (Li et al. 2018) pull towards the STARTING weights: grad += coef * (theta - theta_start), added AFTER gradient clipping
                                                    # (so clipping does not weaken it); with Adam the pull is normalised like any gradient. weight_decay is NOT this: decoupled weight
                                                    # decay shrinks towards ZERO, not towards the starting point.
    track_drift: bool = False                       # log ||theta - theta_start|| (absolute, relative, embedding vs layers) every log_every steps (keeps a copy of the starting weights)
    extra_val_splits: List[str] = field(default_factory=list)   # more validation splits (sub-directories of data_dir, same format as val/); logged as val_<name>_*, never used for stopping
    eval_at_start: bool = False                     # evaluate every validation split once before the first optimizer step (a fresh start only): the baseline of the run
    # ---- schedule file (pre-registered arms A0 / S1 / S2): the packs of every step come from a file, not from the internal sampler. None = unchanged behaviour.
    schedule_file: Optional[str] = None             # .npy int64 [steps, micro_batch_size * grad_accum_steps * world]; its length is the run length (epochs / max_steps are ignored; max_steps must be 0 or equal)
    schedule_sidecar: Optional[str] = None          # default: <schedule_file without .npy>.json, else meta.json in the same folder (sampler.py plan output)
    schedule_require_sidecar: bool = True           # False: run without the sidecar checks (the sha256 is still logged); never for a pre-registered run
    # embeddings-only warm-up INSIDE the schedule (S2: the 3,000 new rows have a mean-of-subtokens init and nothing else should move yet)
    embed_only_steps: int = 0                       # first N steps of the run update only the (tied) embedding matrix: the other gradients are dropped before the clip / optimizer
    embed_only_min_row: int = 0                     # >0: inside the warm-up only embedding rows >= this id receive an update (S2: 256001 = the new ids only)
    # ---- growth 1 (2026-11-01, staged unfreezing; all off by default = unchanged behaviour)
    freeze_regex: Optional[str] = None              # parameters whose NAME matches (re.search) get requires_grad=False before the optimizer is built (no gradient, no momentum, no weight decay)
    train_embedding_rows: List[int] = field(default_factory=list)   # non-empty: of the (tied) embedding matrix ONLY these rows are updated (the gradient of every other row is zeroed before the clip;
                                                    # the embedding is on the no-weight-decay AdamW group, so the other rows cannot drift)
    train_embedding_tags: bool = False              # adds the ids of the 125 dedicated language-tag tokens (gemma_mt.tags / growth.tag_token_ids) to train_embedding_rows


class EarlyStopper:
    """Early stopping on validation loss with a minimum number of epochs.

    update(step, val_loss) -> (improved, stop). An evaluation counts as "bad" only when it does not beat the best
    loss by `min_delta` AND `step >= min_step` (the end of epoch `min_epochs`); evaluations before that never
    accumulate patience. `patience == 0` disables stopping (the best checkpoint is still tracked)."""

    def __init__(self, patience: int, min_delta: float, min_step: int):
        self.patience, self.min_delta, self.min_step = patience, min_delta, min_step
        self.best, self.best_step, self.bad = float("inf"), -1, 0

    def update(self, step: int, val_loss: float):
        improved = val_loss < self.best - self.min_delta
        if improved:
            self.best, self.best_step, self.bad = val_loss, step, 0
        elif step >= self.min_step:
            self.bad += 1
        stop = bool(self.patience) and self.bad >= self.patience and step >= self.min_step
        return improved, stop

    def state(self) -> Dict[str, Any]:
        return {"best": self.best, "best_step": self.best_step, "bad": self.bad}

    def load(self, st: Dict[str, Any]) -> None:
        self.best, self.best_step, self.bad = st["best"], st["best_step"], st["bad"]


def _dtype(name: str):
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


ATTN_IMPLEMENTATIONS = ("sdpa", "eager", "flash_attention_2", "varlen", "flex_attention")


def validate_attention_config(cfg: SFTConfig) -> None:
    """Fail fast on an inconsistent (attn_implementation, mask_mode, loss_impl) triple."""
    from gemma_mt.losses import LOSS_IMPLS

    if cfg.attn_implementation not in ATTN_IMPLEMENTATIONS:
        raise ValueError(f"attn_implementation must be one of {ATTN_IMPLEMENTATIONS}, got {cfg.attn_implementation!r}")
    if cfg.loss_impl not in LOSS_IMPLS:
        raise ValueError(f"loss_impl must be one of {LOSS_IMPLS}, got {cfg.loss_impl!r}")
    mode = resolve_mask_mode(cfg)
    if cfg.attn_implementation in attn.FLASH_IMPLS and mode != "flash":
        raise ValueError(f"attn_implementation={cfg.attn_implementation} needs mask_mode 'flash' (or 'auto'), got {mode!r}")
    if cfg.attn_implementation not in attn.FLASH_IMPLS and mode == "flash":
        raise ValueError(f"mask_mode='flash' needs attn_implementation in {attn.FLASH_IMPLS}, got {cfg.attn_implementation!r}")
    if cfg.attn_implementation == "flex_attention" and mode != "position_ids":
        raise ValueError(f"attn_implementation=flex_attention needs mask_mode 'position_ids' (or 'auto'), got {mode!r}")
    if cfg.flex_kernel_options and cfg.attn_implementation != "flex_attention":
        raise ValueError("flex_kernel_options only applies to attn_implementation='flex_attention'")


def resolve_mask_mode(cfg: SFTConfig) -> str:
    if cfg.mask_mode != "auto":
        return cfg.mask_mode
    return attn.default_mask_mode(cfg.attn_implementation)


class CombinedOptimizer:
    """Several torch optimizers behind the one interface the loop uses (param_groups, zero_grad, step, state_dict)."""

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    @property
    def param_groups(self):
        return [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return {"combined": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, sd):
        if "combined" not in sd:
            raise ValueError("checkpoint holds a single-optimizer state (AdamW); use init_from to start Muon from its weights")
        for o, st in zip(self.optimizers, sd["combined"]):
            o.load_state_dict(st)


def build_optimizer(model, cfg: SFTConfig):
    import torch

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # weight decay on matrices only; not on norms, biases, nor on the (tied) embedding / LM head
        (decay if p.ndim >= 2 and "embed_tokens" not in name else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    use_fused = torch.cuda.is_available() and next(model.parameters()).is_cuda
    adamw_kw = dict(lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, fused=True if use_fused else None)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, **adamw_kw)
    if cfg.optimizer != "muon":
        raise ValueError(f"unknown optimizer {cfg.optimizer!r} (adamw | muon)")
    # Muon orthogonalises the update of a 2-D weight matrix: only the decoder-layer matrices; the (tied) embedding / LM head, norms and
    # any 1-D parameter stay on AdamW. adjust_lr_fn='match_rms_adamw' scales the update to AdamW's RMS so one lr / weight decay serves both.
    muon_p = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim == 2 and ".layers." in n]
    ids = {id(p) for p in muon_p}
    adam_dec = [p for p in decay if id(p) not in ids]
    adam_groups = [g for g in ({"params": adam_dec, "weight_decay": cfg.weight_decay},
                               {"params": no_decay, "weight_decay": 0.0}) if g["params"]]
    muon = torch.optim.Muon([{"params": muon_p, "weight_decay": cfg.weight_decay}], lr=cfg.learning_rate,
                            momentum=cfg.muon_momentum, ns_steps=cfg.muon_ns_steps, adjust_lr_fn="match_rms_adamw")
    return CombinedOptimizer([muon, torch.optim.AdamW(adam_groups, **adamw_kw)])


def _to_device(batch: Dict[str, Any], device) -> Dict[str, Any]:
    import torch

    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
            out[k] = t.to(device)
        else:
            out[k] = v
    return out


def _autocast(cfg: SFTConfig, device):
    import torch

    if cfg.autocast_dtype in (None, "none", "None"):
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=_dtype(cfg.autocast_dtype))


def make_loss_module(model, cfg: SFTConfig, sliding_window, mode: str):
    """nn.Module whose forward returns the summed NLL of a batch. DDP wraps THIS module (not the bare model) so that
    the gradient-sync hooks see the whole forward, including the chunked loss."""
    import torch

    from gemma_mt.losses import sft_nll_sum

    class _SFTLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, b):
            kw = attn.model_inputs(b, attn_implementation=cfg.attn_implementation, mask_mode=mode,
                                   sliding_window=sliding_window, kernel_options=cfg.flex_kernel_options)
            with _autocast(cfg, b["input_ids"].device):
                nll, _ = sft_nll_sum(self.model, kw, b["labels"], chunk=cfg.loss_chunk_tokens, loss_impl=cfg.loss_impl)
            return nll

    return _SFTLoss()


def evaluate(model, ds: PackedDataset, collator, cfg: SFTConfig, dist: Dist, sliding_window, mode: str,
             pack_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """Validation loss over TARGET tokens only (token-weighted), overall and per language."""
    import torch

    from gemma_mt.losses import sft_selected_logp

    was_training = model.training
    model.eval()
    n_packs = len(ds) if cfg.eval_max_packs <= 0 else min(len(ds), cfg.eval_max_packs)
    if pack_ids is not None:
        n_packs = len(pack_ids)
        mine = [int(i) for i in pack_ids[dist.rank::dist.world]]
    else:
        mine = list(range(dist.rank, n_packs, dist.world))
    n_l = len(ALL_ISOS)
    nll = torch.zeros(2 * n_l, dtype=torch.float64, device=dist.device)      # [0, n_l) en2x, [n_l, 2 n_l) x2en
    cnt = torch.zeros(2 * n_l, dtype=torch.float64, device=dist.device)
    with torch.no_grad():
        for s in range(0, len(mine), cfg.micro_batch_size):
            batch = collator([ds[i] for i in mine[s: s + cfg.micro_batch_size]])
            b = _to_device(batch, dist.device)
            kw = attn.model_inputs(b, attn_implementation=cfg.attn_implementation, mask_mode=mode,
                                   sliding_window=sliding_window, kernel_options=cfg.flex_kernel_options)
            with _autocast(cfg, dist.device):
                logp, sel = sft_selected_logp(model, kw, b["labels"], cfg.loss_chunk_tokens, loss_impl=cfg.loss_impl)
            if logp.numel() == 0:
                continue
            iso = b["token_iso"][:, 1:][sel].long()
            assert bool((iso >= 0).all()), "a labelled token has no language id"
            nll.index_add_(0, iso, (-logp).double())
            cnt.index_add_(0, iso, torch.ones_like(logp, dtype=torch.float64))
    if dist.world > 1:
        import torch.distributed as td

        td.all_reduce(nll)
        td.all_reduce(cnt)
    total_n = float(cnt.sum().item())
    out = {"val_loss": float(nll.sum().item() / total_n) if total_n > 0 else float("nan"), "val_tokens": int(total_n),
           "val_packs": n_packs}
    per, per_rev = {}, {}
    for i, iso in enumerate(ALL_ISOS):
        if cnt[i] > 0:
            per[iso] = float(nll[i].item() / cnt[i].item())
        if cnt[n_l + i] > 0:
            per_rev[iso] = float(nll[n_l + i].item() / cnt[n_l + i].item())
    out["val_loss_by_language"] = per                       # English -> language
    ok = [v for i, (iso, v) in enumerate(per.items()) if cnt[ALL_ISOS.index(iso)] >= cfg.min_val_tokens_macro]
    out["val_loss_macro"] = float(sum(ok) / len(ok)) if ok else out["val_loss"]
    out["val_macro_languages"] = len(ok)
    if per_rev:
        out["val_loss_by_language_x2en"] = per_rev          # language -> English (bidirectional datasets only)
        for name, sl in (("en2x", slice(0, n_l)), ("x2en", slice(n_l, 2 * n_l))):
            c = float(cnt[sl].sum().item())
            if c > 0:
                out[f"val_loss_{name}"] = float(nll[sl].sum().item() / c)
    if was_training:
        model.train()
    return out


def _save_state(path: str, model, optimizer, state: Dict[str, Any], keep_prev: bool = False) -> None:
    import torch

    inner = getattr(model, "module", model)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    payload = {"model": inner.state_dict(), "rng": get_rng_state(), "state": state}
    if optimizer is not None:                                     # None = weights-only checkpoint (cfg.save_optimizer false)
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, tmp)
    if keep_prev and os.path.exists(path):
        # the previous checkpoint survives as <dir>_prev/trainer_state.pt: a HARD LINK to the old inode, so `last` is never missing and the
        # extra copy costs no write (the old inode lives on after the atomic replace below)
        prev_dir = os.path.join(os.path.dirname(os.path.dirname(path)), os.path.basename(os.path.dirname(path)) + "_prev")
        os.makedirs(prev_dir, exist_ok=True)
        link = os.path.join(prev_dir, "trainer_state.pt.link")
        if os.path.exists(link):
            os.remove(link)
        os.link(path, link)
        os.replace(link, os.path.join(prev_dir, "trainer_state.pt"))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------- schedule file (A0 / S1 / S2)
SCHEDULE_FORMAT = "sabiyarn-schedule-1"


def packed_structure_sha256(ds: PackedDataset) -> str:
    """Hash of WHICH samples form WHICH pack (pack_start, iso_id, direction), independent of the token ids: A0's packs and the S2 replay
    (same packs re-tokenised) have the same value, a different packing does not. sampler.py computes the same value (a test pins the equality)."""
    h = hashlib.sha256()
    for name in ("pack_start", "iso_id", "direction"):
        a = np.ascontiguousarray(np.asarray(getattr(ds, name)), dtype=np.int64)
        h.update(f"{name}:{a.shape[0]}\n".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def pack_lang_sha256(pack_lang: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(pack_lang), dtype=np.int64).tobytes()).hexdigest()


def _sidecar_candidates(path: str, explicit: Optional[str]) -> List[str]:
    if explicit:
        return [explicit]
    stem = path[:-4] if path.endswith(".npy") else path
    return [stem + ".json", os.path.join(os.path.dirname(os.path.abspath(path)), "meta.json")]


def load_schedule(path: str, ds: PackedDataset, per_step: int, *, sidecar: Optional[str] = None,
                  require_sidecar: bool = True) -> "tuple[np.ndarray, Dict[str, Any]]":
    """Read a schedule file and check it against the packed train set. Returns (int64 [steps, per_step], info).

    Fails (ValueError / FileNotFoundError) on: a non-integer array, a shape that does not fit per_step, any index outside [0, len(ds)), a
    missing sidecar (unless require_sidecar is False), and any sidecar field that disagrees (sha256, steps, per_step, n_packs, language of
    every pack, pack structure). A sidecar of a projected plan (packs from a CSV table, not from a packed dataset) is refused."""
    arr = np.load(path, allow_pickle=False)
    if arr.dtype.kind not in "iu":
        raise ValueError(f"schedule {path}: dtype {arr.dtype} is not an integer type")
    if arr.ndim == 1:
        if arr.size == 0 or arr.size % per_step:
            raise ValueError(f"schedule {path}: {arr.size} draws are not a whole number of steps of {per_step} packs")
        arr = arr.reshape(-1, per_step)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != per_step:
        raise ValueError(f"schedule {path}: shape {tuple(arr.shape)} but the trainer takes {per_step} packs per step "
                         f"(micro_batch_size x grad_accum_steps x world); the schedule must be [steps, {per_step}]")
    n_packs = len(ds)
    lo, hi = int(arr.min()), int(arr.max())
    if lo < 0 or hi >= n_packs:
        bad = np.flatnonzero((arr.reshape(-1) < 0) | (arr.reshape(-1) >= n_packs))
        raise ValueError(f"schedule {path}: {bad.size} indices outside the pack set [0, {n_packs}) (min {lo}, max {hi}; first at "
                         f"step {int(bad[0]) // per_step}, slot {int(bad[0]) % per_step})")
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    sha = hashlib.sha256(arr.tobytes()).hexdigest()
    info: Dict[str, Any] = {"format": SCHEDULE_FORMAT, "file": os.path.abspath(path), "sha256": sha, "steps": int(arr.shape[0]),
                            "per_step": int(per_step), "n_packs": int(n_packs), "sidecar": None, "checks_passed": [], "checks_skipped": []}
    sc_path = next((c for c in _sidecar_candidates(path, sidecar) if os.path.exists(c)), None)
    if sc_path is None:
        if require_sidecar:
            raise FileNotFoundError(f"no sidecar for schedule {path} (looked for {_sidecar_candidates(path, sidecar)}); "
                                    "write one with sampler.py or set schedule_require_sidecar=false")
        info["checks_skipped"].append("sidecar (none found, schedule_require_sidecar=false)")
        return arr, info
    with open(sc_path, encoding="utf-8") as f:
        sc = json.load(f)
    info["sidecar"] = os.path.abspath(sc_path)
    info["sidecar_content"] = sc
    if sc.get("projected") or str(sc.get("source", "")).endswith(".csv"):
        raise ValueError(f"sidecar {sc_path} belongs to a PROJECTED plan (packs per language from a table, not a packed dataset): its indices mean nothing here")
    if sc.get("schedule_sha256") != sha:
        raise ValueError(f"schedule {path}: sha256 {sha} but the sidecar {sc_path} says {sc.get('schedule_sha256')}")
    info["checks_passed"].append("schedule_sha256")
    for key, have in (("steps", arr.shape[0]), ("per_step", per_step), ("n_packs", n_packs)):
        if key in sc and sc[key] is not None:
            if int(sc[key]) != int(have):
                raise ValueError(f"sidecar {sc_path}: {key} = {sc[key]} but the schedule / packed dataset has {have}")
            info["checks_passed"].append(key)
        else:
            info["checks_skipped"].append(key)
    if sc.get("pack_lang_sha256"):
        try:
            got = pack_lang_sha256(ds.pack_languages())
        except ValueError as e:
            raise ValueError(f"sidecar {sc_path} pins the language of every pack but this dataset has mixed-language packs ({e})") from e
        if got != sc["pack_lang_sha256"]:
            raise ValueError(f"the packed dataset's pack languages (sha256 {got[:16]}...) differ from the dataset the schedule was drawn on ({sc['pack_lang_sha256'][:16]}...)")
        info["checks_passed"].append("pack_lang_sha256")
    else:
        info["checks_skipped"].append("pack_lang_sha256")
    if sc.get("packed_structure_sha256"):
        got = packed_structure_sha256(ds)
        if got != sc["packed_structure_sha256"]:
            raise ValueError(f"the packed dataset's pack structure (sha256 {got[:16]}...) differs from the dataset the schedule was drawn on "
                             f"({sc['packed_structure_sha256'][:16]}...): other packs, other order, or other samples")
        info["checks_passed"].append("packed_structure_sha256")
    else:
        info["checks_skipped"].append("packed_structure_sha256")
    return arr, info


def schedule_manifest(arr: np.ndarray, ds: PackedDataset, info: Dict[str, Any], *, run_start: int = 0, embed_only_steps: int = 0,
                      embed_only_min_row: int = 0) -> Dict[str, Any]:
    """Everything the launch manifest needs about a schedule: hash, length, per-language draws / passes / tokens, the checks that passed."""
    flat = arr.reshape(-1)
    seq_len = int(ds.seq_len)
    out = {k: v for k, v in info.items() if k != "sidecar_content"}
    out.update({"draws": int(flat.size), "unique_packs_drawn": int(np.unique(flat).size), "seq_len": seq_len, "padded_tokens": int(flat.size) * seq_len,
                "tokens_per_step_padded": int(arr.shape[1]) * seq_len, "run_start": int(run_start),
                "embed_only_steps": int(embed_only_steps), "embed_only_min_row": int(embed_only_min_row)})
    sl = np.asarray(ds.sample_len, dtype=np.int64)
    tl = np.asarray(ds.target_len, dtype=np.int64)
    ps = np.asarray(ds.pack_start, dtype=np.int64)
    cs_len = np.concatenate([[0], np.cumsum(sl)])
    cs_tgt = np.concatenate([[0], np.cumsum(tl)])
    pack_tokens = cs_len[ps[1:]] - cs_len[ps[:-1]]
    pack_target = cs_tgt[ps[1:]] - cs_tgt[ps[:-1]]
    out["real_tokens"] = int(pack_tokens[flat].sum())
    out["target_tokens"] = int(pack_target[flat].sum())
    try:
        lang = ds.pack_languages()
    except ValueError:
        out["draws_per_language"] = None
        out["note"] = "packs mix languages: no per-language draw counts"
        return out
    n_l = len(ALL_ISOS)
    drawn = np.bincount(lang[flat], minlength=n_l)
    have = np.bincount(lang, minlength=n_l)
    rt = np.bincount(lang[flat], weights=pack_tokens[flat], minlength=n_l)
    tt = np.bincount(lang[flat], weights=pack_target[flat], minlength=n_l)
    out["draws_per_language"] = {ALL_ISOS[i]: {"packs": int(have[i]), "draws": int(drawn[i]), "passes": round(float(drawn[i]) / max(1, int(have[i])), 4),
                                               "real_tokens": int(rt[i]), "target_tokens": int(tt[i])} for i in range(n_l) if have[i] or drawn[i]}
    return out


def embedding_param_ids(model) -> set:
    """ids of the parameters that belong to the embedding: the input matrix and, when it is not tied, the output matrix."""
    ids = set()
    for emb in (model.get_input_embeddings(), model.get_output_embeddings()):
        if emb is not None and getattr(emb, "weight", None) is not None:
            ids.add(id(emb.weight))
    return ids


def keep_only_embedding_grads(model, emb_ids: set, min_row: int = 0) -> None:
    """Embeddings-only step: drop every non-embedding gradient (None: the optimizers skip those parameters, weight decay and momentum included) and,
    with min_row > 0, zero the gradient of embedding rows below min_row. Called after backward (DDP-safe: no parameter changes requires_grad) and
    before the gradient clip, so the clip norm is the embedding gradient's alone."""
    for p in model.parameters():
        if id(p) not in emb_ids:
            p.grad = None
        elif min_row > 0 and p.grad is not None:
            p.grad[:min_row] = 0


def apply_freeze(model, pattern: Optional[str]) -> Dict[str, int]:
    """requires_grad=False for every parameter whose name matches `pattern` (re.search); returns the frozen / trainable parameter counts (the tied embedding is one parameter)."""
    import re

    rx = re.compile(pattern) if pattern else None
    frozen = trainable = 0
    for name, p in model.named_parameters():
        if rx is not None and rx.search(name):
            p.requires_grad_(False)
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    return {"frozen": frozen, "trainable": trainable}


def embedding_rows_to_train(cfg: SFTConfig, tokenizer: Any) -> List[int]:
    """Sorted, de-duplicated rows of cfg.train_embedding_rows plus (train_embedding_tags) the dedicated tag tokens; [] = the whole matrix."""
    rows = [int(r) for r in cfg.train_embedding_rows]
    if cfg.train_embedding_tags:
        from gemma_mt.growth import tag_token_ids

        rows += list(tag_token_ids(tokenizer).values())
    return sorted(set(rows))


def keep_only_embedding_rows(model, emb_ids: set, rows) -> None:
    """Zero the gradient of every embedding row not in `rows` (a LongTensor on the gradient's device). Called after backward and before the clip, so the clip norm is the rows' alone."""
    for p in model.parameters():
        if id(p) in emb_ids and p.grad is not None:
            keep = p.grad[rows].clone()
            p.grad.zero_()
            p.grad[rows] = keep


def train(cfg: SFTConfig, tokenizer: Any = None) -> Dict[str, Any]:
    import torch

    from gemma_mt.data.prepare_sft import tokenizer_fingerprint
    from gemma_mt.modeling import load_model, save_model, sliding_window_of

    dist = Dist(cfg.device)
    set_seed(cfg.seed)
    validate_attention_config(cfg)
    if cfg.attn_implementation == "flash_attention_2" and dist.device.type != "cuda":
        raise ValueError("flash_attention_2 needs a CUDA device")

    # ---- tokenizer + format checks (fail fast, before any GPU time is spent)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.model_name)
    fmt.assert_template_matches_reference(tokenizer)
    fingerprint = tokenizer_fingerprint(tokenizer)

    # ---- data
    mode = resolve_mask_mode(cfg)
    train_ds = PackedDataset(cfg.data_dir, "train", pad=(mode != "flash"), expect_tokenizer_fingerprint=fingerprint)
    val_ds = PackedDataset(cfg.data_dir, "val", pad=(mode != "flash"), expect_tokenizer_fingerprint=fingerprint)
    collator = PackedCollator(attn.collator_mode_for(cfg.attn_implementation), pad_id=train_ds.pad_id)
    extra_val = {name: PackedDataset(cfg.data_dir, name, pad=(mode != "flash"), expect_tokenizer_fingerprint=fingerprint) for name in cfg.extra_val_splits}
    quick_ids: Optional[List[int]] = None
    if cfg.quick_eval_every:
        try:
            vl = val_ds.pack_languages()
        except ValueError:
            vl = None
        if vl is not None and len(vl):
            quick_ids, seen = [], {}
            for pi, l in enumerate(vl.tolist()):
                if seen.get(l, 0) < cfg.quick_eval_packs_per_language:
                    quick_ids.append(pi)
                    seen[l] = seen.get(l, 0) + 1
    spe = steps_per_epoch(len(train_ds), cfg.micro_batch_size, cfg.grad_accum_steps, dist.world)
    if spe <= 0 and not cfg.schedule_file:
        raise ValueError(f"{len(train_ds)} train packs < micro_batch_size x grad_accum_steps x world "
                         f"({cfg.micro_batch_size} x {cfg.grad_accum_steps} x {dist.world})")
    spe = max(spe, 1)
    per_step_packs = cfg.micro_batch_size * cfg.grad_accum_steps * dist.world
    # ---- schedule file: validated here, before the model is loaded; its length is the length of the run
    sched_arr, sched_info, sched_manifest = None, None, None
    if cfg.schedule_file:
        if cfg.language_alpha is not None or cfg.unimax_budget_tokens > 0:
            raise ValueError("schedule_file replaces the internal samplers: language_alpha must be null and unimax_budget_tokens 0")
        sched_arr, sched_info = load_schedule(cfg.schedule_file, train_ds, per_step_packs, sidecar=cfg.schedule_sidecar,
                                              require_sidecar=cfg.schedule_require_sidecar)
        if cfg.max_steps > 0 and cfg.max_steps != sched_arr.shape[0]:
            raise ValueError(f"max_steps={cfg.max_steps} but the schedule has {sched_arr.shape[0]} steps: the file length is the run length "
                             "(use max_steps=0, or cut the schedule file); a different cap would change the LR schedule")
    if cfg.embed_only_steps < 0 or cfg.embed_only_min_row < 0:
        raise ValueError("embed_only_steps and embed_only_min_row must be >= 0")
    # a phase started with init_from continues the step counter of the run it came from (Hub tags, metrics) but has its own length / LR schedule
    run_start = 0
    if cfg.init_from:
        resume_ck = os.path.join(cfg.out_dir, "last", "trainer_state.pt")
        if cfg.resume and os.path.exists(resume_ck):
            run_start = int(torch.load(resume_ck, map_location="cpu", weights_only=False)["state"].get("run_start", 0))
        else:
            run_start = int(torch.load(cfg.init_from, map_location="cpu", weights_only=False)["state"]["step"])
    if sched_arr is not None:
        n_run = int(sched_arr.shape[0])
    elif cfg.unimax_budget_tokens > 0:
        seq_len = int(train_ds.meta["seq_len"])
        n_run = int(cfg.unimax_budget_tokens // (per_step_packs * seq_len))
    else:
        n_run = int(math.ceil(cfg.epochs * spe))
    if cfg.max_steps > 0:
        n_run = min(n_run, cfg.max_steps)
    if cfg.embed_only_steps > n_run:
        raise ValueError(f"embed_only_steps={cfg.embed_only_steps} is longer than the run ({n_run} steps)")
    total_steps = run_start + n_run
    warmup = cfg.warmup_steps or max(1, int(round(cfg.warmup_ratio * n_run)))
    min_epoch_step = run_start + int(math.ceil(cfg.min_epochs * spe))
    schedule = sched_arr
    if sched_arr is not None:
        sched_manifest = schedule_manifest(sched_arr, train_ds, sched_info, run_start=run_start, embed_only_steps=cfg.embed_only_steps,
                                           embed_only_min_row=cfg.embed_only_min_row)
        if dist.is_main:
            top = sorted((sched_manifest.get("draws_per_language") or {}).items(), key=lambda kv: -kv[1]["draws"])
            print(f"[gemma_mt] SCHEDULE FILE {cfg.schedule_file}: sha256 {sched_info['sha256']}, {n_run} steps x {per_step_packs} packs = "
                  f"{sched_manifest['draws']} draws ({sched_manifest['unique_packs_drawn']} distinct packs), "
                  f"{sched_manifest['padded_tokens'] / 1e9:.3f}B padded tokens ({sched_manifest['real_tokens'] / 1e9:.3f}B real, "
                  f"{sched_manifest['target_tokens'] / 1e9:.3f}B target); checks passed: {sched_info['checks_passed']}"
                  + (f"; NOT checked: {sched_info['checks_skipped']}" if sched_info["checks_skipped"] else ""), flush=True)
            print("  draws/passes per language: " + "  ".join(f"{k} {v['draws']}/{v['passes']:.2f}x" for k, v in top), flush=True)
    elif cfg.unimax_budget_tokens > 0:
        prio = {ALL_ISOS.index(i): cfg.unimax_priority_weight for i in cfg.unimax_priority_languages}
        seen = None
        if cfg.unimax_unseen_first and run_start > 0:
            prior, _ = balanced_schedule(train_ds.pack_languages(), cfg.prior_total_steps, per_step_packs, alpha=cfg.prior_language_alpha,
                                         min_share=cfg.prior_min_language_share, max_epochs=cfg.prior_max_language_epochs, seed=cfg.seed)
            seen = np.zeros(len(train_ds), dtype=bool)
            seen[prior[:run_start].ravel()] = True
            if dist.is_main:
                print(f"[gemma_mt] previous phase used {int(seen.sum())} of {len(train_ds)} train packs in its first {run_start} steps; "
                      f"unseen packs are drawn first within each language", flush=True)
        schedule, lang_stats = unimax_schedule(train_ds.pack_languages(), n_run, per_step_packs, min_epochs=cfg.unimax_min_epochs,
                                               max_epochs=cfg.unimax_max_epochs, weights=prio, seed=cfg.seed, seen=seen)
        if dist.is_main:
            rows = sorted(lang_stats.items(), key=lambda kv: -kv[1]["share"])
            print("[gemma_mt] UniMax schedule: %d steps = %.2fB tokens, epochs in [%s, %s], priority x%s on %d languages; "
                  "share/epochs of this phase:" % (n_run, n_run * per_step_packs * seq_len / 1e9, cfg.unimax_min_epochs,
                                                   cfg.unimax_max_epochs, cfg.unimax_priority_weight, len(prio)), flush=True)
            print("  " + "  ".join(f"{ALL_ISOS[i]} {v['share']*100:.2f}%/{v['epochs']:.2f}ep" for i, v in rows), flush=True)
    elif cfg.language_alpha is not None:
        per_step = cfg.micro_batch_size * cfg.grad_accum_steps * dist.world
        schedule, lang_stats = balanced_schedule(train_ds.pack_languages(), total_steps, per_step, alpha=cfg.language_alpha,
                                                 min_share=cfg.min_language_share, max_epochs=cfg.max_language_epochs, seed=cfg.seed)
        if dist.is_main:
            rows = sorted(lang_stats.items(), key=lambda kv: -kv[1]["share"])
            print("[gemma_mt] language-stratified sampling (alpha=%s, floor %.3f, cap %s epochs): %d languages; "
                  "share/effective epochs of the run:" % (cfg.language_alpha, cfg.min_language_share, cfg.max_language_epochs, len(rows)), flush=True)
            print("  " + "  ".join(f"{ALL_ISOS[i]} {v['share']*100:.2f}%/{v['epochs']:.1f}ep" for i, v in rows), flush=True)

    # ---- model
    model = load_model(cfg.model_name, attn_implementation=cfg.attn_implementation, dtype=_dtype(cfg.param_dtype),
                       device=str(dist.device))
    stop = fmt.resolve_stop_tokens(tokenizer, model.generation_config)
    for w in stop.warnings:
        print("[gemma_mt] WARNING:", w, flush=True)
    fmt.ensure_generation_config(model.generation_config, stop, pad_id=tokenizer.pad_token_id)
    sliding_window = sliding_window_of(model)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    # ---- growth 1: staged unfreezing (before the optimizer is built: frozen parameters get no gradient, no momentum, no weight decay)
    freeze_info = apply_freeze(model, cfg.freeze_regex)
    tr_rows = embedding_rows_to_train(cfg, tokenizer)
    tr_rows_t, tr_emb_ids = None, set()
    if tr_rows:
        emb_w = model.get_input_embeddings().weight
        if not emb_w.requires_grad:
            raise ValueError("train_embedding_rows / train_embedding_tags: the embedding matrix is frozen by freeze_regex")
        if tr_rows[0] < 0 or tr_rows[-1] >= emb_w.shape[0]:
            raise ValueError(f"train_embedding_rows: ids must lie in [0, {emb_w.shape[0]})")
        tr_rows_t = torch.tensor(tr_rows, dtype=torch.long, device=dist.device)
        tr_emb_ids = embedding_param_ids(model)
    if freeze_info["trainable"] == 0:
        raise ValueError(f"freeze_regex {cfg.freeze_regex!r} freezes every parameter")
    if (cfg.freeze_regex or tr_rows) and dist.is_main:
        print(f"[gemma_mt] GROWTH freezing: {freeze_info['trainable']:,} trainable / {freeze_info['frozen']:,} frozen parameters"
              + (f"; of the embedding matrix only {len(tr_rows)} rows are updated" if tr_rows else ""), flush=True)
    train_model = make_loss_module(model, cfg, sliding_window, mode)
    if dist.world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        train_model = DDP(train_model, device_ids=[dist.local_rank] if dist.device.type == "cuda" else None)
    optimizer = build_optimizer(model, cfg)

    # ---- reference weights for L2-SP / drift tracking: the weights this run STARTS from (model_name, or init_from when given)
    ref_names: List[str] = []
    ref_params: List[Any] = []

    def capture_reference() -> None:
        if cfg.l2sp_coef > 0 or cfg.track_drift:
            ref_names[:] = [n for n, p in model.named_parameters() if p.requires_grad]
            ref_params[:] = [p.detach().clone() for n, p in model.named_parameters() if p.requires_grad]

    def drift_stats() -> Dict[str, float]:
        with torch.no_grad():
            ps = [p for p in model.parameters() if p.requires_grad]
            d = torch._foreach_sub(ps, ref_params)
            dn = torch.stack(torch._foreach_norm(d)).double()
            rn = torch.stack(torch._foreach_norm(ref_params)).double()
            emb = torch.tensor([("embed_tokens" in n) for n in ref_names], device=dn.device)
            out = {"drift_l2": float(dn.pow(2).sum().sqrt()), "drift_rel": float(dn.pow(2).sum().sqrt() / rn.pow(2).sum().sqrt()),
                   "drift_rel_embed": float(dn[emb].pow(2).sum().sqrt() / rn[emb].pow(2).sum().sqrt()) if bool(emb.any()) else 0.0,
                   "drift_rel_layers": float(dn[~emb].pow(2).sum().sqrt() / rn[~emb].pow(2).sum().sqrt())}
            out["l2sp_penalty"] = 0.5 * cfg.l2sp_coef * float(dn.pow(2).sum())
            return out

    # ---- resume
    os.makedirs(cfg.out_dir, exist_ok=True)
    last_path = os.path.join(cfg.out_dir, "last", "trainer_state.pt")
    step = 0
    rewarm_n, rewarm_from = 0, 0                                      # LR ramp after a resume that did not restore the optimizer state
    stopper = EarlyStopper(cfg.early_stopping_patience, cfg.early_stopping_min_delta, min_epoch_step)
    prev_path = os.path.join(cfg.out_dir, "last_prev", "trainer_state.pt")
    if cfg.resume and os.path.exists(last_path):
        try:
            ck = torch.load(last_path, map_location=dist.device, weights_only=False)
        except Exception as e:                                        # a truncated / corrupt latest checkpoint: fall back to the previous one
            if not os.path.exists(prev_path):
                raise
            print(f"[gemma_mt] WARNING: {last_path} unreadable ({type(e).__name__}); resuming from {prev_path}", flush=True)
            ck = torch.load(prev_path, map_location=dist.device, weights_only=False)
        if cfg.init_from:                                             # the reference of a phase started with init_from is the init_from weights
            capture_reference_from = torch.load(cfg.init_from, map_location=dist.device, weights_only=False)
            model.load_state_dict(capture_reference_from["model"])
            del capture_reference_from
        capture_reference()
        model.load_state_dict(ck["model"])
        if cfg.resume_optimizer and "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        else:                                                         # weights-only resume: momentum buffers restart from zero, the LR ramps up again
            rewarm_n = max(0, int(cfg.optimizer_rewarmup_steps))
        set_rng_state(ck["rng"])
        st = ck["state"]
        step = st["step"]
        rewarm_from = step
        stopper.load(st["stopper"])
        ck_sched = st.get("schedule_sha256")
        now_sched = None if sched_info is None else sched_info["sha256"]
        if ck_sched != now_sched:
            # a STITCHED schedule (sampler.py stitch) keeps the first `prefix_steps` rows of the schedule the checkpoint was trained on, byte for byte, and rebuilds the rest
            # (owner 2026-10-31: extend the data, e.g. with the Efik sheet, without restarting). The checkpoint step must lie inside that identical prefix.
            stitched = ((sched_info or {}).get("sidecar_content") or {}).get("stitched_from") or {}
            prefix = int(stitched.get("prefix_steps", -1))
            same_prefix = (sched_arr is not None and 0 < prefix <= sched_arr.shape[0] and stitched.get("schedule_sha256") == ck_sched and step <= prefix
                           and hashlib.sha256(np.ascontiguousarray(sched_arr[:prefix], dtype=np.int64).tobytes()).hexdigest() == stitched.get("prefix_sha256"))
            if not same_prefix:
                raise ValueError(f"resume: the checkpoint {last_path} was trained with schedule {ck_sched} but this run has {now_sched}; "
                                 "a resumed run must continue the same schedule file (or a stitched one whose first rows are identical)")
            if dist.is_main:
                print(f"[gemma_mt] resume onto a STITCHED schedule: its first {prefix} steps are identical to the schedule of the checkpoint (step {step}); the rest was rebuilt", flush=True)
        if dist.is_main:
            print(f"[gemma_mt] resumed from step {step}" + ("" if rewarm_n == 0 and cfg.resume_optimizer and "optimizer" in ck else
                  f" with a FRESH optimizer (no optimizer state restored); LR re-warmup {rewarm_n} steps"), flush=True)
    elif cfg.init_from:
        ck = torch.load(cfg.init_from, map_location=dist.device, weights_only=False)
        model.load_state_dict(ck["model"])
        capture_reference()
        step = int(ck["state"]["step"])
        stopper.load(ck["state"]["stopper"])
        stopper.bad = 0                                               # the best loss carries over, the patience counter starts again
        del ck
        if dist.is_main:
            print(f"[gemma_mt] new phase from {cfg.init_from}: weights at step {step} (best so far {stopper.best:.4f} @ {stopper.best_step}); "
                  f"optimizer={cfg.optimizer}, fresh optimizer state, {n_run} steps", flush=True)
    else:
        capture_reference()                                           # a fresh start: the reference is the model_name weights

    logger = MetricsLogger(os.path.join(cfg.out_dir, "metrics.jsonl"), cfg.wandb_project, cfg.run_name,
                           dataclasses.asdict(cfg), enabled=dist.is_main)
    if dist.is_main:
        with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
            json.dump({"config": dataclasses.asdict(cfg), "mask_mode": mode, "total_steps": total_steps,
                       "steps_per_epoch": spe, "train_packs": len(train_ds), "val_packs": len(val_ds),
                       "tokenizer_fingerprint": fingerprint, "stop_ids": stop.ids,
                       "schedule": None if sched_manifest is None else {k: sched_manifest[k] for k in ("file", "sha256", "steps", "per_step", "draws")}}, f, indent=2)
            if sched_manifest is not None:                            # for the launch manifest: hash, length, per-language draws, tokens, checks
                with open(os.path.join(cfg.out_dir, "schedule_manifest.json"), "w") as f:
                    json.dump(sched_manifest, f, indent=1)
        print(f"[gemma_mt] sft: {len(train_ds)} train packs, {len(val_ds)} val packs, {spe} steps/epoch, "
              f"{total_steps} total steps, mask_mode={mode}, sliding_window={sliding_window}", flush=True)

    def state_dict_now():
        return {"step": step, "stopper": stopper.state(), "run_start": run_start,
                "schedule_sha256": None if sched_info is None else sched_info["sha256"]}

    from gemma_mt.hub.push_checkpoint import CheckpointPusher

    pusher = CheckpointPusher(cfg.hub_repo_id if dist.is_main else None, private=cfg.hub_private, stage="sft",
                              directions=tuple(train_ds.meta.get("directions") or ("en2x",)))
    if dist.is_main and cfg.hub_repo_id and not pusher.enabled:
        print("[gemma_mt] WARNING: hub_repo_id is set but no write token (HF_WRITE_TOKEN / HF_TOKEN) is in the environment: "
              "checkpoints will NOT be pushed", flush=True)
    push_every = cfg.hub_push_every or cfg.eval_every

    def run_extra_val(at_step: int) -> None:
        for name, vds in extra_val.items():
            ev = evaluate(model, vds, collator, cfg, dist, sliding_window, mode)
            model.train()
            logger.log(at_step, {f"val_{name}_loss": ev["val_loss"], f"val_{name}_loss_macro": ev["val_loss_macro"], f"val_{name}_tokens": ev["val_tokens"],
                                 **{f"val_{name}_loss/{k}": v for k, v in ev["val_loss_by_language"].items()}})
            if dist.is_main:
                print(f"[gemma_mt] eval[{name}] @ {at_step}: val_loss {ev['val_loss']:.4f} macro {ev['val_loss_macro']:.4f} "
                      f"({ev['val_macro_languages']} languages, {ev['val_tokens']} target tokens)", flush=True)

    stopped_early = False
    t0, tokens_seen = time.time(), 0
    last_eval: Dict[str, Any] = {}
    if cfg.eval_at_start and step == 0:                               # the baseline of the run: validation loss of the untouched starting weights
        e0 = evaluate(model, val_ds, collator, cfg, dist, sliding_window, mode)
        model.train()
        logger.log(0, {"val_loss": e0["val_loss"], "val_loss_macro": e0["val_loss_macro"], "val_tokens": e0["val_tokens"],
                       **{f"val_loss/{k}": v for k, v in e0["val_loss_by_language"].items()}})
        if dist.is_main:
            print(f"[gemma_mt] eval @ 0 (start): val_loss {e0['val_loss']:.4f} macro {e0['val_loss_macro']:.4f} ({e0['val_macro_languages']} languages)", flush=True)
        run_extra_val(0)
    emb_ids = embedding_param_ids(model) if cfg.embed_only_steps > 0 else set()
    if cfg.embed_only_steps > 0 and not emb_ids:
        raise ValueError("embed_only_steps > 0 but the model has no embedding parameters")
    if cfg.embed_only_min_row and emb_ids and cfg.embed_only_min_row >= model.get_input_embeddings().weight.shape[0]:
        raise ValueError(f"embed_only_min_row={cfg.embed_only_min_row} is not below the vocabulary size {model.get_input_embeddings().weight.shape[0]}")
    if cfg.embed_only_steps > 0 and dist.is_main:
        print(f"[gemma_mt] embeddings-only warm-up: steps {run_start + 1}..{run_start + cfg.embed_only_steps} of {n_run} update only the embedding"
              + (f" rows >= {cfg.embed_only_min_row}" if cfg.embed_only_min_row else " matrix") + " (part of the schedule, not extra steps)", flush=True)
    while step < total_steps:
        embed_only = step - run_start < cfg.embed_only_steps
        lr = lr_at(step - run_start, n_run, warmup, cfg.learning_rate, cfg.min_lr_ratio)
        if rewarm_n and step - rewarm_from < rewarm_n:
            lr *= (step - rewarm_from + 1) / (rewarm_n + 1)
        for g in optimizer.param_groups:
            g["lr"] = lr
        if schedule is not None:
            idx = schedule[step - run_start].reshape(cfg.grad_accum_steps, dist.world, cfg.micro_batch_size)[:, dist.rank, :].tolist()
        else:
            idx = batch_indices(len(train_ds), step, cfg.micro_batch_size, cfg.grad_accum_steps, dist.rank, dist.world,
                                cfg.seed)
        micro = [collator([train_ds[i] for i in row]) for row in idx]
        n_local = sum(int(m["num_target_tokens"]) for m in micro)
        n_global = float(dist.all_reduce_sum(float(n_local)))
        if n_global <= 0:
            raise RuntimeError(f"step {step}: no target tokens in the accumulated batch")
        optimizer.zero_grad(set_to_none=True)
        loss_sum_total = 0.0
        for k, mb in enumerate(micro):
            sync_ctx = nullcontext()
            if dist.world > 1 and k < len(micro) - 1:
                sync_ctx = train_model.no_sync()
            with sync_ctx:
                n = int(mb["num_target_tokens"])
                if n == 0:
                    continue
                nll_sum = train_model(_to_device(mb, dist.device))
                if not torch.isfinite(nll_sum):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                (nll_sum * (dist.world / n_global)).backward()
            loss_sum_total += float(nll_sum.detach())
        if embed_only:
            keep_only_embedding_grads(model, emb_ids, cfg.embed_only_min_row)
        if tr_rows_t is not None:
            keep_only_embedding_rows(model, tr_emb_ids, tr_rows_t)
        gnorm =torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not torch.isfinite(gnorm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        if cfg.l2sp_coef > 0 and ref_params:
            with torch.no_grad():                                     # L2-SP: after the clip, so the pull is never scaled down by it
                live = [(p, r) for p, r in zip((q for q in model.parameters() if q.requires_grad), ref_params) if p.grad is not None]
                pulls = torch._foreach_sub([p for p, _ in live], [r for _, r in live])
                torch._foreach_add_([p.grad for p, _ in live], pulls, alpha=cfg.l2sp_coef)
        optimizer.step()
        step += 1
        tokens_seen += sum(int(np.asarray(m["input_ids"]).size) for m in micro) * dist.world

        if step % cfg.log_every == 0 or step == 1:
            train_loss = float(dist.all_reduce_sum(loss_sum_total)) / n_global
            el = time.time() - t0
            rec = {"train_loss": train_loss, "lr": lr, "grad_norm": float(gnorm),
                   "target_tokens_step": n_global, "epoch": step / spe,
                   "tokens_per_s": tokens_seen / max(el, 1e-9)}
            if cfg.embed_only_steps:
                rec["embed_only"] = float(embed_only)
            drift_txt = ""
            if ref_params:
                ds_ = drift_stats()
                rec.update(ds_)
                drift_txt = f" drift {ds_['drift_rel'] * 100:.4f}% (layers {ds_['drift_rel_layers'] * 100:.4f}%, embed {ds_['drift_rel_embed'] * 100:.4f}%)"
            logger.log(step, rec)
            if dist.is_main:
                print(f"[gemma_mt] step {step}/{total_steps} loss {train_loss:.4f} lr {lr:.2e} "
                      f"gnorm {float(gnorm):.2f}{drift_txt}", flush=True)

        if (quick_ids and cfg.quick_eval_every and step % cfg.quick_eval_every == 0
                and not (cfg.eval_every and step % cfg.eval_every == 0)):
            q = evaluate(model, val_ds, collator, cfg, dist, sliding_window, mode, pack_ids=quick_ids)
            model.train()
            logger.log(step, {"quick_val_loss_macro": q["val_loss_macro"], "quick_val_loss": q["val_loss"],
                              **{f"quick_val_loss/{k}": v for k, v in q["val_loss_by_language"].items()}})
            if dist.is_main:
                print(f"[gemma_mt] quick-val @ {step}: macro {q['val_loss_macro']:.4f} ({q['val_macro_languages']} languages) "
                      f"micro {q['val_loss']:.4f} on {len(quick_ids)} packs", flush=True)

        if cfg.eval_every and (step % cfg.eval_every == 0 or step == total_steps):
            last_eval = evaluate(model, val_ds, collator, cfg, dist, sliding_window, mode)
            model.train()
            monitored = last_eval["val_loss_macro"] if cfg.val_metric == "macro" else last_eval["val_loss"]
            improved, stopped_early = stopper.update(step, monitored)
            if improved and dist.is_main and not cfg.keep_single_checkpoint:
                save_model(model, tokenizer, os.path.join(cfg.out_dir, "best"), stop_ids=stop.ids)
            logger.log(step, {"val_loss": last_eval["val_loss"], "val_loss_macro": last_eval["val_loss_macro"],
                              "val_tokens": last_eval["val_tokens"], "best_val_loss": stopper.best, "bad_evals": stopper.bad,
                              **{f"val_loss/{k}": v for k, v in last_eval["val_loss_by_language"].items()}})
            if dist.is_main:
                print(f"[gemma_mt] eval @ {step}: val_loss {last_eval['val_loss']:.4f} macro {last_eval['val_loss_macro']:.4f} "
                      f"({last_eval['val_macro_languages']} languages) | monitored={cfg.val_metric} best "
                      f"{stopper.best:.4f} @ {stopper.best_step} bad_evals {stopper.bad}", flush=True)
                due = improved if cfg.hub_push_best_only else (step % push_every == 0 or stopped_early or step == total_steps)
                if cfg.hub_repo_id and due:
                    stage_dir = os.path.join(cfg.out_dir, "hub_stage")
                    save_model(model, tokenizer, stage_dir, stop_ids=stop.ids)
                    pusher.push(stage_dir, step, {"val_loss": last_eval["val_loss"], "val_loss_macro": last_eval["val_loss_macro"],
                                                  "best_val_loss_monitored": stopper.best, "best_step": stopper.best_step,
                                                  "val_metric": cfg.val_metric, "tokens_seen": tokens_seen,
                                                  "val_loss_by_language": last_eval["val_loss_by_language"],
                                                  "language_alpha": cfg.language_alpha, "epoch_equivalents": step / spe,
                                                  "improved": bool(improved), "config": dataclasses.asdict(cfg)},
                                as_best=bool(improved) and not cfg.hub_push_best_only,
                                block=(stopped_early or step == total_steps), cleanup=cfg.keep_single_checkpoint)

        if extra_val and cfg.eval_every and (step % cfg.eval_every == 0 or step == total_steps):
            run_extra_val(step)

        timeboxed = bool(cfg.stop_after_steps) and step >= cfg.stop_after_steps and step < total_steps
        if (cfg.save_every and step % cfg.save_every == 0) or stopped_early or timeboxed or step == total_steps:
            if dist.is_main:
                _save_state(last_path, model, optimizer if cfg.save_optimizer else None, state_dict_now(), keep_prev=cfg.keep_prev_checkpoint)
            dist.barrier()
        if stopped_early and cfg.early_stop_action == "propose":
            if dist.is_main:
                with open(os.path.join(cfg.out_dir, "STOP_PROPOSED"), "w") as f:
                    json.dump({"step": step, "best": stopper.best, "best_step": stopper.best_step, "bad_evals": stopper.bad,
                               "patience": cfg.early_stopping_patience}, f)
                print(f"[gemma_mt] STOP PROPOSED at step {step} (best {stopper.best:.4f} @ {stopper.best_step}, {stopper.bad} bad "
                      f"evaluations): training continues until the owner decides", flush=True)
            stopped_early = False
        if stopped_early:
            if dist.is_main:
                print(f"[gemma_mt] early stop at step {step} (best {stopper.best:.4f} @ {stopper.best_step})",
                      flush=True)
            break
        if timeboxed:
            break

    if dist.is_main:
        pusher.wait()
    summary = {"steps": step, "total_steps": total_steps, "best_val_loss": stopper.best, "best_step": stopper.best_step,
               "stopped_early": stopped_early, "last_eval": last_eval, "mask_mode": mode,
               "seconds": round(time.time() - t0, 1)}
    if sched_info is not None:
        summary["schedule"] = {"sha256": sched_info["sha256"], "steps": int(sched_info["steps"]), "file": sched_info["file"]}
    if dist.is_main:
        if not (cfg.stop_after_steps and step < total_steps and not stopped_early) and not cfg.keep_single_checkpoint:
            save_model(model, tokenizer, os.path.join(cfg.out_dir, "final"), stop_ids=stop.ids)
        with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    logger.close()
    dist.barrier()
    return summary


# ----------------------------------------------------------------------------------------------- smoke / CLI
def run_smoke(workdir: Optional[str] = None) -> Dict[str, Any]:
    """End to end on CPU: synthetic rows -> prepare -> tiny random Gemma3 -> train -> eval -> early-stop logic."""
    import tempfile

    from gemma_mt.data.prepare_sft import PrepareConfig, prepare
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.smoke import synthetic_rows, write_jsonl

    work = workdir or tempfile.mkdtemp(prefix="gemma_mt_smoke_sft_")
    tok = FakeGemmaTokenizer()
    raw = write_jsonl(os.path.join(work, "raw.jsonl"), synthetic_rows(60, seed=1))
    meta = prepare(PrepareConfig(input=raw, out_dir=os.path.join(work, "packed"), seq_len=256, val_fraction=0.2,
                                 seed=3), tokenizer=tok)
    cfg = SFTConfig(model_name="tiny-random", data_dir=os.path.join(work, "packed"), out_dir=os.path.join(work, "run"),
                    attn_implementation="sdpa", autocast_dtype="none", micro_batch_size=2, grad_accum_steps=1,
                    epochs=30, max_steps=40, learning_rate=3e-3, warmup_steps=3, eval_every=10, save_every=20,
                    log_every=5, early_stopping_patience=0, device="cpu", seed=0)
    summary = train(cfg, tokenizer=tok)
    first = [json.loads(l) for l in open(os.path.join(cfg.out_dir, "metrics.jsonl"))]
    losses = [r["train_loss"] for r in first if "train_loss" in r]
    assert all(math.isfinite(x) for x in losses), losses
    assert losses[-1] < losses[0], f"smoke loss did not decrease: {losses}"
    summary["first_loss"], summary["last_loss"] = losses[0], losses[-1]
    summary["workdir"] = work
    summary["packing_efficiency"] = meta["splits"]["train"]["packing_efficiency"]
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true", help="CPU end-to-end run on a tiny random Gemma3")
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    a = ap.parse_args(argv)
    if a.smoke:
        s = run_smoke()
        print(json.dumps({k: v for k, v in s.items() if k != "last_eval"}, indent=2))
        return 0
    if not a.config:
        ap.error("--config is required (or use --smoke)")
    d = load_yaml(a.config)
    d.update(parse_overrides(a.overrides))
    train(dataclass_from_dict(SFTConfig, d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
