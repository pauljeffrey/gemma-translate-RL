# SPDX-License-Identifier: AGPL-3.0-or-later
"""Optimizers of the RL trainer (2026-10-29, Amendment 4): AdamW (the default, unchanged) or Muon with the SAME convention as train_sft.py.

Muon (torch.optim.Muon, adjust_lr_fn="match_rms_adamw") orthogonalises the update of the 2-D DECODER matrices (parameter names containing ".layers."); the (tied) embedding / LM head, the norms and every
1-D parameter stay on AdamW. match_rms_adamw scales the Muon update to AdamW's update RMS, so ONE learning rate (and weight decay) serves both halves and the RL learning rate does not have to be re-tuned.
The state of a Muon run is {"combined": [muon_state, adamw_state]}; a stage always starts with a FRESH optimizer (a new out_dir), only a crash-restart of the same stage reloads it.
"""
from __future__ import annotations

from typing import Any, Dict, List


class CombinedOptimizer:
    """Several torch optimizers behind the one interface the RL loop uses (param_groups, zero_grad, step, state_dict, load_state_dict)."""

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
            raise ValueError("the checkpoint holds a single-optimizer (AdamW) state but optimizer=muon: a new stage starts with a fresh optimizer (new out_dir); do not resume an AdamW run as Muon")
        for o, st in zip(self.optimizers, sd["combined"]):
            o.load_state_dict(st)


def optimizer_kind_of_state(sd: Dict[str, Any]) -> str:
    return "muon" if isinstance(sd, dict) and "combined" in sd else "adamw"


def build_rl_optimizer(policy, cfg, device):
    """AdamW exactly as before (decay group: 2-D except embed_tokens; the rest without decay), or Muon + AdamW for `cfg.optimizer == "muon"`."""
    import torch

    decay = [p for n, p in policy.named_parameters() if p.ndim >= 2 and "embed_tokens" not in n]
    other = [p for n, p in policy.named_parameters() if not (p.ndim >= 2 and "embed_tokens" not in n)]
    adamw_kw = dict(lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), fused=True if device.type == "cuda" else None)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": other, "weight_decay": 0.0}], **adamw_kw)
    if cfg.optimizer != "muon":
        raise ValueError(f"unknown optimizer {cfg.optimizer!r} (adamw | muon)")
    muon_p = [p for n, p in policy.named_parameters() if p.requires_grad and p.ndim == 2 and ".layers." in n]
    ids = {id(p) for p in muon_p}
    adam_dec = [p for p in decay if id(p) not in ids]
    groups: List[Dict[str, Any]] = [g for g in ({"params": adam_dec, "weight_decay": cfg.weight_decay}, {"params": other, "weight_decay": 0.0}) if g["params"]]
    muon = torch.optim.Muon([{"params": muon_p, "weight_decay": cfg.weight_decay}], lr=cfg.learning_rate, momentum=cfg.muon_momentum, ns_steps=cfg.muon_ns_steps,
                            adjust_lr_fn="match_rms_adamw")
    return CombinedOptimizer([muon, torch.optim.AdamW(groups, **adamw_kw)])
