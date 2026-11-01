#!/usr/bin/env python
"""D-59: build the STARTING trainer state of the final science+general GRPO run.

    weights   = the chosen start model (an HF directory: runs/final_1433, or the SFT-stage model written by run_sci_sft_stage.sh finalize)
    optimizer = the Adam moments (exp_avg / exp_avg_sq / step counts) of the final 2c RL run (runs/rl2c/last/trainer_state.pt)
    rng       = the 2c RNG state
    step      = 0, best_eval / best_select = -inf, bench_path = None   (-> the trainer scores the start policy on the new dev benchmark first, `resume_reset_selection`)

The KL reference is NOT part of the state: it is `model_name` in the yaml (runs/final_1433). Written to <out-dir>/last/trainer_state.pt (atomic) + <out-dir>/last/PREPARED.json.

  python scripts/prepare_initial_state.py --start-model /workspace/sft_build/runs/final_1433 --optimizer-from /workspace/sft_build/runs/rl2c/last/trainer_state.pt \
      --out-dir /workspace/jobs/r7/runs/final [--verify]

CPU only (needs ~7 GB RAM). Refuses to overwrite an existing state unless --force. With --verify the written file is re-read and loaded into a fresh model + AdamW built exactly like train_rl.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time


def _sha_head(path: str, n: int = 1 << 20) -> str:
    with open(path, "rb") as f:
        return hashlib.sha1(f.read(n)).hexdigest()[:12]


def build_state(start_model: str, optimizer_from: str, *, attn: str = "sdpa") -> dict:
    import torch

    from gemma_mt.modeling import load_model

    ck = torch.load(optimizer_from, map_location="cpu", weights_only=False)
    model = load_model(start_model, attn_implementation=attn, dtype=torch.float32, device="cpu")
    sd = model.state_dict()
    old = ck["model"]
    if set(sd) != set(old):
        a, b = sorted(set(sd) - set(old)), sorted(set(old) - set(sd))
        raise SystemExit(f"parameter names differ between the start model and the optimizer checkpoint: only in model {a[:5]}, only in checkpoint {b[:5]}")
    bad = [k for k in sd if tuple(sd[k].shape) != tuple(old[k].shape)]
    if bad:
        raise SystemExit(f"shape mismatch for {bad[:5]}")
    # optimizer layout must match the groups of train_rl.py: [decay (ndim>=2, no embed_tokens), other]
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed_tokens" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed_tokens" not in n)]
    pg = ck["optimizer"]["param_groups"]
    if [len(g["params"]) for g in pg] != [len(decay), len(other)]:
        raise SystemExit(f"optimizer groups {[len(g['params']) for g in pg]} do not match the model ({len(decay)}, {len(other)})")
    n_state = len(ck["optimizer"]["state"])
    nonfinite = 0
    for s in ck["optimizer"]["state"].values():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in s and not bool(torch.isfinite(s[k]).all()):
                nonfinite += 1
    if nonfinite:
        raise SystemExit(f"{nonfinite} non-finite Adam moment tensors in {optimizer_from}")
    steps = sorted({int(s["step"]) for s in ck["optimizer"]["state"].values() if "step" in s})
    num = sum(float((sd[k].float() - old[k].float()).pow(2).sum()) for k in sd)
    den = sum(float(old[k].float().pow(2).sum()) for k in sd)
    # weights are NOT cloned: tied tensors (embed_tokens = lm_head) keep sharing one storage, so torch.save writes it once (a clone made the file 0.7 GB larger)
    state = {"model": {k: v.detach() for k, v in sd.items()}, "optimizer": ck["optimizer"], "rng": ck["rng"],
             "state": {"step": 0, "best_eval": -float("inf"), "best_select": -float("inf"), "base_chrf": float("nan"), "bench_path": None,
                       "prepared": {"start_model": start_model, "optimizer_from": optimizer_from, "optimizer_from_step": ck["state"].get("step"),
                                    "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}}
    info = {"start_model": start_model, "optimizer_from": optimizer_from, "optimizer_from_step": ck["state"].get("step"), "adam_state_tensors": n_state,
            "adam_step_counts": steps[:3] + (["..."] if len(steps) > 3 else []), "rel_l2_start_vs_2c_policy": math.sqrt(num / max(den, 1e-30)),
            "n_params": sum(p.numel() for p in model.parameters())}
    return state, info


def verify(path: str, start_model: str, attn: str = "sdpa") -> dict:
    """Load the written file the way train_rl.py does (fresh model + AdamW with the same groups) and check weights, moments and counters."""
    import torch

    from gemma_mt.modeling import load_model

    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = load_model(start_model, attn_implementation=attn, dtype=torch.float32, device="cpu")
    model.load_state_dict(ck["model"])
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "embed_tokens" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "embed_tokens" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.0}, {"params": other, "weight_decay": 0.0}], lr=1e-6, betas=(0.9, 0.95))
    opt.load_state_dict(ck["optimizer"])
    p0 = next(iter(model.parameters()))
    st = opt.state[p0]
    ok = tuple(st["exp_avg"].shape) == tuple(p0.shape) and tuple(st["exp_avg_sq"].shape) == tuple(p0.shape)
    ref = load_model(start_model, attn_implementation=attn, dtype=torch.float32, device="cpu")
    same = all(torch.equal(a, b) for a, b in zip(ref.state_dict().values(), model.state_dict().values()))
    return {"loads_into_model_and_adamw": bool(ok), "weights_equal_start_model": bool(same), "state": {k: (v if k != "prepared" else "...") for k, v in ck["state"].items()},
            "adam_exp_avg_sq_mean": float(sum(float(s["exp_avg_sq"].mean()) for s in opt.state.values()) / max(len(opt.state), 1))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-model", required=True)
    ap.add_argument("--optimizer-from", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out = os.path.join(a.out_dir, "last", "trainer_state.pt")
    if os.path.exists(out) and not a.force:
        print(f"{out} exists (use --force to replace it)", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(out), exist_ok=True)
    need = os.path.getsize(a.optimizer_from) + (1 << 30)      # the new file is written to .tmp next to the old one (replaced atomically); +1 GiB headroom
    free = shutil.disk_usage(os.path.dirname(out)).free
    if free < need:
        print(f"only {free / 2**30:.1f} GiB free, need ~{need / 2**30:.1f}", file=sys.stderr)
        return 3
    import torch

    state, info = build_state(a.start_model, a.optimizer_from, attn=a.attn)
    tmp = out + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, out)
    info["trainer_state_bytes"] = os.path.getsize(out)
    info["start_model_head_sha1"] = _sha_head(os.path.join(a.start_model, "model.safetensors")) if os.path.exists(os.path.join(a.start_model, "model.safetensors")) else None
    if a.verify:
        info["verify"] = verify(out, a.start_model, a.attn)
    json.dump(info, open(os.path.join(a.out_dir, "last", "PREPARED.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
