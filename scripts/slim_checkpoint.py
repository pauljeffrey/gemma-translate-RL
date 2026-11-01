#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Strip the optimizer state from a trainer checkpoint (runs/<run>/last/trainer_state.pt): weights + step + RNG + stopper bookkeeping stay.

    python scripts/slim_checkpoint.py runs/sft_unimax/last/trainer_state.pt /workspace/slim/trainer_state.pt [--bf16]

The result (~1.1 GB fp32 for Gemma-3-270m instead of ~2.7 GB with Muon state) is a valid `init_from` file (new phase) AND a valid `last/trainer_state.pt` for
`resume: true` with a fresh optimizer (the trainer notices that `optimizer` is missing, restarts it and ramps the LR over `optimizer_rewarmup_steps`).
--bf16 halves the file again but rounds the fp32 master weights: use it only for a copy that is NOT meant to continue training.
"""
import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    dropped = ck.pop("optimizer", None)
    if a.bf16:
        ck["model"] = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in ck["model"].items()}
    os.makedirs(os.path.dirname(os.path.abspath(a.dst)), exist_ok=True)
    tmp = a.dst + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, a.dst)
    print(f"step {ck['state']['step']}: wrote {a.dst} ({os.path.getsize(a.dst)/1e9:.2f} GB; source {os.path.getsize(a.src)/1e9:.2f} GB; "
          f"optimizer {'dropped' if dropped is not None else 'was absent'})")


if __name__ == "__main__":
    main()
