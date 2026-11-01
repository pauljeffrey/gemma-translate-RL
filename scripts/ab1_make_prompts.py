#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AB1: the RL prompt set of the ablation = a seeded random subset of rl_prompts_v3, b64 and b256 rows only (both arms read the same files, so every step draws the same prompts).

    python scripts/ab1_make_prompts.py --src /workspace/sft_build/gemma/rl_prompts_v3 --out /workspace/jobs/ab1/data/rl_prompts_ab1 --per-bin 2500 [--seed 1234]

One parquet per language (the v3 layout); rows keep every column. 2,500 per bin and language is 6-10x what a 500-step arm draws from one language (b256: 400 steps x 64 prompts, the
x2.5 weak languages get at most ~800 draws), so almost no prompt repeats inside an arm. Needs pyarrow + pandas (the policy environment has them). No GPU.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bin", type=int, default=2500)
    ap.add_argument("--bins", nargs="*", default=["b64", "b256"])
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tot = 0
    for path in sorted(glob.glob(os.path.join(a.src, "*.parquet"))):
        iso = os.path.basename(path)[:-8]
        df = pd.read_parquet(path)
        keep = []
        for b in a.bins:
            sub = df[df["length_bin"] == b]
            if len(sub) == 0:
                continue
            rng = np.random.default_rng([a.seed, sum(map(ord, iso)), sum(map(ord, b))])
            keep.append(sub.iloc[np.sort(rng.choice(len(sub), size=min(a.per_bin, len(sub)), replace=False))])
        if not keep:
            print("skip", iso, "(no rows in the bins)")
            continue
        out = pd.concat(keep, ignore_index=True)
        out.to_parquet(os.path.join(a.out, f"{iso}.parquet"), index=False)
        tot += len(out)
        print(iso, len(out), {b: int((out["length_bin"] == b).sum()) for b in a.bins}, flush=True)
    print("total rows", tot)


if __name__ == "__main__":
    main()
