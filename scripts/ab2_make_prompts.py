#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AB2: the RL prompt pool of the cheap ablation = b64 windows of 4 languages (default yor kin orm lin) drawn at random from rl_prompts_v3, DECONTAMINATED against every evaluation set.

    python scripts/ab2_make_prompts.py --src /workspace/sft_build/gemma/rl_prompts_v3 --out /workspace/jobs/ab2/data/rl_prompts_ab2 [--langs yor kin orm lin] [--pool 5000] [--seed 1234]

Per language: all b64 rows of the v3 pool -> drop rows whose English side hits the project's hold-out (`/workspace/jobs/r10/clean_stage.build_holdout`: exact sentence or shared word
8-gram with FLORES+ dev/devtest, NTREX, MAFAND, SMOL (sent/doc), GATITOS, SALT, GlotStoryBook, IbomMT, FLORES-MOZ, AfriScience dev/test, AfriLongDoc, bench_dev_clean) -> seeded sample of
`--pool` rows (lin has only 5,000 b64 rows in v3: all that survive). One parquet per language (the v3 layout), identical for both arms. The trainer draws prompts WITH replacement
(PromptSchedule), so the pool is larger than the 49 x 64 draws; `scripts/ab2_drawn_prompts.py` lists what is actually drawn. CPU only (needs pandas/pyarrow, the box's venv_gemma).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/workspace/sft_build/gemma/rl_prompts_v3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*", default=["yor", "kin", "orm", "lin"])
    ap.add_argument("--pool", type=int, default=5000)
    ap.add_argument("--bin", default="b64")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-holdout", action="store_true", help="skip the decontamination (tests only)")
    a = ap.parse_args(argv)
    ho = None
    if not a.no_holdout:
        sys.path[:0] = ["/workspace/jobs/r10", "/workspace/jobs/b1/gemma_mt", "/workspace/jobs/r4/gemma_mt_sft/scripts"]
        import clean_stage as C
        ho, _ = C.build_holdout()
    os.makedirs(a.out, exist_ok=True)
    stats = {}
    for iso in a.langs:
        df = pd.read_parquet(os.path.join(a.src, f"{iso}.parquet"))
        sub = df[df["length_bin"] == a.bin].reset_index(drop=True)
        n0 = len(sub)
        if ho is not None:
            hits = [bool(ho.hit(t)) for t in sub["english"].tolist()]
            sub = sub[[not h for h in hits]].reset_index(drop=True)
        n1 = len(sub)
        rng = np.random.default_rng([a.seed, sum(map(ord, iso)), sum(map(ord, a.bin))])
        keep = np.sort(rng.choice(n1, size=min(a.pool, n1), replace=False))
        out = sub.iloc[keep].reset_index(drop=True)
        out.to_parquet(os.path.join(a.out, f"{iso}.parquet"), index=False)
        stats[iso] = {"v3_rows_in_bin": n0, "after_holdout": n1, "dropped_by_holdout": n0 - n1, "pool": len(out), "median_src_tokens": float(out["src_afr_tokens"].median())}
        print(iso, stats[iso], flush=True)
    json.dump({"seed": a.seed, "bin": a.bin, "langs": a.langs, "stats": stats}, open(os.path.join(a.out, "_pool_stats.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
