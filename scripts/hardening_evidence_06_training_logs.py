#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""H1 step 6: what the training logs say about the hardening terms (no model needed). Reads the step logs of the final run (box/other/r7_runs/final2/metrics.jsonl; steps 251-267 of the
discarded first attempt are dropped: the last record of a step wins) and of phase 2c (paper/data/rl2c_metrics.jsonl, the RL-final run), writes
  training_blocks_final2.csv, training_blocks_rl2c.csv   block means (25 steps for the final run, 100 for 2c) of the failure classes, the detector rates, the ensemble statistics
  training_trend_tests.csv                               slope per 100 steps (ordinary least squares over steps, 95% interval from a block bootstrap) for the series that matter: do the
                                                         detector rates FALL during training (the policy learns to avoid what is penalised) or stay flat (they act as a constant tax)?
Steps 1-150 of the final run ran with the earlier version of the anchors (paper Section 6.7; the padding detector fires on 3-5% of samples as a hard failure there and on none after the
restart), so every trend is also given for steps 151-401 alone.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardening_evidence_common as C  # noqa: E402

warnings.filterwarnings("ignore")
F2 = os.path.join(C.ARCHIVE, "box", "other", "r7_runs", "final2", "metrics.jsonl")
R2C = os.path.join(C.REPO, "paper", "data", "rl2c_metrics.jsonl")
SERIES_F2 = [
    "reward/fail/no_end_of_turn", "reward/fail/copy_source", "reward/fail/degenerate", "reward/fail/wrong_language", "reward/fail/length_ratio", "reward/fail/empty", "reward/fail/stuffed_output",
    "reward/anchors/copy/penalised_rate", "reward/anchors/copy/fraction_mean", "reward/anchors/entity2/penalised_rate", "reward/anchors/entity2/deficit_mean",
    "reward/anchors/func/penalised_rate", "reward/anchors/func/deficit_mean", "reward/anchors/pad/penalised_rate", "reward/anchors/pad/deficit_mean",
    "reward/anchors/repeat/penalised_rate", "reward/anchors/repeat/deficit_mean", "reward/anchors/boiler/penalised_rate", "reward/anchors/boiler/deficit_mean",
    "reward/anchors/penalised_rate", "reward/anchors/penalty_mean",
    "reward/qe_ens/min_share", "reward/qe_ens/min_share_afri_high", "reward/qe_ens/min_share_ssa_high", "reward/qe_ens/abs_diff_mean", "reward/qe_ens/signed_diff_mean",
    "reward/qe_ens/r1_mean", "reward/qe_ens/r2_mean", "reward/qe_ens/raw_mean/africomet", "reward/qe_ens/raw_mean/ssa_mtl", "reward/qe_ens/only1_share", "reward/qe_ens/only2_share",
    "reward/longdoc/long_rate", "reward/longdoc/rep_severity", "reward/longdoc/rep_full_rate", "reward/longdoc/rep_penalised_rate", "reward/longdoc/cover_failed", "reward/longdoc/agg_tail",
    "reward/reward_mean", "reward/qe_mean_scored", "reward/consensus/chrf_mean",
]
SERIES_2C = ["reward/fail/no_end_of_turn", "reward/fail/copy_source", "reward/fail/degenerate", "reward/fail/wrong_language", "reward/fail/length_ratio", "reward/fail/empty",
             "reward/reward_mean", "reward/qe_mean_scored", "reward/consensus/chrf_mean"]


def load(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if "step" in r and any(k.startswith("reward/") for k in r):
            rows[int(r["step"])] = r                          # last record of a step wins (resumed lineage)
    return pd.DataFrame([rows[s] for s in sorted(rows)]).set_index("step")


def blocks(df, cols, width):
    cols = [c for c in cols if c in df]
    b = (df.index.to_numpy() - 1) // width
    g = df[cols].groupby(b).mean()
    g.insert(0, "step_lo", g.index * width + 1)
    g.insert(1, "step_hi", np.minimum((g.index + 1) * width, df.index.max()))
    return g.reset_index(drop=True)


def slope(df, col, lo, hi, B=2000, seed=3):
    x = df.loc[lo:hi, col].dropna()
    if len(x) < 20:
        return dict(slope_per100=np.nan, lo=np.nan, hi=np.nan, n=len(x))
    s = np.polyfit(x.index.values, x.values, 1)[0] * 100
    rng = np.random.default_rng(seed)
    blk = 10                                                  # block bootstrap over 10-step blocks (autocorrelated series)
    vals = x.values
    idx_all = x.index.values
    nb = len(vals) // blk
    sl = []
    for _ in range(B):
        pick = rng.integers(0, nb, size=nb)
        xs = np.concatenate([idx_all[p * blk:(p + 1) * blk] for p in pick])
        ys = np.concatenate([vals[p * blk:(p + 1) * blk] for p in pick])
        sl.append(np.polyfit(xs, ys, 1)[0] * 100 if len(np.unique(xs)) > 2 else np.nan)
    l, h = np.nanpercentile(sl, [2.5, 97.5])
    return dict(slope_per100=s, lo=l, hi=h, n=len(x))


def main():
    f2 = load(F2)
    r2 = load(R2C)
    bf = blocks(f2, SERIES_F2, 25)
    bf.to_csv(os.path.join(C.OUT, "training_blocks_final2.csv"), index=False, float_format="%.5g")
    b2 = blocks(r2, SERIES_2C, 100)
    b2.to_csv(os.path.join(C.OUT, "training_blocks_rl2c.csv"), index=False, float_format="%.5g")
    rows = []
    for col in SERIES_F2:
        if col not in f2:
            continue
        for name, lo, hi in (("final2 steps 1-401", 1, 401), ("final2 steps 151-401", 151, 401)):
            s = slope(f2, col, lo, hi)
            rows.append(dict(run="final2", window=name, series=col, mean_first25=f2.loc[lo:lo + 24, col].mean(), mean_last50=f2.loc[hi - 49:hi, col].mean(), **s))
    for col in SERIES_2C:
        if col not in r2:
            continue
        s = slope(r2, col, 1, int(r2.index.max()))
        rows.append(dict(run="rl2c", window=f"2c steps 1-{int(r2.index.max())}", series=col, mean_first25=r2.loc[1:100, col].mean(), mean_last50=r2.loc[r2.index.max() - 99:r2.index.max(), col].mean(), **s))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.OUT, "training_trend_tests.csv"), index=False, float_format="%.5g")
    pd.set_option("display.width", 250)
    print(out[out.window.isin(["final2 steps 151-401"])][["series", "mean_first25", "mean_last50", "slope_per100", "lo", "hi"]].round(4).to_string())


if __name__ == "__main__":
    main()
