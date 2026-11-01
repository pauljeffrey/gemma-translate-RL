#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Combined validation of the reference-free confidence features over TWO checkpoints (D-47): the RL checkpoint (step 925 of phase 2c) and an early SFT checkpoint (step 2,250, 0.13 epoch-equivalents),
which gives the 15 trained yardstick languages a second, much weaker and graded capability. Rows = (language, checkpoint): 30 trained + 16 new.

    python scripts/newlang_confidence_combined.py --rl DIR/language_features.csv --sft DIR/language_features.csv --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gemma_mt.rl import confidence as C  # noqa: E402


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rl", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    A = pd.read_csv(a.rl)
    A["ckpt"] = "rl925"
    B = pd.read_csv(a.sft)
    B["ckpt"] = "sft2250"
    D = pd.concat([A, B], ignore_index=True)
    P = C.ConfidenceParams()
    D["trust"] = [C.language_trust(r.lid_right_T1, r.logp_sample, r.valid_rate, P) for r in D.itertuples()]
    ref = {c: float(D[(D.ckpt == c) & (D.trained == 1) & (D.lid_right_T1 >= 0.9)]["logp_sample"].median()) for c in ("rl925", "sft2250")}      # the run-relative reference: median of the working languages
    D["trust_rel"] = [C.language_trust(r.lid_right_T1, r.logp_sample, r.valid_rate, P, logp_ref=ref[r.ckpt]) for r in D.itertuples()]
    D.loc[D.trained == 0, "trust_rel"] = [C.language_trust(r.lid_right_T1, r.logp_sample, r.valid_rate, P, logp_ref=ref["rl925"]) for r in D[D.trained == 0].itertuples()]
    out = {"rows": int(len(D)), "trained_rows": int((D.trained == 1).sum())}
    tr = D[D.trained == 1]
    feats = ["lid_right_T1", "valid_rate", "groups_ge3", "cons_nolid", "cons_valid", "logp_sample", "logp_greedy", "len_cv", "finished", "trust", "trust_rel"]
    res = {}
    for name, d in (("trained_both_ckpts", tr), ("trained_sft2250", tr[tr.ckpt == "sft2250"]), ("trained_rl925", tr[tr.ckpt == "rl925"]), ("all_46", D)):
        res[name] = {}
        for f in feats:
            for tgt in ("excess_all", "greedy_chrf"):
                x, y = d[f].values.astype(float), d[tgt].values.astype(float)
                m = ~(np.isnan(x) | np.isnan(y))
                if m.sum() > 4 and np.std(x[m]) > 0:
                    r, p = stats.spearmanr(x[m], y[m])
                    res[name][f"{f}~{tgt}"] = dict(rho=float(r), p=float(p), n=int(m.sum()))
    out["spearman"] = res
    # within-prompt (language-level mean of the per-group Spearman) vs the capability of the checkpoint
    out["within_prompt_mean_rho_cons"] = {c: float(tr[tr.ckpt == c]["rho_cons"].mean()) for c in ("rl925", "sft2250")}
    out["within_prompt_mean_rho_logp"] = {c: float(tr[tr.ckpt == c]["rho_logp"].mean()) for c in ("rl925", "sft2250")}
    # leave-one-LANGUAGE-out linear capability model on the 30 trained rows, then applied to the 16 new languages
    cols = ["cons_nolid", "logp_sample"]
    langs = sorted(set(tr.key))
    pred = pd.Series(index=tr.index, dtype=float)
    for lg in langs:
        tri, tei = tr[tr.key != lg], tr[tr.key == lg]
        w = np.linalg.lstsq(np.c_[np.ones(len(tri)), tri[cols].values], tri.excess_all.values, rcond=None)[0]
        pred[tei.index] = np.c_[np.ones(len(tei)), tei[cols].values] @ w
    w = np.linalg.lstsq(np.c_[np.ones(len(tr)), tr[cols].values], tr.excess_all.values, rcond=None)[0]
    new = D[(D.trained == 0) & (D.key != "nde")]
    pn = np.c_[np.ones(len(new)), new[cols].values] @ w
    out["capability_model_30"] = dict(features=cols, coef=[float(x) for x in w], loo_language_spearman=float(stats.spearmanr(pred, tr.excess_all)[0]), loo_language_mae=float(np.abs(pred - tr.excess_all).mean()),
                                      sd_excess=float(tr.excess_all.std()), new_mean_pred=float(pn.mean()), new_mean_actual=float(new.excess_all.mean()), new_mae=float(np.abs(pn - new.excess_all.values).mean()),
                                      new_pred_spearman=float(stats.spearmanr(pn, new.excess_all)[0]))
    # does the trust (lid x logp x valid) separate "usable" from "not usable" rows? usable := LID right >= 0.8 and excess chrF >= 5
    D["usable"] = ((D.lid_right_T1 >= 0.8) & (D.excess_all >= 5.0)).astype(int)
    out["usable_rows"] = int(D.usable.sum())
    out["logp_reference_median"] = ref
    for nm in ("trust", "trust_rel"):
        out[f"{nm}_vs_usable"] = dict(auc=auc(D.loc[D.usable == 1, nm], D.loc[D.usable == 0, nm]),
                                      usable_mean=float(D.loc[D.usable == 1, nm].mean()), unusable_mean=float(D.loc[D.usable == 0, nm].mean()),
                                      share_zero_among_usable=float((D.loc[D.usable == 1, nm] == 0).mean()), share_positive_among_unusable=float((D.loc[D.usable == 0, nm] > 0).mean()))
    D.to_csv(os.path.join(a.out, "language_features_both_checkpoints.csv"), index=False)
    json.dump(out, open(os.path.join(a.out, "combined_validation.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "spearman"}, indent=1))
    for name in ("trained_both_ckpts", "all_46"):
        print(name)
        for k, v in res[name].items():
            if k.endswith("excess_all"):
                print("  ", k, round(v["rho"], 2), v["n"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
