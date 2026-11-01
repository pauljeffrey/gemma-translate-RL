#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""E2 model comparison, step 3 (box, CPU): per-language and macro chrF++ / spBLEU / AfriCOMET with bootstrap CIs for every system scored by
e2_score.py (scores/NAME.pkl) and e2_comet.py (scores/NAME.comet.pkl), FLORES+ devtest English -> X.

    python e2_model_comparison_aggregate.py --scores /workspace/jobs/e2/scores --sets sets.json --out /workspace/jobs/e2/out [--boot 1000]

Metrics: chrF++ = sacrebleu CHRF(word_order=2) on NFC text, spBLEU = BLEU(tokenize="flores200"), AfriCOMET = masakhane/africomet-stl-1.1 reference based
(mean of the sentence scores; only languages whose `africomet_code` is set; others are n/a, never imputed).
Subsets (same as reports/data/final_model_per_language.csv): all (1,012 sentences), no_pair (reference pair never seen in SFT; the headline), clean (English sentence not in
the SFT text; 48 per language). AfriCOMET is computed on no_pair and clean only.
Bootstrap: B resamples of the sentences of each (language, subset) with ONE index matrix per (language, subset), shared by all systems (paired); the CI of a macro is
the 2.5 / 97.5 percentile of the macro of the replicates (language sampling noise is NOT included). Macros over a language set are reported only when the system
covers every language of the set (`full`); otherwise `k_covered/k_set` is given and the value is over the covered languages.

Outputs (out/): per_language.csv, macro.csv, paired_vs.csv (macro difference to --ref-model with CI).
"""
import argparse
import csv
import json
import os
import pickle
import zlib

import numpy as np

import sys
sys.path.append("/workspace/sft_build/pylibs")
from sacrebleu.metrics import CHRF, BLEU  # noqa: E402

CH, BL = CHRF(word_order=2), BLEU(tokenize="flores200")


def score_from(m, st):
    return m._compute_score_from_stats(list(st)).score


def boot_stat(m, st, ix, chunk=50):
    out = np.empty(ix.shape[0])
    for s in range(0, ix.shape[0], chunk):
        sums = st[ix[s:s + chunk]].sum(1)
        out[s:s + chunk] = [score_from(m, r) for r in sums]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="/workspace/jobs/e2/scores")
    ap.add_argument("--sets", default="/workspace/jobs/e2/sets.json")
    ap.add_argument("--out", default="/workspace/jobs/e2/out")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--ref-model", default="rl")
    ap.add_argument("--ref-models", default="rl,sft,a2w,live250")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    B = args.boot
    sets = json.load(open(args.sets))
    names = sorted(f[:-4] for f in os.listdir(args.scores) if f.endswith(".pkl") and not f.endswith((".comet.pkl", ".ssa.pkl")))
    per = []                       # per-language rows
    reps = {}                      # (model, iso, subset, metric) -> (point, replicates)
    for name in names:
        S = pickle.load(open(f"{args.scores}/{name}.pkl", "rb"))
        NEURAL = {}
        for suffix, key in ((".comet.pkl", "africomet"), (".ssa.pkl", "ssa_comet")):
            cp = f"{args.scores}/{name}{suffix}"
            if os.path.exists(cp):
                NEURAL[key] = pickle.load(open(cp, "rb"))
        for iso, d in sorted(S.items()):
            n_all = len(d["ids"])
            masks = {"all": np.ones(n_all, bool),
                     "no_pair": np.array([p is False for p in d["pair_seen"]]),
                     "clean": np.array([c is False for c in d["contaminated"]])}
            for sub, mk in masks.items():
                n = int(mk.sum())
                if n == 0 or (sub == "all" and n_all < 1000):     # a system run on the no_pair rows only has no `all` row
                    continue
                rng = np.random.default_rng(zlib.crc32(f"{iso}|{sub}".encode()))
                ix = rng.integers(0, n, size=(B, n))
                row = {"model": name, "iso": iso, "subset": sub, "n": n}
                for key, m, st in (("chrf", CH, d["chrf_st"][mk]), ("spbleu", BL, d["bleu_st"][mk])):
                    pt = score_from(m, st.sum(0))
                    rp = boot_stat(m, st, ix)
                    row[key] = pt
                    row[key + "_lo"], row[key + "_hi"] = np.percentile(rp, [2.5, 97.5])
                    reps[(name, iso, sub, key)] = (pt, rp)
                for key, C in NEURAL.items():
                    if iso in C and sub in ("no_pair", "clean"):
                        ids = [i for i, k in zip(d["ids"], mk) if k]
                        v = np.array([C[iso].get(i, np.nan) for i in ids])
                        if not np.isnan(v).any():
                            pt = float(v.mean()); rp = v[ix].mean(1)
                            row.update({key: pt, key + "_lo": float(np.percentile(rp, 2.5)), key + "_hi": float(np.percentile(rp, 97.5))})
                            if key == "africomet":
                                row["n_comet"] = n
                            reps[(name, iso, sub, key)] = (pt, rp)
                per.append(row)
    cols = ["model", "iso", "subset", "n", "chrf", "chrf_lo", "chrf_hi", "spbleu", "spbleu_lo", "spbleu_hi", "africomet", "africomet_lo", "africomet_hi", "n_comet", "ssa_comet", "ssa_comet_lo", "ssa_comet_hi"]
    with open(f"{args.out}/per_language.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in per:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    # macros
    macro, paired = [], []
    for name in names:
        for sname, isos in sets.items():
            for sub in ("no_pair", "clean", "all"):
                for key in ("chrf", "spbleu", "africomet", "ssa_comet"):
                    have = [i for i in isos if (name, i, sub, key) in reps]
                    if not have:
                        continue
                    pts = np.mean([reps[(name, i, sub, key)][0] for i in have])
                    rp = np.mean([reps[(name, i, sub, key)][1] for i in have], axis=0)
                    macro.append({"model": name, "set": sname, "subset": sub, "metric": key, "k_covered": len(have), "k_set": len(isos),
                                  "full": int(len(have) == len(isos)), "value": round(float(pts), 4),
                                  "lo": round(float(np.percentile(rp, 2.5)), 4), "hi": round(float(np.percentile(rp, 97.5)), 4)})
                    for ref in args.ref_models.split(","):
                        if name != ref and all((ref, i, sub, key) in reps for i in have):
                            rr = np.mean([reps[(ref, i, sub, key)][1] for i in have], axis=0)
                            rpt = np.mean([reps[(ref, i, sub, key)][0] for i in have])
                            d = pts - rpt; dr = rp - rr
                            paired.append({"model": name, "vs": ref, "set": sname, "subset": sub, "metric": key, "k": len(have), "full": int(len(have) == len(isos)),
                                           "delta": round(float(d), 4), "lo": round(float(np.percentile(dr, 2.5)), 4), "hi": round(float(np.percentile(dr, 97.5)), 4)})
    for fn, rows in (("macro.csv", macro), ("paired_vs.csv", paired)):
        if rows:
            with open(f"{args.out}/{fn}", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("systems", names, "per-language rows", len(per), "macro rows", len(macro))


if __name__ == "__main__":
    main()
