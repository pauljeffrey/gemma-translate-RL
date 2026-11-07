#!/usr/bin/env python
"""Check that the AfriCOMET scores of the two scoring runs are on the same scale: the 270M-era run (e2, used for NLLB / TranslatePsy) and the 385M-harness run (used for all 385M and E-13 rows).
For rows where the earlier 270M RL / SFT hypothesis (e2 run) is IDENTICAL to the E-13 rerun hypothesis, both runs scored the same (source, hypothesis, reference) triple.
usage: python s02b_comet_scorer_check.py OUT_DIR   (python with numpy)"""
import json, os, sys, unicodedata
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frc_lib import *
OUT = sys.argv[1]
res = {}
for e2name, ev, pred in (("rl", "old_rl", ARC + "box_outputs_2026-11-04/out_old_rl2c/predictions.jsonl"), ("sft", "old_sft", ARC + "box_outputs_2026-11-04/out_old_sft/predictions.jsonl")):
    s2 = load_e2(e2name)
    ev_np = np.load(OURS[ev], allow_pickle=True)
    ek = {(b, i, r): c for b, i, r, c in zip(ev_np["bench"], ev_np["iso"], ev_np["rid"].astype(str), ev_np["comet"])}
    hyp13 = {}
    for line in open(pred, encoding="utf-8"):
        d = json.loads(line)
        if d["benchmark"] == "flores_plus":
            hyp13.setdefault((d["iso"], str(d["id"])), unicodedata.normalize("NFC", d["hyp"]))
    same = tot = 0
    a, b = [], []
    for iso in s2.langs("flores_plus"):
        e = s2.d[("flores_plus", iso)]
        for k, i in enumerate(e["ids"]):
            if (iso, i) not in hyp13:
                continue
            tot += 1
            h2 = unicodedata.normalize("NFC", s2.meta[iso]["hyp"][k].split("</think>")[-1].strip())
            if h2 == hyp13[(iso, i)]:
                same += 1
                c2 = e["comet"][k]
                c13 = ek.get(("flores_plus", iso, i), np.nan)
                if np.isfinite(c2) and np.isfinite(c13):
                    a.append(c2); b.append(c13)
    a, b = np.array(a), np.array(b)
    res[e2name] = dict(rows_compared=int(tot), identical_hypothesis=int(same), rows_with_both_scores=int(len(a)),
                       mean_abs_diff=float(np.abs(a - b).mean()) if len(a) else None, max_abs_diff=float(np.abs(a - b).max()) if len(a) else None,
                       mean_diff_e13_minus_e2=float((b - a).mean()) if len(a) else None, pearson=float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else None)
    print(e2name, res[e2name], flush=True)
json.dump(res, open(os.path.join(OUT, "checks_comet_scorer.json"), "w"), indent=1)
