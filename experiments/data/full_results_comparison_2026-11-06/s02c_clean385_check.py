#!/usr/bin/env python
"""Robustness: language-macro chrF++ of our systems on all rows vs rows not flagged by the 385M audit (NTREX 120, MAFAND 94 flagged rows; FLORES+ none), and on the
clean-for-270M rows with / without the 385M-flagged rows (the NLLB aggregates use the 270M flags only).  usage: python s02c_clean385_check.py CACHE OUT_DIR"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from frc_lib import *
CACHE, OUT = sys.argv[1:3]
sysd = {n: load_ours(n, CACHE) for n in ["sft385", "e03c1", "stageA", "b1best", "mixed", "old_sft", "old_rl"]}
flags = {b: (load_flags(b, "old")[0], load_flags(b, "385")[0]) for b in ["ntrex", "mafand"]}
langs = {"ntrex": sorted({i for (b, i) in sysd["sft385"].d if b == "ntrex"}), "mafand": sorted({i for (b, i) in sysd["sft385"].d if b == "mafand"})}
res = {}
for b in ["ntrex", "mafand"]:
    # same language sets as the main tables (NLLB-covered)
    agg = nllb_aggregates()
    L = [i for i in langs[b] if (b, "all", i) in agg]
    for n, s in sysd.items():
        def macro(pred):
            v = []
            for i in L:
                ids = [x for x in s.d[(b, i)]["ids"] if pred(i, x)]
                v.append(float(chrfpp(s.get(b, i, ids)[0].sum(0))))
            return float(np.mean(v))
        all_ = macro(lambda i, x: True)
        c385 = macro(lambda i, x: flags[b][1].get((i, x), False))
        c270 = macro(lambda i, x: flags[b][0].get((i, x), False))
        c270_385 = macro(lambda i, x: flags[b][0].get((i, x), False) and flags[b][1].get((i, x), False))
        res[f"{b}/{n}"] = dict(all=all_, clean385=c385, d_all_minus_clean385=all_ - c385, clean270=c270, clean270_and_385=c270_385, d_clean270_minus_both=c270 - c270_385)
json.dump(res, open(os.path.join(OUT, "checks_clean385.json"), "w"), indent=1)
print(max(abs(v["d_all_minus_clean385"]) for v in res.values()), max(abs(v["d_clean270_minus_both"]) for v in res.values()))
