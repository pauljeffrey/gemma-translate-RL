#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-68: what do the v2 and v3 stop-word / padding detectors do to the MODEL's own outputs in the languages that failed in the live run? Uses the greedy predictions of the SFT model
(R15's probe: preds/<model>/<iso>.jsonl on tests/<iso>.jsonl) and the human references of the same rows. 'gate' = a func or pad deficit >= 0.9 in a language where the detector is on (v3: only where gate_ok).
    python scripts/policy_flags.py --cal-v2 anchors_v2.json --cal-v3 anchors_v3.json --bigrams anchors_v2_bigrams.npz --langs urh,sid,bas,dua --model final_1433"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemma_mt.rl import anchors_v2 as V
from gemma_mt.rl import textstats as T

ap = argparse.ArgumentParser()
ap.add_argument("--cal-v2", required=True)
ap.add_argument("--cal-v3", required=True)
ap.add_argument("--bigrams", required=True)
ap.add_argument("--langs", default="urh,sid,bas,dua,oke,bbj,bin,ish,kri,kea,nbl,ach,nyn,ndc,dag,tsc,her,ibb,seh,shi")
ap.add_argument("--model", default="final_1433")
ap.add_argument("--probe", default="/workspace/jobs/r15/probe")
a = ap.parse_args()
A2 = V.Anchors2(json.load(open(a.cal_v2)), v2_detectors=("func", "pad", "copy"), bigram_path=a.bigrams)
A3 = V.Anchors2(json.load(open(a.cal_v3)), v2_detectors=("func", "pad", "copy"), bigram_path=a.bigrams)


def flags(A, iso, src, mt, use_gate_ok):
    sw, ow = T.out_tokens(src), T.out_tokens(mt)
    out = {}
    cf = A.copy_fraction(iso, src, mt)
    out["copy"] = None if cf is None else float(cf > 0)
    d = A.func_deficit(iso, sw, ow)
    out["func"] = d
    p = A.pad_deficit(iso, ow)
    out["pad"] = p
    sat = False
    for det, v in (("func", d), ("pad", p)):
        if v is not None and v >= 0.9 and (not use_gate_ok or A.gate_ok(iso, det)):
            sat = True
    out["gate"] = float(sat)
    return out


print("| lang | rows | human refs: v2 graded func/pad, v2 gate | MODEL: v2 graded func/pad, v2 gate | MODEL: v3 graded func/pad/copy, v3 gate |\n|---|---|---|---|---|")
for iso in a.langs.split(","):
    tp, pp = f"{a.probe}/tests/{iso}.jsonl", f"{a.probe}/preds/{a.model}/{iso}.jsonl"
    if not (os.path.exists(tp) and os.path.exists(pp)):
        continue
    tests = {(r["set"], r["id"]): r for r in map(json.loads, open(tp, encoding="utf-8"))}
    preds = [json.loads(l) for l in open(pp, encoding="utf-8")]
    agg = collections.defaultdict(list)
    for p in preds:
        t = tests.get((p["set"], p["id"]))
        if not t or t["set"] == "gold" or not p["hyp"].strip():
            continue
        for name, (A, text, g) in {"h2": (A2, t["reference"], False), "m2": (A2, p["hyp"], False), "m3": (A3, p["hyp"], True)}.items():
            f = flags(A, iso, t["english"], text, g)
            for k, v in f.items():
                if v is not None:
                    agg[(name, k)].append(v)
    if not agg:
        continue
    sh = lambda n, k: ("-" if not agg[(n, k)] else f"{np.mean(np.asarray(agg[(n, k)]) > 0):.2f}")
    print(f"| {iso} | {len(agg[('m2', 'gate')])} | {sh('h2', 'func')}/{sh('h2', 'pad')}, {sh('h2', 'gate')} | {sh('m2', 'func')}/{sh('m2', 'pad')}, {sh('m2', 'gate')} | {sh('m3', 'func')}/{sh('m3', 'pad')}/{sh('m3', 'copy')}, {sh('m3', 'gate')} |")
