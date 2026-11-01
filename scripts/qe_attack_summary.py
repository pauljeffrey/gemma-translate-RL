#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Summary of the QE check of the attacks: per class the mean score of each model and the share of attack rows scoring ABOVE the human reference of the same sentence (and above the policy sample)."""
import argparse
import collections
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--rows", required=True)
ap.add_argument("--scores-dir", required=True)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.rows, encoding="utf-8")]
models = {m: json.load(open(f"{a.scores_dir}/{m}.json")) for m in ("africomet_qe", "ssa_mtl")}
by = collections.defaultdict(dict)
for i, r in enumerate(rows):
    by[(r["iso"], r["src"])][r["cls"]] = {m: v[i] for m, v in models.items()}
classes = [c for c in ("human", "policy", "stuff30", "pad", "pad_fit", "stuff_fw", "gloss", "gloss_pad", "gloss_fw", "en_copy")]
print("| class | n | AfriCOMET-QE mean | share > human | share > policy | SSA-COMET-MTL mean | share > human | share > policy |\n|---|---|---|---|---|---|---|---|")
for c in classes:
    cells = []
    n = 0
    for m in models:
        vs = [d[c][m] for d in by.values() if c in d]
        n = len(vs)
        if not vs:
            cells += ["-", "-", "-"]
            continue
        gh = [d[c][m] > d["human"][m] for d in by.values() if c in d and "human" in d]
        gp = [d[c][m] > d["policy"][m] for d in by.values() if c in d and "policy" in d]
        cells += [f"{np.mean(vs):.3f}", f"{np.mean(gh):.2f}" if c != "human" else "-", f"{np.mean(gp):.2f}" if c != "policy" else "-"]
    print(f"| {c} | {n} | " + " | ".join(cells) + " |")
