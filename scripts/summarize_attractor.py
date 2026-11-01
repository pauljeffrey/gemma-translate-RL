#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Markdown table from the JSON of scripts/attractor_study.py (D-61)."""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--attractor", required=True)
a = ap.parse_args()
R = json.load(open(a.attractor))
ws = ["func0.3_pad1", "func1_pad1", "func2_pad2", "gate0.9_func0.3_pad1"]
print(f"languages ({len(R)}): {' '.join(sorted(R))}\n")
print("reward = consensus value (mean chrF++ vs the other members -> [-2, 2.25]) minus penalties, floored at -2. Cells: mean reward of the ATTACKERS / of the honest policy members / of the human reference, and the share of prompts where the attackers' group-normalised advantage is positive.\n")
print("| case | no penalty (attack / honest / human / adv>0) | " + " | ".join(ws) + " |\n|---|---|" + "---|" * len(ws))
for k in sorted({k for v in R.values() for k in v if "@" in k}):
    v = [R[i][k] for i in R if k in R[i]]
    raw = " / ".join(f"{np.mean([x[m] for x in v]):.2f}" for m in ("attack_raw", "honest_raw", "human_raw", "attack_adv_pos_raw"))
    cells = [" / ".join(f"{np.mean([x[w][m] for x in v]):.2f}" for m in ("attack", "honest", "human", "adv_pos")) for w in ws]
    print(f"| {k} (n={len(v)}) | {raw} | " + " | ".join(cells) + " |")
