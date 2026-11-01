#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-61 report numbers of the graded entity penalty: languages with a channel enabled (v1 strict vs v2 soft), the human mean penalty, and the penalty of the real policy samples (agent A5, T=1) next to
the penalty of the human references of the same sources. Markdown on stdout."""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma_mt.rl import anchors_v2 as V

ap = argparse.ArgumentParser()
ap.add_argument("--cal", required=True)
ap.add_argument("--policy", required=True)
a = ap.parse_args()
cal = json.load(open(a.cal))
A = V.Anchors2(cal, v2_detectors=("entity2",))
A1 = V.Anchors2(cal, detectors=("entity",), v2_detectors=())
weak = set((cal.get("_meta") or {}).get("weak_languages") or [])
L = cal["languages"]
n1 = [i for i in L if A1.enabled(i, "entity")]
n2 = [i for i in L if A.enabled2(i, "entity2")]
w1 = [i for i in n1 if i in weak]
w2 = [i for i in n2 if i in weak]
print(f"entity detector enabled: v1 strict {len(n1)} languages ({len(w1)} weak: {' '.join(sorted(w1))}); v2 soft {len(n2)} languages ({len(w2)} weak of {len(weak)}: {' '.join(sorted(w2))})")
chs = collections.Counter()
for i in n2:
    for c in ("num", "name"):
        if ((L[i].get("entity2") or {}).get(c) or {}).get("enabled"):
            chs[c] += 1
print(f"channels enabled: numbers {chs['num']}, names {chs['name']}\n")
pol = collections.defaultdict(list)
for l in open(a.policy, encoding="utf-8"):
    r = json.loads(l)
    pol[r["iso"]].append(r)
print("| lang | weak | policy samples | policy mean penalty (x0.5) | human mean penalty (x0.5) | policy samples with penalty > 0 | human refs with penalty > 0 |\n|---|---|---|---|---|---|---|")
rows = []
for iso in sorted(n2):
    if iso not in pol:
        continue
    pp, hp, seen = [], [], set()
    for r in pol[iso]:
        pu, nj = A.entity2_penalty(iso, r["src"], r["hyp"])
        if nj:
            pp.append(pu)
        if r["src"] not in seen:
            seen.add(r["src"])
            hu, hj = A.entity2_penalty(iso, r["src"], r["ref"])
            if hj:
                hp.append(hu)
    if len(pp) >= 20 and hp:
        rows.append((iso, iso in weak, len(pp), 0.5 * np.mean(pp), 0.5 * np.mean(hp), np.mean(np.asarray(pp) > 0), np.mean(np.asarray(hp) > 0)))
        print(f"| {iso} | {'w' if iso in weak else ''} | {len(pp)} | {rows[-1][3]:.3f} | {rows[-1][4]:.3f} | {rows[-1][5]:.2f} | {rows[-1][6]:.2f} |")
if rows:
    print(f"\nmean over {len(rows)} languages: policy {np.mean([r[3] for r in rows]):.3f}, human {np.mean([r[4] for r in rows]):.3f}")
