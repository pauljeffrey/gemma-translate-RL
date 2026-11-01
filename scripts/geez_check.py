#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-61: the Ge'ez-script languages (amh, tir) under the v2 detectors. Owner 2026-10-22: they must not be false-positived by the n-gram / stop-word / copy detectors.

Held-out human references (fold B) of amh and tir, in four forms: as they are; word separator U+1361 instead of the space; Ethiopic full stop U+1362 / comma U+1363 replaced by '.' / ','; and mixed with
one Latin-script name inserted. For each: the share of references with a deficit > 0 per detector. The calibration is the fold-A one (scripts/build_anchors_v2_calibration.py --fold A)."""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_anchors_v2 as E
from gemma_mt.rl import anchors_v2 as V
from gemma_mt.rl import textstats as T

ap = argparse.ArgumentParser()
ap.add_argument("--cal", required=True)
ap.add_argument("--human", required=True)
ap.add_argument("--bigrams", default=None)
ap.add_argument("--langs", default="amh,tir")
a = ap.parse_args()
cal = json.load(open(a.cal))
A = V.Anchors2(cal, detectors=("repeat",), bigram_path=a.bigrams)
forms = {
    "as is": lambda s: s,
    "U+1361 separator": lambda s: s.replace(" ", "፡"),
    "U+1362/1363 -> . ,": lambda s: s.replace("።", ".").replace("፣", ",").replace("፤", ";"),
    "Latin name inserted": lambda s: s + " Addis Ababa University",
}
units = collections.defaultdict(list)
seen = set()
for l in open(a.human):
    r = json.loads(l)
    if r["iso"] in a.langs.split(",") and r["set"] != "smol_doc" and not r["set"].endswith("_doc.jsonl") and E.fold_of(r["iso"], r["src"]) == 1:
        k = (r["iso"], r["src"], r["ref"])
        if k not in seen:
            seen.add(k)
            units[r["iso"]].append(r)
print("| lang | form | n | copy | func | pad | repeat (v1) |\n|---|---|---|---|---|---|---|")
for iso, us in sorted(units.items()):
    for name, fn in forms.items():
        c = f = p = rp = n = 0
        for u in us:
            mt = fn(u["ref"])
            sw, ow = T.out_tokens(u["src"]), T.out_tokens(mt)
            cf = A.copy_fraction(iso, u["src"], mt)
            fd = A.func_deficit(iso, sw, ow)
            pd = A.pad_deficit(iso, ow)
            rd = A.repeat_deficit(iso, A.prepare(iso, u["src"], mt))
            n += 1
            c += bool(cf and cf > 0)
            f += bool(fd and fd > 0)
            p += bool(pd and pd > 0)
            rp += bool(rd and rd > 0)
        print(f"| {iso} | {name} | {n} | {c / n:.3f} | {f / n:.3f} | {p / n:.3f} | {rp / n:.3f} |")
