#!/usr/bin/env python
"""D-59: add the `lexicon` block (human low-quantile threshold of the GUARDED lexicon overlap) to an anchors calibration for the languages whose lexicon validated.

  python scripts/add_lexicon_calibration.py --anchors calibration/anchors_v1.json --lexicon-dir lexicon --human human_units.jsonl --out calibration/anchors_v1_lex.json [--sets flores_dev]

The threshold is fitted on FLORES+ dev human references only (open-domain; the lexicons were built without FLORES, so the overlap of these references is an honest human
distribution). Every other block of the input file is copied unchanged. `rl.anchors_calibrate.fit_lexicon` is the fitting rule of the main builder.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--lexicon-dir", required=True)
    ap.add_argument("--human", required=True, help="human_units.jsonl of build_anchors_calibration.py --dump-human: {iso, src, ref, set}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=["flores_dev"])
    ap.add_argument("--alpha", type=float, default=0.5, help="quantile of the HUMAN overlap used as threshold t. Default 0.5 (median): the overlap is a coarse recall signal, 9-32 %% of human FLORES+ references have overlap 0, so a 3 %% quantile is 0 and the detector would be off; deficit = (t - overlap) / t below the median, weight <= 0.25 per language")
    a = ap.parse_args()
    from gemma_mt.rl import anchors as A, anchors_calibrate as K, lexicon as L

    cal = json.load(open(a.anchors, encoding="utf-8"))
    alpha = a.alpha
    units = defaultdict(list)
    for line in open(a.human, encoding="utf-8"):
        d = json.loads(line)
        if d.get("set") in a.sets:
            units[d["iso"]].append(d)
    done, skipped = {}, {}
    for iso in sorted(cal["languages"]):
        lex = L.load_lexicon(iso, a.lexicon_dir)
        if lex is None or not lex.validated:
            continue
        b = K.fit_lexicon(lambda s, r, _i=iso: A.lexicon_overlap(_i, s, r, a.lexicon_dir, True), units.get(iso, []), alpha)
        if b is None:
            skipped[iso] = len(units.get(iso, []))
            continue
        b["guarded"] = True
        b["sets"] = list(a.sets)
        cal["languages"][iso]["lexicon"] = b
        done[iso] = (round(b["t"], 3), round(b["p50"], 3), b["n"], round(b["fp"], 3), b["enabled"])
    cal["_meta"]["lexicon"] = {"lexicon_dir": a.lexicon_dir, "sets": list(a.sets), "alpha": alpha, "languages": sorted(done), "note": "D-59: guarded overlap, FLORES+ dev human references, threshold = human median (the 3 % quantile is 0 for all 10 languages: 9-32 % of human references have no covered word hit)"}
    json.dump(cal, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("lexicon blocks:", json.dumps(done, ensure_ascii=False), "| skipped (too few judged):", skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
