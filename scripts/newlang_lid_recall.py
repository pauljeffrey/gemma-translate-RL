#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GlotLID recall of every reference set of the new-language study (D-47): the measurement that decides whether a GlotLID guard can be switched on for a language (D-37 rule: human recall >= 0.90).

For each (language, reference set): GlotLID v3 top-1 on ALL reference sentences (997 FLORES+ dev / 863 SmolSent / 1,997 NTREX ...), recall = share whose top-1 is in the language's accepted
label set, the three most frequent top-1 labels (the confusion partners), recall on text prefixes of 15 / 30 / 60 / 120 characters, and the share of English text wrongly accepted
(`cross_accept` on 300 English sentences of the same set).

    python scripts/newlang_lid_recall.py --out DIR --lid-model model_v3.bin [--calibration calibration/shaping_flores_dev_v6.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import newlang_probe_sets as S  # noqa: E402
import newlang_registry as R  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--lid-model", default=None)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--keys", nargs="*", default=None)
    a = ap.parse_args(argv)
    from gemma_mt.rl.lid import GlotLIDClassifier
    from gemma_mt.rl.text import nfc

    lid = GlotLIDClassifier(path=a.lid_model)
    lid._load()
    cal = json.load(open(a.calibration, encoding="utf-8")) if a.calibration else {}
    res = {}
    for key, reg in {**R.NEW, **R.YARDSTICK}.items():
        if a.keys and key not in a.keys:
            continue
        labels = tuple(R.NEW[key]["glot"]) if key in R.NEW else tuple(((cal.get(key) or {}).get("lid") or {}).get("labels") or R.YARD_GLOT_FALLBACK[key])
        for spec in reg["sets"]:
            try:
                rows = S.load_set(spec)
            except Exception as e:                                          # noqa: BLE001
                res[f"{key}__{spec}"] = {"error": str(e)[:120]}
                continue
            refs = [nfc(r["reference"]) for r in rows]
            top = lid.top(refs, 1)
            hit = [t[0][0] in labels for t in top]
            conf = {}
            for t in top:
                conf[t[0][0]] = conf.get(t[0][0], 0) + 1
            prefix = {}
            for L in (15, 30, 60, 120):
                sub = [r[:L] for r in refs if len(r) >= L]
                tp = lid.top(sub, 1) if sub else []
                prefix[str(L)] = round(sum(t[0][0] in labels for t in tp) / max(1, len(tp)), 3) if tp else None
            en = [r["english"] for r in rows[:300]]
            eng_top = lid.top(en, 1)
            res[f"{key}__{spec}"] = {"n": len(refs), "accepted": list(labels), "recall": round(sum(hit) / len(hit), 4), "top_labels": dict(sorted(conf.items(), key=lambda kv: -kv[1])[:5]),
                                      "prefix_recall": prefix, "english_accepted": round(sum(t[0][0] in labels for t in eng_top) / max(1, len(eng_top)), 4),
                                      "mean_p1": round(sum(t[0][1] for t in top) / len(top), 3)}
            print(key, spec, res[f"{key}__{spec}"]["recall"], res[f"{key}__{spec}"]["top_labels"], flush=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
