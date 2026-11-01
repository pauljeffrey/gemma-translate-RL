#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thresholds of the science anchor (rl/sci_anchor.py, DECISIONS.md D-52) from the HUMAN references of AfriScience-MT TRAIN (sentence level, six languages).

    python scripts/build_science_anchor_calibration.py --corpus-dir DIR/corpus --out calibration/science_anchor_v1.json

For every (English sentence, human translation) pair the five features of `sci_anchor.features` are computed; the file holds, per feature, the 10th percentile / median / mean over the
pairs that have the feature, pooled over the languages (`default`, used for every language without an entry) and per language (only with >= 150 measured pairs). The reward penalises a
chunk only where its feature is BELOW the human 10th percentile (deficit = (p10 - f) / p10), so a faithful translation is never touched. dev and test are not read.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=200)
    ap.add_argument("--min-n-language", type=int, default=150)
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import eval_afriscience as E
    from gemma_mt.rl import sci_anchor

    data = E.load_corpus(a.corpus_dir, "train")
    rows = [(iso, r["src"], r["ref"]) for iso, blk in data.items() for r in blk["sentences"]]
    blob = sci_anchor.build_calibration(rows, min_n=a.min_n, min_n_language=a.min_n_language)
    blob["source"] = "dsfsi/afriscience_mt train, human references, sentence level; p10 of each feature over the pairs that have it"
    blob["n_pairs"] = len(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=1, ensure_ascii=False)
    print(json.dumps({"pooled": {k: [round(v["p10"], 3), round(v["p50"], 3), v["n"]] for k, v in blob["default"].items()},
                      "per_language_p10": {i: {k: round(v["p10"], 2) for k, v in d.items()} for i, d in blob["languages"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
