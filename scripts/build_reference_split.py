#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Collect the English -> X human reference pairs of one language from the rare-language test sets (D-42), de-duplicate by English, split them
DETERMINISTICALLY (hash of the English sentence) into a CALIBRATION half and a HOLD-OUT half (D-43), and write both as jsonl.

    python scripts/build_reference_split.py --iso efi --spec smol_sent=efi ibom_mt=efi smol_doc=efi \\
        --test-sets /workspace/sft_build/bench/test_sets --out-dir /workspace/sft_build/efi_calib/refs

Only the calibration half may feed a calibration entry. The rows carry the contamination flags of `_contamination/` (English side seen in the SFT windows,
reference pair seen). GATITOS (single words) is not a source here on purpose.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

from gemma_mt.rl import calibrate_refs as C


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iso", required=True)
    ap.add_argument("--spec", nargs="+", required=True, help="set=file_code, e.g. smol_sent=efi ibom_mt=efi smol_doc=efi")
    ap.add_argument("--test-sets", required=True)
    ap.add_argument("--contamination-dir", default=None, help="default: <test-sets>/_contamination")
    ap.add_argument("--frac", type=float, default=0.5, help="share of the pairs in the calibration half")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    spec = dict(s.split("=", 1) for s in a.spec)
    rows = C.load_pairs(a.test_sets, a.iso, spec, a.contamination_dir or os.path.join(a.test_sets, "_contamination"), sources=tuple(spec))
    cal, hold = C.split_rows(rows, a.frac)
    os.makedirs(a.out_dir, exist_ok=True)
    summary = {"iso": a.iso, "spec": spec, "frac": a.frac, "salt": C.SPLIT_SALT, "pairs": len(rows), "calibration": len(cal), "holdout": len(hold)}
    for name, rs in (("calibration", cal), ("holdout", hold)):
        with open(os.path.join(a.out_dir, f"{a.iso}_{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = collections.Counter(r["source"] for r in rs)
        seen = [r["english_seen"] for r in rs if r["english_seen"] is not None]
        pair = [r["pair_seen"] for r in rs if r["pair_seen"] is not None]
        summary[name + "_by_source"] = dict(c)
        summary[name + "_english_seen"] = f"{sum(seen)}/{len(seen)}"
        summary[name + "_pair_seen"] = f"{sum(pair)}/{len(pair)}"
    with open(os.path.join(a.out_dir, f"{a.iso}_split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
