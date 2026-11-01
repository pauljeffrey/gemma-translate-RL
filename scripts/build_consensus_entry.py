#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write a calibration file = BASE + a QE-free consensus entry for one language built from the CALIBRATION half of its human reference sets (D-43).

    python scripts/build_consensus_entry.py --base calibration/shaping_flores_dev_v5.json --iso efi --calibration refs/efi_calibration.jsonl \\
        --weight 0.5 --validation val_efi.json --out calibration/shaping_flores_dev_v6_efi_consensus.json

BASE is never modified. The entry takes the `lid` block of the base entry (measured separately, D-37), the human-reference statistics of
`rl/calibrate_refs.reference_stats` (len_ratio, diacritics, lexical, omission) and the advantage weight given by --weight (the D-43 decision rule applied to the
validation: `calibrate_refs.weight_from_within_prompt_spearman`). Other languages' entries are copied unchanged.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import sys

from gemma_mt.rl import calibrate_refs as C


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--iso", required=True)
    ap.add_argument("--calibration", required=True, help="jsonl of the CALIBRATION half (scripts/build_reference_split.py); never the hold-out")
    ap.add_argument("--weight", type=float, required=True)
    ap.add_argument("--weight-reason", default="")
    ap.add_argument("--validation", default=None, help="validation json (scripts/validate_consensus.py) to cite in the provenance block")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if "holdout" in a.calibration:
        sys.exit("refusing to calibrate on a file named *holdout*")
    base = json.load(open(a.base, encoding="utf-8"))
    rows = [json.loads(l) for l in open(a.calibration, encoding="utf-8") if l.strip()]
    stats = C.reference_stats(rows)
    old = base.get(a.iso) or {}
    prov = {"sets": sorted({r["source"] for r in rows}), "pairs_by_set": {s: sum(1 for r in rows if r["source"] == s) for s in sorted({r["source"] for r in rows})},
            "split": f"calibration half = sha1('{C.SPLIT_SALT}' + NFC English) < 0.5 (hold-out never used)", "english_seen_in_sft": sum(bool(r.get("english_seen")) for r in rows),
            "pair_seen_in_sft": sum(bool(r.get("pair_seen")) for r in rows), "created": datetime.date.today().isoformat(), "weight_reason": a.weight_reason}
    if a.validation:
        v = json.load(open(a.validation, encoding="utf-8"))
        prov["validation"] = v.get("summary", v)
    entry = C.consensus_entry(stats, old.get("lid"), a.weight, prov)
    out = copy.deepcopy(base)
    out[a.iso] = entry
    out["_meta_v6_" + a.iso] = {"created": datetime.date.today().isoformat(), "base": a.base.split("/")[-1] + " (unchanged)",
                                "change": f"{a.iso}: QE-free consensus entry from human references, advantage weight {a.weight}", "decision": "D-43"}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: entry[k] for k in entry if k not in ("diacritics", "lid", "lexical")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
