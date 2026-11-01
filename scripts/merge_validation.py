#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Merge the shards of validate_consensus.py runs of ONE language into a single summary file (D-43): python scripts/merge_validation.py --out val_efi.json val_efi_0.json val_efi_1.json"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_consensus import analyse  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    groups, rows, srcs = [], [], 0
    for p in a.files:
        d = json.load(open(p, encoding="utf-8"))
        groups += d["groups"]
        rows += d["rows"]
        srcs += d["summary"]["sources"]
        iso, model = d["summary"]["iso"], d["summary"]["model"]
    s = analyse(groups)
    s.update({"iso": iso, "model": model, "sources": srcs, "group": d["summary"]["group"],
              "status_counts": {k: sum(1 for r in rows if r["status"] == k) for k in sorted({r["status"] for r in rows})},
              "english_seen_share": sum(bool(r.get("english_seen")) for r in rows) / max(len(rows), 1)})
    json.dump({"summary": s, "groups": groups, "rows": rows}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
