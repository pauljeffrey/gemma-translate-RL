#!/usr/bin/env python3
"""Attach the contamination flags of a flag directory (contamination_<benchmark>.json) to a predictions.jsonl (rows keep everything else).
    python scripts/attach_flags.py --pred in.jsonl --flags DIR --out out.jsonl
Rows of a benchmark without a flag file, or missing in the file, get None (never 'clean'). Prints the counts."""
import argparse, json, os, sys, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemma_mt.eval import contamination as C

ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--flags", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.pred, encoding="utf-8") if l.strip()]
files = C.default_flag_files(a.flags, sorted({r["benchmark"] for r in rows}))
miss = C.apply_flags(rows, files)
cnt = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    b = r["benchmark"]; cnt[b]["n"] += 1
    if r["contaminated"] is None: cnt[b]["noflag"] += 1
    else:
        cnt[b]["contaminated"] += bool(r["contaminated"]); cnt[b]["pair_seen"] += bool(r["pair_seen"]); cnt[b]["near"] += bool(r["near_dup"])
with open(a.out, "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("flag files:", files); print("missing:", miss)
for b, c in sorted(cnt.items()): print(b, dict(c))
