#!/usr/bin/env python3
"""Sanity after the gate: pairs per language (unique, NC removed) BEFORE THE QE CAP, scored (= inside the cap or scored earlier), and AFTER THE GATE, with the table of languages under 2,000 pairs.
    python sanity_after_gate.py --src hub/data hub/data_devhalf hub/specialist hub/extra --scores work/scores_cpu work/scores_gpu --keep work/gate/keep_cal.parquet --out work/gate/sanity_after_gate
A pair that was not scored is not kept (the keep list is the source of truth).
Writes <out>.csv (every language) and <out>_under_2000.csv; prints the under-2,000 table. Needs PYTHONPATH with the chain dir (common.py) and GEMMA_MT_PKG."""
import argparse, collections, csv, os, sys
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, os.environ.get("CHAIN", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chain")))
from common import key_of, shard_files, is_nc, assert_clean_src
ap = argparse.ArgumentParser(); ap.add_argument("--src", nargs="+", required=True); ap.add_argument("--keep", required=True); ap.add_argument("--out", required=True); ap.add_argument("--min", type=int, default=2000); ap.add_argument("--scores", nargs="*", default=[])
a = ap.parse_args()
assert_clean_src(a.src)
keep = set(pq.read_table(a.keep)["key"].to_pylist())
scored_keys = set()
if a.scores:
    import pyarrow.dataset as pads
    for d_ in a.scores:
        if os.path.isdir(d_) and os.listdir(d_): scored_keys.update(pads.dataset(d_, format="parquet").to_table(columns=["key"])["key"].to_pylist())
seen = set(); before = collections.Counter(); after = collections.Counter(); nc = collections.Counter(); scored = collections.Counter()
for f in shard_files(a.src):
    pf = pq.ParquetFile(f); cols = [c for c in ("english", "target", "iso", "key", "licence") if c in pf.schema_arrow.names]
    for rb in pf.iter_batches(batch_size=100000, columns=cols):
        t = rb.to_pydict(); n = len(t["iso"]); lic = t.get("licence") or [None] * n
        for i in range(n):
            if is_nc(lic[i]):
                nc[t["iso"][i]] += 1; continue
            k = t["key"][i] if "key" in t else key_of(t["english"][i], t["iso"][i], t["target"][i])
            if k in seen: continue
            seen.add(k); before[t["iso"][i]] += 1
            if k in scored_keys: scored[t["iso"][i]] += 1
            if k in keep: after[t["iso"][i]] += 1
try:
    from gemma_mt.languages import ALL_ISOS
except Exception:
    ALL_ISOS = tuple()
isos = sorted(set(before) | set(ALL_ISOS) | set(nc), key=lambda i: -before.get(i, 0))
with open(a.out + ".csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["iso", "nc_rows_removed", "pairs_before_cap", "pairs_scored_after_cap", "pairs_after_gate", "kept_pct_of_before_cap", "under_2000"])
    for i in isos: w.writerow([i, nc[i], before[i], scored[i] if a.scores else "", after[i], round(100 * after[i] / max(before[i], 1), 1), int(after[i] < a.min)])
under = [i for i in isos if after[i] < a.min]
with open(a.out + "_under_2000.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["iso", "nc_rows_removed", "pairs_before_cap", "pairs_after_gate", "status"])
    for i in under: w.writerow([i, nc[i], before[i], after[i], "no data" if before[i] == 0 else ("all rows were NC" if before[i] == 0 and nc[i] else "under 2,000 after the cap and gate")])
print(f"{len(seen):,} unique commercial pairs, {sum(scored.values()):,} scored (inside the cap), {sum(after.values()):,} after the gate ({100 * sum(after.values()) / max(len(seen), 1):.1f}%), NC rows removed {sum(nc.values()):,}")
print("| iso | NC removed | pairs after NC | pairs after gate |\n|---|---:|---:|---:|")
for i in under: print(f"| {i} | {nc[i]:,} | {before[i]:,} | {after[i]:,} |")
