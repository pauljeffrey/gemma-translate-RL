#!/usr/bin/env python3
"""Merge specialist_yor/ (yor only, run with --only-langs yor) into specialist/ (19 languages) -> 20 languages, and recompute the per-language tables.
    python merge_yor.py --out out"""
import argparse, csv, os, sys
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import specialist as SP
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--sub-add", default="specialist_yor"); a = ap.parse_args()
t1 = pq.read_table(f"{a.out}/specialist/pairs-0000.parquet"); t2 = pq.read_table(f"{a.out}/{a.sub_add}/pairs-0000.parquet")
assert not set(t1["iso"].to_pylist()) & set(t2["iso"].to_pylist()), "languages overlap"
os.replace(f"{a.out}/specialist/pairs-0000.parquet", f"{a.out}/specialist/pairs-0000.19languages.parquet.bak")
t = pa.concat_tables([t1, t2.select(t1.column_names)])
d = t.to_pydict(); n = len(d["iso"])
SP.write_stats([{k: d[k][i] for k in d} for i in range(n)], a.out, "specialist")
for name in ("excluded.csv", "removed_by_filter_and_duplicates.csv"):
    p1, p2 = f"{a.out}/specialist/stats/{name}", f"{a.out}/{a.sub_add}/stats/{name}"
    if os.path.exists(p2):
        rows = list(csv.reader(open(p2)))[1:]
        with open(p1, "a", newline="") as f:
            csv.writer(f).writerows(rows)
print("merged:", n, "rows,", len(set(d["iso"])), "languages")
