#!/usr/bin/env python3
"""Unique English windows (distinct xxh64 of the English text) before / after the S3 filter, per pool and length bin (the same window is a prompt of many languages).
    python s3_unique_counts.py --out-root out --dest reports/data/clean_rl_science_2026-10-28/unique_windows_by_bin.csv"""
import argparse, collections, csv, glob, os
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--out-root", required=True)
ap.add_argument("--dest", required=True)
a = ap.parse_args()
kept = collections.defaultdict(set)      # (pool, bin) -> hashes
rem = collections.defaultdict(set)
dom = collections.defaultdict(set)
for f in sorted(glob.glob(os.path.join(a.out_root, "clean_*", "*.parquet"))):
    pool = os.path.basename(os.path.dirname(f)).replace("clean_", "")
    t = pq.read_table(f, columns=["prov_en_xxh64", "length_bin"]).to_pydict()
    for h, b in zip(t["prov_en_xxh64"], t["length_bin"]):
        kept[(pool, b)].add(h)
for f in sorted(glob.glob(os.path.join(a.out_root, "removed_log", "*.parquet"))):
    pool = os.path.basename(f).split("__")[0]
    t = pq.read_table(f, columns=["prov_en_xxh64", "length_bin"]).to_pydict()
    for h, b in zip(t["prov_en_xxh64"], t["length_bin"]):
        rem[(pool, b)].add(h)
rows = []
for pool in sorted({k[0] for k in kept}):
    tk, tr = set(), set()
    for b in ("b64", "b256", "b512", "b1024"):
        k, r = kept[(pool, b)], rem[(pool, b)]
        tk |= k; tr |= r
        rows.append([pool, b, len(k) + len(r - k), len(k), len(r - k), round(100 * len(r - k) / max(1, len(k) + len(r - k)), 3)])
    rows.append([pool, "ALL b64-b1024 (distinct windows)", len(tk) + len(tr - tk), len(tk), len(tr - tk), round(100 * len(tr - tk) / max(1, len(tk) + len(tr - tk)), 3)])
os.makedirs(os.path.dirname(a.dest) or ".", exist_ok=True)
with open(a.dest, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["pool", "bin", "unique_windows_before", "unique_windows_after", "unique_windows_removed", "removed_pct"]); w.writerows(rows)
print(rows)
