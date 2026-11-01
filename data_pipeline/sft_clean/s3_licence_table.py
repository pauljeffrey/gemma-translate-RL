#!/usr/bin/env python3
"""Licence / source / domain / bin counts of the CLEAN prompt pools (agent S3, 2026-10-28). Reads the clean parquet files (columns source_dataset, licence, domain, length_bin) and
writes one CSV: pool, source_dataset, domain, licence, length_bin, prompts, unique_english_windows(approx by xxh64 of english, per pool).
    python s3_licence_table.py --dirs out/clean_sci_v2 out/clean_v3 --out licence_by_source.csv
"""
import argparse, collections, csv, glob, os
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
cnt = collections.Counter()
for d in a.dirs:
    pool = os.path.basename(d.rstrip("/")).replace("clean_", "")
    for f in sorted(glob.glob(os.path.join(d, "*.parquet"))):
        t = pq.read_table(f, columns=[c for c in ("source_dataset", "licence", "domain", "length_bin") if c in pq.ParquetFile(f).schema_arrow.names]).to_pydict()
        n = len(t["length_bin"])
        doms = t.get("domain") or [None] * n
        for sd, dm, lc, lb in zip(t["source_dataset"], doms, t["licence"], t["length_bin"]):
            cnt[(pool, sd, dm or "general", lc, lb)] += 1
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["pool", "source_dataset", "domain", "licence", "length_bin", "prompts"])
    for k, v in sorted(cnt.items()):
        w.writerow(list(k) + [v])
print(len(cnt), "rows")
