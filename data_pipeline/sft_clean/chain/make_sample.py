#!/usr/bin/env python3
"""20k-pair test sample for the chain: contiguous runs (up to --per-group consecutive rows) of every (language, source) group, so windows have real consecutive pairs.
    python make_sample.py --src data data_devhalf specialist --out sample_pairs --n 20000 --per-group 120"""
import argparse, collections, os, sys
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import shard_files
ap = argparse.ArgumentParser(); ap.add_argument("--src", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=20000); ap.add_argument("--per-group", type=int, default=120)
a = ap.parse_args()
cnt = collections.Counter(); rows = collections.defaultdict(list); total = 0
for f in shard_files(a.src):
    t = pq.read_table(f).to_pydict(); names = list(t)
    for i in range(len(t["iso"])):
        g = (t["iso"][i], t["source_dataset"][i])
        if cnt[g] < a.per_group:
            cnt[g] += 1; total += 1
            for k in names: rows[k].append(t[k][i])
        if total >= a.n: break
    if total >= a.n: break
os.makedirs(a.out, exist_ok=True)
pq.write_table(pa.table(rows), a.out + "/pairs-0000.parquet", compression="zstd")
print(total, "pairs,", len(set(rows["iso"])), "languages,", len(cnt), "groups")
