#!/usr/bin/env python3
"""SMOKE helper (agent C2): a small sample of the clean SFT pair shards with contiguous runs per (language, source) so that windows.py has real consecutive pairs.
Unlike chain/make_sample.py it keeps only the columns the shards have in common (english target iso source_dataset licence flipped), so main and specialist shards can be mixed.

    python s4_make_sample.py --src /workspace/work/out/final /workspace/work/out/specialist --out sample_pairs --n 40000 --per-group 150
"""
import argparse
import collections
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq

COLS = ["english", "target", "iso", "source_dataset", "licence", "flipped"]
ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=40000)
ap.add_argument("--per-group", type=int, default=150)
a = ap.parse_args()
files = [sorted(glob.glob(os.path.join(d, "*.parquet"))) for d in a.src]
order = [f for grp in zip(*[g for g in files if g]) for f in grp] + [f for g in files for f in g[len(min(files, key=len)):]]
cnt, rows, total = collections.Counter(), {c: [] for c in COLS}, 0
for f in order:
    pf = pq.ParquetFile(f)
    for batch in pf.iter_batches(batch_size=50000, columns=COLS):
        d = batch.to_pydict()
        for i in range(len(d["iso"])):
            g = (d["iso"][i], d["source_dataset"][i])
            if cnt[g] < a.per_group:
                cnt[g] += 1
                total += 1
                for c in COLS:
                    rows[c].append(d[c][i])
    if total >= a.n:
        break
os.makedirs(a.out, exist_ok=True)
pq.write_table(pa.table(rows), os.path.join(a.out, "pairs-0000.parquet"), compression="zstd")
print(total, "pairs,", len(set(rows["iso"])), "languages,", len(cnt), "groups")
