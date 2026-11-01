#!/usr/bin/env python3
"""Split hub/extra2/* shards into rows of languages that exist in the chain registry (gemma_mt.languages.LANGUAGES: they can be gated, windowed and packed with the current tokenizer)
-> hub/extra2k/<source>/..., and rows of languages that do not -> hub/extra2_hold/<source>/... (kept for a later decision on tokenizer tags). Keys are unchanged, so QE scores still apply."""
import glob, os, sys
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, "/workspace/work/code")
from gemma_mt import languages as GL
known = set(GL.LANGUAGES)
n_k = n_h = 0
for f in sorted(glob.glob("/workspace/work/hub/extra2/**/*.parquet", recursive=True)):
    src = os.path.relpath(os.path.dirname(f), "/workspace/work/hub/extra2")
    if src.startswith("_stats"): continue
    t = pq.read_table(f); m = pc.is_in(t["iso"], value_set=pa.array(sorted(known)))
    k, h = t.filter(m), t.filter(pc.invert(m))
    for tbl, root in ((k, "extra2k"), (h, "extra2_hold")):
        if tbl.num_rows:
            d = f"/workspace/work/hub/{root}/{src}"; os.makedirs(d, exist_ok=True)
            pq.write_table(tbl, os.path.join(d, os.path.basename(f)), compression="zstd")
    n_k += k.num_rows; n_h += h.num_rows
print("known-language rows", n_k, "| held (language not in the registry)", n_h)
