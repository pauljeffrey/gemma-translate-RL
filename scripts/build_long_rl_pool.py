#!/usr/bin/env python
"""Pool of the long-document RL stage (2026-11-04): `<out>/{science,science_b2048,general,general_b2048,news,news_long,ref}/<iso>.parquet`.

Every row of the pool comes from a source that already passed the protected-index scan (decon.py R1 exact sentence, R2 contained sentence, R3 shared word 8-gram,
protected-2026-10-28.3): sci_rl and clean_v3_385m (scan of the 385M plan), the news pool (scan of the news build) and the b2048 rows, which were dropped unscanned
before and were scanned on 2026-11-03 (`data_pipeline/sft_clean/scan_long_pool.py`). The loader walks the directory; the domain comes from the `domain` column (general when absent).

    python build_long_rl_pool.py --prompts /workspace/clean_rerun/prompts --extra /workspace/clean_rerun/prompts/_long_extra --out /workspace/clean_rerun/prompts/long_rl
`_long_extra` holds sci_v2_b2048/, v3_b2048/ (scan output) and news_long/ (b512 / b1024 news windows).  Per-language caps (seeded, no replacement) keep the pool small: the stage
consumes 2,560 prompts in all, the cap only has to leave every language and bin enough rows.
"""
import argparse, collections, glob, json, os, shutil, zlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

CAPS = {"science": {"b64": 150, "b256": 150}, "general": {"b64": 150, "b256": 150, "b512": 250, "b1024": 250}, "news": {"b64": 100, "b256": 100}}


def sample(t, caps, seed, key):
    lb = np.asarray(t.column("length_bin").to_pylist())
    keep = []
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])
    for b in sorted(set(lb)):
        ix = np.nonzero(lb == b)[0]
        cap = caps.get(b)
        if cap is not None and len(ix) > cap:
            ix = np.sort(rng.choice(ix, size=cap, replace=False))
        keep.append(ix)
    return t.take(pa.array(np.concatenate(keep))) if keep else t


def put(t, d, iso, domain=None):
    if t.num_rows == 0:
        return 0
    if domain and "domain" not in t.column_names:
        t = t.append_column("domain", pa.array([domain] * t.num_rows, pa.string()))
    os.makedirs(d, exist_ok=True)
    pq.write_table(t, os.path.join(d, iso + ".parquet"), compression="zstd")
    return t.num_rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prompts", required=True); ap.add_argument("--extra", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    stats = collections.defaultdict(collections.Counter)
    def rd(path):
        return pq.read_table(path)
    for f in sorted(glob.glob(os.path.join(a.prompts, "sci_rl", "*.parquet"))):
        iso = os.path.basename(f)[:-8]; t = sample(rd(f), CAPS["science"], a.seed, "sci" + iso)
        for b, n in collections.Counter(t.column("length_bin").to_pylist()).items(): stats["science"][b] += n
        put(t, os.path.join(a.out, "science"), iso, "science")
    for f in sorted(glob.glob(os.path.join(a.prompts, "sci_rl", "ref", "*.parquet"))):
        iso = os.path.basename(f)[:-8]; t = rd(f)
        for b, n in collections.Counter(t.column("length_bin").to_pylist()).items(): stats["ref"][b] += n
        put(t, os.path.join(a.out, "ref"), iso, "ref")
    for f in sorted(glob.glob(os.path.join(a.prompts, "clean_v3_385m", "*.parquet"))):
        iso = os.path.basename(f)[:-8]; t = sample(rd(f), CAPS["general"], a.seed, "gen" + iso)
        for b, n in collections.Counter(t.column("length_bin").to_pylist()).items(): stats["general"][b] += n
        put(t, os.path.join(a.out, "general"), iso, "general")
    for f in sorted(glob.glob(os.path.join(a.prompts, "clean_v3_385m", "news", "*.parquet"))):
        iso = os.path.basename(f)[:-8]; t = sample(rd(f), CAPS["news"], a.seed, "news" + iso)
        for b, n in collections.Counter(t.column("length_bin").to_pylist()).items(): stats["news"][b] += n
        put(t, os.path.join(a.out, "news"), iso, "news")
    for sub, dom, outd in (("sci_v2_b2048", "science", "science_b2048"), ("v3_b2048", "general", "general_b2048"), ("news_long", "news", "news_long")):
        for f in sorted(glob.glob(os.path.join(a.extra, sub, "*.parquet"))):
            iso = os.path.basename(f)[:-8]; t = rd(f)
            for b, n in collections.Counter(t.column("length_bin").to_pylist()).items(): stats[dom][b] += n
            put(t, os.path.join(a.out, outd), iso, dom)
    st = {d: dict(sorted(c.items())) for d, c in stats.items()}
    st["caps"] = CAPS; st["seed"] = a.seed
    json.dump(st, open(os.path.join(a.out, "_pool_stats.json"), "w"), indent=1)
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
