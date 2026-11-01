#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""RL prompt files for the languages added to the consensus mode on 2026-11-01 (guw lue tll zne lub crs mfe arz). No reference translation is needed, so a language's file is a seeded draw of English
sources from the SAME pool the existing prompt files (rl_prompts_v2) were drawn from.

    python scripts/build_rl_prompts_new.py --pool pool/afr.parquet pool/hau.parquet pool/ibo.parquet pool/kin.parquet --out artifacts/rl_prompts_new [--isos guw lue ...] [--per-bin 5000] [--seed 1234]

Pool = the union of the --pool files (columns english, src_afr_tokens, length_bin, n_sentences, source_dataset, licence; parquet or jsonl), duplicates removed (case-folded English), rows with
src_afr_tokens above 1024 dropped. Per language: --per-bin rows from each of the bins b64 b256 b512 b1024 (4 x 5,000 = 20,000 like the 35 files of D-38), drawn without replacement with
numpy default_rng([seed, crc32(iso)], so the same input always gives the same file and a language never shares its draw with another. Output <out>/<iso>.parquet (or .jsonl), columns of the pool
plus `iso` and `language` (canonical display name). Deterministic: running twice gives byte-identical rows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zlib

import numpy as np

from gemma_mt.languages import CONSENSUS_ONLY, LANGUAGES

BINS = ("b64", "b256", "b512", "b1024")
COLS = ("english", "iso", "language", "src_afr_tokens", "length_bin", "n_sentences", "source_dataset", "licence")
DEFAULT_ISOS = ("guw", "lue", "tll", "zne", "lub", "crs", "mfe", "arz")


def read_rows(path):
    if path.endswith(".jsonl"):
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    import pyarrow.parquet as pq
    t = pq.read_table(path).to_pydict()
    return [dict(zip(t, vals)) for vals in zip(*t.values())]


def write_rows(path, rows):
    if path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({c: [r.get(c) for r in rows] for c in COLS}), path, compression="zstd")


def build_pool(paths):
    seen, pool = set(), []
    for p in paths:
        for r in read_rows(p):
            e = r["english"]; k = e.casefold().strip()
            if k in seen or int(r["src_afr_tokens"]) > 1024 or r["length_bin"] not in BINS:
                continue
            seen.add(k); pool.append({c: r.get(c) for c in ("english", "src_afr_tokens", "length_bin", "n_sentences", "source_dataset", "licence")})
    return pool


def draw(pool, iso, per_bin, seed):
    rng = np.random.default_rng([seed, zlib.crc32(iso.encode())])
    rows = []
    for b in BINS:
        cand = [r for r in pool if r["length_bin"] == b]
        assert len(cand) >= per_bin, f"pool has {len(cand)} rows in {b}, need {per_bin}"
        pick = rng.choice(len(cand), size=per_bin, replace=False)
        rows += [{**cand[i], "iso": iso, "language": LANGUAGES[iso].name} for i in sorted(pick.tolist())]
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--isos", nargs="*", default=list(DEFAULT_ISOS))
    ap.add_argument("--per-bin", type=int, default=5000); ap.add_argument("--seed", type=int, default=1234); ap.add_argument("--ext", default="parquet", choices=["parquet", "jsonl"])
    a = ap.parse_args(argv)
    for i in a.isos:
        assert i in CONSENSUS_ONLY, f"{i} is not a consensus RL language of languages.py"
    os.makedirs(a.out, exist_ok=True)
    pool = build_pool(a.pool); print(f"pool: {len(pool)} unique sources", {b: sum(1 for r in pool if r['length_bin'] == b) for b in BINS})
    for iso in a.isos:
        rows = draw(pool, iso, a.per_bin, a.seed); write_rows(os.path.join(a.out, f"{iso}.{a.ext}"), rows)
        print(iso, len(rows), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
