#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""P3 leakage audit: how many of the ENGLISH test sentences of the benchmarks occur in the English prompts of the RL pools (CPU only).

    cd /workspace/jobs/p3 && PYTHONPATH=/workspace/jobs/r7/gemma_mt_run nice -n 19 python p3_rl_leakage_audit.py \
        --rows /workspace/jobs/e1/sys_rl.jsonl.gz --pool v3=/workspace/sft_build/gemma/rl_prompts_v3 --pool sci=/workspace/jobs/r7/assets2/rl_prompts_sci_v2 \
        --out rl_leakage_audit.json --processes 8

Method = the SFT contamination study (`gemma_mt/eval/contamination.py`, same normalisation NFKC + casefold + non-alphanumeric runs -> one space, 8-token n-grams, same
`exact` / `sentence` / `near` definitions) with the RL prompt windows (parquet column `english`, one file per language) in place of the SFT windows:
    exact     the normalised English sentence equals a normalised sentence of a prompt window or occurs inside it as a contiguous token run
    sentence  equals a whole sentence of a window (the stricter, dev-filter rule)
    near      not exact, but >= 50% of its 8-token n-grams occur somewhere in the pool (sentences with fewer than 8 tokens have no n-grams)
    any8      at least one 8-token n-gram occurs in the pool
Every flag is computed against (a) the pool of the SAME language as the test row (the prompts that language was trained on) and (b) the union of all languages of the pool.
The rows are those of agent E1's per-sentence file (benchmark `b`, `iso`, `id`, `en`, `ref`, SFT flags `contaminated` = English in the SFT text, `pair_seen` = pair in the SFT text), i.e. the
exact benchmark rows of the pairwise comparison (161,272 rows). Output: JSON with, per benchmark and subset (all / no_pair / clean), the share of rows and of distinct English sentences.
Needs only pyarrow and the repo's gemma_mt.eval.contamination (standard library otherwise); no GPU, no dataset copy.
"""
import argparse
import glob
import gzip
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict

import pyarrow.parquet as pq

from gemma_mt.eval import contamination as C

_INDEX = None


def _windows(path):
    iso = os.path.basename(path)[:-8]
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=50_000, columns=["english"]):
        for en in batch.column("english").to_pylist():
            yield en, "", iso


def _scan(path):
    t0 = time.time()
    res = C.scan_windows(_INDEX, _windows(path))
    return os.path.basename(path)[:-8], res.exact, res.sentence, res.grams, res.windows, time.time() - t0


def scan_pool(index, directory, processes):
    global _INDEX
    _INDEX = index
    files = sorted(glob.glob(os.path.join(directory, "*.parquet")))
    out = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(processes) as pool:
        for k, (iso, ex, se, gr, nw, dt) in enumerate(pool.imap_unordered(_scan, files), 1):
            out[iso] = dict(exact=ex, sentence=se, grams=gr, windows=nw)
            print(f"[audit] {directory.split('/')[-1]} {k}/{len(files)} {iso}: {nw} windows {dt:.0f}s exact so far {len(ex)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--pool", action="append", required=True, help="NAME=DIR of parquet files (column english, file name = iso)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--processes", type=int, default=8)
    a = ap.parse_args()
    rows = []
    with gzip.open(a.rows, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append(dict(benchmark=r["b"], iso=r["iso"], id=str(r["id"]), english=r["en"], reference=r.get("ref", ""),
                             sft_exact=bool(r.get("contaminated")), pair_seen=bool(r.get("pair_seen"))))
    index = C.BenchIndex([dict(english=r["english"], iso=r["iso"], reference="") for r in rows])
    print(f"[audit] {len(rows)} rows, {len(index.texts)} distinct english sentences, {len(index.gram_to_strings)} distinct 8-grams", flush=True)
    for r in rows:
        r["sid"] = index.ids.get(C.normalise(r["english"]))
    ngrams_of = {}
    for i, s in enumerate(index.texts):
        ngrams_of[i] = set(C._grams(s.split(), index.n))
    pools = {}
    for spec in a.pool:
        name, d = spec.split("=", 1)
        pools[name] = scan_pool(index, d, a.processes)

    def flags(sid, iso, which):
        """exact / sentence / near / any8 of a sentence id against the pool of one language or the union (which = None)."""
        res = {}
        for pname, pool in pools.items():
            if which is None:
                srcs = list(pool.values())
            else:
                srcs = [pool[iso]] if iso in pool else []
            ex = any(sid in s["exact"] for s in srcs)
            se = any(sid in s["sentence"] for s in srcs)
            gset = set().union(*[s["grams"] for s in srcs]) if srcs else set()
            g = ngrams_of[sid]
            frac = (sum(1 for x in g if x in gset) / len(g)) if g else 0.0
            res[pname] = dict(exact=ex, sentence=se, near=(not ex and frac >= 0.5), any8=(len(g) > 0 and frac > 0))
        # union of the pools
        res["any_pool"] = {k: any(res[p][k] for p in pools) for k in ("exact", "sentence", "near", "any8")}
        return res

    # cache per (sid, iso) for the same-language flags, per sid for the union
    union_cache, same_cache = {}, {}
    agg = defaultdict(lambda: defaultdict(int))        # (benchmark, subset, scope, pool, flag) -> counts rows ; distinct handled separately
    seen_distinct = defaultdict(set)
    for r in rows:
        sid = r["sid"]
        if sid is None:
            continue
        if sid not in union_cache:
            union_cache[sid] = flags(sid, None, None)
        key = (sid, r["iso"])
        if key not in same_cache:
            same_cache[key] = flags(sid, r["iso"], r["iso"])
        subsets = ["all"]
        if not r["pair_seen"]:
            subsets.append("no_pair")
        if not r["sft_exact"]:
            subsets.append("clean")
        for sub in subsets:
            for scope, fl in (("same_language_pool", same_cache[key]), ("any_language_pool", union_cache[sid])):
                for pname, d in fl.items():
                    for fk, v in d.items():
                        agg[(r["benchmark"], sub, scope, pname, fk)]["rows_hit"] += int(v)
                    agg[(r["benchmark"], sub, scope, pname, "exact")]["rows"] += 1
            seen_distinct[(r["benchmark"], sub)].add(sid)
    out = dict(meta=dict(rows=len(rows), distinct_english=len(index.texts), pools={k: dict(files=len(v), windows=sum(x["windows"] for x in v.values())) for k, v in pools.items()},
                         method="gemma_mt.eval.contamination definitions, RL prompt windows instead of SFT windows"), results=[])
    # flatten
    table = defaultdict(dict)
    for (b, sub, scope, pname, fk), c in agg.items():
        n = agg[(b, sub, scope, pname, "exact")]["rows"]
        table[(b, sub, scope, pname)]["rows"] = n
        table[(b, sub, scope, pname)][fk] = c["rows_hit"]
    for (b, sub, scope, pname), d in sorted(table.items()):
        n = d["rows"]
        out["results"].append(dict(benchmark=b, subset=sub, scope=scope, pool=pname, rows=n, **{k: d[k] for k in ("exact", "sentence", "near", "any8")},
                                   **{k + "_pct": round(100.0 * d[k] / n, 3) for k in ("exact", "sentence", "near", "any8")}))
    # distinct-sentence view (any-language pool, any pool)
    out["distinct"] = []
    for (b, sub), sids in sorted(seen_distinct.items()):
        n = len(sids)
        row = dict(benchmark=b, subset=sub, distinct_english=n)
        for pname in list(pools) + ["any_pool"]:
            for fk in ("exact", "sentence", "near", "any8"):
                row[f"{pname}_{fk}"] = sum(1 for s in sids if union_cache[s][pname][fk])
        out["distinct"].append(row)
    json.dump(out, open(a.out, "w"), indent=1)
    print("[audit] wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
