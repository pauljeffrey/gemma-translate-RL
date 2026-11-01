#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Download the NLLB-mined `michsethowusu/<lang>-<lang>_sentence-pairs` repos (both languages among the 77) and convert them to parquet shards, one per language pair (D-46).

    python scripts/afri2afri_fetch_mined.py --repos mined_repos.json --out OUT/full_pairs/nllb_mined --bench bench/test_sets --extra-bench ... --workers 8

Per repo: download the single csv (score, <lang A>, <lang B>), NFC + whitespace normalise, drop empty rows, drop every row in which either text matches a held-out evaluation sentence
(`contam.LeakIndex`, exact + word-8-gram, any language), orient the pair so that src_iso < tgt_iso in `ALL_ISOS` order, write `<a>-<b>.parquet` (zstd). The csv is deleted after the
conversion (the cache directory belongs to this script). Resumable: finished shards are skipped. No licence is declared on these cards: `licence` = "unknown" and `licence_note` says why
the rows are treated as non-commercial.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from gemma_mt.afri2afri import contam                                   # noqa: E402
from gemma_mt.afri2afri.text import norm_text                           # noqa: E402
from gemma_mt.languages import ALL_ISOS                                 # noqa: E402

NOTE = "NLLB-v1 mined (CCMatrix-style) pair set republished by michsethowusu; no licence declared on the card; treated as non-commercial upstream (owner policy 2026-10-11 / 2026-10-20)"
_G = {}


def _init(bench, extra, cache):
    li = contam.LeakIndex()
    li.add_all(contam.iter_benchmark_texts(bench, extra))
    _G["leak"] = li
    _G["cache"] = cache


def _work(args):
    repo, a, b, out_dir, suffix = args
    import pyarrow as pa
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    ia, ib = ALL_ISOS.index(a), ALL_ISOS.index(b)
    lo, hi = (a, b) if ia < ib else (b, a)
    if a == b:                                   # e.g. `akan-twi`: one language tag on both sides, not a translation pair
        return repo, lo, hi, -4, 0, 0
    out = os.path.join(out_dir, f"{lo}-{hi}{suffix}.parquet")
    if os.path.exists(out) or os.path.exists(out + ".empty"):
        return repo, lo, hi, -1, 0, 0
    t0 = time.time()
    token = os.environ.get("HF_TOKEN")
    files = [f for f in HfApi(token=token).list_repo_files(repo, repo_type="dataset") if f.lower().endswith((".csv", ".parquet", ".tsv"))]
    if len(files) != 1:
        return repo, lo, hi, -2, len(files), 0
    path = hf_hub_download(repo, files[0], repo_type="dataset", cache_dir=_G["cache"], token=token)
    real = os.path.realpath(path)
    if files[0].lower().endswith(".parquet"):
        t = pq.read_table(real)
    else:
        t = pcsv.read_csv(real, parse_options=pcsv.ParseOptions(delimiter="\t" if files[0].lower().endswith(".tsv") else ","),
                          convert_options=pcsv.ConvertOptions(strings_can_be_null=False))
    names = t.column_names
    if len(names) != 3:
        return repo, lo, hi, -3, 0, 0
    score = t[names[0]].cast(pa.float32()).to_pylist()
    ta, tb = t[names[1]].to_pylist(), t[names[2]].to_pylist()
    leak = _G["leak"]
    n_in = len(ta)
    keep_s, keep_t, keep_sc = [], [], []
    dropped = 0
    for s, x, y in zip(score, ta, tb):
        if not x or not y:
            continue
        x, y = norm_text(str(x)), norm_text(str(y))
        if not x or not y:
            continue
        if leak.is_leak(x) or leak.is_leak(y):
            dropped += 1
            continue
        if ia < ib:
            keep_s.append(x); keep_t.append(y)
        else:
            keep_s.append(y); keep_t.append(x)
        keep_sc.append(s)
    n = len(keep_s)
    if n == 0:                                   # some cards ship a csv with one empty column (e.g. swahili-zulu): nothing usable
        os.makedirs(out_dir, exist_ok=True)
        open(out + ".empty", "w").close()
        for q in (real, path):
            try:
                os.remove(q)
            except OSError:
                pass
        return repo, lo, hi, n_in, dropped, 0
    tbl = pa.table({"src_iso": pa.array([lo] * n).dictionary_encode(), "tgt_iso": pa.array([hi] * n).dictionary_encode(), "src": keep_s, "tgt": keep_t,
                    "score": pa.array(keep_sc, pa.float32()), "source_dataset": pa.array([repo] * n).dictionary_encode(), "licence": pa.array(["unknown"] * n).dictionary_encode(),
                    "licence_class": pa.array(["unspecified"] * n).dictionary_encode(), "licence_note": pa.array([NOTE] * n).dictionary_encode()})
    os.makedirs(out_dir, exist_ok=True)
    pq.write_table(tbl, out + ".tmp", compression="zstd")
    os.replace(out + ".tmp", out)
    try:
        os.remove(real)
        os.remove(path)
    except OSError:
        pass
    return repo, lo, hi, n_in, dropped, n


def main(argv=None) -> int:
    from multiprocessing import Pool

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repos", required=True, help="json with 'repos': [[repo, iso_a, iso_b, rows], ...]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--extra-bench", nargs="*", default=[])
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    repos = json.load(open(a.repos))["repos"]
    if a.limit:
        repos = repos[: a.limit]
    cache = a.cache or os.path.join(a.out, "_hf_cache")
    os.makedirs(a.out, exist_ok=True)
    # several repos can describe the same language pair (e.g. `akan-swahili` and `swahili-twi`): the biggest one owns `<a>-<b>.parquet`, the others get `<a>-<b>__<repo>.parquet`
    seen = set()
    jobs = []
    for r, x, y, _ in sorted(repos, key=lambda r: -r[3]):
        lo, hi = (x, y) if ALL_ISOS.index(x) < ALL_ISOS.index(y) else (y, x)
        suffix = "" if (lo, hi) not in seen else "__" + r.split("/")[1].replace("_sentence-pairs", "")
        seen.add((lo, hi))
        jobs.append((r, x, y, a.out, suffix))
    t0 = time.time()
    tot_in = tot_out = tot_drop = 0
    log = []
    with Pool(a.workers, initializer=_init, initargs=(a.bench, a.extra_bench, cache)) as pool:
        for k, (repo, lo, hi, n_in, dropped, n) in enumerate(pool.imap_unordered(_work, jobs)):
            if n_in >= 0:
                tot_in += n_in; tot_out += n; tot_drop += dropped
            log.append({"repo": repo, "src_iso": lo, "tgt_iso": hi, "rows_in": n_in, "dropped_heldout": dropped, "rows_out": n})
            print(f"{k + 1}/{len(jobs)} {repo} {n_in} -> {n} (held-out {dropped}) total {tot_out:,} {time.time() - t0:.0f}s", flush=True)
    prev = os.path.join(a.out, "_fetch_log.json")
    old = json.load(open(prev)) if os.path.exists(prev) else []
    json.dump(old + log, open(prev, "w"))
    print("done", tot_in, tot_out, tot_drop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
