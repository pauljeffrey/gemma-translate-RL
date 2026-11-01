#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contamination / de-duplication scan of the candidate documents of the long-context translation benchmark (library: gemma_mt/eval/longdoc_contam.py).

    PYTHONPATH=<code dir>:<pylibs> nice -n 10 env CUDA_VISIBLE_DEVICES="" python scripts/longdoc_contam_scan.py \\
        --pool-dir  /workspace/sft_build/longdoc_bench/pool        \\
        --out-dir   /workspace/sft_build/longdoc_bench/contam      \\
        --sft-dir   /workspace/sft_build/gemma/packed_fwd          \\
        --rl-dir    /workspace/sft_build/gemma/rl_prompts_v3       \\
        --bench-root /workspace/sft_build/bench/test_sets          \\
        --tokenizer '/workspace/sft_build/hf/hub/models--google--gemma-3-270m-it/snapshots/*/tokenizer.json' --processes 16

For every `pool/<stem>.jsonl` it writes `<out-dir>/<stem>.flags.jsonl` (one line per document, the pool's `id` unchanged) and `<out-dir>/SUMMARY.json`.
Re-running is incremental: the expensive scan of a pool file (SFT stream, RL windows, benchmarks) is cached in `<out-dir>/cache/<stem>.scan.jsonl` together
with the pool file's size/mtime and the parameters, and is skipped while both still match (`--force` redoes it). New / changed pool files are scanned
together in ONE pass over the training data. The duplicate detection between pool documents is cheap and ALWAYS redone over all pool files, because a new
file can contain the longer copy of an old document (priority: more words first, then file name, then line number); flags are rewritten accordingly.

Flag (per document, all thresholds are options): contaminated if any share >= --weak-below, or a verbatim run >= --run-words words (RL / benchmark / decoded SFT)
or >= --run-tokens tokens (SFT, token space), or dup_share >= --dup-share; weak if the largest share is >= --clean-below; else clean.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import sys
import time
from typing import Any, Dict, List

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("RAYON_NUM_THREADS", "8")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gemma_mt.eval import longdoc_contam as L   # noqa: E402


def rss_gb() -> str:
    """Peak resident memory (GB) of this process and of its largest finished worker."""
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    ch = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    return f"peak RSS main {me:.1f} GB, largest worker {ch:.1f} GB"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rl_parquet_files(rl_dir):
    """The RL prompt files: `*.parquet` of the directory AND of its sub-directories (rl_prompts_v3 has the b2048 prompts in zz_b2048/); hard links to the same
    inode (v3 re-uses the v2 files) are scanned once."""
    if not rl_dir:
        return []
    seen, out = set(), []
    for p in sorted(glob.glob(os.path.join(rl_dir, "*.parquet")) + glob.glob(os.path.join(rl_dir, "*", "*.parquet"))):
        st = os.stat(p)
        if (st.st_dev, st.st_ino) in seen:
            continue
        seen.add((st.st_dev, st.st_ino))
        out.append(p)
    return out


def params_fingerprint(a: argparse.Namespace, bench_sizes: Dict[str, int]) -> Dict[str, Any]:
    return {"word_n": a.word_ngram, "token_n": a.token_ngram, "sft_scope": a.sft_scope, "sft_words": bool(a.sft_words and a.sft_dir),
            "paragraph_min_words": a.paragraph_min_words, "sft": bool(a.sft_dir), "rl": bool(a.rl_dir), "rl_files": len(rl_parquet_files(a.rl_dir)), "bench": bench_sizes, "version": 1}


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sft-dir", default=None, help="packed dataset root (train/ and val/ are scanned); omit to skip the SFT scan")
    ap.add_argument("--rl-dir", default=None, help="directory of RL prompt parquet files (column `english`); omit to skip")
    ap.add_argument("--bench-root", default=None, help="bench/test_sets (flores_plus, ntrex, mafand, smol_*, ...); omit to skip")
    ap.add_argument("--extra-bench", nargs="*", default=[], help="name=path.jsonl (column `english`), e.g. bench_dev=.../bench_dev.jsonl "
                                                                    "flores_dev=.../flores_dev_all_v2.jsonl")
    ap.add_argument("--tokenizer", default=None, help="tokenizer.json (glob allowed); needed for the token path and the decoded SFT word scan")
    ap.add_argument("--processes", type=int, default=16)
    ap.add_argument("--word-ngram", type=int, default=L.WORD_NGRAM)
    ap.add_argument("--token-ngram", type=int, default=L.TOKEN_NGRAM)
    ap.add_argument("--sft-scope", choices=["english", "all"], default="english", help="english = the English side of the en2x samples; all = every token")
    ap.add_argument("--no-sft-words", dest="sft_words", action="store_false", help="skip the decoded word-space scan of the SFT English")
    ap.add_argument("--paragraph-min-words", type=int, default=12)
    th = L.Thresholds()
    ap.add_argument("--clean-below", type=float, default=th.clean_below)
    ap.add_argument("--weak-below", type=float, default=th.weak_below)
    ap.add_argument("--run-words", type=int, default=th.run_words)
    ap.add_argument("--run-tokens", type=int, default=th.run_tokens)
    ap.add_argument("--dup-share", type=float, default=th.dup_share_max)
    ap.add_argument("--paragraph-exact-max", type=int, default=None)
    ap.add_argument("--bench-hit-min-ngrams", type=int, default=th.bench_hit_min_ngrams)
    ap.add_argument("--force", action="store_true", help="redo the cached scans")
    ap.add_argument("--max-sft-samples", type=int, default=None, help="debug: only the first N samples of each split")
    ap.add_argument("--only", nargs="*", default=None, help="only these pool file stems")
    a = ap.parse_args(argv)

    thr = L.Thresholds(a.clean_below, a.weak_below, a.run_words, a.run_tokens, a.dup_share, a.paragraph_exact_max, a.bench_hit_min_ngrams)
    os.makedirs(os.path.join(a.out_dir, "cache"), exist_ok=True)
    files = sorted(p for p in glob.glob(os.path.join(a.pool_dir, "*.jsonl")))
    if a.only:
        files = [p for p in files if os.path.basename(p)[:-6] in a.only]
    if not files:
        log(f"no pool files in {a.pool_dir}")
        return 0
    t_start = time.time()

    # ---- benchmarks
    bench: Dict[str, List[str]] = {}
    if a.bench_root:
        extra = dict(x.split("=", 1) for x in a.extra_bench)
        bench = L.load_benchmark_english(a.bench_root, extra)
        log("benchmark English: " + ", ".join(f"{k}={len(v)}" for k, v in bench.items()))
    bench_sizes = {k: len(v) for k, v in bench.items()}
    fp_params = params_fingerprint(a, bench_sizes)

    encode = None
    tk_path = None
    if a.tokenizer:
        tk_path = sorted(glob.glob(a.tokenizer))[0] if glob.glob(a.tokenizer) else a.tokenizer
        encode = L.make_encoder(L.load_tokenizers_json(tk_path))

    # ---- load all pool files (word path only), decide which need the heavy scan
    wcache = L.new_word_cache()
    records: Dict[str, List[Dict[str, Any]]] = {}
    docs: Dict[str, List[L.PreparedDoc]] = {}
    todo: List[str] = []
    fps: Dict[str, Dict[str, Any]] = {}
    for p in files:
        stem = os.path.basename(p)[:-6]
        fps[stem] = L.file_fingerprint(p)          # BEFORE reading: a pool file replaced during the scan must look stale on the next run
        records[stem] = list(L.iter_jsonl(p))
        docs[stem] = L.prepare_docs(records[stem], None, a.word_ngram, a.token_ngram, wcache)
        meta_p = os.path.join(a.out_dir, "cache", f"{stem}.scan.meta.json")
        ok = False
        if not a.force and os.path.exists(meta_p) and os.path.exists(os.path.join(a.out_dir, "cache", f"{stem}.scan.jsonl")):
            try:
                m = json.load(open(meta_p))
                ok = m.get("pool") == fps[stem] and m.get("params") == fp_params
            except ValueError:
                ok = False
        if not ok:
            todo.append(stem)
        log(f"pool {stem}: {len(records[stem])} docs, {sum(d.n_words for d in docs[stem])} words -> {'SCAN' if not ok else 'cached scan'}")

    # ---- the heavy scan, once for all new / changed files
    if todo:
        all_docs = [d for s in todo for d in docs[s]]
        if encode is not None and a.sft_dir:
            t0 = time.time()
            for s in todo:
                L.attach_tokens(docs[s], records[s], encode, a.token_ngram)
            log(f"tokenised {sum(len(docs[s]) for s in todo)} docs ({sum(d.n_tokens for d in all_docs)} tokens) in {time.time() - t0:.0f}s")
        src = L.Sources(
            sft_dirs=[os.path.join(a.sft_dir, "train"), os.path.join(a.sft_dir, "val")] if a.sft_dir else [],
            sft_tokenizer=tk_path,
            rl_files=rl_parquet_files(a.rl_dir),
            bench=bench)
        cfg = L.ScanConfig(a.word_ngram, a.token_ngram, a.processes, a.sft_scope, bool(a.sft_words), a.paragraph_min_words, a.max_sft_samples)
        rows, info = L.scan_documents(all_docs, src, cfg, log=log)
        k = 0
        for s in todo:
            part = rows[k: k + len(docs[s])]
            k += len(docs[s])
            L.atomic_write_jsonl(os.path.join(a.out_dir, "cache", f"{s}.scan.jsonl"), part)
            with open(os.path.join(a.out_dir, "cache", f"{s}.scan.meta.json"), "w") as f:
                json.dump({"pool": fps[s], "params": fp_params, "info": info,
                           "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1, default=str)
        log(f"scan done: {json.dumps(info, default=str)[:900]} | {rss_gb()}")
        for s in todo:
            for d in docs[s]:      # free the token arrays
                d.tok_ids = d.tok_hash = None

    # ---- duplicates over ALL pool files, flags, summary
    flat = []
    for s in docs:
        for j, d in enumerate(docs[s]):
            flat.append((s, j, d))
    ranking = sorted(range(len(flat)), key=lambda i: (-flat[i][2].n_words, flat[i][0], flat[i][1]))
    rank = [0] * len(flat)
    for r, i in enumerate(ranking):
        rank[i] = r
    t0 = time.time()
    dup_of, dup_share = L.find_duplicates([d.words_hash for _, _, d in flat], rank, a.dup_share)
    log(f"duplicate detection over {len(flat)} docs in {time.time() - t0:.0f}s: {sum(1 for x in dup_of if x >= 0)} docs are >= {a.dup_share:.0%} copies of a higher-priority doc")
    ids = [d.id for _, _, d in flat]
    summary: Dict[str, Any] = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "params": fp_params, "thresholds": vars(thr), "files": {}}
    pos = 0
    for s in docs:
        n = len(docs[s])
        scan_rows = list(L.iter_jsonl(os.path.join(a.out_dir, "cache", f"{s}.scan.jsonl")))
        assert [r["id"] for r in scan_rows] == [d.id for d in docs[s]], f"cache of {s} does not match the pool file (use --force)"
        rows = L.finalize_rows(scan_rows, dup_of[pos: pos + n], dup_share[pos: pos + n], ids, thr)     # dup_of holds GLOBAL indices into `ids`
        L.atomic_write_jsonl(os.path.join(a.out_dir, f"{s}.flags.jsonl"), rows)
        summary["files"][s] = L.summarise_rows(rows)
        try:
            m = json.load(open(os.path.join(a.out_dir, "cache", f"{s}.scan.meta.json")))
            summary["files"][s]["scan"] = {"scanned_at": m.get("created"), "pool_file": m.get("pool"), "sources": m.get("info")}
        except (OSError, ValueError):
            pass
        log(f"{s}: {summary['files'][s]['flags']}")
        pos += n
    val_p = os.path.join(a.out_dir, "validation.json")
    if os.path.exists(val_p):
        summary["validation"] = json.load(open(val_p))
    with open(os.path.join(a.out_dir, "SUMMARY.json.part"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    os.replace(os.path.join(a.out_dir, "SUMMARY.json.part"), os.path.join(a.out_dir, "SUMMARY.json"))
    summary["run"] = {"seconds": round(time.time() - t_start, 1), "memory": rss_gb(), "scanned_files": todo}
    with open(os.path.join(a.out_dir, "SUMMARY.json.part"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    os.replace(os.path.join(a.out_dir, "SUMMARY.json.part"), os.path.join(a.out_dir, "SUMMARY.json"))
    log(f"done in {time.time() - t_start:.0f}s | {rss_gb()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
