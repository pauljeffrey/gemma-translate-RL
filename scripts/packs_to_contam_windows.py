#!/usr/bin/env python3
"""Packed SFT shards -> the parquet windows (english, target, iso) that `gemma_mt.eval.contamination` scans.

    python scripts/packs_to_contam_windows.py --shard /workspace/clean_sft/run/shards/s1 [--shard ... s6] [--replay DIR] --out-dir /workspace/eval_385m/windows \
        --tokenizer gemma_mt/artifacts/tokenizer_gemma3_tags/tokenizer.json --processes 16 [--limit-samples 2000]
    python -m gemma_mt.eval.contamination --windows /workspace/eval_385m/windows --bench-root BENCH/test_sets --benchmarks flores_plus ntrex mafand smol_sent ibom_mt flores_moz glot_story salt \
        --out-dir /workspace/eval_385m/contamination --processes 16

A shard is the packed layout of the growth-1 stage-B data (train/{tokens.u32.bin, sample_offset, sample_len, prompt_len, target_len, iso_id, direction}.npy + meta.json with `iso_table`).
EVERY sample is written, translation or instruction: the benchmark English can sit in either. `english` = the decoded English side (the prompt of an en2x sample, the target of an x2en
sample; for the prompt that includes the instruction wrapper, which adds no benchmark n-gram), `target` = the other side, `iso` = the sample's language. The scanner matches a benchmark
sentence as an exact sentence or as a contiguous token run inside a window, and by 8-gram overlap, so the wrapper words do not hide a hit. Special tokens (language tags, turn markers)
are dropped. The tokenizer only decodes: the ids of the original and the renamed Gemma tokenizer are the same.

NOT run on a real shard from the laptop (no numpy/pyarrow/tokenizers stack there): the decoding mirrors data_pipeline/sft_clean/fresh/lid_measure.py; smoke it with --limit-samples first.
"""
import argparse, json, os
from multiprocessing import Pool

import numpy as np

_G = {}


def _init(shard, tok_path):
    from tokenizers import Tokenizer
    _G["tk"] = Tokenizer.from_file(tok_path)
    d = os.path.join(shard, "train")
    _G["A"] = {n: np.load(f"{d}/{n}.npy", mmap_mode="r") for n in ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction")}
    _G["tok"] = np.memmap(f"{d}/tokens.u32.bin", dtype=np.uint32, mode="r")
    _G["iso_table"] = json.load(open(os.path.join(shard, "meta.json")))["iso_table"]


def _chunk(job):
    lo, hi, out = job
    A, tok, tk, table = _G["A"], _G["tok"], _G["tk"], _G["iso_table"]
    so = np.asarray(A["sample_offset"][lo:hi]).astype(np.int64); pl = np.asarray(A["prompt_len"][lo:hi]).astype(np.int64)
    tl = np.asarray(A["target_len"][lo:hi]).astype(np.int64); iso = np.asarray(A["iso_id"][lo:hi]); dr = np.asarray(A["direction"][lo:hi])
    prompts = [tok[s: s + p].tolist() for s, p in zip(so, pl)]
    targets = [tok[s + p: s + p + t].tolist() for s, p, t in zip(so, pl, tl)]
    P = tk.decode_batch(prompts, skip_special_tokens=True); T = tk.decode_batch(targets, skip_special_tokens=True)
    en = [t if d == 1 else p for p, t, d in zip(P, T, dr)]; ot = [p if d == 1 else t for p, t, d in zip(P, T, dr)]
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.table({"english": en, "target": ot, "iso": [table[i] if 0 <= i < len(table) else "" for i in iso]}), out, compression="zstd")
    return hi - lo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", action="append", required=True, help="packed shard directory (repeat for several; --replay dirs are also accepted here)")
    ap.add_argument("--replay", action="append", default=[], help="replay pack directory (same layout)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json (original or renamed Gemma tokenizer)")
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--rows-per-file", type=int, default=200_000)
    ap.add_argument("--limit-samples", type=int, default=0, help="first N samples of every shard only (smoke test)")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True); total = 0
    for k, shard in enumerate(a.shard + a.replay):
        n = len(np.load(os.path.join(shard, "train", "sample_offset.npy"), mmap_mode="r"))
        if a.limit_samples: n = min(n, a.limit_samples)
        jobs = [(lo, min(lo + a.rows_per_file, n), os.path.join(a.out_dir, f"w{k:02d}_{lo // a.rows_per_file:05d}.parquet")) for lo in range(0, n, a.rows_per_file)]
        with Pool(a.processes, initializer=_init, initargs=(shard, a.tokenizer)) as pool:
            done = sum(pool.imap_unordered(_chunk, jobs))
        total += done; print(f"{shard}: {done} samples -> {len(jobs)} files", flush=True)
    print(f"windows written: {total} samples in {a.out_dir}")


if __name__ == "__main__":
    main()
