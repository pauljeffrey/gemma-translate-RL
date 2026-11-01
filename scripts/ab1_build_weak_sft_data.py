#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AB1 (RL-vs-SFT-data ablation): build the data of the WEAK SFT run without unpacking the 8.8 GB packed SFT set.

The packed set `gemma/packed_fwd` (538,480 train packs x 4,096 tokens, one `tokens.u32.bin` of 8,816,750,880 bytes + 7 small index arrays + val/ + meta.json)
is read as a STREAM (a `tar` on stdin). Stage A of the SFT (configs/sft_long.yaml: AdamW, language_alpha 0.5, floor 0.002, cap 4 epochs, seed 1234) drew its packs with
`balanced_schedule`, a pure function whose draws are sequential: the first N steps of the 16,827-step schedule are the same as the whole N-step schedule. The weak run
(configs/ab1_sft_weak.yaml, max_steps N) therefore trains on exactly the packs of the FIRST N steps of the real stage A. This script computes that schedule from the index
arrays, and writes into a SPARSE copy of tokens.u32.bin only the byte ranges of the packs that schedule draws (the rest of the file is a hole that is never read);
index arrays, val/ and meta.json are copied unchanged. The trainer therefore sees the full-size dataset and reproduces the schedule bit for bit.

    curl -sSL "$PRESIGNED_URL" | tee >(sha256sum > sha.txt) | zstd -dc | python scripts/ab1_build_weak_sft_data.py --out /workspace/jobs/ab1/data/packed_fwd --steps 2764

Disk: about N x 32 x 16 KB = 1.4 GB for N = 2764 (sparse; `du` shows the real size, `ls` the apparent 8.8 GB). Never copy the output with a tool that is not sparse-aware.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time

import numpy as np

INDEX = ["direction", "iso_id", "pack_start", "prompt_len", "sample_len", "sample_offset", "target_len"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dataset root (train/, val/, meta.json)")
    ap.add_argument("--steps", type=int, required=True, help="optimizer steps of the weak run (32 packs per step)")
    ap.add_argument("--per-step", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--min-share", type=float, default=0.002)
    ap.add_argument("--max-epochs", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chunk-mb", type=int, default=64)
    a = ap.parse_args(argv)

    from gemma_mt.data.packed_dataset import balanced_schedule
    from gemma_mt.languages import ALL_ISOS

    os.makedirs(os.path.join(a.out, "train"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "val"), exist_ok=True)
    idx = {}
    tar = tarfile.open(fileobj=sys.stdin.buffer, mode="r|")
    t0 = time.time()
    for m in tar:
        if not m.isfile():
            continue
        parts = m.name.split("/")
        name, split = parts[-1], (parts[-2] if len(parts) > 1 else "")
        if name == "tokens.u32.bin" and split == "train":
            missing = [k for k in INDEX if k not in idx]
            if missing:
                raise SystemExit(f"tokens.u32.bin reached before the index arrays {missing}")
            iso, ps, so, sl = idx["iso_id"], idx["pack_start"], idx["sample_offset"], idx["sample_len"]
            starts_s = np.asarray(ps[:-1], dtype=np.int64)
            lo = np.minimum.reduceat(np.asarray(iso, dtype=np.int64), starts_s)
            hi = np.maximum.reduceat(np.asarray(iso, dtype=np.int64), starts_s)
            if bool((lo != hi).any()):
                raise SystemExit("some packs mix languages")
            pack_lang = lo
            sched, stats = balanced_schedule(pack_lang, a.steps, a.per_step, alpha=a.alpha, min_share=a.min_share, max_epochs=a.max_epochs, seed=a.seed)
            need = np.unique(sched.ravel())
            first = np.asarray(so[starts_s], dtype=np.int64)                           # token index of the first token of every pack
            last_s = np.asarray(ps[1:], dtype=np.int64) - 1
            end = np.asarray(so[last_s], dtype=np.int64) + np.asarray(sl[last_s], dtype=np.int64)
            tok_need = int((end[need] - first[need]).sum())
            print(f"[ab1] schedule: {a.steps} steps x {a.per_step} packs = {sched.size} draws, {len(need)} distinct packs of {len(starts_s)} "
                  f"({100.0 * len(need) / len(starts_s):.1f}%), {tok_need} real tokens = {tok_need * 4 / 1e9:.2f} GB to write", flush=True)
            by_lang = sorted(stats.items(), key=lambda kv: -kv[1]["share"])
            with open(os.path.join(a.out, "weak_schedule_stats.json"), "w") as f:
                json.dump({"steps": a.steps, "per_step": a.per_step, "alpha": a.alpha, "min_share": a.min_share, "max_epochs": a.max_epochs, "seed": a.seed,
                           "distinct_packs": int(len(need)), "total_packs": int(len(starts_s)),
                           "languages": {ALL_ISOS[i]: v for i, v in by_lang}}, f, indent=1)
            # merge the byte ranges of the needed packs (uint32 tokens -> *4 bytes), in file order
            order = need  # np.unique is sorted by pack index = file order
            b0, b1 = first[order] * 4, end[order] * 4
            ranges = []
            for s, e in zip(b0.tolist(), b1.tolist()):
                if ranges and ranges[-1][1] == s:
                    ranges[-1][1] = e
                else:
                    ranges.append([s, e])
            out_path = os.path.join(a.out, "train", "tokens.u32.bin")
            fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.ftruncate(fd, m.size)                                                  # sparse: the apparent size must equal the original
            src = tar.extractfile(m)
            pos, written, chunk = 0, 0, a.chunk_mb << 20

            def skip(n):
                nonlocal pos
                while n > 0:
                    b = src.read(min(chunk, n))
                    if not b:
                        raise SystemExit("unexpected end of the token stream")
                    n -= len(b)
                    pos += len(b)

            for s, e in ranges:
                skip(s - pos)
                n = e - s
                off = s
                while n > 0:
                    b = src.read(min(chunk, n))
                    if not b:
                        raise SystemExit("unexpected end of the token stream")
                    os.pwrite(fd, b, off)
                    off += len(b)
                    n -= len(b)
                    pos += len(b)
                    written += len(b)
            skip(m.size - pos)
            os.close(fd)
            print(f"[ab1] tokens.u32.bin: wrote {written / 1e9:.2f} GB in {len(ranges)} ranges, read {pos / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)
            continue
        data = tar.extractfile(m).read()
        dest = os.path.join(a.out, split, name) if split in ("train", "val") else os.path.join(a.out, name)
        with open(dest, "wb") as f:
            f.write(data)
        if split == "train" and name.endswith(".npy"):
            idx[name[:-4]] = np.load(io.BytesIO(data))
        print(f"[ab1] wrote {dest} ({len(data)} bytes)", flush=True)
    print("[ab1] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
