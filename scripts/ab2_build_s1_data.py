#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AB2: data of the CONTINUED-SFT controls, cut from the 8.8 GB packed SFT set `gemma/packed_fwd` (538,480 train packs x 4,096 tokens) as a STREAM (a `tar` on stdin).

    curl -sSL "$PRESIGNED_URL" | zstd -dc | python scripts/ab2_build_s1_data.py --out data/packed_s1 --pair-out data/packed_s1_pair --from-step 3500 --steps 4100

Stage A of the SFT drew its packs with `balanced_schedule` (language_alpha 0.5, floor 0.002, cap 4 epochs, seed 1234), a pure function whose draws are sequential: rows [0, 3500) are what
the weak model W trained on (AB1 / AB2 re-create exactly those), rows [3500, 4100) are the packs the real run drew NEXT. Two outputs:
  S1  (--out)       a SPARSE copy of tokens.u32.bin that holds only the packs of schedule rows [from-step, steps) (index arrays, val/ and meta.json unchanged; the rest of the file
                    is a hole that is never read): the run `configs/ab2_sft_s1*.yaml` (schedule_start_step 3500, max_steps N <= steps - from-step) trains on them in the real order.
  S1P (--pair-out)  the PAIR-MATCHED control: `--pair-per-lang` (784 = 49 steps x 64 prompts / 4 languages) SHORT samples (prompt tokens <= --pair-max-prompt) per chosen language, drawn at
                    random from the packs W never drew (rows [0, from-step) excluded), re-packed into new single-language packs of the same format. The English source of each sample is
                    one window of the commercial-clean SFT data (not the RL prompt itself: the SFT windows and the RL prompts are different samples of the same English text pools).
Both are written from ONE pass over the stream. Disk: S1 about (steps - from-step) x 32 x 16 KB = 0.3 GB for 600 steps (sparse: never copy it with a tool that is not sparse-aware); S1P < 5 MB.
`ab2_s1_stats.json` (in --out) records the share of the S1 draws that W had already seen (balanced_schedule repeats packs of small languages) and the S1P selection.
"""
from __future__ import annotations

import argparse
import bisect
import io
import json
import os
import shutil
import sys
import tarfile
import time

import numpy as np

INDEX = ["direction", "iso_id", "pack_start", "prompt_len", "sample_len", "sample_offset", "target_len"]


def merge(rs):
    rs = sorted(rs)
    out = []
    for s, e in rs:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def select_pair_samples(idx, pack_lang, seen_w_mask, langs_ids, per_lang, max_prompt, seed):
    """Sample indices (into the index arrays) of SHORT samples of packs W never drew, per chosen language, seeded."""
    ps = np.asarray(idx["pack_start"], dtype=np.int64)
    pack_of_sample = np.repeat(np.arange(len(ps) - 1), np.diff(ps))
    iso = np.asarray(idx["iso_id"], dtype=np.int64)
    pl = np.asarray(idx["prompt_len"], dtype=np.int64)
    direction = np.asarray(idx["direction"], dtype=np.int64)
    ok = (~seen_w_mask[pack_of_sample]) & (pl <= max_prompt) & (direction == 0)
    chosen = {}
    for lid in langs_ids:
        cand = np.flatnonzero(ok & (iso == lid))
        rng = np.random.default_rng([seed, int(lid), 4242])
        if len(cand) < per_lang:
            raise SystemExit(f"language id {lid}: only {len(cand)} short unseen samples, need {per_lang}")
        chosen[int(lid)] = np.sort(rng.choice(cand, size=per_lang, replace=False))
    return chosen, len(np.flatnonzero(ok))


def repack(chosen, idx, tokens_of, seq_len, seed):
    """Greedy single-language packs of <= seq_len tokens from the chosen samples (random order inside a language). Returns the new index arrays and the token array."""
    sl = np.asarray(idx["sample_len"], dtype=np.int64)
    arr = {k: [] for k in INDEX}
    toks = []
    off = 0
    n_samples = 0
    pack_start = [0]
    for lid, sel in chosen.items():
        order = np.random.default_rng([seed, lid, 1]).permutation(sel)
        cur, cur_len = [], 0
        packs = []
        for s in order.tolist():
            L = int(sl[s])
            if L > seq_len:
                continue
            if cur_len + L > seq_len:
                packs.append(cur)
                cur, cur_len = [], 0
            cur.append(s)
            cur_len += L
        if cur:
            packs.append(cur)
        for p in packs:
            for s in p:
                for k in INDEX:
                    if k == "pack_start":
                        continue
                    if k == "sample_offset":
                        arr[k].append(off)
                    else:
                        arr[k].append(idx[k][s])
                toks.append(tokens_of[s])
                off += int(sl[s])
                n_samples += 1
            pack_start.append(n_samples)
    out = {}
    for k in INDEX:
        if k == "pack_start":
            out[k] = np.asarray(pack_start, dtype=np.asarray(idx[k]).dtype)
        else:
            out[k] = np.asarray(arr[k], dtype=np.asarray(idx[k]).dtype)
    return out, np.concatenate(toks).astype(np.uint32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="S1 dataset root (train/, val/, meta.json)")
    ap.add_argument("--pair-out", default=None, help="pair-matched dataset root (omit to skip)")
    ap.add_argument("--steps", type=int, required=True, help="END of the schedule: rows [from-step, steps) are written (32 packs per step)")
    ap.add_argument("--from-step", type=int, default=3500)
    ap.add_argument("--per-step", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--min-share", type=float, default=0.002)
    ap.add_argument("--max-epochs", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--chunk-mb", type=int, default=64)
    ap.add_argument("--pair-langs", nargs="*", default=["yor", "kin", "orm", "lin"])
    ap.add_argument("--pair-per-lang", type=int, default=784)
    ap.add_argument("--pair-max-prompt", type=int, default=90, help="max prompt tokens (template ~15 + English) of a pair-matched sample: the b64 RL windows")
    ap.add_argument("--pair-seed", type=int, default=77)
    ap.add_argument("--seq-len", type=int, default=4096)
    a = ap.parse_args(argv)

    from gemma_mt.data.packed_dataset import balanced_schedule
    from gemma_mt.languages import ALL_ISOS

    os.makedirs(os.path.join(a.out, "train"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "val"), exist_ok=True)
    if a.pair_out:
        os.makedirs(os.path.join(a.pair_out, "train"), exist_ok=True)
        os.makedirs(os.path.join(a.pair_out, "val"), exist_ok=True)
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
            seen_w = np.zeros(len(starts_s), dtype=bool)
            seen_w[sched[:a.from_step].ravel()] = True
            new_draws = sched[a.from_step:].ravel()
            unseen_share = float((~seen_w[new_draws]).mean())
            need = np.unique(new_draws)
            first = np.asarray(so[starts_s], dtype=np.int64)
            last_s = np.asarray(ps[1:], dtype=np.int64) - 1
            end = np.asarray(so[last_s], dtype=np.int64) + np.asarray(sl[last_s], dtype=np.int64)
            tok_need = int((end[need] - first[need]).sum())
            print(f"[ab2] S1 = schedule rows [{a.from_step}, {a.steps}) x {a.per_step}: {len(need)} distinct packs, {100 * unseen_share:.1f}% of the draws are packs W never drew, "
                  f"{tok_need} tokens = {tok_need * 4 / 1e9:.2f} GB to write", flush=True)
            r1 = merge([[int(first[p]) * 4, int(end[p]) * 4] for p in need.tolist()])
            r2, chosen, pair_info = [], {}, {}
            if a.pair_out:
                lid = [ALL_ISOS.index(i) for i in a.pair_langs]
                chosen, n_ok = select_pair_samples(idx, pack_lang, seen_w, lid, a.pair_per_lang, a.pair_max_prompt, a.pair_seed)
                sof = np.asarray(so, dtype=np.int64)
                slen = np.asarray(sl, dtype=np.int64)
                for l_, sel in chosen.items():
                    for s in sel.tolist():
                        r2.append([int(sof[s]) * 4, int(sof[s] + slen[s]) * 4, s])
                pair_info = {"languages": a.pair_langs, "per_language": a.pair_per_lang, "max_prompt_tokens": a.pair_max_prompt, "eligible_samples_all_chosen_languages": int(n_ok),
                             "mean_sample_len": {ALL_ISOS[l_]: float(slen[sel].mean()) for l_, sel in chosen.items()},
                             "mean_prompt_len": {ALL_ISOS[l_]: float(np.asarray(idx['prompt_len'])[sel].mean()) for l_, sel in chosen.items()}}
                print(f"[ab2] S1P = {a.pair_per_lang} short samples (prompt <= {a.pair_max_prompt} tokens) x {a.pair_langs} from packs W never drew ({n_ok} eligible)", flush=True)
            with open(os.path.join(a.out, "ab2_s1_stats.json"), "w") as f:
                json.dump({"from_step": a.from_step, "steps": a.steps, "per_step": a.per_step, "alpha": a.alpha, "min_share": a.min_share, "max_epochs": a.max_epochs, "seed": a.seed,
                           "unseen_by_W_share_of_draws": unseen_share, "distinct_packs": int(len(need)), "total_packs": int(len(starts_s)), "pair": pair_info,
                           "languages_in_s1_draws": {ALL_ISOS[int(l_)]: float((pack_lang[new_draws] == l_).mean()) for l_ in np.unique(pack_lang[new_draws])}}, f, indent=1)
            blocks = merge([[s, e] for s, e in r1] + [[s, e] for s, e, _ in r2])
            r1s = [r[0] for r in r1]
            r2.sort()
            r2s = [r[0] for r in r2]
            out_path = os.path.join(a.out, "train", "tokens.u32.bin")
            fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.ftruncate(fd, m.size)
            src = tar.extractfile(m)
            pos, written = 0, 0
            sample_tok = {}

            def read_exact(n):
                nonlocal pos
                buf = bytearray()
                while len(buf) < n:
                    b = src.read(min(a.chunk_mb << 20, n - len(buf)))
                    if not b:
                        raise SystemExit("unexpected end of the token stream")
                    buf += b
                pos += len(buf)
                return bytes(buf)

            for bs, be in blocks:
                if bs > pos:
                    # skip in chunks
                    n = bs - pos
                    while n > 0:
                        b = src.read(min(a.chunk_mb << 20, n))
                        if not b:
                            raise SystemExit("unexpected end of the token stream")
                        n -= len(b)
                        pos += len(b)
                data = read_exact(be - bs)
                i = max(0, bisect.bisect_right(r1s, bs) - 1)
                while i < len(r1) and r1[i][0] < be:
                    s, e = max(r1[i][0], bs), min(r1[i][1], be)
                    if s < e:
                        os.pwrite(fd, data[s - bs: e - bs], s)
                        written += e - s
                    i += 1
                j = max(0, bisect.bisect_left(r2s, bs))
                while j < len(r2) and r2[j][0] < be:
                    s, e, sid = r2[j]
                    if e <= be:
                        sample_tok[sid] = np.frombuffer(data[s - bs: e - bs], dtype=np.uint32)
                    j += 1
            n = m.size - pos
            while n > 0:
                b = src.read(min(a.chunk_mb << 20, n))
                if not b:
                    raise SystemExit("unexpected end of the token stream")
                n -= len(b)
                pos += len(b)
            os.close(fd)
            print(f"[ab2] tokens.u32.bin: wrote {written / 1e9:.3f} GB (S1), kept {len(sample_tok)} S1P samples, read {pos / 1e9:.2f} GB, {time.time() - t0:.0f} s", flush=True)
            if a.pair_out:
                missing = [s for _, _, s in r2 if s not in sample_tok]
                if missing:
                    raise SystemExit(f"{len(missing)} S1P samples were not read (straddle?)")
                new_idx, toks = repack(chosen, idx, sample_tok, a.seq_len, a.pair_seed)
                for k, v in new_idx.items():
                    np.save(os.path.join(a.pair_out, "train", k + ".npy"), v)
                toks.tofile(os.path.join(a.pair_out, "train", "tokens.u32.bin"))
                npk = len(new_idx["pack_start"]) - 1
                print(f"[ab2] S1P: {len(new_idx['sample_len'])} samples in {npk} single-language packs, {len(toks)} tokens", flush=True)
                with open(os.path.join(a.out, "ab2_s1_stats.json")) as f:
                    st = json.load(f)
                st["pair"].update({"packs": int(npk), "samples": int(len(new_idx["sample_len"])), "tokens": int(len(toks))})
                json.dump(st, open(os.path.join(a.out, "ab2_s1_stats.json"), "w"), indent=1)
            continue
        data = tar.extractfile(m).read()
        dests = [os.path.join(a.out, split, name) if split in ("train", "val") else os.path.join(a.out, name)]
        if a.pair_out and not (split == "train" and name.endswith(".npy")):
            dests.append(os.path.join(a.pair_out, split, name) if split in ("train", "val") else os.path.join(a.pair_out, name))
        for dest in dests:
            with open(dest, "wb") as f:
                f.write(data)
        if split == "train" and name.endswith(".npy"):
            idx[name[:-4]] = np.load(io.BytesIO(data))
        print(f"[ab2] wrote {dests[0]} ({len(data)} bytes)", flush=True)
    print("[ab2] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
