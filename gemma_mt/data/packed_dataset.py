# SPDX-License-Identifier: AGPL-3.0-or-later
"""Memory-mapped loader for the packed dataset written by `prepare_sft`, plus the deterministic batch schedule.

`PackedDataset[i]` is ONE packed sequence (several samples back to back):

    input_ids     int64  [L]   token ids (padding = pad_id)
    labels        int64  [L]   same alignment as input_ids; -100 on prompt tokens, on the template's trailing "\\n"
                               and on padding; the HF convention applies (the loss shifts: logits[t] predicts
                               labels[t + 1]), see gemma_mt.losses
    position_ids  int64  [L]   0..len-1 INSIDE every sample (resets at each sample start); padding is its own
                               segment counting 0..k-1, so no padding token ever continues a sample's positions
    segment_ids   int32  [L]   1..k for the k samples of the pack, 0 for padding
    cu_seqlens    int32  [k+1 or k+2]  sample boundaries (a final entry closes the padding segment when padded)
    token_iso     int16  [L]   language id (gemma_mt.languages.ALL_ISOS index) of the sample a token belongs to, -1 on
                               padding; used only for per-language validation loss
    length        int    real (non-padding) tokens
    n_samples     int

With `pad=True` (default) every row has exactly `seq_len` tokens; with `pad=False` rows have their real length
(used by the padding-free flash-attention path).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import math
import numpy as np

from gemma_mt.data.prepare_sft import DTYPE, FORMAT_VERSION

IGNORE = -100


class PackedDataset:
    def __init__(self, directory: str, split: Optional[str] = None, *, pad: bool = True, pad_id: Optional[int] = None,
                 expect_tokenizer_fingerprint: Optional[str] = None):
        """`directory` is either the dataset root (then pass `split`) or a split directory itself."""
        root = directory
        split_dir = os.path.join(directory, split) if split else directory
        meta_path = os.path.join(root if split else os.path.dirname(os.path.abspath(directory)), "meta.json")
        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.meta.get("format_version") != FORMAT_VERSION or self.meta.get("dtype") != DTYPE:
            raise ValueError(f"unsupported packed format in {meta_path}")
        fp = self.meta["tokenizer"]["fingerprint"]
        if expect_tokenizer_fingerprint is not None and fp != expect_tokenizer_fingerprint:
            raise ValueError("the packed dataset was built with a different tokenizer "
                             f"(dataset {fp[:12]}..., current {expect_tokenizer_fingerprint[:12]}...)")
        self.seq_len = int(self.meta["seq_len"])
        self.pad = pad
        self.pad_id = int(self.meta["tokenizer"]["pad_id"] if pad_id is None else pad_id)
        ld = lambda n: np.load(os.path.join(split_dir, n), mmap_mode="r")
        self.sample_offset, self.sample_len = ld("sample_offset.npy"), ld("sample_len.npy")
        self.prompt_len, self.target_len = ld("prompt_len.npy"), ld("target_len.npy")
        self.iso_id, self.pack_start = ld("iso_id.npy"), ld("pack_start.npy")
        dir_path = os.path.join(split_dir, "direction.npy")                     # absent in older datasets: all en2x
        self.direction = ld("direction.npy") if os.path.exists(dir_path) else np.zeros(len(self.iso_id), np.uint8)
        self.n_langs = len(self.meta.get("iso_table") or ())
        tok_path = os.path.join(split_dir, "tokens.u32.bin")
        self.n_tokens_total = os.path.getsize(tok_path) // 4
        self.tokens = np.memmap(tok_path, dtype=np.uint32, mode="r") if self.n_tokens_total else np.zeros(0, np.uint32)
        self.n_packs = len(self.pack_start) - 1
        self.n_samples = len(self.sample_len)

    def __len__(self) -> int:
        return self.n_packs

    def sample_range(self, i: int) -> Tuple[int, int]:
        return int(self.pack_start[i]), int(self.pack_start[i + 1])

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        if not 0 <= i < self.n_packs:
            raise IndexError(i)
        s0, s1 = self.sample_range(i)
        lens = np.asarray(self.sample_len[s0:s1], dtype=np.int64)
        total = int(lens.sum())
        start = int(self.sample_offset[s0])
        ids = np.asarray(self.tokens[start: start + total], dtype=np.int64)
        labels = np.full(total, IGNORE, dtype=np.int64)
        pos = np.empty(total, dtype=np.int64)
        seg = np.empty(total, dtype=np.int32)
        tiso = np.empty(total, dtype=np.int16)
        c = 0
        for j, L in enumerate(lens):
            L = int(L)
            p, t = int(self.prompt_len[s0 + j]), int(self.target_len[s0 + j])
            labels[c + p: c + p + t] = ids[c + p: c + p + t]
            pos[c: c + L] = np.arange(L)
            seg[c: c + L] = j + 1
            # language id; the reverse direction is offset by n_langs so validation can report both directions
            tiso[c: c + L] = int(self.iso_id[s0 + j]) + self.n_langs * int(self.direction[s0 + j])
            c += L
        cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
        if self.pad and total < self.seq_len:
            n_pad = self.seq_len - total
            ids = np.concatenate([ids, np.full(n_pad, self.pad_id, np.int64)])
            labels = np.concatenate([labels, np.full(n_pad, IGNORE, np.int64)])
            pos = np.concatenate([pos, np.arange(n_pad, dtype=np.int64)])
            seg = np.concatenate([seg, np.zeros(n_pad, np.int32)])
            tiso = np.concatenate([tiso, np.full(n_pad, -1, np.int16)])
            cu = np.concatenate([cu, np.asarray([self.seq_len], np.int32)])
        return {"input_ids": ids, "labels": labels, "position_ids": pos, "segment_ids": seg,
                "cu_seqlens": cu, "token_iso": tiso, "length": total, "n_samples": s1 - s0}

    def iso_ids_of_pack(self, i: int) -> np.ndarray:
        s0, s1 = self.sample_range(i)
        return np.asarray(self.iso_id[s0:s1])

    def total_target_tokens(self) -> int:
        return int(np.asarray(self.target_len, dtype=np.int64).sum())

    def pack_languages(self) -> np.ndarray:
        """Language id of every pack, for datasets built with `--pack-by-language` (one language per pack).
        Raises ValueError when some pack mixes languages (the stratified sampler cannot be used then)."""
        iso = np.asarray(self.iso_id, dtype=np.int64)
        starts = np.asarray(self.pack_start[:-1], dtype=np.int64)
        if len(starts) == 0:
            return np.zeros(0, np.int64)
        lo = np.minimum.reduceat(iso, starts)
        hi = np.maximum.reduceat(iso, starts)
        if bool((lo != hi).any()):
            raise ValueError(f"{int((lo != hi).sum())} packs mix languages: rebuild the data with prepare_sft --pack-by-language")
        return lo


# ----------------------------------------------------------------------------------------------- schedule
def steps_per_epoch(n_packs: int, micro_batch: int, accum: int, world: int = 1) -> int:
    """Optimizer steps in one epoch; the incomplete tail is dropped (a different tail every epoch, because the
    permutation changes)."""
    return n_packs // (micro_batch * accum * world)


def epoch_permutation(n_packs: int, epoch: int, seed: int) -> np.ndarray:
    return np.random.default_rng([seed, epoch]).permutation(n_packs)


def batch_indices(n_packs: int, step: int, micro_batch: int, accum: int, rank: int = 0, world: int = 1,
                  seed: int = 0) -> List[List[int]]:
    """Pack indices of optimizer step `step` (0-based, GLOBAL) for `rank`: `accum` micro-batches of `micro_batch`.

    A pure function of (step, rank, seed): resuming from a checkpoint needs only the step counter, and different
    ranks always get disjoint packs."""
    spe = steps_per_epoch(n_packs, micro_batch, accum, world)
    if spe <= 0:
        raise ValueError(f"{n_packs} packs are not enough for micro_batch={micro_batch} x accum={accum} x world={world}")
    epoch, j = divmod(step, spe)
    used = spe * accum * world * micro_batch
    arr = epoch_permutation(n_packs, epoch, seed)[:used].reshape(spe, accum, world, micro_batch)
    return [list(map(int, row)) for row in arr[j, :, rank, :]]


# ----------------------------------------------------------------------------------------------- language-stratified schedule
def balanced_schedule(pack_lang: np.ndarray, n_steps: int, per_step: int, *, alpha: float = 0.5, min_share: float = 0.002,
                      max_epochs: float = 4.0, seed: int = 0) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """Pack indices for every optimizer step: [n_steps, per_step] (the order inside a step is the order of the draws).

    Languages are drawn with probability  p_l  proportional to  packs_l ** alpha  (1.0 = natural frequency, 0.0 = uniform),
    raised to at least `min_share` so that EVERY language shows up in every evaluation interval, and renormalised. Inside a
    language the packs go round in a seeded permutation (no pack repeats before all of that language's packs were used; a
    fresh permutation after each pass). A language that has been used `max_epochs` times is retired from the mixture (its
    probability is redistributed) so a tiny language is not memorised; when every language is retired the caps restart.

    Pure function of its arguments: resuming a run only needs the step counter."""
    pack_lang = np.asarray(pack_lang, dtype=np.int64)
    langs, counts = np.unique(pack_lang, return_counts=True)
    n_l = len(langs)
    if n_l == 0:
        raise ValueError("no packs")
    rng = np.random.default_rng([seed, 7919])
    members = [np.flatnonzero(pack_lang == l) for l in langs]
    perm = [rng.permutation(m) for m in members]
    ptr = np.zeros(n_l, dtype=np.int64)
    draws = np.zeros(n_l, dtype=np.int64)
    cap = np.maximum(1, np.ceil(max_epochs * counts)).astype(np.int64) if max_epochs else None
    base = counts.astype(np.float64) ** float(alpha)
    out = np.empty((n_steps, per_step), dtype=np.int64)
    for t in range(n_steps):
        active = np.ones(n_l, bool) if cap is None else draws < cap
        if not active.any():
            draws[:] = 0
            active[:] = True
        w = np.where(active, base, 0.0)
        p = w / w.sum()
        if min_share > 0:
            floor = min(min_share, 1.0 / int(active.sum()))
            p = np.where(active, np.maximum(p, floor), 0.0)
            p = p / p.sum()
        pick = rng.choice(n_l, size=per_step, p=p)
        for j, l in enumerate(pick):
            if ptr[l] >= len(perm[l]):
                perm[l] = rng.permutation(members[l])
                ptr[l] = 0
            out[t, j] = perm[l][ptr[l]]
            ptr[l] += 1
            draws[l] += 1
    total = float(out.size)
    # `draws` is reset when every language retires, so recompute the statistics from the schedule itself
    flat_lang = pack_lang[out.reshape(-1)]
    stats = {}
    for i, l in enumerate(langs):
        d = int((flat_lang == l).sum())
        stats[int(l)] = {"packs": int(counts[i]), "draws": d, "share": d / total, "epochs": d / counts[i]}
    return out, stats


# ----------------------------------------------------------------------------------------------- UniMax schedule
def unimax_allocation(counts: np.ndarray, budget: int, *, min_epochs: float = 1.0, max_epochs: float = 2.0,
                      weights: Optional[np.ndarray] = None) -> np.ndarray:
    """Draws per language under a fixed budget (UniMax, Chung et al. 2023, with a per-language floor and priority weights).

        a_l = clip(lambda * w_l, ceil(min_epochs * n_l), floor(max_epochs * n_l)),   sum_l a_l = budget

    `lambda` is the uniform per-language share (water level): small languages sit at their cap (max_epochs passes), large
    languages at their floor (min_epochs passes), the rest at lambda * w_l. With w = 1 and min_epochs = 0 this is plain
    UniMax; `weights` > 1 give a language a higher water level (priority). Integer result, sum == budget exactly."""
    n = np.asarray(counts, dtype=np.float64)
    w = np.ones_like(n) if weights is None else np.asarray(weights, dtype=np.float64)
    lo = np.ceil(min_epochs * n)
    hi = np.maximum(lo, np.floor(max_epochs * n))
    if budget < lo.sum() or budget > hi.sum():
        raise ValueError(f"budget {budget} packs is outside [{int(lo.sum())}, {int(hi.sum())}] "
                         f"(min_epochs={min_epochs}, max_epochs={max_epochs})")
    lam_lo, lam_hi = 0.0, float(hi.max() / w.min()) + 1.0
    for _ in range(200):
        mid = 0.5 * (lam_lo + lam_hi)
        if np.clip(mid * w, lo, hi).sum() > budget:
            lam_hi = mid
        else:
            lam_lo = mid
    a = np.clip(lam_lo * w, lo, hi)
    out = np.floor(a).astype(np.int64)
    short = int(budget - out.sum())                                   # hand the rounding remainder to the languages with room, largest fraction first
    if short > 0:
        room = np.flatnonzero(out < hi)
        order = room[np.argsort(-(a[room] - out[room]), kind="stable")]
        k = 0
        while short > 0 and len(order):
            i = order[k % len(order)]
            if out[i] < hi[i]:
                out[i] += 1
                short -= 1
            k += 1
    return out


def unimax_schedule(pack_lang: np.ndarray, n_steps: int, per_step: int, *, min_epochs: float = 1.0, max_epochs: float = 2.0,
                    weights: Optional[Dict[int, float]] = None, seed: int = 0,
                    seen: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
    """Pack indices for every optimizer step [n_steps, per_step] under a UniMax token budget (n_steps * per_step packs).

    Every language gets exactly `unimax_allocation` draws: whole passes over its packs, each pass in a fresh seeded permutation,
    the last pass partial. The draws of a language are spread evenly over the whole run (key = (k + u) / a_l), so every
    evaluation interval sees every language in proportion to its allocation. With min_epochs >= 1 every pack of every
    language is used at least once. `seen` (bool per pack: used by an earlier phase) puts a language's UNSEEN packs first in
    its first pass, so the new phase meets new data before it repeats old data. Pure function of its arguments (exact resume needs only the step counter)."""
    pack_lang = np.asarray(pack_lang, dtype=np.int64)
    langs, counts = np.unique(pack_lang, return_counts=True)
    w = None if not weights else np.array([float(weights.get(int(l), 1.0)) for l in langs])
    alloc = unimax_allocation(counts, n_steps * per_step, min_epochs=min_epochs, max_epochs=max_epochs, weights=w)
    rng = np.random.default_rng([seed, 104729])
    idx, keys = [], []
    for l, a in zip(langs, alloc):
        members = np.flatnonzero(pack_lang == l)
        reps = int(math.ceil(a / len(members)))
        if seen is None:
            first = rng.permutation(members)
        else:
            sn = np.asarray(seen, dtype=bool)[members]
            first = np.concatenate([rng.permutation(members[~sn]), rng.permutation(members[sn])])
        seq = np.concatenate([first] + [rng.permutation(members) for _ in range(reps - 1)])[:a]
        idx.append(seq)
        keys.append((np.arange(a) + rng.random()) / a)
    idx, keys = np.concatenate(idx), np.concatenate(keys)
    order = np.argsort(keys, kind="stable")
    out = idx[order].reshape(n_steps, per_step)
    stats = {int(l): {"packs": int(c), "draws": int(a), "share": float(a) / float(alloc.sum()), "epochs": float(a) / float(c)}
             for l, c, a in zip(langs, counts, alloc)}
    return out, stats
