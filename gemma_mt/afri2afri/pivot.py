# SPDX-License-Identifier: AGPL-3.0-or-later
"""Join of the en->X pair table on the English sentence: every two languages that translated the same English sentence give X<->Y pairs.

All functions work on flat numpy arrays (one entry per en->X pair), never on strings, so 26 M pairs fit in a few GB:
    key    uint64   hash of `text.pivot_key(english)`
    iso    uint8    language id (position in `languages.ALL_ISOS`)
    score  float    ranking inside one (key, iso): higher wins when the same English sentence has several translations in one language
    tier   uint8    provenance tier (`provenance.KINDS`)

`best_per_key_iso` keeps one row per (key, iso); `group_bounds` finds the groups of equal key; `pair_count_tensor` counts the ordered pairs
(tier, src, tgt) without ever materialising them; `iter_group_pairs` materialises (row_a, row_b) for the groups a caller selects.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def best_per_key_iso(key: np.ndarray, iso: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Indices (into the input arrays) of the best-scoring row of every (key, iso), ordered by (key, iso)."""
    if len(key) == 0:
        return np.zeros(0, np.int64)
    order = np.lexsort((-np.asarray(score, np.float64), iso, key))      # last key is the primary one
    k, i = key[order], iso[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = (k[1:] != k[:-1]) | (i[1:] != i[:-1])
    return order[first]


def group_bounds(sorted_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(starts, sizes) of the runs of equal values in an array that is already sorted by key."""
    n = len(sorted_key)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    change = np.flatnonzero(sorted_key[1:] != sorted_key[:-1]) + 1
    starts = np.concatenate([[0], change]).astype(np.int64)
    sizes = np.diff(np.concatenate([starts, [n]])).astype(np.int64)
    return starts, sizes


def pair_count_tensor(iso: np.ndarray, tier: np.ndarray, starts: np.ndarray, sizes: np.ndarray, n_lang: int, n_tier: int = 4) -> np.ndarray:
    """counts[t, a, b] = number of ORDERED pairs (a -> b) whose pair tier (the min of the two sides) is t. Groups of size 1 give nothing.
    Inside a group every language appears once (after `best_per_key_iso`), so a group of k languages gives k (k - 1) ordered pairs."""
    out = np.zeros(n_tier * n_lang * n_lang, dtype=np.int64)
    L = n_lang
    for k in np.unique(sizes):
        k = int(k)
        if k < 2:
            continue
        sel = starts[sizes == k]
        idx = sel[:, None] + np.arange(k)[None, :]
        G = iso[idx].astype(np.int64)
        T = tier[idx].astype(np.int64)
        for p in range(k):
            for q in range(p + 1, k):
                t = np.minimum(T[:, p], T[:, q])
                a, b = G[:, p], G[:, q]
                out += np.bincount((t * L + a) * L + b, minlength=out.size)
                out += np.bincount((t * L + b) * L + a, minlength=out.size)
    return out.reshape(n_tier, L, L)


def iter_group_pairs(starts: np.ndarray, sizes: np.ndarray, group_ids: np.ndarray, chunk: int = 200_000) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For the selected groups yield (group_id, row_a, row_b) with a < b positions (unordered pairs), as index arrays into the sorted arrays.
    The caller decides the direction(s) later."""
    gids = np.asarray(group_ids, np.int64)
    for s in range(0, len(gids), chunk):
        part = gids[s:s + chunk]
        sz = sizes[part]
        for k in np.unique(sz):
            k = int(k)
            if k < 2:
                continue
            sel = part[sz == k]
            idx = starts[sel][:, None] + np.arange(k)[None, :]
            for p in range(k):
                for q in range(p + 1, k):
                    yield sel, idx[:, p], idx[:, q]


def enumerate_pairs(starts: np.ndarray, sizes: np.ndarray, min_size: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All UNORDERED pairs of rows inside every group: (group index, row a, row b) as int32 arrays (rows index the arrays the groups were computed on).
    A group of k rows gives k (k - 1) / 2 pairs, in no particular order."""
    gs, ras, rbs = [], [], []
    gid_all = np.arange(len(starts), dtype=np.int64)
    for k in np.unique(sizes):
        k = int(k)
        if k < min_size:
            continue
        sel = gid_all[sizes == k]
        idx = starts[sel][:, None] + np.arange(k)[None, :]
        for p in range(k):
            for q in range(p + 1, k):
                gs.append(sel.astype(np.int32))
                ras.append(idx[:, p].astype(np.int32))
                rbs.append(idx[:, q].astype(np.int32))
    if not gs:
        z = np.zeros(0, np.int32)
        return z, z, z
    return np.concatenate(gs), np.concatenate(ras), np.concatenate(rbs)


def pick_by_quota(segment: np.ndarray, rand: np.ndarray, quota: np.ndarray) -> np.ndarray:
    """Boolean mask: inside every segment keep the `quota[segment]` records with the smallest `rand`. (Uniform sampling without replacement per segment.)"""
    n = len(segment)
    if n == 0:
        return np.zeros(0, bool)
    order = np.lexsort((rand, segment))
    seg_sorted = segment[order]
    first = np.ones(n, bool)
    first[1:] = seg_sorted[1:] != seg_sorted[:-1]
    start_pos = np.maximum.accumulate(np.where(first, np.arange(n), 0))
    rank = np.arange(n) - start_pos
    keep_sorted = rank < quota[seg_sorted]
    mask = np.zeros(n, bool)
    mask[order[keep_sorted]] = True
    return mask
