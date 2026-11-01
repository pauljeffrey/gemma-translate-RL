# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bin packing of tokenized samples into fixed-capacity sequences. Pure python, no numpy needed.

Best-fit-decreasing: samples are visited longest first; each goes into the bin whose REMAINING capacity is the
smallest one that still fits it (a new bin when none fits). A sample is never split and never truncated here.
Complexity is O(n log n) thanks to a sorted list of the distinct remaining capacities and a bucket per capacity.
"""

from __future__ import annotations

import random
from bisect import bisect_left, insort
from typing import Dict, List, Sequence


def best_fit_decreasing(lengths: Sequence[int], capacity: int) -> List[List[int]]:
    """Return bins as lists of indices into `lengths`. Every index appears exactly once; no bin exceeds `capacity`.

    Raises ValueError if a sample is longer than `capacity` or has non-positive length (callers drop those first)."""
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    order = sorted(range(len(lengths)), key=lambda i: (-int(lengths[i]), i))   # deterministic tie-break
    bins: List[List[int]] = []
    buckets: Dict[int, List[int]] = {}          # remaining capacity -> bin ids with exactly that much left
    rems: List[int] = []                         # sorted distinct remaining capacities that have a bin
    for i in order:
        L = int(lengths[i])
        if L <= 0:
            raise ValueError(f"sample {i} has length {L}")
        if L > capacity:
            raise ValueError(f"sample {i} has {L} tokens > capacity {capacity}")
        pos = bisect_left(rems, L)
        if pos == len(rems):
            bins.append([i])
            b, new_rem = len(bins) - 1, capacity - L
        else:
            r = rems[pos]
            b = buckets[r].pop()
            if not buckets[r]:
                del buckets[r]
                del rems[pos]
            bins[b].append(i)
            new_rem = r - L
        if new_rem > 0:
            if new_rem not in buckets:
                buckets[new_rem] = []
                insort(rems, new_rem)
            buckets[new_rem].append(b)
    return bins


def pack_buffered(lengths: Sequence[int], capacity: int, buffer_size: int = 100_000, seed: int = 0) -> List[List[int]]:
    """Shuffle, cut the stream into buffers of `buffer_size` samples, best-fit-decreasing inside each buffer.

    Bigger buffers pack tighter (more candidates per bin) but mix samples from a wider window; with a global shuffle
    first, the window is random anyway, so the buffer only trades memory/time against efficiency. Returns bins as
    lists of GLOBAL indices; the bins are returned in a deterministic order (buffer by buffer)."""
    n = len(lengths)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    out: List[List[int]] = []
    for s in range(0, n, max(1, buffer_size)):
        chunk = idx[s: s + buffer_size]
        sub = best_fit_decreasing([lengths[i] for i in chunk], capacity)
        for b in sub:
            # keep a stable, shuffled order of samples inside a bin
            out.append([chunk[j] for j in b])
    return out


def packing_stats(lengths: Sequence[int], bins: Sequence[Sequence[int]], capacity: int) -> Dict[str, float]:
    n_tokens = sum(int(lengths[i]) for b in bins for i in b)
    n_slots = len(bins) * capacity
    return {
        "n_samples": sum(len(b) for b in bins),
        "n_bins": len(bins),
        "n_tokens": n_tokens,
        "capacity": capacity,
        "efficiency": (n_tokens / n_slots) if n_slots else 0.0,       # fraction of slots holding real tokens
        "padding_fraction": 1.0 - (n_tokens / n_slots) if n_slots else 0.0,
        "mean_samples_per_bin": (sum(len(b) for b in bins) / len(bins)) if bins else 0.0,
    }
