# SPDX-License-Identifier: AGPL-3.0-or-later
import random

from gemma_mt.data.packing import best_fit_decreasing, pack_buffered, packing_stats
import pytest


def _check(lengths, bins, cap):
    seen = sorted(i for b in bins for i in b)
    assert seen == list(range(len(lengths))), "every sample exactly once"
    for b in bins:
        assert sum(lengths[i] for i in b) <= cap
        assert len(b) >= 1


def test_every_sample_once_and_capacity_respected():
    rng = random.Random(0)
    lengths = [rng.randint(5, 300) for _ in range(2000)]
    bins = best_fit_decreasing(lengths, 512)
    _check(lengths, bins, 512)


def test_efficiency_beats_sequential_first_fit_and_is_high():
    rng = random.Random(1)
    lengths = [rng.randint(20, 200) for _ in range(5000)]
    cap = 2048
    bins = best_fit_decreasing(lengths, cap)
    st = packing_stats(lengths, bins, cap)
    # naive: pack in arrival order, open a new bin when the next sample does not fit
    naive, cur = 0, 0
    for L in lengths:
        if cur + L > cap:
            naive += 1
            cur = 0
        cur += L
    naive += 1
    assert len(bins) <= naive
    assert st["efficiency"] > 0.99


def test_sample_exactly_capacity_and_oversize_rejected():
    assert best_fit_decreasing([10, 10], 10) == [[0], [1]]
    with pytest.raises(ValueError):
        best_fit_decreasing([11], 10)
    with pytest.raises(ValueError):
        best_fit_decreasing([0], 10)


def test_best_fit_picks_tightest_bin():
    # capacity 10: 6, 5 -> two bins (rem 4, rem 5); a 4 must go to the bin with remaining 4 (the 6-bin)
    bins = best_fit_decreasing([6, 5, 4], 10)
    assert sorted(sorted(b) for b in bins) == [[0, 2], [1]]


def test_buffered_packing_is_deterministic_and_complete():
    rng = random.Random(2)
    lengths = [rng.randint(5, 100) for _ in range(1500)]
    a = pack_buffered(lengths, 256, buffer_size=400, seed=7)
    b = pack_buffered(lengths, 256, buffer_size=400, seed=7)
    c = pack_buffered(lengths, 256, buffer_size=400, seed=8)
    assert a == b and a != c
    _check(lengths, a, 256)
