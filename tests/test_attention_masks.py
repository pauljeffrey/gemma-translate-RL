# SPDX-License-Identifier: AGPL-3.0-or-later
"""Numpy-only checks of the block-diagonal mask construction (run everywhere, no torch needed)."""
import numpy as np

from gemma_mt.attention import block_diagonal_causal_mask_np, segments_from_position_ids


def test_segments_from_position_ids_matches_boundaries():
    pos = np.array([[0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1]])   # last two = padding segment
    seg = segments_from_position_ids(pos)
    assert seg.tolist() == [[0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3]]
    assert segments_from_position_ids(np.array([0, 0, 0])).tolist() == [0, 1, 2]    # length-1 samples


def test_block_diagonal_causal_mask_properties():
    seg = np.array([[1, 1, 1, 2, 2, 3]])
    m = block_diagonal_causal_mask_np(seg)[0, 0]
    T = seg.shape[1]
    for q in range(T):
        for k in range(T):
            assert m[q, k] == (k <= q and seg[0, q] == seg[0, k])
    assert m.diagonal().all(), "a token always attends to itself (no all-masked rows)"


def test_sliding_window_inside_a_segment_uses_relative_distance():
    seg = np.array([[1] * 6 + [2] * 6])
    m = block_diagonal_causal_mask_np(seg, sliding_window=3)[0, 0]
    # inside segment 2 (indices 6..11): token 11 sees 9, 10, 11 only (distance < 3)
    assert np.nonzero(m[11])[0].tolist() == [9, 10, 11]
    # the first token of segment 2 sees only itself, even though indices 4, 5 are within the window
    assert np.nonzero(m[6])[0].tolist() == [6]
    # the window never crosses into segment 1
    assert np.nonzero(m[7])[0].tolist() == [6, 7]
    # a full-attention mask differs from the windowed one only inside segments longer than the window
    full = block_diagonal_causal_mask_np(seg)[0, 0]
    assert full[11, 6] and not m[11, 6]
