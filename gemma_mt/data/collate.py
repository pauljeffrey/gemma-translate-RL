# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batches of packed sequences (numpy in, numpy out; the trainer moves them to torch).

Two layouts, chosen by the attention path (see gemma_mt.attention):

  padded    [B, T] rows padded to the longest row of the batch (or to `pad_to`). Used with eager / SDPA, which need
            a rectangular batch and an explicit block-diagonal mask built from `segment_ids`.
  flatten   [1, N] every row concatenated, padding REMOVED (padding-free). Used with FlashAttention-2 varlen:
            `position_ids` restart at every sample and `cu_seqlens` gives the boundaries, so no token attends
            across a sample. N is the number of real tokens of the whole batch.

Both carry `num_target_tokens`: how many labels carry loss, counted AFTER the one-position shift
(labels[:, 1:]), i.e. exactly the number of terms in the summed loss.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

IGNORE = -100


def count_target_tokens(labels: np.ndarray) -> int:
    return int((labels[:, 1:] != IGNORE).sum())


def collate_padded(items: Sequence[Dict], pad_id: int, pad_to: Optional[int] = None,
                   pad_multiple_of: int = 1) -> Dict[str, np.ndarray]:
    T = max(int(it["input_ids"].shape[0]) for it in items)
    if pad_to is not None:
        if T > pad_to:
            raise ValueError(f"row of {T} tokens exceeds pad_to={pad_to}")
        T = pad_to
    if pad_multiple_of > 1:
        T = -(-T // pad_multiple_of) * pad_multiple_of
    B = len(items)
    ids = np.full((B, T), pad_id, np.int64)
    lab = np.full((B, T), IGNORE, np.int64)
    pos = np.zeros((B, T), np.int64)
    seg = np.zeros((B, T), np.int32)
    tiso = np.full((B, T), -1, np.int16)
    for r, it in enumerate(items):
        n = int(it["input_ids"].shape[0])
        ids[r, :n], lab[r, :n], pos[r, :n], seg[r, :n] = it["input_ids"], it["labels"], it["position_ids"], it["segment_ids"]
        tiso[r, :n] = it["token_iso"]
        if n < T:                                     # padding: its own segment (0), positions 0..k-1
            pos[r, n:] = np.arange(T - n)
    return {"input_ids": ids, "labels": lab, "position_ids": pos, "segment_ids": seg, "token_iso": tiso,
            "num_target_tokens": count_target_tokens(lab)}


def collate_flatten(items: Sequence[Dict]) -> Dict[str, np.ndarray]:
    ids, lab, pos, seg, lens, tiso = [], [], [], [], [], []
    seg_base = 0
    for it in items:
        n = int(it["length"])                         # drop padding
        ids.append(it["input_ids"][:n])
        lab.append(it["labels"][:n])
        pos.append(it["position_ids"][:n])
        tiso.append(it["token_iso"][:n])
        seg.append(it["segment_ids"][:n] + seg_base)
        cu = np.asarray(it["cu_seqlens"])
        # only the sample boundaries up to the real length (drop a trailing padding segment)
        cu = cu[cu <= n]
        lens.extend(np.diff(cu).tolist())
        seg_base += int(it["n_samples"])
    input_ids = np.concatenate(ids)[None]
    labels = np.concatenate(lab)[None]
    cu_seqlens = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    assert int(cu_seqlens[-1]) == input_ids.shape[1], "cu_seqlens does not cover the flattened batch"
    return {"input_ids": input_ids, "labels": labels, "position_ids": np.concatenate(pos)[None],
            "segment_ids": np.concatenate(seg)[None].astype(np.int32), "cu_seqlens": cu_seqlens,
            "token_iso": np.concatenate(tiso)[None],
            "max_seqlen": int(max(lens)) if lens else 0, "num_target_tokens": count_target_tokens(labels)}


class PackedCollator:
    def __init__(self, mode: str = "padded", pad_id: int = 0, pad_to: Optional[int] = None, pad_multiple_of: int = 1):
        if mode not in ("padded", "flatten"):
            raise ValueError("mode must be 'padded' or 'flatten'")
        self.mode, self.pad_id, self.pad_to, self.pad_multiple_of = mode, pad_id, pad_to, pad_multiple_of

    def __call__(self, items: Sequence[Dict]) -> Dict[str, np.ndarray]:
        if self.mode == "flatten":
            return collate_flatten(items)
        return collate_padded(items, self.pad_id, self.pad_to, self.pad_multiple_of)
