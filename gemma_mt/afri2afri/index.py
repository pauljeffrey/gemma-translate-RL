# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage 1 of the inter-African SFT build: turn the en->X pair shards into flat numpy arrays (hash of the English sentence, language id, ...).

Input: the shards of `gemma/pairs_final` (columns english, target, iso, source_dataset, licence, flipped, qe) -- the exact pairs the en->X model was
trained on (after the licence class, dedupe and QE gate of `gemma_gate_pairs.py`). Output (one .npz per shard, rows in shard order):
    key uint64 (pivot_key hash of english) | tkey uint64 (pivot_key hash of target) | iso uint8 | ds uint16 (id in the dataset table)
    | lic uint8 | qe float32 | enlen uint16 | tlen uint16 | flipped bool
A row is globally addressed as (shard number, row number in the shard).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from gemma_mt.afri2afri.text import hash64, pivot_key
from gemma_mt.languages import ALL_ISOS, canonical_iso

ISO_ID = {iso: i for i, iso in enumerate(ALL_ISOS)}
NO_LANG = 255
LICENCE_CLASSES = ("cc0", "mit", "apache", "cc-by", "odc-by", "mpl", "other", "unspecified", "share_alike", "non_commercial")


def iso_id_of(tag: str) -> int:
    c = canonical_iso(tag)
    return ISO_ID[c] if c in ISO_ID else NO_LANG


def index_shard(path: str, ds_table: Dict[str, int], min_key_chars: int = 3) -> Dict[str, np.ndarray]:
    """Index one shard. `ds_table` (dataset name -> id) is extended in place (call it from ONE process, or merge tables afterwards)."""
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["english", "target", "iso", "source_dataset", "licence", "flipped", "qe"]).to_pydict()
    n = len(t["english"])
    key = np.zeros(n, np.uint64)
    tkey = np.zeros(n, np.uint64)
    iso = np.full(n, NO_LANG, np.uint8)
    ds = np.zeros(n, np.uint16)
    lic = np.zeros(n, np.uint8)
    enlen = np.zeros(n, np.uint16)
    tlen = np.zeros(n, np.uint16)
    iso_cache: Dict[str, int] = {}
    lic_id = {c: i for i, c in enumerate(LICENCE_CLASSES)}
    for i in range(n):
        en, tg = t["english"][i], t["target"][i]
        k = pivot_key(en)
        key[i] = hash64(k) if len(k) >= min_key_chars else 0           # 0 = unusable key
        tkey[i] = hash64(pivot_key(tg))
        tag = t["iso"][i]
        v = iso_cache.get(tag)
        if v is None:
            v = iso_cache[tag] = iso_id_of(tag)
        iso[i] = v
        name = t["source_dataset"][i]
        d = ds_table.get(name)
        if d is None:
            d = ds_table[name] = len(ds_table)
        ds[i] = d
        lic[i] = lic_id.get(t["licence"][i], lic_id["other"])
        enlen[i] = min(len(en), 65535)
        tlen[i] = min(len(tg), 65535)
    return {"key": key, "tkey": tkey, "iso": iso, "ds": ds, "lic": lic, "qe": np.asarray(t["qe"], np.float32),
            "enlen": enlen, "tlen": tlen, "flipped": np.asarray(t["flipped"], bool)}


def merge_ds_tables(tables: Sequence[Dict[str, int]]) -> Tuple[Dict[str, int], List[np.ndarray]]:
    """Merge per-process dataset tables into one; returns (merged, [remap array per input table]) so shard `ds` columns can be re-mapped."""
    merged: Dict[str, int] = {}
    remaps = []
    for tb in tables:
        inv = {v: k for k, v in tb.items()}
        rm = np.zeros(max(inv) + 1 if inv else 0, np.uint16)
        for old in sorted(inv):
            name = inv[old]
            if name not in merged:
                merged[name] = len(merged)
            rm[old] = merged[name]
        remaps.append(rm)
    return merged, remaps


def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
