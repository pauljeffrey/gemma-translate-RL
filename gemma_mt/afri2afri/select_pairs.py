# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pair selection of the inter-African SFT (D-46): from candidate pair RECORDS (index arrays only, no text) to the chosen train / validation pairs.

A record is one UNORDERED pair {a, b} of two translations of the same thing:
    origin 0  "pivot"   two rows of the en->X table that share the English sentence (ref_a / ref_b = row numbers of the caller's row table)
    origin 1  "direct"  one human- or machine-made row of the corpus that already pairs two African languages (ref_a = row of the direct table)
    origin 2  "mined"   one row of the NLLB-mined candidate table (ref_a = its row; class `mined_nllb`)
Steps (`select_records`): drop records whose two texts are identical -> de-duplicate on the text pair (a direct record beats a pivot one, a better
provenance class beats a worse one) -> split off validation GROUPS (all pairs of one English sentence are validation together, so a validation sentence
is never trained on with another partner) -> UniMax allocation over the unordered language pairs -> per pair, `policy.compose` quotas over the
provenance classes -> uniform sampling without replacement inside every (pair, class).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from gemma_mt.afri2afri import pivot as pv, policy

CLASS_OF_TIER = np.array([4, 2, 3, 0], np.int8)           # provenance tier (0 synthetic, 1 mixed, 2 religious, 3 human) -> index in policy.CLASSES (1 = mined_nllb has no tier: its records carry the class directly)
MINED_CLASS = 1
ORIGIN_PRIORITY = np.array([2, 0, 1])                       # duplicate text pairs: origin 1 (direct) wins over 2 (mined) over 0 (pivot)
N_CLASS = len(policy.CLASSES)
_M = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def mix64(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """splitmix64 finaliser of an array of uint64 (vectorised hash; deterministic)."""
    add = np.uint64(((int(seed) + 1) * 0x9E3779B97F4A7C15) % (1 << 64))
    z = np.asarray(x, np.uint64) + add
    z = (z ^ (z >> np.uint64(30))) * _M
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def unit_fraction(x: np.ndarray, seed: int = 0) -> np.ndarray:
    """Uniform [0, 1) from uint64 keys."""
    return (mix64(x, seed) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def records_from_groups(group: np.ndarray, iso: np.ndarray, tier: np.ndarray, tkey: np.ndarray, key: np.ndarray, valid: np.ndarray) -> Dict[str, np.ndarray]:
    """Pivot records of every group (rows are sorted by `group`; `valid` masks rows that failed LID / filters). Returned refs index the INPUT arrays."""
    idx = np.flatnonzero(valid)
    starts, sizes = pv.group_bounds(group[idx])
    _, ra, rb = pv.enumerate_pairs(starts, sizes)
    a, b = idx[ra], idx[rb]
    swap = iso[a] > iso[b]
    a2, b2 = np.where(swap, b, a), np.where(swap, a, b)
    t = np.minimum(tier[a2], tier[b2])
    return {"iso_a": iso[a2].astype(np.uint8), "iso_b": iso[b2].astype(np.uint8), "cls": CLASS_OF_TIER[t].astype(np.int8),
            "h": mix64(tkey[a2]) * np.uint64(3) + mix64(tkey[b2]), "gkey": key[a2], "ref_a": a2.astype(np.int64), "ref_b": b2.astype(np.int64),
            "origin": np.zeros(len(a2), np.uint8)}


def concat_records(*parts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def select_records(rec: Dict[str, np.ndarray], n_lang: int, budget: float, max_epochs: float = 3.0, val_frac: float = 0.02, val_cap: int = 40,
                   seed: int = 0, caps: Optional[Mapping[str, float]] = None, min_total: int = 1) -> Dict[str, object]:
    """Returns {'train': index array into `rec`, 'val': index array, 'pid_info': {pid: {...}}, 'epochs': {pid: planned passes over the chosen pairs}, 'dropped': {...}}."""
    same = rec["iso_a"] == rec["iso_b"]                       # a pair of one language with itself (e.g. the mined `akan-twi` repo: both are tag `twi`) is no translation pair
    if same.any():
        rec = {k: v[~same] for k, v in rec.items()}
    n = len(rec["iso_a"])
    dropped: Dict[str, int] = {"same_language_pair": int(same.sum())}
    pid = rec["iso_a"].astype(np.int64) * n_lang + rec["iso_b"].astype(np.int64)
    # --- de-duplicate on (pair id, text-pair hash): direct first, then the better class
    order = np.lexsort((rec["cls"], ORIGIN_PRIORITY[rec["origin"]], rec["h"], pid))      # per (pid, h): direct first, then mined, then pivot; then the lowest class index = best
    first = np.ones(n, bool)
    if n:
        first[1:] = (pid[order][1:] != pid[order][:-1]) | (rec["h"][order][1:] != rec["h"][order][:-1])
    keep = np.zeros(n, bool)
    keep[order[first]] = True
    dropped["duplicate_text_pair"] = int(n - keep.sum())
    idx = np.flatnonzero(keep)
    # --- validation groups
    is_val_group = unit_fraction(rec["gkey"][idx], seed + 17) < val_frac
    val_idx, train_idx = idx[is_val_group], idx[~is_val_group]
    # validation: at most val_cap per language pair, random
    if len(val_idx):
        keep_v = pv.pick_by_quota(pid[val_idx], unit_fraction(rec["h"][val_idx], seed + 5), np.full(n_lang * n_lang, val_cap, np.int64))
        dropped["val_over_cap"] = int(len(val_idx) - keep_v.sum())
        val_idx = val_idx[keep_v]
    # --- allocation over unordered language pairs (train records only)
    tp, tc = pid[train_idx], rec["cls"][train_idx].astype(np.int64)
    counts = np.zeros((n_lang * n_lang, N_CLASS), np.int64)
    np.add.at(counts, (tp, tc), 1)
    totals = counts.sum(1)
    active = np.flatnonzero(totals >= min_total)
    alloc = policy.unimax(totals[active], budget, max_epochs)
    quota = np.zeros((n_lang * n_lang, N_CLASS), np.int64)
    epochs: Dict[int, float] = {}
    info: Dict[int, dict] = {}
    for p, al in zip(active, alloc):
        distinct = int(min(totals[p], round(al)))
        comp = policy.compose({c: int(counts[p, i]) for i, c in enumerate(policy.CLASSES)}, distinct, caps)
        for i, c in enumerate(policy.CLASSES):
            quota[p, i] = comp[c]
        taken = int(sum(comp.values()))
        epochs[int(p)] = (al / taken) if taken else 0.0
        info[int(p)] = {"available": int(totals[p]), "alloc": float(al), "distinct": taken, "by_class": {c: int(comp[c]) for c in policy.CLASSES}}
    seg = tp * N_CLASS + tc
    pick = pv.pick_by_quota(seg, unit_fraction(rec["h"][train_idx], seed + 3), quota.reshape(-1))
    return {"train": train_idx[pick], "val": val_idx, "pid_info": info, "epochs": epochs, "dropped": dropped, "pid": pid}
