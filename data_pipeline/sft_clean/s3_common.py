#!/usr/bin/env python3
"""Shared helpers of agent S3 (clean RL prompt pools + clean AfriScience-MT train set, 2026-10-28).

* `load_strict`   loads the protected index WITHOUT the FLORES+ dev train-half exemption (the strictest reading of the index; the exemption object is returned
                  separately so that the removals that exist only because of the train half can be counted and reported).
* `Checker`       fork-based pool: every worker holds the (copy-on-write shared) index, checks chunks of texts with `decon.Index.check_texts` (the unchanged decon.py
                  logic, rules R1 exact / R2 contained sentence / R3 shared word 8-gram, two normal forms) and optionally a small auxiliary index that only FLAGS.
* `Checker.run`   -> list of tuples per text: (r1, r2, r3, exempt_clears, a1, a2, a3)
                  r1..r3  protected-set bitmasks of the strict check (0 = no hit); exempt_clears: None when no hit, True when the same text has no hit once the
                  FLORES+ dev train-half exemption is attached (i.e. the text is removed only because of the train half), False otherwise;
                  a1..a3  masks of the auxiliary (flag-only) index.
Nothing here writes benchmark text anywhere.
"""
from __future__ import annotations

import functools
import multiprocessing as mp
import os
import sys
import time
from typing import List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decon as D  # noqa: E402
import protected_index_loader as L  # noqa: E402

RULES = ("R1_exact", "R2_contiguous", "R3_8gram")

_IX = None
_EX = None
_AUX = None


def load_strict(index: str, manifest: str):
    """-> (index without exemption, exemption index or None, manifest). The index file sha256 is verified by the loader against the manifest."""
    ix, m = L.load(index, manifest)
    ex = ix.exempt
    ix.exempt = None
    return ix, ex, m


def _work(texts: List[str]):
    r = _IX.check_texts(texts)
    hit = [i for i, t in enumerate(r) if (t[0] | t[1] | t[2])]
    clears = {}
    if hit and _EX is not None:
        _IX.exempt = _EX
        try:
            r2 = _IX.check_texts([texts[i] for i in hit])
        finally:
            _IX.exempt = None
        clears = {i: not (a[0] | a[1] | a[2]) for i, a in zip(hit, r2)}
    elif hit:
        clears = {i: False for i in hit}
    a = _AUX.check_texts(texts) if _AUX is not None else None
    D.norm.cache_clear()
    out = []
    for i, t in enumerate(r):
        if a is not None:
            out.append((t[0], t[1], t[2], clears.get(i), a[i][0], a[i][1], a[i][2]))
        else:
            out.append((t[0], t[1], t[2], clears.get(i), 0, 0, 0))
    return out


class Checker:
    def __init__(self, ix, ex=None, aux=None, nproc: int = 2):
        global _IX, _EX, _AUX
        _IX, _EX, _AUX = ix, ex, aux
        # a bounded cache of normalised strings: the auxiliary check re-uses the work of the main check (same texts, same arguments)
        if not hasattr(D, "_norm_orig"):
            D._norm_orig = D.norm
        D.norm = functools.lru_cache(maxsize=300000)(D._norm_orig)
        self.ix, self.aux = ix, aux
        self.nproc = nproc
        self.pool = mp.get_context("fork").Pool(nproc) if nproc > 1 else None

    def run(self, texts: Sequence[str], max_chars: int = 400_000, max_n: int = 128) -> List[Tuple]:
        """Check texts (any order, duplicates allowed); the result is aligned with `texts`."""
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        chunks, cur, cur_c = [], [], 0
        for i in order:
            L_ = len(texts[i])
            if cur and (cur_c + L_ > max_chars or len(cur) >= max_n):
                chunks.append(cur)
                cur, cur_c = [], 0
            cur.append(i)
            cur_c += L_
        if cur:
            chunks.append(cur)
        res: List[Optional[Tuple]] = [None] * len(texts)
        payload = [[texts[i] for i in ch] for ch in chunks]
        it = self.pool.imap(_work, payload) if self.pool else map(_work, payload)
        for ch, out in zip(chunks, it):
            for i, o in zip(ch, out):
                res[i] = o
        return res

    def close(self):
        if self.pool:
            self.pool.close()
            self.pool.join()


def names_of(ix, mask: int) -> List[str]:
    return ix.names(mask) if mask else []


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
