# SPDX-License-Identifier: AGPL-3.0-or-later
"""Disk-cached pair scorer for the long-document reward study: wraps a QEPairScorer (AfriCOMET-QE-STL-1.1 + SSA-COMET-MTL) and remembers every (model, source, translation)."""
from __future__ import annotations

import hashlib
import os
import pickle
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence


def _k(m: int, s: str, t: str) -> str:
    return hashlib.md5(f"{m}\x00{s}\x00{t}".encode()).hexdigest()[:20]


class CachedPairScorer:
    """Same interface as qe_ensemble.QEPairScorer: `__call__` = AfriCOMET-QE, `score_secondary` = SSA-COMET-MTL. `inner=None` = cache-only (a miss raises)."""

    def __init__(self, inner=None, cache_dir: Optional[str] = None, chunk: int = 4096):
        self.inner, self.dir, self.chunk = inner, cache_dir, chunk
        self.mem: Dict[str, float] = {}
        self.new: Dict[str, float] = {}
        self.secondary = SimpleNamespace(last_stats={})
        self.seconds = [0.0, 0.0]
        self.total_seconds = [0.0, 0.0]
        self.total_pairs = [0, 0]
        self.misses = [0, 0]
        self.hits = [0, 0]
        if cache_dir and os.path.isdir(cache_dir):
            for fn in sorted(os.listdir(cache_dir)):
                if fn.endswith(".pkl"):
                    with open(os.path.join(cache_dir, fn), "rb") as f:
                        self.mem.update(pickle.load(f))
        print(f"[qe_cache] {len(self.mem)} cached scores", flush=True)

    def _score(self, m: int, srcs: Sequence[str], mts: Sequence[str]) -> List[float]:
        keys = [_k(m, s, t) for s, t in zip(srcs, mts)]
        need = {}
        for k, s, t in zip(keys, srcs, mts):
            if k not in self.mem and k not in need:
                need[k] = (s, t)
        self.hits[m] += len(keys) - len(need)
        if need:
            if self.inner is None:
                raise KeyError(f"{len(need)} pairs not in the cache (cache-only scorer)")
            ks = list(need)
            t0 = time.time()
            for a in range(0, len(ks), self.chunk):
                part = ks[a: a + self.chunk]
                S = [need[k][0] for k in part]
                T = [need[k][1] for k in part]
                vals = self.inner(S, T) if m == 0 else self.inner.score_secondary(S, T)
                for k, v in zip(part, vals):
                    self.mem[k] = float(v)
                    self.new[k] = float(v)
            self.seconds[m] += time.time() - t0
            self.total_seconds[m] += time.time() - t0
            self.total_pairs[m] += len(ks)
            self.misses[m] += len(ks)
            self.flush()
        return [self.mem[k] for k in keys]

    def __call__(self, srcs, mts, refs=None):
        return self._score(0, srcs, mts)

    def score_secondary(self, srcs, mts):
        return self._score(1, srcs, mts)

    def flush(self):
        if self.dir and self.new:
            os.makedirs(self.dir, exist_ok=True)
            fn = os.path.join(self.dir, f"c{int(time.time() * 1000)}.pkl")
            with open(fn + ".tmp", "wb") as f:
                pickle.dump(self.new, f)
            os.replace(fn + ".tmp", fn)
            self.new = {}

    def pop_times(self):
        t = tuple(self.seconds)
        self.seconds = [0.0, 0.0]
        return t

    def close(self):
        self.flush()
        if self.inner is not None and hasattr(self.inner, "close"):
            self.inner.close()


def build_scorer(cache_dir: str, gpu: bool = True, batch_size: int = 128):
    """The real two-model worker (venv_gemma python, transformers 4 on the child's PYTHONPATH) behind the cache."""
    from gemma_mt.rl.qe_ensemble import build_pair_scorer

    spec = {"type": "worker", "python": "/workspace/venv_gemma/bin/python", "pythonpath": "/workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs",
            "model_id": "masakhane/africomet-qe-stl-1.1", "ensemble_model_id": "McGill-NLP/ssa-comet-mtl", "batch_size": batch_size,
            "ensemble_batch_size": batch_size, "max_batch_chars": 64000}
    return CachedPairScorer(build_pair_scorer(spec) if gpu else None, cache_dir)
