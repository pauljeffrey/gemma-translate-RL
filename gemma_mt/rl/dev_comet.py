# SPDX-License-Identifier: AGPL-3.0-or-later
"""Held-out reference-based AfriCOMET monitor on the dev translations (patch 3, owner 2026-11-03; config `heldout_comet`).

The policy trains on the reference-free QE ensemble (masakhane/africomet-qe-stl-1.1 + McGill-NLP/ssa-comet-mtl). `masakhane/africomet-stl-1.1` is a DIFFERENT checkpoint (reference-based: source, translation and the
human dev reference), so a rise of the QE reward that is not a rise of this score is a warning sign of reward hacking. At every dev evaluation the greedy dev translations (bench set and the aux sets) are scored against
the human dev references for the languages AfriCOMET supports (`gemma_mt.languages.QE_LANGUAGES` intersected with the dev languages).

Where it runs: a SEPARATE comet worker process (`python -m gemma_mt.rl.comet_worker --reference-based`, started with the same python / pythonpath as the QE worker: unbabel-comet needs the transformers-4.x directory)
that is started for the dev evaluation and released afterwards (`keep_loaded: false`, the default; frees ~2-3 GB between evaluations) or kept resident (`keep_loaded: true`; no model load per evaluation). The existing QE worker
cannot host it: `--reference-based` is a flag of the whole process, and its checkpoints are reference-free. A failure (no download, OOM, timeout) is logged and returns {}: the monitor never stops the training, and the rules
that need it (hacking alarm, combined stop) fall back to the chrF++-only behaviour.

Output keys (the same names as the chrF++ ones): `<set>/comet_macro`, `<set>/comet/<iso>`, `<set>/comet_n/<iso>`, `<set>/comet_macro_sd` (bootstrap over the dev sentences, `bootstrap` resamples), with `<set>` =
`bench` or `bench_aux/<name>`; and `heldout_comet/seconds|load_seconds|n|languages`. n per language is small (~40-60): the per-language values are noisy, the macro sd tells how much."""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from gemma_mt.languages import QE_LANGUAGES

REF_MODEL_ID = "masakhane/africomet-stl-1.1"


def macro_with_sd(per_lang: Dict[str, Sequence[float]], bootstrap: int = 200, seed: int = 0) -> Dict[str, float]:
    """Unweighted mean over languages of the per-language mean score, and its bootstrap sd (sentences resampled within each language)."""
    langs = sorted(per_lang)
    means = [float(np.mean(per_lang[l])) for l in langs]
    macro = float(np.mean(means)) if means else float("nan")
    sd = float("nan")
    if bootstrap > 1 and langs:
        rng = np.random.default_rng(seed)
        arrs = [np.asarray(per_lang[l], dtype=np.float64) for l in langs]
        boots = np.stack([a[rng.integers(0, len(a), size=(bootstrap, len(a)))].mean(axis=1) for a in arrs], axis=0).mean(axis=0)
        sd = float(boots.std())
    return {"macro": macro, "sd": sd}


class HeldoutComet:
    def __init__(self, model_id: str = REF_MODEL_ID, batch_size: int = 64, python: Optional[str] = None, pythonpath: Optional[str] = None, keep_loaded: bool = False, bootstrap: int = 200,
                 seed: int = 0, languages: Optional[Sequence[str]] = None, fake: bool = False, scorer: Any = None, startup_timeout: float = 900.0, request_timeout: float = 900.0):
        self.model_id, self.batch_size, self.python, self.pythonpath = model_id, int(batch_size), python or sys.executable, pythonpath
        self.keep_loaded, self.bootstrap, self.seed, self.fake = bool(keep_loaded), int(bootstrap), int(seed), bool(fake)
        self.languages = set(languages if languages is not None else QE_LANGUAGES)
        self.startup_timeout, self.request_timeout = startup_timeout, request_timeout
        self._scorer = scorer                      # an injected scorer (tests): callable (srcs, mts, refs) -> scores
        self._owned = False
        self.failures = 0

    def _get(self):
        if self._scorer is None:
            from gemma_mt.rl.comet import WorkerScorer

            self._scorer = WorkerScorer(python=self.python, model_id=self.model_id, batch_size=self.batch_size, fake=self.fake, pythonpath=self.pythonpath, reference_based=True,
                                        startup_timeout=self.startup_timeout, request_timeout=self.request_timeout)
            self._owned = True
        return self._scorer

    def close(self) -> None:
        if self._owned and self._scorer is not None:
            try:
                self._scorer.close()
            except Exception:                                  # noqa: BLE001
                pass
        if self._owned:
            self._scorer = None
            self._owned = False

    def score(self, preds: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        """`preds`: dicts {set, iso, src, hyp, ref} (set = "bench" or an aux name). Returns the metrics above ({} on failure or nothing to score)."""
        rows = [p for p in preds if p["iso"] in self.languages and str(p.get("ref", "")).strip()]
        if not rows:
            return {}
        t0 = time.time()
        try:
            sc = self._get()
            t_load = time.time() - t0
            t1 = time.time()
            scores = [float(x) for x in sc([p["src"] for p in rows], [p["hyp"] for p in rows], [p["ref"] for p in rows])]
            t_score = time.time() - t1
        except Exception as exc:                               # noqa: BLE001
            self.failures += 1
            print(f"[gemma_mt] WARNING: the held-out AfriCOMET monitor failed ({type(exc).__name__}: {exc}); no AfriCOMET at this evaluation", flush=True)
            self.close()
            return {}
        if not self.keep_loaded:
            self.close()
        if len(scores) != len(rows):
            return {}
        out: Dict[str, float] = {"heldout_comet/seconds": time.time() - t0, "heldout_comet/load_seconds": t_load, "heldout_comet/score_seconds": t_score, "heldout_comet/n": float(len(rows))}
        by_set: Dict[str, Dict[str, List[float]]] = {}
        for p, s in zip(rows, scores):
            by_set.setdefault(p["set"], {}).setdefault(p["iso"], []).append(s)
        for name, per_lang in by_set.items():
            prefix = "bench" if name == "bench" else f"bench_aux/{name}"
            m = macro_with_sd(per_lang, self.bootstrap, self.seed)
            out[f"{prefix}/comet_macro"], out[f"{prefix}/comet_macro_sd"] = m["macro"], m["sd"]
            for iso, v in per_lang.items():
                out[f"{prefix}/comet/{iso}"] = float(np.mean(v))
                out[f"{prefix}/comet_n/{iso}"] = float(len(v))
            if name == "bench":
                out["heldout_comet/languages"] = float(len(per_lang))
        return out
