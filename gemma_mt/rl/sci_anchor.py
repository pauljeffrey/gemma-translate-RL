# SPDX-License-Identifier: AGPL-3.0-or-later
"""Science anchor (D-52, owner 2026-10-21, science run): numbers, units, formulas and protected scientific tokens must survive the translation of a chunk.

AfriCOMET-QE is nearly blind to a dropped or altered number, a lost unit or a mangled formula, and the long scientific text of the science run is full of them.
This module scores ONE chunk (a source sentence and its translation) with the string-level metrics of the hard long-document split
(`gemma_mt.eval.longdoc_hard`, the same definitions the benchmark uses, so what the reward pushes towards is what the benchmark measures):

    num_recall        share of the source numerals (small integers 0..10 excluded: they may be spelled out) found in the output; separators and numeral scripts normalised
    num_precision     share of the output numerals (same rule) that come from the source (an invented figure lowers it)
    pair_recall       share of the source `number + unit symbol` pairs (5 mg, 37.5 C, 12 %) found as the same pair in the output
    protected_recall  share of the source's protected tokens (acronyms, gene / chemical names, binomials, drug stems, citation markers) found verbatim in the output;
                      classes marked latin_only are scored only when the output is Latin script (Amharic / Tigrinya transliterate names, they must not be punished)
    formula_recall    share of the source's `$..$` / `\\(..\\)` / LaTeX formula spans reproduced exactly (None when the source has none)

Like the lexical penalty (rl/lexical.py) it is a PENALTY only, never a bonus (copying the source or stuffing it with its tokens buys nothing): the deficit of a feature is
max(0, t - f) / t where t is the human 10th percentile of the feature (`calibration/science_anchor_v1.json`, built by `scripts/build_science_anchor_calibration.py` from the
AfriScience-MT human references; pooled over the six languages, per-language values where the sample is large enough) and the deficit of a chunk is the mean over the
features the chunk has (a feature the source side does not have is skipped). The reward subtracts `reward.science_anchor_penalty` x deficit per chunk (floor: the failure
penalty), only for samples whose prompt domain is in `reward.science_anchor_domains` (default ["science"]; general replay prompts are untouched). Off by default (weight 0).
"""

from __future__ import annotations

import json
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

FEATURES = ("num_recall", "num_precision", "pair_recall", "protected_recall", "formula_recall")
LATIN_MIN_SHARE = 0.8         # protected tokens are judged only when at least this share of the output letters is Latin
MIN_THRESHOLD = 0.2            # a feature whose human P10 is below this carries no usable signal (humans themselves fail it): skipped
# used when no calibration file is given: "humans keep (nearly) everything that is a number / a unit / a protected token"
DEFAULT_THRESHOLDS: Dict[str, float] = {"num_recall": 0.9, "num_precision": 0.9, "pair_recall": 0.9, "protected_recall": 0.8, "formula_recall": 0.9}


def features(src: str, mt: str) -> Dict[str, Optional[float]]:
    """The five features of one (source, translation) chunk; None where the source (or, for the precision, the output) has nothing to measure."""
    from gemma_mt.eval import longdoc_hard as H

    src = unicodedata.normalize("NFC", src or "")
    mt = unicodedata.normalize("NFC", mt or "")
    nm = H.number_metrics(src, mt)
    pm = H.protected_metrics(src, mt)
    fm = H.formula_metrics(src, mt)
    # a non-Latin output (Amharic, Tigrinya: Ge'ez script) transliterates acronyms and names: the protected-token recall is not judged there (numbers, units, formulas still are)
    return {
        "num_recall": nm.get("num_ge11_recall"),
        "num_precision": nm.get("num_ge11_precision"),
        "pair_recall": nm.get("pair_recall"),
        "protected_recall": pm.get("protected_recall") if H.latin_share(mt) >= LATIN_MIN_SHARE else None,
        "formula_recall": fm.get("formula_exact_share"),
    }


def deficit(feats: Dict[str, Optional[float]], thresholds: Dict[str, float]) -> float:
    """Mean over the enabled features of max(0, t - f) / t, 0..1 (0 = every feature at or above its human 10th percentile)."""
    ds = []
    for k, t in thresholds.items():
        v = feats.get(k)
        if v is None or t < MIN_THRESHOLD:
            continue
        ds.append(max(0.0, t - v) / t)
    return float(sum(ds) / len(ds)) if ds else 0.0


class ScienceAnchor:
    """Thresholds per language (calibration file) and the per-chunk deficit."""

    def __init__(self, path: Optional[str] = None):
        self.default: Dict[str, float] = dict(DEFAULT_THRESHOLDS)
        self.languages: Dict[str, Dict[str, float]] = {}
        self.source = "built-in defaults"
        if path:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
            self.default = {k: float(v["p10"]) for k, v in (blob.get("default") or {}).items() if k in FEATURES}
            for iso, blk in (blob.get("languages") or {}).items():
                self.languages[iso] = {k: float(v["p10"]) for k, v in blk.items() if k in FEATURES}
            self.source = path
            if not self.default:
                raise ValueError(f"{path}: no pooled thresholds")

    def thresholds(self, lang: str) -> Dict[str, float]:
        return {**self.default, **self.languages.get(lang, {})}

    def feats(self, src: str, mt: str) -> Dict[str, Optional[float]]:
        return features(src, mt)

    def deficit(self, lang: str, src: str, mt: str) -> float:
        return deficit(features(src, mt), self.thresholds(lang))


def build_calibration(rows: Iterable[Tuple[str, str, str]], min_n: int = 200, min_n_language: int = 150, q_low: float = 0.10) -> Dict:
    """rows of (iso, english sentence, human reference sentence) -> the calibration blob: pooled and per-language {feature: {p10, p50, n}}.
    A per-language entry exists only for features with at least `min_n_language` measured sentences; the pooled entry needs `min_n`."""
    import numpy as np

    by_lang: Dict[str, Dict[str, List[float]]] = {}
    pooled: Dict[str, List[float]] = {k: [] for k in FEATURES}
    for iso, s, r in rows:
        f = features(s, r)
        d = by_lang.setdefault(iso, {k: [] for k in FEATURES})
        for k in FEATURES:
            if f[k] is not None:
                d[k].append(f[k])
                pooled[k].append(f[k])

    def summ(v: Sequence[float]) -> Dict[str, float]:
        return {"p10": float(np.quantile(v, q_low)), "p50": float(np.median(v)), "mean": float(np.mean(v)), "n": len(v)}

    return {"default": {k: summ(v) for k, v in pooled.items() if len(v) >= min_n},
            "languages": {iso: {k: summ(v) for k, v in d.items() if len(v) >= min_n_language} for iso, d in sorted(by_lang.items())}}
