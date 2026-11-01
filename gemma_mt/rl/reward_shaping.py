# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reward shaping for the AfriCOMET-QE content score: teach "avoid failures" AND "good versus great".

Ported from the repo's rl/shaping.py (and rlhf/scripts/calibrate_shaping.py). The mapping is LANGUAGE-DEPENDENT
because AfriCOMET-QE score levels differ by language: the anchors are measured per language on human reference
translations, not guessed.

    c  = raw AfriCOMET-QE score of (english source, translation)          roughly 0 .. 1
    t_fail  = P10 of QE(source, HUMAN reference) in that language, never below FAIL_FLOOR (0.46)
    t_mid   = P50 of the same
    t_great = P90 of the same
    (calibrate_shaping.py further raises t_fail to the P90 of QE on MISMATCHED translations: a fluent but wrong
     sentence must never land above the failure line)

Piecewise-linear and monotone non-decreasing in c, so the ranking of completions inside a GRPO group is never
changed by the shaping; only the spacing is (it spends resolution where the policy lives and punishes failures).

PRESETS (OPEN QUESTION Q-01 in DECISIONS.md: which one to train with)

  repo        DEFAULT, CONFIRMED by the owner 2026-10-12 ("i believe it is better"). Exactly the repo's curve, range [-2, +2.25]
                 c = 0        -> -2      (neg_scale)
                 c = t_fail   ->  0
                 c = t_mid    -> +1
                 c = t_great  -> +2
                 c = 1        -> +2.25   (beyond "as good as a human": a tiny slope, a score above what human
                                          translations get is where a learned metric is gamed)
  owner_note  Same per-language anchors, positive side capped at +1, range [-2, +1]
                 c = 0 -> -2,  t_fail -> 0,  t_mid -> +1,  flat at +1 above t_mid.
              This is how the project owner remembers the repo's range; it keeps good-vs-great unresolved above
              t_mid (all translations at least as good as the median human one get the same reward).

`adaptive` mode (repo): t_fail stays anchored to the calibration while t_mid / t_great follow EMA-smoothed rolling
quantiles (P50 / P90) of the policy's own scores for that language, after `warmup` scores.

Calibration file (written by gemma_mt.rl.calibrate_shaping), JSON:
    { "yor": {"anchors": [t_fail, t_mid, t_great], "usable": true, "reliability": 0.93, ...}, "hau": {...}, ... }
Keys are our language tags; keys of the repo's form "eng>yor" are accepted too (the part after ">" is used).
A language with `usable: false` (QE cannot separate real from mismatched translations there, or no reference set exists) gets advantage
weight 0 from `reliability()`; a language absent from the file uses `default_anchors` and weight 1. Keys starting with "_" (`_meta`) are skipped.
Further per-language blocks the loader reads: `len_ratio` {p05, p95 -> length window; corpus, s2 -> language-aware length model of the omission
detector}, `omission` {cost}, `lexical` {feature: p10}, `diacritics` (rl/diacritics.py), `content` {mode: qe|blend|consensus, beta} (D-38, rl/consensus.py),
`lid` {labels, enabled, min_chars, recall, ...} (D-37, rl/lid.py). `fail_floor` replaces the constant FAIL_FLOOR as the lowest
t_fail (reward.fail_floor; D-32).
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

FAIL_FLOOR = 0.46
DEFAULT_ANCHORS: Tuple[float, float, float] = (FAIL_FLOOR, 0.60, 0.75)     # (t_fail, t_mid, t_great)

PRESET_Y: Dict[str, Tuple[float, float, float, float]] = {
    # reward at t_fail, t_mid, t_great and at c = 1  (the value at c = 0 is -neg_scale)
    "repo": (0.0, 1.0, 2.0, 2.25),
    "owner_note": (0.0, 1.0, 1.0, 1.0),
}


def preset_range(preset: str, neg_scale: float = 2.0) -> Tuple[float, float]:
    """(min, max) reward the preset can produce."""
    y = PRESET_Y[_check_preset(preset)]
    return (-neg_scale, max(y))


def _check_preset(preset: str) -> str:
    if preset not in PRESET_Y:
        raise ValueError(f"unknown shaping preset {preset!r}; choose one of {sorted(PRESET_Y)}")
    return preset


def make_curve(anchors: Sequence[float], neg_scale: float = 2.0, preset: str = "repo", floor: float = FAIL_FLOOR):
    t_fail, t_mid, t_great = anchors
    t_fail = max(float(t_fail), floor)
    eps = 1e-3                                       # keep the knots strictly increasing whatever the calibration says
    t_mid = max(float(t_mid), t_fail + eps)
    t_great = max(float(t_great), t_mid + eps)
    xs = [0.0, t_fail, t_mid, t_great, max(1.0, t_great + eps)]
    ys = [-neg_scale, *PRESET_Y[_check_preset(preset)]]
    return xs, ys


def shape(c: float, anchors: Sequence[float] = DEFAULT_ANCHORS, neg_scale: float = 2.0, preset: str = "repo") -> float:
    xs, ys = make_curve(anchors, neg_scale, preset)
    if c <= xs[0]:
        return ys[0]
    if c >= xs[-1]:
        return ys[-1]
    return float(np.interp(c, xs, ys))


def _lang_key(key: str) -> str:
    return key.split(">")[-1].strip()


class ContentShaper:
    """mode: calibrated | adaptive. `calibration` maps language tag -> (t_fail, t_mid, t_great)."""

    def __init__(self, mode: str = "calibrated", calibration: Optional[Dict[str, Sequence[float]]] = None,
                 preset: str = "repo", neg_scale: float = 2.0, window: int = 2000, warmup: int = 200, ema: float = 0.9,
                 p_mid: float = 0.5, p_great: float = 0.9, default_anchors: Sequence[float] = DEFAULT_ANCHORS,
                 fail_floor: float = FAIL_FLOOR):
        if mode not in ("calibrated", "adaptive"):
            raise ValueError("mode must be 'calibrated' or 'adaptive'")
        self.preset = _check_preset(preset)
        self.fail_floor = float(fail_floor)                      # t_fail is never below this (0.46 = the AfriCOMET-scale floor of the first 12 languages; D-32)
        self.default = tuple(default_anchors)
        self.mode, self.cal, self.neg, self.warmup, self.ema = mode, dict(calibration or {}), neg_scale, warmup, ema
        self.p_mid, self.p_great = p_mid, p_great
        self._win: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._live: Dict[str, Tuple[float, float]] = {}
        self.rel: Dict[str, float] = {}
        self.unusable: set = set()
        self.len_window: Dict[str, Tuple[float, float]] = {}     # language -> (p05, p95) of len(reference) / len(source) in characters
        self.lexical: Dict[str, Dict[str, float]] = {}           # language -> {feature: human p10} (rl/lexical.py); empty = no lexical penalty
        self.diacritics: Dict[str, dict] = {}                    # language -> calibration block `diacritics` (rl/diacritics.py); empty = no diacritic penalty
        self.word_ratio: Dict[str, Dict[str, float]] = {}        # language -> {p95, p99, n} of len(reference.split()) / len(source.split()) per sentence (2026-10-21, word_excess); '*' = pooled prior
        self.len_stats: Dict[str, Dict[str, float]] = {}         # language -> {corpus, s2, p05, p95} human length statistics (omission detector, segments.LengthModel)
        self.content: Dict[str, dict] = {}                       # language -> calibration block `content` {mode: qe|blend|consensus, beta, ...} (D-38); empty = every language "qe"
        self.lid: Dict[str, dict] = {}                           # language -> calibration block `lid` {labels, enabled, min_chars, recall, ...} (D-37); empty = languages.py labels

    # -- construction
    @classmethod
    def from_file(cls, path: Optional[str], reliability_from: str = "auc", **kw) -> "ContentShaper":
        """reliability_from: "auc" (cross-sentence AUC of QE, the repo's choice) or "same_source" (accuracy with which QE ranks a human
        reference above corrupted versions of the SAME source, which is what a GRPO group ranking uses; falls back to AUC when absent)."""
        cal, rel, unusable, win, lex, dia, lst, content, lid = {}, {}, set(), {}, {}, {}, {}, {}, {}
        wr: Dict[str, Dict[str, float]] = {}
        if path and not os.path.exists(path):
            print(f"[reward_shaping] calibration file {path!r} not found: using the default anchors for every "
                  f"language (run gemma_mt.rl.calibrate_shaping first)", file=sys.stderr)
            path = None
        if path:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                pooled = data.get("_word_ratio_pooled")
                if isinstance(pooled, dict) and pooled.get("p99"):
                    wr["*"] = {k2: float(v2) for k2, v2 in pooled.items() if isinstance(v2, (int, float))}
                for k, v in data.items():
                    if k.startswith("_"):                        # `_meta` and other file-level keys
                        continue
                    k = _lang_key(k)
                    if isinstance(v, dict):
                        if v.get("lid"):
                            lid[k] = v["lid"]                    # D-37 (read for unusable languages too: the guard does not need QE)
                        if v.get("content"):
                            content[k] = v["content"]            # D-38
                        if not v.get("usable", True):
                            unusable.add(k)
                            continue
                        if v.get("anchors"):                     # a consensus-only language has no QE anchors
                            cal[k] = tuple(v["anchors"])
                        rel[k] = float(v.get("reliability", 1.0))
                        if reliability_from == "same_source" and "reliability_same_source" in v:
                            rel[k] = float(v["reliability_same_source"])
                        lr = v.get("len_ratio")
                        if lr and lr.get("p05") and lr.get("p95"):
                            # clamp: a few over-long references (zul p95 = 8.1) would otherwise make the window useless
                            win[k] = (max(float(lr["p05"]), 0.3), min(float(lr["p95"]), 2.5))
                        if isinstance(v.get("word_ratio"), dict) and v["word_ratio"].get("p99"):
                            wr[k] = {k2: float(v2) for k2, v2 in v["word_ratio"].items() if isinstance(v2, (int, float))}
                        if v.get("lexical"):
                            lex[k] = {f: float(d["p10"]) for f, d in v["lexical"].items() if "p10" in d}
                        if v.get("diacritics"):
                            dia[k] = v["diacritics"]
                        if lr and lr.get("corpus"):
                            lst[k] = {"corpus": float(lr["corpus"]), "s2": float(lr.get("s2") or 6.8),
                                      "p05": float(lr.get("p05") or 0.0), "p95": float(lr.get("p95") or 0.0)}
                            if (v.get("omission") or {}).get("cost"):
                                lst[k]["omit_cost"] = float(v["omission"]["cost"])         # per-language operating point (omission_eval --calibrate-omission)
                    else:
                        cal[k] = tuple(v)
        sh = cls(calibration=cal, **kw)
        sh.rel, sh.unusable, sh.len_window, sh.lexical = rel, unusable, win, lex
        sh.diacritics, sh.len_stats = dia, lst
        sh.word_ratio = wr
        sh.content, sh.lid = content, lid
        return sh

    # -- queries
    def is_usable(self, lang: str) -> bool:
        return lang not in self.unusable

    def reliability(self, lang: str) -> float:
        """Advantage weight in [0, 1]: how much to trust QE for this language (from its measured AUC)."""
        return 0.0 if lang in self.unusable else float(self.rel.get(lang, 1.0))

    def length_excess(self, lang: str, src_chars: int, mt_chars: int, slack: float = 0.15) -> float:
        """log-units by which len(mt)/len(src) lies outside the language's HUMAN window [p05, p95] (widened by `slack`); 0 inside / unknown."""
        w = self.len_window.get(lang) or self.len_window.get("*")
        if not w or src_chars <= 0 or mt_chars <= 0:
            return 0.0
        r = mt_chars / src_chars
        lo, hi = w[0] * (1.0 - slack), w[1] * (1.0 + slack)
        return float(max(0.0, np.log(lo / r), np.log(r / hi)))

    def word_excess(self, lang: str, src: str, mt: str, slack: float = 0.15, min_src_words: int = 4) -> float:
        """log-units by which words(mt) / words(src) lies ABOVE the language's human 99th percentile (widened by `slack`); 0 below / unknown (2026-10-21).
        One-sided on purpose: too short is the char-length window's and the omission detector's job. Words = whitespace-separated, the unit differs a lot
        between orthographies (Zulu 0.75 words per English word, Setswana 1.29, Fon 1.37 at the median), hence a measured limit per language, never a global one."""
        w = self.word_ratio.get(lang) or self.word_ratio.get("*")
        if not w or not w.get("p99"):
            return 0.0
        ns, nm = len(src.split()), len(mt.split())
        if ns < min_src_words or nm <= 0:
            return 0.0
        hi = float(w["p99"]) * (1.0 + slack)
        return float(max(0.0, np.log((nm / ns) / hi)))

    def lexical_deficit(self, lang: str, src: str, mt: str) -> float:
        """0..1: how far the output falls short of the lexical overlap human references of this language show (rl/lexical.py); 0 when unknown."""
        th = self.lexical.get(lang)
        if not th:
            return 0.0
        from gemma_mt.rl import lexical

        return lexical.deficit(lexical.features(src, mt), th)

    def diacritic_deficit(self, lang: str, src: str, mt: str) -> float:
        """0..1: stripped / fabricated diacritics of the chunk against the language's human references (rl/diacritics.py); 0 for a language
        without diacritics, without calibration, or when the English chunk has < 3 content words."""
        e = self.diacritics.get(lang)
        if not e or not e.get("applies"):
            return 0.0
        from gemma_mt.rl import diacritics

        if not diacritics.applicable(src):
            return 0.0
        return diacritics.deficit(mt, e)

    def length_model(self, lang: str, omit_cost: Optional[float] = None, extra_cost: Optional[float] = None):
        """segments.LengthModel of the language (human corpus ratio, variance per source character) or None when it has no calibration.
        `omit_cost` None = the language's calibrated omission cost (calibration block `omission`), else 4.0; `extra_cost` None = 4.0 (the calibration
        only tunes the omission side: an omission bead needs fewer output than source sentences, an extra bead more, so the two never compete)."""
        st = self.len_stats.get(lang)
        if not st:
            return None
        from gemma_mt.rl.segments import LengthModel

        return LengthModel(ratio=st["corpus"], s2=st["s2"], omit_cost=st.get("omit_cost", 4.0) if omit_cost is None else omit_cost,
                           extra_cost=4.0 if extra_cost is None else extra_cost)

    def anchors(self, lang: str) -> Tuple[float, float, float]:
        base = tuple(self.cal.get(lang) or self.cal.get("*") or self.default)
        if self.mode == "adaptive" and lang in self._live:
            mid, great = self._live[lang]
            return (max(base[0], self.fail_floor), max(mid, base[0] + 1e-3), max(great, mid + 1e-3))
        return (max(base[0], self.fail_floor), base[1], base[2])

    def observe(self, lang: str, scores: Iterable[float]) -> None:
        """adaptive only: feed the raw QE scores of gate-open rows."""
        scores = list(scores)
        if self.mode != "adaptive" or not scores:
            return
        w = self._win[lang]
        w.extend(scores)
        if len(w) >= self.warmup:
            mid, great = (float(x) for x in np.quantile(np.fromiter(w, float), [self.p_mid, self.p_great]))
            old = self._live.get(lang)
            self._live[lang] = (mid, great) if old is None else (self.ema * old[0] + (1 - self.ema) * mid,
                                                                  self.ema * old[1] + (1 - self.ema) * great)

    def __call__(self, c: float, lang: str) -> float:
        xs, ys = make_curve(self.anchors(lang), self.neg, self.preset, self.fail_floor)
        if c <= xs[0]:
            return ys[0]
        if c >= xs[-1]:
            return ys[-1]
        return float(np.interp(c, xs, ys))

    @property
    def reward_range(self) -> Tuple[float, float]:
        return preset_range(self.preset, self.neg)

    def state(self) -> dict:
        return {k: {"t_mid": v[0], "t_great": v[1], "n": len(self._win[k])} for k, v in self._live.items()}


# ------------------------------------------------------------------------------------------- calibration helpers
def compute_anchors(scores: Sequence[float], q=(0.10, 0.50, 0.90)) -> Tuple[float, float, float]:
    a = np.quantile(np.asarray(scores, dtype=float), q)
    return (float(a[0]), float(a[1]), float(a[2]))


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score(pos) > score(neg)): how well the metric separates real translations from mismatched ones (0.5 =
    chance). Rank-sum (Mann-Whitney) form with average ranks for ties."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    both = np.concatenate([p, n])
    order = np.argsort(both, kind="mergesort")
    ranks = np.empty(len(both))
    ranks[order] = np.arange(1, len(both) + 1)
    # average the ranks of ties
    sorted_vals = both[order]
    i = 0
    while i < len(both):
        j = i
        while j + 1 < len(both) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i: j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def reliability_from_auc(a: float, low: float = 0.65, high: float = 0.90) -> float:
    """Linear trust in the metric for a language: AUC <= 0.65 -> 0, AUC >= 0.90 -> 1."""
    return float(min(1.0, max(0.0, (a - low) / (high - low))))


def pair_anchors(human: Sequence[float], mismatched: Sequence[float], floor: float = FAIL_FLOOR):
    """t_fail = max(floor, P90 of QE on mismatched-but-fluent translations); t_mid = P50 and t_great = P90 of QE on
    human references (kept a margin above the previous anchor)."""
    m90 = float(np.quantile(np.asarray(mismatched, dtype=float), 0.90)) if len(mismatched) else floor
    t_fail = max(floor, m90)
    h = np.quantile(np.asarray(human, dtype=float), [0.5, 0.9])
    t_mid = max(float(h[0]), t_fail + 0.05)
    t_great = max(float(h[1]), t_mid + 0.05)
    return (t_fail, t_mid, t_great)
