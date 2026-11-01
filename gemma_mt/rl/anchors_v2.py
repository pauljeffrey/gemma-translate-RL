# SPDX-License-Identifier: AGPL-3.0-or-later
"""Anchors v2 (D-61, owner 2026-10-21/22): graded entity preservation for more languages, substring COPY detection, the stop-word profile and the padded / stuffed-output penalty.

Extends rl/anchors.py (D-56, which stays untouched: v1 detectors and calibration keep their meaning). `Anchors2` is an `Anchors` with a second set of calibration blocks (calibration/anchors_v2.json =
anchors_v1.json + the blocks below, written by scripts/build_anchors_v2_calibration.py) and four more detectors; everything is flag-gated in `RewardConfig` (default OFF), PENALTIES only.

    entity2   chunk level. The v1 entity detector punishes only below the human 3 % quantile of the recall and is therefore off wherever humans drop whole channels (names: 15-50 %). v2 punishes the
              share of the source's numbers / names that the output lacks (deficit = 1 - recall), with a weight that shrinks with the language's HUMAN DROP RATE d (share of the judged human chunks
              that lost every entity of the channel): w = base x (1 - d), channel off when d > `max_drop` (0.6). Numbers base 1, names base `name_scale` (0.5) (the capitalised-word heuristic is noisier).
    copy      chunk level. Share of the English source's UNPROTECTED words covered by substrings that occur verbatim in the output (word runs >= k, character runs >= m; numbers, URLs, names,
              acronyms, name connectors and the language's learned loanwords are protected). Above the language's human threshold t (99.5 % quantile) the chunk's reward is capped at
              -scale x fraction (owner: "-2 x the percentage copied"; fraction 1 -> -2). Jaccard `is_copy_of_source` (reward.copy_jaccard) stays the hard gate; its items never reach this detector.
    func      sample level. Stop-word profile: share of target function words (dictionary = most frequent words of the language's training text), share of ENGLISH function words that the language
              does not use, and target function words per English function word of the source; thresholds are low / high human quantiles per length bucket.
    pad       sample level. Padded / stuffed output: share of the output's word bigrams that the language's training text attests (a stuffed bag of words has almost none) and the type / token ratio,
              thresholds from the human low quantile per length bucket.
"""

from __future__ import annotations

import json
import os
import threading
import zlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from gemma_mt.rl import textstats as T
from gemma_mt.rl.anchors import Anchors, _clip01, entity_counts

DETECTORS_V2 = ("entity2", "copy", "func", "pad")
V2_LENGTH_EDGES = (7, 15, 30, 60)                   # word-count bucket edges of the func / pad thresholds (5 buckets); outputs below FUNC_MIN_WORDS are not judged
FUNC_MIN_WORDS = 8
FUNC_MIN_SRC_WORDS = 5
ENTITY2_MAX_DROP = 0.6
ENTITY2_NAME_SCALE = 0.5
COPY_SCALE = 2.0
BIGRAM_LAST = 2


def v2_bucket(n_words: int) -> int:
    for i, e in enumerate(V2_LENGTH_EDGES):
        if n_words <= e:
            return i
    return len(V2_LENGTH_EDGES)


def bigram_hashes(w: Sequence[str]) -> np.ndarray:
    """Stable (cross-process) uint32 hashes of the word bigrams of `w` (crc32 of 'w1\\x00w2')."""
    return np.fromiter((zlib.crc32((a + "\x00" + b).encode("utf-8")) for a, b in zip(w, w[1:])), dtype=np.uint32, count=max(len(w) - 1, 0))


BIGRAM_SHUFFLES = 4


def attested_share(tab: np.ndarray, w: Sequence[str]) -> float:
    h = bigram_hashes(w)
    if h.size == 0:
        return float("nan")
    idx = np.searchsorted(tab, h)
    idx[idx >= tab.size] = tab.size - 1
    return float((tab[idx] == h).mean())


def bigram_gain(tab: np.ndarray, w: Sequence[str]) -> Tuple[float, float]:
    """(attested-bigram share of the output, share of its word-shuffled copies): the second number is what the SAME words would score in a random order, so the difference (the "gain") measures word
    ORDER and is not affected by how well the training domain matches the text. The shuffles are seeded by the text (deterministic, resume-safe)."""
    import random

    obs = attested_share(tab, w)
    rng = random.Random(zlib.crc32(" ".join(w).encode("utf-8")))
    sh = []
    ww = list(w)
    for _ in range(BIGRAM_SHUFFLES):
        rng.shuffle(ww)
        sh.append(attested_share(tab, ww))
    return obs, float(np.mean(sh))


def pad_ttr(words: Sequence[str], mode: str = "plain") -> float:
    """Type / token ratio of an output. mode 'plain': distinct / n (v2). mode 'seg100' (anchors_v3.json, D-68): the type / token ratio of a text falls with its length (human documents of 100 / 200 /
    400 / 800 words: 0.66 / 0.51 / 0.43 / 0.32 in the same language), so a threshold fitted on windows of <= 250 words flagged long documents. 'seg100' = the mean over `n // 100` equal segments
    (each 100-199 words) for texts of >= 100 words, the plain ratio below; length independent."""
    n = len(words)
    if n == 0:
        return 1.0
    if mode != "seg100" or n < 100:
        return len(set(words)) / n
    k = n // 100
    vals = []
    for j in range(k):
        seg = words[j * n // k: (j + 1) * n // k]
        vals.append(len(set(seg)) / len(seg))
    return float(sum(vals) / len(vals))


def _ramp(x: float, lo: float, hi: float, clip: bool = True) -> float:
    """0 at lo, 1 at hi (linear, either direction)."""
    if hi == lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return _clip01(v) if clip else v


def func_deficit_from(blk: Dict[str, Any], n: int, fw: float, en: float, ratio: float, clip: bool = True) -> float:
    """The stop-word-profile deficit of one output (0..1; unclipped raw score when clip=False, for AUC): max of
      (a) target function-word share below the human low quantile of the length bucket (0 at the threshold, 1 at a quarter of it),
      (b) English function-word share above the human high quantile (1 at `cap_en`),
      (c) target function words per English function word of the source below the human low quantile (1 at a quarter of it),
      (d) target function-word share ABOVE the human high quantile (a bag of function words; 1 at `cap_fw`)."""
    k = v2_bucket(n)
    d = -9.0 if not clip else 0.0
    t = (blk.get("t_fw") or [None] * 5)[k]
    if t and t > 0:
        d = max(d, _ramp(fw, t, 0.25 * t, clip))
    t = (blk.get("t_en") or [None] * 5)[k]
    if t is not None and float(blk.get("cap_en", 0.3)) > t:
        d = max(d, _ramp(en, t, float(blk.get("cap_en", 0.3)), clip))
    t = (blk.get("t_ratio") or [None] * 5)[k]
    if t and t > 0:
        d = max(d, _ramp(ratio, t, 0.25 * t, clip))
    t = (blk.get("t_fw_hi") or [None] * 5)[k]
    if t is not None and float(blk.get("cap_fw", 0.95)) > t:
        d = max(d, _ramp(fw, t, float(blk.get("cap_fw", 0.95)), clip))
    return d


def pad_deficit_from(blk: Dict[str, Any], n: int, gain: Optional[float], ttr: float, clip: bool = True) -> float:
    """The padding deficit (0..1): max of (a) bigram-order gain (attested share of the output minus that of its shuffled copy) below the human low quantile of the length bucket (1 at a quarter of it),
    (b) type / token ratio below the human low quantile (1 at half of it)."""
    k = v2_bucket(n)
    d = -9.0 if not clip else 0.0
    t = (blk.get("t_gn") or [None] * 5)[k]
    if t and t > 0 and gain is not None and gain == gain:
        d = max(d, _ramp(gain, t, 0.25 * t, clip))
    t = (blk.get("t_ttr") or [None] * 5)[k]
    if t and t > 0:
        d = max(d, _ramp(ttr, t, 0.5 * t, clip))
    return d


# ------------------------------------------------------------------------------------------------ entity v2
def entity2_penalty(blk: Optional[Dict[str, Any]], src: str, mt: str) -> Tuple[float, int]:
    """(penalty units 0..1, number of source entities judged): count-weighted mean over the enabled channels the chunk is long enough for of w_c x (1 - recall_c), where w_c = (1 - d_c) x
    scale_c (the calibration stores w). The reward multiplies the result by `anchors_entity_weight`. (0.0, 0) when nothing is judged."""
    if not blk:
        return 0.0, 0
    ec = entity_counts(src, mt)
    tot = wn = 0.0
    for ch, n, r in (("num", ec.n_num, ec.recall_num), ("name", ec.n_name, ec.recall_name)):
        c = blk.get(ch) or {}
        if not c.get("enabled") or r is None or n < int(c.get("min_n", 1)):
            continue
        tot += n * float(c["w"]) * (1.0 - r)
        wn += n
    return (tot / wn if wn else 0.0), int(wn)


# ------------------------------------------------------------------------------------------------ the object used by the reward
class _LRU(OrderedDict):
    """Small thread-safe LRU of source tokenisations (a pure cache: results never depend on its content)."""

    def __init__(self, cap: int):
        super().__init__()
        self.cap = cap
        self._lock = threading.Lock()

    def get_or(self, key, fn):
        with self._lock:
            if key in self:
                self.move_to_end(key)
                return self[key]
        v = fn()
        with self._lock:
            self[key] = v
            if len(self) > self.cap:
                self.popitem(last=False)
        return v


class Anchors2(Anchors):
    """`Anchors` + the v2 blocks of `calibration/anchors_v2.json` (`languages[iso]["entity2" | "copy" | "func" | "pad"]`, `pooled[...]`).

    `v2_modes`: content modes the v2 detectors apply to (default all three). `v2_off_languages`: extra CONTROL languages for the v2 detectors only (`off_languages` of the base class is honoured too).
    `bigram_path`: npz with `<iso>` -> sorted uint32 bigram hashes (calibration/anchors_v2_bigrams.npz); without it `pad` judges the type / token ratio only."""

    def __init__(self, calibration: Optional[Dict[str, Any]] = None, *, v2_detectors: Sequence[str] = DETECTORS_V2, v2_modes: Sequence[str] = ("qe", "blend", "consensus"),
                 v2_off_languages: Sequence[str] = (), copy_scale: float = COPY_SCALE, copy_mode: str = "cap", bigram_path: Optional[str] = None, **kw: Any):
        super().__init__(calibration, **kw)
        self.v2_detectors = frozenset(v2_detectors)
        self.v2_modes = tuple(v2_modes)
        self.v2_off = frozenset(v2_off_languages)
        self.copy_scale, self.copy_mode = float(copy_scale), copy_mode
        if copy_mode not in ("cap", "subtract"):
            raise ValueError("copy_mode must be 'cap' or 'subtract'")
        self.bigram_path = bigram_path
        self._bigrams: Dict[str, np.ndarray] = {}
        self._bg_file = None
        self._src_cache = _LRU(4096)
        self._fw_cache: Dict[str, Tuple[FrozenSet[str], FrozenSet[str]]] = {}
        self._wl_cache: Dict[str, FrozenSet[str]] = {}

    # -- which languages / detectors
    def applies2(self, lang: str, mode: str) -> bool:
        return mode in self.v2_modes and lang not in self.v2_off and lang not in self.off

    def _blk2(self, lang: str, name: str) -> Optional[Dict[str, Any]]:
        """Like `_blk`, for the v2 blocks (the same proxy / pooled-prior rule); a block with `enabled: false` or (anchors_v3.json) `usable: false` is None."""
        b = self._blk(lang, name)
        return None if b is not None and b.get("usable") is False else b

    def gate_ok(self, lang: str, det: str) -> bool:
        """May a saturated `det` deficit of this language fail the sample (anchors_stuff_gate)? Blocks of anchors_v3.json carry `gate_ok: false` for languages calibrated on a single domain."""
        b = self._blk2(lang, det)
        return bool(b) and b.get("gate_ok", True) is not False

    def enabled2(self, lang: str, det: str) -> bool:
        if det not in self.v2_detectors:
            return False
        b = self._blk2(lang, det)
        if not b:
            return False
        if det == "entity2":
            return any((b.get(c) or {}).get("enabled") for c in ("num", "name"))
        return True

    # -- entity v2
    def entity2_penalty(self, lang: str, src: str, mt: str) -> Tuple[float, int]:
        return entity2_penalty(self._blk2(lang, "entity2"), src, mt)

    # -- copy
    def _whitelist(self, lang: str, b: Dict[str, Any]) -> FrozenSet[str]:
        wl = self._wl_cache.get(lang)
        if wl is None:
            wl = self._wl_cache[lang] = frozenset(b.get("whitelist") or ())
        return wl

    def copy_fraction(self, lang: str, src: str, mt: str) -> Optional[float]:
        """The copied fraction of the chunk if it exceeds the language's human threshold, 0.0 if not, None when the detector is off for the language or the source has too few unprotected words.
        The reward then caps the chunk reward at -copy_scale x fraction."""
        b = self._blk2(lang, "copy")
        if not b:
            return None
        wl = self._whitelist(lang, b)
        st = self._src_cache.get_or((src, id(wl)), lambda: T.source_tokens(src, wl))
        k, m = int(b.get("k", 3)), int(b.get("m", 0))
        r = T.copy_fractions(st, T.out_tokens(mt), k=k, m=m, min_unprot=int(b.get("min_unprot", 2)))
        if r.frac_w is None and r.frac_c is None:
            return None
        out = 0.0
        tw, tc = float(b.get("t_w", 1.0)), float(b.get("t_c", 1.0))
        if r.frac_w is not None and r.frac_w > tw:
            out = max(out, r.frac_w)
        if r.frac_c is not None and r.frac_c > tc:
            out = max(out, r.frac_c)
        return out

    def copy_apply(self, reward: float, frac: float, floor: float) -> float:
        """Chunk reward after the copy rule: 'cap' (owner): min(reward, -scale x frac); 'subtract': reward - scale x frac; never below `floor` (the failure penalty)."""
        pen = self.copy_scale * frac
        r = min(reward, -pen) if self.copy_mode == "cap" else reward - pen
        return max(floor, r)

    # -- function-word profile + padding, sample level
    def _fw_sets(self, lang: str, b: Dict[str, Any]) -> Tuple[FrozenSet[str], FrozenSet[str]]:
        s = self._fw_cache.get(lang)
        if s is None:
            s = self._fw_cache[lang] = (frozenset(b.get("fw") or ()), frozenset(b.get("en_excl") or ()))
        return s

    def func_features(self, lang: str, src_words: Sequence[str], out_words: Sequence[str]) -> Optional[Dict[str, float]]:
        b = self._blk2(lang, "func")
        if not b or len(out_words) < FUNC_MIN_WORDS or len(src_words) < FUNC_MIN_SRC_WORDS:
            return None
        fw, ex = self._fw_sets(lang, b)
        f = T.func_features(out_words, fw, ex)
        es = sum(1 for w in src_words if w in T.EN_FUNC)
        f["ratio"] = f["nfw"] / (es + 1.0)
        return f

    def func_deficit(self, lang: str, src_words: Sequence[str], out_words: Sequence[str]) -> Optional[float]:
        """0..1 (see `func_deficit_from`); None when off / not judged."""
        b = self._blk2(lang, "func")
        f = self.func_features(lang, src_words, out_words)
        if f is None:
            return None
        return func_deficit_from(b, int(f["n"]), f["fw"], f["en"], f["ratio"])

    def _bigram_table(self, lang: str) -> Optional[np.ndarray]:
        if lang in self._bigrams:
            return self._bigrams[lang]
        tab = None
        if self.bigram_path and os.path.exists(self.bigram_path):
            if self._bg_file is None:
                self._bg_file = np.load(self.bigram_path)
            if lang in self._bg_file.files:
                tab = self._bg_file[lang]
        self._bigrams[lang] = tab
        return tab

    def pad_features(self, lang: str, out_words: Sequence[str]) -> Optional[Dict[str, float]]:
        b = self._blk2(lang, "pad")
        if not b or len(out_words) < FUNC_MIN_WORDS:
            return None
        f = {"n": len(out_words), "ttr": pad_ttr(out_words, b.get("ttr_mode", "plain"))}
        tab = self._bigram_table(lang) if b.get("has_bigrams") else None
        if tab is not None and tab.size and len(out_words) > 1:
            obs, shuf = bigram_gain(tab, out_words)
            f["bg"], f["gain"] = obs, obs - shuf
        return f

    def pad_deficit(self, lang: str, out_words: Sequence[str]) -> Optional[float]:
        """0..1 (see `pad_deficit_from`); None when off / not judged."""
        b = self._blk2(lang, "pad")
        f = self.pad_features(lang, out_words)
        if f is None:
            return None
        return pad_deficit_from(b, int(f["n"]), f.get("gain"), f["ttr"])
