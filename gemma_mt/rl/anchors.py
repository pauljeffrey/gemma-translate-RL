# SPDX-License-Identifier: AGPL-3.0-or-later
"""Constraint penalties for the languages whose reward has no (or a weak) AfriCOMET signal (D-56, owner 2026-10-21).

For the 23 consensus-only and the 22 blend languages the content signal is the agreement of the group's samples (rl/consensus.py) with, at best, half a QE score. Agreement
alone rewards fluent text that is the same whatever the source says. The RL-vs-SFT check (reports/rl_vs_sft_weak_signal_2026-10-21) measured exactly that drift on top of a real
gain (+5 chrF++ over SFT): the 5-gram boilerplate rate of the outputs rose from 25 to 32 % (human text: 3 %) and the output-vs-output-of-another-source similarity from 13 to 19 chrF++.
This module adds PENALTIES only (never a bonus: the human references keep only 44.5 % of the source's numbers and names, the model 73 %, so a recall REWARD would push towards
copying), each with its own weight, each switched on per language only where its false-positive rate on HUMAN references is small (calibration/anchors_v1.json, built by
scripts/build_anchors_calibration.py):

    entity_num / entity_name   (a) chunk level. Share of the source chunk's numbers (digit strings) and capitalised names (not sentence-initial, >= 3 letters, diacritic-folded,
                               1-character tolerance for names >= 5 letters) that are missing from the translation. Penalised only below the language's human low quantile of that recall
                               (alpha 3 % by default: the human 10th percentile is 0 at sentence level, "all names dropped" is normal for a transliterating orthography; the
                               quantile is taken over the human chunks that have the channel's minimum number of entities), linearly to 0. Names and numbers are separate channels
                               because their retention differs a lot (numbers 86 %, names 52 % pooled); each is enabled per language only when its human threshold is > 0.
    repeat                     (b) sample level. Share of the output's word 3-, 4- and 5-gram positions that repeat an earlier n-gram of the same output (the quantity of
                               guards.repeated_ngram_fraction), against the human high quantile for texts of that LENGTH in that language (length buckets: repetition grows with
                               length), never below the source's own repetition. Linear from the threshold to `cap` (the level at which loops are certain; the hard near-loop
                               guard of the consensus languages fires at 4-gram 0.35).
    boiler                     (c) sample level, needs the rolling buffer. Share of the output's word 5-grams (those not copied from the source) that occur in the outputs of >= 3 OTHER
                               prompts of the same language among the last ~600 outputs. Prompts whose sources share an English word 6-gram with the current one are not "other"
                               (the same boilerplate sentence in two papers is translated the same way, legitimately). No penalty until the buffer holds >= 150 outputs from >= 30 prompts.
    lexicon (hook)             (d) `lexicon_overlap(lang, src, mt)`: share of the source's content words that a bilingual lexicon covers whose translation is in the output
                               (rl/lexicon.py of R3, `lexicon/<iso>.json`, only for languages whose lexicon validated). Penalty below the language's human low quantile (the calibration
                               has to have measured it: build_anchors_calibration.py --lexicon-dir); weight 0 and detector off by default.
    lid_prob (hook)            (e) minimum GlotLID probability per language (calibration `lid_prob.pmin`, measured as the p05 of the probability on human references whose label is
                               accepted); soft penalty, weight 0 by default.

Everything here is pure python + numpy; the buffer is in memory and starts empty in a resumed run (it warms up in ~30 steps). One `Anchors` instance is shared by the reward and the
evaluation calls; the buffer only grows through `observe` (the trainer's sampling calls), never through the greedy evaluation.
"""

from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

DETECTORS = ("entity", "repeat", "boiler", "lexicon", "lid_prob")
DEFAULT_MODES = ("consensus", "blend")
LENGTH_EDGES = (25, 60, 150, 400, 1000)             # word-count bucket edges of the repetition thresholds (6 buckets)
REPEAT_NS = (3, 4, 5)
REPEAT_MIN_WORDS = 12                                # shorter outputs are not judged for repetition
ENTITY_MIN_NUM = 1                                   # numbers needed in a chunk for the number channel to be judged
ENTITY_MIN_NAME = 3                                  # names needed for the name channel (a single dropped name is normal)
BOILER_N = 5
BOILER_RELATED_N = 6                                 # English word n-gram shared by two sources = "related prompts" (>= BOILER_RELATED_MIN of them)
BOILER_RELATED_MIN = 1
BOILER_PER_PROMPT = 4                                # outputs of one prompt kept in the buffer (a GRPO group has 16: the buffer must hold many prompts, not many copies of few)


# ------------------------------------------------------------------------------------------------ text helpers
def fold(s: str) -> str:
    """NFKD, combining marks removed, casefolded: diacritics and case never matter for anchors."""
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c)).casefold()


_W = re.compile(r"\w+", re.UNICODE)


def words(s: str) -> List[str]:
    return _W.findall(fold(s))


# ------------------------------------------------------------------------------------------------ (a) entities
_WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_SENT_END = ".!?:;\n…"


def numkey(s: str) -> str:
    return re.sub(r"[.,]", "", s)


def extract_entities(src: str) -> Tuple[List[str], List[str]]:
    """(numbers, names) of a source chunk. Numbers: digit strings without their separators. Names: capitalised words of >= 3 letters that do not start a sentence, folded.
    (The definition of the RL-vs-SFT check, so thresholds and the measured 44.5 % / 73 % retention refer to the same thing; unlike rl/lexical.anchors it keeps accents
    and all-caps acronyms and matches with a tolerance.)"""
    nums = [numkey(m.group()) for m in _NUM.finditer(src)]
    names: List[str] = []
    for m in _WORD.finditer(src):
        w = m.group()
        if len(w) < 3 or not w[0].isupper():
            continue
        head = src[: m.start()]
        pre = head.rstrip(" \t\"'“‘([{«")
        if not pre or pre[-1] in _SENT_END or head.rstrip(" \t").endswith("\n"):
            continue
        names.append(fold(w))
    return nums, names


def _edit_le1(a: str, b: str) -> bool:
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) <= 1
    if la > lb:
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def _name_in(name: str, ftxt: str, toks: Sequence[str]) -> bool:
    if name in ftxt:
        return True
    L = len(name)
    if L < 5:
        return False
    h = L // 2                                          # one edit lies in at most one half: the other half must occur verbatim (cheap prefilter)
    if name[:h] not in ftxt and name[h:] not in ftxt:
        return False
    for t in toks:
        if len(t) < L - 1:
            continue
        for w in (L - 1, L, L + 1):
            for s in range(0, len(t) - w + 1):
                if _edit_le1(name, t[s: s + w]):
                    return True
    return False


@dataclass
class EntityCounts:
    n_num: int
    hit_num: int
    n_name: int
    hit_name: int

    @property
    def recall_num(self) -> Optional[float]:
        return self.hit_num / self.n_num if self.n_num else None

    @property
    def recall_name(self) -> Optional[float]:
        return self.hit_name / self.n_name if self.n_name else None


def entity_counts(src: str, mt: str) -> EntityCounts:
    nums, names = extract_entities(src)
    if not nums and not names:
        return EntityCounts(0, 0, 0, 0)
    ft = fold(mt)
    toks = _W.findall(ft)
    tn = {numkey(m.group()) for m in _NUM.finditer(mt)}
    return EntityCounts(len(nums), sum(1 for n in nums if n in tn), len(names), sum(1 for n in names if _name_in(n, ft, toks)))


# ------------------------------------------------------------------------------------------------ (b) repetition
def repeat_shares(w: Sequence[str], ns: Sequence[int] = REPEAT_NS) -> Dict[int, float]:
    """{n: share of the word n-gram positions that repeat an earlier n-gram of the same text}; 0.0 when the text has fewer than n+1 words."""
    out: Dict[int, float] = {}
    for n in ns:
        m = len(w) - n + 1
        if m < 2:
            out[n] = 0.0
            continue
        out[n] = 1.0 - len({tuple(w[i: i + n]) for i in range(m)}) / m
    return out


def length_bucket(n_words: int) -> int:
    for i, e in enumerate(LENGTH_EDGES):
        if n_words <= e:
            return i
    return len(LENGTH_EDGES)


# ------------------------------------------------------------------------------------------------ (c) cross-prompt boilerplate
def gram_hashes(w: Sequence[str], n: int, skip: Optional[Sequence[bool]] = None) -> np.ndarray:
    """Unique int64 hashes of the word n-grams of `w`; an n-gram whose words all have skip[j] True (copied from the source) is left out."""
    m = len(w) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.int64)
    if skip is None:
        hs = [hash(tuple(w[i: i + n])) for i in range(m)]
    else:
        pre = np.concatenate([[0], np.cumsum(np.asarray(skip, dtype=np.int64))])
        hs = [hash(tuple(w[i: i + n])) for i in range(m) if pre[i + n] - pre[i] < n]
    return np.unique(np.asarray(hs, dtype=np.int64)) if hs else np.zeros(0, dtype=np.int64)


class _LangBuffer:
    def __init__(self) -> None:
        self.entries: Deque[Tuple[int, np.ndarray]] = deque()      # (prompt key, unique n-gram hashes of one output)
        self.total = 0
        self.prompt_count: Counter = Counter()
        self.prompt_src: Dict[int, np.ndarray] = {}                 # prompt key -> unique English 6-gram hashes of its source
        self.version = 0
        self._table: Optional[Tuple[int, np.ndarray, np.ndarray]] = None
        self._related: Dict[int, Tuple[int, Set[int]]] = {}

    # -- table: for every n-gram hash the number of DISTINCT prompts whose outputs contain it
    def table(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._table is not None and self._table[0] == self.version:
            return self._table[1], self._table[2]
        if not self.entries:
            t = (self.version, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        else:
            pid = {k: i for i, k in enumerate(self.prompt_count)}
            g = np.concatenate([e[1] for e in self.entries])
            p = np.concatenate([np.full(len(e[1]), pid[e[0]], dtype=np.int64) for e in self.entries])
            order = np.lexsort((p, g))
            g, p = g[order], p[order]
            new = np.ones(len(g), dtype=bool)
            new[1:] = (g[1:] != g[:-1]) | (p[1:] != p[:-1])
            ug, cnt = np.unique(g[new], return_counts=True)
            t = (self.version, ug, cnt)
        self._table = t
        return t[1], t[2]

    def related(self, key: int, src_hashes: np.ndarray, min_shared: int = 1) -> Set[int]:
        hit = self._related.get(key)
        if hit is not None and hit[0] == self.version:
            return hit[1]
        rel = {k for k, s in self.prompt_src.items() if k != key and src_hashes.size and s.size and np.intersect1d(s, src_hashes, assume_unique=True).size >= min_shared}
        self._related[key] = (self.version, rel)
        return rel

    def prompt_grams(self, keys: Set[int]) -> np.ndarray:
        parts = [e[1] for e in self.entries if e[0] in keys]
        return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)

    def add(self, key: int, grams: np.ndarray, src_hashes: np.ndarray, capacity: int, max_grams: int, per_prompt: int = 10 ** 9) -> None:
        if self.prompt_count.get(key, 0) >= per_prompt:
            return
        self.entries.append((key, grams))
        self.total += len(grams)
        self.prompt_count[key] += 1
        self.prompt_src.setdefault(key, src_hashes)
        while len(self.entries) > capacity or (self.total > max_grams and len(self.entries) > 1):
            k, g = self.entries.popleft()
            self.total -= len(g)
            self.prompt_count[k] -= 1
            if self.prompt_count[k] <= 0:
                del self.prompt_count[k]
                self.prompt_src.pop(k, None)
        self.version += 1


class BoilerplateBuffer:
    """Rolling per-language buffer of the last `capacity` outputs (and at most `max_grams` distinct-per-output n-gram hashes). Thread-safe."""

    def __init__(self, capacity: int = 600, min_outputs: int = 150, min_prompts: int = 30, min_other: int = 3, max_grams: int = 250_000, per_prompt: int = BOILER_PER_PROMPT,
                 related_min: int = BOILER_RELATED_MIN):
        self.capacity, self.min_outputs, self.min_prompts, self.min_other, self.max_grams = int(capacity), int(min_outputs), int(min_prompts), int(min_other), int(max_grams)
        self.per_prompt, self.related_min = int(per_prompt), int(related_min)
        self._langs: Dict[str, _LangBuffer] = {}
        self._lock = threading.RLock()

    def reset(self) -> None:
        with self._lock:
            self._langs.clear()

    def size(self, lang: str) -> Tuple[int, int]:
        """(outputs, distinct prompts) in the buffer of `lang`."""
        with self._lock:
            b = self._langs.get(lang)
            return (len(b.entries), len(b.prompt_count)) if b else (0, 0)

    def ready(self, lang: str) -> bool:
        n, p = self.size(lang)
        return n >= self.min_outputs and p >= self.min_prompts

    def score(self, lang: str, key: int, grams: np.ndarray, src_hashes: np.ndarray) -> Optional[float]:
        """Share of `grams` found in the outputs of >= min_other prompts other than `key` / its related prompts; None while the buffer is not ready or `grams` is empty."""
        if grams.size == 0:
            return None
        with self._lock:
            b = self._langs.get(lang)
            if b is None or len(b.entries) < self.min_outputs or len(b.prompt_count) < self.min_prompts:
                return None
            ug, cnt = b.table()
            if ug.size == 0:
                return 0.0
            idx = np.searchsorted(ug, grams)
            idx[idx >= ug.size] = ug.size - 1
            c = np.where(ug[idx] == grams, cnt[idx], 0).astype(np.int64)
            excl = {key} | b.related(key, src_hashes, self.related_min)
            for k in excl:                                       # each excluded prompt that contains the gram is not "another" prompt
                if k in b.prompt_count:
                    c -= np.isin(grams, b.prompt_grams({k})).astype(np.int64)
            return float((c >= self.min_other).sum()) / grams.size

    def add(self, lang: str, key: int, grams: np.ndarray, src_hashes: np.ndarray) -> None:
        if grams.size == 0:
            return
        with self._lock:
            self._langs.setdefault(lang, _LangBuffer()).add(key, grams, src_hashes, self.capacity, self.max_grams, self.per_prompt)


# ------------------------------------------------------------------------------------------------ (d) lexicon hook
def lexicon_overlap(lang: str, src: str, mt: str, lexicon_dir: Optional[str] = None, guard: bool = True) -> Optional[float]:
    """The hook: share (0..1, IDF-weighted) of the English source's lexicon-covered content words with a candidate in `mt` (rl/lexicon.py, R3); None when the language has no
    validated lexicon or the chunk has too few covered words. `guard` (default ON, D-59): the overlap is scaled down by R3's stuffing guard (output density of lexicon candidates above
    the human p95 `meta.density_max` -> linearly 0 at density 1), i.e. `lexicon_reward`; without it the raw overlap is a gameable reward. `lexicon_dir` (default: $GEMMA_MT_LEXICON_DIR or ./lexicon)."""
    from gemma_mt.rl import lexicon as L

    lex = L.load_lexicon(lang, lexicon_dir) if lexicon_dir else (L.load_lexicon(lang) if L.lexicon_available(lang) else None)
    if lex is None or not lex.validated:
        return None
    r = lex.overlap(src, mt)
    return lex.guarded(r) if guard else r.overlap


def lexicon_language_weight(lang: str, lexicon_dir: Optional[str] = None, cap: float = 0.25) -> float:
    """Per-language weight of the lexicon penalty = min(cap, the validation's proposed weight `meta.weight`); 0 for a language without a validated lexicon (D-59)."""
    from gemma_mt.rl import lexicon as L

    lex = L.load_lexicon(lang, lexicon_dir) if lexicon_dir else L.load_lexicon(lang)
    if lex is None or not lex.validated:
        return 0.0
    return min(float(cap), max(0.0, float(lex.meta.get("weight", 0.0))))


# ------------------------------------------------------------------------------------------------ the object used by the reward
@dataclass
class Prepared:
    """Everything about one (source, output) sample that the sample-level detectors need, computed once."""
    w: List[str]
    src_w: List[str]
    key: int
    grams: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    src_hashes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def prompt_key(lang: str, src: str) -> int:
    return hash((lang, unicodedata.normalize("NFC", src or "")))


def prepare(lang: str, src: str, mt: str, key: Optional[int] = None, with_grams: bool = True) -> Prepared:
    w, sw = words(mt), words(src)
    p = Prepared(w, sw, prompt_key(lang, src) if key is None else key)
    if with_grams:
        sset = set(sw)
        p.grams = gram_hashes(w, BOILER_N, [x in sset for x in w])
        p.src_hashes = gram_hashes(sw, BOILER_RELATED_N)
    return p


def _clip01(x: float) -> float:
    return 0.0 if x <= 0.0 else 1.0 if x >= 1.0 else float(x)


class Anchors:
    """Calibration + buffer + the deficit functions (all return 0..1; the reward multiplies by the detector's weight).

    `detectors`: which detectors are active at all (default entity + repeat + boiler; `lexicon` and `lid_prob` are opt-in hooks). `modes`: content modes the penalties apply to.
    `off_languages`: the CONTROL list (languages that never get any anchor penalty). `lexicon_fn(lang, src, mt) -> Optional[float]`: replaces the rl/lexicon.py hook (tests)."""

    def __init__(self, calibration: Optional[Dict[str, Any]] = None, *, modes: Sequence[str] = DEFAULT_MODES, off_languages: Sequence[str] = (),
                 detectors: Sequence[str] = ("entity", "repeat", "boiler"), lexicon_dir: Optional[str] = None, buffer_capacity: int = 600,
                 buffer_min_outputs: int = 150, buffer_min_prompts: int = 30, buffer_min_other: int = 3, buffer_per_prompt: int = BOILER_PER_PROMPT,
                 related_min: int = BOILER_RELATED_MIN, lexicon_fn: Optional[Any] = None,
                 lexicon_guard: bool = True, lexicon_per_language: bool = False, lexicon_weight_cap: float = 0.25):
        self.cal = calibration or {}
        self.pooled = self.cal.get("pooled") or {}
        self.langs = self.cal.get("languages") or {}
        self.proxies = self.cal.get("proxies") or {}        # language without references -> a related language whose blocks it borrows (swc -> swh)
        self.modes = tuple(modes)
        self.off = frozenset(off_languages)
        self.detectors = frozenset(detectors)
        self.buffer = BoilerplateBuffer(buffer_capacity, buffer_min_outputs, buffer_min_prompts, buffer_min_other, per_prompt=buffer_per_prompt, related_min=related_min)
        self.lexicon_dir = lexicon_dir
        self._lexicon_fn = lexicon_fn
        self.lexicon_guard, self.lexicon_per_language, self.lexicon_weight_cap = bool(lexicon_guard), bool(lexicon_per_language), float(lexicon_weight_cap)

    # -- construction
    @classmethod
    def from_file(cls, path: Optional[str], **kw: Any) -> "Anchors":
        cal = None
        if path:
            with open(path, encoding="utf-8") as f:
                cal = json.load(f)
        return cls(cal, **kw)

    # -- which languages
    def applies(self, lang: str, mode: str) -> bool:
        return mode in self.modes and lang not in self.off

    def _blk(self, lang: str, name: str) -> Optional[Dict[str, Any]]:
        """The language's calibration block `name` if it has one and it is enabled; a language without references borrows its proxy's block, else the pooled prior; else None."""
        if lang not in self.langs and lang in self.proxies:
            lang = self.proxies[lang]
        b = (self.langs.get(lang) or {}).get(name)
        if b is None:
            b = self.pooled.get(name)
        return b if b and b.get("enabled", True) else None

    def enabled(self, lang: str, det: str) -> bool:
        if det not in self.detectors:
            return False
        if det == "entity":
            b = self._blk(lang, "entity")
            return bool(b) and any(c.get("enabled") for c in (b.get("num") or {}, b.get("name") or {}))
        if det in ("repeat", "boiler", "lid_prob", "lexicon"):
            return self._blk(lang, det) is not None
        return False

    # -- (a) entities, per chunk
    def entity_deficit(self, lang: str, src: str, mt: str) -> Tuple[float, int]:
        """(deficit 0..1, number of source entities judged). The count-weighted mean of the channel deficits max(0, t - recall) / t over the enabled channels the chunk is
        long enough for; (0.0, 0) when nothing is judged."""
        b = self._blk(lang, "entity")
        if not b:
            return 0.0, 0
        ec = entity_counts(src, mt)
        tot = wsum = 0
        for ch, n, r, nmin in (("num", ec.n_num, ec.recall_num, ENTITY_MIN_NUM), ("name", ec.n_name, ec.recall_name, ENTITY_MIN_NAME)):
            c = b.get(ch) or {}
            t = float(c.get("t") or 0.0)
            if not c.get("enabled") or t <= 0 or r is None or n < int(c.get("min_n", nmin)):
                continue
            tot += n * _clip01((t - r) / t)
            wsum += n
        return (tot / wsum if wsum else 0.0), wsum

    # -- sample level: preparation, (b), (c), (e)
    def prepare(self, lang: str, src: str, mt: str, key: Optional[int] = None) -> Prepared:
        return prepare(lang, src, mt, key, with_grams="boiler" in self.detectors)

    def repeat_deficit(self, lang: str, p: Prepared) -> Optional[float]:
        b = self._blk(lang, "repeat")
        if not b or len(p.w) < REPEAT_MIN_WORDS:
            return None
        k = length_bucket(len(p.w))
        sh = repeat_shares(p.w)
        src_sh = repeat_shares(p.src_w) if len(p.src_w) >= REPEAT_MIN_WORDS else {n: 0.0 for n in REPEAT_NS}
        d = 0.0
        for n in REPEAT_NS:
            thr = (b.get("thr") or {}).get(str(n))
            t = float(thr[k]) if thr else 1.0
            cap = float((b.get("cap") or {}).get(str(n), 0.5))
            t = max(t, src_sh[n])                                    # a source that repeats itself is allowed a translation that does
            if cap > t:
                d = max(d, _clip01((sh[n] - t) / (cap - t)))
        return d

    def boiler_deficit(self, lang: str, p: Prepared) -> Optional[float]:
        b = self._blk(lang, "boiler")
        if not b:
            return None
        s = self.buffer.score(lang, p.key, p.grams, p.src_hashes)
        if s is None:
            return None
        t, cap = float(b.get("thr", 0.03)), float(b.get("cap", 0.4))
        return _clip01((s - t) / (cap - t)) if cap > t else 0.0

    def observe(self, lang: str, p: Prepared) -> None:
        """Add one OK output of a training rollout to the language's buffer (at most `per_prompt` outputs of one prompt are kept)."""
        self.buffer.add(lang, p.key, p.grams, p.src_hashes)

    def lid_prob_deficit(self, lang: str, label_ok: bool, prob: Optional[float]) -> Optional[float]:
        b = self._blk(lang, "lid_prob")
        if not b or prob is None or not label_ok:
            return None
        pmin = float(b.get("pmin") or 0.0)
        return _clip01((pmin - prob) / pmin) if pmin > 0 else 0.0

    # -- (d) lexicon
    def lexicon_overlap(self, lang: str, src: str, mt: str) -> Optional[float]:
        if self._lexicon_fn is not None:
            return self._lexicon_fn(lang, src, mt)
        return lexicon_overlap(lang, src, mt, self.lexicon_dir, self.lexicon_guard)

    def lexicon_weight_factor(self, lang: str) -> float:
        """Multiplier of `anchors_lexicon_weight` for this language: 1 unless `lexicon_per_language`, then min(cap, the lexicon's validated weight) (0 without a validated lexicon)."""
        if not self.lexicon_per_language:
            return 1.0
        return lexicon_language_weight(lang, self.lexicon_dir, self.lexicon_weight_cap)

    def lexicon_deficit(self, lang: str, src: str, mt: str) -> Optional[float]:
        """Deficit 0..1 = (t - overlap) / t below the language's human low quantile `lexicon.t`; None unless the detector is active, the language has a calibrated threshold and the
        lexicon judges the chunk."""
        if "lexicon" not in self.detectors:
            return None
        b = self._blk(lang, "lexicon")
        if not b or not float(b.get("t") or 0) > 0:
            return None
        ov = self.lexicon_overlap(lang, src, mt)
        if ov is None:
            return None
        t = float(b["t"])
        return _clip01((t - ov) / t)
