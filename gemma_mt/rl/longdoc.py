# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reward v2 for LONG documents (D-60; agent R8, 2026-10-22). Everything here is OFF unless `reward.longdoc_v2` is set; nothing in the v1 path calls it.

Why (reports/longdoc_reward_v2_2026-10-22.md): at b1024 - b4096 the v1 reward (a) fails good, loop-free translations with the whole-text repetition guards
(`guards.is_repetitive`: distinct-word ratio < 0.3 and zlib ratio < 0.18 are LENGTH-dependent - every natural 3,000-word text trips the first one; NLLB's own
translations get -0.96 at b4096), (b) is a plain source-length weighted mean over up to 300 sentences, blind to a localised failure, (c) aligns sentences by length
alone and is thrown off by a duplicated sentence, a looped tail or an omission, and (d) punishes loops all-or-nothing (-2) so a document that loops in its last
10 % scores like one that loops from the first sentence.

Parts (pure python, stdlib only; the reward calls them behind flags):

  1. REPETITION PROFILE at four levels (`repetition_profile`): word 4-gram repeat share, repeated-8-gram coverage, exact duplicate sentences, repeated blocks of
     >= 2 sentences, periodic word cycles ('muna muna muna'). Each level is a share (0..1) of the OUTPUT that is a redundant copy.
  2. CONTINUOUS SEVERITY (`LongdocCalibration.severity`): per level a ramp from the HUMAN limit of that length bucket (p99.5 of human documents, calibration/longdoc_v2.json,
     scripts/longdoc_v2/calibrate_repetition.py) to a 'clearly degenerate' share; the sample's severity s in [0, 1] is the largest level. The reward becomes
     r - s * (r - penalty): s = 0 leaves it alone, s = 1 is the failure penalty (-2), in between it is linear, so a loop in the last 10 % costs a tenth of the way.
  3. TAIL-SENSITIVE AGGREGATION (`aggregate_tail`): mean over the chunks blended with the mean of the worst quartile (CVaR 25) - failed chunks count as the penalty.
  4. COVERAGE: the fraction of the SOURCE (characters) that has no output (omitted, copied, shortfall) scales an extra drop towards the penalty (`coverage_severity`).
  5. ANCHORED ALIGNMENT (`anchored_beads`): redundant output sentences (copies) are removed first, then numbers / names / lexicon words that occur in both texts pin
     source and output sentences together and the length model (segments.gale_church) aligns only the gaps between the anchors.
"""

from __future__ import annotations

import bisect
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from gemma_mt.rl.guards import normalise
from gemma_mt.rl.segments import LengthModel, gale_church, split_sentences

LEVELS = ("rep4", "cover8", "dup_sent", "dup_block", "cycle")      # the repetition levels, in the order of RepProfile.levels
MIN_WORDS_NGRAM = 24                                                # no n-gram verdict below this many words (as guards.is_self_repetitive)
DUP_MIN_WORDS = 4                                                   # a sentence shorter than this may legitimately repeat ('Thank you.', 'Yes.')


# ------------------------------------------------------------------------------------------------ 1. repetition profile
@dataclass
class RepProfile:
    n_words: int = 0
    n_sents: int = 0
    rep4: float = 0.0          # share of word 4-grams that repeat an earlier 4-gram (guards.repeated_ngram_fraction)
    cover8: float = 0.0        # share of the words inside a repeated word 8-gram (second and later occurrences)
    dup_sent: float = 0.0      # share of the characters in sentences (>= DUP_MIN_WORDS words) that repeat an earlier sentence exactly (after normalisation)
    dup_block: float = 0.0     # share of the characters in sentences that belong to a repeated block of >= 2 consecutive sentences
    cycle: float = 0.0         # share of the words in the extra copies of a periodic word cycle (>= 3 repeats of a 1-30 word unit, >= 6 words)
    cycle_period: int = 0      # words of the longest cycle found
    cycle_at_tail: bool = False
    redundant_sents: List[int] = field(default_factory=list)   # indices of sentences that are redundant copies (duplicate sentences, repeated blocks)

    @property
    def levels(self) -> Dict[str, float]:
        return {"rep4": self.rep4, "cover8": self.cover8, "dup_sent": self.dup_sent, "dup_block": self.dup_block, "cycle": self.cycle}

    @property
    def redundant(self) -> float:
        """Largest level except rep4 (rep4 is the noisiest, it only enters through its own calibrated ramp)."""
        return max(self.cover8, self.dup_sent, self.dup_block, self.cycle)


def _words(text: str) -> List[str]:
    return normalise(text).split()


def ngram_repeat_share(words: Sequence[str], n: int) -> float:
    if len(words) < n + 1:
        return 0.0
    seen: Set[Tuple[str, ...]] = set()
    rep = 0
    tot = len(words) - n + 1
    for i in range(tot):
        g = tuple(words[i: i + n])
        if g in seen:
            rep += 1
        else:
            seen.add(g)
    return rep / tot


def ngram_cover_share(words: Sequence[str], n: int) -> float:
    """Share of the words that lie inside a repeated n-gram occurrence (an n-gram that occurred before; the first occurrence is not counted)."""
    m = len(words)
    if m < n + 1:
        return 0.0
    seen: Set[Tuple[str, ...]] = set()
    covered = bytearray(m)
    for i in range(m - n + 1):
        g = tuple(words[i: i + n])
        if g in seen:
            for k in range(i, i + n):
                covered[k] = 1
        else:
            seen.add(g)
    return sum(covered) / m


def word_cycles(words: Sequence[str], max_period: int = 30, min_repeats: int = 3, min_words: int = 6, min_repeats_p1: int = 6) -> Tuple[float, int, bool]:
    """(share of the words in the extra copies of periodic cycles, smallest period found, whether a cycle reaches the end of the text). A cycle = a unit of p words (1..max_period)
    repeated at least `min_repeats` times in a row (period 1: `min_repeats_p1`) covering at least `min_words` words. Copies beyond the first count as redundant; the period reported is the smallest."""
    n = len(words)
    if n < min_words:
        return 0.0, 0, False
    ids: Dict[str, int] = {}
    t = [ids.setdefault(w, len(ids)) for w in words]
    red = bytearray(n)
    best_p = 0
    tail = False
    for p in range(1, min(max_period, n // min_repeats) + 1):
        need = (min_repeats_p1 if p == 1 else min_repeats) - 1
        run = 0
        start = 0
        done = -1                                              # last position already marked in this pass (linear time even for a loop of thousands of words)
        for i in range(n - p):
            if t[i] == t[i + p]:
                if run == 0:
                    start = i
                run += 1
            else:
                run = 0
            if run >= need * p and run + p >= min_words:
                # words start+p .. i+p are copies of the unit before them
                for k in range(max(start + p, done + 1), i + p + 1):
                    red[k] = 1
                done = i + p
                if best_p == 0 or p < best_p:
                    best_p = p
                if i + p >= n - 1:
                    tail = True
    return sum(red) / n, best_p, tail


def _sent_keys(sents: Sequence[str]) -> List[str]:
    return [" ".join(_words(s)) for s in sents]


def sentence_repetition(sents: Sequence[str]) -> Tuple[float, float, List[int]]:
    """(dup_sent share, dup_block share, indices of redundant sentences). Shares are character shares of the (normalised) output."""
    keys = _sent_keys(sents)
    lens = [len(k) for k in keys]
    tot = sum(lens)
    if tot == 0:
        return 0.0, 0.0, []
    first: Dict[str, int] = {}
    dup = [False] * len(keys)
    for i, k in enumerate(keys):
        if not k:
            continue
        if k in first and len(k.split()) >= DUP_MIN_WORDS:
            dup[i] = True
        else:
            first.setdefault(k, i)
    # blocks: a bigram of consecutive sentences seen before marks both sentences of the second occurrence
    seen2: Dict[Tuple[str, str], int] = {}
    blk = [False] * len(keys)
    for i in range(len(keys) - 1):
        if not keys[i] or not keys[i + 1]:
            continue
        b = (keys[i], keys[i + 1])
        if b in seen2 and (len(keys[i].split()) + len(keys[i + 1].split())) >= 2 * DUP_MIN_WORDS:
            blk[i] = blk[i + 1] = True
        else:
            seen2.setdefault(b, i)
    d = sum(l for l, x in zip(lens, dup) if x) / tot
    b = sum(l for l, x in zip(lens, blk) if x) / tot
    red = [i for i in range(len(keys)) if dup[i] or blk[i]]
    return d, b, red


def repetition_profile(text: str, sents: Optional[Sequence[str]] = None) -> RepProfile:
    """All repetition levels of `text` (cost: linear in the number of words for the n-gram levels, 30 passes for the cycle level)."""
    words = _words(text)
    p = RepProfile(n_words=len(words))
    sents = split_sentences(text) if sents is None else list(sents)
    p.n_sents = len(sents)
    if len(words) >= MIN_WORDS_NGRAM:
        p.rep4 = ngram_repeat_share(words, 4)
        p.cover8 = ngram_cover_share(words, 8)
    p.cycle, p.cycle_period, p.cycle_at_tail = word_cycles(words)
    if len(sents) >= 2:
        p.dup_sent, p.dup_block, p.redundant_sents = sentence_repetition(sents)
    return p


def has_stutter(text: str, max_run: int = 4, cycle_share: float = 0.3) -> bool:
    """A chunk-level loop: one word repeated more than `max_run` times in a row, or at least `cycle_share` of the words are extra copies of a periodic cycle.
    In v2 a stuttering CHUNK is a failed chunk (its source weight counts as the penalty) instead of failing the whole document."""
    ws = text.split()
    run = 1
    for i in range(1, len(ws)):
        run = run + 1 if ws[i] == ws[i - 1] else 1
        if run > max_run:
            return True
    return word_cycles(_words(text))[0] >= cycle_share


def drop_redundant(sents: Sequence[str], keep: int = 0) -> Tuple[List[str], List[int]]:
    """The sentences without the redundant copies (exact duplicates of an earlier sentence, second copies of repeated blocks), and the removed indices.
    `keep`: the number of redundant sentences the SOURCE has itself (a document that repeats a line is translated into a text that repeats it): the first `keep` redundant
    sentences stay, only the surplus is removed (loops sit at the end, so the earliest copies are the likeliest legitimate ones)."""
    _, _, red = sentence_repetition(sents)
    red = red[max(0, keep):]
    if not red:
        return list(sents), []
    rs = set(red)
    return [s for i, s in enumerate(sents) if i not in rs], red


# ------------------------------------------------------------------------------------------------ 2. calibration and continuous severity
DEFAULT_BUCKET_EDGES = (60, 230, 560, 1000, 2000)      # source WORDS: b64 / b256 / b512 / b1024 / b2048 / b4096 (an English token is ~0.7 words)
DEFAULT_T1 = {"rep4": 0.60, "cover8": 0.50, "dup_sent": 0.50, "dup_block": 0.50, "cycle": 0.50}   # share at which the level is 'clearly degenerate' (severity 1)
LEVEL_FLOOR = {"rep4": 0.10, "cover8": 0.04, "dup_sent": 0.03, "dup_block": 0.03, "cycle": 0.02}      # a calibrated limit is never below this (a small human sample must not give a zero limit)
DEFAULT_T0 = {"rep4": 0.35, "cover8": 0.10, "dup_sent": 0.06, "dup_block": 0.05, "cycle": 0.02}   # fallback human limits when no calibration file is given


DEFAULT_TAIL_GAP = (0.85, 0.85, 0.9, 0.95, 0.95, 1.0)      # fallback when the calibration file has none (measured medians of human documents, reports/longdoc_reward_v2_2026-10-22.md)


@dataclass
class LongdocCalibration:
    """Per length bucket: the limit t0 of every repetition level (human p99.5, widened by `margin`) and the share t1 that counts as fully degenerate."""
    edges: Tuple[int, ...] = DEFAULT_BUCKET_EDGES
    t0: List[Dict[str, float]] = field(default_factory=lambda: [dict(DEFAULT_T0) for _ in range(len(DEFAULT_BUCKET_EDGES) + 1)])
    t1: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_T1))
    meta: Dict[str, Any] = field(default_factory=dict)
    tail_gap: List[float] = field(default_factory=lambda: list(DEFAULT_TAIL_GAP))      # per bucket: median over HUMAN documents of (mean chunk reward - CVaR25 of the chunk rewards), the tail deficit that is normal

    @classmethod
    def from_file(cls, path: Optional[str]) -> "LongdocCalibration":
        if not path:
            return cls()
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        edges = tuple(int(x) for x in d["edges"])
        t0 = [{k: float(v) for k, v in b.items()} for b in d["t0"]]
        if len(t0) != len(edges) + 1:
            raise ValueError(f"{path}: {len(t0)} buckets for {len(edges)} edges")
        t1 = {k: float(v) for k, v in (d.get("t1") or DEFAULT_T1).items()}
        tg = [float(x) for x in (d.get("tail_gap") or DEFAULT_TAIL_GAP)]
        if len(tg) != len(edges) + 1:
            raise ValueError(f"{path}: {len(tg)} tail gaps for {len(edges) + 1} buckets")
        return cls(edges, t0, t1, d.get("meta") or {}, tg)

    def gap0(self, src_words: int) -> float:
        return self.tail_gap[self.bucket(src_words)]

    def bucket(self, src_words: int) -> int:
        return bisect.bisect_left(self.edges, src_words)

    def ramp(self, level: str, share: float, src_words: int) -> float:
        a = self.t0[self.bucket(src_words)][level]
        b = max(self.t1.get(level, 0.5), a + 1e-6)
        return min(1.0, max(0.0, (share - a) / (b - a)))

    def severity(self, prof: RepProfile, src_words: int, src_prof: Optional[RepProfile] = None) -> Tuple[float, str]:
        """(s in [0, 1], the level that set it). With the profile of the English SOURCE the level shares are measured as the EXCESS over the source's own repetition
        (an exercise list that repeats 'Which of the following' is translated into a text that repeats it too; a human translation keeps the source's repetition)."""
        best, name = 0.0, ""
        for lv, x in prof.levels.items():
            if src_prof is not None:
                x = max(0.0, x - src_prof.levels[lv])
            s = self.ramp(lv, x, src_words)
            if s > best:
                best, name = s, lv
        return best, name


def apply_severity(r: float, s: float, penalty: float) -> float:
    """r - s * (r - penalty) for r above the penalty: s = 1 gives the failure penalty exactly, s = 0 leaves r. Never raises a reward (r <= penalty stays)."""
    if s <= 0.0 or r <= penalty:
        return r
    return r - s * (r - penalty)


# ------------------------------------------------------------------------------------------------ 3. tail-sensitive aggregation
def weighted_tail_mean(vals: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Weighted mean of the worst fraction q of the total weight (CVaR_q; a value is split at the boundary). q >= 1 is the plain weighted mean."""
    tot = sum(weights)
    if tot <= 0 or not vals:
        return float(sum(vals) / max(len(vals), 1))
    if q >= 1.0:
        return sum(v * w for v, w in zip(vals, weights)) / tot
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    budget = q * tot
    acc = s = 0.0
    for i in order:
        take = min(weights[i], budget - acc)
        if take <= 0:
            break
        s += take * vals[i]
        acc += take
        if acc >= budget - 1e-12:
            break
    return s / max(acc, 1e-12)


def weighted_quantile(vals: Sequence[float], weights: Sequence[float], q: float) -> float:
    tot = sum(weights)
    if tot <= 0 or not vals:
        return float(sum(vals) / max(len(vals), 1))
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    acc = 0.0
    for i in order:
        acc += weights[i]
        if acc >= q * tot - 1e-12:
            return vals[i]
    return vals[order[-1]]


def aggregate_tail(items: Sequence[Tuple[Optional[int], float]], seg_rewards: Dict[int, float], penalty: float, *, mode: str = "cvar",
                   alpha: float = 0.5, q: float = 0.25, min_chunks: int = 8, gap0: float = 0.9) -> float:
    """Chunk rewards -> one reward. `items` as chunk_plan.PlanItem; a failed chunk (None) is `penalty`.
      mean     weighted mean (v1)
      cvar     (1 - alpha) * mean + alpha * mean of the worst fraction q of the weight (default: half mean, half worst-quartile mean)
      p25      (1 - alpha) * mean + alpha * weighted q-quantile
      min      (1 - alpha) * mean + alpha * worst chunk
      cvar_excess   mean - alpha * max(0, (mean - CVaR_q) - gap0): only the tail deficit BEYOND what human documents show (`gap0`, calibration `tail_gap` per length bucket) costs reward,
                    so a human-quality translation keeps its mean (a plain blend lowers every document by ~0.4 whatever its quality)
    Fewer than `min_chunks` chunks: the plain mean (a tail of 3 sentences is the sentence itself)."""
    vals = [penalty if seg is None else seg_rewards[seg] for seg, _ in items]
    ws = [w for _, w in items]
    tot = sum(ws)
    if not vals:
        return float(penalty)
    if tot <= 0:
        return float(sum(vals) / len(vals))
    mean = sum(v * w for v, w in zip(vals, ws)) / tot
    if mode == "mean" or len(vals) < min_chunks:
        return mean
    if mode == "cvar_excess":
        return mean - alpha * max(0.0, (mean - weighted_tail_mean(vals, ws, q)) - gap0)
    if mode == "cvar":
        tail = weighted_tail_mean(vals, ws, q)
    elif mode == "p25":
        tail = weighted_quantile(vals, ws, q)
    elif mode == "min":
        tail = min(vals)
    else:
        raise ValueError(f"unknown longdoc aggregation {mode!r}")
    return (1.0 - alpha) * mean + alpha * tail


# ------------------------------------------------------------------------------------------------ 4. coverage
def coverage_severity(items: Sequence[Tuple[Optional[int], float]], kappa: float = 1.0) -> float:
    """kappa * (weight of the failed chunks / total weight): the share of the source (characters) with no usable output, in [0, 1]. Failed chunks that stand
    for EXTRA output text (no source side) are included at their output-character weight like the aggregation does, so the share can only over-count them."""
    tot = sum(w for _, w in items)
    if tot <= 0:
        return 0.0
    failed = sum(w for seg, w in items if seg is None)
    return min(1.0, max(0.0, kappa * failed / tot))


def combine_severity(*sev: float) -> float:
    """1 - prod(1 - s): independent failure modes."""
    p = 1.0
    for s in sev:
        p *= 1.0 - min(max(s, 0.0), 1.0)
    return 1.0 - p


# ------------------------------------------------------------------------------------------------ 5. anchored alignment
_NUMBER = re.compile(r"(?<![A-Za-z\d])\d+(?:[.,]\d+)*")
_TOKEN_EDGE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def _fold(s: str) -> str:
    """Lowercase, accents removed (NFKD without combining marks): 'Müller' == 'muller', tone marks of the target do not matter for names / numbers."""
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def number_items(sent: str) -> List[str]:
    """Numbers of a sentence that can pin it: >= 2 digits, or a decimal / grouped number (a lone '1' or '2' is in every other sentence)."""
    out = []
    for m in _NUMBER.finditer(sent):
        d = m.group(0)
        digits = re.sub(r"\D", "", d)
        if len(digits) >= 2 or (len(d) > len(digits) and len(digits) >= 2):
            out.append(digits)
    return out


def name_items(sent: str) -> List[str]:
    """Names, acronyms and letter+digit codes of an English sentence (kept unchanged by most translations): capitalised words that do not start the sentence, ALL-CAPS words
    of >= 2 letters, tokens that mix letters and digits. Folded."""
    toks = sent.split()
    out = []
    for k, t in enumerate(toks):
        w = _TOKEN_EDGE.sub("", t)
        if len(w) < 3 and not (w.isupper() and len(w) >= 2):
            continue
        has_d = any(c.isdigit() for c in w)
        has_a = any(c.isalpha() for c in w)
        if has_d and has_a and any(c.isupper() for c in w):
            out.append(_fold(w))
        elif w.isupper() and len(w) >= 2 and has_a:
            out.append(_fold(w))
        elif w[:1].isupper() and k > 0 and len(w) >= 3 and w[1:].islower() and toks[k - 1][-1:] not in ".!?:":
            out.append(_fold(w))
    return out


class _OutIndex:
    """Inverted index of the output sentences: number -> sentence set, folded token -> sentence set."""

    def __init__(self, sents: Sequence[str]):
        self.num: Dict[str, Set[int]] = {}
        self.tok: Dict[str, Set[int]] = {}
        for j, s in enumerate(sents):
            for d in number_items(s):
                self.num.setdefault(d, set()).add(j)
            for t in s.split():
                w = _fold(_TOKEN_EDGE.sub("", t))
                if w:
                    self.tok.setdefault(w, set()).add(j)
        self._long = None

    def name_hits(self, name: str) -> Set[int]:
        hit = self.tok.get(name)
        if hit:
            return hit
        if len(name) >= 5:                                   # agglutinative prefixes / suffixes around a kept name (Zulu 'eCranfield', 'Cranfieldini')
            if self._long is None:
                self._long = [(t, s) for t, s in self.tok.items() if len(t) >= 5]
            acc: Set[int] = set()
            for t, s in self._long:
                if name in t:
                    acc |= s
            return acc
        return set()


def anchor_scores(src: Sequence[str], out: Sequence[str], lexicon: Any = None, max_occ: int = 4, w_num: float = 1.0, w_name: float = 0.8,
                  w_lex: float = 0.2, lex_min_idf: float = 3.0) -> Dict[Tuple[int, int], float]:
    """Sparse evidence that source sentence i and output sentence j belong together: shared numbers, kept names, lexicon words (rare English content words whose
    lexicon translations occur in j). An item that occurs in many sentences carries 1 / (source occurrences x output occurrences) of its weight."""
    idx = _OutIndex(out)
    s_items: Dict[Tuple[str, str], Set[int]] = {}
    for i, s in enumerate(src):
        for d in number_items(s):
            s_items.setdefault(("n", d), set()).add(i)
        for nm in name_items(s):
            s_items.setdefault(("w", nm), set()).add(i)
    pairs: Dict[Tuple[int, int], float] = {}
    for (kind, key), si in s_items.items():
        oj = idx.num.get(key, set()) if kind == "n" else idx.name_hits(key)
        if not oj or len(si) > max_occ or len(oj) > max_occ:
            continue
        w = (w_num if kind == "n" else w_name) / (len(si) * len(oj))
        for i in si:
            for j in oj:
                pairs[(i, j)] = pairs.get((i, j), 0.0) + w
    if lexicon is not None:
        from gemma_mt.rl.lexicon import english_content_stems, token_matches  # noqa: F401  (token_matches kept for affix lexicons)

        lex_src: Dict[str, Set[int]] = {}
        for i, s in enumerate(src):
            for st in english_content_stems(s):
                e = lexicon.entries.get(st)
                if e is not None and e[0] >= lex_min_idf:
                    lex_src.setdefault(st, set()).add(i)
        if lex_src:
            from gemma_mt.rl.lexicon import x_folded_tokens

            otok: Dict[str, Set[int]] = {}
            for j, s in enumerate(out):
                for t in set(x_folded_tokens(s)):
                    otok.setdefault(t, set()).add(j)
            for st, si in lex_src.items():
                if len(si) > max_occ:
                    continue
                _, _, cands = lexicon.entries[st]
                oj: Set[int] = set()
                for c, conf in cands[:3]:
                    oj |= otok.get(c, set())
                if not oj or len(oj) > max_occ:
                    continue
                w = w_lex / (len(si) * len(oj))
                for i in si:
                    for j in oj:
                        pairs[(i, j)] = pairs.get((i, j), 0.0) + w
    return pairs


def anchor_chain(pairs: Dict[Tuple[int, int], float], min_weight: float = 0.5, offset_cost: float = 0.01, strict: bool = False) -> List[Tuple[int, int]]:
    """Heaviest chain of anchor pairs that is non-decreasing in both coordinates (equal in one = a merged / split sentence). Pairs below `min_weight` are not anchors on their own;
    a jump of the offset (j - i) between consecutive anchors costs `offset_cost` per unit (omissions and extra sentences shift it legitimately, a random far match is expensive)."""
    pts = sorted((p for p, w in pairs.items() if w >= min_weight), key=lambda p: (p[0], p[1]))
    if not pts:
        return []
    best = [0.0] * len(pts)
    prev = [-1] * len(pts)
    for a, (i, j) in enumerate(pts):
        w = pairs[(i, j)]
        best[a] = w
        for b in range(a):
            i0, j0 = pts[b]
            if j0 > j or (i0 == i and j0 == j) or (strict and (i0 == i or j0 == j)):
                continue
            c = best[b] + w - offset_cost * abs((j - i) - (j0 - i0))
            if c > best[a]:
                best[a], prev[a] = c, b
    a = max(range(len(pts)), key=lambda k: best[k])
    chain = []
    while a >= 0:
        chain.append(pts[a])
        a = prev[a]
    return chain[::-1]


Bead = Tuple[Tuple[int, ...], Tuple[int, ...]]


def _gap_beads(src_lo: int, src_hi: int, out_lo: int, out_hi: int, src: Sequence[str], out: Sequence[str], lm: Optional[LengthModel]) -> List[Bead]:
    """Beads for the source range [src_lo, src_hi) against the output range [out_lo, out_hi) (length model; omissions / extras at the ends of an empty side)."""
    ns, no = src_hi - src_lo, out_hi - out_lo
    if ns <= 0 and no <= 0:
        return []
    if ns <= 0:
        return [((), (j,)) for j in range(out_lo, out_hi)]
    if no <= 0:
        return [((i,), ()) for i in range(src_lo, src_hi)]
    if ns == 1 and no == 1:
        return [((src_lo,), (out_lo,))]
    if ns == no:
        return [((src_lo + k,), (out_lo + k,)) for k in range(ns)]
    beads = gale_church(src[src_lo:src_hi], out[out_lo:out_hi], lm)
    return [(tuple(src_lo + i for i in si), tuple(out_lo + j for j in oi)) for si, oi in beads]


def anchored_beads(src: Sequence[str], out: Sequence[str], lm: Optional[LengthModel] = None, *, lexicon: Any = None, dedupe: bool = True, anchors: bool = True,
                   trust_equal: bool = True, min_anchor_weight: float = 0.5, max_pairs: int = 400) -> List[Bead]:
    """Beads (source sentence indices, output sentence indices) over the ORIGINAL output indices.
      1. dedupe: redundant output copies (duplicate sentences, repeated blocks) are taken out first and come back as extra beads ((), (j,)); they cannot shift the alignment
      2. equal counts and the anchors agree with the identity (or there are none): i-th to i-th, as v1
      3. otherwise numbers / names / lexicon words pin sentences together (heaviest monotone chain) and the language-aware length model aligns each gap between two anchors"""
    n, m = len(src), len(out)
    if n == 0 or m == 0:
        return [(tuple(range(n)), tuple(range(m)))] if (n or m) else []
    removed: List[int] = []
    kept_idx = list(range(m))
    if dedupe and m >= 2:
        kept, removed = drop_redundant(out, keep=len(sentence_repetition(src)[2]))
        if removed:
            rs = set(removed)
            kept_idx = [j for j in range(m) if j not in rs]
    o2 = [out[j] for j in kept_idx]
    m2 = len(o2)
    chain: List[Tuple[int, int]] = []
    pairs: Dict[Tuple[int, int], float] = {}
    if anchors and n >= 2 and m2 >= 2:
        pairs = anchor_scores(src, o2, lexicon)
        if len(pairs) > max_pairs:                               # keep the heaviest (bounded chain cost)
            pairs = dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:max_pairs])
        chain = anchor_chain(pairs, min_anchor_weight)
    beads: List[Bead] = []
    ident = False
    if n == m2 and trust_equal:
        if not chain:
            ident = True
        else:                                                    # equal counts: identity unless a strictly monotone chain of anchors outweighs it (a drop + a duplicate that cancel in the count)
            sc = anchor_chain(pairs, min_anchor_weight, strict=True)
            ident = sum(pairs.get((i, i), 0.0) for i in range(n)) >= 0.8 * sum(pairs[p] for p in sc)
    if ident:
        beads = [((i,), (i,)) for i in range(n)]
    elif n < 2:
        beads = [((0,), tuple(range(m2)))]
    elif m2 <= 1:
        beads = [(tuple(range(n)), tuple(range(m2)))]
    else:
        # anchor groups: consecutive anchors that share a coordinate form one bead (merged / split sentence)
        groups: List[Tuple[int, int, int, int]] = []             # (i_lo, i_hi, j_lo, j_hi) inclusive
        for (i, j) in chain:
            if groups and (groups[-1][1] == i or groups[-1][3] == j):
                g = groups[-1]
                groups[-1] = (g[0], max(g[1], i), g[2], max(g[3], j))
            else:
                groups.append((i, i, j, j))
        pi, pj = 0, 0                                            # next unaligned source / output sentence
        for (i0, i1, j0, j1) in groups:
            beads += _gap_beads(pi, i0, pj, j0, src, o2, lm)
            beads.append((tuple(range(i0, i1 + 1)), tuple(range(j0, j1 + 1))))
            pi, pj = i1 + 1, j1 + 1
        beads += _gap_beads(pi, n, pj, m2, src, o2, lm)
    # back to the original output indices; the removed copies become extra beads at their position
    beads = [(si, tuple(kept_idx[j] for j in oi)) for si, oi in beads]
    if removed:
        beads += [((), (j,)) for j in removed]
        beads.sort(key=lambda b: (min(b[1]) if b[1] else 10 ** 9, min(b[0]) if b[0] else 10 ** 9))
    return beads


def make_study_aligner(name: str) -> Callable:
    """Aligners of scripts/longdoc_v2/align_study.py: fn(src_sents, out_sents, lm, ctx) -> beads. ctx = {"iso", ...}."""
    def lex_for(ctx):
        try:
            from gemma_mt.rl.lexicon import load_lexicon

            return load_lexicon(ctx.get("iso"))
        except Exception:
            return None

    if name == "dedupe":
        return lambda s, o, lm, ctx: anchored_beads(s, o, lm, anchors=False)
    if name == "anchored":
        return lambda s, o, lm, ctx: anchored_beads(s, o, lm)
    if name == "anchored_lex":
        return lambda s, o, lm, ctx: anchored_beads(s, o, lm, lexicon=lex_for(ctx))
    if name == "anchored_nodedupe":
        return lambda s, o, lm, ctx: anchored_beads(s, o, lm, dedupe=False)
    raise KeyError(f"unknown study aligner {name!r}")


def beads_fn(length_model: Optional[LengthModel] = None, lexicon_iso: Optional[str] = None, **kw) -> Callable:
    """The `beads_fn` of chunk_plan.plan_chunks: fn(source, text) -> ([(source text, output text, source idx, output idx)], status) with the anchored alignment of this module.
    `lexicon_iso`: also use the validated lexicon of that language as anchors (None = numbers and names only)."""
    lex = None
    if lexicon_iso:
        try:
            from gemma_mt.rl.lexicon import lexicon_available, load_lexicon

            if lexicon_available(lexicon_iso):
                lex = load_lexicon(lexicon_iso)
        except Exception:
            lex = None

    def fn(source: str, text: str):
        s, o = split_sentences(source), split_sentences(text)
        if len(s) < 2:
            return [(source.strip(), text.strip(), tuple(range(len(s))), tuple(range(len(o))))], "single"
        if not o:
            return [(" ".join(s), "", tuple(range(len(s))), ())], "whole"
        beads = anchored_beads(s, o, length_model, lexicon=lex, **kw)
        how = "gale_church"
        if len(s) == len(o) and all(len(si) == 1 and si == oi for si, oi in beads):
            how = "aligned"
        elif len(o) <= 1:
            how = "whole"
        return [(" ".join(s[i] for i in si), " ".join(o[j] for j in oi), si, oi) for si, oi in beads], how

    return fn
