# SPDX-License-Identifier: AGPL-3.0-or-later
"""QE-free consensus signal (minimum-Bayes-risk style) for languages AfriCOMET cannot judge well (D-38).

For every completion of a GRPO group (G samples of the same prompt) that passed the hard failure checks, the consensus is the MEAN sentence-level chrF++ of
the completion (as hypothesis) against the OTHER valid completions of the group (as references): the translation that most of its siblings agree with is the
one a minimum-Bayes-risk decoder would pick, and agreement between independent samples of the same model needs no reference set and no learned metric.

    chrF++ here = sacrebleu `CHRF(word_order=2)` (character 6-grams, word 2-grams, beta 2, no lower-casing, whitespace ignored) on the texts the reward
    already derived (NFC, tag stripped, cut at the stop id). `chrf_sentence(h, r)` is verified equal to `sacrebleu.CHRF(word_order=2).sentence_score(h, [r])`
    (tests/test_consensus.py). It is re-implemented only for speed: the n-gram counters of each text are built ONCE per sample (a group of 8 needs 8, not
    8 x 7 x 2) and the match counts of a pair are computed once and used for both directions.

    consensus_value = lo + (hi - lo) * chrF / 100            the FIXED linear map [0, 100] -> the reward scale [-2, +2.25] (`lo`, `hi` = the range of
                                                             the shaping preset). Group standardisation removes the offset anyway; the map matters
                                                             where the value shares the scale with penalties and failure classes (-2).

Pool rules (`group_consensus`): only valid samples enter the pool, they neither receive nor give consensus otherwise; a group needs at least `min_valid`
(3) valid samples, else every sample of the group gets `None` (the reward then falls back to the QE-only or the rubric-only value). Identical samples
agree perfectly (chrF 100): a collapsed group is flat and gets zero advantage; near-loops are removed before the pool by the repetition guards
(`guards.is_repetitive`, `guards.is_self_repetitive`).
"""

from __future__ import annotations

import time
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

CHAR_ORDER = 6
WORD_ORDER = 2
BETA = 2
_PUNCTS = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
ORDER = CHAR_ORDER + WORD_ORDER


def _words(sent: str) -> List[str]:
    """sacrebleu `CHRF._remove_punctuation`: a punctuation character at the end (else at the start) of a word becomes a token of its own."""
    out: List[str] = []
    for w in sent.split():
        if len(w) == 1:
            out.append(w)
        elif w[-1] in _PUNCTS:
            out += [w[:-1], w[-1]]
        elif w[0] in _PUNCTS:
            out += [w[0], w[1:]]
        else:
            out.append(w)
    return out


class Profile:
    """The 8 n-gram counters of one text (char 1..6, word 1..2) and their totals."""

    __slots__ = ("grams", "totals")

    def __init__(self, text: str):
        line = "".join(text.split())                                    # whitespace is ignored for the character n-grams (sacrebleu default)
        grams: List[Counter] = [Counter(line[i: i + n] for i in range(len(line) - n + 1)) for n in range(1, CHAR_ORDER + 1)]
        words = _words(text)
        for n in range(1, WORD_ORDER + 1):
            grams.append(Counter(" ".join(words[i: i + n]) for i in range(len(words) - n + 1)))
        self.grams = grams
        self.totals = [sum(c.values()) for c in grams]


def matches(a: Profile, b: Profile) -> List[int]:
    """Clipped n-gram match counts per order (symmetric)."""
    out = []
    for ca, cb in zip(a.grams, b.grams):
        if len(ca) > len(cb):
            ca, cb = cb, ca
        out.append(sum(min(c, cb[g]) for g, c in ca.items() if g in cb))
    return out


def f_score(n_hyp: Sequence[int], n_ref: Sequence[int], n_match: Sequence[int], beta: int = BETA) -> float:
    """sacrebleu `CHRF._compute_f_score` (eps_smoothing False): per order precision / recall, averaged over the orders that exist on both sides, then F_beta; 0..100."""
    factor = beta ** 2
    avg_p = avg_r = 0.0
    eff = 0
    for h, r, m in zip(n_hyp, n_ref, n_match):
        h = h if r > 0 else 0                                          # sacrebleu: hyp count is 0 when the reference has no n-gram of this order
        if h > 0 and r > 0:
            avg_p += m / h
            avg_r += m / r
            eff += 1
    if eff == 0:
        return 0.0
    avg_p /= eff
    avg_r /= eff
    if avg_p + avg_r == 0:
        return 0.0
    return 100.0 * (1 + factor) * avg_p * avg_r / (factor * avg_p + avg_r)


def chrf_profiles(hyp: Profile, ref: Profile, m: Optional[List[int]] = None) -> float:
    return f_score(hyp.totals, ref.totals, matches(hyp, ref) if m is None else m)


def chrf_sentence(hyp: str, ref: str) -> float:
    """Sentence chrF++ (== sacrebleu CHRF(word_order=2).sentence_score(hyp, [ref]).score)."""
    return chrf_profiles(Profile(hyp), Profile(ref))


class ProfileCache:
    """LRU cache text -> Profile (the reward sees every sample once per step; the cache only serves repeated texts, e.g. identical samples of a group)."""

    def __init__(self, size: int = 4096):
        self.size, self._d = size, OrderedDict()

    def get(self, text: str) -> Profile:
        p = self._d.get(text)
        if p is None:
            p = Profile(text)
            self._d[text] = p
            if len(self._d) > self.size:
                self._d.popitem(last=False)
        else:
            self._d.move_to_end(text)
        return p


def group_consensus(profiles: Sequence[Optional[Profile]], min_valid: int = 3) -> List[Optional[float]]:
    """Mean chrF++ (0..100) of each profile against the OTHER non-None profiles of the group; `None` for the invalid ones, and for everybody when the
    group has fewer than `min_valid` valid members."""
    idx = [i for i, p in enumerate(profiles) if p is not None]
    out: List[Optional[float]] = [None] * len(profiles)
    if len(idx) < max(min_valid, 2):
        return out
    acc = {i: 0.0 for i in idx}
    for a in range(len(idx)):
        pa = profiles[idx[a]]
        for b in range(a + 1, len(idx)):
            pb = profiles[idx[b]]
            m = matches(pa, pb)
            acc[idx[a]] += f_score(pa.totals, pb.totals, m)             # a as hypothesis, b as reference
            acc[idx[b]] += f_score(pb.totals, pa.totals, m)             # b as hypothesis, a as reference
    k = len(idx) - 1
    for i in idx:
        out[i] = acc[i] / k
    return out


def consensus_value(chrf: float, lo: float = -2.0, hi: float = 2.25) -> float:
    """The fixed linear map of a chrF++ in [0, 100] to the reward scale."""
    return lo + (hi - lo) * min(max(chrf, 0.0), 100.0) / 100.0


def split_groups(keys: Sequence, group_size: Optional[int]) -> List[List[int]]:
    """Row indices of the GRPO groups: consecutive blocks of `group_size` rows; a block is further split wherever the key (prompt identity: iso + source) changes,
    and a last block may be shorter (ragged batch). `group_size` None / < 2 -> every row alone (no consensus possible)."""
    n = len(keys)
    if not group_size or group_size < 2:
        return [[i] for i in range(n)]
    groups: List[List[int]] = []
    for s in range(0, n, group_size):
        cur: List[int] = []
        for i in range(s, min(s + group_size, n)):
            if cur and keys[i] != keys[cur[-1]]:
                groups.append(cur)
                cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
    return groups
