# SPDX-License-Identifier: AGPL-3.0-or-later
"""Joint-length budget of UnifiedMetric scorers (SSA-COMET-MTL), as PURE functions (D-55).

AfriCOMET-QE-STL encodes the source and the translation SEPARATELY and truncates each at 510 tokens. A UnifiedMetric (SSA-COMET-MTL, reference-free) encodes ONE
joint sequence  <s> mt </s></s> src </s>  of at most 512 positions (comet/encoders/base.py `concat_sequences`: the parts are truncated at 510 each, then the joint
sequence is cut at 512), so a long translation silently pushes the END of the source (and the closing </s>) out of the window: the model then judges a half-source.

`split_pair` keeps every scored pair inside the joint budget instead: a pair whose content tokens `tok(src) + tok(mt)` exceed `JOINT_CONTENT_BUDGET` (= 512 - 4
special tokens) is cut, recursively at the middle of BOTH texts (sentence boundary nearest to the middle, else a word boundary), into pieces that fit; the caller scores the
pieces and averages them with `combine` (weights = source characters of the piece, the rule of `chunk_agg: source_length`). A pair that fits is returned untouched, so
the path of every normal (sentence-sized) chunk is bit-for-bit what it was without this module.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from gemma_mt.rl.segments import split_sentences

MAX_POSITIONS = 512
JOINT_SPECIALS = 4                                   # <s> mt </s> </s> src </s>
JOINT_CONTENT_BUDGET = MAX_POSITIONS - JOINT_SPECIALS    # 508 content tokens for src + mt together
MAX_DEPTH = 12                                       # 2^12 pieces: never reached by a real text, only a guard against a degenerate input


def _cut_index(units: Sequence[str], frac: float, joiner: str) -> int:
    """Index j in 1..len(units)-1 such that joiner.join(units[:j]) is closest to `frac` of the total length."""
    total = sum(len(u) for u in units) + max(0, len(units) - 1) * len(joiner)
    best, best_d, acc = 1, None, 0
    for j in range(1, len(units)):
        acc += len(units[j - 1]) + (len(joiner) if j > 1 else 0)
        d = abs(acc / max(total, 1) - frac)
        if best_d is None or d < best_d:
            best, best_d = j, d
    return best


def cut_text(text: str, frac: float = 0.5) -> Tuple[str, str]:
    """(left, right): `text` cut near the character fraction `frac`, at a sentence boundary when there are two or more sentences, else at a space, else in the
    middle of the string. Whitespace is normalised to single spaces. A text that cannot be cut (< 2 characters) returns (text, "")."""
    sents = split_sentences(text)
    if len(sents) >= 2:
        j = _cut_index(sents, frac, " ")
        return " ".join(sents[:j]), " ".join(sents[j:])
    words = (text or "").split()
    if len(words) >= 2:
        j = _cut_index(words, frac, " ")
        return " ".join(words[:j]), " ".join(words[j:])
    t = (text or "").strip()
    if len(t) >= 2:
        k = max(1, min(len(t) - 1, int(round(len(t) * frac))))
        return t[:k], t[k:]
    return t, ""


def split_pair(src: str, mt: str, tok_len: Callable[[str], int], budget: int = JOINT_CONTENT_BUDGET, _depth: int = 0) -> List[Tuple[str, str]]:
    """The pieces (src_i, mt_i) of one pair, each with tok_len(src_i) + tok_len(mt_i) <= budget (best effort: a piece that cannot be cut further is returned as is).
    `tok_len(text)` = content tokens WITHOUT special tokens. A pair that fits is returned as [(src, mt)]."""
    if tok_len(src) + tok_len(mt) <= budget or _depth >= MAX_DEPTH:
        return [(src, mt)]
    s1, s2 = cut_text(src, 0.5)
    m1, m2 = cut_text(mt, 0.5)
    if (not s2 and not m2) or (not s1 and not m1):                 # nothing left to cut on either side
        return [(src, mt)]
    return split_pair(s1, m1, tok_len, budget, _depth + 1) + split_pair(s2, m2, tok_len, budget, _depth + 1)


def piece_weights(pieces: Sequence[Tuple[str, str]]) -> List[float]:
    """Weights of the pieces of one pair: source characters (the `chunk_agg: source_length` rule); all zero (blank sources) -> plain mean."""
    w = [float(len(s)) for s, _ in pieces]
    return w if sum(w) > 0 else [1.0] * len(pieces)


def combine(scores: Sequence[float], weights: Sequence[float]) -> float:
    return float(sum(s * w for s, w in zip(scores, weights)) / sum(weights))


def plan_pieces(srcs: Sequence[str], mts: Sequence[str], tok_len_many: Callable[[Sequence[str]], List[int]], budget: int = JOINT_CONTENT_BUDGET,
                ) -> Tuple[List[List[Tuple[str, str]]], int]:
    """For a batch: `plans[i]` = pieces of pair i (a single (src, mt) when it fits) and the number of pairs that had to be split. `tok_len_many(texts)` tokenizes in one
    call (content tokens, no specials); only the pairs over budget are tokenized again piece by piece."""
    ls, lm = tok_len_many(list(srcs)), tok_len_many(list(mts))
    plans: List[List[Tuple[str, str]]] = []
    n_split = 0
    for s, m, a, b in zip(srcs, mts, ls, lm):
        if a + b <= budget:
            plans.append([(s, m)])
            continue
        pieces = split_pair(s, m, lambda t: tok_len_many([t])[0], budget)
        n_split += len(pieces) > 1
        plans.append(pieces)
    return plans, n_split
