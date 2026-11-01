# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sentence segmentation used by the chunked reward (owner 2026-10-12): the source window and the generated translation are
split at sentence ends, i-th source sentence is scored against the i-th output sentence, and the chunk scores are averaged.

No newline convention is involved: the model never has to emit (or receive) line breaks, a newline inside the text is
ordinary white space. The boundary is a full stop; the ASCII `?` and `!` and the Ethiopic full stop `።` (Amharic,
Tigrinya) are boundaries too, because the same text may contain them in both languages.

A terminator ends a sentence only when
  * it is followed by white space or the end of the text (so `3.5`, `www.example.org`, `a.b` never split),
  * the next word does not start with a lowercase letter (abbreviation continuations: `e.g. the`, `etc. and`),
  * the word it ends is not a known abbreviation / a single capital initial (`Dr.`, `St.`, `J.`, `U.S.`),
  * it is not a list number (`1.`, `12.`).
Closing quotes / brackets directly after the terminator stay with the sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

_TERMINATORS = ".!?።"
_CLOSERS = "\"'”’»)]}"
_ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "e.g", "i.e", "no", "nos", "fig", "figs", "inc",
           "ltd", "co", "corp", "gen", "col", "capt", "lt", "sgt", "rev", "hon", "gov", "pres", "dept", "est", "approx", "ca", "cf",
           "al", "vol", "vols", "pp", "ed", "eds", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}
_CANDIDATE = re.compile(r"([.!?።]+)([\"'”’»)\]}]*)(\s+|$)")
_LAST_WORD = re.compile(r"([^\s]+)$")
_LAST_WORD_SPAN = 256        # characters looked at before a terminator; a word that long is no abbreviation, initial or list number, so the decision cannot change


def _is_boundary(text: str, start: int, end_term: int, term: str, next_pos: int) -> bool:
    """Decide whether the terminator run text[start:end_term] closes a sentence."""
    if next_pos < len(text):
        nxt = text[next_pos]
        if nxt.isalpha() and nxt.islower() and "።" not in term:
            return False                                           # `etc. and ...`, `e.g. the ...`
    head = text[:start]
    m = _LAST_WORD.search(head, max(0, start - _LAST_WORD_SPAN))     # D-45: the whole `head` was scanned for every candidate (quadratic in the text: ~8 ms for a b2048 window, 0.1 s for a document)
    word = m.group(1) if m else ""
    bare = word.strip("\"'“‘(«[{").lower()
    if term == ".":
        if bare in _ABBREV or bare.rstrip(".") in _ABBREV:
            return False
        if re.fullmatch(r"[A-Za-z]", bare or "x") and word[:1].isupper():  # single capital initial: `J. Smith`
            return False
        if re.fullmatch(r"(?:[A-Za-z]\.){2,}", bare + "."):               # `U.S.`, `e.g`, `i.e`, `p.m`
            return False
        if bare.isdigit() and len(bare) <= 3:
            before = head[: m.start(1)].rstrip()
            if not before or before[-1] in _TERMINATORS + ":;":
                return False                                       # list number that starts a segment: `1. Item`, `. 2. Item`
    return True


def split_sentences(text: str) -> List[str]:
    """Sentences of `text` in order, each stripped, with internal white space (incl. newlines) normalised to single spaces."""
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return []
    out: List[str] = []
    last = 0
    for m in _CANDIDATE.finditer(t):
        end = m.end(2)                                             # include closing quotes / brackets
        if _is_boundary(t, m.start(1), m.end(1), m.group(1), m.end()):
            seg = t[last:end].strip()
            if seg:
                out.append(seg)
            last = m.end()
    tail = t[last:].strip()
    if tail:
        out.append(tail)
    return out


# --------------------------------------------------------------------------------------------- length-based alignment
# Gale & Church (1993): dynamic programme over sentence "beads" with a character-length model. Used when the number of
# source and output sentences differs (merged / split / dropped sentences), so a pairing never shifts silently:
#   (1,1) one-to-one   (2,1) (3,1) (4,1) several source sentences translated as one   (1,2) (1,3) (1,4) the reverse
#   (2,2) (3,2) (2,3)  mixed                    (1,0) a source sentence with NO output (omission)   (0,1) an extra output sentence
_BEADS = {(1, 1): 0.89, (1, 0): 0.01, (0, 1): 0.01, (2, 1): 0.045, (1, 2): 0.045, (2, 2): 0.011, (3, 1): 0.006, (1, 3): 0.006,
          (3, 2): 0.002, (2, 3): 0.002, (4, 1): 0.001, (1, 4): 0.001}
_S2 = 6.8                                                    # variance of the length difference per source character


@dataclass(frozen=True)
class LengthModel:
    """Language-aware length model for `gale_church` (omission detector, D-30). The legacy model has no per-language constant: it uses
    the observed output/source ratio of the very pair it aligns, so a translation that DROPS sentences is simply "shorter overall" and the
    cheaper explanation is always a 2:1 merge of the neighbours (finding F2: an omission bead won in 2 of 20,000 trials). Here
      ratio       expected output characters per source character, measured on the language's human references (calibration `len_ratio.corpus`)
      s2          variance of the length difference per source character (calibration `len_ratio.s2`, default = Gale & Church's 6.8)
      omit_cost   cost in nats of a source sentence with no output (bead (1,0)); NOT length dependent: under the length model an empty
                  output is 4+ sigma away for every sentence, which is why the legacy cost never selects it
      extra_cost  same for an output sentence with no source (bead (0,1))
    """
    ratio: float
    s2: float = 6.8
    omit_cost: float = 4.0
    extra_cost: float = 4.0
    dp: bool = True            # False: keep the legacy alignment and use the model only for the shortfall check of chunk_plan
    directional: bool = True   # an omission bead only when the output has FEWER sentences than the source, an extra bead only when it has MORE


def _norm_sf(x: float) -> float:
    import math
    return 0.5 * math.erfc(x / math.sqrt(2.0))              # P(Z > x)


def _cost(l_src: int, l_out: int, c: float, prior: float, s2: float = _S2) -> float:
    import math
    if l_src == 0 and l_out == 0:
        return 1e9
    delta = (l_out - l_src * c) / math.sqrt(max(l_src, 1) * s2)
    p = max(2.0 * _norm_sf(abs(delta)), 1e-12)
    return -math.log(prior) - math.log(p)


def gale_church(src: Sequence[str], out: Sequence[str], model: Optional[LengthModel] = None
                ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Beads [(source indices, output indices), ...] covering both sequences in order. `c` (output chars per source char)
    is the observed ratio of the two texts, so no per-language constant is needed. Banded DP: |i*m/n - j| <= max(8, 0.25*max(n, m)).
    With `model` (a LengthModel with dp=True) `c` and the variance come from the language and the omission / extra beads cost a constant."""
    n, m = len(src), len(out)
    if n == 0 or m == 0:
        return [(tuple(range(n)), tuple(range(m)))] if (n or m) else []
    ls = [len(x) for x in src]
    lo = [len(x) for x in out]
    c = max(sum(lo) / max(sum(ls), 1), 1e-3)
    if model is not None:
        c = max(float(model.ratio), 1e-3)
    ps, po = [0], [0]
    for v in ls:
        ps.append(ps[-1] + v)
    for v in lo:
        po.append(po[-1] + v)
    band = max(8, int(0.25 * max(n, m)))
    INF = float("inf")
    best = {(0, 0): (0.0, None)}
    for i in range(n + 1):
        centre = i * m / n
        j_lo, j_hi = max(0, int(centre - band)), min(m, int(centre + band) + 1)
        for j in range(j_lo, j_hi + 1):
            if (i, j) == (0, 0):
                continue
            cand = INF
            arg = None
            for (di, dj), prior in _BEADS.items():
                pi, pj = i - di, j - dj
                if pi < 0 or pj < 0 or (pi, pj) not in best:
                    continue
                if model is None:
                    step = _cost(ps[i] - ps[pi], po[j] - po[pj], c, prior)
                elif di == 1 and dj == 0:
                    step = 1e9 if (model.directional and m > n) else model.omit_cost
                elif di == 0 and dj == 1:
                    step = 1e9 if (model.directional and n > m) else model.extra_cost
                else:
                    step = _cost(ps[i] - ps[pi], po[j] - po[pj], c, prior, model.s2)
                cost = best[(pi, pj)][0] + step
                if cost < cand:
                    cand, arg = cost, (pi, pj)
            if arg is not None:
                best[(i, j)] = (cand, arg)
    if (n, m) not in best:                                   # band too narrow for a degenerate input: one big bead
        return [(tuple(range(n)), tuple(range(m)))]
    beads, cur = [], (n, m)
    while cur != (0, 0):
        prev = best[cur][1]
        beads.append((tuple(range(prev[0], cur[0])), tuple(range(prev[1], cur[1]))))
        cur = prev
    return beads[::-1]


def align_chunks(source: str, translation: str, mode: str = "gale_church") -> Tuple[List[Tuple[str, str]], int, str]:
    """Scoring chunks of one (source, translation) pair.

    Returns (chunks, n_failed, status). `chunks` are (source text, output text) pairs to score (an empty output text marks an
    omitted source sentence, which gets the failure score); `n_failed` counts omitted source sentences. `status`:
      single       one source sentence: the whole pair is one chunk
      aligned      equal sentence counts: plain i-th-to-i-th pairing
      gale_church  counts differ: length-based alignment (many-to-one beads, omissions)
      whole        counts differ and the output is a single sentence (or mode == "whole"): one chunk for the whole text"""
    s, o = split_sentences(source), split_sentences(translation)
    if len(s) < 2:
        return [(source.strip(), translation.strip())], 0, "single"
    if len(s) == len(o):
        return list(zip(s, o)), 0, "aligned"
    if mode == "whole" or len(o) <= 1:                       # one output sentence for several source sentences: score it as a whole
        return [(" ".join(s), " ".join(o))], 0, "whole"
    chunks, failed = [], 0
    for si, oi in gale_church(s, o):
        if si and oi:
            chunks.append((" ".join(s[i] for i in si), " ".join(o[j] for j in oi)))
        elif si:                                             # omitted source sentence(s)
            chunks.append((" ".join(s[i] for i in si), ""))
            failed += len(si)
    return chunks, failed, "gale_church"


def align_beads(source: str, translation: str, mode: str = "gale_church", model: Optional[LengthModel] = None
                ) -> Tuple[List[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]]], str]:
    """Like `align_chunks`, but every chunk also says which sentences each side holds: [(source text, output text, source sentence
    indices, output sentence indices)] plus the status. An omitted source sentence has no output indices ("" text); an extra output
    sentence has no source indices (`align_chunks` drops those). `model` selects the language-aware Gale-Church costs (LengthModel);
    without it the alignment is the legacy one."""
    s, o = split_sentences(source), split_sentences(translation)
    if len(s) < 2:
        return [(source.strip(), translation.strip(), tuple(range(len(s))), tuple(range(len(o))))], "single"
    if len(s) == len(o):
        return [(a, b, (i,), (i,)) for i, (a, b) in enumerate(zip(s, o))], "aligned"
    if mode == "whole" or len(o) <= 1:
        return [(" ".join(s), " ".join(o), tuple(range(len(s))), tuple(range(len(o))))], "whole"
    beads = gale_church(s, o) if (model is None or not model.dp) else gale_church(s, o, model)
    return [(" ".join(s[i] for i in si), " ".join(o[j] for j in oi), tuple(si), tuple(oi)) for si, oi in beads], "gale_church"


def align(source: str, translation: str) -> Tuple[List[str], List[str], bool]:
    """(source sentences, output sentences, aligned). `aligned` = same number of sentences, at least two: only then the
    i-th source sentence is paired with the i-th output sentence."""
    s, o = split_sentences(source), split_sentences(translation)
    return s, o, (len(s) == len(o) and len(s) >= 2)
