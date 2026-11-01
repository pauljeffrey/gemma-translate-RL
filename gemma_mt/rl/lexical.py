# SPDX-License-Identifier: AGPL-3.0-or-later
"""Lexical fidelity of a translation to its English source: the second opinion next to AfriCOMET-QE (owner 2026-10-13, D-25).

AfriCOMET-QE is nearly blind to two things in Pidgin and weak in the other languages: text that is fluent but about something
else (human reference vs mismatched reference: QE ranks the human one first only 76-81% of the time) and text appended to a
correct translation (47% = chance). A translation, however, has to carry over what does not change between languages:
numbers, names of people / places / organisations, and (for Pidgin, whose vocabulary is largely English) the content words.

    anchor_recall       share of the source's numbers and mid-sentence capitalised words found in the output
    anchor_precision    share of the output's numbers and mid-sentence capitalised words that come from the source
                        (1.0 when the output has none)
    cognate_recall      share of the source's content words (>= 4 letters, not a stop word) found in the output
    cognate_precision   share of the output's content words that come from the source (hallucinated / unrelated text lowers it)

The reward uses them only as a PENALTY (never a bonus), so copying the source or stuffing it with its own words cannot buy
reward: `deficit` is 0 when every enabled feature is at or above the language's human 10th percentile, and rises linearly
to 1 when it is 0.  Which features are enabled per language is decided by the calibration (human references of that
language must actually carry the overlap, `min_median`).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional, Sequence, Set

STOP = frozenset("""a an the and or but if then than so as of to in on at by for with from into onto over under about after before
between through during without within is are was were be been being am do does did done have has had having will would shall should
can could may might must not no nor yes i me my mine we us our ours you your yours he him his she her hers it its they them their
theirs this that these those there here who whom whose which what when where why how all any both each few more most other some such
only own same too very just also again further once up down out off while because until against above below""".split())

FEATURES = ("anchor_recall", "anchor_precision", "cognate_recall", "cognate_precision")
# cognate_recall is reported but never penalised: a Pidgin paraphrase legitimately drops source words (its fitted weight was negative)
PENALTY_FEATURES = ("anchor_recall", "anchor_precision", "cognate_precision")
# a feature is "enabled" for a language when the human references' median is at least this (they actually carry the overlap)
MIN_MEDIAN = {"anchor_recall": 0.50, "anchor_precision": 0.50, "cognate_recall": 0.25, "cognate_precision": 0.25}
MIN_CONTENT_WORDS = 3          # cognate features need this many content words in the shorter side to mean anything

_WORD = re.compile(r"[A-Za-z0-9']+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _tokens(s: str):
    return _WORD.findall(s)


def anchors(s: str) -> Set[str]:
    """numbers and capitalised words that are not sentence-initial (names, places): what a translation carries over unchanged."""
    out: Set[str] = set()
    for sent in _SENT.split(s):
        for k, w in enumerate(_tokens(sent)):
            if any(c.isdigit() for c in w):
                out.add(w.lower())
            elif k > 0 and w[:1].isupper() and len(w) > 1 and not w.isupper():
                out.add(w.lower())
    return out


def content_words(s: str) -> Set[str]:
    return {w.lower() for w in _tokens(s) if len(w) >= 4 and w.lower() not in STOP and not any(c.isdigit() for c in w)}


def features(src: str, mt: str) -> Dict[str, Optional[float]]:
    """the four features; None where the side has too little to say anything (so no penalty is derived from it)."""
    a_s, a_m = anchors(src), anchors(mt)
    c_s, c_m = content_words(src), content_words(mt)
    f: Dict[str, Optional[float]] = {
        "anchor_recall": len(a_s & a_m) / len(a_s) if a_s else None,
        "anchor_precision": len(a_m & a_s) / len(a_m) if a_m else 1.0,
        "cognate_recall": None, "cognate_precision": None,
    }
    if len(c_s) >= MIN_CONTENT_WORDS and len(c_m) >= MIN_CONTENT_WORDS:
        f["cognate_recall"] = len(c_s & c_m) / len(c_s)
        f["cognate_precision"] = len(c_m & c_s) / len(c_m)
    return f


def deficit(feats: Dict[str, Optional[float]], thresholds: Dict[str, float]) -> float:
    """mean over the enabled features of max(0, t - f) / t  (0 = every feature at / above its human 10th percentile, 1 = all zero)."""
    ds = []
    for k, t in thresholds.items():
        v = feats.get(k)
        if v is None or t <= 0:
            continue
        ds.append(max(0.0, t - v) / t)
    return float(sum(ds) / len(ds)) if ds else 0.0


def summarise(rows: Iterable[Sequence[str]], q_low: float = 0.10) -> Dict[str, Dict[str, float]]:
    """rows of (english, human reference) -> {feature: {p10, p50}} for the features the references actually carry (median >= MIN_MEDIAN)."""
    import numpy as np

    cols: Dict[str, list] = {k: [] for k in FEATURES}
    for s, r in rows:
        f = features(s, r)
        for k in FEATURES:
            if f[k] is not None:
                cols[k].append(f[k])
    out: Dict[str, Dict[str, float]] = {}
    for k, v in cols.items():
        if k not in PENALTY_FEATURES or len(v) < 30:
            continue
        p50 = float(np.median(v))
        if p50 >= MIN_MEDIAN[k]:
            out[k] = {"p10": float(np.quantile(v, q_low)), "p50": p50, "n": len(v)}
    return out
