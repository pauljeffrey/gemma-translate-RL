# SPDX-License-Identifier: AGPL-3.0-or-later
"""Known-alignment documents and corruptions for the long-document reward study (R8).

A *unit list* is [(text, src_idx)]: the translation of source sentence `src_idx` (None = text with no source sentence: a duplicate, a loop, text of another
document). Joining the units and re-splitting with the reward's splitter gives the output sentences the reward sees; `labels[j]` = the set of source indices
whose units overlap output sentence j - the ground truth of the alignment."""
from __future__ import annotations

import random
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from gemma_mt.rl.segments import split_sentences

Units = List[Tuple[str, Optional[int]]]


def units_from(texts: Sequence[str]) -> Units:
    return [(" ".join(t.split()), i) for i, t in enumerate(texts)]


def join_units(units: Units) -> str:
    return " ".join(t for t, _ in units if t)


def sentence_labels(units: Units) -> Tuple[str, List[str], List[FrozenSet[int]]]:
    """(text, output sentences, labels) after joining and re-splitting; a sentence that spans several units gets all their source indices."""
    parts, spans = [], []
    pos = 0
    for t, idx in units:
        if not t:
            continue
        if parts:
            pos += 1
        spans.append((pos, pos + len(t), idx))
        parts.append(t)
        pos += len(t)
    text = " ".join(parts)
    sents = split_sentences(text)
    labels: List[FrozenSet[int]] = []
    cur = 0
    for s in sents:
        k = text.find(s, cur)
        if k < 0:
            k = cur
        a, b = k, k + len(s)
        cur = b
        labels.append(frozenset(idx for (u0, u1, idx) in spans if idx is not None and u0 < b and u1 > a))
    return text, sents, labels


# ------------------------------------------------------------------------------------------------ corruptions
def drop(units: Units, k: int, rng: random.Random) -> Units:
    n = len(units)
    gone = set(rng.sample(range(n), min(k, n - 1)))
    return [u for i, u in enumerate(units) if i not in gone]


def drop_frac(units: Units, frac: float, rng: random.Random) -> Units:
    return drop(units, max(1, int(round(frac * len(units)))), rng)


def duplicate(units: Units, n_sent: int, rng: random.Random) -> Units:
    """A run of n_sent consecutive sentences is output twice in a row (an inserted copy has no source index)."""
    n = len(units)
    n_sent = min(n_sent, n)
    a = rng.randrange(0, n - n_sent + 1)
    run = units[a: a + n_sent]
    return units[: a + n_sent] + [(t, None) for t, _ in run] + units[a + n_sent:]


def loop_tail_sentences(units: Units, extra_frac: float = 0.3, period: int = 2, rng: Optional[random.Random] = None) -> Units:
    """The document ends with its last `period` sentences repeated until the copies add `extra_frac` of the document's characters."""
    base = sum(len(t) for t, _ in units)
    run = units[-period:]
    out = list(units)
    added = 0
    while added < extra_frac * base:
        for t, _ in run:
            out.append((t, None))
            added += len(t)
    return out


def loop_tail_words(units: Units, extra_frac: float = 0.3, period_words: int = 3) -> Units:
    """The last sentence runs into a word-cycle loop ('... muna muna muna'), adding extra_frac of the document's characters."""
    base = sum(len(t) for t, _ in units)
    words = units[-1][0].split()
    cyc = " ".join(words[-period_words:]) if len(words) >= period_words else (words[-1] if words else "a")
    add = []
    n = 0
    while n < extra_frac * base:
        add.append(cyc)
        n += len(cyc) + 1
    t, idx = units[-1]
    return units[:-1] + [(t + " " + " ".join(add), idx)]


def truncate(units: Units, keep_frac: float) -> Units:
    n = max(1, int(round(keep_frac * len(units))))
    return units[:n]


def replace_frac(units: Units, frac: float, other: Sequence[str], rng: random.Random) -> Units:
    """frac of the sentences are replaced by sentences of ANOTHER document in the same language (no source index)."""
    n = len(units)
    k = max(1, int(round(frac * n)))
    pos = sorted(rng.sample(range(n), min(k, n)))
    out = list(units)
    for p in pos:
        out[p] = (" ".join(rng.choice(other).split()), None)
    return out


CORRUPTIONS = ["none", "drop1", "drop3", "drop10", "dup1", "dup3", "loop_tail_sent", "loop_tail_word", "trunc80", "trunc50", "repl10", "repl30"]


def apply(name: str, units: Units, rng: random.Random, other: Sequence[str]) -> Units:
    if name == "none":
        return list(units)
    if len(units) < 3 and name != "none":
        return list(units)
    if name == "drop1":
        return drop(units, 1, rng)
    if name == "drop3":
        return drop(units, 3, rng)
    if name == "drop10":
        return drop_frac(units, 0.10, rng)
    if name == "dup1":
        return duplicate(units, 1, rng)
    if name == "dup3":
        return duplicate(units, 3, rng)
    if name == "loop_tail_sent":
        return loop_tail_sentences(units, 0.3, 2)
    if name == "loop_tail_word":
        return loop_tail_words(units, 0.3, 3)
    if name == "trunc80":
        return truncate(units, 0.8)
    if name == "trunc50":
        return truncate(units, 0.5)
    if name == "repl10":
        return replace_frac(units, 0.10, other, rng)
    if name == "repl30":
        return replace_frac(units, 0.30, other, rng)
    raise KeyError(name)
