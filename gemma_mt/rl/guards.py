# SPDX-License-Identifier: AGPL-3.0-or-later
"""Degenerate-output and language guards for the RL reward. Pure python (the LID model is injected)."""

from __future__ import annotations

import re
import unicodedata
import zlib
from typing import Callable, Optional, Sequence

_PUNCT = re.compile(r"[\W_]+", re.UNICODE)


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return " ".join(_PUNCT.sub(" ", s).split())


def is_copy_of_source(src: str, mt: str, jaccard_threshold: float = 0.9) -> bool:
    """True when the output IS the source (after case / punctuation normalisation) or shares >= threshold of its
    word set with it. Copying the input is the classic way to fool a reference-free QE metric."""
    a, b = normalise(src), normalise(mt)
    if not b:
        return False
    if a == b:
        return True
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) >= jaccard_threshold


def is_repetitive(mt: str, max_run: int = 4, min_unique_ratio: float = 0.3, min_words_for_ratio: int = 8,
                  min_compress_ratio: float = 0.18, min_bytes_for_compress: int = 80) -> bool:
    """Loops / stuttering: one word repeated > max_run times in a row, very few distinct words, or text that
    compresses far better than natural language does."""
    words = mt.split()
    if not words:
        return False
    run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        if run > max_run:
            return True
    if len(words) >= min_words_for_ratio and len(set(w.lower() for w in words)) / len(words) < min_unique_ratio:
        return True
    raw = mt.encode("utf-8")
    if len(raw) >= min_bytes_for_compress and len(zlib.compress(raw)) / len(raw) < min_compress_ratio:
        return True
    return False


def length_ratio_bad(src: str, mt: str, lo: float = 0.2, hi: float = 5.0) -> bool:
    """Character length of the translation vs the source outside [lo, hi]: omission or runaway output."""
    ls, lm = len(src.strip()), len(mt.strip())
    if ls == 0:
        return False
    r = lm / ls
    return r < lo or r > hi


_ROLE_TAIL = re.compile(r"\n[ \t]*(?:model|user|assistant)[ \t]*:?\s*$", re.IGNORECASE)


def has_role_tail(mt: str, src: str = "") -> bool:
    """True when the text ends with a bare role name on a line of its own (`...\\nmodel`, `...\\nuser:`): the model started
    writing the next turn's header before its end token. Turn MARKERS (`<start_of_turn>`...) are caught from the raw ids
    (`text.decode_completion`); this is for the case where only the plain role word is left. A source that itself ends
    with such a line (a translation of a dialogue script) is exempt, and a role word inside a sentence never matches."""
    return bool(_ROLE_TAIL.search(mt)) and not _ROLE_TAIL.search(src)


LidFn = Callable[[Sequence[str]], Sequence[str]]


def repeated_ngram_fraction(text: str, n: int = 4) -> float:
    """Share of the word n-grams of `text` (lower-cased, split on white space) that repeat an earlier n-gram: 0 for text without repetition, near 1 for a loop."""
    words = text.lower().split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i: i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def is_self_repetitive(text: str, n: int = 4, max_fraction: float = 0.35, min_words: int = 24) -> bool:
    """Near-loops that `is_repetitive` misses (a phrase repeated with small variations, a paragraph repeated inside the answer): more than `max_fraction`
    of the word n-grams repeat an earlier one. Used where the consensus signal is on (D-38): a majority of such samples would otherwise agree with each other.
    The threshold is set by measurement (human FLORES+ dev documents of 3-30 sentences: scripts/check_selfrep.py)."""
    return len(text.split()) >= min_words and repeated_ngram_fraction(text, n) > max_fraction


class GlotLID:
    """Kept for backward compatibility: the classifier now lives in `gemma_mt.rl.lid` (numpy-2-safe, batched, cached)."""

    def __new__(cls, *a, **kw):
        from gemma_mt.rl.lid import GlotLIDClassifier

        return GlotLIDClassifier(*a, **kw)
