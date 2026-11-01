# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text normalisation and 64-bit hashing used by every join and every leak filter of the inter-African data.

Two normal forms:
  * `pivot_key(text)`  NFC, casefold, every run of characters that is not a letter, digit or combining mark -> one space. Used for the JOIN
    on the English (or French) sentence, so "Hello, World!" and "hello world" are the same sentence.
  * `exact_key(text)`  NFC, casefold, whitespace collapsed (punctuation kept). Used for exact-duplicate removal of target texts.

Combining marks are kept in the pivot key on purpose: Yoruba / Igbo text (dot-below + tone mark) has no precomposed form for many letters, so a
`\\W` class that dropped marks would cut words in two.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, List, Sequence

_NON_WORD = re.compile(r"[^\ẁ-ͯ᪰-᫿᷀-᷿⃐-⃿︠-︯]+|_+")
_WS = re.compile(r"\s+")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def norm_text(text: str) -> str:
    """NFC + collapsed whitespace + stripped (case and punctuation kept): the form written to the training data."""
    return _WS.sub(" ", nfc(text)).strip()


def exact_key(text: str) -> str:
    return _WS.sub(" ", nfc(text).casefold()).strip()


def pivot_key(text: str) -> str:
    # U+0329 (combining vertical line below) is how some Yoruba corpora (MAFAND) write the under-dot that FLORES+ writes as U+0323
    return _NON_WORD.sub(" ", nfc(text).replace("\u0329", "\u0323").casefold()).strip()


def hash64(text: str) -> int:
    """Stable 64-bit hash (blake2b, little endian) of a string."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def words(text: str) -> List[str]:
    return pivot_key(text).split()


def shingles(text: str, n: int = 8) -> List[int]:
    """Hashes of the word n-grams of `text` (empty when it has fewer than n words)."""
    w = words(text)
    if len(w) < n:
        return []
    return [hash64(" ".join(w[i:i + n])) for i in range(len(w) - n + 1)]


def jaccard_words(a: str, b: str) -> float:
    sa, sb = set(words(a)), set(words(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def split_key_fraction(key: str, seed: int = 0, salt: str = "afri2afri-split-v1") -> float:
    """Deterministic uniform number in [0, 1) from (salt, seed, key): the train / validation split of the pivot groups."""
    d = hashlib.blake2b(f"{salt}|{seed}|{key}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(d, "big") / 2.0 ** 64
