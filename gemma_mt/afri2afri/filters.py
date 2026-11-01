# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quality filters of one X<->Y pair (numpy / pure python; no model is loaded here).

  * `lid_policy` / `lid_verdict`   GlotLID check of ONE side against the language's accepted labels. The accepted labels, the per-language "enabled" flag
    and the minimum text length are the MEASURED ones of the reward (`calibration/*.json`, block `lid`, D-37): a language whose LID is unreliable
    (pcm, kon) is not checked, a short text is not checked. Verdicts: LID_OK / LID_BAD / LID_SKIP.
  * `expected_log_ratio` / `ratio_ok`   length-ratio window. The expected character-length ratio of (src, tgt) is the ratio of the two languages'
    median "characters per English character" (taken from the en->X index, so Ethiopic text is not compared with Latin text as if both were Latin);
    a pair is kept when its log ratio is within `tol` of the expectation (default e^0.8 = 2.2x).
  * `is_copy`   the two sides are identical after normalisation, or share >= 0.9 of their word set (names, numbers, URLs).
  * `basic_ok`  length bounds, letter share, no URL, no markup debris -- the same rules `gemma_sft_pairs.ok()` applied to the English side.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from gemma_mt.afri2afri.text import exact_key, jaccard_words

LID_OK, LID_BAD, LID_SKIP = 1, 0, 2
_URL = re.compile(r"https?://|www\.|\.com/|@\w+\.", re.I)
_DEBRIS = re.compile(r"[{}<>\\\[\]]{3,}|&[a-z]+;|\b(?:lorem ipsum|click here|subscribe)\b", re.I)


def lid_policy(calibration: dict, fallback: Optional[Dict[str, Sequence[str]]] = None) -> Dict[str, Tuple[frozenset, int, bool]]:
    """{iso: (accepted labels, min_chars, enabled)} from a calibration file's per-language `lid` blocks; `fallback` = languages.py glotlid."""
    out: Dict[str, Tuple[frozenset, int, bool]] = {}
    for iso, entry in calibration.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("lid"), dict):
            continue
        b = entry["lid"]
        labels = frozenset(list(b.get("labels") or ()) + list(b.get("extra_labels") or ()))
        out[iso] = (labels, int(b.get("min_chars") or 0), bool(b.get("enabled")) and bool(labels))
    for iso, labs in (fallback or {}).items():
        out.setdefault(iso, (frozenset(labs), 30, bool(labs)))
    return out


def lid_verdict(iso: str, text_len: int, label: str, policy: Dict[str, Tuple[frozenset, int, bool]]) -> int:
    """LID_OK if `label` is accepted for `iso`; LID_SKIP when the language has no reliable LID or the text is too short; else LID_BAD."""
    entry = policy.get(iso)
    if entry is None:
        return LID_SKIP
    labels, min_chars, enabled = entry
    if not enabled or text_len < min_chars:
        return LID_SKIP
    return LID_OK if label in labels else LID_BAD


def chars_per_english_char(enlen: np.ndarray, tlen: np.ndarray, iso: np.ndarray, n_lang: int, min_rows: int = 200) -> np.ndarray:
    """Median of tlen / enlen per language (float array of n_lang; NaN when a language has fewer than `min_rows` rows)."""
    out = np.full(n_lang, np.nan)
    ratio = tlen.astype(np.float64) / np.maximum(enlen, 1)
    for i in range(n_lang):
        r = ratio[iso == i]
        if len(r) >= min_rows:
            out[i] = float(np.median(r))
    return out


def expected_log_ratio(cpe: np.ndarray, src: int, tgt: int) -> float:
    a, b = cpe[src], cpe[tgt]
    if not (a > 0 and b > 0):
        return 0.0
    return math.log(b / a)


def ratio_ok(src_len: int, tgt_len: int, expected_log: float, tol: float = 0.8) -> bool:
    if src_len <= 0 or tgt_len <= 0:
        return False
    return abs(math.log(tgt_len / src_len) - expected_log) <= tol


def is_copy(src: str, tgt: str, jaccard: float = 0.9) -> bool:
    a, b = exact_key(src), exact_key(tgt)
    if a == b:
        return True
    return jaccard_words(src, tgt) >= jaccard


def basic_ok(text: str, min_chars: int = 3, max_chars: int = 2500, min_letter_share: float = 0.35) -> bool:
    n = len(text)
    if n < min_chars or n > max_chars:
        return False
    if _URL.search(text) or _DEBRIS.search(text):
        return False
    letters = sum(c.isalpha() for c in text)
    return letters / n >= min_letter_share


def script_ok(text: str, iso: str) -> bool:
    """Ethiopic languages must be mostly Ethiopic script, the others mostly NOT Ethiopic (catches swapped / mislabelled rows)."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 5:
        return True
    eth = sum(1 for c in letters if "ሀ" <= c <= "፿" or "ᎀ" <= c <= "᎟" or "ⶀ" <= c <= "⷟")
    share = eth / len(letters)
    return share >= 0.5 if iso in ("amh", "tir") else share <= 0.2
