# SPDX-License-Identifier: AGPL-3.0-or-later
"""Diacritics score of a translation chunk: a PENALTY for stripped or fabricated diacritics (owner 2026-10-17, D-31).

Yoruba tone marks and under-dots, Ewe / Fon / Akan open vowels and hooked consonants ... are part of the spelling, and a model that "saves
tokens" by dropping them still produces text AfriCOMET-QE rates highly. The reward therefore looks at the DIACRITIC DENSITY of every scored
chunk, as the human references of the language show it:

    units(text)   = number of combining marks (category Mn) + special letters (`SPECIAL`) in the NFD form of the text
    density(text) = units / number of NFD characters

The logic of `units` / `density` is a copy of data_pipeline/sft_new/text_restore.py (no import across packages: SPECIAL, is_mark, diacritic_units,
diacritic_density), so the data-side restoration tasks and this reward agree on what a diacritic is.

Calibration (`summarise`, called by calibrate_shaping on the HUMAN references of each language, FLORES+ dev) writes the block `diacritics`:
    applies    median human density >= MIN_DENSITY (0.008, the threshold text_restore uses to call a language a diacritic language)
    density    mean / p01 / p10 / p50 / p90 / p99 of the per-sentence human density
    allowed    the inventory the humans actually use: combining marks and special letters with their counts (hex code points), and the
               (base letter + marks) clusters
The reward (`deficit`) applies ONLY to languages with `applies` and only to chunks whose English source has >= MIN_CONTENT_WORDS content words:
    deficit = max(0, t - d) / t                      t = `threshold` (the human P10; see `summarise` for the two languages-types where P10 is 0),
                                                     d = density of the chunk (so a long output cannot hide stripped chunks)
    deficit = 1.0                                    anti-hacking guards: a mark / special letter outside the language's inventory, a combining
                                                     mark that is not attached to a letter, a (letter, marks) cluster no human reference has,
                                                     or a density above the human p99 (over-decoration)
and is subtracted like the lexical penalty (`diacritic_penalty`, never below the failure floor). It never adds reward.

`diacritic_report(text, lang)` is the plain, reusable form (the future "best translation archive" can call it): density, deficit, flags.
"""

from __future__ import annotations

import json
import os
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# letters that are "diacritics" for restoration purposes: a plain-keyboard writer types the base letter instead (copy of text_restore.SPECIAL)
SPECIAL = {"ɛ": "e", "ɔ": "o", "ɖ": "d", "ƒ": "f", "ɣ": "g", "ŋ": "n", "ʋ": "v", "ɓ": "b", "ɗ": "d", "ƙ": "k", "ɠ": "g", "ɲ": "n",
           "ə": "e", "ɩ": "i", "ʊ": "u", "ɨ": "i", "ʉ": "u", "ɱ": "m", "ʃ": "s", "ʒ": "z", "ɦ": "h",
           "Ɛ": "E", "Ɔ": "O", "Ɖ": "D", "Ƒ": "F", "Ɣ": "G", "Ŋ": "N", "Ʋ": "V", "Ɓ": "B", "Ɗ": "D", "Ƙ": "K", "Ɠ": "G", "Ɲ": "N",
           "Ə": "E", "Ɩ": "I", "Ʊ": "U", "’": "'", "ʼ": "'"}
_PUNCT_LIKE = frozenset("’ʼ")   # apostrophes are in SPECIAL (the restoration tasks fold them) but are punctuation: never a "foreign letter"
MIN_DENSITY = 0.008          # a language "uses diacritics" when a full strip changes >= 0.8% of its characters (text_restore.MIN_DENSITY)
MIN_CONTENT_WORDS = 3        # the English chunk must carry at least this many content words (rl/lexical.py), otherwise nothing is judged
MIN_SAMPLES = 30             # sentences needed to calibrate a language
MIN_COUNT = 1                # a mark / letter / cluster enters the inventory when it occurs at least this often in the references
MIN_GUARD_CHARS = 25         # density guards (stripped / over-decorated) need at least this many NFD characters in the chunk


def is_mark(c: str) -> bool:
    return unicodedata.category(c) == "Mn"


def _nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def diacritic_units(text: str) -> int:
    """How many characters a full strip would remove or fold (combining marks + special letters), on the NFD form."""
    return sum(1 for c in _nfd(text) if is_mark(c) or c in SPECIAL)


def diacritic_density(text: str) -> float:
    return diacritic_units(text) / max(len(_nfd(text)), 1)


def _hex(c: str) -> str:
    return f"{ord(c):04X}"


def clusters(text: str) -> List[Tuple[str, Tuple[str, ...]]]:
    """(base character, marks attached to it) for every character of the NFD text that has at least one mark; a mark with nothing before it
    (start of text) has base ''."""
    out: List[Tuple[str, Tuple[str, ...]]] = []
    base, marks = "", []
    for c in _nfd(text):
        if is_mark(c):
            marks.append(c)
            continue
        if marks:
            out.append((base, tuple(marks)))
        base, marks = c, []
    if marks:
        out.append((base, tuple(marks)))
    return out


def _cluster_key(base: str, marks: Sequence[str]) -> str:
    return base.lower() + "|" + ",".join(_hex(m) for m in marks)


def inventory(text: str) -> Dict[str, Counter]:
    """What a text uses: combining marks, special letters, (letter|marks) clusters. Keys are hex code points / cluster keys."""
    nfd = _nfd(text)
    marks = Counter(_hex(c) for c in nfd if is_mark(c))
    letters = Counter(_hex(c) for c in nfd if c in SPECIAL)
    cl = Counter(_cluster_key(b, m) for b, m in clusters(text))
    return {"marks": marks, "letters": letters, "clusters": cl}


def _quantiles(v: Sequence[float]) -> Dict[str, float]:
    s = sorted(v)
    q = lambda p: float(s[min(len(s) - 1, int(round(p * (len(s) - 1))))])
    return {"mean": float(sum(v) / len(v)), "p01": q(0.01), "p10": q(0.10), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99)}


DENSE_MIN = 0.05             # a language whose MARKED sentences have a median density >= this cannot have zero marks by chance (Yoruba 0.24, Fon 0.21, Ibo 0.09)


def summarise(refs: Iterable[str], min_density: float = MIN_DENSITY, extra_refs: Iterable[str] = ()) -> Optional[Dict[str, Any]]:
    """Calibration block `diacritics` from the human references of one language (None when there are too few). Text is taken as NFC
    (the reward path normalises to NFC) and measured on its NFD form. `extra_refs` (other dev sets of the language, e.g. MAFAND news) only
    WIDEN the allowed inventory: Yoruba MAFAND writes the under-dot as U+0329, which FLORES+ never does.

    `threshold` is the density below which `deficit` starts: the human P10 (the owner's rule) wherever it is > 0. Where it is 0 the density of a
    single sentence is either uninformative (Hausa, Kinyarwanda, Tumbuka: ~1% of the characters, whole sentences without any special letter are
    normal -> `threshold` 0 = no density penalty, the guards still apply) or the references themselves are a MIX of marked and unmarked text
    (FLORES+ Yoruba: 15% of the sentences carry no tone mark at all although the others are at 0.24) -> `threshold` = P10 of the marked sentences."""
    refs = [unicodedata.normalize("NFC", r) for r in refs if r and r.strip()]
    if len(refs) < MIN_SAMPLES:
        return None
    dens = [diacritic_density(r) for r in refs]
    q = _quantiles(dens)
    inv = {"marks": Counter(), "letters": Counter(), "clusters": Counter()}
    for r in list(refs) + [unicodedata.normalize("NFC", r) for r in extra_refs if r and r.strip()]:
        for k, c in inventory(r).items():
            inv[k].update(c)
    units = sum(diacritic_units(r) for r in refs)
    chars = sum(len(_nfd(r)) for r in refs)
    pos = sorted(d for d in dens if d > 0)
    dense = bool(pos) and pos[len(pos) // 2] >= DENSE_MIN
    if q["p10"] > 0:
        thr, src = q["p10"], "p10"
    elif dense:
        thr, src = _quantiles(pos)["p10"], "p10_of_marked_sentences"
    else:
        thr, src = 0.0, "disabled"
    return {"applies": bool(q["p50"] >= min_density), "n": len(refs), "density": q, "corpus_density": units / max(chars, 1),
            "zero_share": 1.0 - len(pos) / len(dens), "threshold": thr, "threshold_source": src,
            "allowed": {k: {h: n for h, n in sorted(c.items(), key=lambda kv: -kv[1]) if n >= MIN_COUNT} for k, c in inv.items()}}


# ------------------------------------------------------------------------------------------------ scoring
def flags(text: str, entry: Dict[str, Any], use_clusters: bool = True) -> Dict[str, bool]:
    """The anti-hacking guards for one chunk against the language's calibration `entry`."""
    nfd = _nfd(text)
    allowed = entry.get("allowed", {})
    ok_marks, ok_letters, ok_clusters = allowed.get("marks", {}), allowed.get("letters", {}), allowed.get("clusters", {})
    foreign = False
    orphan = False
    bad_cluster = False
    for c in nfd:
        if is_mark(c) and _hex(c) not in ok_marks:
            foreign = True
        elif c in SPECIAL and c not in _PUNCT_LIKE and _hex(c) not in ok_letters:
            foreign = True
    for b, ms in clusters(text):
        if not b or not (b.isalpha() or b in SPECIAL):
            orphan = True
        elif use_clusters and ok_clusters and _cluster_key(b, ms) not in ok_clusters:
            bad_cluster = True
    dens = diacritic_density(text)
    over = len(nfd) >= MIN_GUARD_CHARS and dens > float(entry.get("density", {}).get("p99", 1.0))
    return {"foreign_mark": foreign, "orphan_mark": orphan, "unseen_cluster": bad_cluster, "over_decorated": over}


def deficit(text: str, entry: Optional[Dict[str, Any]], use_clusters: bool = True) -> float:
    """0..1 penalty weight of one chunk (see the module docstring); 0 for a language that does not use diacritics / has no calibration."""
    if not entry or not entry.get("applies"):
        return 0.0
    if any(flags(text, entry, use_clusters).values()):
        return 1.0
    thr = float(entry.get("threshold", entry.get("density", {}).get("p10", 0.0)))
    if thr <= 0.0 or len(_nfd(text)) < MIN_GUARD_CHARS:
        return 0.0
    return float(max(0.0, thr - diacritic_density(text)) / thr)


def applicable(src: str) -> bool:
    """Source-aware rule: the English chunk must have >= MIN_CONTENT_WORDS content words."""
    from gemma_mt.rl import lexical

    return len(lexical.content_words(src)) >= MIN_CONTENT_WORDS


# default calibration: the file shipped with the repo (calibration/shaping_flores_dev_v4.json), loaded lazily
_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "calibration", "shaping_flores_dev_v4.json")
_CACHE: Dict[str, Dict[str, Any]] = {}


def load_blocks(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """language -> `diacritics` block of a calibration file (default: the repo's calibration/shaping_flores_dev_v4.json); {} when missing."""
    path = path or _DEFAULT_PATH
    if path not in _CACHE:
        blocks: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    if isinstance(v, dict) and v.get("diacritics"):
                        blocks[k.split(">")[-1].strip()] = v["diacritics"]
        _CACHE[path] = blocks
    return _CACHE[path]


def diacritic_report(text: str, lang: str, stats: Optional[Dict[str, Dict[str, Any]]] = None, src: Optional[str] = None) -> Dict[str, Any]:
    """Plain report for one text: {lang, applies, density, p10, p99, deficit, flags, units, chars}. `stats` = language -> `diacritics` block (default:
    the repo calibration file). With `src` (the English text) the source-aware rule is applied (deficit 0 when it has < 3 content words)."""
    stats = load_blocks() if stats is None else stats
    entry = stats.get(lang)
    text = unicodedata.normalize("NFC", text)
    nfd = _nfd(text)
    rep: Dict[str, Any] = {"lang": lang, "applies": bool(entry and entry.get("applies")), "chars": len(nfd), "units": diacritic_units(text),
                           "density": diacritic_density(text), "p10": None, "threshold": None, "p99": None, "flags": {}, "deficit": 0.0}
    if entry:
        rep["p10"], rep["p99"] = entry["density"]["p10"], entry["density"]["p99"]
        rep["threshold"] = entry.get("threshold", rep["p10"])
        rep["flags"] = flags(text, entry)
        d = deficit(text, entry)
        if src is not None and not applicable(src):
            d = 0.0
        rep["deficit"] = d
    return rep
