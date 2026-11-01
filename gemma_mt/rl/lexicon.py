# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dictionary-overlap signal (D-57): does the translation contain the words a bilingual lexicon expects for the source's content words?

The AfriCOMET signal is missing (23 consensus-only languages) or weak (22 blend languages) for ~45 of the 77 target languages. The owner proposed
a dictionary-overlap check: build an English -> X lexicon from the English-pivot pairs of `<HF_ORG_LEGACY>/interlaanguage-translation` and measure
the share of the source's content words whose X translation occurs in the output. This module is the RUNTIME half (stdlib only); the builder is
`scripts/build_lexicons.py`, the lexicons are `lexicon/<iso>.json` (next to `calibration/`; override the directory with $GEMMA_MT_LEXICON_DIR), the
validation (which languages may use the signal) is `reports/lexicon_overlap_2026-10-21.md`.

Hook contract (rl/anchors.py of agent B2):

    from gemma_mt.rl.lexicon import lexicon_overlap, lexicon_available
    if lexicon_available(iso):                     # a lexicon exists AND the validation passed (meta.validated); validated_only=False: file exists
        r = lexicon_overlap(iso, english_source, translation)
        r.overlap      share (0..1, IDF-weighted) of the source's lexicon-covered content words with a candidate in the output; None when fewer
                       than MIN_COVERED words are covered (the signal says nothing about this sentence: do not reward, do not punish)
        r.coverage     covered content words / content words of the source
        r.n_content, r.n_covered, r.n_hit

A RECALL signal only: appending text, or stuffing, cannot lower it, so it is used next to precision guards (length window, LID, consensus), never alone.

Normalisation (build and run time are the same code): NFC, lowercase, apostrophes folded, whitespace tokens with edge punctuation stripped; tone / accent marks are
FOLDED for the comparison (the dot below stays: ọ is not o) (the lexicon keeps the surface variants in `v`, the diacritics reward is a separate term, rl/diacritics.py). English words are
reduced by `en_stem` (light suffix stripping; the stem is only a key).
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

MIN_COVERED = 2            # fewer lexicon-covered content words than this: overlap is None (nothing to judge)
MIN_STEM_LEN = 4           # affix matching: the shorter of candidate / token must have at least this many letters ...
MIN_STEM_RATIO = 0.6       # ... and at least this share of the longer one (so a 4-letter stem matches at most a 6-letter token)
DEFAULT_TOPK = 3
DEFAULT_MIN_CONF = 0.0     # candidates below this confidence are ignored (the lexicon files already carry the builder's floor)
MAX_IDF = 8.0

# ------------------------------------------------------------------------------------------------------ English side
STOP: FrozenSet[str] = frozenset("""a an the and or but if then than so as of to in on at by for with from into onto over under about after before
between through during without within is are was were be been being am do does did done doing have has had having will would shall should can could
may might must not no nor yes i me my mine myself we us our ours ourselves you your yours yourself he him his himself she her hers herself it its itself
they them their theirs themselves this that these those there here who whom whose which what when where why how all any both each few more most other
some such only own same too very just also again further once up down out off while because until against above below
said say says saying get gets got getting make makes made making take takes took taken go goes went gone going come comes came let lets
however therefore thus whether either neither another every across along around among upon toward towards although though unless since whereas
already yet still even ever never often always really quite rather perhaps maybe per via etc thing things something anything nothing everything
one two three four five six seven eight nine ten first second third many much several able like likely according""".split())

_EN_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
_VOWELS = "aeiou"


def en_stem(w: str) -> str:
    """Light suffix stripping of an English word. NOT linguistically exact: it only has to map the inflected forms of a word to the same key
    (houses/house -> hous, walking/walked/walk -> walk) at build time and at run time."""
    w = w.lower().replace("’", "'")
    if w.endswith("'s"):
        w = w[:-2]
    n = len(w)
    if n <= 3:
        return w
    if w.endswith("ies") and n > 4:
        w = w[:-3] + "y"
    elif w.endswith("ied") and n > 4:
        w = w[:-3] + "y"
    elif w.endswith(("sses", "xes", "ches", "shes", "zes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    elif w.endswith("ing") and n > 5 and any(c in _VOWELS for c in w[:-3]):
        w = w[:-3]
        if len(w) >= 3 and w[-1] == w[-2] and w[-1] not in "lsz":
            w = w[:-1]
    elif w.endswith("ed") and n > 4 and any(c in _VOWELS for c in w[:-2]):
        w = w[:-2]
        if len(w) >= 3 and w[-1] == w[-2] and w[-1] not in "lsz":
            w = w[:-1]
    if w.endswith("e") and len(w) > 3:
        w = w[:-1]
    return w


def english_content_stems(text: str) -> List[str]:
    """Unique stems (in order of first occurrence) of the content words of an English sentence: alphabetic, >= 3 letters, not a stop word, and not a
    mid-sentence capitalised word (names are carried over unchanged, rl/lexical.py anchors judge them). The first word of a sentence is lowercased."""
    out: List[str] = []
    seen: Set[str] = set()
    text = unicodedata.normalize("NFC", text)
    for m in _EN_WORD.finditer(text):
        w = m.group(0)
        prev = text[:m.start()].rstrip()
        start = (not prev) or prev[-1] in ".!?:;\"\u201c\u201d\u2018(["
        if w[0].isupper() and (not start or (len(w) > 1 and w.isupper())):
            continue
        lw = w.lower().strip("'-’")
        if len(lw) < 3 or lw in STOP:
            continue
        st = en_stem(lw)
        if len(st) < 3 or st in STOP or st in seen:
            continue
        seen.add(st)
        out.append(st)
    return out


# ------------------------------------------------------------------------------------------------------ X side
_APOS = str.maketrans({"’": "'", "ʼ": "'", "‘": "'", "`": "'", "´": "'"})


# combining marks that are PART OF THE LETTER in several of our orthographies (dot below: Yoruba / Edo / Urhobo / Esan / Okpe / Efik ọ ẹ ṣ, ogonek, cedilla,
# ring/diaeresis below) are kept; tone and stress accents (grave, acute, circumflex, tilde, macron, breve, diaeresis, caron, double grave ...) are folded
_KEEP_MARKS = frozenset("\u0323\u0324\u0325\u0327\u0328")


def fold(s: str) -> str:
    """NFC lowercase, apostrophes folded, tone / accent marks removed (they are the DIACRITICS reward's job, rl/diacritics.py); the dot below and
    similar letter-forming marks stay (ọ is not o). U+0329 (vertical line below, how some corpora write the Yoruba dot) is read as the dot below."""
    s = unicodedata.normalize("NFD", unicodedata.normalize("NFC", s).lower().translate(_APOS)).replace("\u0329", "\u0323")
    return unicodedata.normalize("NFC", "".join(c for c in s if unicodedata.category(c) != "Mn" or c in _KEEP_MARKS))


def _is_edge_junk(c: str) -> bool:
    return unicodedata.category(c)[0] in "PSZC"


def x_tokens(text: str) -> List[str]:
    """Whitespace tokens of the target text: NFC, lowercase, edge punctuation stripped, internal apostrophes / hyphens kept, tokens without a letter
    dropped. NOT folded (see `fold`)."""
    out: List[str] = []
    for tok in unicodedata.normalize("NFC", text).lower().translate(_APOS).split():
        i, j = 0, len(tok)
        while i < j and _is_edge_junk(tok[i]):
            i += 1
        while j > i and _is_edge_junk(tok[j - 1]):
            j -= 1
        tok = tok[i:j]
        if tok and any(c.isalpha() for c in tok):
            out.append(tok)
    return out


def x_folded_tokens(text: str) -> List[str]:
    return [t for t in (fold(t) for t in x_tokens(text)) if t]


def token_matches(cand: str, tok: str, mode: str = "exact") -> bool:
    """`mode` exact: equal; affix: also a stem relation (one contains the other, the shorter one has >= MIN_STEM_LEN letters and >= MIN_STEM_RATIO of the
    longer one), which covers prefixing / suffixing agglutination (Bantu noun classes, verb extensions) without matching short function morphemes."""
    if cand == tok:
        return True
    if mode != "affix":
        return False
    a, b = (cand, tok) if len(cand) <= len(tok) else (tok, cand)
    if len(a) < MIN_STEM_LEN or len(a) < MIN_STEM_RATIO * len(b):
        return False
    return a in b


# ------------------------------------------------------------------------------------------------------ the lexicon object
class LexiconOverlap(NamedTuple):
    overlap: Optional[float]     # IDF-weighted share of covered content words with a candidate in the output; None when n_covered < MIN_COVERED
    coverage: float              # covered content words / content words of the source (0 when the source has none)
    n_content: int
    n_covered: int
    n_hit: int
    density: Optional[float] = None   # share of the OUTPUT's tokens that are a candidate of ANY lexicon entry (stuffing detector, see `Lexicon.density_max`)


class Lexicon:
    """entries: English stem -> (idf, biblical flag, ((candidate, confidence), ...)) with candidates folded; meta: builder / validation record."""

    def __init__(self, iso: str, entries: Dict[str, Tuple[float, int, Tuple[Tuple[str, float], ...]]], meta: Dict):
        self.iso = iso
        self.entries = entries
        self.meta = meta
        self._cands: Optional[FrozenSet[str]] = None

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def match(self) -> str:
        return self.meta.get("match", "exact")

    @property
    def candidates(self) -> FrozenSet[str]:
        if self._cands is None:
            self._cands = frozenset(c for _, _, cs in self.entries.values() for c, _ in cs)
        return self._cands

    @property
    def density_max(self) -> float:
        """95th percentile of `density` over human FLORES+ dev references (validation record); 0.5 when unknown."""
        return float(self.meta.get("density_max", 0.5))

    def guarded(self, r: LexiconOverlap) -> Optional[float]:
        """overlap x stuffing factor (1 up to `density_max`, linearly to 0 at density 1); None when `r` says nothing."""
        if r.overlap is None or r.density is None:
            return None
        dm = min(max(self.density_max, 0.0), 0.99)
        return r.overlap * (1.0 if r.density <= dm else max(0.0, 1.0 - (r.density - dm) / (1.0 - dm)))

    @property
    def validated(self) -> bool:
        return bool(self.meta.get("validated", False))

    def overlap(self, src: str, mt: str, *, match: Optional[str] = None, topk: int = DEFAULT_TOPK, min_conf: float = DEFAULT_MIN_CONF,
                weighting: str = "idf", exclude_biblical: bool = False) -> LexiconOverlap:
        stems = english_content_stems(src)
        n_content = len(stems)
        mode = match or self.match
        tok_seq = x_folded_tokens(mt)
        toks = set(tok_seq)
        dens = (sum(t in self.candidates for t in tok_seq) / len(tok_seq)) if tok_seq else None
        tok_list = sorted(toks) if mode == "affix" else ()
        n_cov = n_hit = 0
        wsum = whit = 0.0
        for st in stems:
            e = self.entries.get(st)
            if e is None:
                continue
            idf, bib, cands = e
            if exclude_biblical and bib:
                continue
            cands = [c for c, s in cands[:topk] if s >= min_conf]
            if not cands:
                continue
            n_cov += 1
            w = min(idf, MAX_IDF) if weighting == "idf" else 1.0
            wsum += w
            hit = any((c in toks) or (mode == "affix" and any(token_matches(c, t, "affix") for t in tok_list)) for c in cands)
            if hit:
                n_hit += 1
                whit += w
        cov = n_cov / n_content if n_content else 0.0
        if n_cov < MIN_COVERED or wsum <= 0:
            return LexiconOverlap(None, cov, n_content, n_cov, n_hit, dens)
        return LexiconOverlap(whit / wsum, cov, n_content, n_cov, n_hit, dens)


# ------------------------------------------------------------------------------------------------------ files
def lexicon_dir() -> str:
    d = os.environ.get("GEMMA_MT_LEXICON_DIR")
    if d:
        return d
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lexicon"))


def parse_lexicon(iso: str, obj: Dict) -> Lexicon:
    ents: Dict[str, Tuple[float, int, Tuple[Tuple[str, float], ...]]] = {}
    for k, v in obj.get("e", {}).items():
        idf, bib, cands = v[0], v[1], v[2:]
        ents[k] = (float(idf), int(bib), tuple((str(c[0]), float(c[1])) for c in cands))
    return Lexicon(iso, ents, obj.get("meta", {}))


@lru_cache(maxsize=None)
def load_lexicon(iso: str, directory: Optional[str] = None) -> Optional[Lexicon]:
    p = os.path.join(directory or lexicon_dir(), f"{iso}.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return parse_lexicon(iso, json.load(f))


def lexicon_available(iso: str, validated_only: bool = True) -> bool:
    """True when `lexicon/<iso>.json` exists and (default) its validation record says the signal is usable for this language."""
    lex = load_lexicon(iso)
    return lex is not None and (lex.validated or not validated_only)


def lexicon_weight(iso: str) -> float:
    """Advantage weight proposed by the validation (0 for a language that did not validate)."""
    lex = load_lexicon(iso)
    return float(lex.meta.get("weight", 0.0)) if lex is not None and lex.validated else 0.0


def lexicon_reward(iso: str, src: str, mt: str, **kw) -> Optional[float]:
    """`overlap` with the stuffing guard: an output made of lexicon candidates (density above the human 95th percentile `meta.density_max`) is scaled
    linearly down to 0 at density 1. None when the signal says nothing (no lexicon / too few covered words / empty output). Not weighted: see `lexicon_weight`."""
    lex = load_lexicon(iso)
    if lex is None:
        return None
    return lex.guarded(lex.overlap(src, mt, **kw))


def lexicon_overlap(iso: str, src: str, mt: str, **kw) -> LexiconOverlap:
    """Overlap of the translation `mt` with the lexicon translations of the content words of the English `src` (see the module docstring).
    A language without a lexicon returns overlap None, coverage 0."""
    lex = load_lexicon(iso)
    if lex is None:
        return LexiconOverlap(None, 0.0, len(english_content_stems(src)), 0, 0)
    return lex.overlap(src, mt, **kw)
