# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text primitives of the D-61 anchors (rl/anchors_v2.py): tokenisation with the information the copy detector needs, the substring copy fractions, the English function-word list and the
function-word profile features. Pure python, no numpy, no model.

Tokens are the `\\w+` runs of the text after NFKD with the combining marks removed (the same convention as rl/anchors.words), case-folded; the SOURCE side also keeps a protection flag per
token (numbers, URLs / e-mails, capitalised names that are not sentence-initial, acronyms, the lowercase connectors inside a name, and the language's learned loanword whitelist): protected
tokens are never counted as "copied" (a human translation keeps them).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

_TOK = re.compile(r"\w+", re.UNICODE)
_URL = re.compile(r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w-]+\.[\w.-]+|\b[\w-]+\.(?:com|org|net|edu|gov|info|io|int)\b\S*", re.IGNORECASE)
_SENT_END = ".!?:;\n…"
_QUOTES = " \t\"'“‘”’([{«»"
# lowercase words that sit INSIDE a proper name ("University of Oxford", "Médecins sans Frontières", "Bill & Melinda"): protected when both neighbours are protected
CONNECTORS = frozenset("of the and for de la le du des del von van der den al el bin ibn and & in on at to".split())
MAX_CONNECTOR_RUN = 2
ABBREV = frozenset("dr mr mrs ms prof st jr sr mt vs inc ltd gen col sen rep gov capt lt sgt rev hon fr mme mlle no".split())
_PREV_WORD = re.compile(r"(\w+)\.\s*$")


def _after_abbrev(head: str) -> bool:
    """The text before the token ends with an abbreviation point ('Dr.', 'J.'): the period does not end a sentence."""
    m = _PREV_WORD.search(head)
    return bool(m) and (m.group(1).casefold() in ABBREV or (len(m.group(1)) == 1 and m.group(1).isupper()))


def plain(s: str) -> str:
    """NFKD without the combining marks, case kept."""
    return "".join(c for c in unicodedata.normalize("NFKD", s or "") if not unicodedata.combining(c))


def out_tokens(s: str) -> List[str]:
    """Output side: case-folded tokens."""
    return [t.casefold() for t in _TOK.findall(plain(s))]


@dataclass
class SrcToks:
    words: List[str]                  # case-folded tokens
    prot: List[bool]                  # protected (never counted as copied)
    spans: List[Tuple[int, int]]      # [start, end) of every token in " ".join(words)

    @property
    def n_unprot(self) -> int:
        return sum(1 for p in self.prot if not p)

    @property
    def text(self) -> str:
        return " ".join(self.words)


def source_tokens(s: str, whitelist: FrozenSet[str] = frozenset()) -> SrcToks:
    p = plain(s)
    ms = list(_TOK.finditer(p))
    url_spans = [(m.start(), m.end()) for m in _URL.finditer(p)]
    words = [m.group().casefold() for m in ms]
    prot = [False] * len(ms)
    ui = 0
    for i, m in enumerate(ms):
        w, raw = words[i], m.group()
        if any(ch.isdigit() for ch in raw):
            prot[i] = True
            continue
        while ui < len(url_spans) and url_spans[ui][1] <= m.start():
            ui += 1
        if ui < len(url_spans) and url_spans[ui][0] <= m.start() < url_spans[ui][1]:
            prot[i] = True
            continue
        if w in whitelist:
            prot[i] = True
            continue
        if len(raw) >= 2 and raw.isupper():                               # acronym (any position)
            prot[i] = True
            continue
        if len(raw) >= 2 and raw[0].isupper():
            head = p[: m.start()].rstrip(_QUOTES)
            first = ((not head) or head[-1] in _SENT_END) and not _after_abbrev(head)
            if not first:
                prot[i] = True
            elif i + 1 < len(ms) and ms[i + 1].group()[0].isupper() and p[m.end(): ms[i + 1].start()].strip() == "":
                prot[i] = True                                              # sentence-initial but followed by another capitalised word: a name ("Stanford University ...")
    # connectors between two protected tokens
    i = 0
    while i < len(ms):
        if not prot[i] and words[i] in CONNECTORS:
            j = i
            while j < len(ms) and not prot[j] and words[j] in CONNECTORS and j - i < MAX_CONNECTOR_RUN:
                j += 1
            if i > 0 and prot[i - 1] and j < len(ms) and prot[j]:
                for t in range(i, j):
                    prot[t] = True
            i = j if j > i else i + 1
        else:
            i += 1
    spans, pos = [], 0
    for w in words:
        spans.append((pos, pos + len(w)))
        pos += len(w) + 1
    return SrcToks(words, prot, spans)


@dataclass
class CopyFrac:
    frac_w: Optional[float]           # share of the unprotected source words covered by a run of >= k words that occurs verbatim in the output (None: fewer than MIN_SRC_WORDS)
    frac_c: Optional[float]           # same at character level (runs of >= m characters), None when m == 0 / too short
    n_unprot: int


MIN_SRC_WORDS = 3


def copy_fractions(src: "SrcToks | str", out: "Sequence[str] | str", k: int = 3, m: int = 14, min_unprot: int = 2, whitelist: FrozenSet[str] = frozenset(),
                   min_letters_frac: float = 0.6) -> CopyFrac:
    """Fraction of the (unprotected) English source covered by substrings that appear verbatim in the output.

    word level: a source window of `k` consecutive words that also occurs as `k` consecutive words of the output covers its words; windows with fewer than `min_unprot` unprotected
    words are ignored (a run of names / numbers is not a copy); frac_w = covered unprotected words / unprotected words.
    char level (m > 0): a source window of `m` characters (over the normalised token string, spaces included) that occurs in the normalised output covers its characters; a window with fewer
    than `min_letters_frac * m` unprotected letters is ignored; frac_c = covered unprotected letters / unprotected letters. (Catches runs inside inflected / glued words.)"""
    st = src if isinstance(src, SrcToks) else source_tokens(src, whitelist)
    ot = out if not isinstance(out, str) else out_tokens(out)
    nu = st.n_unprot
    if nu < MIN_SRC_WORDS or not ot:
        return CopyFrac(None, None, nu)
    w, prot = st.words, st.prot
    fw = None
    if k > 0 and len(w) >= k:
        oset = {tuple(ot[i: i + k]) for i in range(len(ot) - k + 1)}
        cov = [False] * len(w)
        for i in range(len(w) - k + 1):
            if tuple(w[i: i + k]) in oset and sum(1 for t in range(i, i + k) if not prot[t]) >= min_unprot:
                for t in range(i, i + k):
                    cov[t] = True
        fw = sum(1 for i in range(len(w)) if cov[i] and not prot[i]) / nu
    elif k > 0:
        fw = 0.0
    fc = None
    if m > 0:
        s, o = st.text, " ".join(ot)
        if len(s) >= m and len(o) >= m:
            letter = [False] * len(s)
            for (a, b), p in zip(st.spans, prot):
                if not p:
                    for t in range(a, b):
                        letter[t] = True
            nl = sum(letter)
            if nl:
                oset = {o[i: i + m] for i in range(len(o) - m + 1)}
                covc = [False] * len(s)
                need = min_letters_frac * m
                run = sum(letter[:m])
                for i in range(len(s) - m + 1):
                    if i:
                        run += letter[i + m - 1] - letter[i - 1]
                    if run >= need and s[i: i + m] in oset:
                        for t in range(i, i + m):
                            covc[t] = True
                fc = sum(1 for i in range(len(s)) if covc[i] and letter[i]) / nl
        else:
            fc = 0.0
    return CopyFrac(fw, fc, nu)


# ------------------------------------------------------------------------------------------------ English function words
# Pure function words (no content-ish verbs like said / make / go; no number words): the English side of the stop-word profile and the detector of English leakage.
EN_FUNC: FrozenSet[str] = frozenset("""a an the and or but if then than so as of to in on at by for with from into onto over under about after before between through during without within
is are was were be been being am do does did done doing have has had having will would shall should can could may might must not no nor i me my mine myself we us our ours ourselves you your yours
yourself he him his himself she her hers herself it its itself they them their theirs themselves this that these those there here who whom whose which what when where why how all any both each few
more most other some such only own same too very just also again once up down out off while because until against above below however therefore thus whether either neither another every
across along around among upon toward towards although though unless since whereas already yet still even ever never often always s t don didn doesn isn aren wasn weren won wouldn couldn
shouldn can't cannot""".split())


def en_func_stats(words: Sequence[str], excl: FrozenSet[str] = frozenset()) -> Tuple[int, int]:
    """(number of tokens that are English function words not in `excl`, number of tokens)."""
    return sum(1 for w in words if w in EN_FUNC and w not in excl), len(words)


def func_features(words: Sequence[str], fw: FrozenSet[str], en_excl: FrozenSet[str]) -> Dict[str, float]:
    """Profile of an output: n tokens, share of target function words, share of English-only function words, share of distinct tokens."""
    n = len(words)
    if n == 0:
        return {"n": 0, "fw": 0.0, "en": 0.0, "ttr": 1.0}
    nf = sum(1 for w in words if w in fw)
    ne = sum(1 for w in words if w in EN_FUNC and w not in en_excl and w not in fw)
    return {"n": n, "fw": nf / n, "en": ne / n, "ttr": len(set(words)) / n, "nfw": nf}
