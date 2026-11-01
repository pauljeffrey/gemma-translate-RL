# SPDX-License-Identifier: AGPL-3.0-or-later
"""The optional HARD / DOMAIN split of the long-document benchmark (medicine, law, maths-science): cleaning for domain text, candidate building and the
domain metric configuration.  Everything here is pure Python (no model, no network); the heavy lifting (fetching, contamination scan, selection on the
rental) is `scripts/longdoc_fetch_hard.py` and `scripts/longdoc_build_hard.py`.

PART 1  formulas, domain cleaning (a variant of `longdoc_select` / the fetcher rules in which formulas, numbers with units, symbols, citation markers and
        section numbers STAY, and URLs / e-mails / telephone numbers are removed)
PART 2  metrics of a translated document that matter for domain text (numbers and units, protected tokens, formulas, term consistency, modality / negation
        audit list) and `evaluate_hard`, which adds the general metrics of `longdoc.evaluate_language`
PART 3  `METRIC_CONFIG` (thresholds, regexes) saved as `hard/metric_config.json`

Honest scope.  The metrics are string-level checks on the SOURCE side (what must survive) and on the TRANSLATION side (what survived).  They need no
word alignment: numbers, protected tokens and formulas are language independent, term consistency uses the sentence beads of `longdoc.plan_document`
and a surface-form heuristic, modality is a flag list for a human audit.  None of them proves that a translation is correct; a translation can keep every
number and still say the opposite.  Medical and legal translations produced or scored with this split must not be used for clinical, diagnostic,
treatment, legal or compliance purposes; no output has been verified by a domain expert.
"""

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from gemma_mt.eval import longdoc as L

DOMAINS = ("medicine", "law", "math_science")
HARD_BINS = (512, 1024, 2048, 4096)

# ====================================================================================================================== PART 1: formulas
# One alternation, display forms first so that `$$x$$` is not read as two empty inline formulas.  An inline `$...$` must not start or end with white space
# and must not be glued to a digit (currency: "$5 and $10"), which keeps prices out of the formula spans.
FORMULA_RE = re.compile(
    r"\$\$.+?\$\$"
    r"|\\\[.+?\\\]"
    r"|\\\(.+?\\\)"
    r"|\\begin\{(?P<env>equation\*?|align\*?|gather\*?|eqnarray\*?|multline\*?|displaymath|math)\}.+?\\end\{(?P=env)\}"
    r"|<math\b[^>]*>.+?</math>"
    r"|(?<![\\$\w])\$(?![\s$])(?:[^$\n\\]|\\.)+?(?<![\s\\])\$(?![\d$])",
    re.S)
_LATEX_CMD = re.compile(r"\\[A-Za-z]+")
_FORMULA_SYMBOLS = "∑∫∏√∞≈≠≤≥±×÷∂∇∈∉⊂⊆⊇⊃∪∩∀∃→←↔⇒⇔≡∝∼∥⊥°"


def find_formulas(text: str) -> List[Tuple[int, int, str]]:
    """Non-overlapping formula spans (start, end, string) of `text`: `$..$`, `$$..$$`, `\\(..\\)`, `\\[..\\]`, LaTeX environments, `<math>..</math>`.  Spans that
    contain no letter, digit or backslash (an empty `$ $`) are ignored."""
    out = []
    for m in FORMULA_RE.finditer(text or ""):
        s = m.group(0)
        if not re.search(r"[A-Za-z0-9\\]", s.strip("$ ")):
            continue
        out.append((m.start(), m.end(), s))
    return out


def mask_formulas(text: str, repl: str = " ") -> str:
    """`text` with every formula span replaced by `repl` (default one space)."""
    out, last = [], 0
    for a, b, _ in find_formulas(text):
        out.append(text[last:a])
        out.append(repl)
        last = b
    out.append(text[last:])
    return "".join(out)


def normalise_formula(s: str) -> str:
    """Canonical form of a formula span for the exact-match test: NFC, spaces removed (LaTeX ignores them, translators re-space them), delimiters kept."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))


# ====================================================================================================================== PART 1: cleaning
_ZW = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_WS = re.compile(r"[ \t\r\f\v\u00a0\u2002\u2003\u2009\u202f]+")
URL_RE = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\"')\]]+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE_RE = re.compile(r"(?<![\w.])(?:\+?\d{1,3}[ .-])?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?![\w])")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
BARE_DOMAIN_RE = re.compile(r"\b[\w-]+\.(?:com|net|gov|edu|int|org|info|io|co\.uk|gov\.uk)(?:/\S*)?\b", re.I)
_STOP = frozenset("the of and to in a is that for it as was with be by on not he she they this are from at or an have has had his her which were but their these those been "
                  "shall may such any".split())
_REF_LINE = re.compile(r"^\s*(?:\[\d{1,3}\]|\d{1,3}[.)])\s+[A-Z][\w'’-]+,?\s+[A-Z]\.", re.U)
_REF_MARK = re.compile(r"\b(?:doi:|DOI:|ISBN|ISSN|PMID|pp\.\s*\d|Vol\.\s*\d|arXiv:\d)", re.U)
_BAD_START = re.compile(r"^(?:by the end of this (?:section|chapter)|learning objectives|key terms|chapter outline|section summary|table of contents|contents\b|references\b|bibliography\b|"
                        r"acknowledg|funding|conflicts? of interest|competing interests?|author contributions?|supplementary|data availability|keywords?:|"
                        r"copyright|all rights reserved|figure \d+[.:]|fig\. \d+[.:]|table \d+[.:])", re.I)
_TEX_LEFTOVER = re.compile(r"\\(?:cite|ref|label|footnote|textbf|textit|emph|section|subsection|includegraphics|caption|begin\{(?:figure|table|tabular|itemize|enumerate|abstract)|"
                           r"bibitem|hline|multicolumn|url|href)\b|\{\{|\}\}|\[\[|\]\]|&[a-z]+;|<[a-z/][^>]*>|\b[a-z]{3,}\d{2,4}[a-z]{3,}\b")


def clean_hard_paragraph(p: str) -> str:
    """Whitespace/NFC normalisation and removal of URLs, e-mail addresses, telephone and social-security-like numbers.  Formulas, numbers, units, symbols,
    citation markers `[12]` / `(Smith et al., 2019)` and section numbers are left alone."""
    p = unicodedata.normalize("NFC", _ZW.sub("", p or ""))
    p = _WS.sub(" ", p.replace("\n", " "))
    keep = [(a, b, s) for a, b, s in find_formulas(p)]
    if keep:                      # do not rewrite inside formulas
        out, last = [], 0
        for a, b, s in keep:
            out.append(_scrub(p[last:a]))
            out.append(s)
            last = b
        out.append(_scrub(p[last:]))
        p = "".join(out)
    else:
        p = _scrub(p)
    return _WS.sub(" ", p).strip()


def _scrub(s: str) -> str:
    s = URL_RE.sub("", s)
    s = EMAIL_RE.sub("", s)
    s = SSN_RE.sub("", s)
    s = PHONE_RE.sub("", s)
    s = BARE_DOMAIN_RE.sub("", s)
    s = re.sub(r"\(\s*[,;]?\s*\)|\[\s*\]", "", s)             # brackets emptied by the removals
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return s


def _word_stats(masked: str) -> Tuple[int, float, float]:
    w = re.findall(r"[A-Za-z][A-Za-z'’-]*", masked)
    if not w:
        return 0, 0.0, 0.0
    stop = sum(1 for x in w if x.lower() in _STOP) / len(w)
    single = sum(1 for x in w if len(x) == 1 and x not in "aAI") / len(w)
    return len(w), stop, single


def paragraph_ok_hard(p: str) -> Tuple[bool, str]:
    """(ok, reason).  Per-paragraph guard for domain text: running prose only (no reference entries, tables, learning-objective blocks, shouting, OCR garbage,
    template remnants).  Formulas, numbers and symbols do NOT make a paragraph bad."""
    masked = mask_formulas(p)
    w = masked.split()
    if len(w) < 8 and not (find_formulas(p) and len(p.split()) >= 8):
        return False, "short"
    if _TEX_LEFTOVER.search(p):
        return False, "markup"
    if _BAD_START.match(p.strip()) or _REF_LINE.match(p):
        return False, "boilerplate"
    if len(_REF_MARK.findall(p)) >= 2 or len(re.findall(r"\bet al\b", p)) >= 4:
        return False, "reference_entry"
    if p.count("|") >= 3 or p.count("\t") >= 2:
        return False, "table"
    letters = sum(c.isalpha() for c in masked)
    nonspace = sum(not c.isspace() for c in masked)
    if nonspace and letters / nonspace < 0.55:
        return False, "low_letter_share"
    n_words, stop, single = _word_stats(masked)
    if n_words >= 12 and stop < 0.12:
        return False, "not_english_prose"
    if single > 0.12:
        return False, "ocr_single_letters"
    up = [c for c in masked if c.isalpha()]
    if up and sum(c.isupper() for c in up) / len(up) > 0.45:
        return False, "shouting"
    non_latin = sum(1 for c in masked if c.isalpha() and ord(c) > 0x24F and not (0x370 <= ord(c) <= 0x3FF))   # Greek letters are fine
    if non_latin / max(letters, 1) > 0.03:
        return False, "non_latin"
    t = p.rstrip()
    t = re.sub(r"[\s$]+$", "", t) if t.endswith("$") and not t.endswith("$$") else t
    if t[-1:] not in ".!?:;\"'\u201d\u2019)]" and not (t.endswith("$$") or t.endswith("\\]") or t.endswith("</math>")):
        return False, "unterminated"
    return True, ""


def text_quality_hard(text: str, sents: Sequence[str]) -> Tuple[bool, str]:
    """(ok, reason) for a whole excerpt: English prose, sane sentence lengths, a formula share that leaves translatable text, no tables."""
    if len(sents) < 6:
        return False, "few_sentences"
    masked = mask_formulas(text)
    letters = sum(c.isalpha() for c in masked)
    if len(masked) and letters / len(masked) < 0.60:
        return False, "low_letter_share"
    f_chars = sum(b - a for a, b, _ in find_formulas(text))
    if f_chars / max(len(text), 1) > 0.40:
        return False, "formula_share"
    n_words, stop, single = _word_stats(masked)
    if stop < 0.16:
        return False, "not_english"
    if single > 0.08:
        return False, "ocr_single_letters"
    wl = [len(mask_formulas(s, "X").split()) for s in sents]
    mean = sum(wl) / len(wl)
    if mean < 6 or mean > 55:
        return False, "sentence_length"
    if sum(1 for n in wl if n > 140) / len(wl) > 0.04:
        return False, "long_sentences"
    if sum(1 for n in wl if n < 3) / len(wl) > 0.25:
        return False, "fragments"
    if re.search(r"https?://|www\.\S+|\S+@\S+\.\S+", masked):
        return False, "url_or_email"
    return True, ""


def formula_density(text: str) -> Dict[str, Any]:
    """Number of formula spans, their share of the characters and the LaTeX commands in them (for the report and the math_science preference)."""
    fs = find_formulas(text)
    return {"n_formulas": len(fs), "formula_char_share": round(sum(b - a for a, b, _ in fs) / max(len(text), 1), 4), "n_latex_cmds": sum(len(_LATEX_CMD.findall(s)) for _, _, s in fs)}


# ====================================================================================================================== PART 3: configuration
# Thresholds and regular expressions of the domain metrics, saved as `hard/metric_config.json` next to the documents so that every score table can quote them.
METRIC_CONFIG: Dict[str, Any] = {
    "version": "0.1",
    "note": "string-level checks; none of them proves that a translation is correct (see the module docstring)",
    "numbers": {
        "small_int_max": 10,                    # integers 0..10 may be written as words in the target language: reported separately, not counted as numeric errors in the lenient variant
        "year_range": [1000, 2100],
        "thousands_separators_source": [","],
        "decimal_separators_source": ["."],
        "target_conventions": ["en: 1,234.5", "eu: 1.234,5", "space: 1 234,5 (no-break / thin / narrow no-break space only)"],
        "units": ["%", "°C", "°F", "K", "mg", "g", "kg", "ug", "ml", "l", "mmol", "mol", "mmHg", "cm", "mm", "m", "km", "Hz", "kHz", "MHz", "GHz", "J", "kJ", "W", "kW", "V", "mV", "A", "mA", "eV", "s", "ms",
                  "h", "min", "bp", "kb", "Mb", "Gb", "IU", "$", "€", "£", "USD", "EUR", "GBP"],
        "unit_note": "symbols only; unit WORDS (years, days, percent, dollars) are translated and therefore not checked",
        "any_numeric_error_strict": "recall < 1 or precision < 1 on the multiset of numerals",
        "any_numeric_error_lenient": "as strict, ignoring integers <= small_int_max (they may be spelled out) on both sides",
    },
    "protected": {
        "classes": {
            "acronym": {"regex": r"\b[A-Z][A-Z0-9]{1,7}\b", "latin_only": False, "note": "all-caps tokens of 2-8 characters (WHO, DNA, BRCA1, USD); roman numerals included"},
            "chem_gene": {"regex": r"\b(?:[A-Z][a-z]?\d+(?:[A-Z][a-z]?\d*)*[+\-]{0,2}|[A-Z]{1,5}-\d+[A-Za-z]?|[A-Za-z]{2,5}-[A-Z]{2,4}\d?)\b", "latin_only": False,
                          "note": "CO2, H2O, CD4, IL-6, COVID-19, SARS-CoV, Ca2+"},
            "drug": {"regex": r"\b[a-z]{3,}(?:mab|nib|vir|cillin|mycin|statin|pril|sartan|olol|azole|prazole|dipine|oxetine|formin|gliptin|parin|caine|profen|barbital|azepam|triptan|tidine|floxacin|cycline|"
                              r"thiazide|setron|lukast|terol|sone|olone)s?\b", "latin_only": True, "note": "INN-style stems (heuristic): imatinib, amoxicillin, atorvastatin ..."},
            "latin_name": {"regex": r"\b(?:[A-Z][a-z]{3,}\s+[a-z]{4,}(?:us|um|ae|is|ii|ensis|icus|ans|ens|ata|ica|i)|[A-Z]\.\s?[a-z]{4,}(?:us|um|ae|is|ii|i|a))\b", "latin_only": True,
                           "note": "binomials (heuristic: capitalised genus + Latin-looking epithet): Plasmodium falciparum, E. coli; sentence-initial false positives are possible"},
            "statute_ref": {"regex": r"(?:§+\s?|\b(?:Sections?|Sec\.|Articles?|Art\.|Rules?|Clauses?|Paragraphs?|Chapters?|Annex(?:es)?|Schedules?|Parts?|Regulations?|Directives?)\s)\d+[A-Za-z]?(?:\.\d+)*(?:\([A-Za-z0-9]+\))*", "latin_only": False,
                            "value_group": "locator", "note": "reference locator after the keyword is checked (Section 5(1)(a), Article 12); the keyword itself is translated"},
            "eu_number": {"regex": r"\b\d{1,4}/\d{2,4}(?:/(?:EU|EC|EEC|CFSP|EURATOM))?\b|\b[CT]-\d+/\d+\b", "latin_only": False, "note": "2016/635, 2011/833/EU, C-123/45"},
            "us_cite": {"regex": r"\b\d{1,3}\s+(?:U\.S\.C\.|C\.F\.R\.|U\.S\.|S\.\s?Ct\.|F\.(?:2d|3d|4th)?|F\.\s?Supp\.(?:\s?[23]d)?|L\.\s?Ed\.(?:\s?2d)?|N\.[EW]\.(?:2d|3d)?|P\.(?:2d|3d)?|A\.(?:2d|3d)?|So\.(?:\s?[23]d)?)\s+(?:§\s?)?\d+(?:\.\d+)?",
                        "latin_only": False, "note": "123 U.S.C. 456, 410 U.S. 113, 5 F.3d 100"},
            "citation_bracket": {"regex": r"\[\d{1,3}(?:\s*[–\-,]\s*\d{1,3})*\]", "latin_only": False, "note": "numeric citation markers [12], [3-5], [1,2]"},
            "citation_author_year": {"regex": r"\([A-Z][A-Za-z\-]+(?:\s+et al\.?|\s+and\s+[A-Z][A-Za-z\-]+)?,?\s+(?:19|20)\d\d[a-z]?\)", "latin_only": True, "note": "(Smith et al., 2019): copied for Latin-script targets"},
            "url": {"regex": r"(?:https?://|www\.)[^\s<>\"')\]]+", "latin_only": False, "note": "the documents of the split contain none (removed); kept for other uses of the metric"},
        },
        "latin_script_min_share": 0.80,
        "latin_only_note": "classes marked latin_only are scored only when >= 80% of the letters of the translation are Latin/Greek (names are transliterated in other scripts)",
    },
    "formulas": {"spans": ["$..$", "$$..$$", "\\(..\\)", "\\[..\\]", "LaTeX math environments", "<math>..</math> (MathML)"], "compared_on": "content without delimiters and white space",
                 "exact_share": "multiset overlap / number of source spans", "order": "longest common subsequence of the span sequence / number of source spans; edit distance of the sequence"},
    "term_consistency": {"min_occurrences": 3, "max_terms": 20, "min_contrast": 0.30, "rendering_prefix_chars": 5, "top_common_target_words_ignored": 40, "common_word_share": 0.6,
                         "heuristic": "word-alignment-free: for each English term (1-2 content words, >= 3 occurrences in sentences/beads), the candidate rendering is a target word "
                                      "(5-character prefix, for inflection) whose contrast (share of the term's beads containing it minus share of the other beads containing it) is >= min_contrast; "
                                      "the one covering most of the term's beads is the modal rendering; consistency = share of the term's beads that contain it. Undetermined terms (contrast < min_contrast) are not counted. "
                                      "The same statistic is computed on the reference as a baseline."},
    "modality": {"cues": {"negation": ["not", "no", "never", "cannot", "without", "neither", "nor"], "obligation": ["must", "shall", "required", "prohibited"], "permission": ["may"]},
                 "lexicon_learning": {"min_support": 12, "min_contrast": 0.35, "top_tokens_per_cue": 2},
                 "short_ratio_vs_ref": 0.6,
                 "note": "a FLAG LIST for human audit, not a score: a bead is flagged when the source bead has a cue, and the learned target equivalent(s) of the cue (learned from the "
                         "sentence-aligned reference) occur in the reference bead but not in the translation bead, or (no lexicon) the translation bead is much shorter than the reference bead"},
    "bootstrap": {"n_boot": 1000, "seed": 0, "alpha": 0.05, "resampling_unit": "document"},
}


_ORDER = ("us_cite", "eu_number", "statute_ref", "citation_bracket", "citation_author_year", "url", "chem_gene", "drug", "latin_name", "acronym")
METRIC_CONFIG["protected"]["classes"] = {k: METRIC_CONFIG["protected"]["classes"][k] for k in _ORDER}      # order = priority when two patterns overlap


def save_config(path: str, config: Optional[Dict[str, Any]] = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config or METRIC_CONFIG, f, ensure_ascii=False, indent=1, sort_keys=False)


# ====================================================================================================================== PART 2: numbers, units, dates
def to_ascii_digits(s: str) -> str:
    """Every decimal digit of every script (Arabic-Indic, Devanagari, Nko, Adlam ...) -> ASCII 0-9; minus sign variants -> '-'."""
    out = []
    for ch in s:
        if ch.isdigit() and not ch.isascii():
            try:
                out.append(str(unicodedata.digit(ch)))
                continue
            except ValueError:
                pass
        out.append(ch)
    return "".join(out).replace("−", "-").replace("–", "–")


_NUM_RE = re.compile(r"(?<![\w.,])(?P<sign>[-−+]?)(?P<num>\d{1,3}(?:[   ]\d{3})+(?:[.,]\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)(?![\w])", re.U)


def _strip_zeros(x: str) -> str:
    if "." in x:
        x = x.rstrip("0").rstrip(".")
    return x or "0"


def number_variants(tok: str, source: bool) -> List[str]:
    """Canonical string(s) of a numeral token ('1,234.5' -> '1234.5').  English source: ',' is the thousands and '.' the decimal separator.  Target: both the English and the
    European reading ('1.234,5' -> '1234.5', '3,5' -> '3.5' and also '35'); all readings are returned and the matcher accepts any of them."""
    t = re.sub(r"[   ]", "", tok)
    outs: List[str] = []
    has_c, has_d = "," in t, "." in t

    def add(x: str):
        x = _strip_zeros(x)
        if x not in outs:
            outs.append(x)
    if has_c and has_d:
        if t.rfind(",") > t.rfind("."):
            add(t.replace(".", "").replace(",", "."))
        else:
            add(t.replace(",", ""))
        return outs
    if has_c:
        parts = t.split(",")
        thousands_like = all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3
        add(t.replace(",", ""))                            # English reading
        if not source and (len(parts) == 2):
            add(t.replace(",", "."))                       # decimal comma
        elif not source and thousands_like:
            pass
        return outs
    if has_d:
        parts = t.split(".")
        thousands_like = len(parts) > 1 and all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3
        add(t)                                              # decimal point
        if not source and thousands_like:
            add(t.replace(".", ""))                        # European thousands point
        return outs
    add(t)
    return outs


def extract_numbers(text: str, source: bool = True, masked: bool = True) -> List[Tuple[str, List[str], int, str]]:
    """[(canonical primary value, all accepted variants, position, raw token)] of the numerals of `text` (formulas masked when `masked`).  Digits of every script are normalised first."""
    t = to_ascii_digits(mask_formulas(text) if masked else text)
    out = []
    for m in _NUM_RE.finditer(t):
        sign = "-" if m.group("sign") in ("-", "\u2212") else ""
        vs = number_variants(m.group("num"), source)
        out.append((sign + vs[0], [sign + v for v in vs], m.start(), m.group('num')))
    return out


def _is_small_int(v: str, small: int) -> bool:
    try:
        return "." not in v and abs(int(v)) <= small
    except ValueError:
        return False


def _multiset_match(src: Sequence[str], tgt_variants: Sequence[Sequence[str]]) -> Tuple[int, List[int], List[int]]:
    """Greedy multiset match of source values to target numerals (a target numeral matches a source value if ANY of its variants is equal).  Returns (matched, unmatched source
    indices, unmatched target indices).  Exact matches of the primary variant are consumed first, so '1,000' (en) is not stolen by a '1.000' later."""
    avail: Dict[str, List[int]] = defaultdict(list)
    for j, vs in enumerate(tgt_variants):
        for v in vs:
            avail[v].append(j)
    used_t, matched = set(), 0
    # two phases: the primary reading of the target first, then any reading
    done_s = set()
    for j, vs in enumerate(tgt_variants):                      # phase 1: the primary reading of the target equals a source value
        for i, s in enumerate(src):
            if i not in done_s and j not in used_t and vs and vs[0] == s:
                done_s.add(i); used_t.add(j); matched += 1
                break
    for i, s in enumerate(src):                                # phase 2: any other reading
        if i in done_s:
            continue
        for j in avail.get(s, []):
            if j not in used_t:
                done_s.add(i); used_t.add(j); matched += 1
                break
    return matched, [i for i in range(len(src)) if i not in done_s], [j for j in range(len(tgt_variants)) if j not in used_t]


_UNITS = METRIC_CONFIG["numbers"]["units"]
_UNIT_ALIASES = {"µg": "ug", "μg": "ug", "mcg": "ug", "mL": "ml", "L": "l", "℃": "°C", "℉": "°F", "µm": "um", "μm": "um", " ": " "}
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(set(_UNITS) | set(_UNIT_ALIASES) - {" "}, key=len, reverse=True))
_PAIR_AFTER = re.compile(rf"(?<![\w.,])(?P<num>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:[.,]\d+)?)[\s  ]?(?P<unit>{_UNIT_ALT})(?![A-Za-zÀ-ɏ\d])", re.U)
_PAIR_BEFORE = re.compile(rf"(?P<unit>[$€£]|USD|EUR|GBP)[\s  ]?(?P<num>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:[.,]\d+)?)(?![\d])", re.U)


def extract_number_unit_pairs(text: str, source: bool = True) -> List[Tuple[str, List[str], str]]:
    """[(primary number, number variants, normalised unit)] for 'number unit' (5 mg, 37.5 °C, 12%) and 'unit number' ($5, EUR 10) pairs, symbol units only."""
    t = to_ascii_digits(mask_formulas(text))
    out = []
    for rx in (_PAIR_AFTER, _PAIR_BEFORE):
        for m in rx.finditer(t):
            u = _UNIT_ALIASES.get(m.group("unit"), m.group("unit"))
            if u == "$":
                u = "USD"
            if u == "€":
                u = "EUR"
            if u == "£":
                u = "GBP"
            vs = number_variants(m.group("num"), source)
            out.append((vs[0], vs, u))
    return out


def _prf(matched: int, n_src: int, n_tgt: int) -> Dict[str, Optional[float]]:
    p = matched / n_tgt if n_tgt else None
    r = matched / n_src if n_src else None
    f = (2 * p * r / (p + r)) if (p is not None and r is not None and (p + r) > 0) else (0.0 if (n_src and n_tgt) else None)
    return {"precision": p, "recall": r, "f1": f}


def number_metrics(src: str, hyp: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Number / unit / date preservation of ONE document.  Keys: n_src, n_hyp, num_precision/recall/f1 (all numerals), the same for numerals that are not small integers
    (`_ge11`), the share of missing small integers (word-numeral risk), year recall, number+unit pair PRF, and `any_error_strict` / `any_error_lenient`."""
    cfg = (config or METRIC_CONFIG)["numbers"]
    small = int(cfg["small_int_max"])
    lo, hi = cfg["year_range"]
    s = extract_numbers(src, True)
    h = extract_numbers(hyp, False)
    sv = [x[0] for x in s]
    matched, un_s, un_t = _multiset_match(sv, [x[1] for x in h])
    res: Dict[str, Any] = {"n_src": len(s), "n_hyp": len(h)}
    res.update({f"num_{k}": v for k, v in _prf(matched, len(s), len(h)).items()})
    # lenient: ignore small integers on both sides
    keep_s = [i for i, v in enumerate(sv) if not _is_small_int(v, small)]
    kept_vals = {sv[i] for i in keep_s}
    keep_t = [j for j, x in enumerate(h) if not _is_small_int(x[0], small) or any(v in kept_vals for v in x[1])]
    m2, un_s2, un_t2 = _multiset_match([sv[i] for i in keep_s], [h[j][1] for j in keep_t])
    res.update({f"num_ge11_{k}": v for k, v in _prf(m2, len(keep_s), len(keep_t)).items()})
    miss_small = sum(1 for i in un_s if _is_small_int(sv[i], small))
    res["missing_small_int_share"] = (miss_small / len(un_s)) if un_s else None
    res["n_missing"] = len(un_s)
    res["n_extra"] = len(un_t)
    ys = [x[0] for x in s if re.fullmatch(r"\d{4}", x[3]) and lo <= int(x[3]) <= hi]
    if ys:
        ym, _, _ = _multiset_match(ys, [x[1] for x in h])
        res["year_recall"] = ym / len(ys)
    else:
        res["year_recall"] = None
    sp = extract_number_unit_pairs(src, True)
    hp = extract_number_unit_pairs(hyp, False)
    pm = 0
    used = set()
    for num, vs, u in sp:
        for j, (n2, v2, u2) in enumerate(hp):
            if j not in used and u2 == u and num in v2:
                used.add(j); pm += 1
                break
    res["n_pairs_src"], res["n_pairs_hyp"] = len(sp), len(hp)
    res.update({f"pair_{k}": v for k, v in _prf(pm, len(sp), len(hp)).items()})
    res["any_error_strict"] = (len(un_s) > 0 or len(un_t) > 0) if (s or h) else None
    res["any_error_lenient"] = (len(un_s2) > 0 or len(un_t2) > 0) if (keep_s or keep_t) else (False if (s or h) else None)
    res["copy_rate"] = (matched / len(s)) if s else None
    return res


# ====================================================================================================================== PART 2: protected tokens
_PROT_RX = {k: re.compile(v["regex"]) for k, v in METRIC_CONFIG["protected"]["classes"].items()}
_LOCATOR = re.compile(r"\d+[A-Za-z]?(?:\.\d+)*(?:\([A-Za-z0-9]+\))*")
_COMMON_CAPS = frozenset("I A II III IV VI VII VIII IX XI XII OK TV PM AM".split())


def latin_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for c in letters if ord(c) < 0x250 or 0x370 <= ord(c) <= 0x3FF or 0x1E00 <= ord(c) <= 0x1EFF) / len(letters)


def protected_tokens(text: str, config: Optional[Dict[str, Any]] = None) -> List[Tuple[str, str]]:
    """[(class, token)] protected tokens of an English text (formulas masked; a token belongs to the first class that matches it)."""
    cfg = (config or METRIC_CONFIG)["protected"]["classes"]
    t = mask_formulas(text)
    out, taken = [], []

    def free(a, b):
        return all(b <= x or a >= y for x, y in taken)
    for cls, spec in cfg.items():
        rx = _PROT_RX[cls] if cls in _PROT_RX and spec["regex"] == METRIC_CONFIG["protected"]["classes"][cls]["regex"] else re.compile(spec["regex"])
        for m in rx.finditer(t):
            a, b = m.start(), m.end()
            if not free(a, b):
                continue
            tok = m.group(0)
            if cls == "statute_ref":
                lm = _LOCATOR.search(tok)
                if not lm:
                    continue
                tok = lm.group(0)
            if cls == "acronym" and (tok in _COMMON_CAPS or tok.isdigit() or re.fullmatch(r"[A-Z]\d*", tok)):
                continue
            if cls == "chem_gene" and re.fullmatch(r"[A-Z]\d{4}", tok):
                continue
            taken.append((a, b))
            out.append((cls, tok))
    return out


def _count_in(text: str, tok: str) -> int:
    if not tok:
        return 0
    return len(re.findall(rf"(?<![\w]){re.escape(tok)}(?![\w])", text)) if (tok[0].isalnum() and tok[-1].isalnum()) else text.count(tok)


def protected_metrics(src: str, hyp: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Recall of the protected-token list extracted from the source: per class and overall (count-capped: min(source count, translation count) / source count).
    Classes marked `latin_only` are scored only when the translation is >= 80% Latin-script letters; `protected_scored` says whether they were."""
    cfg = config or METRIC_CONFIG
    classes = cfg["protected"]["classes"]
    latin = latin_share(hyp) >= cfg["protected"]["latin_script_min_share"]
    toks = protected_tokens(src, cfg)
    h = unicodedata.normalize("NFC", hyp)
    per: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    by_tok: Counter = Counter((c, t) for c, t in toks)
    lost: List[Tuple[str, str, int, int]] = []
    for (cls, tok), n in by_tok.items():
        if classes[cls].get("latin_only") and not latin:
            continue
        got = min(n, _count_in(h, tok))
        per[cls][0] += got
        per[cls][1] += n
        if got < n:
            lost.append((cls, tok, n, got))
    tot_got = sum(v[0] for v in per.values())
    tot = sum(v[1] for v in per.values())
    return {"protected_recall": (tot_got / tot) if tot else None, "n_protected": tot, "protected_latin_target": latin,
            "protected_by_class": {k: {"recall": v[0] / v[1], "n": v[1]} for k, v in sorted(per.items()) if v[1]},
            "protected_lost": sorted(lost, key=lambda x: -x[2])[:15]}


# ====================================================================================================================== PART 2: formulas
def formula_core(span: str) -> str:
    """Content of a formula span without delimiters and white space ('$ x^2 $' == '\\(x^2\\)' == '$$x^2$$')."""
    s = span.strip()
    for a, b in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if s.startswith(a) and s.endswith(b) and len(s) >= len(a) + len(b):
            s = s[len(a):len(s) - len(b)]
            break
    s = re.sub(r"^<math\b[^>]*>|</math>$", "", s)
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s))


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def formula_metrics(src: str, hyp: str) -> Dict[str, Any]:
    """Formula preservation of ONE document.  None values when the source has no formula."""
    s = [formula_core(x[2]) for x in find_formulas(src)]
    h = [formula_core(x[2]) for x in find_formulas(hyp)]
    if not s:
        return {"n_formulas_src": 0, "n_formulas_hyp": len(h), "formula_exact_share": None, "formula_ordered_share": None, "formula_edit_distance": None, "formula_edit_norm": None,
                "formula_extra": len(h)}
    cs, ch = Counter(s), Counter(h)
    exact = sum(min(cs[k], ch[k]) for k in cs)
    lcs = _lcs(s, h)
    ed = _edit_distance(s, h)
    return {"n_formulas_src": len(s), "n_formulas_hyp": len(h), "formula_exact_share": exact / len(s), "formula_ordered_share": lcs / len(s), "formula_edit_distance": ed,
            "formula_edit_norm": ed / max(len(s), len(h)), "formula_extra": max(0, len(h) - exact)}


def mask_formulas_with_ids(text: str) -> Tuple[str, List[str]]:
    """For a system that should not touch formulas: the text with every span replaced by a placeholder `[[F1]]`, `[[F2]]` ... and the list of the spans."""
    spans, out, last = [], [], 0
    for a, b, s in find_formulas(text):
        out.append(text[last:a])
        spans.append(s)
        out.append(f"[[F{len(spans)}]]")
        last = b
    out.append(text[last:])
    return "".join(out), spans


def unmask_formulas(text: str, spans: Sequence[str]) -> str:
    return re.sub(r"\[\[F(\d+)\]\]", lambda m: spans[int(m.group(1)) - 1] if 0 < int(m.group(1)) <= len(spans) else m.group(0), text)


# ====================================================================================================================== PART 2: term consistency (alignment-free heuristic)
_EN_STOP = frozenset(("the of and to in a an is are was were be been being that this these those for with as by on at from or not but which who whom whose it its their his her they them we our you your "
                      "has have had having do does did can could may might must shall should will would than then there here also such into over under between among about after before during other "
                      "more most less any all each both either neither one two three first second new used using use based show shown results result study studies data").split())


def _stem(w: str) -> str:
    w = w.lower()
    for suf in ("ies", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def _content_words(sent: str) -> List[str]:
    return [_stem(w) for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", mask_formulas(sent)) if w.lower() not in _EN_STOP]


def english_terms(src_sents: Sequence[str], min_occ: int = 3, max_terms: int = 20) -> List[Tuple[str, List[int]]]:
    """Candidate terms: content unigrams / bigrams occurring in >= `min_occ` different sentences; returns [(term, sentence indices)] best first.  A bigram that explains a
    unigram (same sentences) replaces it."""
    uni: Dict[str, set] = defaultdict(set)
    bi: Dict[str, set] = defaultdict(set)
    for i, s in enumerate(src_sents):
        ws = _content_words(s)
        raw = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", mask_formulas(s))]
        for w in ws:
            if len(w) >= 4:
                uni[w].add(i)
        for a, b in zip(raw, raw[1:]):
            if a not in _EN_STOP and b not in _EN_STOP and len(a) >= 3 and len(b) >= 3:
                bi[_stem(a) + " " + _stem(b)].add(i)
    terms = {k: v for k, v in bi.items() if len(v) >= min_occ}
    for k, v in uni.items():
        if len(v) >= min_occ and not any(k in t.split() and len(tv) >= 0.8 * len(v) for t, tv in terms.items()):
            terms[k] = v
    ranked = sorted(terms.items(), key=lambda kv: (-(len(kv[1]) * (1 + 0.5 * (" " in kv[0]))), kv[0]))
    return [(k, sorted(v)) for k, v in ranked[:max_terms]]


def _target_words(s: str) -> List[str]:
    return [w for w in re.findall(r"\w[\ẁ-ͯ'’\-]*", unicodedata.normalize("NFC", s).lower()) if not w.isdigit()]


def term_consistency(src_beads: Sequence[str], tgt_beads: Sequence[str], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Alignment-free term consistency over sentence beads (see METRIC_CONFIG['term_consistency'] for the heuristic).  `src_beads[i]` and `tgt_beads[i]` are aligned."""
    cfg = (config or METRIC_CONFIG)["term_consistency"]
    n = len(src_beads)
    if n < 6 or len(tgt_beads) != n:
        return {"term_consistency": None, "n_terms": 0, "n_terms_determined": 0, "terms": []}
    terms = english_terms(src_beads, cfg["min_occurrences"], cfg["max_terms"])
    tw = [set(_target_words(t)) for t in tgt_beads]
    pre = int(cfg["rendering_prefix_chars"])
    # ignore function words: among the 40 most frequent target words of the document those that occur in more than 60% of the beads
    df = Counter(w for ws in tw for w in ws)
    common = {w for w, c in df.most_common(int(cfg["top_common_target_words_ignored"])) if c > cfg["common_word_share"] * n}
    tpre = [set(w[:pre] for w in ws if len(w) >= 3 and w not in common) for ws in tw]
    pre_df = Counter(p for ps in tpre for p in ps)
    rows, vals = [], []
    for term, idx in terms:
        inside = set(idx)
        k_in = len(inside)
        outside = n - k_in
        if outside < 3:
            continue
        cnt_in: Counter = Counter()
        for i in inside:
            cnt_in.update(tpre[i])
        best, best_c, best_cov = None, -1.0, -1.0
        for p, c in cnt_in.items():
            if c < 2:
                continue
            contrast = c / k_in - (pre_df[p] - c) / outside
            if contrast >= cfg["min_contrast"] and (c / k_in, contrast) > (best_cov, best_c):
                best, best_c, best_cov = p, contrast, c / k_in
        if best is None:
            rows.append({"term": term, "n_beads": k_in, "rendering": None, "consistency": None, "contrast": None})
            continue
        cons = cnt_in[best] / k_in
        rows.append({"term": term, "n_beads": k_in, "rendering": best, "consistency": cons, "contrast": best_c})
        vals.append(cons)
    return {"term_consistency": (sum(vals) / len(vals)) if vals else None, "n_terms": len(terms), "n_terms_determined": len(vals), "terms": rows[:12]}


def protected_term_consistency(src: str, hyp: str, min_occ: int = 3) -> Dict[str, Any]:
    """Protected tokens that occur >= `min_occ` times in the source must keep ONE surface form: share of their occurrences found verbatim in the translation (Latin targets)."""
    toks = Counter(protected_tokens(src))
    got = tot = 0
    for (cls, tok), n in toks.items():
        if n < min_occ or METRIC_CONFIG["protected"]["classes"][cls].get("latin_only") and latin_share(hyp) < 0.8:
            continue
        tot += n
        got += min(n, _count_in(hyp, tok))
    return {"protected_term_consistency": (got / tot) if tot else None, "n_protected_repeated": tot}


# ====================================================================================================================== PART 2: modality / negation audit
_CUES = METRIC_CONFIG["modality"]["cues"]
_CUE_ALL = {w: k for k, ws in _CUES.items() for w in ws}
_CUE_RX = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _CUE_ALL), key=len, reverse=True)) + r")\b|\b\w+n't\b", re.I)


def source_cues(sent: str) -> List[str]:
    """Modality / negation cue words of an English sentence (lower-case; contractions n't -> 'not')."""
    out = []
    for m in _CUE_RX.finditer(sent):
        w = m.group(0).lower()
        out.append("not" if w.endswith("n't") else w)
    return out


def learn_cue_lexicon(src_sents: Sequence[str], ref_sents: Sequence[str], config: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """Target-side equivalents of the cues learned from the sentence-aligned reference of a language: for each cue, the target 5-character prefixes with the highest contrast
    (share of cue sentences containing it minus share of the other sentences containing it).  Returns {} entries for cues with too little support / contrast."""
    cfg = (config or METRIC_CONFIG)["modality"]["lexicon_learning"]
    n = len(src_sents)
    tw = [set(w[:5] for w in _target_words(s) if len(w) >= 2) for s in ref_sents]
    df = Counter(p for ps in tw for p in ps)
    cues_by_sent = [set(source_cues(s)) for s in src_sents]
    lex: Dict[str, List[str]] = {}
    for cue in _CUE_ALL:
        idx = [i for i, cs in enumerate(cues_by_sent) if cue in cs]
        if len(idx) < cfg["min_support"] or n - len(idx) < cfg["min_support"]:
            continue
        cnt: Counter = Counter()
        for i in idx:
            cnt.update(tw[i])
        scored = sorted(((c / len(idx) - (df[p] - c) / (n - len(idx)), p) for p, c in cnt.items() if c >= 3), reverse=True)
        top = [p for sc, p in scored[: cfg["top_tokens_per_cue"]] if sc >= cfg["min_contrast"]]
        if top:
            lex[cue] = top
    return lex


def modality_flags(src_beads: Sequence[str], tgt_beads: Sequence[str], ref_beads: Sequence[str], lexicon: Optional[Dict[str, List[str]]] = None,
                   config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Flag list (for a human audit, NOT a score) of the beads whose English side has a modality / negation cue that the translation seems to have dropped.
    Reason `cue_missing`: the learned target equivalent(s) of the cue are in the reference bead but not in the translation bead.  Reason `short_vs_ref`: no lexicon for the cue and the
    translation bead is shorter than `short_ratio_vs_ref` x the reference bead.  Reason `copied`: the bead is the English text."""
    cfg = (config or METRIC_CONFIG)["modality"]
    lex = lexicon or {}
    out = []
    for i, (s, t, r) in enumerate(zip(src_beads, tgt_beads, ref_beads)):
        cues = sorted(set(source_cues(s)))
        if not cues:
            continue
        tp = set(w[:5] for w in _target_words(t))
        rp = set(w[:5] for w in _target_words(r))
        if not t.strip():
            out.append({"bead": i, "cue": cues, "reason": "empty", "src": s, "hyp": t, "ref": r})
            continue
        if t.strip().lower() == s.strip().lower():
            out.append({"bead": i, "cue": cues, "reason": "copied", "src": s, "hyp": t, "ref": r})
            continue
        for cue in cues:
            eq = lex.get(cue)
            if eq:
                if any(e in rp for e in eq) and not any(e in tp for e in eq):
                    out.append({"bead": i, "cue": [cue], "reason": "cue_missing", "equivalents": eq, "src": s, "hyp": t, "ref": r})
                    break
            elif r and len(t) < cfg["short_ratio_vs_ref"] * len(r):
                out.append({"bead": i, "cue": [cue], "reason": "short_vs_ref", "src": s, "hyp": t, "ref": r})
                break
    return out


# ====================================================================================================================== evaluate_hard
def _beads(d: "L.RefDoc", hyp_text: str, lm) -> Tuple[List[str], List[str], List[str]]:
    """(source, hypothesis, reference) beads of a document: the segments of the chunk plan (omitted / copied chunks are not in them)."""
    plan = L.plan_document(d.src_sents, d.ref_sents, hyp_text, lm)
    return [a for a, _, _ in plan.segments], [b for _, b, _ in plan.segments], [c for _, _, c in plan.segments]


METRIC_KEYS = ("num_f1", "num_recall", "num_precision", "num_ge11_f1", "num_ge11_recall", "num_ge11_precision", "year_recall", "pair_f1", "pair_recall", "copy_rate",
               "protected_recall", "protected_term_consistency", "formula_exact_share", "formula_ordered_share", "formula_edit_norm", "term_consistency", "term_consistency_ref",
               "missing_small_int_share")
RATE_KEYS = ("any_error_strict", "any_error_lenient")


def evaluate_hard(refdocs: Sequence["L.RefDoc"], hyps: Dict[str, "L.Hyp"], *, iso: Optional[str] = None, config: Optional[Dict[str, Any]] = None, with_core: bool = True,
                  n_boot: int = 1000, seed: int = 0, scorer_ref: Optional[Callable] = None, scorer_qe: Optional[Callable] = None, keep_rows: bool = True,
                  max_flags: int = 200, **core_kw) -> Dict[str, Any]:
    """Domain metrics of ONE language for ONE system.  `refdocs`: the reference documents of the language (`longdoc.RefDoc`); `hyps`: doc_id -> `longdoc.Hyp` (a document without
    a hypothesis counts as an empty translation).  Returns {"iso", "n_docs", "per_doc": [...], "metrics": {name: {value, lo, hi, n}}, "rates": {...}, "by_domain": {...},
    "modality_flags": [...], "lexicon": {...}, "core": <longdoc.evaluate_language result> (with_core), "config_version"}."""
    cfg = config or METRIC_CONFIG
    docs = [d for d in refdocs if iso is None or d.iso == iso]
    iso = iso or (docs[0].iso if docs else "")
    if not docs:
        return {"iso": iso, "n_docs": 0}
    all_src = [s for d in docs for s in d.src_sents]
    all_ref = [s for d in docs for s in d.ref_sents]
    lm = L.make_length_model(all_src, all_ref)
    lexicon = learn_cue_lexicon(all_src, all_ref, cfg)
    rows, flags = [], []
    for d in docs:
        h = hyps.get(d.doc_id)
        text = h.text if h is not None else ""
        src_text = d.source_text
        row: Dict[str, Any] = {"doc_id": d.doc_id, "iso": iso, "domain": d.domain, "bin": d.bin, "missing_hypothesis": h is None}
        row.update(number_metrics(src_text, text, cfg))
        row.update(protected_metrics(src_text, text, cfg))
        row.update(protected_term_consistency(src_text, text))
        row.update(formula_metrics(src_text, text))
        sb, hb, rb = _beads(d, text, lm) if text.strip() else ([], [], [])
        tc = term_consistency(sb, hb, cfg) if hb else {"term_consistency": None, "n_terms": 0, "n_terms_determined": 0, "terms": []}
        tr = term_consistency(d.src_sents, d.ref_sents, cfg)
        row.update({"term_consistency": tc["term_consistency"], "n_terms": tc["n_terms"], "n_terms_determined": tc["n_terms_determined"], "term_consistency_ref": tr["term_consistency"],
                    "terms": tc["terms"]})
        mf = modality_flags(sb, hb, rb, lexicon, cfg) if hb else []
        row["n_modality_flags"] = len(mf)
        row["n_cue_beads"] = sum(1 for s in sb if source_cues(s))
        for f in mf:
            if len(flags) < max_flags:
                flags.append({"doc_id": d.doc_id, "iso": iso, **f})
        rows.append(row)
    metrics = {k: L.bootstrap_mean([r.get(k) for r in rows], n_boot, seed) for k in METRIC_KEYS}
    rates = {k: L._rate([r.get(k) for r in rows]) for k in RATE_KEYS}
    rates["docs_with_modality_flag"] = L._rate([r["n_modality_flags"] > 0 for r in rows if r["n_cue_beads"] > 0])
    by_domain: Dict[str, Any] = {}
    for dom in sorted({r["domain"] for r in rows}):
        rs = [r for r in rows if r["domain"] == dom]
        by_domain[dom] = {"n_docs": len(rs), **{k: L.bootstrap_mean([r.get(k) for r in rs], min(n_boot, 500), seed + 1) for k in ("num_f1", "num_ge11_f1", "protected_recall", "formula_exact_share", "term_consistency")},
                          "any_error_lenient": L._rate([r.get("any_error_lenient") for r in rs])}
    res: Dict[str, Any] = {"iso": iso, "n_docs": len(docs), "n_missing": sum(1 for r in rows if r["missing_hypothesis"]), "metrics": metrics, "rates": rates, "by_domain": by_domain,
                           "modality_flags": flags, "lexicon": lexicon, "config_version": cfg.get("version")}
    if keep_rows:
        res["per_doc"] = rows
    if with_core:
        res["core"] = L.evaluate_language(docs, hyps, iso=iso, scorer_ref=scorer_ref, scorer_qe=scorer_qe, n_boot=n_boot, seed=seed, keep_rows=keep_rows, **core_kw)
    return res


def evaluate_hard_run(refdocs: Sequence["L.RefDoc"], hyps_by_iso: Dict[str, Dict[str, "L.Hyp"]], **kw) -> Dict[str, Any]:
    """`evaluate_hard` for every language of `hyps_by_iso` (iso -> doc_id -> Hyp)."""
    return {iso: evaluate_hard(refdocs, hyps, iso=iso, **kw) for iso, hyps in sorted(hyps_by_iso.items())}
