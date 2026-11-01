# SPDX-License-Identifier: AGPL-3.0-or-later
"""Long-document translation evaluation (D-44): the metrics side of the public long-context benchmark English -> African languages.

A benchmark document is an English text of about 512 / 1024 / 2048 / 4096 Gemma tokens. Its REFERENCE is a list of target sentences aligned 1:1 to the
canonical English sentences of the document (`split_sentences`, the same splitter as the RL reward):
  * pseudo reference  every English sentence translated on its own by the best open translator of the language, then joined (`translate_documents`);
  * human reference   SmolDoc documents (sentence-aligned by the publisher), FLORES+ passages, or several of them stitched together.
A system under test translates the WHOLE document in one go; this module compares that full-document translation with the reference.

Metrics (per language, document level, bootstrap CIs over DOCUMENTS)
  chrf, bleu      corpus chrF++ (char 6, word 2) and BLEU (13a) with every document as ONE segment (d-chrF++, d-BLEU); NFC + collapsed whitespace
  africomet_ref   AfriCOMET-STL-1.1 (reference based) on aligned chunks: the hypothesis is split into sentences, aligned to the source sentences with the
                  RL reward's Gale-Church beads (language-aware length model of the omission detector, D-30), each bead (source, hypothesis, reference)
                  is scored, the document score is the source-length weighted mean in which an omitted / copied / missing part counts `penalty`
                  (default 0.0); `africomet_ref_covered` leaves the failed parts out; `africomet_qe` is the reference-free variant (no reference needed)
  coverage        share of the source characters that have a translated counterpart (1 - failed weight of the chunk plan), omitted sentences, copied
                  (untranslated) sentences, shortfall beads, extra (appended / unmatched) output
  loop_rate       share of documents whose translation contains a degenerate repetition loop (a unit of 1-30 words repeated >= 4 times in a row and >= 8
                  words long) that the reference does not contain
  finished_rate   share of documents whose generation stopped at its own end token inside the budget (given by the generator; None when unknown)
  length_ratio    characters of the translation / characters of the reference (mean, median, p05, p95, share outside [0.6, 1.6])
Tiers: a language whose best pseudo-reference translator is weak on FLORES+ dev is "C" (reported, never ranked); see `assign_tier`.

Everything here is CPU only. sacrebleu is needed for chrF++/BLEU (imported lazily); AfriCOMET comes in through any `scorer(srcs, mts, refs)` callable
(`gemma_mt.rl.comet`: CometScorer / WorkerScorer / FakeScorer).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from gemma_mt.rl.chunk_plan import aggregate
from gemma_mt.rl.guards import is_copy_of_source, is_repetitive
from gemma_mt.rl import segments as _seg
from gemma_mt.rl.segments import LengthModel, gale_church

# ------------------------------------------------------------------------------------------------ constants
BINS = (512, 1024, 2048, 4096)                  # nominal Gemma tokens of the English source
BIN_RANGES = {512: (410, 640), 1024: (820, 1280), 2048: (1640, 2560), 4096: (3280, 5120)}   # accepted token range per nominal bin (nominal -20% .. +25%)
SHORT_BINS = (64, 256)                           # owner decision 2026-10-20 (evening): the short RL bins b64 / b256 are part of the benchmark
ALL_BINS = SHORT_BINS + BINS
SHORT_EDGES_AFRO = {64: (1, 64), 256: (65, 256)}  # AfroXLM-R (AfriCOMET encoder) tokens of the English source: the edges of the RL bins b64 / b256
REF_KINDS = ("pseudo", "human", "human_stitched")
TIERS = ("A", "B", "C", "N")                    # A confident, B usable (flagged), C low confidence (never ranked), N no open system / no pseudo reference
RANKED_TIERS = ("A", "B")
DEFAULT_TIER_THRESHOLDS = {                     # PROVISIONAL (not measured): to be replaced by the distribution of the bake-off on FLORES+ dev
    "A": {"chrf": 45.0, "comet": 0.65},         # chrF++ vs human references >= 45 and AfriCOMET-ref >= 0.65 (the latter only where the language is covered)
    "B": {"chrf": 30.0, "comet": 0.50},
}
# ------------------------------------------------------------------------------------------------ linear-time sentence splitter
def _is_boundary_fast(text: str, start: int, term: str, next_pos: int) -> bool:
    r"""`segments._is_boundary` with the last word found by a backward scan instead of `re.search(r"([^\s]+)$", text[:start])`, which copies and scans the whole prefix
    for EVERY candidate boundary (quadratic: 0.2 s for a 40,000-character text, > 1 s for a long generated document). Same decisions (tested against the original)."""
    if next_pos < len(text):
        nxt = text[next_pos]
        if nxt.isalpha() and nxt.islower() and "።" not in term:
            return False
    i = start
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    word = text[i:start]
    bare = word.strip("\"'“‘(«[{").lower()
    if term == ".":
        if bare in _seg._ABBREV or bare.rstrip(".") in _seg._ABBREV:
            return False
        if re.fullmatch(r"[A-Za-z]", bare or "x") and word[:1].isupper():
            return False
        if re.fullmatch(r"(?:[A-Za-z]\.){2,}", bare + "."):
            return False
        if bare.isdigit() and len(bare) <= 3:
            j = i
            while j > 0 and text[j - 1].isspace():
                j -= 1
            if j == 0 or text[j - 1] in _seg._TERMINATORS + ":;":
                return False
    return True


def split_sentences(text: str) -> List[str]:
    """Identical to `rl.segments.split_sentences` (the splitter of the RL reward) but linear in the length of the text."""
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return []
    out: List[str] = []
    last = 0
    for m in _seg._CANDIDATE.finditer(t):
        end = m.end(2)
        if _is_boundary_fast(t, m.start(1), m.group(1), m.end()):
            seg = t[last:end].strip()
            if seg:
                out.append(seg)
            last = m.end()
    tail = t[last:].strip()
    if tail:
        out.append(tail)
    return out


# ------------------------------------------------------------------------------------------------ text helpers


def norm_text(s: str) -> str:
    """NFC and every run of white space (incl. newlines) -> one space. This is what the metrics see."""
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def paragraphs(text: str) -> List[str]:
    """Paragraphs of `text` (separated by a blank line), whitespace-normalised, empty ones dropped."""
    return [p for p in (norm_text(x) for x in re.split(r"\n[ \t]*\n", text or "")) if p]


def canonical_sentences(text: str) -> List[str]:
    """The sentence units of the benchmark: `split_sentences` of the whole text (paragraph breaks are plain white space), the splitter of the RL reward.
    A fixed point: re-splitting the joined result gives the same list (checked by `sentences_are_stable`)."""
    s = split_sentences(text)
    for _ in range(3):
        t = split_sentences(" ".join(s))
        if t == s:
            break
        s = t
    return s


def sentences_are_stable(sents: Sequence[str]) -> bool:
    return split_sentences(" ".join(sents)) == list(sents)


def paragraph_starts(text: str, sents: Sequence[str]) -> List[int]:
    """Indices of the sentences that start a paragraph (always begins with 0). [] when the paragraphs cannot be mapped onto `sents` (then the text is one block)."""
    starts, k = [], 0
    for p in paragraphs(text):
        n = len(split_sentences(p))
        starts.append(k)
        k += n
    return starts if (k == len(sents) and starts) else ([0] if sents else [])


def join_sentences(sents: Sequence[str], starts: Optional[Sequence[int]] = None) -> str:
    """Sentences joined with a space; with `starts` a blank line before every paragraph start (except the first)."""
    if not starts or len(starts) < 2:
        return " ".join(sents)
    out, cut = [], set(starts[1:])
    for i, s in enumerate(sents):
        if i in cut:
            out.append("\n\n")
        elif i:
            out.append(" ")
        out.append(s)
    return "".join(out)


_SPLIT_LEVELS = ("; ", ": ", " \u2014 ", " - ", ", ", " ")


def split_long_sentence(sent: str, max_chars: int = 700) -> List[str]:
    """Pieces of at most ~max_chars characters, cut after the strongest available separator nearest to the middle (translators are trained on sentences; a
    1,500-character 'sentence' without a full stop is split for the translation call and re-joined; the alignment unit stays the sentence)."""
    if len(sent) <= max_chars:
        return [sent]
    for sep in _SPLIT_LEVELS:
        cuts = [m.end() for m in re.finditer(re.escape(sep), sent)]
        cuts = [c for c in cuts if 20 < c < len(sent) - 20]
        if cuts:
            mid = len(sent) / 2
            c = min(cuts, key=lambda x: abs(x - mid))
            left, right = sent[:c].strip(), sent[c:].strip()
            if left and right:
                return split_long_sentence(left, max_chars) + split_long_sentence(right, max_chars)
    h = len(sent) // 2
    return [sent[:h], sent[h:]]


# ------------------------------------------------------------------------------------------------ documents and references
@dataclass
class RefDoc:
    doc_id: str
    iso: str
    source: str                          # English text (paragraphs separated by blank lines allowed)
    src_sents: List[str]                 # canonical sentence units of `source`
    ref_sents: List[str]                 # reference, aligned 1:1 to src_sents
    ref_kind: str = "pseudo"             # pseudo | human | human_stitched
    ref_system: str = ""                 # translator name, "smoldoc", "flores_plus"
    domain: str = ""
    bin: int = 0                         # nominal length bin of the English source (0 = unbinned)
    para_starts: List[int] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)       # per sentence quality flags of a pseudo reference ("" = fine)
    contaminated: Optional[bool] = None  # human-reference documents: English seen in the training data

    def __post_init__(self):
        if len(self.src_sents) != len(self.ref_sents):
            raise ValueError(f"{self.doc_id}/{self.iso}: {len(self.src_sents)} source sentences but {len(self.ref_sents)} reference sentences")
        if self.ref_kind not in REF_KINDS:
            raise ValueError(f"unknown ref_kind {self.ref_kind!r}")

    @property
    def ref_text(self) -> str:
        return join_sentences(self.ref_sents, self.para_starts)

    @property
    def source_text(self) -> str:
        return join_sentences(self.src_sents, self.para_starts)

    @property
    def flagged_share(self) -> float:
        return sum(1 for f in self.flags if f) / max(len(self.ref_sents), 1)


@dataclass
class Hyp:
    doc_id: str
    iso: str
    text: str
    finished: Optional[bool] = None
    system: str = ""
    new_tokens: Optional[int] = None


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def load_refdocs(docs_path: str, refs_path: str, only_valid: bool = False, max_flagged: float = 0.05) -> List[RefDoc]:
    """Join the English document file (rows: doc_id, text, sents, para_starts, domain, bin, ...) with a per-language reference file (rows: doc_id, iso,
    ref_sents, ref_kind, ref_system, flags, ...). `only_valid` drops pseudo-reference documents whose flagged-sentence share exceeds `max_flagged`."""
    docs = {d["doc_id"]: d for d in read_jsonl(docs_path)}
    out = []
    for r in read_jsonl(refs_path):
        d = docs.get(r["doc_id"])
        if d is None:
            continue
        rd = RefDoc(doc_id=r["doc_id"], iso=r["iso"], source=d["text"], src_sents=d["sents"], ref_sents=r["ref_sents"],
                    ref_kind=r.get("ref_kind", "pseudo"), ref_system=r.get("ref_system", ""), domain=d.get("domain", ""), bin=int(d.get("bin", 0) or 0),
                    para_starts=d.get("para_starts", []), flags=r.get("flags", []), contaminated=r.get("contaminated"))
        if only_valid and rd.ref_kind == "pseudo" and rd.flagged_share > max_flagged:
            continue
        out.append(rd)
    return out


def load_hyps(path: str) -> Dict[Tuple[str, str], Hyp]:
    out = {}
    for r in read_jsonl(path):
        out[(r["iso"], r["doc_id"])] = Hyp(r["doc_id"], r["iso"], r.get("hyp", r.get("text", "")), r.get("finished"), r.get("system", ""), r.get("new_tokens"))
    return out


# ------------------------------------------------------------------------------------------------ translate sentence by sentence, then join
@dataclass
class TranslatedDoc:
    doc_id: str
    iso: str
    system: str
    src_sents: List[str]
    tgt_sents: List[str]
    para_starts: List[int]
    flags: List[str]

    @property
    def text(self) -> str:
        return join_sentences(self.tgt_sents, self.para_starts)

    @property
    def flagged_share(self) -> float:
        return sum(1 for f in self.flags if f) / max(len(self.flags), 1)

    def to_ref_row(self, ref_kind: str = "pseudo") -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "iso": self.iso, "ref_kind": ref_kind, "ref_system": self.system, "ref_sents": self.tgt_sents, "flags": self.flags,
                "flagged_share": round(self.flagged_share, 4)}


def sentence_flag(src: str, tgt: str, ratio_range: Tuple[float, float] = (0.25, 4.0)) -> str:
    """Quality flag of ONE translated sentence: '' (fine), 'empty', 'copy' (English left untranslated), 'loop' (degenerate repetition), 'short', 'long'."""
    if not tgt or not tgt.strip():
        return "empty"
    if len(src.split()) >= 4 and is_copy_of_source(src, tgt, 0.9):
        return "copy"
    if is_repetitive(tgt) and not is_repetitive(src):
        return "loop"
    r = len(tgt.strip()) / max(len(src.strip()), 1)
    if len(src) >= 20 and r < ratio_range[0]:
        return "short"
    if r > ratio_range[1]:
        return "long"
    return ""


def translate_documents(docs: Sequence[Dict[str, Any]], translate_fn: Callable[[List[str]], List[str]], *, iso: str, system: str, batch_size: int = 64,
                        max_sentence_chars: int = 700, retry_fn: Optional[Callable[[List[str]], List[str]]] = None,
                        progress: Optional[Callable[[int, int], None]] = None) -> List[TranslatedDoc]:
    """Translate every sentence of every document on its own and join them (the pseudo reference), recording the sentence alignment.

    docs         [{"doc_id", "text"}]  (paragraphs separated by blank lines); the sentence units are `canonical_sentences(text)`.
    translate_fn list of English strings -> list of translations (same length, same order); the language is fixed by the caller's closure.
    Identical sentences (across documents) are translated once; sentences longer than `max_sentence_chars` are cut into pieces for the call and the
    pieces are re-joined, so the alignment unit never changes. `retry_fn` (same signature) is called once for the sentences that came back flagged
    (e.g. the same system with sampling / a different beam); its answer replaces the first one only if it is not flagged.
    Returns one TranslatedDoc per input document, in input order: tgt_sents[i] translates src_sents[i]."""
    per_doc, uniq = [], {}
    for d in docs:
        sents = list(d["sents"]) if d.get("sents") else canonical_sentences(d["text"])      # a document may bring its own alignment units (human references)
        starts = d.get("para_starts") or paragraph_starts(d["text"], sents)
        per_doc.append((d, sents, starts))
        for s in sents:
            uniq.setdefault(s, None)
    pieces: Dict[str, List[str]] = {s: split_long_sentence(s, max_sentence_chars) for s in uniq}
    flat = sorted({p for ps in pieces.values() for p in ps}, key=len)
    out_piece: Dict[str, str] = {}
    for i in range(0, len(flat), batch_size):
        chunk = flat[i: i + batch_size]
        res = translate_fn(list(chunk))
        if len(res) != len(chunk):
            raise RuntimeError(f"translate_fn returned {len(res)} outputs for {len(chunk)} inputs")
        for a, b in zip(chunk, res):
            out_piece[a] = (b or "").strip()
        if progress:
            progress(min(i + batch_size, len(flat)), len(flat))
    done = {s: norm_text(" ".join(out_piece[p] for p in ps)) for s, ps in pieces.items()}
    if retry_fn is not None:
        bad = [s for s in uniq if sentence_flag(s, done[s])]
        if bad:
            again = retry_fn(bad)
            if len(again) != len(bad):
                raise RuntimeError("retry_fn returned a wrong number of outputs")
            for s, t in zip(bad, again):
                t = norm_text(t or "")
                if not sentence_flag(s, t):
                    done[s] = t
    flag_of = {s: sentence_flag(s, done[s]) for s in uniq}
    return [TranslatedDoc(d["doc_id"], iso, system, sents, [done[s] for s in sents], starts, [flag_of[s] for s in sents]) for d, sents, starts in per_doc]


# ------------------------------------------------------------------------------------------------ chrF++ / BLEU sufficient statistics and the document bootstrap
@functools.lru_cache(maxsize=1)
def _metrics():
    from sacrebleu.metrics import BLEU, CHRF

    return CHRF(word_order=2), BLEU(tokenize="13a")


def doc_stat_vectors(hyps: Sequence[str], refs: Sequence[str]):
    """Per-document sufficient statistics of chrF++ and BLEU (numpy arrays n x k): the corpus score of ANY subset of documents is the metric applied to
    the SUM of the rows, which makes the document bootstrap cheap. Texts are NFC + whitespace-collapsed first."""
    import numpy as np

    chrf, bleu = _metrics()
    h = [norm_text(x) for x in hyps]
    r = [norm_text(x) for x in refs]
    cs = np.array([chrf._extract_corpus_statistics([a], [[b]])[0] for a, b in zip(h, r)], dtype=np.float64)
    bs = np.array([bleu._extract_corpus_statistics([a], [[b]])[0] for a, b in zip(h, r)], dtype=np.float64)
    return cs, bs


def scores_from_stats(chrf_stats, bleu_stats, idx=None) -> Tuple[float, float]:
    """(chrF++, BLEU) of the documents `idx` (all when None) from the statistic rows."""
    chrf, bleu = _metrics()
    c = chrf_stats if idx is None else chrf_stats[idx]
    b = bleu_stats if idx is None else bleu_stats[idx]
    if len(c) == 0:
        return float("nan"), float("nan")
    return float(chrf._aggregate_and_compute([c.sum(0).tolist()]).score), float(bleu._aggregate_and_compute([b.sum(0).tolist()]).score)


def _resample_indices(n: int, rng, strata: Optional[Sequence[Any]] = None):
    import numpy as np

    if strata is None:
        return rng.randint(0, n, size=n)
    groups: Dict[Any, List[int]] = defaultdict(list)
    for i, s in enumerate(strata):
        groups[s].append(i)
    parts = [np.array(ix)[rng.randint(0, len(ix), size=len(ix))] for ix in groups.values()]
    return np.concatenate(parts)


def _ci(vals: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    xs = sorted(v for v in vals if v == v)
    if not xs:
        return float("nan"), float("nan")
    lo = xs[int(math.floor((alpha / 2) * (len(xs) - 1)))]
    hi = xs[int(math.ceil((1 - alpha / 2) * (len(xs) - 1)))]
    return lo, hi


def bootstrap_scores(chrf_stats, bleu_stats, n_boot: int = 1000, seed: int = 0, strata: Optional[Sequence[Any]] = None, alpha: float = 0.05) -> Dict[str, Any]:
    """Document-level percentile bootstrap of corpus chrF++ and BLEU. `strata`: resample inside each stratum (e.g. the length bin)."""
    import numpy as np

    n = len(chrf_stats)
    point = scores_from_stats(chrf_stats, bleu_stats)
    out = {"n_docs": n, "chrf": {"value": point[0]}, "bleu": {"value": point[1]}}
    if n < 2 or n_boot <= 0:
        for k in ("chrf", "bleu"):
            out[k].update({"lo": float("nan"), "hi": float("nan"), "se": float("nan")})
        return out
    rng = np.random.RandomState(seed)
    cs, bs = [], []
    for _ in range(n_boot):
        c, b = scores_from_stats(chrf_stats, bleu_stats, _resample_indices(n, rng, strata))
        cs.append(c)
        bs.append(b)
    for k, v in (("chrf", cs), ("bleu", bs)):
        lo, hi = _ci(v, alpha)
        out[k].update({"lo": lo, "hi": hi, "se": float(np.std(v, ddof=1))})
    return out


def paired_bootstrap(stats_a: Tuple[Any, Any], stats_b: Tuple[Any, Any], n_boot: int = 1000, seed: int = 0, strata: Optional[Sequence[Any]] = None,
                     alpha: float = 0.05) -> Dict[str, Any]:
    """Paired document bootstrap of system A minus system B (same documents, same resamples): delta, CI, share of resamples with A > B."""
    import numpy as np

    (ca, ba), (cb, bb) = stats_a, stats_b
    n = len(ca)
    if n != len(cb):
        raise ValueError("the two systems must be scored on the same documents")
    rng = np.random.RandomState(seed)
    d_c, d_b = [], []
    for _ in range(n_boot):
        ix = _resample_indices(n, rng, strata)
        a1, a2 = scores_from_stats(ca, ba, ix)
        b1, b2 = scores_from_stats(cb, bb, ix)
        d_c.append(a1 - b1)
        d_b.append(a2 - b2)
    pa, pb = scores_from_stats(ca, ba), scores_from_stats(cb, bb)
    res = {"n_docs": n}
    for k, v, p in (("chrf", d_c, pa[0] - pb[0]), ("bleu", d_b, pa[1] - pb[1])):
        lo, hi = _ci(v, alpha)
        res[k] = {"delta": p, "lo": lo, "hi": hi, "p_a_better": float(np.mean([x > 0 for x in v]))}
    return res


def bootstrap_mean(values: Sequence[float], n_boot: int = 1000, seed: int = 0, strata: Optional[Sequence[Any]] = None, alpha: float = 0.05) -> Dict[str, float]:
    """Mean of per-document values with a percentile bootstrap CI (values that are None / NaN are dropped, with their strata)."""
    import numpy as np

    keep = [i for i, v in enumerate(values) if v is not None and v == v]
    xs = np.array([values[i] for i in keep], dtype=np.float64)
    st = None if strata is None else [strata[i] for i in keep]
    if len(xs) == 0:
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    out = {"value": float(xs.mean()), "n": int(len(xs))}
    if len(xs) < 2 or n_boot <= 0:
        out.update({"lo": float("nan"), "hi": float("nan")})
        return out
    rng = np.random.RandomState(seed)
    ms = [float(xs[_resample_indices(len(xs), rng, st)].mean()) for _ in range(n_boot)]
    out["lo"], out["hi"] = _ci(ms, alpha)
    return out


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ------------------------------------------------------------------------------------------------ failure metrics: loops, length
def repetition_loop(text: str, max_period: int = 30, min_repeats: int = 4, min_run_words: int = 8) -> Dict[str, Any]:
    """Degenerate repetition: a unit of 1..max_period words repeated at least `min_repeats` times in a row covering at least `min_run_words` words
    (the '... muna muna muna ...' failure of the SFT model). Returns {"loop", "period", "repeats", "covered_words", "start_word", "at_tail", "covered_share"}."""
    toks = norm_text(text).lower().split()
    n = len(toks)
    best = {"loop": False, "period": 0, "repeats": 0, "covered_words": 0, "start_word": -1, "at_tail": False, "covered_share": 0.0}
    if n < min_run_words:
        return best
    ids: Dict[str, int] = {}
    t = [ids.setdefault(w, len(ids)) for w in toks]
    for p in range(1, min(max_period, n // 2) + 1):
        run, start = 0, 0
        for i in range(n - p):
            if t[i] == t[i + p]:
                if run == 0:
                    start = i
                run += 1
            else:
                run = 0
            # a run of `run` matching positions = a unit of p words repeated (run + p) / p times
            if run >= (min_repeats - 1) * p and run + p >= min_run_words:
                covered = run + p
                if covered > best["covered_words"]:
                    best.update({"loop": True, "period": p, "repeats": covered // p, "covered_words": covered, "start_word": start,
                                 "at_tail": start + covered >= n - p, "covered_share": covered / n})
    return best


def length_stats(ratios: Sequence[float], lo: float = 0.6, hi: float = 1.6) -> Dict[str, float]:
    xs = sorted(r for r in ratios if r is not None and r == r)
    if not xs:
        return {"mean": float("nan"), "median": float("nan"), "p05": float("nan"), "p95": float("nan"), "share_outside": float("nan"), "n": 0}
    q = lambda p: xs[int(round(p * (len(xs) - 1)))]
    return {"mean": sum(xs) / len(xs), "median": q(0.5), "p05": q(0.05), "p95": q(0.95), "share_outside": sum(1 for r in xs if r < lo or r > hi) / len(xs), "n": len(xs)}


def ends_cleanly(text: str) -> bool:
    t = (text or "").rstrip()
    return bool(t) and t[-1] in ".!?।።\"'”’»)"


# ------------------------------------------------------------------------------------------------ chunk plan on explicit sentence lists
@dataclass
class DocPlan:
    """The scoring plan of one (source sentences, hypothesis) pair, the chunk plan of the RL reward (rl/chunk_plan.py, D-23/D-24/D-30) on explicit
    sentence lists. `items` = [(segment index | None = failed chunk, weight)]; `segments` = (source, hypothesis, reference) triples of the scored chunks."""
    how: str
    items: List[Tuple[Optional[int], float]]
    segments: List[Tuple[str, str, str]]
    n_src: int
    n_hyp: int
    omitted_sents: int          # source sentences with no output at all
    copied_sents: int           # source sentences whose output is a copy of the source
    short_beads: int            # shortfall beads (part of the chunk has no counterpart)
    failed_w: float             # source-side failed weight (omitted + copied + shortfall), source characters
    total_w: float              # source characters in the plan
    extra_w: float              # output-side failed weight (unmatched output + appended text), output characters
    hyp_chars: int
    extra_tags: List[str] = field(default_factory=list)
    omitted_idx: List[int] = field(default_factory=list)   # source sentence indices with no output / copied output
    short_idx: List[int] = field(default_factory=list)     # source sentence indices of shortfall beads (part of the bead is missing; cannot be pinned to one sentence)

    @property
    def coverage_chars(self) -> float:
        return 1.0 - self.failed_w / self.total_w if self.total_w > 0 else float("nan")

    @property
    def coverage_sents(self) -> float:
        return 1.0 - (self.omitted_sents + self.copied_sents) / self.n_src if self.n_src else float("nan")

    @property
    def extra_share(self) -> float:
        return self.extra_w / self.hyp_chars if self.hyp_chars else 0.0


def make_length_model(src_sents: Sequence[str], ref_sents: Sequence[str], s2: float = 6.8, omit_cost: float = 4.0, extra_cost: float = 4.0) -> LengthModel:
    """Language-aware Gale-Church model (D-30 'dp4'): expected output characters per source character = corpus ratio of the reference."""
    ls, lr = sum(len(s) for s in src_sents), sum(len(r) for r in ref_sents)
    return LengthModel(ratio=max(lr / max(ls, 1), 1e-3), s2=s2, omit_cost=omit_cost, extra_cost=extra_cost)


def plan_document(src_sents: Sequence[str], ref_sents: Sequence[str], hyp: str, lm: LengthModel, *, short_frac: float = 0.7, short_min: float = 0.15,
                  extra_ratio_limit: float = 1.8, length_slack: float = 0.15, w_hi: Optional[float] = None, copy_jaccard: float = 0.9) -> DocPlan:
    """Align the hypothesis sentences to the source sentences (Gale-Church with the language length model, omission / extra beads) and decide which chunks are
    scored and which are failed (omitted, copied, shortfall, appended text) with the RULES of `rl.chunk_plan.plan_chunks` (source-length weights). It works on
    explicit sentence lists, so the reference of every bead is known; `tests/test_longdoc.py` asserts that it agrees with `plan_chunks` on stable documents."""
    s = list(src_sents)
    o = split_sentences(hyp)
    n, m = len(s), len(o)
    if n == 0:
        return DocPlan("empty", [], [], 0, m, 0, 0, 0, 0.0, 0.0, 0.0, len(" ".join(o)))
    if m == 0:                                                 # nothing was generated: the whole document is one failed chunk
        tw = float(len(" ".join(s)))
        return DocPlan("empty_output", [(None, tw)], [], n, 0, n, 0, 0, tw, tw, 0.0, 0, ["omitted_or_copied"], list(range(n)), [])
    if n < 2:
        beads, how = [((0,), tuple(range(m)))], "single"
    elif n == m:
        beads, how = [((i,), (i,)) for i in range(n)], "aligned"
    elif m <= 1:
        beads, how = [(tuple(range(n)), tuple(range(m)))], "whole"
    else:
        beads, how = gale_church(s, o, lm), "gale_church"
    pairs = []                                                 # (source text, output text, source idx, output idx) of beads that have a source side
    for si, oi in beads:
        if si:
            pairs.append((" ".join(s[i] for i in si), " ".join(o[j] for j in oi), si, oi))
    total_out = len(" ".join(o))
    matched = len(" ".join(b for _, b, _, _ in pairs))
    items: List[Tuple[Optional[int], float]] = []
    segs: List[Tuple[str, str, str]] = []
    tags: List[str] = []
    omitted = copied = short = 0
    omitted_idx: List[int] = []
    short_idx: List[int] = []
    failed_w = extra_w = total_w = 0.0
    extra = max(0, total_out - matched)
    if len(pairs) > 1 and extra > 0 and extra >= 0.05 * max(total_out, 1):
        items.append((None, float(extra)))
        extra_w += extra
        tags.append("unmatched_output")
    for k, (a, b, si, oi) in enumerate(pairs):
        w = float(len(a))
        total_w += w
        miss_share = 0.0
        if how in ("gale_church", "whole") and a and b.strip() and (len(si) >= 2 or how == "whole"):
            ratio = max(float(lm.ratio), 1e-3)
            if len(b) < short_frac * ratio * len(a):
                miss = max(0.0, len(a) - len(b) / ratio)
                if miss >= short_min * len(a):
                    miss_share = miss / len(a)
        if a and b:
            lim = w_hi * (1.0 + length_slack) if w_hi else extra_ratio_limit
            over = len(b) - lim * len(a)
            if over > 0.10 * len(b):
                items.append((None, float(over)))
                extra_w += over
                tags.append("appended_text")
        if len(pairs) > 1 and (not b.strip() or is_copy_of_source(a, b, copy_jaccard)):
            items.append((None, w))
            failed_w += w
            if not b.strip():
                omitted += len(si)
            else:
                copied += len(si)
            omitted_idx.extend(si)
            tags.append("omitted_or_copied")
            continue
        segs.append((a, b, " ".join(ref_sents[i] for i in si)))
        if miss_share > 0.0:
            items.append((len(segs) - 1, w * (1.0 - miss_share)))
            items.append((None, w * miss_share))
            failed_w += w * miss_share
            short += 1
            short_idx.extend(si)
            tags.append("shortfall")
        else:
            items.append((len(segs) - 1, w))
    return DocPlan(how, items, segs, n, m, omitted, copied, short, failed_w, total_w, extra_w, total_out, tags, omitted_idx, short_idx)


# ------------------------------------------------------------------------------------------------ AfriCOMET over chunks
class _NoTokenLengths:
    @staticmethod
    def token_lengths(texts):
        return [len(t) // 3 + 2 for t in texts]


def comet_over_plans(plans: Sequence[DocPlan], scorer: Callable, *, reference_based: bool, penalty: float = 0.0, max_tokens: int = 510,
                     source_of_segment: Optional[Callable] = None) -> Dict[str, Any]:
    """Score every chunk of every plan with ONE batched scorer call and aggregate per document with `chunk_plan.aggregate` (failed chunk = `penalty`).
    scorer(srcs, mts, refs) -> list of floats (reference based) or scorer(srcs, mts) (QE). Returns {"doc": [penalised mean], "covered": [mean over scored
    chunks only], "n_truncated": chunks over the encoder limit (the encoder truncates source and translation separately at 510 tokens), "n_segments"}."""
    srcs, mts, refs, owner = [], [], [], []
    for d, p in enumerate(plans):
        for a, b, r in p.segments:
            srcs.append(a)
            mts.append(b)
            refs.append(r)
            owner.append(d)
    if srcs:
        scores = scorer(srcs, mts, refs) if reference_based else scorer(srcs, mts)
        if len(scores) != len(srcs):
            raise RuntimeError("the scorer returned a wrong number of scores")
    else:
        scores = []
    tl = getattr(scorer, "token_lengths", None) or _NoTokenLengths.token_lengths
    n_trunc = 0
    if srcs:
        n_trunc = sum(1 for x, y in zip(tl(srcs), tl(mts)) if x > max_tokens + 2 or y > max_tokens + 2)
    by_doc: Dict[int, List[float]] = defaultdict(list)
    for d, s in zip(owner, scores):
        by_doc[d].append(float(s))
    doc, covered = [], []
    for d, p in enumerate(plans):
        seg_scores = by_doc.get(d, [])
        if not p.items:
            doc.append(float("nan"))
            covered.append(float("nan"))
            continue
        doc.append(aggregate(p.items, seg_scores, penalty))
        ok = [(i, w) for i, w in p.items if i is not None]
        covered.append(aggregate(ok, seg_scores, penalty) if ok else float("nan"))
    return {"doc": doc, "covered": covered, "n_truncated": n_trunc, "n_segments": len(srcs)}


# ------------------------------------------------------------------------------------------------ evaluation of one language
def _rate(flags: Sequence[Optional[bool]]) -> Dict[str, float]:
    xs = [bool(f) for f in flags if f is not None]
    if not xs:
        return {"value": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    lo, hi = wilson_interval(sum(xs), len(xs))
    return {"value": sum(xs) / len(xs), "lo": lo, "hi": hi, "n": len(xs)}


def evaluate_language(docs: Sequence[RefDoc], hyps: Dict[str, Hyp], *, iso: str, scorer_ref: Optional[Callable] = None, scorer_qe: Optional[Callable] = None,
                      n_boot: int = 1000, seed: int = 0, penalty: float = 0.0, short_frac: float = 0.7, tier: Optional[Dict[str, Any]] = None,
                      stratify_bins: bool = False, keep_rows: bool = True, w_hi: Optional[float] = None) -> Dict[str, Any]:
    """Metrics of ONE language for ONE system. `docs`: reference documents of the language (all with the same ref_kind, see `evaluate_run`);
    `hyps`: doc_id -> Hyp. A document without a hypothesis counts as an empty translation (scores 0, coverage 0) and is counted in n_missing."""
    docs = [d for d in docs if d.iso == iso]
    if not docs:
        return {"iso": iso, "n_docs": 0}
    all_src = [s for d in docs for s in d.src_sents]
    all_ref = [s for d in docs for s in d.ref_sents]
    lm = make_length_model(all_src, all_ref)
    if w_hi is None:                                                     # the language's human p95 of the sentence length ratio, like the calibration's w_hi
        rs = sorted(len(r) / max(len(s), 1) for s, r in zip(all_src, all_ref) if len(s) >= 10)
        w_hi = min(rs[int(0.95 * (len(rs) - 1))], 2.5) if rs else None
    hy_text, rows, plans = [], [], []
    n_missing = 0
    for d in docs:
        h = hyps.get(d.doc_id)
        if h is None:
            n_missing += 1
            h = Hyp(d.doc_id, iso, "", None)
        hy_text.append(h.text)
        plans.append(plan_document(d.src_sents, d.ref_sents, h.text, lm, short_frac=short_frac, w_hi=w_hi))
    cs, bs = doc_stat_vectors(hy_text, [d.ref_text for d in docs])
    strata = [d.bin for d in docs] if stratify_bins else None
    boot = bootstrap_scores(cs, bs, n_boot, seed, strata)
    doc_chrf = [scores_from_stats(cs, bs, [i])[0] for i in range(len(docs))]
    comet_ref = comet_qe = None
    if scorer_ref is not None:
        comet_ref = comet_over_plans(plans, scorer_ref, reference_based=True, penalty=penalty)
    if scorer_qe is not None:
        qplans = [DocPlan(p.how, p.items, [(a, b, "") for a, b, _ in p.segments], p.n_src, p.n_hyp, p.omitted_sents, p.copied_sents, p.short_beads,
                          p.failed_w, p.total_w, p.extra_w, p.hyp_chars, p.extra_tags, p.omitted_idx, p.short_idx) for p in plans]
        comet_qe = comet_over_plans(qplans, scorer_qe, reference_based=False, penalty=penalty)
    loops, ref_loops, ratios, finished, clean_end = [], [], [], [], []
    for i, d in enumerate(docs):
        h = hyps.get(d.doc_id)
        hl = repetition_loop(h.text) if h else repetition_loop("")
        rl = repetition_loop(d.ref_text)
        loops.append(bool(hl["loop"]) and not rl["loop"])
        ref_loops.append(bool(rl["loop"]))
        rc = len(norm_text(d.ref_text))
        ratios.append(len(norm_text(h.text)) / rc if (h and rc) else 0.0)
        finished.append(None if h is None else h.finished)
        clean_end.append(ends_cleanly(h.text) if h else False)
        rows.append({"doc_id": d.doc_id, "iso": iso, "domain": d.domain, "bin": d.bin, "ref_kind": d.ref_kind, "ref_system": d.ref_system, "n_sents": len(d.src_sents),
                     "doc_chrf": doc_chrf[i], "doc_bleu": scores_from_stats(cs, bs, [i])[1], "coverage_chars": plans[i].coverage_chars, "coverage_sents": plans[i].coverage_sents,
                     "omitted_sents": plans[i].omitted_sents, "copied_sents": plans[i].copied_sents, "short_beads": plans[i].short_beads,
                     "extra_share": plans[i].extra_share, "loop": loops[-1], "loop_in_ref": ref_loops[-1], "length_ratio": ratios[-1],
                     "finished": finished[-1], "ends_cleanly": clean_end[-1], "n_hyp_sents": plans[i].n_hyp, "align": plans[i].how,
                     "africomet_ref": comet_ref["doc"][i] if comet_ref else None, "africomet_ref_covered": comet_ref["covered"][i] if comet_ref else None,
                     "africomet_qe": comet_qe["doc"][i] if comet_qe else None, "flagged_ref_share": d.flagged_share})
    bm = lambda key: bootstrap_mean([r[key] for r in rows], n_boot, seed + 1, strata)
    res: Dict[str, Any] = {
        "iso": iso, "n_docs": len(docs), "n_missing": n_missing, "ref_kinds": dict(Counter(d.ref_kind for d in docs)),
        "ref_systems": dict(Counter(d.ref_system for d in docs)), "source_tokens_note": "bins are nominal Gemma tokens of the English source",
        "chrf": boot["chrf"], "bleu": boot["bleu"], "doc_chrf_mean": bm("doc_chrf"),
        "coverage_chars": bm("coverage_chars"), "coverage_sents": bm("coverage_sents"), "extra_share": bm("extra_share"),
        "omitted_sents_total": sum(r["omitted_sents"] for r in rows), "copied_sents_total": sum(r["copied_sents"] for r in rows),
        "docs_with_omission": _rate([(r["omitted_sents"] + r["short_beads"]) > 0 for r in rows]),
        "loop_rate": _rate(loops), "ref_loop_rate": _rate(ref_loops), "finished_rate": _rate(finished), "ends_cleanly_rate": _rate(clean_end),
        "length_ratio": length_stats(ratios), "length_model_ratio": lm.ratio, "w_hi": w_hi,
    }
    if comet_ref:
        res["africomet_ref"] = bm("africomet_ref")
        res["africomet_ref_covered"] = bm("africomet_ref_covered")
        res["africomet_truncated_chunks"] = comet_ref["n_truncated"]
        res["africomet_chunks"] = comet_ref["n_segments"]
    if comet_qe:
        res["africomet_qe"] = bm("africomet_qe")
    res["by_bin"] = _breakdown(rows, "bin", n_boot, seed)
    res["by_domain"] = _breakdown(rows, "domain", n_boot, seed)
    if tier is not None:
        res["tier"] = tier.get("tier")
        res["ranked"] = tier.get("tier") in RANKED_TIERS
        res["tier_info"] = tier
    if keep_rows:
        res["rows"] = rows
    return res


def _breakdown(rows: Sequence[Dict[str, Any]], key: str, n_boot: int, seed: int) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    for k, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        d = {"n_docs": len(rs)}
        for m in ("doc_chrf", "coverage_chars", "africomet_ref", "africomet_qe", "length_ratio"):
            vals = [r[m] for r in rs if r.get(m) is not None]
            if vals:
                d[m] = bootstrap_mean(vals, min(n_boot, 500), seed + 2)
        d["loop_rate"] = _rate([r["loop"] for r in rs])
        d["finished_rate"] = _rate([r["finished"] for r in rs])
        out[str(k)] = d
    return out


# ------------------------------------------------------------------------------------------------ tiers from the bake-off
def assign_tier(best: Optional[Dict[str, Any]], thresholds: Optional[Dict[str, Dict[str, float]]] = None, *, use_lower_bound: bool = False) -> Dict[str, Any]:
    """Confidence tier of a language's PSEUDO reference from the bake-off of its best open system on FLORES+ dev (human references):
        A  chrF++ >= th[A].chrf and (AfriCOMET-ref >= th[A].comet, or the language has no AfriCOMET score)
        B  same with th[B]
        C  below: reported as a low-confidence pseudo reference and NOT ranked
        N  no open system supports the language (best is None): human references only
    `best` = {"system", "chrf", "chrf_lo" (optional), "africomet" (optional), "n"}; `use_lower_bound` judges chrF++ by its CI lower bound (conservative)."""
    th = thresholds or DEFAULT_TIER_THRESHOLDS
    if not best or best.get("chrf") is None:
        return {"tier": "N", "system": None, "reason": "no open system covers the language"}
    chrf = best.get("chrf_lo", best["chrf"]) if use_lower_bound else best["chrf"]
    comet = best.get("africomet")
    tier, why = "C", f"chrF++ {chrf:.1f} below the B threshold {th['B']['chrf']}"
    for t in ("A", "B"):
        if chrf >= th[t]["chrf"] and (comet is None or comet >= th[t]["comet"]):
            tier, why = t, f"chrF++ {chrf:.1f} >= {th[t]['chrf']}" + ("" if comet is None else f", AfriCOMET {comet:.3f} >= {th[t]['comet']}")
            break
    else:
        if chrf >= th["B"]["chrf"] and comet is not None:
            why = f"chrF++ {chrf:.1f} ok but AfriCOMET {comet:.3f} below the B threshold {th['B']['comet']}"
    return {"tier": tier, "system": best.get("system"), "chrf": best["chrf"], "africomet": comet, "n": best.get("n"), "reason": why}


def load_tiers(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("tiers", d)


# ------------------------------------------------------------------------------------------------ whole run (many languages)
def evaluate_run(refdocs: Sequence[RefDoc], hyps: Dict[Tuple[str, str], Hyp], *, tiers: Optional[Dict[str, Dict[str, Any]]] = None,
                 scorer_ref: Optional[Callable] = None, scorer_qe: Optional[Callable] = None, n_boot: int = 1000, seed: int = 0, penalty: float = 0.0,
                 keep_rows: bool = False, log: Callable[[str], None] = lambda m: None) -> Dict[str, Any]:
    """All languages. The pseudo and the human reference sets are scored SEPARATELY (a language appears once per group). When a language has pseudo references of
    SEVERAL translators, the primary one (the system the tiers file chose, else the first by name) is `run["pseudo"][iso]` and the others are scored too, under
    `run["pseudo_alt"][iso][system]` (the agreement of a model with each pseudo-reference system). `hyps` keyed by (iso, doc_id)."""
    tiers = tiers or {}
    pseudo: Dict[str, Dict[str, List[RefDoc]]] = defaultdict(lambda: defaultdict(list))
    human: Dict[str, List[RefDoc]] = defaultdict(list)
    for d in refdocs:
        if d.ref_kind == "pseudo":
            pseudo[d.iso][d.ref_system].append(d)
        else:
            human[d.iso].append(d)
    out: Dict[str, Any] = {"pseudo": {}, "pseudo_alt": {}, "human": {}}

    def one(iso, docs, tier, keep):
        per_lang = {d.doc_id: hyps[(iso, d.doc_id)] for d in docs if (iso, d.doc_id) in hyps}
        return evaluate_language(docs, per_lang, iso=iso, scorer_ref=scorer_ref, scorer_qe=scorer_qe, n_boot=n_boot, seed=seed, penalty=penalty, tier=tier, keep_rows=keep)

    for iso, by_sys in sorted(pseudo.items()):
        want = (tiers.get(iso) or {}).get("system")
        primary = want if want in by_sys else sorted(by_sys)[0]
        res = one(iso, by_sys[primary], tiers.get(iso), keep_rows)
        res["ref_system"] = primary
        out["pseudo"][iso] = res
        for name in sorted(by_sys):
            if name != primary:
                alt = one(iso, by_sys[name], tiers.get(iso), False)
                alt["ref_system"] = name
                out["pseudo_alt"].setdefault(iso, {})[name] = alt
        log(f"[longdoc] pseudo/{iso} ({primary}): {len(by_sys[primary])} docs chrF++ {res['chrf']['value']:.1f}")
    for iso, docs in sorted(human.items()):
        out["human"][iso] = one(iso, docs, {"tier": "H", "reason": "human references"}, keep_rows)
        log(f"[longdoc] human/{iso}: {len(docs)} docs chrF++ {out['human'][iso]['chrf']['value']:.1f}")
    out["macro"] = macro_average(out)
    return out


def macro_average(run: Dict[str, Any]) -> Dict[str, Any]:
    """Unweighted mean over languages for the headline metrics: pseudo-reference languages of the RANKED tiers (A, B), all pseudo languages, human."""
    def mean_of(group: Dict[str, Any], key: str, only_ranked: bool = False):
        vals = []
        for iso, r in group.items():
            if only_ranked and not r.get("ranked", True):
                continue
            v = r.get(key)
            v = v["value"] if isinstance(v, dict) else v
            if v is not None and v == v:
                vals.append(v)
        return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)

    keys = ("chrf", "bleu", "africomet_ref", "africomet_qe", "coverage_chars", "loop_rate", "finished_rate")
    res: Dict[str, Any] = {}
    for name, group, ranked in (("pseudo_ranked", run["pseudo"], True), ("pseudo_all", run["pseudo"], False), ("human", run["human"], False)):
        res[name] = {}
        for k in keys:
            v, n = mean_of(group, k, ranked)
            if n:
                res[name][k] = {"value": v, "n_languages": n}
    return res


def report_markdown(run: Dict[str, Any], names: Optional[Dict[str, str]] = None) -> str:
    """Per-language table (pseudo and human groups), the metrics with 95% CIs, the tier and the failure metrics."""
    f = lambda d, p=1: "-" if not isinstance(d, dict) or d.get("value") != d.get("value") else f"{d['value']:.{p}f}"
    fc = lambda d, p=1: "-" if not isinstance(d, dict) or d.get("value") != d.get("value") else (f"{d['value']:.{p}f} [{d['lo']:.{p}f}, {d['hi']:.{p}f}]" if d.get("lo") == d.get("lo") else f"{d['value']:.{p}f}")
    out = []
    for grp in ("pseudo", "human"):
        rs = run.get(grp) or {}
        if not rs:
            continue
        out += [f"### {grp} references", "",
                "| lang | tier | docs | chrF++ [95% CI] | BLEU | AfriCOMET-ref | AfriCOMET-QE | coverage | loops | finished | len ratio |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for iso, r in sorted(rs.items()):
            nm = (names or {}).get(iso, iso)
            out.append(f"| {nm} (`{iso}`) | {r.get('tier', '-')}{'' if r.get('ranked', True) else ' (unranked)'} | {r['n_docs']} | {fc(r['chrf'])} | {f(r['bleu'])} | "
                       f"{f(r.get('africomet_ref'), 3)} | {f(r.get('africomet_qe'), 3)} | {f(r['coverage_chars'], 3)} | {f(r['loop_rate'], 3)} | "
                       f"{f(r['finished_rate'], 3)} | {f({'value': r['length_ratio'].get('mean', float('nan'))}, 2)} |")
        out.append("")
    alt = run.get("pseudo_alt") or {}
    if alt:
        out += ["### the same outputs against the pseudo references of the other translators", "", "| lang | primary reference | other reference | chrF++ [95% CI] vs primary | chrF++ [95% CI] vs other |", "|---|---|---|---|---|"]
        for iso, d in sorted(alt.items()):
            for name, r in sorted(d.items()):
                prim = run["pseudo"][iso]
                out.append(f"| `{iso}` | {prim.get('ref_system', '-')} | {name} | {fc(prim['chrf'])} | {fc(r['chrf'])} |")
        out.append("")
    out += ["### macro averages (mean over languages)", "", "| group | metric | value | languages |", "|---|---|---|---|"]
    for g, ms in run.get("macro", {}).items():
        for k, v in ms.items():
            out.append(f"| {g} | {k} | {v['value']:.3f} | {v['n_languages']} |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------------------------ tokenizer helper (Gemma token counts for the length bins)
class TokenCounter:
    """Gemma token counter: `tc(text)` -> int, `tc.many([texts])` -> [int] (batched in Rust; set RAYON_NUM_THREADS to bound its threads on a shared machine)."""

    def __init__(self, tokenizer):
        self.tok = tokenizer

    def __call__(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def many(self, texts: Sequence[str]) -> List[int]:
        return [len(e.ids) for e in self.tok.encode_batch(list(texts), add_special_tokens=False)]


def load_token_counter(tokenizer_json: Optional[str] = None) -> TokenCounter:
    """Number of Gemma tokens of a text (no BOS) from a `tokenizer.json` (default: the one in the Hugging Face cache of google/gemma-3-270m-it)."""
    import glob

    from tokenizers import Tokenizer

    path = tokenizer_json
    if not path:
        hits = sorted(glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub", "models--google--gemma-3-270m-it", "snapshots", "*", "tokenizer.json")))
        if not hits:
            raise FileNotFoundError("tokenizer.json of google/gemma-3-270m-it not found; pass --tokenizer")
        path = hits[-1]
    return TokenCounter(Tokenizer.from_file(path))


def load_afro_counter(tokenizer_json: Optional[str] = None) -> TokenCounter:
    """Token counter of the AfriCOMET encoder (Davlan/afro-xlmr-large-76L, no special tokens): the unit of the RL length bins b64 / b256."""
    import glob

    from tokenizers import Tokenizer

    path = tokenizer_json
    if not path:
        hits = sorted(glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub", "models--Davlan--afro-xlmr-large-76L", "snapshots", "*", "tokenizer.json")))
        if not hits:
            raise FileNotFoundError("tokenizer.json of Davlan/afro-xlmr-large-76L not found; pass --afro-tokenizer")
        path = hits[-1]
    return TokenCounter(Tokenizer.from_file(path))


def bin_of(n_tokens: int) -> int:
    """Nominal bin of an English source of `n_tokens` Gemma tokens (0 if outside every accepted range)."""
    for b, (lo, hi) in BIN_RANGES.items():
        if lo <= n_tokens <= hi:
            return b
    return 0


# ------------------------------------------------------------------------------------------------ CLI
def _make_scorer(kind: str, python: Optional[str], pythonpath: Optional[str], reference_based: bool, batch_size: int):
    from gemma_mt.rl import comet as C

    if kind == "none":
        return None
    if kind == "fake":
        return C.FakeScorer()
    if kind == "inproc":
        return C.CometScorer(C.REF_MODEL_ID if reference_based else C.QE_MODEL_ID, batch_size=batch_size, reference_based=reference_based)
    if kind == "worker":
        return C.WorkerScorer(python=python or sys.executable, model_id=C.REF_MODEL_ID if reference_based else C.QE_MODEL_ID, batch_size=batch_size,
                              pythonpath=pythonpath, reference_based=reference_based, gpu=False)
    raise ValueError(kind)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score", help="score the full-document translations of one system")
    sc.add_argument("--docs", required=True, help="docs.jsonl of the English benchmark documents")
    sc.add_argument("--refs", required=True, nargs="+", help="per-language reference files (rows doc_id, iso, ref_sents, ref_kind, ...)")
    sc.add_argument("--preds", required=True, help="jsonl rows {doc_id, iso, hyp, finished?, system?}")
    sc.add_argument("--tiers", default=None, help="tiers.json written by the bake-off")
    sc.add_argument("--out", required=True)
    sc.add_argument("--comet-ref", default="none", choices=["none", "fake", "inproc", "worker"])
    sc.add_argument("--comet-qe", default="none", choices=["none", "fake", "inproc", "worker"])
    sc.add_argument("--python", default=None)
    sc.add_argument("--pythonpath", default=None)
    sc.add_argument("--batch-size", type=int, default=32)
    sc.add_argument("--n-boot", type=int, default=1000)
    sc.add_argument("--penalty", type=float, default=0.0)
    sc.add_argument("--only-valid", action="store_true", help="drop pseudo-reference documents with > 5%% flagged reference sentences")
    sc.add_argument("--keep-rows", action="store_true")
    a = ap.parse_args(argv)
    refdocs = []
    for p in a.refs:
        refdocs += load_refdocs(a.docs, p, only_valid=a.only_valid)
    hyps = load_hyps(a.preds)
    sref = _make_scorer(a.comet_ref, a.python, a.pythonpath, True, a.batch_size)
    sqe = _make_scorer(a.comet_qe, a.python, a.pythonpath, False, a.batch_size)
    run = evaluate_run(refdocs, hyps, tiers=load_tiers(a.tiers), scorer_ref=sref, scorer_qe=sqe, n_boot=a.n_boot, penalty=a.penalty, keep_rows=a.keep_rows,
                       log=lambda m: print(m, flush=True))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=1)
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_markdown(run))
    print(report_markdown(run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
