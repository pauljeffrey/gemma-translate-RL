# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text metrics (sacrebleu) and aggregation. No model, no torch."""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Sequence

TEXT_METRICS = ("chrf", "spbleu", "bleu")
COMET_METRICS = ("comet", "comet_qe")
WORD_METRICS = ("exact",)                  # word-level lexicon benchmarks (GATITOS): share of rows whose output equals an accepted translation
ALL_METRICS = TEXT_METRICS + COMET_METRICS + WORD_METRICS


def nfc(texts: Sequence[str]) -> List[str]:
    """Unicode NFC. Reference sets are not uniformly composed (FLORES+ Yoruba ~28% and Fon ~37% of the sentences carry decomposed
    sequences, measured by another agent 2026-10-14) while the model emits composed text; without this the character n-grams of
    the same sentence differ and chrF++ / BLEU are artificially low for those languages."""
    return [unicodedata.normalize("NFC", t) for t in texts]


def chrf_pp(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """Corpus chrF++ (character n-gram 6, word n-gram 2), 0..100. Hypotheses and references are NFC-normalised first."""
    from sacrebleu.metrics import CHRF

    return float(CHRF(word_order=2).corpus_score(nfc(hyps), [nfc(refs)]).score)


def chrf_pp_multi(hyps: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    """Corpus chrF++ with SEVERAL accepted references per row (sacrebleu scores each segment against its best reference). Rows with fewer
    references are padded by repeating their first one."""
    from sacrebleu.metrics import CHRF

    width = max((len(r) for r in refs), default=1)
    streams = [nfc([(r[k] if k < len(r) else r[0]) for r in refs]) for k in range(width)]
    return float(CHRF(word_order=2).corpus_score(nfc(hyps), streams).score)


_WORD_STRIP = " \t\r\n.,;:!?\"'\u201c\u201d\u2018\u2019\u00ab\u00bb()[]{}\u2026"


def normalise_word(text: str) -> str:
    """Key of the exact-match test: NFC, case-folded, surrounding whitespace / quotes / sentence punctuation removed (the model often adds a full stop)."""
    return unicodedata.normalize("NFC", text or "").strip(_WORD_STRIP).casefold()


def exact_match(hyps: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    """Share (0..1) of rows whose hypothesis equals ANY accepted reference after `normalise_word`."""
    if not hyps:
        return float("nan")
    hit = 0
    for h, rs in zip(hyps, refs):
        key = normalise_word(h)
        hit += bool(key) and key in {normalise_word(r) for r in rs}
    return hit / len(hyps)


def bleu(hyps: Sequence[str], refs: Sequence[str], tokenize: str = "13a") -> float:
    from sacrebleu.metrics import BLEU

    return float(BLEU(tokenize=tokenize).corpus_score(nfc(hyps), [nfc(refs)]).score)


_SPBLEU_TOK: Optional[str] = None


def spbleu(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """BLEU on the FLORES-200 SentencePiece tokenization (sacrebleu tokenize="flores200"). The tokenizer model is
    downloaded by sacrebleu on first use; if that is impossible (offline) the score falls back to the "13a"
    tokenizer and `spbleu_tokenizer()` says so. Never mix the two in one comparison."""
    global _SPBLEU_TOK
    if _SPBLEU_TOK in (None, "flores200"):
        try:
            v = bleu(hyps, refs, tokenize="flores200")
            _SPBLEU_TOK = "flores200"
            return v
        except Exception:
            _SPBLEU_TOK = "13a"
    return bleu(hyps, refs, tokenize="13a")


def spbleu_tokenizer() -> str:
    return _SPBLEU_TOK or "flores200"


def compute_text_metrics(hyps: Sequence[str], refs: Sequence[str]) -> Dict[str, float]:
    if len(hyps) != len(refs):
        raise ValueError("hyps and refs differ in length")
    if not hyps:
        return {}
    return {"chrf": chrf_pp(hyps, refs), "spbleu": spbleu(hyps, refs), "bleu": bleu(hyps, refs)}


def mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if x is not None]
    return float(sum(xs) / len(xs)) if xs else float("nan")


def macro_average(per_language: Dict[str, Dict[str, float]], metrics: Sequence[str] = ALL_METRICS) -> Dict[str, float]:
    """Unweighted mean over languages (each language counts once), per metric, over the languages that have it."""
    out: Dict[str, float] = {}
    for m in metrics:
        vals = [d[m] for d in per_language.values() if m in d and d[m] is not None and d[m] == d[m]]
        if vals:
            out[m] = float(sum(vals) / len(vals))
            out[f"{m}_n_languages"] = len(vals)
    return out
