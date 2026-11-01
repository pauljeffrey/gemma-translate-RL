# SPDX-License-Identifier: AGPL-3.0-or-later
"""Leak filter against the evaluation sets (they are the benchmarks of the en->X model AND the inter-African benchmark: nothing of them may be trained on).

Policy (D-46): a candidate pair / prompt is DROPPED when any of its texts (the English or French pivot sentence, the source text, the target text)
  (a) equals a benchmark sentence in ANY language of ANY held-out set after `text.pivot_key` normalisation (exact rule; texts with fewer than
      `min_words` words are only compared when the benchmark entry itself has at least that many words, so one-word GATITOS entries cannot delete half of
      the corpus), or
  (b) shares at least `ngram_frac` of its word 8-grams with the benchmark 8-gram set (near-duplicate rule, for texts with >= 8 words; catches a benchmark
      sentence embedded in a longer window).
The 8-gram test looks at the text on its own, whatever its language: FLORES+/NTREX/SMOL are multi-way, so a Hausa training sentence that is the
FLORES+ Hausa line is as much a leak for the inter-African benchmark as the English line.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from gemma_mt.afri2afri.text import hash64, pivot_key, shingles

# subdirectories / files of the benchmark root and how to read them
BENCH_SETS = ("flores_plus", "ntrex", "mafand", "smol_sent", "smol_doc", "gatitos", "ibom_mt", "flores_moz", "glot_story", "salt")


def _jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def iter_benchmark_texts(root: str, extra_jsonl: Sequence[str] = ()) -> Iterator[Tuple[str, str, str]]:
    """Yield (benchmark, language label, text) for every sentence of every held-out set under `root` (the box layout `bench/test_sets`).
    `extra_jsonl`: more files (FLORES+ DEV copies, `bench_dev.jsonl`) read with the same key guesses."""
    # FLORES+ style: <set>/{devtest,dev}/<code>.jsonl with `text`
    for sub in ("flores_plus", "flores_moz"):
        for p in sorted(glob.glob(os.path.join(root, sub, "*", "*.jsonl"))):
            lang = os.path.basename(p)[:-6]
            for r in _jsonl(p):
                if r.get("text"):
                    yield sub, lang, r["text"]
    for p in sorted(glob.glob(os.path.join(root, "ntrex", "*", "newstest2019-*.txt"))):
        lang = os.path.basename(p).split(".")[-2]
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield "ntrex", lang, line
    for p in sorted(glob.glob(os.path.join(root, "mafand", "*", "*.json"))):
        for r in _jsonl(p):
            for lg, t in (r.get("translation") or {}).items():
                if t:
                    yield "mafand", lg, t
    for sub in ("smol_sent", "gatitos", "ibom_mt", "glot_story", "salt"):
        for p in sorted(glob.glob(os.path.join(root, sub, "*.jsonl"))):
            lang = os.path.basename(p)[:-6]
            for r in _jsonl(p):
                for k in ("src", "trg"):
                    if r.get(k):
                        yield sub, lang, r[k]
                for t in r.get("trgs") or ():
                    yield sub, lang, t
    for p in sorted(glob.glob(os.path.join(root, "smol_doc", "*.jsonl"))):
        lang = os.path.basename(p)[:-6]
        for r in _jsonl(p):
            for k in ("srcs", "trgs"):
                for t in r.get(k) or ():
                    if t:
                        yield "smol_doc", lang, t
    for p in extra_jsonl:
        lang = os.path.basename(p)[:-6]
        for r in _jsonl(p):
            for k in ("text", "english", "reference", "src", "trg"):
                if isinstance(r.get(k), str) and r.get(k):
                    yield os.path.basename(os.path.dirname(p)) or "extra", r.get("iso") or lang, r[k]
            for sent in r.get("sents") or ():            # long-document benchmark: documents carry their sentences
                if isinstance(sent, str) and sent:
                    yield os.path.basename(os.path.dirname(p)) or "extra", lang, sent


class LeakIndex:
    """Hash sets of the normalised benchmark sentences and of their word n-grams."""

    def __init__(self, n: int = 8, min_words: int = 3, ngram_frac: float = 0.5):
        self.n, self.min_words, self.ngram_frac = n, min_words, ngram_frac
        self.exact: Set[int] = set()
        self.grams: Set[int] = set()
        self.count = 0
        self.by_bench: Dict[str, int] = {}

    def add(self, text: str, bench: str = "") -> None:
        k = pivot_key(text)
        if not k:
            return
        self.count += 1
        self.by_bench[bench] = self.by_bench.get(bench, 0) + 1
        if len(k.split()) >= self.min_words:
            self.exact.add(hash64(k))
        for h in shingles(text, self.n):
            self.grams.add(h)

    def add_all(self, items: Iterable[Tuple[str, str, str]]) -> "LeakIndex":
        for bench, _lang, text in items:
            self.add(text, bench)
        return self

    def exact_hit(self, text: str) -> bool:
        return hash64(pivot_key(text)) in self.exact

    def ngram_share(self, text: str) -> float:
        sh = shingles(text, self.n)
        if not sh:
            return 0.0
        return sum(1 for h in sh if h in self.grams) / len(sh)

    def is_leak(self, text: str) -> Optional[str]:
        """None when clean, else the rule that fired ('exact' or 'ngram')."""
        if self.exact_hit(text):
            return "exact"
        if self.ngram_share(text) >= self.ngram_frac:
            return "ngram"
        return None

    def key_hits(self, keys: Sequence[int]) -> List[bool]:
        """Vectorisable shortcut for already-hashed pivot keys."""
        return [k in self.exact for k in keys]
