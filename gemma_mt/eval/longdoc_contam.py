# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contamination / de-duplication scanner for the candidate documents of the long-context translation benchmark.

For every candidate English document it measures how much of it occurs in

    sft    the English side of the SFT training stream (`gemma/packed_fwd`, TOKEN space: the text windows were deleted)
    rl     the English windows of the RL prompts (`gemma/rl_prompts_v2/<iso>.parquet`, WORD space)
    bench  the English text of the existing benchmarks (FLORES+ dev/devtest, NTREX, MAFAND, SMOL, ...; WORD space)
    dup    the other pool documents (WORD space; pairwise, against documents of higher priority only)

Method (all four use the same primitive): a document is turned into a sequence of units (words after the `contamination.normalise` normalisation, or Gemma
token ids), every window of `n` consecutive units is hashed to a uint64 (vectorised polynomial rolling hash over splitmix64-mixed unit ids), and the
fraction of the document's windows that occur in the other side is its `share`; the longest run of consecutive matching windows gives the longest
verbatim stretch (`run` = matches + n - 1 units). The CANDIDATE side is hashed once and kept as a sorted unique uint64 array; the (huge) training side is
STREAMED in chunks and every chunk is looked up with `np.searchsorted` (no python set of hundreds of millions of ints). A hit marks the candidate hash
as "seen"; the per-document statistics are then read back through the document's own hash positions.

Units. Word path: NFKC + casefold + every run of non-alphanumerics -> one space, n = 8 words (same as `contamination.py`). Token path: the document's
whitespace is collapsed to single spaces (training windows are single-line: no newline token appears in the English side of the stream), tokenised
alone with the Gemma tokenizer (`add_special_tokens=False`), n = `TOKEN_NGRAM` tokens. Windows never cross a training SAMPLE boundary; across the
paragraph boundaries of a candidate they do (a paragraph break is a plain space in the training text).

Packed layout (see `gemma_mt.data.prepare_sft` / `packed_dataset`): sample i occupies tokens[sample_offset[i] : +sample_len[i]]
    <bos><start_of_turn>user\\nTranslate the text to {Language}:  {english}<end_of_turn>\\n<start_of_turn>model\\n | {iso} {target}<end_of_turn>\\n
`prompt_len` ends right after `model\\n`; the English is what lies between the double-space token (id 138, the only one in the header) and the fixed
5-token tail `<end_of_turn> \\n <start_of_turn> model \\n`. numpy is required; pyarrow is imported only inside the parquet readers.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from gemma_mt.eval.contamination import NGRAM, normalise

WORD_NGRAM = NGRAM                    # 8 words, as contamination.py
TOKEN_NGRAM = 12                      # Gemma tokens; justified in reports/longdoc_benchmark_plan_2026-10-20_parts/contamination.md
_M64 = (1 << 64) - 1
_BASE = 0xD6E8FEB86659FD93            # odd multiplier of the polynomial hash
_C0, _C1, _C2 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB   # splitmix64 constants

# --------------------------------------------------------------------------------------------------- hashing primitives
_U = np.uint64


def mix64_int(x: int) -> int:
    """splitmix64 finaliser on a python int (reference implementation of `mix64`)."""
    z = (x + _C0) & _M64
    z = ((z ^ (z >> 30)) * _C1) & _M64
    z = ((z ^ (z >> 27)) * _C2) & _M64
    return z ^ (z >> 31)


def mix64(x: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 finaliser: a bijection of uint64 that spreads small integers over all 64 bits."""
    z = np.asarray(x, dtype=np.uint64) + _U(_C0)
    z = (z ^ (z >> _U(30))) * _U(_C1)
    z = (z ^ (z >> _U(27))) * _U(_C2)
    return z ^ (z >> _U(31))


_TOKEN_TABLE: Optional[np.ndarray] = None
_TOKEN_TABLE_SIZE = 1 << 19


def token_ids64(tokens: np.ndarray) -> np.ndarray:
    """token ids -> mixed uint64 (a table lookup for ids < 2**19, i.e. every Gemma id)."""
    global _TOKEN_TABLE
    a = np.asarray(tokens)
    if a.size == 0:
        return np.zeros(0, dtype=np.uint64)
    if int(a.max()) < _TOKEN_TABLE_SIZE:
        if _TOKEN_TABLE is None:
            _TOKEN_TABLE = mix64(np.arange(_TOKEN_TABLE_SIZE, dtype=np.uint64))
        return _TOKEN_TABLE[a]
    return mix64(a.astype(np.uint64))


class _WordIds(dict):
    """word -> stable uint64 id (blake2b, 8 bytes): the same in every process and every run, unlike python's hash()."""

    def __missing__(self, w: str) -> int:
        v = int.from_bytes(hashlib.blake2b(w.encode("utf-8"), digest_size=8).digest(), "little")
        if len(self) > 6_000_000:        # bound the memory of a long-lived worker
            self.clear()
        self[w] = v
        return v


def new_word_cache() -> Dict[str, int]:
    return _WordIds()


def word_ids64(words: Sequence[str], cache: Optional[Dict[str, int]] = None) -> np.ndarray:
    cache = cache if cache is not None else new_word_cache()
    if not words:
        return np.zeros(0, dtype=np.uint64)
    return np.fromiter((cache[w] for w in words), dtype=np.uint64, count=len(words))


def naive_ngram_hash(unit_ids64: Sequence[int]) -> int:
    """Reference (python int) value of one window hash: `((x0*B + x1)*B + x2) ... mod 2**64`."""
    h = 0
    for x in unit_ids64:
        h = (h * _BASE + int(x)) & _M64
    return h


def rolling_hashes(ids64: np.ndarray, n: int) -> np.ndarray:
    """Hash of every window of `n` consecutive items (len - n + 1 values), Horner form, uint64 wrap-around arithmetic."""
    ids64 = np.asarray(ids64, dtype=np.uint64)
    m = len(ids64) - n + 1
    if n < 1:
        raise ValueError("n must be >= 1")
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = ids64[:m].copy()
    base = _U(_BASE)
    for j in range(1, n):
        h *= base
        h += ids64[j: j + m]
    return h


def region_ngram_hashes(ids64: np.ndarray, lengths: Sequence[int], n: int) -> np.ndarray:
    """Windows of `n` items that lie INSIDE one region; `ids64` is the concatenation of the regions (their lengths in `lengths`). The windows come out in
    order, region by region; region r contributes max(length_r - n + 1, 0) of them."""
    lengths = np.asarray(lengths, dtype=np.int64)
    h = rolling_hashes(ids64, n)
    if h.size == 0:
        return h
    starts = np.cumsum(lengths) - lengths
    ok = lengths >= n
    diff = np.zeros(len(h) + 1, dtype=np.int64)
    np.add.at(diff, starts[ok], 1)
    np.add.at(diff, starts[ok] + lengths[ok] - n + 1, -1)
    return h[np.cumsum(diff[:-1]) > 0]


def counts_per_region(lengths: Sequence[int], n: int) -> np.ndarray:
    return np.maximum(np.asarray(lengths, dtype=np.int64) - n + 1, 0)


# --------------------------------------------------------------------------------------------------- membership
def mark_hits(uniq: np.ndarray, hit: np.ndarray, queries: np.ndarray, block: int = 1 << 24) -> int:
    """For every query found in the sorted unique array `uniq`, set hit[position] = 1 (hit: uint8/bool array of len(uniq), may be a shared buffer).
    Returns the number of matching queries."""
    if uniq.size == 0 or queries.size == 0:
        return 0
    total = 0
    last = len(uniq) - 1
    for s in range(0, len(queries), block):
        q = queries[s: s + block]
        idx = np.searchsorted(uniq, q)
        np.minimum(idx, last, out=idx)
        m = uniq[idx] == q
        hit[idx[m]] = 1
        total += int(m.sum())
    return total


def membership(sorted_unique: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """bool array: query in sorted_unique."""
    if sorted_unique.size == 0:
        return np.zeros(len(queries), dtype=bool)
    idx = np.searchsorted(sorted_unique, queries)
    idx[idx == len(sorted_unique)] = 0
    return sorted_unique[idx] == queries


def longest_run(b: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D bool array."""
    if b.size == 0 or not b.any():
        return 0
    x = np.concatenate(([0], b.astype(np.int8), [0]))
    d = np.diff(x)
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def run_units(run_ngrams: int, n: int) -> int:
    """n-gram run -> number of units it spans (r matches in a row cover r + n - 1 units)."""
    return run_ngrams + n - 1 if run_ngrams > 0 else 0


def hit_stats(hit: np.ndarray, n: int) -> Tuple[float, int, int]:
    """(share of True, longest run in UNITS, number of True) of one document's per-window hit array; an empty array gives (0, 0, 0)."""
    if hit.size == 0:
        return 0.0, 0, 0
    k = int(hit.sum())
    return k / hit.size, run_units(longest_run(hit), n), k


# --------------------------------------------------------------------------------------------------- documents
_PARA_SPLIT = re.compile(r"\n\s*\n")


def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in _PARA_SPLIT.split(text or "") if p.strip()]


def collapse_ws(text: str) -> str:
    return " ".join((text or "").split())


@dataclass
class PreparedDoc:
    id: str
    n_words: int
    word_ids: np.ndarray                       # uint64 id of each normalised word, paragraphs back to back
    para_bounds: List[Tuple[int, int]]         # [start, end) word indices of each paragraph that has words
    tok_ids: Optional[np.ndarray] = None       # Gemma token ids of the whitespace-collapsed text (None if the token path is off)
    n_tokens: int = 0
    words_hash: Optional[np.ndarray] = None    # word n-gram hashes (filled by `prepare_docs`)
    tok_hash: Optional[np.ndarray] = None      # token n-gram hashes


def prepare_docs(records: Sequence[Dict[str, Any]], encode: Optional[Callable[[List[str]], List[List[int]]]] = None, word_n: int = WORD_NGRAM,
                 token_n: int = TOKEN_NGRAM, cache: Optional[Dict[str, int]] = None) -> List[PreparedDoc]:
    """records: dicts with `id` and `text`. `encode(list of str) -> list of token-id lists` enables the token path (None = word path only)."""
    cache = cache if cache is not None else new_word_cache()
    docs: List[PreparedDoc] = []
    for r in records:
        ids_parts, bounds, pos = [], [], 0
        for p in split_paragraphs(r.get("text", "")):
            w = normalise(p).split()
            if not w:
                continue
            ids_parts.append(word_ids64(w, cache))
            bounds.append((pos, pos + len(w)))
            pos += len(w)
        wid = np.concatenate(ids_parts) if ids_parts else np.zeros(0, dtype=np.uint64)
        d = PreparedDoc(id=str(r["id"]), n_words=int(pos), word_ids=wid, para_bounds=bounds)
        d.words_hash = rolling_hashes(wid, word_n)
        docs.append(d)
    if encode is not None and docs:
        attach_tokens(docs, records, encode, token_n)
    return docs


def attach_tokens(docs: Sequence[PreparedDoc], records: Sequence[Dict[str, Any]], encode: Callable[[List[str]], List[List[int]]],
                  token_n: int = TOKEN_NGRAM, batch: int = 512) -> None:
    """Token path of already prepared documents: the whitespace-collapsed text is tokenised alone and its token n-gram hashes stored."""
    for s in range(0, len(docs), batch):
        toks = encode([collapse_ws(r.get("text", "")) for r in records[s: s + batch]])
        for d, t in zip(docs[s: s + batch], toks):
            d.tok_ids = np.asarray(t, dtype=np.uint32)
            d.n_tokens = len(t)
            d.tok_hash = rolling_hashes(token_ids64(d.tok_ids), token_n)


class CandidateIndex:
    """All n-gram hashes of a list of documents as ONE sorted unique array (`uniq`) plus, for every document, the positions of its windows in `uniq`."""

    def __init__(self, per_doc_hashes: Sequence[np.ndarray]):
        self.counts = np.array([len(h) for h in per_doc_hashes], dtype=np.int64)
        self.offsets = np.concatenate(([0], np.cumsum(self.counts)))
        allh = np.concatenate(per_doc_hashes) if len(per_doc_hashes) else np.zeros(0, dtype=np.uint64)
        self.uniq = np.unique(allh)
        self.inv = np.searchsorted(self.uniq, allh).astype(np.int64)

    def new_hit(self) -> np.ndarray:
        return np.zeros(len(self.uniq), dtype=np.uint8)

    def doc_hits(self, hit: np.ndarray, i: int) -> np.ndarray:
        """bool array (one per window of document i): the window was seen."""
        return hit[self.inv[self.offsets[i]: self.offsets[i + 1]]].astype(bool)


def benchmark_hash_array(texts: Iterable[str], n: int = WORD_NGRAM, cache: Optional[Dict[str, int]] = None) -> np.ndarray:
    """Sorted unique word n-gram hashes of a collection of English texts (each text hashed on its own) PLUS the n-grams that cross the boundary between two
    consecutive texts (FLORES / SMOL rows come in document order, so a candidate that copies two neighbouring sentences is matched across the join)."""
    cache = cache if cache is not None else new_word_cache()
    parts: List[np.ndarray] = []
    prev: Optional[np.ndarray] = None
    for t in texts:
        w = normalise(t).split()
        if not w:
            continue
        ids = word_ids64(w, cache)
        parts.append(rolling_hashes(ids, n))
        if prev is not None and n > 1:
            parts.append(rolling_hashes(np.concatenate([prev[-(n - 1):], ids[: n - 1]]), n))
        prev = ids
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.uint64)


# --------------------------------------------------------------------------------------------------- the packed SFT stream
@dataclass(frozen=True)
class PromptLayout:
    sep_id: int = 138                                       # the double-space token right before the English
    tail: Tuple[int, ...] = (106, 107, 105, 4368, 107)      # <end_of_turn> \n <start_of_turn> model \n
    max_header: int = 24


def english_bounds(slab: np.ndarray, rel_offset: np.ndarray, prompt_len: np.ndarray, layout: PromptLayout = PromptLayout()
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For samples starting at `rel_offset` (indices into `slab`, a contiguous slice of the token stream) return (start, end, valid): the English text of
    sample i is slab[start[i]:end[i]] when valid[i]. A sample is invalid when the separator is not in its first `max_header` tokens or the 5-token tail does
    not follow the English (a layout other than the one documented in the module docstring)."""
    m = len(rel_offset)
    W = layout.max_header
    idx = rel_offset[:, None] + np.arange(W)[None, :]
    np.minimum(idx, len(slab) - 1, out=idx)
    is_sep = slab[idx] == layout.sep_id
    has = is_sep.any(1)
    start = rel_offset + is_sep.argmax(1) + 1
    end = rel_offset + prompt_len.astype(np.int64) - len(layout.tail)
    valid = has & (end > start)
    ok_idx = np.flatnonzero(valid)
    for k, t in enumerate(layout.tail):
        pos = np.minimum(end[ok_idx] + k, len(slab) - 1)
        valid[ok_idx[slab[pos] != t]] = False
    return start, end, valid


def gather_regions(slab: np.ndarray, start: np.ndarray, end: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate slab[start:end] for all regions (vectorised): (tokens, lengths)."""
    lengths = (end - start).astype(np.int64)
    total = int(lengths.sum())
    if total == 0:
        return slab[:0], lengths
    first = np.cumsum(lengths) - lengths
    pos = np.arange(total, dtype=np.int64) + np.repeat(start.astype(np.int64) - first, lengths)
    return slab[pos], lengths


class PackedSplit:
    """Read-only view of one split directory of `gemma/packed_fwd` (train/ or val/)."""

    def __init__(self, split_dir: str):
        self.dir = split_dir
        ld = lambda n: np.load(os.path.join(split_dir, n), mmap_mode="r")
        self.sample_offset, self.sample_len, self.prompt_len = ld("sample_offset.npy"), ld("sample_len.npy"), ld("prompt_len.npy")
        p = os.path.join(split_dir, "direction.npy")
        self.direction = ld("direction.npy") if os.path.exists(p) else None
        self.iso_id = ld("iso_id.npy") if os.path.exists(os.path.join(split_dir, "iso_id.npy")) else None
        self.tokens = np.memmap(os.path.join(split_dir, "tokens.u32.bin"), dtype=np.uint32, mode="r")
        self.n_samples = len(self.sample_len)

    def chunk(self, a: int, b: int, scope: str = "english", layout: PromptLayout = PromptLayout(), direction: Optional[int] = 0
              ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Samples [a, b): (tokens, region lengths, n invalid). scope='english' -> the English side of the en2x samples (direction==0), 'all' -> the whole sample
        (every direction)."""
        so = np.asarray(self.sample_offset[a:b], dtype=np.int64)
        sl = np.asarray(self.sample_len[a:b], dtype=np.int64)
        lo, hi = int(so[0]), int(so[-1] + sl[-1])
        slab = np.asarray(self.tokens[lo:hi])
        rel = so - lo
        if scope == "all":
            return gather_regions(slab, rel, rel + sl) + (0,)
        pl = np.asarray(self.prompt_len[a:b], dtype=np.int64)
        start, end, valid = english_bounds(slab, rel, pl, layout)
        keep = valid.copy()
        if direction is not None and self.direction is not None:
            keep &= np.asarray(self.direction[a:b]) == direction
        n_invalid = int((~valid).sum()) if direction is None or self.direction is None else int(((~valid) & (np.asarray(self.direction[a:b]) == direction)).sum())
        toks, lengths = gather_regions(slab, start[keep], end[keep])
        return toks, lengths, n_invalid


def english_region_hashes(split: PackedSplit, a: int, b: int, n: int, scope: str = "english") -> Tuple[np.ndarray, int, int, int]:
    """(window hashes, n tokens scanned, n regions, n invalid samples) of samples [a, b)."""
    toks, lengths, bad = split.chunk(a, b, scope)
    if toks.size == 0:
        return np.zeros(0, dtype=np.uint64), 0, 0, bad
    return region_ngram_hashes(token_ids64(toks), lengths, n), int(toks.size), int(len(lengths)), bad


# --------------------------------------------------------------------------------------------------- duplicates within the pool
def find_duplicates(hash_lists: Sequence[np.ndarray], rank: Sequence[int], threshold: float = 0.5, max_group: int = 25
                    ) -> Tuple[List[int], np.ndarray]:
    """Near-duplicate detection between documents. hash_lists[i]: window hashes of document i. rank[i]: priority (lower = kept, "earlier").
    For every document j: dup_of[j] = the document i with rank[i] < rank[j] that shares the largest number of unique windows with j (-1 if none),
    share[j] = that number / unique windows of j. (dup_of is only set when share >= threshold; share is reported always.)
    Windows contained in more than `max_group` documents (boilerplate: licence footers, stock phrases) are ignored."""
    D = len(hash_lists)
    uh = [np.unique(h) for h in hash_lists]
    sizes = np.array([len(h) for h in uh], dtype=np.int64)
    dup_of = [-1] * D
    share = np.zeros(D, dtype=np.float64)
    if D < 2 or sizes.sum() == 0:
        return dup_of, share
    rank = np.asarray(rank, dtype=np.int64)
    allh = np.concatenate(uh)
    doc = np.repeat(np.arange(D, dtype=np.int64), sizes)
    order = np.lexsort((rank[doc], allh))
    allh, doc = allh[order], doc[order]
    change = np.flatnonzero(np.concatenate(([True], allh[1:] != allh[:-1], [True])))
    gstart, gsize = change[:-1], np.diff(change)
    keys: List[np.ndarray] = []
    for k in np.unique(gsize[(gsize >= 2) & (gsize <= max_group)]):
        S = gstart[gsize == k]
        ia, ja = np.triu_indices(int(k), 1)
        early = doc[(S[:, None] + ia[None, :]).ravel()]
        late = doc[(S[:, None] + ja[None, :]).ravel()]
        keys.append(early * D + late)
    if not keys:
        return dup_of, share
    uk, cnt = np.unique(np.concatenate(keys), return_counts=True)
    early, late = uk // D, uk % D
    best = {}
    for e, l, c in zip(early.tolist(), late.tolist(), cnt.tolist()):
        if l not in best or c > best[l][1] or (c == best[l][1] and rank[e] < rank[best[l][0]]):
            best[l] = (e, c)
    for l, (e, c) in best.items():
        share[l] = c / max(int(sizes[l]), 1)
        if share[l] >= threshold:
            dup_of[l] = e
    return dup_of, share


# --------------------------------------------------------------------------------------------------- flag rules
@dataclass
class Thresholds:
    """clean = every share < clean_below; weak = max share < weak_below; otherwise contaminated. A verbatim run of >= run_words words (word path: rl, bench) or
    >= run_tokens tokens (sft), or dup_share >= dup_share_max, makes the document contaminated whatever its shares."""
    clean_below: float = 0.02
    weak_below: float = 0.10
    run_words: int = 50
    run_tokens: int = 64
    dup_share_max: float = 0.5
    paragraph_exact_max: Optional[int] = None     # >= this many verbatim paragraphs => contaminated (None: informational only)
    bench_hit_min_ngrams: int = 8                 # a benchmark counts as hit when >= this many windows of the document occur in it


def flag_row(row: Dict[str, Any], th: Thresholds = Thresholds()) -> Tuple[str, List[str]]:
    """(flag, reasons) of one per-document result row (keys as in the module's output schema)."""
    reasons: List[str] = []
    shares = {k: float(row.get(k, 0.0) or 0.0) for k in ("sft_share", "sft_word_share", "rl_share", "bench_share", "dup_share")}
    if (row.get("sft_longest_run", 0) or 0) >= th.run_tokens:
        reasons.append(f"sft_run>={th.run_tokens}tok")
    if (row.get("sft_word_longest_run", 0) or 0) >= th.run_words:
        reasons.append(f"sft_word_run>={th.run_words}w")
    for k in ("rl", "bench"):
        if (row.get(f"{k}_longest_run", 0) or 0) >= th.run_words:
            reasons.append(f"{k}_run>={th.run_words}w")
    if shares["dup_share"] >= th.dup_share_max:
        reasons.append("dup")
    if th.paragraph_exact_max is not None and (row.get("paragraph_exact", 0) or 0) >= th.paragraph_exact_max:
        reasons.append("paragraphs")
    mx = max(shares.values())
    top = max(shares, key=shares.get)
    if mx >= th.weak_below:
        reasons.append(f"{top}={mx:.3f}>={th.weak_below}")
    if reasons:
        return "contaminated", reasons
    if mx >= th.clean_below:
        return "weak", [f"{top}={mx:.3f}"]
    return "clean", []


def quantiles(values: Sequence[float], qs: Sequence[float] = (0.5, 0.9, 0.95, 0.99, 1.0)) -> Dict[str, float]:
    if len(values) == 0:
        return {}
    a = np.asarray(values, dtype=np.float64)
    return {f"q{int(q * 100)}": round(float(np.quantile(a, q)), 5) for q in qs}


# --------------------------------------------------------------------------------------------------- per-document aggregation
def paragraph_exact_count(hit: np.ndarray, para_bounds: Sequence[Tuple[int, int]], n: int, min_words: int = 12) -> int:
    """Number of paragraphs (>= `min_words` words) all of whose windows are in `hit` (the per-window bool array of the document): the paragraph occurs
    verbatim (up to the 64-bit hash) in the other side. Windows are indexed by the word position where they start."""
    c = 0
    for a, b in para_bounds:
        if b - a < max(min_words, n):
            continue
        if hit[a: b - n + 1].all():
            c += 1
    return c


def summarise_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Flag counts, share quantiles and per-benchmark hit counts of a list of per-document result rows."""
    out: Dict[str, Any] = {"n_docs": len(rows), "flags": {}, "quantiles": {}, "bench_hits": {}}
    for r in rows:
        out["flags"][r["flag"]] = out["flags"].get(r["flag"], 0) + 1
        for b in r.get("bench_hit_benchmarks") or []:
            out["bench_hits"][b] = out["bench_hits"].get(b, 0) + 1
    for k in ("sft_share", "sft_word_share", "rl_share", "bench_share", "dup_share"):
        vals = [r[k] for r in rows if k in r and r[k] is not None]
        if vals:
            out["quantiles"][k] = quantiles(vals)
    for k in ("sft_longest_run", "rl_longest_run", "bench_longest_run"):
        vals = [r[k] for r in rows if k in r]
        if vals:
            out["quantiles"][k] = quantiles(vals)
    return out


def atomic_write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------------------------------- text sources (RL windows, benchmarks)
def iter_parquet_column(path: str, column: str = "english", batch_size: int = 50_000) -> Iterator[str]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=[column]):
        for s in batch.column(0).to_pylist():
            if s:
                yield s


def text_chunk_hashes(texts: Iterable[str], n: int = WORD_NGRAM, cache: Optional[Dict[str, int]] = None, batch: int = 4000) -> Iterator[np.ndarray]:
    """Word n-gram hashes of independent texts, yielded per batch of `batch` texts (windows never cross a text)."""
    cache = cache if cache is not None else new_word_cache()
    buf_ids: List[np.ndarray] = []
    buf_len: List[int] = []
    for t in texts:
        w = normalise(t).split()
        if len(w) < n:
            continue
        buf_ids.append(word_ids64(w, cache))
        buf_len.append(len(w))
        if len(buf_len) >= batch:
            yield region_ngram_hashes(np.concatenate(buf_ids), buf_len, n)
            buf_ids, buf_len = [], []
    if buf_len:
        yield region_ngram_hashes(np.concatenate(buf_ids), buf_len, n)


def file_fingerprint(path: str) -> Dict[str, Any]:
    st = os.stat(path)
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


# --------------------------------------------------------------------------------------------------- parallel streaming scans
import multiprocessing as mp   # noqa: E402

_G: Dict[str, Any] = {}


def _shared_hit(size: int):
    raw = mp.RawArray("B", max(int(size), 1))
    return raw, np.frombuffer(raw, dtype=np.uint8)[: int(size)]


def _sft_worker(task: Tuple[str, int, int]) -> Tuple[int, int, int]:
    split_dir, a, b = task
    splits = _G.setdefault("splits", {})
    if split_dir not in splits:
        splits[split_dir] = PackedSplit(split_dir)
    toks, lengths, bad = splits[split_dir].chunk(a, b, _G["scope"])
    if toks.size:
        ids = token_ids64(toks)
        for n, (uniq, hit) in _G["targets"].items():
            mark_hits(uniq, hit, region_ngram_hashes(ids, lengths, n))
    return int(toks.size), int(len(lengths)), bad


def scan_sft_stream(split_dirs: Sequence[str], targets: Dict[int, np.ndarray], processes: int = 8, chunk_samples: int = 100_000, scope: str = "english",
                    log: Callable[[str], None] = print, max_samples: Optional[int] = None) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    """Stream the packed SFT splits and mark which of the candidate hashes occur. targets: {n: sorted unique uint64 candidate hashes of n-token windows}.
    Returns ({n: uint8 hit array aligned with targets[n]}, stats). Several n are served by one pass over the data."""
    tasks: List[Tuple[str, int, int]] = []
    for sd in split_dirs:
        ns = len(np.load(os.path.join(sd, "sample_len.npy"), mmap_mode="r"))
        if max_samples:
            ns = min(ns, max_samples)
        tasks += [(sd, a, min(a + chunk_samples, ns)) for a in range(0, ns, chunk_samples)]
    raws, hits = {}, {}
    for n, uniq in targets.items():
        raws[n], hits[n] = _shared_hit(len(uniq))
    _G["targets"] = {n: (targets[n], hits[n]) for n in targets}
    _G["scope"] = scope
    t0 = time.time()
    tot = np.zeros(3, dtype=np.int64)
    if processes <= 1:
        results = map(_sft_worker, tasks)
        pool = None
    else:
        pool = mp.get_context("fork").Pool(processes)
        results = pool.imap_unordered(_sft_worker, tasks)
    for k, r in enumerate(results, 1):
        tot += np.asarray(r, dtype=np.int64)
        if k % max(1, len(tasks) // 10) == 0 or k == len(tasks):
            log(f"[sft-scan] {k}/{len(tasks)} chunks, {tot[0] / 1e6:.0f} M tokens, {time.time() - t0:.0f}s")
    if pool is not None:
        pool.close()
        pool.join()
    stats = {"tokens": int(tot[0]), "regions": int(tot[1]), "invalid_samples": int(tot[2]), "seconds": round(time.time() - t0, 1), "chunks": len(tasks),
             "scope": scope}
    return {n: np.array(hits[n], copy=True) for n in hits}, stats


def _text_worker(task: Tuple[str, str]) -> Tuple[str, int, int]:
    path, column = task
    uniq, hit, n = _G["text_uniq"], _G["text_hit"], _G["text_n"]
    cache = _G.setdefault("cache", new_word_cache())
    seen = set()
    n_rows = n_unique = 0

    def gen():
        nonlocal n_rows, n_unique
        for s in iter_parquet_column(path, column):
            n_rows += 1
            h = hash(s)
            if h in seen:
                continue
            seen.add(h)
            n_unique += 1
            yield s

    for hs in text_chunk_hashes(gen(), n, cache):
        mark_hits(uniq, hit, hs)
    return path, n_rows, n_unique


def scan_parquet_texts(files: Sequence[str], uniq: np.ndarray, column: str = "english", n: int = WORD_NGRAM, processes: int = 8,
                       log: Callable[[str], None] = print) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Word-n-gram scan of the `column` of parquet files (one worker task per file; exact duplicate texts inside a file are scanned once)."""
    raw, hit = _shared_hit(len(uniq))
    _G.update(text_uniq=uniq, text_hit=hit, text_n=n)
    t0 = time.time()
    rows = unique = 0
    tasks = [(f, column) for f in files]
    if processes <= 1:
        results, pool = map(_text_worker, tasks), None
    else:
        pool = mp.get_context("fork").Pool(min(processes, max(1, len(files))))
        results = pool.imap_unordered(_text_worker, tasks)
    for k, (p, r, u) in enumerate(results, 1):
        rows += r
        unique += u
        if k % max(1, len(tasks) // 8) == 0 or k == len(tasks):
            log(f"[text-scan] {k}/{len(tasks)} files, {time.time() - t0:.0f}s")
    if pool is not None:
        pool.close()
        pool.join()
    return np.array(hit, copy=True), {"files": len(files), "rows": rows, "unique_in_file": unique, "seconds": round(time.time() - t0, 1)}


def load_tokenizers_json(path_or_glob: str):
    """`tokenizers.Tokenizer` from a tokenizer.json path (glob allowed)."""
    from tokenizers import Tokenizer

    p = sorted(glob.glob(path_or_glob))
    return Tokenizer.from_file(p[0] if p else path_or_glob)


def make_encoder(tk) -> Callable[[List[str]], List[List[int]]]:
    return lambda texts: [e.ids for e in tk.encode_batch(list(texts), add_special_tokens=False)]


def _sftword_worker(task: Tuple[str, int, int]) -> Tuple[int, int, int]:
    split_dir, a, b = task
    splits = _G.setdefault("splits", {})
    if split_dir not in splits:
        splits[split_dir] = PackedSplit(split_dir)
    if "tk" not in _G:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"       # the parent used rayon before the fork: parallelism in a forked child would deadlock
        _G["tk"] = load_tokenizers_json(_G["tk_path"])
    toks, lengths, bad = splits[split_dir].chunk(a, b, "english")
    if toks.size:
        cuts = np.cumsum(lengths)[:-1]
        texts = _G["tk"].decode_batch([c.tolist() for c in np.split(toks, cuts)], skip_special_tokens=False)
        cache = _G.setdefault("cache", new_word_cache())
        for hs in text_chunk_hashes(texts, _G["sw_n"], cache):
            mark_hits(_G["sw_uniq"], _G["sw_hit"], hs)
    return int(toks.size), int(len(lengths)), bad


def scan_sft_words(split_dirs: Sequence[str], uniq: np.ndarray, tokenizer_path: str, n: int = WORD_NGRAM, processes: int = 8, chunk_samples: int = 50_000,
                   log: Callable[[str], None] = print, max_samples: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """WORD-space scan of the SFT English side: every English region of the packed stream is DECODED back to text with the Gemma tokenizer, normalised and
    hashed like the RL / benchmark text. Robust to formatting differences that change the tokenisation (quotes, hyphens, spaces before punctuation)."""
    tasks: List[Tuple[str, int, int]] = []
    for sd in split_dirs:
        ns = len(np.load(os.path.join(sd, "sample_len.npy"), mmap_mode="r"))
        if max_samples:
            ns = min(ns, max_samples)
        tasks += [(sd, a, min(a + chunk_samples, ns)) for a in range(0, ns, chunk_samples)]
    raw, hit = _shared_hit(len(uniq))
    _G.update(sw_uniq=uniq, sw_hit=hit, sw_n=n, tk_path=tokenizer_path)
    _G.pop("tk", None)
    t0 = time.time()
    tot = np.zeros(3, dtype=np.int64)
    if processes <= 1:
        results, pool = map(_sftword_worker, tasks), None
    else:
        pool = mp.get_context("fork").Pool(processes)
        results = pool.imap_unordered(_sftword_worker, tasks)
    for k, r in enumerate(results, 1):
        tot += np.asarray(r, dtype=np.int64)
        if k % max(1, len(tasks) // 10) == 0 or k == len(tasks):
            log(f"[sft-word-scan] {k}/{len(tasks)} chunks, {tot[0] / 1e6:.0f} M tokens, {time.time() - t0:.0f}s")
    if pool is not None:
        pool.close()
        pool.join()
    return np.array(hit, copy=True), {"tokens": int(tot[0]), "regions": int(tot[1]), "invalid_samples": int(tot[2]), "seconds": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------------------------------- existing benchmark English
def _jsonl_field(path: str, getter: Callable[[Dict[str, Any]], Optional[str]]) -> Iterator[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = getter(json.loads(line))
            except (ValueError, KeyError, TypeError):
                continue
            if s:
                yield s


def load_benchmark_english(bench_root: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, List[str]]:
    """{benchmark name: English texts in file order (deduplicated)} of every English side found under `bench_root` (the layout of `bench/test_sets`).
    `extra`: {name: path of a jsonl with an `english` column} (bench_dev.jsonl, the FLORES+ dev file ...). Benchmarks without the directory are skipped."""
    out: Dict[str, List[str]] = {}

    def add(name: str, texts: Iterable[str]) -> None:
        seen, lst = set(), out.setdefault(name, [])
        for t in texts:
            if t not in seen:
                seen.add(t)
                lst.append(t)

    j = os.path.join
    for name in ("flores_plus", "flores_moz"):
        for p in sorted(glob.glob(j(bench_root, name, "*", "eng_Latn.jsonl"))):
            add(name, _jsonl_field(p, lambda r: r["text"]))
    for p in sorted(glob.glob(j(bench_root, "ntrex", "*", "newstest2019-src.eng.txt")) + glob.glob(j(bench_root, "ntrex", "newstest2019-src.eng.txt"))):
        with open(p, encoding="utf-8") as f:
            add("ntrex", (ln.strip() for ln in f))
    for p in sorted(glob.glob(j(bench_root, "mafand", "*", "*.json"))):
        add("mafand", _jsonl_field(p, lambda r: r["translation"]["en"]))
    for p in sorted(glob.glob(j(bench_root, "smol_doc", "en_*.jsonl"))):
        add("smol_doc", _jsonl_field(p, lambda r: " ".join(r["srcs"]) if r.get("sl", "en") == "en" else None))
    for name in ("smol_sent", "glot_story", "salt", "ibom_mt"):
        for p in sorted(glob.glob(j(bench_root, name, "en_*.jsonl"))):
            add(name, _jsonl_field(p, lambda r: r["src"] if r.get("sl", "en") == "en" else None))
    for name, path in (extra or {}).items():
        if os.path.isfile(path):
            add(name, _jsonl_field(path, lambda r: r["english"]))
    return {k: v for k, v in out.items() if v}


# --------------------------------------------------------------------------------------------------- the scan of a batch of prepared documents
@dataclass
class Sources:
    """Where the training / benchmark side lives. Any field may be empty/None: the corresponding columns are then omitted from the result."""
    sft_dirs: Sequence[str] = ()                       # split directories of the packed SFT dataset (train/, val/)
    sft_tokenizer: Optional[str] = None                # tokenizer.json (needed for the decoded word-space SFT scan)
    rl_files: Sequence[str] = ()                       # parquet files with an `english` column
    rl_texts: Sequence[str] = ()                       # in-memory alternative / addition (tests)
    bench: Dict[str, Sequence[str]] = field(default_factory=dict)      # benchmark name -> English texts


@dataclass
class ScanConfig:
    word_n: int = WORD_NGRAM
    token_n: int = TOKEN_NGRAM
    processes: int = 8
    sft_scope: str = "english"
    sft_words: bool = True                             # also scan the decoded English of the SFT stream in word space
    paragraph_min_words: int = 12
    max_sft_samples: Optional[int] = None


def scan_documents(docs: Sequence[PreparedDoc], src: Sources, cfg: ScanConfig = ScanConfig(), log: Callable[[str], None] = print
                   ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Per-document scan statistics (no duplicate detection, no flag) and a dict of timings / source statistics.
    Row keys: id, n_tokens, n_words, n_ngrams_tok, n_ngrams_word, sft_share, sft_longest_run, sft_word_share, sft_word_longest_run, rl_share, rl_longest_run,
    bench_share, bench_longest_run, bench_hit_benchmarks, bench_hit_ngrams, paragraph_exact."""
    info: Dict[str, Any] = {}
    D = len(docs)
    wn, tn = cfg.word_n, cfg.token_n
    wcand = CandidateIndex([d.words_hash if d.words_hash is not None else np.zeros(0, np.uint64) for d in docs])
    have_tok = all(d.tok_hash is not None for d in docs) and D > 0
    tcand = CandidateIndex([d.tok_hash for d in docs]) if have_tok else None
    hits: Dict[str, np.ndarray] = {}
    # existing benchmarks: the small side is the benchmark, the lookup is a plain membership test
    bench_hit = np.zeros(len(wcand.uniq), dtype=np.uint8)
    bench_each: Dict[str, np.ndarray] = {}
    t0 = time.time()
    for name, texts in src.bench.items():
        arr = benchmark_hash_array(texts, wn)
        h = membership(arr, wcand.uniq).astype(np.uint8)
        bench_each[name] = h
        bench_hit |= h
    info["bench_seconds"] = round(time.time() - t0, 1)
    info["bench_sizes"] = {k: len(v) for k, v in src.bench.items()}
    # RL windows
    rl_hit = None
    if len(src.rl_files) or len(src.rl_texts):
        rl_hit = np.zeros(len(wcand.uniq), dtype=np.uint8)
        if len(src.rl_files):
            h, st = scan_parquet_texts(list(src.rl_files), wcand.uniq, n=wn, processes=cfg.processes, log=log)
            rl_hit |= h
            info["rl"] = st
        if len(src.rl_texts):
            for hs in text_chunk_hashes(src.rl_texts, wn):
                mark_hits(wcand.uniq, rl_hit, hs)
    # SFT stream: token space (+ decoded word space)
    sft_hit = None
    if len(src.sft_dirs) and tcand is not None:
        res, st = scan_sft_stream(list(src.sft_dirs), {tn: tcand.uniq}, processes=cfg.processes, scope=cfg.sft_scope, log=log, max_samples=cfg.max_sft_samples)
        sft_hit = res[tn]
        info["sft"] = st
    sftw_hit = None
    if len(src.sft_dirs) and cfg.sft_words and src.sft_tokenizer:
        sftw_hit, st = scan_sft_words(list(src.sft_dirs), wcand.uniq, src.sft_tokenizer, n=wn, processes=cfg.processes, log=log, max_samples=cfg.max_sft_samples)
        info["sft_word"] = st
    rows: List[Dict[str, Any]] = []
    for i, d in enumerate(docs):
        row: Dict[str, Any] = {"id": d.id, "n_tokens": int(d.n_tokens), "n_words": int(d.n_words), "n_ngrams_word": int(wcand.counts[i])}
        if tcand is not None:
            row["n_ngrams_tok"] = int(tcand.counts[i])
            if sft_hit is not None:
                s, r, _ = hit_stats(tcand.doc_hits(sft_hit, i), tn)
                row["sft_share"], row["sft_longest_run"] = round(s, 5), r
        if sftw_hit is not None:
            s, r, _ = hit_stats(wcand.doc_hits(sftw_hit, i), wn)
            row["sft_word_share"], row["sft_word_longest_run"] = round(s, 5), r
        if rl_hit is not None:
            s, r, _ = hit_stats(wcand.doc_hits(rl_hit, i), wn)
            row["rl_share"], row["rl_longest_run"] = round(s, 5), r
        row["bench_hit_benchmarks"], row["bench_hit_ngrams"] = [], {}
        if src.bench:
            hb = wcand.doc_hits(bench_hit, i)
            s, r, _ = hit_stats(hb, wn)
            row["bench_share"], row["bench_longest_run"] = round(s, 5), r
            for name, h in bench_each.items():
                k = int(wcand.doc_hits(h, i).sum())
                if k:
                    row["bench_hit_ngrams"][name] = k
            row["bench_hit_benchmarks"] = sorted(row["bench_hit_ngrams"], key=lambda b: -row["bench_hit_ngrams"][b])
        # paragraphs that occur verbatim in RL / benchmark English (all windows of the paragraph seen)
        union = np.zeros(int(wcand.counts[i]), dtype=bool)
        for h in (rl_hit, bench_hit if src.bench else None):
            if h is not None:
                union |= wcand.doc_hits(h, i)
        row["paragraph_exact"] = paragraph_exact_count(union, d.para_bounds, wn, cfg.paragraph_min_words)
        rows.append(row)
    return rows, info


def finalize_rows(scan_rows: Sequence[Dict[str, Any]], dup_of: Sequence[int], dup_share: Sequence[float], ids: Sequence[str],
                  th: Thresholds = Thresholds()) -> List[Dict[str, Any]]:
    """Add dup_of / dup_share / flag / flag_reasons to the scan rows (same order as ids)."""
    out = []
    for i, r in enumerate(scan_rows):
        r = dict(r)
        r["dup_of"] = ids[dup_of[i]] if dup_of[i] >= 0 else ""
        r["dup_share"] = round(float(dup_share[i]), 5)
        r["bench_hit_benchmarks"] = [b for b in r.get("bench_hit_benchmarks", []) if r.get("bench_hit_ngrams", {}).get(b, 0) >= th.bench_hit_min_ngrams]
        r["flag"], r["flag_reasons"] = flag_row(r, th)
        out.append(r)
    return out
