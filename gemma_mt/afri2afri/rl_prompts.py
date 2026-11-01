# SPDX-License-Identifier: AGPL-3.0-or-later
"""RL prompts for the inter-African model (D-46): SOURCE-language windows with a target-language tag.

File layout = the `rl_prompts_v2` family of the en->X model (one parquet per TARGET language, so `PromptSchedule` can draw languages by n ** alpha and
bins by stage), with `english` renamed `source` and three new columns:

    source           str    source-language text (a window of whole sentences, joined by single spaces)
    src_iso          str    canonical tag of the source language           (NEW)
    iso              str    canonical tag of the TARGET language (the tag the model has to write)
    language         str    fixed instruction name of the target language
    src_language     str    fixed instruction name of the source language  (NEW)
    src_afr_tokens   int    length of `source` in Davlan/afro-xlmr-large-76L tokens with special tokens (the AfriCOMET encoder), as in rl_prompts_v2
    length_bin       str    b64 [24,64] | b256 [65,256] | b512 [257,512] | b1024 [513,1024] | (b2048 [1025,2048] reserved)
    n_sentences      int
    source_dataset   str
    licence          str
    src_qe_valid / tgt_qe_valid   bool  the language is QE_VALIDATED / QE_LANGUAGES in `languages.py` (decides the reward family of the direction)

Windows come from documents (`clean_doc` + `windows_from_doc`, ported from `gemma_rl_windows.py` of the en->X build) or from the held-out pair texts.
`assign_targets` spreads the windows of one (source language, bin) over the target languages with a cyclic walk, so every window is reused about equally often
and a (window, target) pair never repeats.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gemma_mt.rl.segments import split_sentences

BINS = {"b64": (24, 65), "b256": (65, 257), "b512": (257, 513), "b1024": (513, 1025)}      # AfriCOMET tokens, upper bound in the name
ORDER = ("b1024", "b512", "b256", "b64")
RL_X2X_COLUMNS = ("source", "src_iso", "iso", "language", "src_language", "src_afr_tokens", "length_bin", "n_sentences", "source_dataset", "licence",
                  "src_qe_valid", "tgt_qe_valid")
HEAD = re.compile(r"^[^.!?።]{0,60}$")
BAD = re.compile(r"https?://|www\.|\{\{|\}\}|\|\||&[a-z]+;|<[a-z/]+|\[\[|edit source|\bclick\b", re.I)


def bin_of_tokens(n: int) -> Optional[str]:
    for b, (lo, hi) in BINS.items():
        if lo <= n < hi:
            return b
    return None


def clean_doc(text: str, min_sentence_chars: int = 12) -> List[str]:
    """Document -> whole clean sentences (section titles, short headings, markup debris removed). Same rules as the en->X RL windows."""
    keep = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or (len(line.split()) < 7 and HEAD.match(line)):
            continue
        keep.append(line)
    text = re.sub(r"\(\s*[;,:]*\s*\)", "", " ".join(keep))
    text = re.sub(r"\s+([,.;:!?])", r"\1", re.sub(r"\s{2,}", " ", text))
    return [s for s in split_sentences(text) if len(s) >= min_sentence_chars and not BAD.search(s)]


def windows_from_doc(sents: Sequence[str], rng: np.random.Generator, count_tokens: Callable[[List[str]], List[int]], chars_per_token: float = 4.0,
                     max_per_doc: int = 4, p_take: Optional[Mapping[str, float]] = None) -> List[dict]:
    """Up to `max_per_doc` windows of consecutive whole sentences per document, one per bin at most, never overlapping."""
    p_take = dict(p_take or {"b1024": 0.55, "b512": 0.4, "b256": 0.3, "b64": 0.2})
    lens = np.fromiter((len(s) + 1 for s in sents), dtype=np.int64, count=len(sents))
    cum = np.concatenate([[0], np.cumsum(lens)])
    total_tok = cum[-1] / chars_per_token
    out, used = [], []
    for b in ORDER:
        if len(out) >= max_per_doc:
            break
        lo, hi = BINS[b]
        if total_tok < lo * 1.05 or rng.random() > p_take[b]:
            continue
        budget = rng.uniform(lo, hi - 1)
        need = budget * chars_per_token
        starts = np.nonzero(cum[-1] - cum[:-1] >= need * 0.9)[0]
        if len(starts) == 0:
            continue
        for _ in range(4):
            s0 = int(starts[rng.integers(len(starts))])
            if any(not (s0 >= e or s0 + 1 <= a) for a, e in used):
                continue
            e = min(max(int(np.searchsorted(cum, cum[s0] + need, side="right")) - 1, s0 + 1), len(sents))
            n = 0
            text = ""
            for _try in range(6):
                text = " ".join(sents[s0:e])
                n = count_tokens([text])[0]
                if n > hi - 1 and e - s0 > 1:
                    e = s0 + max(1, int((e - s0) * (hi - 1) / n) - (1 if _try > 2 else 0))
                    continue
                if n < lo and e < len(sents):
                    e = min(len(sents), e + max(1, int((e - s0) * (lo - n) / max(n, 1)) + 1))
                    continue
                break
            if lo <= n <= hi - 1 and not any(not (s0 >= e2 or e <= a) for a, e2 in used):
                out.append({"source": text, "src_afr_tokens": n, "length_bin": b, "n_sentences": e - s0})
                used.append((s0, e))
                break
    return out


def direction_weights(pair_alloc: Mapping[str, float], langs: Sequence[str], lo: float = 0.25, hi: float = 2.0) -> Dict[Tuple[str, str], float]:
    """Per ORDERED direction weight, mean 1 over the directions that exist, from the SFT allocation of the unordered pair (key 'a-b' with a < b in ALL_ISOS
    order): a pair UniMax gave few draws (little data) gets a proportionally smaller RL share, clipped to [lo, hi]. Pairs without an SFT allocation get `lo`."""
    raw: Dict[Tuple[str, str], float] = {}
    for x in langs:
        for y in langs:
            if x == y:
                continue
            a, b = (x, y) if (x + "-" + y) in pair_alloc else (y, x)
            raw[(x, y)] = float(pair_alloc.get(f"{a}-{b}", 0.0))
    vals = np.array([v for v in raw.values() if v > 0])
    mean = float(vals.mean()) if len(vals) else 1.0
    return {k: float(np.clip(v / mean, lo, hi)) if v > 0 else lo for k, v in raw.items()}


def assign_targets(n_rows: int, targets: Sequence[str], quota: Mapping[str, int], rng: np.random.Generator, max_reuse: int = 8) -> Dict[str, np.ndarray]:
    """For one (source language, bin) with `n_rows` windows: {target: row indices}. Each target gets `quota[target]` DISTINCT rows (capped at n_rows), and the
    walk over a random permutation is cyclic, so the number of times a window is used differs by at most one between windows. `max_reuse` scales all quotas down
    when the pool is too small (a window serving more than `max_reuse` targets would let the policy memorise it)."""
    total = sum(min(int(quota.get(t, 0)), n_rows) for t in targets)
    scale = min(1.0, max_reuse * n_rows / total) if total else 1.0
    perm = rng.permutation(n_rows)
    out: Dict[str, np.ndarray] = {}
    pos = 0
    for t in rng.permutation(len(targets)):
        tgt = targets[int(t)]
        q = int(round(min(int(quota.get(tgt, 0)), n_rows) * scale))
        if q <= 0:
            continue
        idx = (pos + np.arange(q)) % n_rows
        out[tgt] = perm[idx]
        pos = (pos + q) % n_rows
    return out
