# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-sentence windows for the inter-African SFT, the same idea as `gemma_windows.py` of the en->X build (D-?? owner 2026-10-12: "SFT must not be
single sentences"): sentence pairs of one language pair are concatenated, SOURCE SIDE AND TARGET SIDE IN THE SAME ORDER, joined by a single space,
until the next pair would push either side over the window's token budget. The windows are balanced over the length bins b64 / b256 / b512 / b1024
by TOKENS (25 % of a pair's tokens in each bin), not by count.

Differences to the en->X build: (1) token lengths are Gemma tokens (what the SFT sees; the RL prompts keep AfroXLM-R tokens because the QE scorer
limits them), (2) the unit that is concatenated is a pair from the SAME pivot class (so a window never mixes a Bible verse with a government statement
unless the caller puts them in one run), (3) pairs where only one side ends in a sentence terminator stay standalone (no sentence boundary is invented).
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

BINS = {"b64": (24, 65), "b256": (65, 257), "b512": (257, 513), "b1024": (513, 1025)}
CYCLE = ("b64", "b256", "b512", "b1024")
TERM = re.compile(r"[.!?።…][\"'”’»)\]}]*\s*$")


def bin_of(n: int) -> str:
    for b in CYCLE:
        if n < BINS[b][1]:
            return b
    return "b1024"


def build_windows(src: Sequence[str], tgt: Sequence[str], src_tok: Sequence[int], tgt_tok: Sequence[int], rng: np.random.Generator,
                  used: Dict[str, int] = None, add_stop: bool = True) -> List[dict]:
    """Greedy windows over an ordered run of pairs. Returns dicts: src, tgt, n_pairs, src_tokens, tgt_tokens, bin, punct_added.
    `used` (bin -> tokens so far) is the running token balance of the language pair; pass the same dict for successive runs of that pair."""
    used = used if used is not None else {b: 0 for b in CYCLE}
    out: List[dict] = []
    cur = None

    def new_budget() -> Tuple[int, str]:
        b = min(CYCLE, key=lambda x: (used[x], CYCLE.index(x)))
        lo, hi = BINS[b]
        budget = int(rng.integers(lo, hi))
        used[b] += budget
        return budget, b

    def emit(w):
        n = max(w["ts"], w["tt"]) + 2
        b = bin_of(n)
        if "budget" in w:
            used[w["bin"]] -= w["budget"]
        used[b] += n
        out.append({"src": " ".join(w["s"]), "tgt": " ".join(w["t"]), "n_pairs": len(w["s"]), "src_tokens": w["ts"] + 2, "tgt_tokens": w["tt"] + 2,
                    "bin": b, "punct_added": w["pa"]})

    for s, t, ls, lt in zip(src, tgt, src_tok, tgt_tok):
        a_term, b_term = bool(TERM.search(s)), bool(TERM.search(t))
        if a_term != b_term:                                   # one side without a sentence end: standalone, unchanged
            if cur:
                emit(cur)
                cur = None
            emit({"s": [s], "t": [t], "ts": int(ls), "tt": int(lt), "pa": False})
            continue
        added = False
        if not a_term and add_stop:
            s, t, added = s.rstrip() + ".", t.rstrip() + ".", True
            ls, lt = ls + 1, lt + 1
        if cur is not None and (cur["ts"] + ls + 2 > cur["budget"] or cur["tt"] + lt + 2 > cur["budget"]):
            emit(cur)
            cur = None
        if cur is None:
            budget, b = new_budget()
            cur = {"s": [], "t": [], "ts": 0, "tt": 0, "pa": False, "budget": budget, "bin": b}
        cur["s"].append(s)
        cur["t"].append(t)
        cur["ts"] += int(ls)
        cur["tt"] += int(lt)
        cur["pa"] = cur["pa"] or added
    if cur:
        emit(cur)
    return out
