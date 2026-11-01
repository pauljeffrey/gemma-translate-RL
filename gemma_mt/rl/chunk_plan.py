# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chunk planning and aggregation of the chunked reward (D-23 / D-24), as PURE functions.

This is the logic that used to be inline in `TranslationReward.__call__`, factored out so that exactly the same
semantics can be reused (and tested without a tokenizer, a scorer or a model):

    plan_chunks(source, text, ...)  ->  ChunkPlan   which (source, output) pairs the QE model must score, and which
                                                    parts of the text count as FAILED chunks (omitted / copied /
                                                    appended text), each with its weight
    aggregate(items, values, ...)   ->  float       the weighted mean of the per-chunk rewards (failed chunk = penalty)

The SabiYarn RL reward (repo root, `rl/chunking.py`) carries a file-for-file port of `segments.py` + this module;
`tests/test_chunking.py` in the repo root asserts that the two stay identical on a fuzz corpus.

Weights (`chunk_agg`): "source_length" = number of SOURCE characters of the chunk (the owner's rule, 2026-10-12);
"mean" = 1 per chunk. Failed chunks that stand for text with no source sentence (extra / appended output) weigh their
OUTPUT characters (resp. 1): they have no source side to be weighed by.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from gemma_mt.rl.guards import is_copy_of_source
from gemma_mt.rl.segments import LengthModel, align_beads, align_chunks, split_sentences

# (segment index into ChunkPlan.segments, or None = a FAILED chunk that gets the penalty, weight)
PlanItem = Tuple[Optional[int], float]


@dataclass
class ChunkPlan:
    segments: List[Tuple[str, str]]            # (source text, output text) pairs to score with the QE model
    items: List[PlanItem]                      # in order; the aggregation walks this list
    how: str = "whole"                         # off | single | aligned | gale_church | whole (align_chunks status)
    n_failed: int = 0                          # source sentences with no output at all (Gale-Church omissions)
    extra: List[str] = field(default_factory=list)   # reasons of the failed items that are not omissions (diagnostics)
    n_short: int = 0                           # beads flagged by the shortfall check (omission detector, D-30)
    flagged_src: List[int] = field(default_factory=list)   # source sentence indices flagged as dropped (omission beads + shortfall beads; detector only)


def plan_chunks(source: str, text: str, *, chunking: str = "sentences", chunk_align: str = "gale_church",
                chunk_agg: str = "source_length", extra_output_penalty: bool = True, extra_ratio_limit: float = 1.8,
                length_slack: float = 0.15, copy_jaccard: float = 0.9, w_hi: Optional[float] = None,
                extra_on_single: bool = True,
                is_copy: Callable[[str, str, float], bool] = is_copy_of_source,
                length_model: Optional[LengthModel] = None, short_frac: Optional[float] = None,
                short_min: float = 0.15, beads_fn: Optional[Callable] = None) -> ChunkPlan:
    """The scoring plan of one (English source, translation) pair.

    chunking "off": one segment = the whole pair (no sentence splitting, no extra-text rules).
    chunking "sentences": `align_chunks` (equal counts: i-th to i-th; different counts: Gale-Church; one output sentence
    for several source sentences: scored as a whole), then
      * text of the output that no source sentence is matched to ("extra" sentences, dropped by the alignment): one failed
        chunk weighing the unmatched characters, unless they are < 5% of the output (stray white space / punctuation)
      * inside every pair with both sides non-empty: characters beyond `w_hi * (1 + length_slack)` times the source length
        (`w_hi` = the language's human p95 length ratio from the calibration; `extra_ratio_limit` when uncalibrated) form
        a failed chunk when they are > 10% of the output side (QE does not see appended text)
      * with several pairs, an empty or copied output side is a failed chunk with the weight of its source side.
    `extra_on_single=False` skips the "inside every pair" rule when the source is a single sentence (status "single"):
    the SabiYarn port uses this to keep one-chunk texts numerically identical to whole-text scoring.

    Omission detector (D-30, off unless `length_model` is given; the SabiYarn port does not have it): only when the sentence counts of
    source and output DIFFER (equal counts / one source sentence are planned exactly as before, bit for bit)
      * `length_model`: the Gale-Church alignment uses the language's expected length ratio and variance and a constant cost for an
        omission / extra bead (segments.LengthModel), so a dropped sentence is found as an omission bead instead of being merged
        into its neighbour (2:1 bead); the omission bead is a failed chunk weighing the source characters of the dropped sentence(s)
      * `short_frac`: shortfall check on what the alignment still merged: a bead of >= 2 source sentences (or the single-output-sentence
        "whole" case) whose output is shorter than `short_frac * length_model.ratio * len(source side)` and whose missing share
        (source-equivalent characters `len(a) - len(b)/ratio`) is >= `short_min` of the bead is split into a scored chunk of the
        source characters that are covered and a FAILED chunk weighing the missing source characters (total weight unchanged).
    """
    if chunking != "sentences":
        return ChunkPlan([(source, text)], [(0, float(len(source)) if chunk_agg == "source_length" else 1.0)], "off", 0)
    n_src: List[int] = []                                     # sentences of the source side of each pair (omission detector only)
    bead_src: List[Tuple[int, ...]] = []
    if beads_fn is not None:                                  # D-60 (rl/longdoc.py): the alignment is supplied (anchored, redundant copies removed); same bead format as align_beads
        beads, how = beads_fn(source, text)
        pairs = [(a, b) for a, b, si, oi in beads if si]
        n_src = [len(si) for a, b, si, oi in beads if si]
        bead_src = [si for a, b, si, oi in beads if si]
        n_failed = sum(len(si) for a, b, si, oi in beads if si and not oi)
    else:
        pairs, n_failed, how = align_chunks(source, text, chunk_align)
    if beads_fn is None and length_model is not None and how in ("gale_church", "whole"):
        beads, how = align_beads(source, text, chunk_align, length_model)
        pairs = [(a, b) for a, b, si, oi in beads if si]      # extra-output beads (no source) are accounted for below, as before
        n_src = [len(si) for a, b, si, oi in beads if si]
        bead_src = [si for a, b, si, oi in beads if si]
        n_failed = sum(len(si) for a, b, si, oi in beads if si and not oi)
    plan = ChunkPlan([], [], how, n_failed)
    for k_pair, (a, b) in enumerate(pairs):                   # diagnostics: which source sentences the omission detector flagged
        if bead_src and not b.strip():
            plan.flagged_src.extend(bead_src[k_pair])
    if extra_output_penalty and len(pairs) > 1:
        total_out = len(" ".join(split_sentences(text)))
        matched = len(" ".join(b for _, b in pairs))
        extra = max(0, total_out - matched)
        if extra > 0 and extra >= 0.05 * max(total_out, 1):          # a few stray characters of white space / punctuation do not count
            plan.items.append((None, float(extra) if chunk_agg == "source_length" else 1.0))
            plan.extra.append("unmatched_output")
    for k_pair, (a, b) in enumerate(pairs):
        w = float(len(a)) if chunk_agg == "source_length" else 1.0
        miss_share = 0.0
        if (short_frac is not None and length_model is not None and n_src and a and b.strip()
                and (n_src[k_pair] >= 2 or how == "whole")):
            ratio = max(float(length_model.ratio), 1e-3)
            if len(b) < short_frac * ratio * len(a):
                miss = max(0.0, len(a) - len(b) / ratio)
                if miss >= short_min * len(a):
                    miss_share = miss / len(a)
        if extra_output_penalty and a and b and (extra_on_single or how != "single"):
            lim = w_hi * (1.0 + length_slack) if w_hi else extra_ratio_limit    # the language's human p95 when calibrated
            over = len(b) - lim * len(a)                                  # text appended inside a chunk (QE does not see it)
            if over > 0.10 * len(b):
                plan.items.append((None, float(over) if chunk_agg == "source_length" else 1.0))
                plan.extra.append("appended_text")
        if len(pairs) > 1 and (not b.strip() or is_copy(a, b, copy_jaccard)):
            plan.items.append((None, w))               # an omitted or copied sentence is a failed chunk, not a scored one
            plan.extra.append("omitted_or_copied")
            continue
        plan.segments.append((a, b))
        if miss_share > 0.0:                                   # shortfall: part of the source side has no counterpart in the output
            plan.items.append((len(plan.segments) - 1, w * (1.0 - miss_share)))
            plan.items.append((None, w * miss_share))
            plan.extra.append("shortfall")
            plan.n_short += 1
            plan.flagged_src.extend(bead_src[k_pair])
        else:
            plan.items.append((len(plan.segments) - 1, w))
    return plan


def aggregate(items: Sequence[PlanItem], seg_rewards: Sequence[float], penalty: float, exact_single: bool = False) -> float:
    """Weighted mean of the chunk rewards; a failed chunk (index None) contributes `penalty`. `seg_rewards[k]` is the
    reward of segment k. `exact_single`: a plan of one item returns its reward untouched (no w * r / w rounding).
    Zero total weight (a blank source) falls back to the plain mean instead of dividing by zero."""
    if exact_single and len(items) == 1:
        seg = items[0][0]
        return float(penalty if seg is None else seg_rewards[seg])
    tot = wsum = 0.0
    for seg, w in items:
        tot += w * (penalty if seg is None else seg_rewards[seg])
        wsum += w
    if wsum <= 0.0:
        vals = [penalty if seg is None else seg_rewards[seg] for seg, _ in items]
        return float(sum(vals) / len(vals)) if vals else float(penalty)
    return tot / wsum
