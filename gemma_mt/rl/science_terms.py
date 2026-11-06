# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference terms of the SCIENCE RL stage (owner 2026-11-03, arm 2: science SFT then RL): small, separately weighted terms added to the reward of rows that carry a HUMAN reference.

    ssa      reference-based SSA-COMET-MTL (src + mt + ref, `McGill-NLP/ssa-comet-mtl`; NOT africomet-stl-1.1, which stays the held-out monitor): + w_ssa * slope * (score - center)
    omit     chrF++ character / word n-gram RECALL against the reference below a floor (dropped content, omission):   - w_omit * (floor - recall) / floor
    add      chrF++ n-gram PRECISION below a floor (added or hallucinated content):                                  - w_add  * (floor - precision) / floor
    length   |log(len(hyp) / len(ref))| beyond a tolerance (characters without spaces):                              - w_len  * min(1, (|log ratio| - tol) / tol)
    preserve numbers, units, acronyms / formulas and named entities of the source that are missing from the translation (share of the source's items, judged only with >= min_items):
                                                                                                                     - w_pres * share_missing

Every term is PENALTY or small shaping, weights are small (default sum of the maxima < 2 reward units against a reward range of 4.25), the result is clipped so that a sample never falls below the
failure penalty, and nothing is applied to a sample that failed a hard check. Rates and means are returned per batch (`stats`) so that gaming can be watched (copy rate, length drift, preservation
judged / missing). Pure functions: no model, no I/O. Reuses the chrF++ n-gram machinery of `rl/consensus.py` and the entity extraction of `rl/anchors.py`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from gemma_mt.rl import consensus as cons
from gemma_mt.rl.anchors import _edit_le1, extract_entities, numkey

_UNITS = (r"mg|µg|ug|kg|g|ml|mL|dl|dL|l|L|mm|cm|km|m|mol|mmol|µmol|nm|ha|kPa|Pa|°C|°F|K|Hz|kHz|MHz|W|kW|MW|V|mV|A|mA|kcal|kJ|J|IU|ppm|ppb|rpm|GB|MB|KB|TB|min|h|s|yr|%")
_NUM_UNIT = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)*)\s?(" + _UNITS + r")(?![A-Za-z])")
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:-[0-9A-Z]+)?\b")                       # HIV, DNA, COVID-19, CO2, H2O, PM2
_FORMULA = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")                                    # NaCl, CaCO3, H2SO4, KOH


def chrf_recall_precision(hyp: str, ref: str) -> Tuple[float, float]:
    """(recall, precision) of the chrF++ statistics (char 1..6 and word 1..2 n-grams, sacrebleu conventions, average over the orders that exist on both sides), each 0..1."""
    h, r = cons.Profile(hyp), cons.Profile(ref)
    m = cons.matches(h, r)
    ap = ar = 0.0
    eff = 0
    for nh, nr, nm in zip(h.totals, r.totals, m):
        nh = nh if nr > 0 else 0
        if nh > 0 and nr > 0:
            ap += nm / nh
            ar += nm / nr
            eff += 1
    return (ar / eff, ap / eff) if eff else (0.0, 0.0)


def length_log_ratio(hyp: str, ref: str) -> float:
    a, b = len("".join(hyp.split())), len("".join(ref.split()))
    return math.log((a + 1.0) / (b + 1.0))


def source_items(src: str) -> Dict[str, List[str]]:
    """Items of the source that a translation should carry over: numbers (digit strings, separators removed), number+unit pairs (normalised without spaces), acronyms / chemical formulas
    (case kept), named entities (folded, not sentence-initial; rl/anchors.extract_entities)."""
    nums, names = extract_entities(src)
    units = [numkey(m.group(1)) + m.group(2) for m in _NUM_UNIT.finditer(src)]
    acr = sorted({m.group() for m in _ACRONYM.finditer(src)} | {m.group() for m in _FORMULA.finditer(src) if any(c.isdigit() for c in m.group()) or sum(c.isupper() for c in m.group()) >= 2})
    return {"numbers": nums, "units": units, "acronyms": acr, "names": names}


def _present(item: str, kind: str, hyp: str, hyp_nums: set, hyp_fold: str, hyp_words: List[str]) -> bool:
    if kind == "numbers":
        return item in hyp_nums
    if kind == "units":
        return item in "".join(hyp.split()).replace(",", "").replace(".", "") or item in hyp_fold.replace(" ", "")
    if kind == "acronyms":
        return item in hyp
    return item in hyp_fold or any(_edit_le1(item, w) for w in hyp_words if abs(len(w) - len(item)) <= 1)


def preservation(src: str, hyp: str, min_items: int = 2) -> Tuple[Optional[float], int, Dict[str, int]]:
    """(share of the source's items missing from the translation, number of items, missing per kind); share is None when the source has fewer than `min_items` items."""
    items = source_items(src)
    n = sum(len(v) for v in items.values())
    if n < min_items:
        return None, n, {}
    from gemma_mt.rl.anchors import fold

    hyp_nums = {numkey(m) for m in re.findall(r"\d+(?:[.,]\d+)*", hyp)}
    hf = fold(hyp)
    words = re.findall(r"\w+", hf)
    miss = {k: sum(1 for it in v if not _present(it, k, hyp, hyp_nums, hf, words)) for k, v in items.items()}
    return sum(miss.values()) / n, n, miss


@dataclass
class ScienceTermsConfig:
    ssa_weight: float = 0.5            # w_ssa (x slope = reward units per SSA-COMET score unit; the QE shaping slope of the stages is about 4)
    ssa_slope: float = 4.0
    ssa_center: float = 0.5            # offset (cancels inside a GRPO group; only matters for the clip at the failure penalty)
    omit_weight: float = 0.3
    recall_floor: float = 0.5
    add_weight: float = 0.3
    precision_floor: float = 0.5
    length_weight: float = 0.3
    length_tol: float = math.log(1.35)
    preserve_weight: float = 0.5
    preserve_min_items: int = 2


def science_adjustment(cfg: ScienceTermsConfig, src: str, hyp: str, ref: str, ssa: Optional[float]) -> Tuple[float, Dict[str, float]]:
    """(reward adjustment, details) for one row that passed every hard check. `ssa` = the reference-based SSA-COMET-MTL score (None = not available: that term is skipped)."""
    rec, prec = chrf_recall_precision(hyp, ref)
    lr = length_log_ratio(hyp, ref)
    d: Dict[str, float] = {"recall": rec, "precision": prec, "log_len_ratio": lr}
    adj = 0.0
    if ssa is not None:
        t = cfg.ssa_weight * cfg.ssa_slope * (ssa - cfg.ssa_center)
        d["ssa"], d["ssa_term"] = ssa, t
        adj += t
    om = max(0.0, cfg.recall_floor - rec) / cfg.recall_floor
    ad = max(0.0, cfg.precision_floor - prec) / cfg.precision_floor
    ln = min(1.0, max(0.0, abs(lr) - cfg.length_tol) / cfg.length_tol)
    d.update(omit=om, add=ad, length=ln)
    adj -= cfg.omit_weight * om + cfg.add_weight * ad + cfg.length_weight * ln
    miss, n_items, _ = preservation(src, hyp, cfg.preserve_min_items)
    d["items"] = float(n_items)
    if miss is not None:
        d["preserve_missing"] = miss
        adj -= cfg.preserve_weight * miss
    d["adjustment"] = adj
    return adj, d


def batch_stats(details: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Means and rates of one batch for the step record (gaming watch)."""
    n = len(details)
    out: Dict[str, float] = {"sci/n": float(n)}
    if not n:
        return out
    mean = lambda k: sum(d[k] for d in details if k in d) / max(1, sum(1 for d in details if k in d))
    for k in ("recall", "precision", "log_len_ratio", "ssa", "adjustment"):
        if any(k in d for d in details):
            out[f"sci/{k}_mean"] = mean(k)
    out["sci/omit_rate"] = sum(1 for d in details if d["omit"] > 0) / n
    out["sci/add_rate"] = sum(1 for d in details if d["add"] > 0) / n
    out["sci/length_rate"] = sum(1 for d in details if d["length"] > 0) / n
    j = [d for d in details if "preserve_missing" in d]
    out["sci/preserve_judged"] = len(j) / n
    out["sci/preserve_missing_mean"] = sum(d["preserve_missing"] for d in j) / len(j) if j else 0.0
    return out
