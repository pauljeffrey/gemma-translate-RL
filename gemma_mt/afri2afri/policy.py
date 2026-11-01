# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sampling policy over the language-pair directions (D-46): UniMax allocation + per-pair composition quotas.

`unimax` (Chung et al. 2023): spread a budget UNIFORMLY over the items, but never give an item more than `max_epochs` passes over its own data;
what a small item cannot absorb is redistributed over the larger ones. Applied here to the UNORDERED language pairs {X, Y}: both directions of a pair
use the same pairs (like the en->X / X->en samples of one sentence pair), so a pair's allocation is the number of sentence pairs drawn, and the
dataset contains twice as many directional samples.

`compose` fills one pair's allocation from provenance classes in a priority order with caps, so that machine-generated (`synthetic_mt`) and
religious / Bible text cannot take over a direction that also has other text, but a direction that has ONLY such text still gets filled (the caps are
relaxed in a second pass, and the shortfall is reported, never hidden).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

CLASSES = ("human_curated", "mined_nllb", "mixed", "religious_jw", "synthetic_mt")        # fill priority (highest first)


def unimax(sizes: Sequence[float], budget: float, max_epochs: float = 3.0) -> np.ndarray:
    """Allocation per item (float, same unit as `sizes`). Items are visited from smallest to largest; each gets min(equal share of what is left, max_epochs x size).
    The sum is min(budget, max_epochs x sum(sizes))."""
    sizes = np.asarray(sizes, np.float64)
    n = len(sizes)
    alloc = np.zeros(n)
    left = float(budget)
    for rank, i in enumerate(np.argsort(sizes, kind="stable")):
        share = left / (n - rank)
        a = min(share, max_epochs * sizes[i])
        alloc[i] = a
        left -= a
    return alloc


def compose(available: Mapping[str, int], alloc: int, caps: Optional[Mapping[str, float]] = None) -> Dict[str, int]:
    """How many of the `alloc` draws of one pair come from each class. `available[c]` = distinct pairs of class c.
    Pass 1 takes classes in CLASSES order up to min(available, cap x alloc); pass 2 (relaxation) fills any shortfall from what is still unused, same order.
    Draws never exceed `available` (repetition for small pairs is applied by the caller on top, as epochs)."""
    caps = {"human_curated": 1.0, "mined_nllb": 0.4, "mixed": 1.0, "religious_jw": 0.6, "synthetic_mt": 0.15, **(dict(caps or {}))}
    take = {c: 0 for c in CLASSES}
    need = int(alloc)
    for c in CLASSES:
        k = min(int(available.get(c, 0)), int(caps.get(c, 1.0) * alloc), need)
        take[c] = k
        need -= k
    for c in CLASSES:
        if need <= 0:
            break
        extra = min(int(available.get(c, 0)) - take[c], need)
        if extra > 0:
            take[c] += extra
            need -= extra
    return take


def epochs_for(total_available: int, alloc: float) -> float:
    """Passes over the pair's own data implied by an allocation (>1 means the thin pair is repeated)."""
    return alloc / total_available if total_available else 0.0
