# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dynamic sampling for GRPO (DAPO-style; owner 2026-11-03, default OFF in `RLConfig.dynamic_sampling`).

A group (the G completions of one prompt) whose rewards have (almost) no spread has advantage 0 in every member: it costs generation, scoring and backward passes and teaches nothing
(and a group with a tiny spread is worse: `A = (r - mean) / (std + eps)` turns jitter in the 3rd decimal of a QE score into advantages of order 1). With dynamic sampling a step
    1. draws `ceil(n_target * oversample)` prompts and generates + scores them as usual,
    2. marks the groups whose reward spread is below `min_std` as UNINFORMATIVE (flat),
    3. while fewer than `n_target` informative groups exist and fewer than `max_refills` refill rounds were spent: draws one REPLACEMENT prompt for each dropped group (same language, same
       domain and length bin as the dropped prompt where the pool has one, so the language / domain / bin mixture of the schedule is kept in expectation; `PromptSchedule.replacements`),
       generates and scores only those,
    4. keeps the first `n_target` informative groups (in generation order) for the policy update; all other groups only feed the monitoring (reward, finished / loop rates, per-language
       statistics, the translation archive), which must not be biased by the filter (a flat all-failed group is exactly the group the filter removes).
If after the refills fewer than `n_target` informative groups exist the step trains on those (a smaller batch; counted); with none at all the first `n_target` groups of round 0 are kept
(zero advantages, the step is wasted but never empty). Everything here is pure python / numpy: the tensors stay in train_rl.

Informative = std of the group's rewards >= min_std, where the group is the SAMPLED members only when `greedy_stats == "sampled"` (the greedy member is a baseline-relative extra there, exactly as in
`grpo.group_advantages_sampled_baseline`), and, with `hard_fail_mask == "exclude"`, the non-failed members only (fewer than 2 of them: uninformative).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

FLAT_THRESHOLD = 1e-6                           # = grpo.group_advantages' flat_threshold: a group with std below this has advantage 0 whatever min_std says


def informative_groups(rewards: Sequence[float], group_size: int, min_std: float, sampled_only: bool = False,
                       failed: Optional[Sequence[bool]] = None, mask_mode: str = "off", drop: Optional[Sequence[bool]] = None) -> List[bool]:
    """One flag per group (rows [kG, (k+1)G) of `rewards`): True = the group's advantages are not all (nearly) zero. `drop`: members that are not part of the group (a hard-failed beam-guide member);
    the spread is measured without them, exactly as the advantage does. A guide that is NOT dropped counts, so a group that is informative only because of the guide is kept."""
    r = np.asarray(list(rewards), dtype=np.float64)
    if r.size % group_size:
        raise ValueError(f"{r.size} rewards are not divisible into groups of {group_size}")
    r = r.reshape(-1, group_size)
    f = np.zeros_like(r, dtype=bool) if failed is None else np.asarray(list(failed), dtype=bool).reshape(-1, group_size)
    dr = np.zeros_like(r, dtype=bool) if drop is None else np.asarray(list(drop), dtype=bool).reshape(-1, group_size)
    out: List[bool] = []
    for row, fr, dd in zip(r, f, dr):
        if dd.any():
            if sampled_only:
                row, dd = row[:-1], dd[:-1]
            row = row[~dd]
            if row.size < 2:
                out.append(False)
                continue
            out.append(bool(row.std() >= max(min_std, FLAT_THRESHOLD)))
            continue
        if mask_mode == "exclude":
            row = row[~fr]
            if row.size < 2:
                out.append(False)
                continue
        elif sampled_only:
            row = row[:-1]
        out.append(bool(row.std() >= max(min_std, FLAT_THRESHOLD)))
    return out


# ------------------------------------------------------------------------------------------------ row selection / concatenation
def _pick(x: Any, rows: List[int], n: int) -> Any:
    """x[rows] for a per-row list of length n; everything else (empty lists = feature off, dicts, scalars) unchanged."""
    if isinstance(x, list) and len(x) == n and n > 0:
        return [x[r] for r in rows]
    return x


def _rows_of(groups: Sequence[int], group_size: int) -> List[int]:
    return [g * group_size + j for g in groups for j in range(group_size)]


def subset_dataclass(obj: Any, groups: Sequence[int], group_size: int, n: int) -> Any:
    """Copy of a Rollout / RewardOutput / list-of-Sample holder with only the rows of `groups` (per-row list fields; the dict fields are copied as they are)."""
    rows = _rows_of(groups, group_size)
    return dataclasses.replace(obj, **{f.name: _pick(getattr(obj, f.name), rows, n) for f in dataclasses.fields(obj)})


def subset_rollout_output(roll: Any, samples: List[Any], out: Any, groups: Sequence[int], group_size: int) -> Tuple[Any, List[Any], Any]:
    n = len(roll.completions)
    rows = _rows_of(groups, group_size)
    return subset_dataclass(roll, groups, group_size, n), [samples[r] for r in rows], subset_dataclass(out, groups, group_size, n)


def _cat(a: Any, b: Any, na: int, nb: int) -> Any:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) == na and len(b) == nb:
            return a + b
        if len(a) == na and not b:                       # the second batch has the optional column empty: pad (None / "" are what "not judged" is everywhere)
            return a + [None] * nb
        if not a and len(b) == nb:
            return [None] * na + b
    return a


def _merge_stats(a: Dict[str, float], b: Dict[str, float], na: int, nb: int) -> Dict[str, float]:
    """Sample-weighted mean of two stats dicts (the stats of a reward call are rates / means over its samples); keys of one side only are kept."""
    out = dict(a)
    for k, v in b.items():
        out[k] = (a[k] * na + v * nb) / max(na + nb, 1) if k in a and isinstance(a[k], (int, float)) and isinstance(v, (int, float)) else v
    return out


def concat_rollout_output(first: Tuple[Any, List[Any], Any], second: Tuple[Any, List[Any], Any]) -> Tuple[Any, List[Any], Any]:
    (r1, s1, o1), (r2, s2, o2) = first, second
    n1, n2 = len(r1.completions), len(r2.completions)
    roll = dataclasses.replace(r1, **{f.name: _cat(getattr(r1, f.name), getattr(r2, f.name), n1, n2) for f in dataclasses.fields(r1)})
    skip = {"stats", "confidence", "fluency"}
    out = dataclasses.replace(o1, **{f.name: _cat(getattr(o1, f.name), getattr(o2, f.name), n1, n2) for f in dataclasses.fields(o1) if f.name not in skip},
                              stats=_merge_stats(o1.stats, o2.stats, n1, n2), confidence=dict(o1.confidence, **o2.confidence), fluency=dict(o1.fluency, **o2.fluency))
    return roll, list(s1) + list(s2), out


# ------------------------------------------------------------------------------------------------ the step
@dataclass
class DynamicResult:
    roll: Any                                   # ALL generated rows (every round), in generation order
    samples: List[Any]
    out: Any
    prompt_idx: List[int]                       # prompt index per group of the all-rows lists
    informative: List[bool]                     # per group of the all-rows lists
    keep: List[int]                             # the groups (indices into the all-rows lists) used for the update, in order
    stats: Dict[str, float] = field(default_factory=dict)
    by_lang: Dict[str, Dict[str, int]] = field(default_factory=dict)       # iso -> {generated, dropped_flat, surplus, kept}


def dynamic_rollout(first_idx: Sequence[int], n_target: int, group_size: int, max_refills: int, min_std: float,
                    generate: Callable[[List[int], int], Tuple[Any, List[Any], Any]], redraw: Callable[[List[int], int], List[int]],
                    iso_of: Callable[[int], str], sampled_only: bool = False, mask_mode: str = "off") -> DynamicResult:
    """`generate(indices, round)` -> (roll, samples, out) for those prompts (round 0 = the first draw); `redraw(dropped_prompt_indices, round)` -> one replacement prompt index per dropped prompt."""
    G = group_size
    roll, samples, out = generate(list(first_idx), 0)
    prompt_idx = list(first_idx)
    rounds = 0

    def flags():
        failed = [st != "ok" for st in out.status] if mask_mode == "exclude" else None
        drop = [bool(getattr(smp, "guide", False)) and st != "ok" for smp, st in zip(samples, out.status)]            # a hard-failed beam-guide member leaves its group
        return informative_groups(out.rewards, G, min_std, sampled_only, failed, mask_mode, drop if any(drop) else None)

    fl = flags()
    replaced = set()
    while sum(fl) < n_target and rounds < max_refills:
        deficit = n_target - sum(fl)
        todo = [g for g, ok in enumerate(fl) if not ok and g not in replaced][:deficit]
        if not todo:
            break
        replaced.update(todo)
        rounds += 1
        new_idx = list(redraw([prompt_idx[g] for g in todo], rounds))
        if len(new_idx) != len(todo):
            raise ValueError(f"redraw returned {len(new_idx)} prompts for {len(todo)} dropped groups")
        roll, samples, out = concat_rollout_output((roll, samples, out), generate(new_idx, rounds))
        prompt_idx += new_idx
        fl = flags()
    inf = [g for g, ok in enumerate(fl) if ok]
    fallback = not inf
    keep = (list(range(min(n_target, len(fl)))) if fallback else inf[:n_target])
    keep_set = set(keep)
    lens = [len(c) for c in roll.completions]
    by_lang: Dict[str, Dict[str, int]] = {}
    wasted = 0
    for g, p in enumerate(prompt_idx):
        d = by_lang.setdefault(iso_of(p), {"generated": 0, "dropped_flat": 0, "surplus": 0, "kept": 0})
        d["generated"] += 1
        if g in keep_set:
            d["kept"] += 1
        elif not fl[g]:
            d["dropped_flat"] += 1
            wasted += sum(lens[g * G:(g + 1) * G])
        else:
            d["surplus"] += 1
            wasted += sum(lens[g * G:(g + 1) * G])
    n_groups = len(prompt_idx)
    n_flat = sum(1 for ok in fl if not ok)
    stats = {"ds/requested": float(n_target), "ds/generated_groups": float(n_groups), "ds/kept_groups": float(len(keep)),
             "ds/dropped_flat": float(sum(1 for g, ok in enumerate(fl) if not ok and g not in keep_set)),           # flat and not used (== flat unless the all-flat fallback kept some)
             "ds/surplus": float(sum(1 for g, ok in enumerate(fl) if ok and g not in keep_set)), "ds/refill_rounds": float(rounds), "ds/short_batch": float(len(keep) < n_target),
             "ds/fallback": float(fallback), "ds/flat_frac": n_flat / max(n_groups, 1),
             "ds/wasted_completion_tokens": float(wasted), "ds/wasted_token_frac": wasted / max(sum(lens), 1)}
    return DynamicResult(roll, samples, out, prompt_idx, fl, keep, stats, by_lang)


class LanguageCounters:
    """Running per-language counters of the filter (written to <out_dir>/dynamic_sampling_by_language.json): the kept rate of a language must not collapse (a language whose groups are all flat would
    otherwise lose its share of the update silently)."""

    def __init__(self):
        self.c: Dict[str, Dict[str, int]] = {}

    def add(self, by_lang: Dict[str, Dict[str, int]]) -> None:
        for iso, d in by_lang.items():
            t = self.c.setdefault(iso, {"generated": 0, "dropped_flat": 0, "surplus": 0, "kept": 0})
            for k, v in d.items():
                t[k] += v

    def table(self) -> Dict[str, Dict[str, float]]:
        return {iso: dict(d, flat_rate=d["dropped_flat"] / max(d["generated"], 1), kept_rate=d["kept"] / max(d["generated"], 1)) for iso, d in sorted(self.c.items())}

    def step_metrics(self, by_lang: Dict[str, Dict[str, int]], top: int = 5) -> Dict[str, float]:
        """Per-language flat share of THIS step, only for the languages that lost groups (at most `top` of them, the worst first)."""
        worst = sorted(((d["dropped_flat"] / max(d["generated"], 1), iso) for iso, d in by_lang.items() if d["dropped_flat"]), reverse=True)[:top]
        return {f"ds_flat/{iso}": float(v) for v, iso in worst}
