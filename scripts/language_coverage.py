#!/usr/bin/env python
"""D-59: how many prompts / sampled completions does every language get over a run? Simulates the real PromptSchedule of a yaml (CPU, ~10 s).

  python scripts/language_coverage.py configs/rl_science_final.yaml [--steps N] [--json out.json] [--no-floor]

Reports per language: prompts, completions (x group_size), share, the minimum over aligned blocks of `language_floor_steps` steps, and the number of steps (not blocks) in which the language was absent.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-floor", action="store_true")
    a = ap.parse_args()
    from gemma_mt.languages import SUPPORTED_RL
    from gemma_mt.rl.data import PromptSchedule, load_rl_prompts
    from gemma_mt.rl.train_rl import RLConfig, apply_overrides
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    cfg = dataclass_from_dict(RLConfig, apply_overrides(load_yaml(a.config), {}))
    steps = a.steps or cfg.steps
    full = load_rl_prompts(cfg.prompts_path, max_src_tokens=cfg.max_src_tokens)
    fs, fp = (0, 1) if a.no_floor else (cfg.language_floor_steps, cfg.language_floor_prompts)
    sched = PromptSchedule(full.iso, cfg.language_alpha, cfg.seed, length_bin=full.length_bin, curriculum=cfg.curriculum_steps or None, language_weights=cfg.language_weights or None,
                           domain=full.domain, domain_weights=cfg.domain_weights or None, domain_bin_weights=cfg.domain_bin_weights or None, floor_steps=fs, floor_prompts=fp)
    nps = cfg.prompts_per_step_by_bin.get("mixed", cfg.prompts_per_step)
    tot = collections.Counter()
    sci = collections.Counter()
    absent = collections.Counter()
    blockmin = {l: 10 ** 9 for l in SUPPORTED_RL}
    P = cfg.language_floor_steps or 20
    blk = collections.Counter()
    for st in range(steps):
        picks = [full.iso[i] for i in sched.indices(st, nps)]
        c = collections.Counter(picks)
        tot.update(c); blk.update(c)
        sci.update(full.iso[i] for i in sched.indices(st, nps) if full.domain[i] == "science")
        for l in SUPPORTED_RL:
            if c[l] == 0:
                absent[l] += 1
        if (st + 1) % P == 0:
            for l in SUPPORTED_RL:
                blockmin[l] = min(blockmin[l], blk[l])
            blk = collections.Counter()
    g = cfg.group_size
    rows = [{"iso": l, "prompts": tot[l], "completions": tot[l] * g, "share": tot[l] / max(sum(tot.values()), 1), "science_prompts": sci[l], f"min_per_{P}_steps": blockmin[l] if steps >= P else None,
             "steps_absent": absent[l]} for l in SUPPORTED_RL]
    rows.sort(key=lambda r: r["prompts"])
    vals = [r["prompts"] for r in rows]
    print(f"{steps} steps x {nps} prompts x group {g}: {sum(vals)} prompts, {sum(vals) * g} sampled completions; floor {'off' if not fs else f'{fp} per {fs} steps'}")
    print(f"per language prompts: min {vals[0]} ({rows[0]['iso']}), median {vals[len(vals) // 2]}, max {vals[-1]} ({rows[-1]['iso']}); completions min {vals[0] * g}; min per {P}-step block over all languages: {min(r[f'min_per_{P}_steps'] for r in rows)}")
    print("lowest 8:", ", ".join(f"{r['iso']} {r['prompts']}" for r in rows[:8]))
    print("highest 8:", ", ".join(f"{r['iso']} {r['prompts']}" for r in rows[-8:]))
    if a.json:
        json.dump({"steps": steps, "prompts_per_step": nps, "group_size": g, "floor": [fs, fp], "languages": rows}, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
