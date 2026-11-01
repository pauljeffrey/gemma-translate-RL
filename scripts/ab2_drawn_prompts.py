#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AB2: which prompts does an RL arm actually draw?  Replays the trainer's own prompt loading, hold-out split and PromptSchedule (a pure function of seed and step: the same for
arm F and arm W and independent of the model) for steps 0..steps-1 of a config, CPU only.

    python scripts/ab2_drawn_prompts.py --config configs/rl_ab2_full.yaml --out data/rl_drawn_prompts.jsonl
Writes one JSON line per DRAW (step, iso, english) and prints draws / distinct sources per language and the overlap with the held-out reward-eval prompts.
"""
import argparse
import collections
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from gemma_mt.rl.data import PromptSchedule, load_rl_prompts
    from gemma_mt.rl.train_rl import RLConfig, _subset, split_eval_prompts, validate_config
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    cfg = dataclass_from_dict(RLConfig, load_yaml(a.config))
    validate_config(cfg)
    prompts = load_rl_prompts(cfg.prompts_path, max_src_tokens=cfg.max_src_tokens)
    tr_idx, ev_idx = split_eval_prompts(prompts, min(cfg.eval_prompts, len(prompts) // 10), cfg.seed, cfg.eval_min_per_bin)
    ev_text = {prompts.english[i] for i in ev_idx}
    prompts = _subset(prompts, tr_idx)
    sch = PromptSchedule(prompts.iso, cfg.language_alpha, cfg.seed, length_bin=prompts.length_bin if cfg.length_bins else None,
                         curriculum=cfg.curriculum_steps or None, language_weights=cfg.language_weights or None)
    draws = collections.defaultdict(list)
    rows = []
    for step in range(cfg.steps):
        n = cfg.prompts_per_step_by_bin.get(sch.bin_of_step(step), cfg.prompts_per_step) if cfg.length_bins else cfg.prompts_per_step
        for i in sch.indices(step, n):
            draws[prompts.iso[i]].append(prompts.english[i])
            rows.append({"step": step, "iso": prompts.iso[i], "english": prompts.english[i]})
    tot = sum(len(v) for v in draws.values())
    dist = len({r["english"] for r in rows})
    print(f"pool {len(prompts)} prompts ({len(ev_idx)} held out for the reward eval), {cfg.steps} steps -> {tot} draws, {dist} distinct sources, overlap with the held-out eval prompts: "
          f"{len({r['english'] for r in rows} & ev_text)}")
    for iso, v in sorted(draws.items()):
        print(f"  {iso}: {len(v)} draws, {len(set(v))} distinct")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
