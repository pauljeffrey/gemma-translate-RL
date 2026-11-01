#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AB1: make a gemma_mt tree fit for the RL-vs-SFT-data ablation. Idempotent text patches of `gemma_mt/rl/train_rl.py`; nothing else changes.

    python scripts/ab1_patch_tree.py TREE_ROOT [--run-controls-from OTHER_TREE]

1. RUN CONTROLS (only for the phase-2c tree, which predates D-56): `rl/run_controls.py` (copied from --run-controls-from) + the 12 lines that read `<out_dir>/WINDDOWN` and
   `<out_dir>/LR_SCALE` once per step (identical to the D-56 hunks of the final-run tree). Skipped when train_rl.py already imports RunControls.
2. SNAPSHOT EXPORT: config keys `export_steps: [250]` and `export_dtype: bfloat16`. At those steps, and at the last step of the run (also a wound-down one), the policy is written as a
   Hugging Face directory `<out_dir>/snap_step<N>` (weights cast to bf16 = what every evaluation loads anyway, 0.54 GB; greedy generation_config when the tree has D-67). Needed because
   `keep_single_checkpoint: true` writes only the resumable `last/` (3.2 GB with the optimizer) and `hub_repo_id: null` writes nothing else.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


def sub(text: str, old: str, new: str, count: int = 1, what: str = "") -> str:
    if text.count(old) < 1 or (count and text.count(old) != count):
        raise SystemExit(f"patch anchor not found {count}x ({what or old[:60]!r}): found {text.count(old)}")
    return text.replace(old, new)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree")
    ap.add_argument("--run-controls-from", default=None)
    a = ap.parse_args(argv)
    p = os.path.join(a.tree, "gemma_mt", "rl", "train_rl.py")
    s = open(p, encoding="utf-8").read()
    notes = []

    if "RunControls" not in s:
        if not a.run_controls_from:
            raise SystemExit("this tree has no run controls: pass --run-controls-from <tree that has gemma_mt/rl/run_controls.py>")
        shutil.copy(os.path.join(a.run_controls_from, "gemma_mt", "rl", "run_controls.py"), os.path.join(a.tree, "gemma_mt", "rl", "run_controls.py"))
        s = sub(s, "from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward\n",
                "from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward\nfrom gemma_mt.rl.run_controls import RunControls\n", what="import")
        s = sub(s, "    keep_single_checkpoint: bool = False            # local disk: only last/ (resumable); no best/ or final/ copies\n",
                "    keep_single_checkpoint: bool = False            # local disk: only last/ (resumable); no best/ or final/ copies\n"
                "    run_controls: bool = True                       # D-56: <out_dir>/WINDDOWN and <out_dir>/LR_SCALE are read at every step (rl/run_controls.py); nothing happens without those files\n",
                what="field run_controls")
        s = sub(s, "    while step < cfg.steps:\n        t0 = time.time()\n        lr = cfg.learning_rate * lr_scale * min(1.0, (step + 1) / max(cfg.warmup_steps, 1))\n",
                "    controls = RunControls(cfg.out_dir, cfg.steps) if cfg.run_controls else None      # D-56: WINDDOWN / LR_SCALE files in out_dir\n"
                "    end_step = cfg.steps\n    ctl = None\n"
                "    while step < end_step:\n        t0 = time.time()\n"
                "        if controls is not None:\n            ctl = controls.update(step)\n            end_step = ctl.end_step\n"
                "            if step >= end_step:                       # a wind-down that was already complete when this (restarted) run began\n                break\n"
                "        lr = cfg.learning_rate * lr_scale * min(1.0, (step + 1) / max(cfg.warmup_steps, 1)) * (ctl.lr_mult if ctl is not None else 1.0)\n", what="loop head")
        s = sub(s, "print(f\"[gemma_mt] rl step {step}/{cfg.steps} reward", "print(f\"[gemma_mt] rl step {step}/{end_step} reward", what="progress line")
        s = sub(s, "(step % cfg.eval_every == 0 or step == cfg.steps) and len(eval_prompts)", "(step % cfg.eval_every == 0 or step == end_step) and len(eval_prompts)", what="eval cadence")
        s = sub(s, "(step % b_every == 0 or step == cfg.steps)", "(step % b_every == 0 or step == end_step)", what="bench cadence")
        s = sub(s, "(step % cfg.hub_latest_every == 0 or step == cfg.steps)", "(step % cfg.hub_latest_every == 0 or step == end_step)", what="latest cadence")
        s = sub(s, "block=(step == cfg.steps))", "block=(step == end_step))", what="latest block")
        s = sub(s, "        if (cfg.save_every and step % cfg.save_every == 0) or step == cfg.steps:\n            save_last()\n",
                "        if (cfg.save_every and step % cfg.save_every == 0) or step == end_step:\n            save_last()\n", what="save_last")
        s = sub(s, "    if archive is not None:\n        summary[\"archive\"] = archive.close()\n    with open(os.path.join(cfg.out_dir, \"summary.json\"), \"w\") as f:\n",
                "    if controls is not None:\n        wd = controls.finish(step)\n        if wd is not None:\n            summary[\"winddown\"] = wd\n"
                "    if archive is not None:\n        summary[\"archive\"] = archive.close()\n    with open(os.path.join(cfg.out_dir, \"summary.json\"), \"w\") as f:\n", what="summary")
        notes.append("run controls grafted")

    if "export_steps" not in s:
        s = sub(s, "    run_controls: bool = True",
                "    export_steps: List[int] = field(default_factory=list)   # AB1: also write <out_dir>/snap_step<N> (HF directory, export_dtype) at these steps and at the last step\n"
                "    export_dtype: str = \"bfloat16\"                  # AB1: bfloat16 (0.54 GB, what the evaluations load) | float32\n"
                "    run_controls: bool = True", count=1, what="export fields")
        s = sub(s, "        if (cfg.save_every and step % cfg.save_every == 0) or step == end_step:\n            save_last()\n",
                "        if step in cfg.export_steps or (cfg.export_steps and step == end_step):\n"
                "            save_model(policy, tokenizer, os.path.join(cfg.out_dir, f\"snap_step{step}\"), stop_ids=stop.ids,\n"
                "                       dtype={\"bfloat16\": torch.bfloat16, \"float32\": None}[cfg.export_dtype])\n"
                "            print(f\"[gemma_mt] AB1 snapshot written: {cfg.out_dir}/snap_step{step} ({cfg.export_dtype})\", flush=True)\n"
                "        if (cfg.save_every and step % cfg.save_every == 0) or step == end_step:\n            save_last()\n", what="export hook")
        notes.append("export_steps added")
    open(p, "w", encoding="utf-8").write(s)
    print("ab1_patch_tree:", "; ".join(notes) if notes else "nothing to do (already patched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
