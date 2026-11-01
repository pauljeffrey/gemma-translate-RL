#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AB2: idempotent patch of gemma_mt/train_sft.py in the AB2 tree (never the shared tree) for the CONTINUED-SFT controls.

    python scripts/ab2_patch_sft.py [--tree /workspace/jobs/ab2/gemma_mt]

Adds two config fields and nothing else (a config without them behaves exactly as before, byte for byte):
  schedule_start_step: int = 0   > 0: a NEW phase that starts from the weights in `model_name` (an HF directory, e.g. the weak SFT) and CONTINUES the step counter and the
                                 language-stratified data schedule of the run that produced those weights: the phase trains on rows [start, start + max_steps) of the SAME
                                 `balanced_schedule` (a pure function of seed and step: row t of an N-step schedule is row t of every longer one). Fresh optimizer; the LR
                                 warm-up of the phase is `warmup_steps`. Resumable (the checkpoint stores run_start).
  lr_hold_fraction: float = 0.0  > 0: constant-then-anneal LR: warm-up, then `learning_rate` for this share of the remaining steps, then a LINEAR decay to
                                 learning_rate * min_lr_ratio over the rest (0 = the usual cosine).
"""
import argparse
import sys

FIELDS_ANCHOR = "    prior_total_steps: int = 0                      # total_steps of the previous phase (its schedule length)\n"
FIELDS_NEW = FIELDS_ANCHOR + (
    "    # AB2 (continued-SFT controls): weights from `model_name` (an HF directory), step counter + balanced_schedule continue at this step (fresh optimizer)\n"
    "    schedule_start_step: int = 0\n"
    "    lr_hold_fraction: float = 0.0                   # > 0: constant LR for this share of the post-warm-up steps, then linear decay to learning_rate * min_lr_ratio\n")

HELPER_ANCHOR = "class EarlyStopper:\n"
HELPER_NEW = '''def lr_const_anneal(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr_ratio: float, hold_fraction: float) -> float:
    """AB2: linear warm-up to `peak_lr`, `peak_lr` for `hold_fraction` of the remaining steps, then a linear decay to peak_lr * min_lr_ratio at the last step."""
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    start = warmup_steps + hold_fraction * max(0, total_steps - warmup_steps)
    if step < start or total_steps <= start:
        return peak_lr
    t = min(1.0, max(0.0, (step - start) / max(1.0, (total_steps - 1) - start)))
    return peak_lr * (1.0 - (1.0 - min_lr_ratio) * t)


''' + HELPER_ANCHOR

RUN_START_OLD = """        else:
            run_start = int(torch.load(cfg.init_from, map_location="cpu", weights_only=False)["state"]["step"])
"""
RUN_START_NEW = RUN_START_OLD + """    elif cfg.schedule_start_step > 0:                     # AB2: weights from model_name, schedule continues at this step
        resume_ck = os.path.join(cfg.out_dir, "last", "trainer_state.pt")
        if cfg.resume and os.path.exists(resume_ck):
            run_start = int(torch.load(resume_ck, map_location="cpu", weights_only=False)["state"].get("run_start", cfg.schedule_start_step))
        else:
            run_start = int(cfg.schedule_start_step)
"""

SCHED_OLD = """        schedule, lang_stats = balanced_schedule(train_ds.pack_languages(), total_steps, per_step, alpha=cfg.language_alpha,
                                                 min_share=cfg.min_language_share, max_epochs=cfg.max_language_epochs, seed=cfg.seed)
"""
SCHED_NEW = SCHED_OLD + """        if cfg.schedule_start_step > 0 and not cfg.init_from:      # AB2: this phase uses rows [run_start, total_steps) of the schedule (indexed below as step - run_start)
            schedule = schedule[run_start:]
            assert len(schedule) == n_run, (len(schedule), n_run)
"""

STEP0_OLD = """    elif cfg.init_from:
        ck = torch.load(cfg.init_from, map_location=dist.device, weights_only=False)
"""
STEP0_NEW = """    elif cfg.schedule_start_step > 0 and not cfg.init_from:
        step = run_start                                              # AB2: weights = model_name, the counter continues at the step of the run that made them
        if dist.is_main:
            print(f"[gemma_mt] AB2 continuation: weights from {cfg.model_name}, step counter and data schedule continue at step {step} "
                  f"({n_run} steps, fresh optimizer)", flush=True)
""" + STEP0_OLD

LR_OLD = "        lr = lr_at(step - run_start, n_run, warmup, cfg.learning_rate, cfg.min_lr_ratio)\n"
LR_NEW = """        if cfg.lr_hold_fraction > 0:
            lr = lr_const_anneal(step - run_start, n_run, warmup, cfg.learning_rate, cfg.min_lr_ratio, cfg.lr_hold_fraction)
        else:
            lr = lr_at(step - run_start, n_run, warmup, cfg.learning_rate, cfg.min_lr_ratio)
"""


def sub(s, old, new, what):
    if new in s:
        return s, False
    if s.count(old) != 1:
        raise SystemExit(f"patch anchor for {what!r} found {s.count(old)} times (expected 1)")
    return s.replace(old, new), True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", default="/workspace/jobs/ab2/gemma_mt")
    a = ap.parse_args(argv)
    p = f"{a.tree}/gemma_mt/train_sft.py"
    s = open(p, encoding="utf-8").read()
    done = []
    for old, new, what in ((FIELDS_ANCHOR, FIELDS_NEW, "fields"), (HELPER_ANCHOR, HELPER_NEW, "helper"), (RUN_START_OLD, RUN_START_NEW, "run_start"),
                           (SCHED_OLD, SCHED_NEW, "schedule slice"), (STEP0_OLD, STEP0_NEW, "step0"), (LR_OLD, LR_NEW, "lr")):
        if what == "helper" and "def lr_const_anneal" in s:
            continue
        s, ch = sub(s, old, new, what)
        if ch:
            done.append(what)
    open(p, "w", encoding="utf-8").write(s)
    print("ab2_patch_sft:", ", ".join(done) if done else "nothing to do (already patched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
