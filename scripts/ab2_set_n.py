#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AB2: choose N, the number of SFT steps of the GPU-time-matched control S1, from the MEASURED times (rule of DECISION_RULE.txt section 2: the S1 job's wall-clock equals one RL arm's).

    python scripts/ab2_set_n.py [--ab /workspace/jobs/ab2] [--write] [--arm full] [--dry-n 420]

  T_arm   = wall-clock of the RL arm job (first 'start' to 'finished' line of its tag in logs/launch_ab2.log: start-up, step 0 dev evaluation, 49 steps, dev evaluations, snapshot)
  rate    = seconds per SFT step of the weak SFT (runs/sft_weak/metrics.jsonl, 'time' of step 100 -> step 3500)
  fixed   = job wall-clock of the weak SFT minus 3,500 x rate, minus 4 validation passes (S1 evaluates 3 times, W 7 times), floor 120 s
  N       = floor((T_arm - fixed) / rate), clipped to [100, steps available in data/packed_s1 (ab2_s1_stats.json)]
With --write the max_steps / eval_every of configs/ab2_sft_s1.yaml are rewritten. Also prints the dev curve of the RL arm(s) (the only input to the owner's optional extension, DECISION_RULE.txt 8).
Without any run on disk (--dry-n or no logs) it prints the central-case N of ab2_cost.py.
"""
import argparse
import datetime
import json
import os
import re
import sys


def ts(line):
    return datetime.datetime.strptime(line.split(" ")[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()


def job_seconds(log, tag):
    start = fin = None
    for l in open(log, encoding="utf-8", errors="replace"):
        if f"[launch-{tag}] start (" in l:
            start, fin = ts(l), None
        elif f"[launch-{tag}] finished" in l and start is not None:
            fin = ts(l)
    return (fin - start) if (start and fin) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ab", default="/workspace/jobs/ab2")
    ap.add_argument("--arm", default="full", choices=["full", "weak"])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--dry-n", type=int, default=0)
    a = ap.parse_args(argv)
    log = os.path.join(a.ab, "logs", "launch_ab2.log")
    cfgp = os.path.join(a.ab, "gemma_mt", "configs", "ab2_sft_s1.yaml")
    stats = os.path.join(a.ab, "data", "packed_s1", "ab2_s1_stats.json")
    avail = 600
    if os.path.exists(stats):
        st = json.load(open(stats))
        avail = int(st["steps"]) - int(st["from_step"])
    for arm in ("full", "weak"):
        p = os.path.join(a.ab, "runs", f"ab2_{arm}", "metrics.jsonl")
        if os.path.exists(p):
            rows = [json.loads(x) for x in open(p) if x.strip()]
            ev = [r for r in rows if "bench/chrf_macro" in r]
            print(f"dev curve arm {arm} (macro chrF++ on the clean dev rows of yor kin orm, greedy 256 tokens, step:value):", ", ".join(f"{r['step']}:{r['bench/chrf_macro']:.2f}" for r in ev))
    n = a.dry_n
    if not n:
        t_arm = job_seconds(log, "AB2_F" if a.arm == "full" else "AB2_W") if os.path.exists(log) else None
        t_sft = job_seconds(log, "AB2_SFT") if os.path.exists(log) else None
        mp = os.path.join(a.ab, "runs", "sft_weak", "metrics.jsonl")
        if not (t_arm and t_sft and os.path.exists(mp)):
            print("no measured times yet (need the finished weak SFT and the finished RL arm): using the cost-model central N = 420")
            n = 420
        else:
            rows = [json.loads(x) for x in open(mp) if x.strip()]
            tt = {r["step"]: r["time"] for r in rows if "train_loss" in r}
            s0, s1 = (100, 3500) if 100 in tt and 3500 in tt else (min(tt), max(tt))
            rate = (tt[s1] - tt[s0]) / (s1 - s0)
            fixed = max(120.0, t_sft - 3500 * rate - 4 * 60.0)
            n = int((t_arm - fixed) // rate)
            print(f"T_arm {t_arm:.0f} s, SFT rate {rate:.2f} s/step, fixed per SFT job {fixed:.0f} s -> N = {n}")
    n = max(100, min(n, avail))
    print(f"N = {n} SFT steps for S1 (data available: {avail} steps)")
    if a.write:
        t = open(cfgp).read()
        t = re.sub(r"^max_steps:.*$", f"max_steps: {n}", t, flags=re.M)
        t = re.sub(r"^eval_every:.*$", f"eval_every: {max(50, n // 3)}", t, flags=re.M)
        open(cfgp, "w").write(t)
        print("rewrote", cfgp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
