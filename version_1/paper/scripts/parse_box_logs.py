#!/usr/bin/env python3
"""Parse the plain-text trainer logs of the SFT / RL phases 1, 2, 2b (copied READ-ONLY from the GPU box, /workspace/tmp/logs/)
into small CSVs under paper/data/.  Only needed once: the raw logs are not kept in the repo.

    python paper/scripts/parse_box_logs.py LOGDIR
LOGDIR must contain rl_rtxpro.log (phase 1), rl2_rtxpro.log (phase 2), rl2b_rtxpro.log (phase 2b),
sft_unimax_rtxpro.log and sft_unimax_rtxpro_prev_*.log (SFT).
Output: sft_eval.csv, rl{1,2,2b}_steps.csv, rl{1,2,2b}_evals.csv (last occurrence of a step wins: restarts replay steps)."""
import csv, glob, os, re, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
STEP = re.compile(r"rl step (\d+)/(\d+) reward (\S+) kl (\S+) fin (\S+) len (\S+) flat (\S+) gnorm (\S+)")
EVAL = re.compile(r"eval @ (\d+): reward (\S+) finished (\S+)(?: \| dev chrF\+\+ macro (\S+) \(SFT start (\S+)\) BLEU (\S+))?")
SFTEV = re.compile(r"eval @ (\d+): val_loss (\S+) macro (\S+)")
START = re.compile(r"starting point on (\d+) dev sentences: chrF\+\+ macro (\S+) BLEU (\S+) finished (\S+)")


def parse_rl(path):
    steps, evals, start = {}, {}, None
    for line in open(path, errors="replace"):
        m = STEP.search(line)
        if m:
            s = int(m.group(1)); steps[s] = dict(step=s, total=int(m.group(2)), reward=float(m.group(3)), kl=float(m.group(4)),
                                                 finished=float(m.group(5)), completion_len=float(m.group(6)), flat=float(m.group(7)), grad_norm=float(m.group(8)))
            continue
        m = EVAL.search(line)
        if m:
            s = int(m.group(1)); evals[s] = dict(step=s, eval_reward=float(m.group(2)), eval_finished=float(m.group(3)),
                                                 dev_chrf=float(m.group(4)) if m.group(4) else "", dev_bleu=float(m.group(6)) if m.group(6) else "",
                                                 sft_start=float(m.group(5)) if m.group(5) else "")
            continue
        m = START.search(line)
        if m and start is None:
            start = dict(n=int(m.group(1)), chrf=float(m.group(2)), bleu=float(m.group(3)), finished=float(m.group(4)))
    return [steps[k] for k in sorted(steps)], [evals[k] for k in sorted(evals)], start


def write(name, rows):
    if not rows:
        return
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", name, len(rows))


def main(logdir):
    for tag, fn in (("rl1", "rl_rtxpro.log"), ("rl2", "rl2_rtxpro.log"), ("rl2b", "rl2b_rtxpro.log")):
        steps, evals, start = parse_rl(os.path.join(logdir, fn))
        write(f"{tag}_steps.csv", steps); write(f"{tag}_evals.csv", evals)
        print(tag, "start bar:", start)
    ev = {}
    for p in sorted(glob.glob(os.path.join(logdir, "sft_unimax_rtxpro*.log"))):
        for line in open(p, errors="replace"):
            m = SFTEV.search(line)
            if m:
                ev[int(m.group(1))] = dict(step=int(m.group(1)), val_loss=float(m.group(2)), val_macro=float(m.group(3)))
    write("sft_eval.csv", [ev[k] for k in sorted(ev)])


if __name__ == "__main__":
    main(sys.argv[1])
