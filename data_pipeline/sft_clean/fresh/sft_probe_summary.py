#!/usr/bin/env python3
"""Combined throughput of SFT runs from their metrics.jsonl (written by train_sft.py: {step, time, tokens_per_s, ...} every log_every steps).
    python sft_probe_summary.py runs/probe_1job runs/probe_2job_a runs/probe_2job_b [--skip-steps 10] [--group 1job=runs/probe_1job --group 2jobs=runs/probe_2job_a,runs/probe_2job_b]
Steady-state steps/s of a run = (last_step - first_step) / (last_time - first_time) over the records after --skip-steps (warm-up, compile, first-touch of the data excluded).
A group's combined steps/s = the SUM over its runs, valid only while the runs overlapped in time (the script reports the overlap fraction of the measurement windows).
"""
import argparse, json, os, sys


def rate(d, skip):
    recs = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl")) if l.strip()]
    recs = [r for r in recs if "tokens_per_s" in r and r["step"] > skip]
    if len(recs) < 2: return None
    a, b = recs[0], recs[-1]
    dt = b["time"] - a["time"]
    tok_step = None
    return {"dir": d, "t0": a["time"], "t1": b["time"], "steps": b["step"] - a["step"], "steps_per_s": (b["step"] - a["step"]) / dt, "avg_tokens_per_s_run": b["tokens_per_s"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-steps", type=int, default=10); ap.add_argument("--group", action="append", default=[], help="label=dir1,dir2")
    ap.add_argument("dirs", nargs="*"); ap.add_argument("--seq-tokens-per-step", type=int, default=131072)
    a = ap.parse_args()
    groups = {}
    for g in a.group:
        lab, ds = g.split("=", 1); groups[lab] = ds.split(",")
    if not groups: groups = {"all": a.dirs}
    res = {}
    print(f"{'group':10s} {'runs':>4s} {'combined steps/s':>17s} {'combined tokens/s':>18s} {'overlap':>8s}   per-run steps/s")
    for lab, ds in groups.items():
        rs = [r for r in (rate(d, a.skip_steps) for d in ds) if r]
        if not rs: print(f"{lab:10s}  no usable metrics"); continue
        lo, hi = max(r["t0"] for r in rs), min(r["t1"] for r in rs)
        ov = max(0.0, hi - lo) / max(1e-9, min(r["t1"] - r["t0"] for r in rs))
        comb = sum(r["steps_per_s"] for r in rs)
        res[lab] = {"combined_steps_per_s": comb, "combined_tokens_per_s": comb * a.seq_tokens_per_step, "overlap": ov, "runs": rs}
        print(f"{lab:10s} {len(rs):4d} {comb:17.3f} {comb * a.seq_tokens_per_step:18,.0f} {ov:8.2f}   " + ", ".join(f"{r['steps_per_s']:.3f}" for r in rs))
    labs = list(res)
    if len(labs) >= 2:
        base = res[labs[0]]["combined_steps_per_s"]
        for l in labs[1:]:
            x = res[l]["combined_steps_per_s"] / base
            print(f"{l} vs {labs[0]}: combined throughput x{x:.2f}  -> " + ("run two jobs at once" if x >= 1.15 and res[l]["overlap"] > 0.8 else "one job at a time (gain < 15% or runs did not overlap)"))
    json.dump(res, open("sft_probe_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
