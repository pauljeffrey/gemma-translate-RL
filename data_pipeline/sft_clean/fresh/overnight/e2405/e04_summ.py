#!/usr/bin/env python
"""Summary of a finished (or running) RL run directory: dev evaluations, KL / guard / speed statistics, per-step reward series.  python e04_summ.py RUN_DIR OUT.json"""
import json, statistics, sys
run, out = sys.argv[1:3]; tr, ev = [], []
for l in open(run + "/metrics.jsonl"):
    try: r = json.loads(l)
    except Exception: continue
    if "reward" in r and "kl" in r: tr.append(r)
    if "bench/chrf_macro" in r: ev.append(r)
tr.sort(key=lambda r: r["step"]); med = lambda x: statistics.median(x) if x else None
dts = [b["time"] - a["time"] for a, b in zip(tr, tr[1:]) if b["step"] == a["step"] + 1 and 0 < b["time"] - a["time"] < 600]
S = {"run": run, "steps_logged": len(tr), "last_step": tr[-1]["step"] if tr else None,
     "dev": [{k: r.get(k) for k in ("step", "bench/chrf_macro", "bench/bleu_macro", "bench/finished_rate")} | {"weak_dev_chrf": r.get("bench_aux/weak_dev/chrf_macro")} for r in sorted(ev, key=lambda r: r["step"])],
     "dev_per_language_chrf": {r["step"]: {k.split("/")[-1]: v for k, v in r.items() if k.startswith("bench/chrf/")} for r in ev},
     "step_time_median_s": med(dts), "kl_median_by_10": {s: med([r["kl"] for r in tr if s < r["step"] <= s + 10]) for s in range(0, 100, 10)},
     "kl_max_guard_median": max([float(r.get("kl_guard_median") or 0) for r in tr] or [0]), "kl_guard_cuts": sum(1 for r in tr if r.get("kl_guard_cut")),
     "reward_median_by_10": {s: med([r["reward"] for r in tr if s < r["step"] <= s + 10]) for s in range(0, 100, 10)},
     "finished_median_by_10": {s: med([r["finished_rate"] for r in tr if s < r["step"] <= s + 10 and "finished_rate" in r]) for s in range(0, 100, 10)},
     "greedy_loop_median_by_10": {s: med([r["greedy_loop_rate"] for r in tr if s < r["step"] <= s + 10 and "greedy_loop_rate" in r]) for s in range(0, 100, 10)},
     "guide_overhead_median": med([r["guide/overhead_frac"] for r in tr if "guide/overhead_frac" in r]), "grad_norm_median": med([r["grad_norm"] for r in tr if "grad_norm" in r])}
json.dump(S, open(out, "w"), indent=1, default=str); print(json.dumps({k: S[k] for k in ("steps_logged", "last_step", "step_time_median_s", "kl_max_guard_median", "kl_guard_cuts")}))
