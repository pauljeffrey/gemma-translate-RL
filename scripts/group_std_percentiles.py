#!/usr/bin/env python3
"""Percentiles of the within-group reward std from <run>/group_stats.jsonl (config log_group_stats). Use it to (re)set `adv_std_floor` (about the 10th percentile: only near-flat groups are damped),
`dynamic_sampling_min_std` and to see which languages have the flattest groups.

    python scripts/group_std_percentiles.py RUN_DIR [--from-step 20] [--by-language]
"""
import argparse
import collections
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--from-step", type=int, default=0)
ap.add_argument("--by-language", action="store_true")
a = ap.parse_args()
rows = [json.loads(l) for l in open(os.path.join(a.run, "group_stats.jsonl")) if l.strip()]
rows = [r for r in rows if r["step"] >= a.from_step]
std = np.asarray([r["std"] for r in rows])
print(f"{len(rows)} groups, steps {min(r['step'] for r in rows)}..{max(r['step'] for r in rows)}")
print("group std percentiles:", {f"p{q}": round(float(np.percentile(std, q)), 4) for q in (1, 5, 10, 25, 50, 75, 90)})
print("share exactly flat (< 1e-6):", round(float((std < 1e-6).mean()), 4), " share below 0.01:", round(float((std < 0.01).mean()), 4), " below 0.02:", round(float((std < 0.02).mean()), 4), " below 0.05:", round(float((std < 0.05).mean()), 4))
g = [r for r in rows if r["guide"]]
if g:
    print(f"guided groups {len(g)}: median std {np.median([r['std'] for r in g]):.4f} (all groups {np.median(std):.4f})")
if a.by_language:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["iso"]].append(r["std"])
    for iso, v in sorted(by.items(), key=lambda kv: np.median(kv[1])):
        print(f"  {iso}: n={len(v)} median {np.median(v):.3f} p10 {np.percentile(v, 10):.3f} flat {np.mean(np.asarray(v) < 1e-6):.2f}")
