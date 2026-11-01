# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare the loss curves / validation losses / gradient norms of several runs (metrics.jsonl) against a reference run.

    python scripts/compare_runs.py runs/eq_sdpa runs/eq_varlen runs/eq_sdpa_repeat ...   (first = reference)"""
import json
import os
import sys


def load(d):
    rec = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
    f = lambda key: {r["step"]: r[key] for r in rec if key in r}
    return f("train_loss"), f("val_loss"), f("grad_norm")


def main(dirs):
    runs = {os.path.basename(d.rstrip("/")).replace("eq_", ""): load(d) for d in dirs}
    names = list(runs)
    ref = names[0]
    print("step " + "".join(f"{n:>16s}" for n in names))
    for s in sorted(runs[ref][0]):
        if s in (1, 2, 3, 5, 10, 15, 20, 25, 30):
            print(f"{s:4d} " + "".join(f"{runs[n][0].get(s, float('nan')):16.5f}" for n in names))
    for s in sorted(runs[ref][1]):
        print(f"val{s:3d}" + "".join(f"{runs[n][1].get(s, float('nan')):16.5f}" for n in names))
    print(f"\nvs {ref}:      max|dloss|  mean|dloss|  loss30  mean(last10) diff   max val diff   max gnorm rel diff   step-1 |dloss|")
    for n in names[1:]:
        a, b = runs[ref], runs[n]
        d = [abs(a[0][s] - b[0][s]) for s in a[0] if s in b[0]]
        l10 = sorted(a[0])[-10:]
        m10 = sum(b[0][s] - a[0][s] for s in l10) / 10
        vd = max(abs(a[1][s] - b[1][s]) for s in a[1] if s in b[1])
        gd = max(abs(a[2][s] - b[2][s]) / a[2][s] for s in a[2] if s in b[2])
        print(f"  {n:22s} {max(d):10.4f} {sum(d) / len(d):11.4f} {b[0][max(b[0])]:8.4f} {m10:+12.4f} {vd:15.4f} {gd:20.3f} {abs(a[0][1] - b[0][1]):15.5f}")


if __name__ == "__main__":
    main(sys.argv[1:])
