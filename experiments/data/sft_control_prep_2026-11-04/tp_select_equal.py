#!/usr/bin/env python
"""E-03 control, equal-samples-per-language TranslatePsy half.

Takes the cleaned en->xx rows (tp_prepare.py output, one jsonl per language), shuffles each language with a fixed seed and writes EXACTLY N training rows and V validation rows per
language (the same N for all 19 languages: "fixed n_samples per language"), plus a per-language table.

    python tp_select_equal.py --in tp_clean --out-train tp_train --out-val tp_val --n 281000 --val 600 [--seed 1234] [--table table.csv]
"""
import argparse, csv, json, os, random, sys

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", nargs="+", required=True, help="one or more tp_prepare output dirs")
ap.add_argument("--out-train", required=True); ap.add_argument("--out-val", required=True)
ap.add_argument("--n", type=int, required=True); ap.add_argument("--val", type=int, default=600); ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--table", default=None)
a = ap.parse_args()
os.makedirs(a.out_train, exist_ok=True); os.makedirs(a.out_val, exist_ok=True)
files = {}
for d in a.inp:
    for f in os.listdir(d):
        if f.startswith("en2x_") and f.endswith(".jsonl"): files[f[5:-6]] = os.path.join(d, f)
rows = []
for iso in sorted(files):
    L = [l for l in open(files[iso], encoding="utf-8")]
    rng = random.Random(f"{a.seed}-{iso}"); rng.shuffle(L)
    if len(L) < a.n + a.val: sys.exit(f"{iso}: only {len(L)} clean rows, need {a.n + a.val}")
    with open(f"{a.out_train}/en2x_{iso}.jsonl", "w", encoding="utf-8") as f: f.writelines(L[:a.n])
    with open(f"{a.out_val}/en2x_{iso}.jsonl", "w", encoding="utf-8") as f: f.writelines(L[a.n:a.n + a.val])
    rows.append({"iso": iso, "clean_rows_available": len(L), "n_train_samples": a.n, "n_val_samples": a.val})
    print(iso, len(L), "->", a.n, "+", a.val, flush=True)
if a.table:
    with open(a.table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
