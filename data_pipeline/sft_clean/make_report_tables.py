#!/usr/bin/env python3
"""Merge the rebuild statistics with the old per-language numbers (training plan 2026-10-14 table, language_token_proportions_2026-10-25.csv) -> comparison CSV + markdown tables.
Run on the laptop after copying out/stats/* to reports/data/sft_clean_2026-10-28/ .  Old numbers are windows and Gemma target tokens AFTER the QE gate and windowing;
the rebuilt numbers are pairs and ESTIMATED target tokens (chars / chars-per-token from a Gemma-3 tokenizer sample) BEFORE the QE gate: indicative, not like for like."""
import csv, json, os, re, sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
D = os.path.join(REPO, "reports", "data", "sft_clean_2026-10-28")
plan = open(os.path.join(REPO, "reports", "gemma_sft_training_plan_2026-10-14.md"), encoding="utf-8").read()
old = {}
for line in plan.splitlines():
    m = re.match(r"\| ([a-z]{3}) \| ([^|]+) \| ([\d,]+) / ([\d,]+) / ([\d,]+) / ([\d,]+) \| ([\d,]+) \| ([\d,]+) \|", line)
    if m:
        old[m.group(1)] = {"name": m.group(2).strip(), "windows_all": sum(int(m.group(i).replace(",", "")) for i in (3, 4, 5, 6)), "train_samples": int(m.group(7).replace(",", "")),
                           "tokens": int(m.group(8).replace(",", ""))}
for r in csv.DictReader(open(os.path.join(REPO, "reports", "data", "language_token_proportions_2026-10-25.csv"))):
    old.setdefault(r["iso"], {"name": r["name"]})["tokens_csv"] = int(r["sft_target_tokens"])
new = {r["iso"]: r for r in csv.DictReader(open(os.path.join(D, "per_language.csv")))}
isos = sorted(set(old) | set(new), key=lambda i: -int(new.get(i, {}).get("est_tokens_before", 0) or 0))
rows = []
for i in isos:
    o, n = old.get(i, {}), new.get(i)
    ob = o.get("tokens") or o.get("tokens_csv") or 0
    tb = int(n["est_tokens_before"]) if n else 0
    ta = int(n["est_tokens_after"]) if n else 0
    pb = int(n["pairs_before"]) if n else 0
    pa_ = int(n["pairs_after"]) if n else 0
    loss = 100 * (pb - pa_) / pb if pb else 0.0
    flag = []
    if (n is None) or pa_ < 2000:
        flag.append("BELOW_2000_PAIRS")
    if loss > 20:
        flag.append("LOSS_GT_20PCT")
    if ob and ta < 0.2 * ob:
        flag.append("COVERAGE_LT_20PCT_OF_OLD_TOKENS")
    rows.append([i, o.get("name", ""), o.get("train_samples", ""), ob, pb, pa_, round(loss, 1), tb, ta, round(100 * ta / ob, 1) if ob else "", "; ".join(flag),
                 n["removed_R1"] if n else "", n["removed_only_R3_8gram"] if n else ""])
out = os.path.join(D, "comparison_old_vs_rebuilt.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["iso", "name", "old_train_windows", "old_target_tokens", "pairs_before_filter", "pairs_after_filter", "pairs_removed_pct", "est_tokens_before", "est_tokens_after",
                "rebuilt_after_as_pct_of_old_tokens", "flags", "removed_R1_exact", "removed_only_by_R3_8gram"])
    w.writerows(rows)
md = ["| iso | language | old windows | old target tokens | pairs before | pairs after | removed % | est. tokens after | after / old tokens % | flags |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
for r in rows:
    md.append(f"| {r[0]} | {r[1]} | {r[2]:,} | {r[3]:,} | {r[4]:,} | {r[5]:,} | {r[6]} | {r[8]:,} | {r[9]} | {r[10]} |".replace("| ,", "| ").replace(" |  |", " | |") if isinstance(r[2], int) else
              f"| {r[0]} | {r[1]} | - | {r[3]:,} | {r[4]:,} | {r[5]:,} | {r[6]} | {r[8]:,} | {r[9]} | {r[10]} |")
open(os.path.join(D, "comparison_old_vs_rebuilt.md"), "w").write("\n".join(md) + "\n")
print(len(rows), "languages ->", out)
