#!/usr/bin/env python3
"""W1 (2026-10-26): render per_language.csv into the markdown ranking table (reports/data/weak_selection_2026-10-26/ranking_table.md)."""
import csv, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "reports", "data", "weak_selection_2026-10-26")
rows = list(csv.DictReader(open(os.path.join(OUT, "per_language.csv"), encoding="utf-8")))
f = lambda x, d=1: "-" if x in ("", None) else f"{float(x):.{d}f}"
pc = lambda x: "-" if x in ("", None) else f"{100*float(x):.0f}"
L = ["| rank | iso | language | tier | chrF++ [95% CI] | P(in 20 lowest) | near cut | spBLEU | finished % | loop % | wrong-lang % | no-pair chrF++ (pair-seen %) | SFT token share % | devtest chrF++ (post hoc) | weak | why |", "|" + "---|" * 16]
for r in rows:
    rank = r["rank_asc"] if r["rank_asc"] else {"B": "B", "C": "C"}[r["tier"]]
    ci = f"{f(r['chrf'])} [{f(r['ci_lo'])}, {f(r['ci_hi'])}]" if r["chrf"] else "n/a (no reference)"
    npair = f"{f(r['chrf_nopair'])} ({pc(r['pair_seen_rate'])})" if r.get("chrf_nopair") else ("n/a" if r["tier"] != "C" else "-")
    L.append(f"| {rank} | {r['iso']} | {r['name']} | {r['tier']} | {ci} | {f(r['p_bottom20'],2)} | {'yes' if r['near_cut']=='True' else ''} | {f(r['spbleu'])} | {pc(r['finished'])} | {pc(r['loop'])} | {pc(r['wrong_lang'])} | {npair} | {f(r['sft_share_pct'],2)} | {f(r['devtest_chrf_posthoc'])} | {'**weak**' if r['weak']=='True' else ''} | {r['reasons']} |")
open(os.path.join(OUT, "ranking_table.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print(len(rows), "rows")
