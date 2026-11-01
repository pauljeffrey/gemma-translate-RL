#!/usr/bin/env python3
"""Task 3 of the 2026-10-26 contamination audit: per-benchmark tables from reports/data/final_model_per_language.csv.

Reads (all in the repository, nothing is downloaded):
  reports/data/final_model_per_language.csv        benchmark, subset (all/no_pair/clean), checkpoint (sft_step27638, rl_final_step1433)
  reports/data/final_model_runs/macro_boot.json    paired sentence-bootstrap CI of RL - SFT, macro over languages, per benchmark x subset
  reports/data/final_model_runs/bench_boot.csv     the same per language
  reports/data/model_comparison_master.csv         NLLB-3.3B / 600M / final2 / A2W re-run rows (FLORES+ only has a `clean` subset)
Writes reports/data/contamination_audit_2026-10-26/{benchmark_subset_table.csv, nllb_clean_table.csv, highlight_languages.csv}.
Macro = unweighted mean over languages with at least MIN_ROWS rows in the subset (same rule as the paper's `clean` macro: >= 30 rows is the
SFT report's rule; here 20, and the number of languages is printed so the reader can see both).
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from contam_audit_common import OUT, REPO  # noqa: E402

MIN_ROWS = 20
D = os.path.join(REPO, "reports", "data")
os.makedirs(OUT, exist_ok=True)

pl = pd.read_csv(os.path.join(D, "final_model_per_language.csv"))
boot = json.load(open(os.path.join(D, "final_model_runs", "macro_boot.json")))
BENCH = ["flores_plus", "ntrex", "mafand", "smol_sent", "glot_story", "salt", "ibom_mt", "flores_moz"]
rows = []
for b in BENCH:
    sub = pl[pl.benchmark == b]
    allrows = sub[(sub.subset == "all") & (sub.checkpoint == "sft_step27638")].set_index("iso").n
    n_all = int(allrows.sum())
    rec = {"benchmark": b, "languages": int(allrows.shape[0]), "rows_all": n_all}
    for s in ("no_pair", "clean"):
        ss = sub[(sub.subset == s) & (sub.checkpoint == "sft_step27638")].set_index("iso").n
        rec[f"rows_{s}"] = int(ss.sum())
        rec[f"share_{s}_pct"] = round(100 * ss.sum() / n_all, 1)
    for s in ("all", "no_pair", "clean"):
        vals = {}
        for ck in ("sft_step27638", "rl_final_step1433"):
            t = sub[(sub.subset == s) & (sub.checkpoint == ck) & (sub.n >= MIN_ROWS)].set_index("iso")
            vals[ck] = t
        common = vals["sft_step27638"].index.intersection(vals["rl_final_step1433"].index)
        rec[f"langs_{s}"] = len(common)
        for ck, tag in (("sft_step27638", "sft"), ("rl_final_step1433", "rl")):
            rec[f"{tag}_chrf_{s}"] = round(vals[ck].loc[common, "chrf"].mean(), 2) if len(common) else None
            rec[f"{tag}_spbleu_{s}"] = round(vals[ck].loc[common, "spbleu"].mean(), 2) if len(common) else None
        k = f"{b}|{s}|all"
        if k in boot:
            rec[f"d_chrf_{s}"] = round(boot[k]["chrf"]["d"], 2)
            rec[f"d_chrf_{s}_lo"] = round(boot[k]["chrf"]["lo"], 2)
            rec[f"d_chrf_{s}_hi"] = round(boot[k]["chrf"]["hi"], 2)
            rec[f"d_spbleu_{s}"] = round(boot[k]["spbleu"]["d"], 2)
    rows.append(rec)
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(OUT, "benchmark_subset_table.csv"), index=False)
print(tab.T.to_string())

# ---- NLLB / final2 / A2W on FLORES+ (the only benchmark where model_comparison_master has a clean subset)
m = pd.read_csv(os.path.join(D, "model_comparison_master.csv"))
m = m[m.benchmark == "FLORES+ devtest en->xx"]
short = {"NLLB-200-3.3B (re-run: CT2 int8, beam 4)": "nllb3.3b", "NLLB-200-distilled-600M (re-run: CT2 fp16, beam 4)": "nllb600m",
         "SabiYarn-270M RL-final (GRPO step 1433)": "rl_final", "SabiYarn-270M SFT-final (step 27638)": "sft_final",
         "SabiYarn-270M final run (final2 step 401)": "final2", "SabiYarn-270M science-stage start (A2W alpha 0.7)": "a2w"}
m = m[m.model.isin(short)].copy()
m["sys"] = m.model.map(short)
out = []
nl = set(m[m.sys == "nllb3.3b"].language)
for s in ("all", "no_pair", "clean"):
    piv = m[m.subset == s].pivot_table(index="language", columns="sys", values="chrF++")
    piv = piv.loc[piv.index.isin(nl)].dropna(subset=["nllb3.3b", "rl_final", "sft_final"])
    r = {"subset": s, "languages": len(piv)}
    for c in ("sft_final", "rl_final", "final2", "a2w", "nllb3.3b", "nllb600m"):
        if c in piv:
            r[c] = round(piv[c].mean(), 2)
    r["rl_minus_nllb3.3b"] = round((piv.rl_final - piv["nllb3.3b"]).mean(), 2)
    r["rl_minus_sft"] = round((piv.rl_final - piv.sft_final).mean(), 2)
    r["langs_rl_above_nllb"] = int((piv.rl_final > piv["nllb3.3b"]).sum())
    out.append(r)
pd.DataFrame(out).to_csv(os.path.join(OUT, "nllb_clean_table.csv"), index=False)
print(pd.DataFrame(out).to_string())

# ---- highlight languages (kea kbp nus kam luo) SFT -> RL on FLORES+ by subset, from the per-language paired bootstrap
bb = pd.read_csv(os.path.join(D, "final_model_runs", "bench_boot.csv"))
hl = bb[(bb.benchmark == "flores_plus") & (bb.iso.isin(["kea", "kbp", "nus", "kam", "luo", "kik", "bam", "kmb"]))]
hl.to_csv(os.path.join(OUT, "highlight_languages.csv"), index=False)
print(hl.pivot_table(index="iso", columns="subset", values=["d_chrf", "d_chrf_lo", "d_chrf_hi", "n"]).round(2).to_string())
