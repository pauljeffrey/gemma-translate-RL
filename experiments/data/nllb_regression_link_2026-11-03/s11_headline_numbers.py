#!/usr/bin/env python3
"""Step 11: the numbers quoted in the report (macro regressions per benchmark and stage, counts, AfriScience, power) -> headline_numbers.json, tables_extra.md"""
import json, os
import numpy as np, pandas as pd
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__))
L = pd.read_csv(os.path.join(H, "link_language_table_full.csv"))
E = pd.read_csv(os.path.join(H, "eval_by_language_benchmark.csv"))
out = {}
# macro regressions per benchmark and stage (old-unseen sentences; equal language weight)
mac = E.groupby("bm").agg(n_lang=("iso", "count"), n_sent_median=("n_clean", "median"), m385_sft=("m385_sft_clean", "mean"), old_sft=("old_sft", "mean"), old_rl=("old_rl", "mean"),
                          R_sft=("R_sft", "mean"), R_sft_vs_oldrl=("R_sft_vs_oldrl", "mean"), R_A_est=("R_A_est", "mean"), R_rl_est=("R_rl_est", "mean"), a0=("a0_all", "mean"), m385_all=("m385_sft_all", "mean"),
                          b1_all=("b1_all", "mean")).round(2)
mac.to_csv(os.path.join(H, "macro_regressions_by_benchmark.csv"))
print(mac.to_string())
for b in ["ntrex", "mafand", "smol_sent", "flores_plus"]:
    d = E[E.bm == b]
    out[b] = dict(n=len(d), R_sft_mean=float(d.R_sft.mean()), n_R_sft_lt_minus3=int((d.R_sft < -3).sum()), n_R_sft_lt_minus5=int((d.R_sft < -5).sum()),
                  R_rl_mean=float(d.R_rl_est.mean()) if d.R_rl_est.notna().any() else None, n_R_rl_lt_minus3=int((d.R_rl_est < -3).sum()), n_R_rl_lt_minus5=int((d.R_rl_est < -5).sum()))
out["pooled_langs"] = dict(n_all4=int(L.R_sft_all4.notna().sum()), n_big=int(L.R_sft_big.notna().sum()),
                           n_R_sft_all4_lt_minus5=int((L.R_sft_all4 < -5).sum()), n_R_rl_all3_lt_minus3=int((L.R_rl_all3 < -3).sum()), n_R_rl_big_lt_minus3=int((L.R_rl_big < -3).sum()),
                           R_rl_all3_list=L[L.R_rl_all3 < -3][["iso", "R_rl_all3", "share_nllb_mined"]].round(2).values.tolist())
reg = L[L.R_sft_all4 < -5]
out["regressed_gt5"] = dict(n=len(reg), mean_R_sft=float(reg.R_sft_all4.mean()), mean_R_old2a0=float(reg.R_old2a0.mean()), mean_R_a0_2sft=float(reg.R_a0_2sft.mean()),
                            n_a0_already_below_minus5=int((reg.R_old2a0 < -5).sum()), n_stageB_nonneg=int((reg.R_a0_2sft >= 0).sum()), isos=reg.iso.tolist(),
                            mean_share=float(reg.share_nllb_mined.mean()), share_gt0=int((reg.share_nllb_mined > 0).sum()))
non = L[(L.R_sft_all4.notna()) & ~L.iso.isin(reg.iso)]
out["not_regressed"] = dict(n=len(non), mean_share=float(non.share_nllb_mined.mean()), share_gt0=int((non.share_nllb_mined > 0).sum()))
# AfriScience (test, sentence, plain chrF): Table 5 of reports/eval_post_B1_385m_2026-11-03.md
afs = pd.DataFrame({"iso": ["amh", "hau", "lug", "nso", "yor", "zul"], "m385_sft": [20.30, 47.33, 23.31, 41.27, 25.36, 58.42], "stageA": [28.75, 54.44, 36.92, 51.98, 37.55, 60.72], "b1": [28.85, 55.35, 37.22, 52.15, 37.72, 60.53],
                    "old_rl_final": [25.3, 59.4, 46.2, 55.9, 37.4, 63.4], "old_A2W_science_stage": [26.2, 62.0, 48.9, 58.3, 39.8, 64.7]})
afs = afs.merge(L[["iso", "share_nllb_mined"]], on="iso")
afs["R_b1_vs_old_rl"] = afs.b1 - afs.old_rl_final
afs["R_sft_vs_old_rl"] = afs.m385_sft - afs.old_rl_final
afs.to_csv(os.path.join(H, "afriscience_regression.csv"), index=False)
r = stats.spearmanr(afs.share_nllb_mined, afs.R_b1_vs_old_rl)
out["afriscience"] = dict(n=6, rho_share_vs_R_b1=float(r.statistic), p=float(r.pvalue), note="old RL-final / science stage trained on AfriScience-style domain data (the 385M has had no science stage); n=6; not a clean regression test")
# power: smallest |rho| detected with 80% power (two-sided 5%) at n
for n in (15, 28, 45, 57):
    z = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) / np.sqrt(n - 3)
    out.setdefault("power_min_abs_rho", {})[n] = float(np.tanh(z))
json.dump(out, open(os.path.join(H, "headline_numbers.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:4000])
print(afs.round(2).to_string())
