#!/usr/bin/env python3
"""Step 9: how much of the SFT-stage regression is unfinished / looping output (a decoding failure that RL removes), and is NLLB share or the NLLB duplicate-target rate linked to it?

Per language, all-rows greedy runs of the 385M SFT and A0 (FLORES devtest, NTREX, MAFAND, SmolSent pooled by mean over the benchmarks a language has):
  unfinished = 1 - finished_m385 ; repeat_m385 ; d_repeat = repeat_m385 - repeat_a0 ; d_unfinished = (1-finished_m385) - (1-finished_a0)
Spearman (bootstrap 95% CI over languages) against NLLB-mined share, NLLB target-repeat rate (rows whose target occurs >= 3 times in the language's NLLB file), the SFT-stage regression R_sft
and the RL gain.  Also: the cross-language slope of R_sft on unfinished share.
"""
import os
import numpy as np, pandas as pd
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__))
D = os.path.abspath(os.path.join(H, ".."))
rng = np.random.default_rng(11)
g = pd.read_csv(os.path.join(D, "eval_385m_results_2026-11-03", "greedy_per_language_all_benchmarks.csv"))
g = g[(g.subset == "all") & g.benchmark.isin(["flores_plus", "ntrex", "mafand", "smol_sent"])]
g["unfin_m"] = 1 - g.finished_m385; g["unfin_a"] = 1 - g.finished_a0
p = g.groupby("iso").agg(unfin_m385=("unfin_m", "mean"), unfin_a0=("unfin_a", "mean"), repeat_m385=("repeat_m385", "mean"), repeat_a0=("repeat_a0", "mean")).reset_index()
p["d_repeat"] = p.repeat_m385 - p.repeat_a0
p["d_unfin"] = p.unfin_m385 - p.unfin_a0
L = pd.read_csv(os.path.join(H, "link_language_table.csv"))
q = pd.read_csv(os.path.join(H, "nllb_quality_by_language.csv"))[["iso", "nllb_rows_in_targets_repeated_3plus", "nllb_distinct_target_share"]]
d = L.merge(p, on="iso").merge(q, on="iso", how="left")
d["nllb_dup3"] = d.nllb_rows_in_targets_repeated_3plus.fillna(0)
d["nllb_dup_tokens_share"] = d.nllb_dup3 * d.share_nllb_mined
d.to_csv(os.path.join(H, "decoding_confound_table.csv"), index=False)


def sp(x, y, B=4000):
    m = d[[x, y]].dropna()
    r = stats.spearmanr(m[x], m[y])
    bs = [stats.spearmanr(m.iloc[i][x], m.iloc[i][y]).statistic for i in (rng.integers(0, len(m), len(m)) for _ in range(B))]
    return dict(x=x, y=y, n=len(m), rho=r.statistic, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5), p=r.pvalue)

res = pd.DataFrame([sp(x, y) for x, y in [
    ("unfin_m385", "R_sft_all4"), ("unfin_m385", "R_sft_big"), ("d_unfin", "R_a0_2sft"), ("d_unfin", "dB_pool"),
    ("share_nllb_mined", "unfin_m385"), ("share_nllb_mined", "d_unfin"), ("share_nllb_mined", "d_repeat"),
    ("nllb_dup_tokens_share", "d_repeat"), ("nllb_dup_tokens_share", "d_unfin"), ("nllb_dup3", "d_repeat"),
    ("share_nllb_mined", "rl_gain"), ("unfin_m385", "rl_gain"), ("R_sft_all4", "rl_gain")]])
res.to_csv(os.path.join(H, "decoding_confound_corr.csv"), index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string())
m = d[["R_sft_all4", "unfin_m385"]].dropna()
b = np.polyfit(m.unfin_m385, m.R_sft_all4, 1)
print("slope of R_sft_all4 on unfinished share:", b, "n", len(m), "R2", np.corrcoef(m.unfin_m385, m.R_sft_all4)[0, 1] ** 2)
m = d[["R_sft_big", "unfin_m385"]].dropna()
print("big set: slope", np.polyfit(m.unfin_m385, m.R_sft_big, 1), "R2", np.corrcoef(m.unfin_m385, m.R_sft_big)[0, 1] ** 2, "n", len(m))
