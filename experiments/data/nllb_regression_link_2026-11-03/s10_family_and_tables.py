#!/usr/bin/env python3
"""Step 10: within-family Spearman (bootstrap CI) and the markdown tables of the report (tables.md)."""
import os, json
import numpy as np, pandas as pd
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__))
D = os.path.abspath(os.path.join(H, ".."))
rng = np.random.default_rng(3)
L = pd.read_csv(os.path.join(H, "link_language_table.csv"))
own = pd.read_csv(os.path.join(D, "eval_385m_plan_2026-11-02", "own_270m_per_language.csv"))
fam = own.drop_duplicates("iso").set_index("iso").family
names = pd.read_csv(os.path.join(D, "eval_385m_results_2026-11-03", "greedy_per_language_all_benchmarks.csv")).drop_duplicates("iso").set_index("iso").language
L["family"] = L.iso.map(fam).fillna("unlabelled")
L["name"] = L.iso.map(names)
sh = pd.read_csv(os.path.join(H, "nllb_share_by_language.csv"))
A0 = pd.read_csv(os.path.join(D, "token_table_provisional_2026-11-01.csv"))[["iso", "A0_trained_Mtok"]]
L = L.merge(A0, on="iso", how="left")
q = pd.read_csv(os.path.join(H, "nllb_quality_by_language.csv"))
L = L.merge(q[["iso", "lid_target_ok_nllb_mined", "lid_target_ok_other", "lid_target_eng_nllb_mined", "nllb_distinct_target_share", "nllb_rows_in_targets_repeated_3plus", "nllb_qe_mean", "lang_median_qe_all_sources", "nllb_keep_rate"]], on="iso", how="left")
L.to_csv(os.path.join(H, "link_language_table_full.csv"), index=False)


def sp(d, x, y, B=4000):
    m = d[[x, y]].dropna()
    if len(m) < 7 or m[x].nunique() < 3:
        return None
    r = stats.spearmanr(m[x], m[y])
    bs = [stats.spearmanr(m.iloc[i][x], m.iloc[i][y]).statistic for i in (rng.integers(0, len(m), len(m)) for _ in range(B))]
    return dict(n=len(m), rho=r.statistic, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5), p=r.pvalue)

rows = []
for f in ["Bantu", "Atlantic-Congo (non-Bantu)", "Afroasiatic", "all Bantu+Atlantic-Congo", "all"]:
    d = L if f == "all" else (L[L.family.isin(["Bantu", "Atlantic-Congo (non-Bantu)"])] if f == "all Bantu+Atlantic-Congo" else L[L.family == f])
    for y in ["R_sft_big", "R_sft_all4", "R_old2a0", "R_a0_2sft", "dB_pool", "R_rl_all3"]:
        r = sp(d, "share_nllb_mined", y)
        if r:
            rows.append(dict(family=f, target=y, **r))
fc = pd.DataFrame(rows)
fc.to_csv(os.path.join(H, "link_family_spearman.csv"), index=False)
print(fc.round(3).to_string())

# ---- markdown tables
def md(df, fmt=None):
    cols = df.columns.tolist()
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.2f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(out)

ev_isos = set(pd.read_csv(os.path.join(D, "eval_385m_results_2026-11-03", "greedy_per_language_all_benchmarks.csv")).iso)
T = L[L.iso.isin(ev_isos)].copy()
T["B_Mtok"] = T.tokens_B / 1e6
T["NLLB_rows_k"] = T.rows_nllb_mined / 1e3
T["NLLB_Mtok"] = T.nllb_mined_tokens / 1e6
T["share_mined_%"] = T.share_nllb_mined * 100
T["share_strict_%"] = T.share_nllb_strict * 100
T["share_of_transl_%"] = T.share_nllb_mined_of_translation * 100
T["instr_%"] = T.instr_share * 100
T = T.sort_values("share_nllb_mined", ascending=False)
t2 = T[["iso", "name", "B_Mtok", "NLLB_rows_k", "NLLB_Mtok", "share_mined_%", "share_of_transl_%", "share_strict_%", "instr_%", "A0_trained_Mtok"]].round(1)
t2.to_csv(os.path.join(H, "nllb_share_eval_languages.csv"), index=False)
open(os.path.join(H, "tables.md"), "w").write("## T2 NLLB share per eval language\n" + md(t2) + "\n")
print(md(t2.head(5)))
