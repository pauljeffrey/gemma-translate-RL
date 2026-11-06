#!/usr/bin/env python
"""E-12 step 3: per-language table for all 125 registry languages, coverage by Glottolog family and script, figure."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/path/to/repo"
D = ROOT + "/reports/data"
OUT = D + "/family_script_resource_analysis_2026-11-04"
R = pd.read_csv(OUT + "/registry_125.csv")
CELLS = pd.read_csv(OUT + "/cells_gain_sft_rl_chrfpp.csv")
F = pd.read_csv(OUT + "/flores54_regression_table.csv")

# per-language gains from the three benchmarks that have an RL model (FLORES+, NTREX, MAFAND)
piv = CELLS.pivot(index="iso", columns="benchmark", values="gain")
piv.columns = ["gain_" + c for c in piv.columns]
sft = CELLS.pivot(index="iso", columns="benchmark", values="sft")
sft.columns = ["sft_" + c for c in sft.columns]
A = R.merge(piv, left_on="iso", right_index=True, how="left").merge(sft, left_on="iso", right_index=True, how="left")
A["has_rl_gain_any"] = A[["gain_flores_plus", "gain_ntrex", "gain_mafand"]].notna().any(axis=1).astype(int)
A["mean_gain_available_benchmarks"] = A[["gain_flores_plus", "gain_ntrex", "gain_mafand"]].mean(axis=1)
keep = ["iso", "language", "glottocode", "glottolog_name", "glottolog_top_family", "family_branch", "family_reg", "our_group_evid_table4", "script",
        "script_source", "sft_tokens_M", "sft_tokens_source", "share_nllb_mined", "africomet_available_flores", "nllb_3p3b_published_chrfpp",
        "n_benchmarks", "benchmarks_present", "sft_flores_plus", "gain_flores_plus", "sft_ntrex", "gain_ntrex", "sft_mafand", "gain_mafand",
        "has_rl_gain_any", "mapping_note"]
A[keep].to_csv(OUT + "/languages_125_family_script_gain.csv", index=False)


def cover(col):
    g = A.groupby(col)
    out = g.agg(n_registry=("iso", "size"), n_with_flores=("gain_flores_plus", lambda s: int(s.notna().sum())),
                n_with_any_rl_gain=("has_rl_gain_any", "sum"), tokens_M_sum=("sft_tokens_M", "sum"),
                tokens_M_median=("sft_tokens_M", "median"), n_no_token_data=("sft_tokens_M", lambda s: int(s.isna().sum())),
                n_no_reference_test=("n_benchmarks", lambda s: int((s == 0).sum()))).reset_index()
    out.insert(0, "grouping", col)
    out = out.rename(columns={col: "group"})
    return out


COV = pd.concat([cover(c) for c in ["glottolog_top_family", "family_reg", "family_branch", "script", "our_group_evid_table4"]])
COV.to_csv(OUT + "/coverage_125_by_group.csv", index=False)
print(COV.round(1).to_string())

# figure: gain vs SFT level and vs tokens, coloured by family_reg; marker by script
cols = {"Atlantic-Congo: Narrow Bantu": "#1b6ca8", "Atlantic-Congo: non-Bantu": "#59a14f", "Afro-Asiatic": "#e15759", "Nilotic": "#b07aa1",
        "Mande": "#f28e2b", "Indo-European (creoles, Afrikaans)": "#76b7b2", "Other (Austronesian, Saharan, Songhay, Ta-Ne-Omotic)": "#9c755f"}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for fam, d in F.groupby("family_reg"):
    for scr, dd in d.groupby("script_nonlatin"):
        ax[0].scatter(dd.log_tok, dd.gain, c=cols[fam], marker="o" if scr == 0 else "^", s=40, label=fam if scr == 0 else None, alpha=.85)
        ax[1].scatter(dd.sft, dd.gain, c=cols[fam], marker="o" if scr == 0 else "^", s=40, alpha=.85)
        ax[2].scatter(dd.log_tok, dd.sft, c=cols[fam], marker="o" if scr == 0 else "^", s=40, alpha=.85)
for iso, r in F.iterrows():
    if r.iso in ("kea", "orm", "nso", "taq", "zgh", "aeb", "ary", "yor", "afr", "amh", "dik"):
        ax[0].annotate(r.iso, (r.log_tok, r.gain), fontsize=7)
        ax[1].annotate(r.iso, (r.sft, r.gain), fontsize=7)
ax[0].set_xlabel("log10 SFT tokens (M)"); ax[0].set_ylabel("RL gain, mixed best - SFT (FLORES+ chrF++)")
ax[1].set_xlabel("SFT chrF++ (FLORES+)"); ax[1].set_ylabel("RL gain")
ax[2].set_xlabel("log10 SFT tokens (M)"); ax[2].set_ylabel("SFT chrF++ (FLORES+)")
ax[0].set_title("gain vs token budget (n=54)", fontsize=10); ax[1].set_title("gain vs SFT level", fontsize=10); ax[2].set_title("SFT level vs token budget", fontsize=10)
ax[1].legend(*ax[0].get_legend_handles_labels(), fontsize=6, loc="upper right"); fig.text(0.01, 0.01, "triangles: non-Latin script (Ethiopic, Arabic, Tifinagh). Families: Glottolog 5.3 top-level (Narrow Bantu split out).", fontsize=7)
plt.tight_layout(rect=(0, 0.03, 1, 1))
plt.savefig(OUT + "/fig_gain_sft_tokens_family.png", dpi=150)

# bar figure: gain by Glottolog branch
G = pd.read_csv(OUT + "/gain_by_group_flores54.csv")
g = G[G.group_col == "family_branch"].sort_values("mean_gain")
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh([f"{r.group} (n={r.n_lang})" for r in g.itertuples()], g.mean_gain, xerr=[g.mean_gain - g.boot_lo.fillna(g.mean_gain), g.boot_hi.fillna(g.mean_gain) - g.mean_gain], color="#4e79a7")
ax.set_xlabel("mean RL gain (mixed best - SFT), FLORES+ chrF++; bars = language-bootstrap 95% CI where n>1")
plt.tight_layout(); plt.savefig(OUT + "/fig_gain_by_glottolog_branch.png", dpi=150)
