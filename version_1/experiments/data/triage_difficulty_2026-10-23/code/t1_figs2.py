from t1load import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = OUT + "/figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})
COL = {"rl": "#0072B2", "nllb3b3": "#D55E00", "sft": "#7f7f7f", "nllb600m": "#009E73", "a2w": "#56B4E9", "tpsy08": "#CC79A7"}
# F5 dimensions
t = pd.read_csv(OUT + "/06_dimension_partial_correlations.csv")
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
dims = DIMS
for ax, met in zip(axs, ("AfriCOMET", "chrF++")):
    for j, (ds, c) in enumerate((("FLORES+ devtest", "#0072B2"), ("NTREX-128", "#E69F00"), ("MAFAND-MT", "#009E73"), ("pooled (all benchmarks)", "#000000"))):
        d = t[(t.dataset == ds) & (t.k == 1) & (t.model == "rl") & (t.metric == met)].set_index("dimension")
        y = np.arange(len(dims)) + (j - 1.5) * 0.18
        v = d.loc[dims, "partial_r_given_length_and_other_four_dims"].values; lo = d.loc[dims, "partial_r_given_length_and_other_four_dims_lo"].values; hi = d.loc[dims, "partial_r_given_length_and_other_four_dims_hi"].values
        ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="o", ms=3.5, color=c, lw=1, label=ds)
    ax.axvline(0, color="k", lw=0.6); ax.set_yticks(range(len(dims))); ax.set_yticklabels(dims); ax.set_title(f"RL-final {met}: partial correlation with each dimension\n(given source length and the other four dimensions)", fontsize=9)
axs[0].legend(frameon=False, fontsize=7); axs[0].invert_yaxis()
fig.tight_layout(); fig.savefig(f"{FIG}/fig5_dimension_partial_correlations.png"); plt.close(fig)
# F6 agreement
a = pd.read_csv(OUT + "/08_agreement_with_nllb3b3_by_band.csv")
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, ds in zip(axs, ["FLORES+ devtest", "NTREX-128", "MAFAND-MT"]):
    d = a[(a.dataset == ds) & (a.k == 1) & (a.group == "all") & (a.band.isin(["very easy", "easy", "moderately difficult"]))]
    systems = [s for s in ["rl", "sft", "nllb600m", "tpsy08"] if s in set(d.system)]
    w = 0.8 / len(systems)
    for i, s in enumerate(systems):
        dd = d[d.system == s].set_index("band").reindex(["very easy", "easy", "moderately difficult"])
        x = np.arange(3) + (i - (len(systems) - 1) / 2) * w
        ax.bar(x, dd["chrf_vs_nllb3b3_output(pseudo_ref)"], width=w * 0.9, color=COL[s], alpha=0.9, label=f"{s}: vs NLLB-3.3B output")
        ax.bar(x, dd["chrf_vs_human_ref"], width=w * 0.9, color="white", edgecolor="k", lw=0.8, alpha=0.0)
        ax.scatter(x, dd["chrf_vs_human_ref"], marker="_", s=120, color="k", zorder=3, label="vs human reference" if i == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels(["very easy", "easy", "moderate"]); ax.set_title(ds, fontsize=9)
axs[0].set_ylabel("chrF++ (bar: against NLLB-3.3B output; tick: against human reference)", fontsize=7); axs[0].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(f"{FIG}/fig6_agreement_with_nllb3b3_vs_human_reference.png"); plt.close(fig)
# F9 long docs
L = pd.read_csv(OUT + "/12c_longdoc_chrf_by_score_bin_sentence_mode.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
for a_, st in zip(ax, ("human", "sci")):
    d = L[L.set == st]
    for m in ("rl", "nllb3b3", "sft"):
        dd = d[d.model == m]; xs = range(len(dd))
        a_.errorbar(dd.score_bin, dd.mean_chrf, yerr=[dd.mean_chrf - dd.ci_lo, dd.ci_hi - dd.mean_chrf], marker="o", ms=3, color=COL[m], label=m, lw=1.2, capsize=2)
    a_.set_title({"human": "SmolDoc documents (human references)", "sci": "AfriScience-MT dev+test documents"}[st], fontsize=9); a_.set_xlabel("difficulty score of the document")
ax[0].set_ylabel("document chrF++ (sentence-by-sentence mode)"); ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/fig9_long_documents.png"); plt.close(fig)
print("ok")
