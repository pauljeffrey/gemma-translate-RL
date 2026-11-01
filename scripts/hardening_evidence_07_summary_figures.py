#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""H1 step 7: summary tables (verdict matrix, decomposition of RL-final -> final2 into the science stage and the hardened RL run, per-language table) and figures.
Reads the CSVs of steps 3-6 from $HE_OUT, writes summary_*.csv and fig*.png there."""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardening_evidence_common as C  # noqa: E402

warnings.filterwarnings("ignore")
O = C.OUT
# Okabe-Ito (colour-blind safe); one hue per entity, same hue in every figure
COL = {"sft": "#999999", "rl": "#0072B2", "a2w": "#E69F00", "live250": "#009E73", "live": "#009E73", "final2": "#D55E00", "human reference": "#000000"}
NAME = {"sft": "SFT-final", "rl": "RL-final", "a2w": "A2W (start)", "live250": "step 250", "live": "step 250", "final2": "final2 (401)", "human reference": "human reference"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight"})
KEY = ["unfinished", "degenerate", "lid_wrong", "copy_flag", "num_loss", "name_loss", "func_flag", "pad_flag", "rep_flag", "rep4", "pen_total"]
KEYLAB = {"unfinished": "unfinished", "degenerate": "degenerate (v1 rule)", "lid_wrong": "wrong language (GlotLID)", "copy_flag": "substring copy flag", "num_loss": "numbers lost",
          "name_loss": "names lost", "func_flag": "stop-word profile flag", "pad_flag": "padding flag", "rep_flag": "repetition flag", "rep4": "repeated 4-grams (share)",
          "pen_total": "anchors penalty (reward units)"}


def sym(r):
    return {"final2 better": "better", "final2 worse": "worse", "equal": "equal"}.get(r, r)


def summary_matrix():
    d = pd.read_csv(os.path.join(O, "paired_rl_vs_final2.csv"))
    rows = []
    for scheme, groups in (("all_languages", ["all"]), ("tier", ["strong", "mid", "weak", "consensus-only"]), ("content_mode", ["qe", "blend", "consensus"])):
        for g in groups:
            for bench in ["ALL", "flores_plus", "ntrex", "mafand", "smol_sent", "salt", "glot_story", "afs_dev", "afs_test"]:
                x = d[(d.scheme == scheme) & (d.group == g) & (d.bench == bench)]
                for r in x.itertuples():
                    rows.append(dict(scheme=scheme, group=g, bench=bench, indicator=r.indicator, n_lang=r.n_lang, n_rows=r.n_rows, rl_final=r.mean_a, final2=r.mean_b, diff_final2_minus_rl=r.diff_b_minus_a,
                                     ci_lo=r.ci_lo, ci_hi=r.ci_hi, lang_ci_lo=r.lang_ci_lo, lang_ci_hi=r.lang_ci_hi, verdict=sym(r.verdict), negligible=r.negligible))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(O, "summary_final2_vs_rl_all.csv"), index=False, float_format="%.5g")
    return out


def decomposition():
    """FLORES+ and AfriScience sentences: total change RL-final -> final2 = (RL-final -> A2W: science SFT stage + interpolation) + (A2W -> final2: the hardened RL run)."""
    rows = []
    for bench in ("flores_plus", "ALL_afs"):
        for scheme, groups in (("all_languages", ["all"]), ("tier", ["strong", "mid", "weak", "consensus-only"])):
            for g in groups:
                rec = {}
                for key, f in (("rl_to_a2w", "paired_rl_vs_a2w.csv"), ("a2w_to_final2", "paired_a2w_vs_final2.csv"), ("rl_to_final2", "paired_rl_vs_final2.csv"), ("a2w_to_live250", "paired_a2w_vs_live250.csv"),
                               ("live250_to_final2", "paired_live250_vs_final2.csv")):
                    d = pd.read_csv(os.path.join(O, f))
                    d = d[(d.scheme == scheme) & (d.group == g)]
                    if bench == "ALL_afs":
                        continue
                    x = d[d.bench == bench].set_index("indicator")
                    for ind in x.index:
                        rec.setdefault(ind, {})[key] = (x.loc[ind, "diff_b_minus_a"], x.loc[ind, "ci_lo"], x.loc[ind, "ci_hi"])
                for ind, v in rec.items():
                    r = dict(bench=bench, scheme=scheme, group=g, indicator=ind)
                    for k, (p, lo, hi) in v.items():
                        r[k], r[k + "_lo"], r[k + "_hi"] = p, lo, hi
                    rows.append(r)
    out = pd.DataFrame(rows)
    # AfriScience (RL-final, A2W, final2 all available): sentences, dev and test pooled
    afs = []
    for key, f in (("rl_to_a2w", "paired_rl_vs_a2w.csv"), ("a2w_to_final2", "paired_a2w_vs_final2.csv"), ("rl_to_final2", "paired_rl_vs_final2.csv")):
        d = pd.read_csv(os.path.join(O, f))
        d = d[(d.scheme == "all_languages") & d.bench.isin(["afs_dev", "afs_test"])]
        d = d.groupby("indicator")[["diff_b_minus_a", "ci_lo", "ci_hi"]].mean()
        for ind, r in d.iterrows():
            afs.append(dict(bench="afs_dev+test (mean of the two)", scheme="all_languages", group="all", indicator=ind, **{key: r.diff_b_minus_a, key + "_lo": r.ci_lo, key + "_hi": r.ci_hi}))
    afs = pd.DataFrame(afs).groupby(["bench", "scheme", "group", "indicator"], as_index=False).first()
    out = pd.concat([out[out.bench != "ALL_afs"], afs], ignore_index=True)
    out.to_csv(os.path.join(O, "summary_decomposition_rl_to_final2.csv"), index=False, float_format="%.5g")
    return out


def per_language(B=1000):
    ind = pd.read_parquet(os.path.join(C.WORK, "indicators.parquet"))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib

    st = importlib.import_module("hardening_evidence_03_sentence_stats")
    ind = st.prep(ind)
    groups = pd.read_csv(os.path.join(O, "language_groups.csv")).set_index("iso")
    d = ind[~ind.pair_seen & ind.system.isin(["rl", "final2"])]
    a = d[d.system == "rl"].set_index(["bench", "iso", "id"])
    b = d[d.system == "final2"].set_index(["bench", "iso", "id"])
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    cols = ["unfinished", "degenerate", "copy_flag", "func_flag", "pad_flag", "rep_flag", "pen_total"]
    iso = idx.get_level_values(1).values
    rng = np.random.default_rng(9)
    rows = []
    for l in np.unique(iso):
        m = iso == l
        for c in cols:
            x = b[c].values[m] - a[c].values[m]
            n = len(x)
            bm = np.array([x[rng.integers(0, n, n)].mean() for _ in range(B)])
            lo, hi = np.percentile(bm, [2.5, 97.5])
            rows.append(dict(iso=l, tier=groups.loc[l, "tier"], content_mode=groups.loc[l, "content_mode"], indicator=c, n=n, rl_final=a[c].values[m].mean(), final2=b[c].values[m].mean(), diff=x.mean(), lo=lo, hi=hi,
                             verdict="better" if hi < 0 else ("worse" if lo > 0 else "equal")))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(O, "summary_per_language_rl_vs_final2.csv"), index=False, float_format="%.5g")
    return out


# ---------------------------------------------------------------------------------------------------------------- figures
def fig_forest(sm):
    fig, axes = plt.subplots(1, 5, figsize=(13, 4.6), sharey=True)
    groups = [("all_languages", "all", "all languages"), ("tier", "strong", "strong"), ("tier", "mid", "mid"), ("tier", "weak", "weak"), ("tier", "consensus-only", "consensus-only")]
    for ax, (scheme, g, lab) in zip(axes, groups):
        x = sm[(sm.scheme == scheme) & (sm.group == g) & (sm.bench == "ALL")].set_index("indicator")
        for k, ind in enumerate(KEY):
            if ind not in x.index:
                continue
            r = x.loc[ind]
            sc = 100.0
            c = COL["final2"] if r.verdict == "better" else ("#000000" if r.verdict == "worse" else "#888888")
            ax.errorbar(r.diff_final2_minus_rl * sc, k, xerr=[[(r.diff_final2_minus_rl - r.ci_lo) * sc], [(r.ci_hi - r.diff_final2_minus_rl) * sc]], fmt="o", color=c, ecolor=c, ms=4, capsize=2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{lab}", fontsize=9)
        ax.set_yticks(range(len(KEY)))
        ax.set_yticklabels([KEYLAB[k] for k in KEY])
        ax.invert_yaxis()
        ax.set_xlabel("final2 minus RL-final, pp\n(penalty: 0.01 reward units)", fontsize=8)
    fig.suptitle("Failure / hack indicators, final2 minus RL-final, all benchmarks pooled (no_pair rows; sentences resampled within language, macro over languages)\n"
                 "orange = final2 clearly lower, black = clearly higher, grey = interval includes 0", fontsize=9)
    fig.savefig(os.path.join(O, "fig1_final2_minus_rl_forest.png"))
    plt.close(fig)


def fig_decomp(dec):
    ind = ["degenerate", "copy_flag", "name_loss", "pad_flag", "rep_flag", "pen_total"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    for ax, g in zip(axes, ["strong", "mid", "weak", "consensus-only"]):
        x = dec[(dec.bench == "flores_plus") & (dec.scheme == "tier") & (dec.group == g)].set_index("indicator")
        for k, i in enumerate(ind):
            if i not in x.index:
                continue
            for j, (key, col) in enumerate((("rl_to_a2w", COL["a2w"]), ("a2w_to_final2", COL["final2"]))):
                v = x.loc[i, key]
                lo, hi = x.loc[i, key + "_lo"], x.loc[i, key + "_hi"]
                ax.barh(k + (j - 0.5) * 0.35, v * 100, height=0.3, color=col, label=("science stage (RL-final to A2W, no RL)" if key == "rl_to_a2w" else "hardened RL run (A2W to final2)") if k == 0 else None)
                ax.plot([lo * 100, hi * 100], [k + (j - 0.5) * 0.35] * 2, color="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{g} (FLORES+)", fontsize=9)
        ax.set_yticks(range(len(ind)))
        ax.set_yticklabels([KEYLAB[i] for i in ind])
        ax.invert_yaxis()
        ax.set_xlabel("change, pp (penalty: 0.01 reward units); consensus-only = 2 languages in FLORES+", fontsize=7)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Where the change from RL-final to final2 comes from (FLORES+, no_pair; bars = point estimate, lines = 95% interval)", fontsize=9)
    fig.savefig(os.path.join(O, "fig2_decomposition_flores.png"))
    plt.close(fig)


def fig_lang(pl):
    x = pl[pl.indicator == "pen_total"]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    palette = {"strong": "#0072B2", "mid": "#56B4E9", "weak": "#E69F00", "consensus-only": "#D55E00"}
    for t, g in x.groupby("tier"):
        ax.scatter(g.rl_final, g.final2, s=22, color=palette.get(t, "#999"), label=f"{t} ({len(g)})")
    m = max(x.rl_final.max(), x.final2.max()) * 1.05
    ax.plot([0, m], [0, m], color="k", lw=0.8)
    for r in x.sort_values("rl_final", ascending=False).head(5).itertuples():
        ax.annotate(r.iso, (r.rl_final, r.final2), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("RL-final: penalty the final reward would subtract\n(reward units, mean per language, all benchmarks)")
    ax.set_ylabel("final2")
    ax.legend(fontsize=7)
    ax.set_title("Per language; below the diagonal = final2 charged less", fontsize=9)
    fig.savefig(os.path.join(O, "fig3_language_penalty_scatter.png"))
    plt.close(fig)


def fig_training():
    b = pd.read_csv(os.path.join(O, "training_blocks_final2.csv"))
    x = (b.step_lo + b.step_hi) / 2
    panels = [("hard failures", ["reward/fail/degenerate", "reward/fail/wrong_language", "reward/fail/stuffed_output"], ["degenerate", "wrong language", "stuffed output"]),
              ("detector rates (share of samples penalised)", ["reward/anchors/copy/penalised_rate", "reward/anchors/pad/penalised_rate", "reward/anchors/repeat/penalised_rate", "reward/anchors/func/penalised_rate"],
               ["copy", "padding", "repetition", "stop-word"]),
              ("entity / boilerplate (share penalised)", ["reward/anchors/entity2/penalised_rate", "reward/anchors/boiler/penalised_rate"], ["graded entity term (share penalised)", "boilerplate"]),
              ("QE ensemble", ["reward/qe_ens/min_share", "reward/qe_ens/min_share_afri_high", "reward/qe_ens/min_share_ssa_high"], ["min branch", "AfriCOMET higher by > 1", "SSA-COMET higher by > 1"]),
              ("shaped judge means", ["reward/qe_ens/r1_mean", "reward/qe_ens/r2_mean"], ["r1 AfriCOMET-QE", "r2 SSA-COMET-MTL"]),
              ("long-document reward v2", ["reward/longdoc/rep_severity", "reward/longdoc/rep_penalised_rate", "reward/longdoc/cover_failed"], ["repetition severity", "share repetition-penalised", "failed share of source"])]
    pal = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, (t, cols, labs) in zip(axes.ravel(), panels):
        for k, (c, l) in enumerate(zip(cols, labs)):
            if c in b:
                ax.plot(x, b[c], marker="o", ms=3, color=pal[k], label=l)
        ax.axvline(150.5, color="k", lw=0.7, ls=":")
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("step; dotted line: restart with anchors v3", fontsize=8)
        ax.legend(fontsize=7)
    fig.suptitle("Training log of the final run (25-step means): detector rates stay flat or rise after the restart at step 151", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(O, "fig4_training_logs.png"))
    plt.close(fig)


def fig_longdoc():
    b = pd.read_csv(os.path.join(O, "longdoc_lc1_by_system.csv"))
    b = b[b["mode"] == "whole"]
    groups = ["short (<= ~512 tokens)", "~1,024 tokens", ">= ~2,048 tokens", "AfriScience papers"]
    inds = [("unfinished", "unfinished"), ("rep_severe", "repetition severity >= 0.5"), ("cover_failed", "failed share of the source")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    systems = ["sft", "rl", "a2w", "live", "final2"]
    for ax, (c, lab) in zip(axes, inds):
        for k, s in enumerate(systems):
            v = [b[(b.system == s) & (b.length_group == g)][c].mean() for g in groups]
            ax.bar(np.arange(len(groups)) + (k - 2) * 0.16, v, width=0.15, color=COL[s], label=NAME[s])
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.replace(" tokens", "").replace("AfriScience papers", "papers") for g in groups], fontsize=7)
        ax.set_title(lab, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.suptitle("Whole-document translation (14 languages, 290 documents, 2,280 items per system)", fontsize=8)
    fig.savefig(os.path.join(O, "fig5_longdoc_whole.png"))
    plt.close(fig)


def fig_judges():
    g = pd.read_csv(os.path.join(O, "judges_B_qe_gap_final2_vs_rl.csv"))
    g = g[g.tier == "all"]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for k, r in enumerate(g.itertuples()):
        ax.errorbar(r.final2_minus_rl, k, xerr=[[r.final2_minus_rl - r.lo], [r.hi - r.final2_minus_rl]], fmt="o", color=COL["final2"], capsize=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([f"{q} ({n} lang.)" for q, n in zip(g.quantity, g.n_lang)], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("final2 minus RL-final (metric units, mean over languages, 95% interval over languages)")
    ax.set_title("Judge gaps on the E1 sample (same rows)", fontsize=9)
    fig.savefig(os.path.join(O, "fig6_judge_gaps.png"))
    plt.close(fig)


def main():
    sm = summary_matrix()
    dec = decomposition()
    pl = per_language()
    fig_forest(sm)
    fig_decomp(dec)
    fig_lang(pl)
    fig_training()
    fig_longdoc()
    fig_judges()
    print("done")


if __name__ == "__main__":
    main()
