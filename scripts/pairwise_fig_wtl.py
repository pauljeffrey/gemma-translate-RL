# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figure PW1 (E1): win / tie / loss of the RL-final model over the SFT-final model, sentence level, per metric, benchmark and training group.

    python gemma_mt/scripts/pairwise_fig_wtl.py          ->  paper/figures/figpw1_wtl.{png,pdf}

Panel A: macro win / tie / loss over the QE-covered languages (all benchmarks pooled per language, `no_pair` subset) for the reference-based metrics (the
independent evidence) and the two QE metrics (the RL reward family). Panel B: share of sentences RL is preferred in among the non-tied sentences, per benchmark.
Panel C: the same per training group. Whiskers: 95% paired-bootstrap intervals (macro: languages resampled independently; micro: strata = language)."""
import numpy as np
from pairwise_fig_common import *  # noqa: F401,F403

plt = setup_matplotlib()
SUB = "no_pair"


def main():
    m = read("macro.csv")
    m = m[(m["subset"] == SUB)]
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.62, wspace=0.42)
    # ---- A
    ax = fig.add_subplot(gs[0, :])
    mets = REF_METRICS + ["ens_ref"] + QE_METRICS
    rows = []
    for k in mets:
        r = m[(m["b"] == "ALL") & (m["scope"] == "all_langs") & (m["metric"] == k)]
        if len(r):
            rows.append((k, r.iloc[0]))
    ypos = np.arange(len(rows))[::-1]
    n_ref = len(REF_METRICS) + 1
    for y, (k, r) in zip(ypos, rows):
        w, t, l = r["macro_win"], r["macro_tie"], r["macro_loss"]
        ax.barh(y, l, left=0, color=LOSS_C, height=0.62)
        ax.barh(y, t, left=l, color=TIE_C, height=0.62)
        ax.barh(y, w, left=l + t, color=WIN_C, height=0.62)
        lo, hi = r["macro_win_lo"], r["macro_win_hi"]
        ax.plot([1 - hi, 1 - lo], [y, y], color="k", lw=1.0, solid_capstyle="butt")      # CI of the win share = CI of the left edge of the green bar
        ax.text(1.01, y, f"{100 * w:.0f} / {100 * t:.0f} / {100 * l:.0f}", va="center", fontsize=6.5)
        if l > 0.06:
            ax.text(l / 2, y, f"{100 * l:.0f}", ha="center", va="center", fontsize=6, color="white")
        ax.text(l + t + w / 2, y, f"{100 * w:.0f}", ha="center", va="center", fontsize=6.5, color="white")
    ax.set_yticks(ypos)
    ax.set_yticklabels([METRIC_LABEL[k].replace("\n", " ") for k, _ in rows], fontsize=7)
    ax.axhline(ypos[n_ref - 1] - 0.5, color="k", lw=0.6, ls=":")
    ax.text(1.01, ypos[0] + 0.62, "win / tie / loss %", fontsize=6.3, color="#333")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of sentences (loss | tie | win of RL-final over SFT-final)")
    ax.set_title("A  Pairwise outcome per metric (macro over languages, no_pair)", loc="left", fontweight="bold")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    # ---- B: by benchmark
    ax = fig.add_subplot(gs[1, :2])
    benches = [b for b in ["flores_plus", "ntrex", "mafand", "smol_sent", "glot_story", "salt"] if b in set(m["b"])]
    lab = {"flores_plus": "FLORES+", "ntrex": "NTREX", "mafand": "MAFAND", "smol_sent": "SmolSent", "glot_story": "GlotStory", "salt": "SALT"}
    cols = {"chrf_sub": OI["blue"], "afri_ref": OI["sky"], "ssa_ref": OI["green"], "afri_qe": OI["orange"], "ssa_qe": OI["verm"]}
    w = 0.16
    for j, k in enumerate(["chrf_sub", "afri_ref", "ssa_ref", "afri_qe", "ssa_qe"]):
        xs, ys, lo, hi = [], [], [], []
        for i, b in enumerate(benches):
            r = m[(m["b"] == b) & (m["scope"] == "all_langs") & (m["metric"] == k)]
            if not len(r):
                continue
            r = r.iloc[0]
            xs.append(i + (j - 2) * w)
            ys.append(r["micro_pref"])
            lo.append(r["micro_pref"] - r["micro_pref_lo"])
            hi.append(r["micro_pref_hi"] - r["micro_pref"])
        ax.bar(xs, [y - 0.5 for y in ys], bottom=0.5, width=w * 0.92, color=cols[k], label=METRIC_LABEL[k].replace("\n", " "))
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="k", elinewidth=0.6, capsize=1.2)
    ax.axhline(0.5, color="k", lw=0.7)
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels([lab[b] for b in benches])
    ax.set_ylabel("RL preferred (non-tied)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("B  By benchmark (micro, sentences pooled)", loc="left", fontweight="bold")
    ax.legend(ncol=5, fontsize=5.4, loc="upper center", bbox_to_anchor=(0.5, -0.14), columnspacing=0.8, handlelength=1.0)
    # ---- C: by group
    ax = fig.add_subplot(gs[1, 2])
    groups = ["qe_validated", "blend", "consensus_only", "sft_only"]
    for j, k in enumerate(["chrf_sub", "afri_ref", "afri_qe"]):
        xs, ys, lo, hi = [], [], [], []
        for i, g in enumerate(groups):
            r = m[(m["b"] == "ALL") & (m["scope"] == "group:" + g) & (m["metric"] == ("chrf" if k == "chrf_sub" else k))]   # chrF++ on all rows (consensus-only / SFT-only have no COMET scores)
            if not len(r):
                continue
            r = r.iloc[0]
            xs.append(i + (j - 1) * 0.26)
            ys.append(r["micro_pref"])
            lo.append(r["micro_pref"] - r["micro_pref_lo"])
            hi.append(r["micro_pref_hi"] - r["micro_pref"])
        ax.bar(xs, [y - 0.5 for y in ys], bottom=0.5, width=0.24, color=cols[k])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="k", elinewidth=0.6, capsize=1.2)
    ax.axhline(0.5, color="k", lw=0.7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(["QE-\nvalid.", "QE-\nblend", "consen-\nsus", "SFT-\nonly"], fontsize=6.2)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("C  By training group", loc="left", fontweight="bold")
    ax.text(0.5, 0.97, "chrF++ / AfriC-ref / AfriC-QE", transform=ax.transAxes, ha="center", va="top", fontsize=5.5)
    save(fig, "figpw1_wtl")


if __name__ == "__main__":
    main()
