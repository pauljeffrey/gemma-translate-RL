# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figure PW3 (E1): (A) agreement of the sentence-level winner between metrics, (B) what the reference-based metrics say about the sentences on which RL
wins by both QE metrics, (C) the long-input check (R6 documents, sentence-by-sentence decoding, per length bin).

    python gemma_mt/scripts/pairwise_fig_agreement.py    ->  paper/figures/figpw3_agreement.{png,pdf}

Panel A: among the sentences where both metrics call a winner, the share on which they name the same one (`no_pair`). Panel B: of the sentences where RL wins on
BOTH AfriCOMET-QE and SSA-COMET-QE: share where the majority of the three reference-based metrics also prefers RL / is neutral / prefers SFT (metric gaming
would show as a large 'prefers SFT' share). Panel C: win / tie / loss of RL over SFT per length bin of the long-document set (segment-level chrF++, from
reports/data/pairwise_benchmark_comparison/longdoc_segments/); skipped when the CSV is absent."""
import os

import numpy as np
from pairwise_fig_common import *  # noqa: F401,F403

plt = setup_matplotlib()
SUB = "no_pair"


def main():
    ag = read("agreement.csv")
    ag = ag[(ag["subset"] == SUB) & (ag["scope"] == "all_langs") & (~ag["metric_1"].astype(str).str.startswith("ens"))]
    mets = ["chrf_sub", "afri_ref", "ssa_ref", "afri_qe", "ssa_qe"]
    A = np.full((5, 5), np.nan)
    for _, r in ag.iterrows():
        i, j = mets.index(r["metric_1"]), mets.index(r["metric_2"])
        A[i, j] = A[j, i] = r["agree_nontie"]
    have_ld = os.path.isfile(os.path.join(DATA, "longdoc_segments", "macro.csv"))
    fig = plt.figure(figsize=(7.0, 2.7))
    gs = fig.add_gridspec(1, 3 if have_ld else 2, width_ratios=[1.0, 0.75, 1.15] if have_ld else [1, 0.8], wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(A, cmap="Blues", vmin=0.5, vmax=1.0)
    for i in range(5):
        for j in range(5):
            if i != j and not np.isnan(A[i, j]):
                ax.text(j, i, f"{100 * A[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="white" if A[i, j] > 0.78 else "black")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    short = {"chrf_sub": "chrF++", "afri_ref": "AfriC-ref", "ssa_ref": "SSA-ref", "afri_qe": "AfriC-QE", "ssa_qe": "SSA-QE"}
    ax.set_xticklabels([short[k] for k in mets], rotation=40, ha="right", fontsize=6)
    ax.set_yticklabels([short[k] for k in mets], fontsize=6)
    ax.grid(False)
    ax.set_title("A  Winner agreement (%)", loc="left", fontweight="bold", fontsize=7.5)
    # B
    ax = fig.add_subplot(gs[0, 1])
    r = ag[ag["metric_1"] == "ens_qe(B wins both QE)"]
    r = read("agreement.csv")
    r = r[(r["subset"] == SUB) & (r["metric_1"].astype(str).str.startswith("ens_qe"))]
    if len(r):
        r = r.iloc[0]
        conf, neut, contra = r["agree_nontie"], r["agree_3class"], r["kappa_3class"]
        ax.bar([0], [conf], color=WIN_C, label="ref-based majority prefers RL")
        ax.bar([0], [neut], bottom=[conf], color=TIE_C, label="neutral")
        ax.bar([0], [contra], bottom=[conf + neut], color=LOSS_C, label="prefers SFT")
        for y0, v in ((0, conf), (conf, neut), (conf + neut, contra)):
            if v > 0.05:
                ax.text(0, y0 + v / 2, f"{100 * v:.0f}%", ha="center", va="center", fontsize=6.5, color="white" if v == conf or v == contra else "black")
    ax.set_xlim(-0.8, 0.8); ax.set_xticks([]); ax.set_ylim(0, 1)
    ax.set_ylabel("share of sentences")
    ax.set_title("B  RL wins both QE", loc="left", fontweight="bold", fontsize=7.5)
    ax.legend(fontsize=5.4, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=1)
    ax.grid(False)
    # C
    if have_ld:
        ax = fig.add_subplot(gs[0, 2])
        m = read("macro.csv", "longdoc_segments")
        m = m[(m["subset"] == "all") & (m["scope"] == "all_langs") & (m["metric"] == "chrf")]
        bins = [b for b in ["core_b64", "core_b256", "core_b512", "core_b1024", "human_sm", "human_st512", "human_st1024"] if b in set(m["b"])]
        for i, b in enumerate(bins):
            r = m[m["b"] == b].iloc[0]
            ax.bar(i, r["micro_loss"], color=LOSS_C, width=0.7)
            ax.bar(i, r["micro_tie"], bottom=r["micro_loss"], color=TIE_C, width=0.7)
            ax.bar(i, r["micro_win"], bottom=r["micro_loss"] + r["micro_tie"], color=WIN_C, width=0.7)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([b.replace("core_", "").replace("human_", "h-") for b in bins], rotation=40, ha="right", fontsize=6)
        ax.set_ylim(0, 1); ax.set_ylabel("segment share")
        ax.set_title("C  Long input, chrF++ (segments)", loc="left", fontweight="bold", fontsize=7.5)
        ax.grid(False)
    save(fig, "figpw3_agreement")


if __name__ == "__main__":
    main()
