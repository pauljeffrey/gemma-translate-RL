# SPDX-License-Identifier: AGPL-3.0-or-later
"""Figure PW2 (E1): per-language pairwise preference for RL-final over SFT-final (all benchmarks pooled per language, `no_pair` subset).

    python gemma_mt/scripts/pairwise_fig_langs.py        ->  paper/figures/figpw2_languages.{png,pdf}

One row per AfriCOMET-covered language, grouped by training group; columns = metrics. Cell = share of the non-tied sentences in which RL is better (0.5 = no
preference; blue = RL better, red = SFT better). A star marks a Benjamini-Hochberg-adjusted sign test q < 0.05 (across languages, per metric). A black dot in
the left strip marks languages where RL wins on a QE metric with a CI above zero but not on every reference-based metric (flags.csv)."""
import numpy as np
from pairwise_fig_common import *  # noqa: F401,F403

plt = setup_matplotlib()
SUB = "no_pair"


def main():
    u = read("unit_stats.csv")
    u = u[(u["subset"] == SUB) & (u["b"] == "ALL")]
    fl = read("flags.csv")
    fl = fl[(fl["subset"] == SUB) & (fl["b"] == "ALL")]
    flagged = set(fl[fl["flag"].notna()]["iso"])
    mets = ["chrf_sub", "afri_ref", "ssa_ref", "afri_qe", "ssa_qe"]
    P = u.pivot(index="iso", columns="metric", values="pref")
    Q = u.pivot(index="iso", columns="metric", values="q_sign")
    N = u.pivot(index="iso", columns="metric", values="n")
    G = u.drop_duplicates("iso").set_index("iso")["group"]
    order_g = list(GROUP_COLOR)
    isos = [i for i in P.index if not np.isnan(P.loc[i, "afri_ref"])]
    isos.sort(key=lambda i: (order_g.index(G[i]), -P.loc[i, "chrf_sub"]))
    M = np.array([[P.loc[i, k] if k in P else np.nan for k in mets] for i in isos])
    fig, ax = plt.subplots(figsize=(3.9, 0.155 * len(isos) + 1.1))
    im = ax.imshow(M, aspect="auto", cmap="RdBu", vmin=0.0, vmax=1.0)
    for r, i in enumerate(isos):
        for c, k in enumerate(mets):
            v = M[r, c]
            if np.isnan(v):
                continue
            q = Q.loc[i, k]
            ax.text(c, r, f"{100 * v:.0f}" + ("*" if q < 0.05 else ""), ha="center", va="center", fontsize=5.2,
                    color="white" if abs(v - 0.5) > 0.3 else "black")
    ax.set_xticks(range(len(mets)))
    ax.set_xticklabels([METRIC_LABEL[k] for k in mets], fontsize=5.8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(isos)))
    ax.set_yticklabels([("● " if i in flagged else "") + i for i in isos], fontsize=5.6)
    for lab, i in zip(ax.get_yticklabels(), isos):
        lab.set_color(GROUP_COLOR[G[i]])
    ax.axvline(2.5, color="k", lw=0.8)
    prev = None
    for r, i in enumerate(isos):
        if G[i] != prev and r:
            ax.axhline(r - 0.5, color="k", lw=0.5)
        prev = G[i]
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("RL preferred among non-tied sentences", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.text(0, len(isos) + 0.2, "reference-based", fontsize=6, va="top")
    ax.text(3, len(isos) + 0.2, "QE (RL reward family)", fontsize=6, va="top")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="s", ls="", color=GROUP_COLOR[g], label=GROUP_LABEL[g]) for g in order_g if g in set(G[isos])],
              loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=5.6)
    save(fig, "figpw2_languages")


if __name__ == "__main__":
    main()
