#!/usr/bin/env python3
"""Compact figures of the short paper (paper/short/figures/*.pdf, 6.7 in = full text width, legible at the printed size).

    <python with matplotlib> paper/scripts/make_figures_short.py

fig1s_pipeline: one-row schematic of the training chain and of the reward (text only, numbers as in the long paper's Figure 1 / Sections 4.1-4.3).
fig2s_gain:     per-language chrF++ SFT-final -> RL-final on FLORES+ devtest (no_pair rows), data: reports/data/final_model_per_language_wide.csv
                (same file and filter as make_figures.fig2), languages sorted by the SFT score; NLLB-200-3.3B as published (published_baselines.csv).
The other two figures of the short paper (fig11_ablation, fig14_longctx_chrf) are copied unchanged from paper/figures/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
plt = setup_matplotlib()
import numpy as np  # noqa
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa
from matplotlib.lines import Line2D  # noqa

OUT = os.path.join(PAPER, "short", "figures")
os.makedirs(OUT, exist_ok=True)
W = 6.7


def save_short(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    print("figure", name)


def box(ax, x, y, w, h, text, fc, ec="#33506B", fs=5.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=0.8, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.2)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.8, color="#33506B"))


def fig1s():
    fig, ax = plt.subplots(figsize=(W, 2.75))
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    ax.text(0, 0.995, "A  Training chain", fontsize=7, fontweight="bold", va="top")
    A = [("Gemma 3 270M-it\n18 layers, d=640,\n262k vocabulary", "#F2F2F2"),
         ("SFT, 77 languages\n3.62 B tokens\n(step 27,638)\n= SFT-final", "#DCEBF7"),
         ("GRPO, length curriculum\nb64 to b2048, then mixed\n(phase 2c, step 1,433)\n= RL-final", "#DDF0E8"),
         ("Science SFT stage\n+ weight interpolation\n(alpha 0.7)", "#DCEBF7"),
         ("Final RL run, 401 steps\nQE ensemble, anchors,\nlong-doc. reward", "#FFF3D0")]
    n = len(A); w = 0.172; gap = (1 - n * w) / (n - 1)
    for i, (t, fc) in enumerate(A):
        x = i * (w + gap)
        box(ax, x, 0.64, w, 0.27, t, fc, fs=4.8)
        if i:
            arrow(ax, x - gap + 0.002, 0.775, x - 0.002, 0.775)
    ax.text(0, 0.575, "B  Reward of one completion (computed per sentence chunk, then source-length-weighted mean)", fontsize=7, fontweight="bold", va="top")
    B = [("1 Sanitise; hard failure\n(-2): no end, wrong tag\nor language, copy,\nloop, length ratio", "#F8DDD3"),
         ("2 Chunk: sentence\nsplit, language-aware\nalignment; omitted or\nextra sentence = -2", "#DCEBF7"),
         ("3 Content signal: QE\n(ensemble: mean, min\nif judges differ > 1)\nor group consensus\nchrF++ (no metric)", "#DDF0E8"),
         ("4 Shaping per language\non human anchors: 0 =\nP90 mismatched, +1 =\nhuman median, +2 =\nhuman P90", "#DDF0E8"),
         ("5 Penalties, no bonus:\nlength, numbers/names,\ndiacritics; anchors:\ncopy, padding, loops", "#FFF3D0"),
         ("6 GRPO: group of 16,\nstandardised advantage\nx language weight,\nKL guard", "#EEE0F0")]
    n = len(B); w = 0.152; gap = (1 - n * w) / (n - 1)
    for i, (t, fc) in enumerate(B):
        x = i * (w + gap)
        box(ax, x, 0.20, w, 0.30, t, fc, fs=4.6)
        if i:
            arrow(ax, x - gap + 0.002, 0.35, x - 0.002, 0.35)
    ax.text(0, 0.12, "Documents longer than about 512 source tokens: decode sentence by sentence (greedy) and re-decode the unfinished sentences.", fontsize=5.6, style="italic", va="top")
    save_short(fig, "fig1s_pipeline")
    plt.close(fig)


def published_nllb():
    out = {}
    for r in read_csv(os.path.join(RDATA, "published_baselines.csv")):
        if r["model"] == "NLLB-200-3.3B" and r["direction"] == "en->xx" and r["benchmark"].startswith("FLORES-200 "):
            v = fnum(r["chrF++"])
            if v is not None:
                out[r["iso"]] = v
    return out


def fig2s():
    rows = [r for r in read_csv(os.path.join(RDATA, "final_model_per_language_wide.csv"))
            if r["benchmark"] == "flores_plus" and r["subset"] == "no_pair"]
    rows.sort(key=lambda r: float(r["sft_chrf"]))
    nl = published_nllb()
    fig, ax = plt.subplots(figsize=(W, 2.45))
    for i, r in enumerate(rows):
        c = GROUP_COLOR.get(r["group"], OI["grey"])
        s, e = float(r["sft_chrf"]), float(r["rl_chrf"])
        ax.plot([i, i], [s, e], color=c, lw=1.4, alpha=0.75, zorder=1)
        ax.scatter([i], [s], s=9, facecolors="white", edgecolors=c, lw=0.8, zorder=2)
        ax.scatter([i], [e], s=10, color=c, zorder=3)
        if r["iso"] in nl:
            ax.scatter([i], [nl[r["iso"]]], marker="_", s=26, color="black", lw=1.0, zorder=4)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["iso"] for r in rows], rotation=90, fontsize=5.2)
    ax.set_xlim(-0.8, len(rows) - 0.2)
    ax.set_ylabel("FLORES+ chrF++ (no_pair rows)", fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6)
    ax.grid(axis="x", alpha=0)
    h = [Line2D([], [], marker="o", color=GROUP_COLOR[g], ls="", markersize=3.5, label=GROUP_LABEL[g]) for g in GROUP_COLOR if any(r["group"] == g for r in rows)]
    h += [Line2D([], [], marker="o", mfc="white", mec="black", ls="", markersize=3.5, label="SFT-final"),
          Line2D([], [], marker="o", color="black", ls="", markersize=3.5, label="RL-final"),
          Line2D([], [], marker="_", color="black", ls="", markersize=6, label="NLLB-200 3.3B (published)")]
    ax.legend(handles=h, loc="upper left", fontsize=5.6, ncol=2)
    save_short(fig, "fig2s_gain")
    plt.close(fig)


if __name__ == "__main__":
    fig1s(); fig2s()
