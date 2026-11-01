#!/usr/bin/env python3
"""All figures of the paper.  Usage:  python paper/scripts/make_figures.py [fig1 fig2 ...]   (default: all)

Every number is read from a file of this repository (see the CLAIMS_LEDGER); nothing is typed in except where a source line is cited
in a comment next to the constant.  fig3 (phase diagnostics) and fig9 (final2) read paper/data/final2_metrics.jsonl and are meant to be
re-run when the final2 run has finished (scripts/refresh_final2.sh copies the newest metrics.jsonl from the GPU box, read-only).
"""
import json
import math
import os
import sys

from common import *  # noqa: F401,F403

plt = setup_matplotlib()
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

COLUMN_W, PAGE_W = 3.4, 7.0


# =================================================================================================== fig1: pipeline
def box(ax, x, y, w, h, text, fc="#EEF3F8", ec="#33506B", fs=6.8, bold=False, tc="black", ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=0.9, ls=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1, color="#33506B", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig1():
    fig, ax = plt.subplots(figsize=(PAGE_W, 6.2))
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)
    ax.text(0.0, 0.995, "A  Training chronology (each arrow: the model that starts the next phase)", fontsize=8, fontweight="bold", va="top")
    w, h = 0.232, 0.12
    xs = [0.0, 0.255, 0.51, 0.765]
    A = [
        ("gemma-3-270m-it\n(base; 18 layers, d = 640,\n262k vocabulary)", "#F2F2F2"),
        ("SFT, 77 languages\nUniMax-style sampler, packed\nblock-diagonal mask\nstep 27,638", "#DCEBF7"),
        ("GRPO phase 1 (300 steps)\nb64 > b256 > b512\n> b1024 > mixed", "#DDF0E8"),
        ("GRPO phase 2\ngroup 8 > 16, consensus\nsignal, 77 languages", "#DDF0E8"),
        ("Phase 2b (lr 3e-6,\nno std-normalisation)\nKL 0.04 > 2.5: DIVERGED", "#F8DDD3"),
        ("Phase 2c: std-norm. advantage,\nlr 2e-6, KL guard, k3c\nb256 > ... > b2048 > mixed\nstep 1,433 = final model", "#DDF0E8"),
        ("Science SFT stage\nAfriScience-MT + replay +\nlong-doc distillation\nWiSE-FT alpha 0.7", "#DCEBF7"),
        ("Science + general RL (final2)\nensemble QE, anchors v3,\nlong-doc reward v2, NLLB signal\n401 steps, wind-down 351-401", "#FFF3D0"),
    ]
    ys = [0.80, 0.62]
    pos = []
    for i, (t, fc) in enumerate(A):
        r, c = divmod(i, 4)
        x, y = xs[c], ys[r]
        pos.append((x, y))
        box(ax, x, y, w, h, t, fc=fc, ec="#B03A1A" if "DIVERGED" in t else "#33506B", fs=5.6, ls="--" if "TODO" in t else "-")
    for i in range(7):
        (x0, y0), (x1, y1) = pos[i], pos[i + 1]
        if i == 3:
            arrow(ax, x0 + w / 2, y0, x1 + w / 2, y1 + h, rad=0.0)
        else:
            arrow(ax, x0 + w, y0 + h / 2, x1 - 0.002, y1 + h / 2)
    ax.text(0.5, 0.585, "2c restarts from the step-488 (b64-stage) model of 2b, not from its diverged end", fontsize=5.8, color="#B03A1A", ha="center", va="top")
    ax.text(0.0, 0.54, "Evaluation: FLORES+, NTREX-128, MAFAND-MT, SmolSent, SALT, IbomMT, GlotStoryBook (contamination-split);\nAfriLongDoc (6 length bins) and SmolDoc for documents; decoding / sampler studies.",
            fontsize=6, va="top", style="italic")
    ax.text(0.0, 0.47, "B  Reward of one rollout (per sentence chunk, then source-length-weighted mean)", fontsize=8, fontweight="bold", va="top")
    B = [
        ("1 Sanitise: cut at <end_of_turn>,\nNFC, strip tag; hard failure = -2\n(no end, copy, loop, wrong tag\nor language, length ratio)", "#F8DDD3"),
        ("2 Chunk: sentence split,\nlanguage-aware Gale-Church;\nomitted / extra sentence =\nfailed chunk (-2)", "#DCEBF7"),
        ("3 Content signal: AfriCOMET-QE\n+ SSA-COMET-MTL (mean, or min\nif they differ by > 1.0);\ngroup-consensus chrF++ for\nblend / consensus languages", "#DDF0E8"),
        ("4 Per-language shaping:\n0 = P90 of mismatched refs,\n+1 = human median,\n+2 = human P90;\nrange [-2, +2.25]", "#DDF0E8"),
        ("5 Penalties: length / word ratio,\nlexical, diacritics, anchors (copy,\nentity, stop-word, padding,\nrepetition, boilerplate)", "#FFF3D0"),
        ("6 Long documents: multi-level\nrepetition severity, anchored\nalignment, cvar-excess\naggregation, coverage", "#FFF3D0"),
        ("7 GRPO: 16 completions\n(15 sampled + greedy),\nstd-normalised advantage\nx language weight, k3c KL,\nlr 1e-6 .. 2e-6", "#EEE0F0"),
    ]
    xsB = [0.0, 0.255, 0.51, 0.765]
    ysB = [0.27, 0.07]
    posB = []
    for i, (t, fc) in enumerate(B):
        r, c = divmod(i, 4)
        x, y = xsB[c], ysB[r]
        posB.append((x, y)); box(ax, x, y, w, h + 0.045, t, fc=fc, fs=5.1)
    for i in range(6):
        (x0, y0), (x1, y1) = posB[i], posB[i + 1]
        if i == 3:
            arrow(ax, x0 + w / 2, y0, x1 + w / 2, y1 + h + 0.03)
        else:
            arrow(ax, x0 + w, y0 + (h + 0.03) / 2, x1 - 0.002, y1 + (h + 0.03) / 2)
    ax.text(0.765, 0.19, "Inference for documents > 512 source tokens:\ntranslate sentence by sentence (greedy),\nre-decode the ~1% unfinished sentences\nby best-of-N at T = 0.7.", fontsize=6, style="italic", va="top")
    save(fig, "fig1_pipeline")
    plt.close(fig)


# =================================================================================================== fig2: per-language gains
def published_flores(model):
    out = {}
    for r in read_csv(os.path.join(RDATA, "published_baselines.csv")):
        if r["model"] == model and r["direction"] == "en->xx" and r["benchmark"].startswith("FLORES-200 "):
            v = fnum(r["chrF++"])
            if v is not None:
                out[r["iso"]] = v
    return out


def flores_wide(subset="no_pair"):
    rows = [r for r in read_csv(os.path.join(RDATA, "final_model_per_language_wide.csv"))
            if r["benchmark"] == "flores_plus" and r["subset"] == subset]
    return rows


def fig2():
    rows = flores_wide("no_pair")
    nllb33 = published_flores("NLLB-200-3.3B")
    afq = published_flores("AfriqueGemma-12B")
    rows.sort(key=lambda r: float(r["rl_chrf"]))
    n = len(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(PAGE_W, 7.3), gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.06), sharey=True)
    ys = np.arange(n)
    for i, r in enumerate(rows):
        g = r["group"]; c = GROUP_COLOR.get(g, OI["grey"])
        s, e = float(r["sft_chrf"]), float(r["rl_chrf"])
        a1.plot([s, e], [i, i], color=c, lw=1.2, alpha=0.7, zorder=1)
        a1.scatter([s], [i], s=14, facecolors="white", edgecolors=c, lw=0.9, zorder=2)
        a1.scatter([e], [i], s=16, color=c, zorder=3)
        if r["iso"] in nllb33:
            a1.scatter([nllb33[r["iso"]]], [i], marker="|", s=55, color="black", lw=1.3, zorder=4)
        if r["iso"] in afq:
            a1.scatter([afq[r["iso"]]], [i], marker="D", s=11, color=OI["pink"], zorder=4)
        d, lo, hi = float(r["d_chrf"]), float(r["d_chrf_lo"]), float(r["d_chrf_hi"])
        a2.barh(i, d, color=c, alpha=0.85, height=0.7)
        a2.plot([lo, hi], [i, i], color="black", lw=0.7)
    a1.set_yticks(ys); a1.set_yticklabels([f"{r['language']} ({r['iso']})" for r in rows], fontsize=5.8)
    a1.set_xlabel("FLORES+ devtest chrF++ (no_pair rows)")
    a2.set_xlabel("RL minus SFT chrF++ (95% CI)")
    a1.set_ylim(-0.7, n - 0.3)
    a2.tick_params(left=False)
    h = [Line2D([], [], marker="o", color=GROUP_COLOR[g], ls="", markersize=4, label=GROUP_LABEL[g]) for g in GROUP_COLOR if any(r["group"] == g for r in rows)]
    h += [Line2D([], [], marker="o", mfc="white", mec="black", ls="", markersize=4, label="SFT-final"),
          Line2D([], [], marker="o", color="black", ls="", markersize=4, label="RL-final"),
          Line2D([], [], marker="|", color="black", ls="", markersize=7, label="NLLB-200 3.3B (published)"),
          Line2D([], [], marker="D", color=OI["pink"], ls="", markersize=4, label="AfriqueGemma-12B (published)")]
    a1.legend(handles=h, loc="lower right", fontsize=6)
    save(fig, "fig2_per_language_gain")
    plt.close(fig)


# =================================================================================================== fig3: phase diagnostics
def load_phase(tag):
    return read_csv(os.path.join(DATA, f"{tag}_steps.csv")), read_csv(os.path.join(DATA, f"{tag}_evals.csv"))


def smooth(y, k=9):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    ker = np.ones(k) / k
    pad = np.concatenate([np.full(k // 2, y[0]), y, np.full(k // 2, y[-1])])
    return np.convolve(pad, ker, mode="valid")[:len(y)]


def rl2c_series():
    rows = read_jsonl(os.path.join(DATA, "rl2c_metrics.jsonl"))
    tr = [r for r in rows if "kl" in r]
    ev = [r for r in rows if "bench/chrf_macro" in r]
    return tr, ev


def final2_series():
    """Rows of the final run with the restarts resolved: the run was OOM-killed twice on 2026-10-23 and resumed from the step-250 checkpoint, so
    metrics.jsonl holds steps 251.. up to three times; a record is dropped when a later record of the same or a lower step follows it (the resumed
    lineage wins).  The steps at which a lineage was replaced are stored in final2_series.restarts."""
    rows = read_jsonl(os.path.join(DATA, "final2_metrics.jsonl"))
    out, restarts = [], []
    for r in rows:
        if "kl" in r and out:
            last = max((x["step"] for x in out if "kl" in x), default=-1)
            if r["step"] <= last:
                restarts.append(int(r["step"]))
                out = [x for x in out if not ("kl" in x and x["step"] >= r["step"])]
        out.append(r)
    final2_series.restarts = sorted(set(restarts))
    tr = [r for r in out if "kl" in r]
    return out, tr
final2_series.restarts = []


def fig3():
    """Dev chrF++ chain over the phases (top) and KL / gradient norm / reward per phase (bottom)."""
    p1s, p1e = load_phase("rl1"); p2s, p2e = load_phase("rl2"); p2bs, p2be = load_phase("rl2b")
    tr2c, ev2c = rl2c_series()
    f2rows, f2tr = final2_series()
    phases = [("phase 1", 300), ("phase 2", 378), ("phase 2b", 839), ("phase 2c", 1433),
              ("final2", max(250, int(max(r["step"] for r in f2tr))))]
    wr = [p[1] for p in phases]
    fig, axes = plt.subplots(4, 5, figsize=(PAGE_W + 0.4, 6.6), gridspec_kw=dict(width_ratios=wr, wspace=0.07, hspace=0.35), sharey="row")
    # ---- row 0: dev chrF++
    def dev_ax(ax, steps, vals, c, label=None):
        ax.plot(steps, vals, "o-", ms=2.5, color=c, lw=1.0, label=label)

    ax = axes[0, 0]; dev_ax(ax, [int(r["step"]) for r in p1e if r["dev_chrf"] != ""], [float(r["dev_chrf"]) for r in p1e if r["dev_chrf"] != ""], OI["blue"])
    ax.axhline(39.65, color=OI["grey"], ls=":", lw=0.9); ax.text(60, 39.2, "SFT-final 39.65", fontsize=5.8, color=OI["grey"])
    ax = axes[0, 1]; dev_ax(ax, [int(r["step"]) for r in p2e if r["dev_chrf"] != ""], [float(r["dev_chrf"]) for r in p2e if r["dev_chrf"] != ""], OI["blue"])
    ax = axes[0, 2]; dev_ax(ax, [int(r["step"]) for r in p2be if r["dev_chrf"] != ""], [float(r["dev_chrf"]) for r in p2be if r["dev_chrf"] != ""], OI["blue"])
    ax.axvline(488, color=OI["verm"], ls="--", lw=0.8); ax.text(492, 40.2, "step 488\n(model kept)", fontsize=5.6, color=OI["verm"])
    ax = axes[0, 3]; dev_ax(ax, [r["step"] for r in ev2c], [r["bench/chrf_macro"] for r in ev2c], OI["blue"])
    # curriculum stages of 2c (config curriculum_steps: b256 391, b512 446, b1024 149, b2048 298, then mixed)
    cs = [("b256", 0, 391), ("b512", 391, 837), ("b1024", 837, 986), ("b2048", 986, 1284), ("mixed", 1284, 1433)]
    cols = ["#FFFFFF", "#F3F3F3", "#FFFFFF", "#F3F3F3", "#FFFFFF"]
    for (nm, a, b), cc in zip(cs, cols):
        axes[0, 3].axvspan(a, b, color=cc, zorder=0)
        axes[0, 3].text((a + b) / 2, 38.65, nm, ha="center", fontsize=5.6)
    ax = axes[0, 4]
    g = [(r["step"], r["bench_aux/general_dev/chrf_macro"]) for r in f2rows if "bench_aux/general_dev/chrf_macro" in r]
    dev_ax(ax, [a for a, _ in g], [b for _, b in g], OI["verm"])
    ax.set_ylim(38.5, 43.0)
    axes[0, 0].set_ylabel("dev chrF++\n(19 lang., 760 sent.)")
    # ---- row 1: KL (log)
    def line(ax, steps, vals, c, sm=7, alpha=0.35):
        ax.plot(steps, vals, color=c, lw=0.5, alpha=alpha)
        ax.plot(steps, smooth(vals, sm), color=c, lw=1.1)
    for ax, (st, c) in zip(axes[1, :3], [(p1s, OI["blue"]), (p2s, OI["blue"]), (p2bs, OI["verm"])]):
        line(ax, [int(r["step"]) for r in st], [max(float(r["kl"]), 1e-4) for r in st], c)
    line(axes[1, 3], [r["step"] for r in tr2c], [max(r["kl"], 1e-4) for r in tr2c], OI["blue"])
    line(axes[1, 4], [r["step"] for r in f2tr], [max(r["kl"], 1e-4) for r in f2tr], OI["verm"])
    for ax in axes[1]:
        ax.set_yscale("log"); ax.axhline(0.15, color=OI["grey"], ls=":", lw=0.8)
    axes[1, 0].set_ylabel("KL to reference\n(log; dotted = guard 0.15)")
    axes[1, 3].annotate("k3 outliers\n(steps 607, 747)", xy=(747, 5), xytext=(900, 20), fontsize=5.6, arrowprops=dict(arrowstyle="->", lw=0.6))
    # ---- row 2: grad norm
    for ax, (st, c) in zip(axes[2, :3], [(p1s, OI["blue"]), (p2s, OI["blue"]), (p2bs, OI["verm"])]):
        line(ax, [int(r["step"]) for r in st], [float(r["grad_norm"]) for r in st], c)
    line(axes[2, 3], [r["step"] for r in tr2c], [r["grad_norm"] for r in tr2c], OI["blue"])
    line(axes[2, 4], [r["step"] for r in f2tr], [r["grad_norm"] for r in f2tr], OI["verm"])
    axes[2, 0].set_ylabel("pre-clip grad norm\n(clip = 1.0)")
    for ax in axes[2]:
        ax.set_yscale("log")
    # ---- row 3: training reward
    for ax, (st, c) in zip(axes[3, :3], [(p1s, OI["blue"]), (p2s, OI["blue"]), (p2bs, OI["verm"])]):
        line(ax, [int(r["step"]) for r in st], [float(r["reward"]) for r in st], c, sm=15)
    line(axes[3, 3], [r["step"] for r in tr2c], [r["reward"] for r in tr2c], OI["blue"], sm=25)
    line(axes[3, 4], [r["step"] for r in f2tr], [r["reward"] for r in f2tr], OI["verm"], sm=15)
    axes[3, 0].set_ylabel("mean training reward\n(not comparable\nacross phases)")
    for i in range(4):   # final2: linear learning-rate wind-down from step 351 (written 2026-10-23 13:42 UTC, launch_FINAL2.log)
        axes[i, 4].axvspan(350.5, 401.5, color=OI["green"], alpha=0.10, lw=0)
    axes[0, 4].text(352, 38.7, "wind-down", fontsize=5.4, color=OI["green"])
    for j, (nm, _) in enumerate(phases):
        axes[0, j].set_title(nm, fontsize=7.5)
        axes[3, j].set_xlabel("step")
        for i in range(4):
            if j > 0:
                axes[i, j].tick_params(labelleft=False)
    for i in range(4):
        for j in range(5):
            axes[i, j].tick_params(axis="x", labelsize=6)
    fig.text(0.5, 0.005, "Phase 2b diverged from step ~490 (b256 stage); phase 2c restarted from the step-488 model. The reward definition changed between phases (calibration, ensemble, penalties), so only within-phase trends are meaningful.",
             ha="center", fontsize=5.8, style="italic")
    save(fig, "fig3_phase_diagnostics")
    plt.close(fig)


# =================================================================================================== fig4: length-bin results
def longdoc_tables():
    path = os.path.join(REPORTS, "longdoc_eval_final_model_2026-10-21.md")
    T = md_tables(path)
    core, human = {}, {}
    for t in T:
        h = t["heading"]
        if h.startswith("core bin b") and "macro over 8 languages" in h:
            b = h.split()[2]
            core[b] = {r[0]: r for r in t["rows"]}
        if h.startswith("human bin") and "macro over" in h:
            b = h.split()[2]
            human[b] = {r[0]: r for r in t["rows"]}
    return core, human


def fig4():
    core, human = longdoc_tables()
    xs = np.arange(len(BINS))
    def get(cfg, col, tabs=core):
        return [num(tabs[b][cfg][col]) for b in BINS]
    fig, axs = plt.subplots(2, 2, figsize=(PAGE_W, 5.2), gridspec_kw=dict(wspace=0.28, hspace=0.55)); axs = axs.ravel()
    sty = dict(
        whole=dict(c=OI["verm"], m="o", ls="-", label="final: whole document, greedy"),
        sent=dict(c=OI["blue"], m="s", ls="-", label="final: sentence by sentence + repair"),
        sftw=dict(c=OI["grey"], m="o", ls="--", label="SFT: whole document, greedy"),
        sfts=dict(c=OI["sky"], m="s", ls="--", label="SFT: sentence by sentence"),
        bon=dict(c=OI["orange"], m="^", ls="-", label="final: whole doc., best of 4 (T 0.7)"),
    )
    cfgs = dict(whole="greedy", sent="sentfix", sftw="sft:greedy", sfts="sft:greedy_sent", bon="bon4")
    for ax, col, ttl, scale in [(axs[0], 4, "finished (end token)", 1), (axs[1], 5, "documents with a loop", 1)]:
        for k, cfg in cfgs.items():
            if k == "bon":
                continue
            ax.plot(xs, get(cfg, col), marker=sty[k]["m"], color=sty[k]["c"], ls=sty[k]["ls"], ms=3.5)
        ax.set_title(ttl); ax.set_ylabel("% of documents"); ax.set_ylim(-3, 103)
    # chrF++ vs NLLB (col 1 has 'x [lo, hi]')
    ax = axs[2]
    for k, cfg in cfgs.items():
        ys = get(cfg, 1)
        ax.plot(xs, ys, marker=sty[k]["m"], color=sty[k]["c"], ls=sty[k]["ls"], ms=3.5, label=sty[k]["label"])
    lo = [nums(core[b]["sentfix"][1])[1] for b in BINS]; hi = [nums(core[b]["sentfix"][1])[2] for b in BINS]
    ax.fill_between(xs, lo, hi, color=OI["blue"], alpha=0.2, lw=0)
    ax.set_title("chrF++ vs NLLB-3.3B pseudo-ref."); ax.set_ylabel("chrF++ (8 languages)")
    # human references
    ax = axs[3]
    hb = ["sm", "st512", "st1024", "st2048"]
    hx = np.arange(len(hb))
    hl = ["~190 words", "~512 tok.", "~1,024 tok.", "~2,048 tok."]
    for k, cfg in cfgs.items():
        if k == "bon":
            continue
        ys = [num(human[b][cfg][1]) for b in hb]
        ax.plot(hx, ys, marker=sty[k]["m"], color=sty[k]["c"], ls=sty[k]["ls"], ms=3.5)
    ax.set_xticks(hx); ax.set_xticklabels(hl, rotation=25, ha="right")
    ax.set_title("chrF++ vs human SmolDoc"); ax.set_ylabel("chrF++ (6 / 4 languages)")
    for ax in axs[:3]:
        ax.set_xticks(xs); ax.set_xticklabels(BINS, rotation=25, ha="right")
        ax.set_xlabel("length bin (source tokens)")
    axs[3].set_xlabel("human document length")
    fig.legend(*axs[2].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=6.3)
    save(fig, "fig4_length_bins")
    plt.close(fig)


# =================================================================================================== fig5: reward-design evidence
def fig5():
    fig = plt.figure(figsize=(8.8, 6.6))
    gs = fig.add_gridspec(2, 3, wspace=0.95, hspace=0.75)
    # (a) QE metric agreement with humans.  qe_metric_choice_2026-10-21.md  sec 3.1 (l.44) and 3.2 (l.72)
    ax = fig.add_subplot(gs[0, 0])
    names = ["AfriCOMET\nQE (ours)", "SSA-COMET\nQE", "SSA-COMET\nMTL", "ensemble\n(z-mean)"]
    ssa_mte = [0.470, 0.550, 0.604, 0.584]; afrimte = [0.413, 0.421, 0.454, 0.465]
    x = np.arange(4); w = 0.38
    ax.bar(x - w / 2, ssa_mte, w, color=OI["blue"], label="SSA-MTE (7 pairs, in-domain for SSA)")
    ax.bar(x + w / 2, afrimte, w, color=OI["sky"], label="AfriMTE-WMT24 (9 pairs)")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=5.8, rotation=25, ha="right"); ax.set_ylim(0.3, 0.72); ax.set_ylabel("Spearman with human DA")
    ax.set_title("(a) agreement with human scores"); ax.legend(fontsize=5.3, loc="upper left")
    # (b) hackability: failure = corrupted scores >= human.  qe_ensemble_2026-10-21.md sec 2 (delta 1.0)
    ax = fig.add_subplot(gs[0, 1])
    classes = ["copy_source", "copy_noisy", "mismatch", "neighbour", "append_unrel.", "append_source", "mix_half_src"]
    afri = [0.9, 3.7, 10.1, 11.3, 25.6, 53.0, 17.2]; ssa = [40.6, 37.5, 1.7, 1.5, 5.1, 16.5, 13.4]
    mean = [8.7, 12.6, 2.9, 3.0, 10.2, 40.2, 12.4]; ens = [0.8, 3.3, 6.1, 7.2, 21.8, 41.6, 17.6]
    y = np.arange(len(classes)); h = 0.2
    for k, (v, c, l) in enumerate([(afri, OI["blue"], "AfriCOMET-QE"), (ssa, OI["orange"], "SSA-COMET-MTL"), (mean, OI["grey"], "plain mean"), (ens, OI["green"], "mean-or-min (used)")]):
        ax.barh(y + (k - 1.5) * h, v, h, color=c, label=l)
    ax.set_yticks(y); ax.set_yticklabels(classes, fontsize=5.8); ax.invert_yaxis(); ax.set_xlabel("corrupted >= human reference (%)")
    ax.set_title("(b) QE exploits (lower is better)"); ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    # (c) anchors: detection vs false positives.  anchors_2026-10-21.md sec 2 tables (l.45-67); anchors_v2 (copy) l.62
    ax = fig.add_subplot(gs[0, 2])
    det = [("other source's translation\n(entity, 3 langs)", 100), ("second half dropped\n(entity)", 59), ("clause repeated 6x\n(repetition)", 100),
           ("translation twice\n(repetition)", 100), ("stock sentence appended\n(boilerplate)", 100), ("exact / half copy of source\n(substring copy)", 100)]
    fp = [("entity", 0.7), ("repetition", 1.8), ("boilerplate", 0.4), ("substring copy", 0.5)]
    yy = np.arange(len(det))
    ax.barh(yy, [d for _, d in det], color=OI["green"], height=0.6)
    ax.set_yticks(yy); ax.set_yticklabels([n for n, _ in det], fontsize=5.3); ax.invert_yaxis(); ax.set_xlim(0, 118)
    ax.set_xlabel("corrupted human refs. detected (%)\nfalse positives on human refs.: entity 0.7%, repetition 1.8%,\nboilerplate 0.4%, substring copy 0.5%", fontsize=6)
    ax.set_title("(c) constraint detectors")
        # (d) long-document repetition guard: v1 vs v2 flag rate on GOOD sentence-by-sentence output.  longdoc_reward_v2_2026-10-22.md l.64-74 and l.54-62
    ax = fig.add_subplot(gs[1, 0])
    bins3 = ["b1024", "b2048", "b4096"]
    v1_good = [1, 11, 47]; v2_good = [0, 0, 0]; v1_loop = [18, 47, 67]; v2_loop = [100, 100, 100]
    x = np.arange(3); w = 0.2
    ax.bar(x - 1.5 * w, v1_good, w, color=OI["verm"], label="v1: good output flagged (false pos.)")
    ax.bar(x - 0.5 * w, v2_good, w, color=OI["green"], label="v2: good output flagged")
    ax.bar(x + 0.5 * w, v1_loop, w, color=OI["orange"], label="v1: +30% loop tail flagged")
    ax.bar(x + 1.5 * w, v2_loop, w, color=OI["blue"], label="v2: +30% loop tail penalised")
    ax.set_xticks(x); ax.set_xticklabels(bins3); ax.set_ylabel("% of documents"); ax.set_title("(d) long-doc. repetition guard")
    ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax.set_ylim(0, 190)
    # (e) padding-detector length bug.  anchors_v3_2026-10-22.md sec 1 (l.7)
    ax = fig.add_subplot(gs[1, 1])
    labs = ["<100", "100-200", "200-400", "400-800", ">800"]
    v2 = [5, 14, 31, 81, 100]
    ax.bar(np.arange(5), v2, color=OI["verm"], label="v2 padding feature (bug)")
    ax.bar(np.arange(5), [0, 0, 0, 0, 0], color=OI["green"], label="v3 (segmental type/token), OOD median 0.0%")
    ax.set_xticks(np.arange(5)); ax.set_xticklabels(labs, rotation=30, ha="right"); ax.set_xlabel("human document length (words)")
    ax.set_ylabel("human documents with deficit > 0 (%)"); ax.set_title("(e) padding detector: length bug")
    ax.legend(fontsize=5.3, loc="upper left"); ax.set_ylim(0, 130)
    # (f) ensemble min-branch share etc: AUC human vs mismatched per mode, qe_ensemble sec 3
    ax = fig.add_subplot(gs[1, 2])
    modes = ["all 44", "qe-mode", "blend-mode"]
    afa = [0.845, 0.885, 0.738]; ssa_a = [0.982, 0.993, 0.953]; ensa = [0.915, 0.945, 0.833]
    x = np.arange(3); w = 0.26
    ax.bar(x - w, afa, w, color=OI["blue"], label="AfriCOMET-QE"); ax.bar(x, ssa_a, w, color=OI["orange"], label="SSA-COMET-MTL"); ax.bar(x + w, ensa, w, color=OI["green"], label="ensemble")
    ax.set_xticks(x); ax.set_xticklabels(modes); ax.set_ylim(0.6, 1.02); ax.set_ylabel("AUC, human vs mismatched ref.")
    ax.set_title("(f) QE discrimination (FLORES+ dev)"); ax.set_ylim(0.6, 1.2); ax.legend(fontsize=5.3, loc="upper center", ncol=1)
    save(fig, "fig5_reward_design")
    plt.close(fig)


# =================================================================================================== fig6: decoding study
def fig6():
    T = md_tables(os.path.join(REPORTS, "final_model_decoding_study_2026-10-21.md"))
    t = [x for x in T if x["header"][:2] == ["config", "chrF++"]][0]
    rows = []
    for r in t["rows"]:
        name = r[0]
        if "cross-check" in name:
            continue
        d = nums(r[2]) if r[2] != "0" else [0.0, 0.0, 0.0]
        rows.append((name, float(r[1]), d[0], d[1] if len(d) > 2 else 0, d[2] if len(d) > 2 else 0))
    fig, axs = plt.subplots(1, 2, figsize=(PAGE_W + 0.4, 3.3), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.9))
    ax = axs[0]
    short = {
        "greedy (vLLM) = reference": "greedy", "greedy + repetition_penalty 1.05 (vLLM, prompt + output)": "rep. penalty 1.05",
        "greedy + repetition_penalty 1.1 (vLLM, prompt + output)": "rep. penalty 1.1", "greedy + repetition penalty 1.05, generated tokens only (HF)": "rep. 1.05 (gen. tokens)",
        "greedy + repetition penalty 1.1, generated tokens only (HF)": "rep. 1.1 (gen. tokens)", "greedy + repetition_penalty 1.1 (HF, prompt + output)": "rep. 1.1 (HF)",
        "greedy + no_repeat_ngram_size 3 (HF)": "no-repeat 3-gram", "greedy + no_repeat_ngram_size 4 (HF)": "no-repeat 4-gram",
        "sampling T 0.3 (vLLM, 1 sample)": "sample T 0.3", "sampling T 0.5 (vLLM, 1 sample)": "sample T 0.5", "sampling T 0.7 (vLLM, 1 sample)": "sample T 0.7",
        "beam search 2 (HF)": "beam 2", "beam search 4 (HF)": "beam 4", "beam search 8 (HF)": "beam 8",
    }
    plot = []
    for n, v, d, lo, hi in rows:
        if n in short:
            plot.append((short[n], d, lo, hi, OI["blue"] if "beam" in short[n] else (OI["verm"] if d < -0.2 else OI["grey"])))
        elif n.startswith("MBR: 8 samples T 0.7,"):
            plot.append(("MBR 8 @T0.7", d, lo, hi, OI["green"]))
        elif n == "MBR: 8 samples T 0.5":
            plot.append(("MBR 8 @T0.5", d, lo, hi, OI["green"]))
    plot = plot[1:] if plot and plot[0][0] == "greedy" else plot
    y = np.arange(len(plot))
    ax.barh(y, [p[1] for p in plot], color=[p[4] for p in plot], height=0.65)
    for i, p in enumerate(plot):
        ax.plot([p[2], p[3]], [i, i], color="black", lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([p[0] for p in plot], fontsize=6.2); ax.invert_yaxis(); ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("macro chrF++ minus greedy (32.52); 20 languages x 200 sentences")
    ax.set_title("(a) decoding rules, sentence level")
    # (b) sampler grid: T x cut-off  (generation_config_study_grid.csv)
    G = read_csv(os.path.join(RDATA, "generation_config_study_grid.csv"))
    ax = axs[1]
    series = {}
    for r in G:
        nm = r["name"]
        if nm.startswith("T") and "_" not in nm and r["T"]:
            series.setdefault("temperature only", []).append((float(r["T"]), float(r["d_chrf_all"])))
    for lab, pat, c in [("T + min-p 0.05", "_m0.05", OI["green"]), ("T + top-p 0.9", "_p0.9", OI["orange"])]:
        pts = [(float(r["T"]), float(r["d_chrf_all"])) for r in G if r["name"].endswith(pat) and r["T"]]
        if pts:
            series[lab] = pts
    for lab, pts in series.items():
        pts = sorted(set(pts))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, label=lab)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("temperature"); ax.set_ylabel("chrF++ minus greedy"); ax.set_title("(b) single-sample samplers")
    ax.legend(fontsize=6)
    save(fig, "fig6_decoding")
    plt.close(fig)


# =================================================================================================== fig7: SFT-stage guardrails
def fig7():
    T = md_tables(os.path.join(REPORTS, "final_run_integration_2026-10-21.md"))
    # first table: mix_v1 sweep; second (addendum): mixtures A / B_v1 / A2W / B_v2_ibom (one line per mixture, 4 alphas per cell)
    t1 = [t for t in T if t["header"][:2] == ["model", "sent. chrF++"]][0]
    t2 = [t for t in T if t["header"][:2] == ["model", "sent. chrF++"] and "weak-blend d [95 % CI]" in t["header"]][0]
    alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
    base = {r[0]: r for r in t1["rows"]}["final_1433"]
    sent0, paper0, fin0 = num(base[1]), num(base[2]), num(base[3])
    mix1 = {"alpha 0.3": 0.3, "alpha 0.5": 0.5, "alpha 0.7": 0.7, "alpha 1.0": 1.0}
    series = {"mix_v1": ([0.0], [sent0], [paper0], [0.0])}
    xs, s, p, b = [0.0], [sent0], [paper0], [0.0]
    for r in t1["rows"]:
        if r[0] in mix1:
            xs.append(mix1[r[0]]); s.append(num(r[1])); p.append(num(r[2])); b.append(num(r[5]))
    series["mix_v1"] = (xs, s, p, b)
    for r in t2["rows"]:
        nm = r[0].replace("*", "").strip()
        if nm.startswith("final_1433"):
            continue
        key = nm.split(" a")[0]
        if not key:
            continue
        S = nums(r[1]); P = nums(r[2]); B = [float(v) for v in __import__("re").findall(r"[-+]\d+\.\d+", r[5].replace("**", "").replace("[", " ").replace("]", " ")) ]
        # weak-blend delta cell lists 4 values "a / b / c / d" with CIs for some; take the signed ones in order
        raw = r[5].replace("**", "")
        parts = [x.strip() for x in raw.split("/")]
        Bv = [num(x) for x in parts]
        if len(S) == 4 and len(Bv) == 4:
            series[key] = ([0.0] + alphas[1:], [sent0] + S, [paper0] + P, [0.0] + Bv)
    fig, axs = plt.subplots(1, 3, figsize=(PAGE_W + 0.4, 2.6), gridspec_kw=dict(wspace=0.42))
    cmap = {"mix_v1": OI["grey"], "A": OI["sky"], "B_v1": OI["orange"], "A2W": OI["blue"], "B_v2_ibom": OI["pink"]}
    for k, (xs_, s_, p_, b_) in series.items():
        c = cmap.get(k, OI["black"])
        axs[0].plot(xs_, s_, marker="o", ms=3, color=c, label=k)
        axs[1].plot(xs_, p_, marker="o", ms=3, color=c)
        axs[2].plot(xs_, b_, marker="o", ms=3, color=c)
    axs[2].axhline(-0.5, color=OI["verm"], ls=":", lw=0.9); axs[2].text(0.02, -0.46, "guardrail: > -0.5", fontsize=5.8, color=OI["verm"])
    axs[2].axhline(0, color="black", lw=0.5)
    axs[0].set_title("AfriScience dev: sentence chrF++"); axs[1].set_title("AfriScience dev: paper-level chrF"); axs[2].set_title("weak-blend dev (19 lang.):\nchrF++ change vs final_1433")
    for a in axs:
        a.set_xlabel("interpolation alpha")
    axs[0].legend(fontsize=6, title="SFT mixture", title_fontsize=6)
    save(fig, "fig7_sft_stage_guardrail")
    plt.close(fig)


# =================================================================================================== fig8: coverage
def fig8():
    sys.path.insert(0, os.path.join(REPO, "gemma_mt"))
    rows = read_csv(os.path.join(RDATA, "final_model_per_language_wide.csv"))
    prim = {}
    for r in rows:
        if r["subset"] == "no_pair" and r["primary"] == "1":
            prim[r["iso"]] = r
    allrows = {}
    for r in read_csv(os.path.join(RDATA, "final_model_per_language.csv")):
        allrows[r["iso"]] = (r["language"], r["group"], r["family"])
    groups = list(GROUP_COLOR)
    fam_order = ["Bantu", "Atlantic-Congo (non-Bantu)", "Afroasiatic", "Nilo-Saharan", "Mande", "Creole / Pidgin", "Other (Indo-European, Austronesian)"]
    import collections
    cnt = collections.defaultdict(lambda: collections.Counter())
    nob = collections.defaultdict(lambda: collections.Counter())
    for iso, (lang, g, fam) in allrows.items():
        cnt[fam][g] += 1
        if iso not in prim:
            nob[fam][g] += 1
    fig, axs = plt.subplots(1, 2, figsize=(PAGE_W + 0.4, 2.6), gridspec_kw=dict(wspace=0.6, width_ratios=[1.6, 1]))
    ax = axs[0]
    bottoms = np.zeros(len(fam_order))
    for g in groups:
        v = np.array([cnt[f][g] for f in fam_order], float)
        ax.barh(range(len(fam_order)), v, left=bottoms, color=GROUP_COLOR[g], label=GROUP_LABEL[g], height=0.65)
        bottoms += v
    ax.set_yticks(range(len(fam_order))); ax.set_yticklabels(fam_order, fontsize=6); ax.invert_yaxis()
    ax.set_xlabel("languages (85 table rows; 77 RL-trained)")
    for i, f in enumerate(fam_order):
        t = sum(cnt[f].values()); z = sum(nob[f].values())
        ax.text(t + 0.4, i, f"{t}" + (f"  ({z} without a human-reference benchmark)" if z else ""), va="center", fontsize=5.5)
    ax.set_xlim(0, 50)
    ax.legend(fontsize=6, loc="lower right")
    ax.set_title("(a) languages by family and training group")
    # (b) benchmark availability
    ax = axs[1]
    bench_n = collections.Counter()
    for r in rows:
        if r["subset"] == "all":
            bench_n[r["benchmark"]] += 1
    order = ["flores_plus", "smol_sent", "ntrex", "mafand", "glot_story", "salt", "ibom_mt", "flores_moz"]
    names = {"flores_plus": "FLORES+ devtest", "smol_sent": "SmolSent", "ntrex": "NTREX-128", "mafand": "MAFAND-MT", "glot_story": "GlotStoryBook", "salt": "SALT", "ibom_mt": "IbomMT", "flores_moz": "FLORES-MOZ"}
    v = [bench_n[k] // 3 if k in ("flores_plus",) else bench_n[k] for k in order]
    # wide csv has 3 subsets per (language, benchmark) pair -> divide by number of subsets present
    v = [len({r["iso"] for r in rows if r["benchmark"] == k}) for k in order]
    ax.barh(range(len(order)), v, color=OI["sky"], height=0.65)
    for i, x in enumerate(v):
        ax.text(x + 0.5, i, str(x), va="center", fontsize=6)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[k] for k in order], fontsize=6); ax.invert_yaxis()
    ax.set_xlabel("languages covered"); ax.set_title("(b) human-reference benchmarks (66 of 85 languages)")
    save(fig, "fig8_coverage")
    plt.close(fig)


# =================================================================================================== fig9: final2
def fig9():
    rows, tr = final2_series()
    ev = [r for r in rows if "bench/chrf_macro" in r]
    aux = [r for r in rows if "bench_aux/afrisci_dev_doc/finished_rate" in r]
    step_max = int(max(r["step"] for r in tr))
    fig, axs = plt.subplots(2, 4, figsize=(8.8, 4.6), gridspec_kw=dict(wspace=0.6, hspace=0.6))
    a = axs[0, 0]
    a.plot([r["step"] for r in tr], smooth([r["reward"] for r in tr], 11), color=OI["verm"]); a.plot([r["step"] for r in tr], [r["reward"] for r in tr], color=OI["verm"], alpha=0.25, lw=0.5)
    a.set_title("training reward"); a.set_xlabel("step")
    a = axs[0, 1]
    a.plot([r["step"] for r in tr], [max(r["kl"], 1e-4) for r in tr], color=OI["blue"], lw=0.8); a.set_yscale("log"); a.axhline(0.15, ls=":", color=OI["grey"])
    a2 = a.twinx(); a2.plot([r["step"] for r in tr], [r.get("kl_guard_lr_scale", 1.0) for r in tr], color=OI["orange"], lw=0.9, drawstyle="steps-post"); a2.set_ylim(0, 1.1); a2.grid(False)
    a2.set_ylabel("KL-guard lr scale", color=OI["orange"], fontsize=6)
    a.set_title("KL to start model\n(guard 0.15)"); a.set_xlabel("step")
    a = axs[0, 2]
    a.plot([r["step"] for r in tr], [r["grad_norm"] for r in tr], color=OI["grey"], lw=0.6); a.set_yscale("log"); a.yaxis.set_minor_formatter(__import__("matplotlib").ticker.NullFormatter()); a.set_title("grad norm (clip 1)"); a.set_xlabel("step")
    a = axs[0, 3]
    a.plot([r["step"] for r in tr], smooth([r["completion_len"] for r in tr], 9), color=OI["green"]); a.set_title("completion length (tokens)"); a.set_xlabel("step")
    def evplot(ax, key, ttl, c, ylab=None):
        pts = [(r["step"], r[key]) for r in rows if key in r]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, color=c)
            ax.axhline(pts[0][1], color=OI["grey"], ls=":", lw=0.8)
        ax.set_title(ttl); ax.set_xlabel("step")
    evplot(axs[1, 0], "bench/chrf_macro", "AfriScience dev:\nsentence chrF++", OI["blue"])
    evplot(axs[1, 1], "bench_aux/afrisci_dev_doc/chrf_macro", "paper-level chrF++\n(25 docs / lang.)", OI["verm"])
    evplot(axs[1, 2], "bench_aux/afrisci_dev_doc/finished_rate", "paper-level\nfinished rate", OI["verm"])
    a = axs[1, 3]
    for key, lab, c in [("bench_aux/general_dev/chrf_macro", "general dev (19 lang.)", OI["blue"]), ("bench_aux/weak_dev/chrf_macro", "weak-language dev", OI["orange"])]:
        pts = [(r["step"], r[key]) for r in rows if key in r]
        if pts:
            a.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, color=c, label=lab)
    a.legend(fontsize=5.5, loc="center right"); a.set_title("forgetting monitors\n(chrF++)"); a.set_xlabel("step")
    # events of the run (steps and times from /workspace/jobs/r7/logs/launch_FINAL2.log and the metrics file):
    #   step 150: planned restart (exit 143 at 2026-10-22 21:22:41 UTC) -> KL guard reset, the padding ("stuffed output") detector no longer fires (anchors v3 in force, see Section 6.7);
    #   step 250: OOM kills at 04:22 and 04:49 UTC on 2026-10-23, resumed from the step-250 checkpoint (the discarded steps 251-267 are removed by final2_series);
    #   step 351: linear learning-rate wind-down (written 13:42 UTC), 50 steps, final evaluation at step 401.
    for ax_ in axs.ravel():
        ax_.axvline(150.5, color=OI["grey"], ls="--", lw=0.7, alpha=0.8)
        for rs in final2_series.restarts:
            ax_.axvline(rs - 0.5, color=OI["verm"], ls="--", lw=0.7, alpha=0.7)
        ax_.axvspan(350.5, step_max + 0.5, color=OI["green"], alpha=0.10, lw=0)
    fig.text(0.5, 0.005, "grey dashed: restart at step 150 (KL guard reset; padding detector v3 in force); red dashed: resume from the step-250 checkpoint after two OOM kills (2026-10-23); green band: learning-rate wind-down, steps 351-401", ha="center", fontsize=5.6, style="italic")
    fig.suptitle(f"final2 (science + general RL), steps 1-{step_max} (700 planned; wound down from step 351 after the stop criteria held at step 350)", fontsize=7.5, y=0.99)
    save(fig, "fig9_final2")
    plt.close(fig)


# =================================================================================================== fig10: RL vs SFT data (research question)
def fig10():
    """Research question 'RL as a data-efficient complement to SFT': gains per stage and per unit of data (from paper/data/rq_budget.json),
    and RL versus extra parallel data for the two Ge'ez-script languages (wide CSV; extra-data numbers from
    reports/final_run2_integration_2026-10-22.md section 1: C3 minus A2W on the 60-sentence weak dev set)."""
    b = json.load(open(os.path.join(DATA, "rq_budget.json")))
    a = b["sets"]["all"]
    tot_t = b["total_tokens"] / 1e9; tot_c = b["total_completions"] / 1e6
    sft20 = b["sft_tokens_b"] - b["late_b"]; late = b["late_b"]
    stages = ["SFT\nearly\n(to 20.5k)", "SFT\nlate\n(last 26%)", "RL\nall\nphases"]
    gains = [a["s20"] - a["base"], a["s27"] - a["s20"], a["rl"] - a["s27"]]
    per_b = [gains[0] / sft20, gains[1] / late, gains[2] / tot_t]
    fig, axs = plt.subplots(1, 3, figsize=(PAGE_W + 1.2, 2.7), gridspec_kw=dict(wspace=0.55, width_ratios=[1, 1, 1.25]))
    cols = [OI["grey"], OI["grey"], OI["verm"]]
    ax = axs[0]; ax.bar(range(3), gains, color=cols); ax.set_xticks(range(3)); ax.set_xticklabels(stages, fontsize=5.8)
    ax.set_ylabel("macro chrF++ gain (45 FLORES+ lang.,\n320-token budget)"); ax.set_title("(a) gain per stage")
    for i, g in enumerate(gains): ax.text(i, g + 0.3, f"{g:+.2f}", ha="center", fontsize=6)
    ax = axs[1]; ax.bar(range(3), per_b, color=cols); ax.set_xticks(range(3)); ax.set_xticklabels(stages, fontsize=5.8)
    ax.set_ylabel("chrF++ per billion tokens\n(training / generated)"); ax.set_title("(b) gain per unit of data\n(not comparable units)")
    for i, g in enumerate(per_b): ax.text(i, g + 0.1, f"{g:.2f}", ha="center", fontsize=6)
    # (c) Ge'ez
    wide = read_csv(os.path.join(RDATA, "final_model_per_language_wide.csv"))
    def d(iso, bench):
        r = [x for x in wide if x["iso"] == iso and x["benchmark"] == bench and x["subset"] == "no_pair"][0]
        return float(r["d_chrf"])
    ax = axs[2]
    rl = [d("amh", "flores_plus"), d("tir", "flores_plus")]
    extra = [0.02, 0.55]   # C3 minus A2W, weak dev chrF++ (reports/final_run2_integration_2026-10-22.md section 1)
    x = np.arange(2); wdt = 0.36
    ax.bar(x - wdt / 2, rl, wdt, color=OI["verm"], label="RL on top of SFT (FLORES+)")
    ax.bar(x + wdt / 2, extra, wdt, color=OI["sky"], label="+4.5 M tokens teacher data (C3 vs A2W)")
    for xi, v in zip(x - wdt / 2, rl): ax.text(xi, v + 0.1, f"{v:+.1f}", ha="center", fontsize=6)
    for xi, v in zip(x + wdt / 2, extra): ax.text(xi, v + 0.1, f"{v:+.2f}", ha="center", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(["Amharic", "Tigrinya"]); ax.set_ylabel("chrF++ gain"); ax.legend(fontsize=5.3, loc="upper right", bbox_to_anchor=(1.02, 1.0))
    ax.set_ylim(0, 6.2); ax.set_title("(c) Ge'ez-script languages:\nRL versus more parallel data")
    save(fig, "fig10_rq_data")
    plt.close(fig)


# =================================================================================================== fig11: controlled ablation (AB2)
def fig11():
    """Controlled ablation (design AB2): RL from SFT-final (arm S = F) versus RL from the re-created weak SFT (arm W), and the continued-SFT controls from W (W_S1 equal GPU time, W_S1P equal number of sentences).
    Reads paper/data/ablation_results.json (copy of AB2's results/ablation_results.json; curves and point estimates) and, for the 95% paired-bootstrap intervals of panel (b),
    paper/data/ab2_results/ab2_results.json (FLORES+ no_pair, language macro).  Schema of the first file:
        {"arms": {"S": {"curve": [[step, dev_macro_chrf], ...], "start_flores": x, "end_flores": y, ...}, "W": {...}, "W_S1": {...}, "W_S1P": {...}}, "units": "RL steps"}
    Without the file the figure shows placeholders."""
    path = os.path.join(DATA, "ablation_results.json")
    fullp = os.path.join(DATA, "ab2_results", "ab2_results.json")
    fig, axs = plt.subplots(1, 2, figsize=(PAGE_W + 0.6, 2.9), gridspec_kw=dict(wspace=0.35, width_ratios=[1.0, 1.2]))
    ax = axs[0]
    have = os.path.exists(path)
    ARMS = (("S", OI["verm"], "F: RL from SFT-final"), ("W", OI["blue"], "W: RL from weak SFT"), ("W_S1", OI["green"], "W + continued SFT (S1)"), ("W_S1P", OI["orange"], "W + continued SFT (S1P)"))
    if have:
        arms = json.load(open(path))["arms"]
        for k, c, lab in ARMS[:2]:
            cv = arms.get(k, {}).get("curve", [])
            if cv:
                ax.plot([p[0] for p in cv], [p[1] for p in cv], "o-", ms=3, color=c, label=lab)
        ax.set_xlabel("RL step"); ax.set_ylabel("dev macro chrF++ (yor kin orm lin)"); ax.legend(fontsize=6, loc="center right"); ax.set_title("(a) dev chrF++ during RL")
    else:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.text(0.5, 0.55, "ablation result missing", ha="center", va="center", fontsize=11, color=OI["grey"])
        ax.set_title("(a) dev chrF++ during RL")
    ax = axs[1]
    if have:
        arms = json.load(open(path))["arms"]
        ci = {}
        if os.path.exists(fullp):
            mac = json.load(open(fullp))["results"]["units"]["FLORES+ no_pair"]["all"]["macro"]["chrf"]
            ci = {"S": (mac["F0"], mac["F_RL"]), "W": (mac["W0"], mac["W_RL"]), "W_S1": (mac["W0"], mac["W_S1"]), "W_S1P": (mac["W0"], mac["W_S1P"])}
        present = [(k, c, lab) for k, c, lab in ARMS if k in arms and arms[k].get("start_flores") is not None]
        x = np.arange(len(present)); wdt = 0.36
        def err(i):
            return [[ci[k][i][0] - ci[k][i][1] for k, _, _ in present], [ci[k][i][2] - ci[k][i][0] for k, _, _ in present]] if ci else None
        ax.bar(x - wdt / 2, [arms[k]["start_flores"] for k, _, _ in present], wdt, color="#CCCCCC", label="start (F0 for F, W0 for the others)", yerr=err(0), error_kw=dict(lw=0.8, capsize=1.5))
        ax.bar(x + wdt / 2, [arms[k].get("end_flores") or 0 for k, _, _ in present], wdt, color=[c for _, c, _ in present], label="end (49 RL steps or continued SFT)", yerr=err(1), error_kw=dict(lw=0.8, capsize=1.5))
        for xi, (k, _, _) in zip(x, present):
            ax.text(xi + wdt / 2, arms[k]["end_flores"] + 1.6, f"{arms[k]['gain_flores']:+.2f}", ha="center", fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels([{"S": "F: RL", "W": "W: RL", "W_S1": "W + SFT (S1)", "W_S1P": "W + SFT (S1P)"}[k] for k, _, _ in present], fontsize=6.5)
        ax.set_ylabel("FLORES+ macro chrF++ (yor kin orm lin)"); ax.set_ylim(0, 42); ax.legend(fontsize=6, loc="upper right")
    else:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.text(0.5, 0.5, "ablation result missing", ha="center", va="center", fontsize=9, color=OI["grey"], transform=ax.transAxes)
    ax.set_title("(b) FLORES+ before and after (95% intervals; gain above bars)")
    save(fig, "fig11_ablation")
    plt.close(fig)


def figlc():
    """Long-context benchmark figures fig14-fig16 (agent LC1): delegated to make_longctx.py."""
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_longctx.py"), "figs"], check=False)


def figcpu():
    """Figures fig17 (tokenisation cost by language) and fig18 (CPU quality-speed trade-off): delegated to make_cpu.py."""
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_cpu.py"), "figs"], check=False)


def figjudge():
    """Figure fig19 (LLM-judge scores of aligned long-document units, Run 3): delegated to make_judge.py."""
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_judge.py"), "figs"], check=False)


ALL = dict(figjudge=figjudge, figcpu=figcpu, figlc=figlc, fig1=fig1, fig2=fig2, fig3=fig3, fig4=fig4, fig5=fig5, fig6=fig6, fig7=fig7, fig8=fig8, fig9=fig9, fig10=fig10, fig11=fig11)

if __name__ == "__main__":
    which = sys.argv[1:] or list(ALL)
    for k in which:
        ALL[k]()
