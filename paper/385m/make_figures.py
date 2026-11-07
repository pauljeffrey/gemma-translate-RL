#!/usr/bin/env python3
"""Figures of paper_385m/paper.md.   python paper_385m/make_figures.py   (matplotlib, numpy, pandas; no network)

Reads ONLY local files under reports/, reports/data/ and research_archive_2026-11-03/ and writes PNG + PDF into
paper_385m/figures/.  Deterministic: no random numbers are drawn (every interval is copied from a report's own bootstrap),
fixed fonts and style, no timestamps in the PDF metadata.  A figure whose input does not exist locally is skipped with a
message; a figure whose experiment result is not yet local would be drawn as a clearly marked PENDING placeholder and redrawn from the data file
the first time that file exists. All experiments with figures (E-15, E-05, E-09b, E-13) now have their data; E-02, E-04, E-11, E-14, E-10 are not run.

Figures (file stem: content: inputs)
  fig17_full_results         (not made here) copied from latex_snippets_full_results/ (script s03_build.py in reports/data/full_results_comparison_2026-11-06/)
  fig15_e02_decoding         E-02 greedy / beam-4 / MBR-16 at 768 tokens, SFT and RL (reports/data/e02_beam_mbr_768_2026-11-05/cmp_all.json)
  fig16_e04_e11_seeds_ablations E-04 seed spread and E-11 ablation arms (reports/data/e04_rl_seeds_2026-11-05/cmp.json, reports/data/e11_rl_ablations_2026-11-05/cmp.json)
  fig14_e03_control          E-03 matched-compute continued-SFT control vs RL, by language panel (reports/data/e03_control_execution_2026-11-04/{verdict_tables.json,evidence_c1/*.json})
  fig13_sft_validation_curve SFT training loss, validation loss and learning rate (reports/data/e23_hygiene_2026-11-04/sft_385m_curve.csv)
  fig1_flores_per_language   FLORES+ chrF++ per language SFT / stage A / B1 best / mixed best; stage deltas with CI
      reports/data/eval_weak_group_2026-11-03/results/results_{sft,stageA,b1best,mixedbest}.csv,
      reports/data/rl_vs_sft_evidence_2026-11-04/evidence.json
  fig2_rl_training_curves    reward, KL, greedy finished rate, dev chrF++ vs step per RL stage
      research_archive_2026-11-03/{runs_385m/{A,B1,mixed},clean_rerun_misc/runs_385m_{weak/mixed,sci/final,long/final}}/metrics.jsonl,
      runs_385m/A/kl_guard_events.jsonl
  fig3_decomposition_guard   RL - SFT decomposition incl. loop-cut and decode-time guard (E-01); dev grid of the 7 guard settings
      reports/data/rl_vs_sft_evidence_2026-11-04/evidence.json, reports/data/e01_loop_guard_2026-11-04/{evidence_e01,guard_dev}.json
  fig4_nllb_43               RL models against NLLB-200 on the 43 shared languages
      reports/data/nllb_comparison_rl_models_2026-11-04/table{1,2,3}_*.csv
  fig5a_science_sentence     AfriScience-MT sentence level: start / science SFT / science RL best+final (test), paired differences
      reports/data/eval_science_rl_2026-11-04/sci_pairs.json, reports/data/eval_science_sft_2026-11-04/tables.md
  fig5b_science_document     AfriScience-MT documents (168): whole vs chunked, 385M models vs the old 270M
      reports/data/eval_science_rl_2026-11-04/doc_tables.md
  fig6_weak_group            continued RL on 8 weak West-African languages: per-language change, dev series
      reports/data/eval_weak_group_2026-11-03/scores/bootstrap_by_language.csv, research_archive_2026-11-03/clean_rerun_misc/runs_385m_weak/mixed/metrics.jsonl
  fig7_loops_finishing       finished / loop rates (reward flag and independent detector), English-output share
      reports/data/rl_vs_sft_evidence_2026-11-04/evidence.json, reports/data/degeneration_audit_independent_2026-11-04/{summary_by_model_benchmark,bootstrap_rates,arabic_tifinagh_english_leak}.csv
  fig8_engine_parity         E-09: vLLM / HF, bf16 / fp32 (SFT and RL)
      reports/data/e09_engine_parity_2026-11-04/{parity_bootstrap,parity_summary}.json
  fig9_long_document_rl      E-15 long-document RL: whole vs chunked, paired differences (reports/data/eval_long_rl_2026-11-04/{doc_tables.md,doc_pairs.json}); placeholder pending_e15_long_document if absent
  fig12_ssa_all_stages       E-05 SSA-COMET-QE / AfriCOMET / chrF for all stages (reports/data/e05_ssa_all_stages_2026-11-04/tables.md); placeholder pending_e05_ssa_qe_stages if absent
  (E-02 beam-4 / MBR-16 at 768 tokens: not run, no figure)
  fig11_fp32_full_set        E-09b full-set fp32 SFT (reports/data/e09b_fp32_2026-11-04/evidence_e09b.json); placeholder pending_e09b_fp32_fullset if absent
  fig10_old270m_same_harness E-13 old 270M SFT / RL re-run in the 385M harness (reports/data/e13_old270m_same_harness_2026-11-04/evidence_e13.json); placeholder pending_e13_old270m if absent
E-20 (B1 best against B1 final) is a table in paper.md (numbers in reports/e20_b1_final_2026-11-04.md), not a figure.
"""
import csv
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "reports", "data")
ARCH = os.path.join(ROOT, "research_archive_2026-11-03")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito colour-blind-safe palette
BLUE, ORANGE, GREEN, VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
SKY, PURPLE, YELLOW, GREY, BLACK = "#56B4E9", "#CC79A7", "#F0E442", "#8C8C8C", "#000000"
SYS_COL = {"SFT": GREY, "stage A": SKY, "B1 best": ORANGE, "mixed best": BLUE, "weak final": GREEN}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 100,
    "savefig.dpi": 200, "pdf.fonttype": 42, "svg.hashsalt": "paper385m",
})
MADE = []


def p(*a):
    return os.path.join(*a)


def save(fig, stem):
    fig.savefig(p(OUT, stem + ".png"), dpi=200, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(p(OUT, stem + ".pdf"), bbox_inches="tight", metadata={"Creator": "make_figures.py", "Producer": "matplotlib", "CreationDate": None})
    plt.close(fig)
    MADE.append(stem)
    print("wrote", stem)


def need(*paths):
    miss = [x for x in paths if not os.path.exists(x)]
    if miss:
        print("SKIP (missing input):", ", ".join(os.path.relpath(m, ROOT) for m in miss))
        return False
    return True


def jload(path):
    with open(path) as f:
        return json.load(f)


def rows_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")


def parse_md_table(path, heading_prefix):
    """Rows (list of lists of str) of the first markdown table after a line starting with heading_prefix."""
    lines = open(path).read().split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith(heading_prefix)), None)
    if i is None:
        return None
    j = i
    while j < len(lines) and not lines[j].startswith("|"):
        j += 1
    out = []
    while j < len(lines) and lines[j].startswith("|"):
        cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
        if not set("".join(cells)) <= set("-: "):
            out.append(cells)
        j += 1
    return out


def num(s):
    m = re.search(r"-?\d+\.?\d*", s.replace("*", ""))
    return float(m.group(0)) if m else float("nan")


# --------------------------------------------------------------------------------------------------------------------
# Figure 1: FLORES+ per language along the RL chain and stage-by-stage macro gain
def fig1():
    base = p(DATA, "eval_weak_group_2026-11-03", "results")
    ev = p(DATA, "rl_vs_sft_evidence_2026-11-04", "evidence.json")
    files = {"SFT": "sft", "stage A": "stageA", "B1 best": "b1best", "mixed best": "mixedbest"}
    if not need(ev, *[p(base, f"results_{v}.csv") for v in files.values()]):
        return
    vals = {}
    for name, f in files.items():
        d = {}
        for r in rows_csv(p(base, f"results_{f}.csv")):
            if r["benchmark"] == "flores_plus" and r["subset"] == "all" and r["iso"] != "MACRO":
                d[r["iso"]] = float(r["chrf"])
        vals[name] = d
    langs = sorted(vals["mixed best"], key=lambda k: vals["mixed best"][k])  # ascending -> best language on top
    E = jload(ev)
    fig = plt.figure(figsize=(11.5, 10.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.3, 1.0], height_ratios=[1, 1], wspace=0.28, hspace=0.32)
    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(langs))
    sft = np.array([vals["SFT"][k] for k in langs]); mix = np.array([vals["mixed best"][k] for k in langs])
    ax.hlines(y, sft, mix, color="#bbbbbb", lw=1.2, zorder=1)
    for name, mk, sz in (("SFT", "o", 22), ("stage A", "s", 20), ("B1 best", "D", 16), ("mixed best", "o", 28)):
        v = np.array([vals[name][k] for k in langs])
        ax.scatter(v, y, s=sz, marker=mk, color=SYS_COL[name] if name != "SFT" else "white",
                   edgecolor=SYS_COL[name], linewidth=1.0, label=name, zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(langs, fontsize=6.5); ax.set_ylim(-0.8, len(langs) - 0.2)
    ax.set_xlabel("chrF++ on FLORES+ devtest (greedy, 768 tokens, 1,012 sentences per language)")
    ax.set_title("FLORES+ per language along the RL chain (n = 54 languages, 1 seed per model)", loc="left")
    ax.legend(loc="lower right", ncol=1)
    panel_label(ax, "a")
    # stage deltas
    ax2 = fig.add_subplot(gs[0, 1])
    benches = [("flores_plus", "FLORES+ (54)"), ("ntrex", "NTREX (24)"), ("mafand", "MAFAND (15)")]
    steps = [("sft->stageA", "SFT -> A", SKY), ("stageA->b1", "A -> B1", ORANGE), ("b1->mixed", "B1 -> mixed", BLUE)]
    w = 0.26
    for si, (key, lab, col) in enumerate(steps):
        for bi, (b, _) in enumerate(benches):
            pp = E["stages"][key][b]["pp"]  # [before, after, delta, lo, hi, n]
            ax2.bar(bi + (si - 1) * w, pp[2], w * 0.92, color=col, label=lab if bi == 0 else None,
                    yerr=[[pp[2] - pp[3]], [pp[4] - pp[2]]], capsize=2, error_kw={"lw": 0.8})
            ax2.text(bi + (si - 1) * w, pp[4] + 0.15, f"{pp[2]:+.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax2.set_xticks(range(3)); ax2.set_xticklabels([b[1] for b in benches], fontsize=7.5)
    ax2.set_ylabel("chrF++ gain of the step"); ax2.set_ylim(-0.5, 10.5)
    ax2.set_title("Stage-by-stage gain\n(95% paired bootstrap; B1 = best, step 50)", loc="left", fontsize=8.5)
    ax2.legend(loc="upper right"); panel_label(ax2, "b")
    # macro level
    ax3 = fig.add_subplot(gs[1, 1])
    names = ["SFT", "stage A", "B1 best", "mixed best"]
    for bi, (b, lab) in enumerate(benches):
        lv = [E["stages"]["sft->stageA"][b]["pp"][0], E["stages"]["sft->stageA"][b]["pp"][1],
              E["stages"]["stageA->b1"][b]["pp"][1], E["stages"]["b1->mixed"][b]["pp"][1]]
        ax3.plot(range(4), lv, marker="o", color=[BLUE, VERM, GREEN][bi], label=lab)
        ax3.text(3.08, lv[-1], f"{lv[-1]:.2f}", fontsize=7, va="center", color=[BLUE, VERM, GREEN][bi])
    ax3.set_xticks(range(4)); ax3.set_xticklabels(names, rotation=20, fontsize=7.5); ax3.set_xlim(-0.2, 3.6)
    ax3.set_ylabel("language-macro chrF++"); ax3.set_title("Macro level per stage", loc="left", fontsize=8.5)
    ax3.legend(loc="lower right", fontsize=7); panel_label(ax3, "c")
    save(fig, "fig1_flores_per_language")


# --------------------------------------------------------------------------------------------------------------------
# Figure 13: SFT training and validation curve (385M growth: SFT-1 and the six SFT-2 shards)
def fig13():
    f = p(DATA, "e23_hygiene_2026-11-04", "sft_385m_curve.csv")
    if not need(f):
        return
    rows = rows_csv(f)
    runs = []
    for r in rows:
        if r["run"] not in runs:
            runs.append(r["run"])
    step = np.array([float(r["step"]) for r in rows]); lr = np.array([float(r["lr"]) if r["lr"] else np.nan for r in rows])
    tl = np.array([float(r["train_loss"]) if r["train_loss"] else np.nan for r in rows])
    vm = np.array([float(r["val_loss_macro"]) if r["val_loss_macro"] else np.nan for r in rows]); vi = np.array([float(r["val_loss_micro"]) if r["val_loss_micro"] else np.nan for r in rows])
    bounds = [min(step[[r["run"] == k for r in rows]]) for k in runs]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    ax.plot(step, tl, color=BLUE, alpha=0.3, lw=0.6); ax.plot(step, smooth(np.nan_to_num(tl, nan=np.nanmean(tl)), 20), color=BLUE, lw=1.3)
    ax.set_ylim(0.9, 2.4); ax.set_ylabel("training loss (per logged step)"); ax.set_xlabel("optimiser step (131,072 tokens)")
    ax.set_title("(a) Training loss", loc="left", fontsize=8.5)
    ax = axes[1]
    ok = ~np.isnan(vm); ok2 = ~np.isnan(vi)
    ax.plot(step[ok], vm[ok], marker="o", ms=2.5, color=VERM, label="macro over languages"); ax.plot(step[ok2], vi[ok2], marker="s", ms=2.5, color=GREEN, label="micro (token-weighted)")
    ax.set_xlabel("optimiser step"); ax.set_ylabel("validation loss"); ax.legend(fontsize=7.5)
    ax.set_title("(b) Validation loss (final: macro 1.3412, micro 1.2846,\n78 languages; end of SFT-1: 1.7847)", loc="left", fontsize=8.5)
    ax = axes[2]
    ax.plot(step, lr, color=PURPLE, lw=1.0); ax.set_xlabel("optimiser step"); ax.set_ylabel("learning rate"); ax.set_title("(c) Learning rate (Muon; fresh warm-up per shard)", loc="left", fontsize=8.5)
    for ax in axes:
        for b in bounds[1:]:
            ax.axvline(b, color=GREY, lw=0.5, ls=":")
    for k, b in zip(runs[1:], bounds[1:]):
        axes[0].text(b, 2.3, " " + k.replace("g1b_", ""), fontsize=6.5, color=GREY, va="top") if k != "g1b_s1" else None
    fig.suptitle("385M SFT (SFT-1: new layers and tag rows only, 1,500 steps; SFT-2: six shards, all weights; 32,853 steps, one seed). The validation macro jumps at the start of SFT-2 because new languages enter the validation set (1.78 at the end of SFT-1, 1.98 at the end of shard 1).", fontsize=8, y=1.04)
    save(fig, "fig13_sft_validation_curve")


# --------------------------------------------------------------------------------------------------------------------
# Figure 14: E-03 matched-compute continued-SFT control
def fig14():
    d = p(DATA, "e03_control_execution_2026-11-04")
    ev = p(d, "evidence_c1"); vt = p(d, "verdict_tables.json")
    files = {"mixed_tp": "evidence_mixed_vs_c6000_TP19.json", "mixed_ot": "evidence_mixed_vs_c6000_OTHERS.json",
             "stageA_tp": "evidence_stageA_vs_c6000_TP19.json", "stageA_ot": "evidence_stageA_vs_c6000_OTHERS.json"}
    if not need(vt, *[p(ev, f) for f in files.values()]):
        return
    J = {k: jload(p(ev, f)) for k, f in files.items()}; V = jload(vt)
    B = [("flores_plus", "FLORES+"), ("ntrex", "NTREX"), ("mafand", "MAFAND")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.15, 1.15, 1.0], "wspace": 0.3})
    for ax, (grp, title) in zip(axes[:2], [("tp", "(a) The 19 TranslatePsy languages (47 cells)"), ("ot", "(b) The other languages (46 cells)")]):
        for k, (lab, col, src, key) in enumerate([("continued SFT (control, 6,000 steps)", ORANGE, "mixed_" + grp, "sft->c1_s6000"), ("RL stage A", SKY, "stageA_" + grp, "sft->stageA"),
                                                  ("RL mixed best", BLUE, "mixed_" + grp, "sft->mixed")]):
            for bi, (b, _) in enumerate(B):
                pp = J[src]["stages"][key][b]["pp"]
                ax.errorbar(bi + (k - 1) * 0.22, pp[2], yerr=[[pp[2] - pp[3]], [pp[4] - pp[2]]], fmt="o", color=col, capsize=3, label=lab if bi == 0 else None)
                ax.annotate(f"{pp[2]:+.1f}", (bi + (k - 1) * 0.22, pp[2]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=6.5)
        ax.axhline(0, color=BLACK, lw=0.7); ax.set_xticks(range(3)); ax.set_xticklabels([f"{l}\n({J['mixed_' + grp]['stages']['sft->mixed'][b]['pp'][5]} languages)" for b, l in B], fontsize=7.5)
        ax.set_xlim(-0.5, 2.5); ax.set_ylabel("chrF++ gain over the SFT (language macro)"); ax.set_title(title, loc="left", fontsize=8.5)
    axes[0].legend(fontsize=7, loc="upper right"); axes[0].set_ylim(-0.5, 9); axes[1].set_ylim(-0.5, 12)
    ax = axes[2]
    pans = [("mixed best", "all 93 cells"), ("mixed best TP19", "19 TranslatePsy\nlanguages (47 cells)"), ("mixed best OTHERS", "other languages\n(46 cells)")]
    for k, (pref, col, lab) in enumerate([("mixed best", BLUE, "RL mixed best - control"), ("stage A", SKY, "RL stage A - control")]):
        for pi, (nm, pl) in enumerate(pans):
            key = nm.replace("mixed best", pref)
            pp = V[key]["pp"]
            ax.errorbar(pi + (k - 0.5) * 0.25, pp[2], yerr=[[pp[2] - pp[3]], [pp[4] - pp[2]]], fmt="o", color=col, capsize=3, label=lab if pi == 0 else None)
            ax.annotate(f"{pp[2]:+.2f}", (pi + (k - 0.5) * 0.25, pp[2]), textcoords="offset points", xytext=(11, 0), va="center", fontsize=6.5)
    ax.axhline(0, color=BLACK, lw=0.7); ax.axhline(1.0, color=GREEN, ls="--", lw=0.9); ax.text(2.45, 1.15, "registered margin +1.0\n(overall endpoint)", fontsize=6.5, color=GREEN, ha="right", va="bottom")
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in pans], fontsize=7); ax.set_xlim(-0.5, 2.5); ax.set_ylabel("chrF++ difference, RL minus control (95% bootstrap)"); ax.legend(fontsize=7, loc="center right")
    ax.set_title("(c) RL minus control; verdict as registered:\ninconclusive (condition d > 0 on the 19 languages fails)", loc="left", fontsize=8.5)
    fig.suptitle("E-03 matched-compute control: continued SFT (LR 1e-5, 6,000 steps, 50% replay + 50% TranslatePsy en->xx for 19 languages, equal pairs per language), one seed, one budget point; clean rows, language macro; "
                 "the 3,100- and 4,800-step points were not evaluated", fontsize=7.8, y=1.03)
    save(fig, "fig14_e03_control")


# --------------------------------------------------------------------------------------------------------------------
# Figure 15: E-02 greedy / beam-4 / MBR-16 at 768 tokens, SFT and RL
def fig15():
    f = p(DATA, "e02_beam_mbr_768_2026-11-05", "cmp_all.json")
    if not need(f):
        return
    D = jload(f)["systems"]
    dec = [("greedy", "greedy", GREY), ("beam4", "beam-4", BLUE), ("beam4hf", "beam-4\nHF rule", SKY), ("mbr16", "MBR-16", ORANGE)]
    mets = [("chrfpp", "chrF++ (MBR utility)"), ("spbleu", "spBLEU"), ("comet", "held-out AfriCOMET-STL")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"wspace": 0.28})
    for ax, (m, lab) in zip(axes, mets):
        for mi, model in enumerate(["sft", "rl"]):
            for di, (d, dl, col) in enumerate(dec):
                r = D[f"{model}_{d}"]["all"][m]
                pt = r["point"]; lo = r.get("lo", pt); hi = r.get("hi", pt)
                x = mi * 5 + di
                ax.errorbar(x, pt, yerr=[[pt - lo], [hi - pt]], fmt="o", color=col, capsize=3)
                ax.annotate(f"{pt:.2f}" if m != "comet" else f"{pt:.4f}", (x, pt), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=6)
        ax.set_xticks([0, 1, 2, 3, 5, 6, 7, 8]); ax.set_xticklabels([dl for _, dl, _ in dec] * 2, fontsize=7)
        ax.text(1.5, ax.get_ylim()[0], "SFT", ha="center", va="bottom", fontsize=9, fontweight="bold"); ax.text(6.5, ax.get_ylim()[0], "RL (mixed best)", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(lab, fontsize=8.5)
    axes[0].set_ylabel("language macro, 20 languages x 200 FLORES+ devtest sentences")
    fig.suptitle("Beam-4 and MBR-16 at 768 new tokens (E-02; 4,000 sentences per system, 95% paired bootstrap over source sentences; one SFT and one RL checkpoint, vLLM bf16). "
                 "beam-4 = stop when 4 hypotheses have ended; HF rule = Hugging Face early_stopping=False; MBR is metric-matched (utility chrF++). Not comparable with the earlier 256-token study, whose beam loop ran past the end token.", fontsize=8, y=1.04)
    save(fig, "fig15_e02_decoding")


# --------------------------------------------------------------------------------------------------------------------
# Figure 16: E-04 seeds and E-11 ablations
def fig16():
    f4 = p(DATA, "e04_rl_seeds_2026-11-05", "cmp.json"); f11 = p(DATA, "e11_rl_ablations_2026-11-05", "cmp.json")
    if not need(f4, f11):
        return
    E4, E11 = jload(f4)["pairs"], jload(f11)["pairs"]
    g = lambda P, k: (P[k]["all"]["chrfpp"]["point"], P[k]["all"]["chrfpp"]["lo"], P[k]["all"]["chrfpp"]["hi"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.3})
    ax = axes[0]
    items = [("seed 2\n(full)", g(E4, "s2:sft"), BLUE), ("seed 3\n(full)", g(E4, "s3:sft"), BLUE), ("(a) no\nfeatures", g(E11, "a_base:sft"), VERM), ("(c) no reference\ndomain", g(E11, "c_noref:sft"), ORANGE),
             ("(d) no beam\nguide", g(E11, "d_noguide:sft"), GREEN)]
    for i, (lab, (pt, lo, hi), col) in enumerate(items):
        ax.errorbar(i, pt, yerr=[[pt - lo], [hi - pt]], fmt="o", color=col, capsize=3); ax.annotate(f"{pt:.2f}", (i, pt), textcoords="offset points", xytext=(10, 0), va="center", fontsize=7)
    lo_b, hi_b = min(items[0][1][0], items[1][1][0]), max(items[0][1][0], items[1][1][0])
    ax.axhspan(lo_b, hi_b, color=BLUE, alpha=0.15); ax.text(4.45, hi_b + 0.02, "range of the two\nfull-feature seeds", fontsize=6.5, color=BLUE, ha="right", va="bottom")
    ax.set_xticks(range(len(items))); ax.set_xticklabels([l for l, _, _ in items], fontsize=7); ax.set_xlim(-0.5, 4.8); ax.set_ylim(5.35, 6.3)
    ax.set_ylabel("chrF++ gain over the SFT, 49 cells of 20 languages (95% bootstrap)")
    ax.set_title("(a) Gain over the SFT after 100 steps (b64 recipe, one run per arm;\narms (a), (c), (d) fork seed 2's step-40 state)", loc="left", fontsize=8.5)
    ax = axes[1]
    s23 = E4["s2:s3"]["all"]["chrfpp"]["point"]
    eff = [("seed spread\ns2 - s3", g(E4, "s2:s3"), GREY), ("(a) no\nfeatures\n- full", g(E11, "a_base:s2"), VERM), ("(c) no ref.\ndomain\n- full", g(E11, "c_noref:s2"), ORANGE),
           ("(d) no beam\nguide\n- full", g(E11, "d_noguide:s2"), GREEN), ("B1 best\n- stage A", g(E4, "b1best:stageA"), SKY), ("mixed best\n- B1 best", g(E4, "mixed:b1best"), BLUE), ("stage A (400)\n- seed 2 (100)", g(E4, "stageA:s2"), PURPLE)]
    for i, (lab, (pt, lo, hi), col) in enumerate(eff):
        ax.errorbar(i, pt, yerr=[[pt - lo], [hi - pt]], fmt="o", color=col, capsize=3); ax.annotate(f"{pt:+.2f}", (i, pt), textcoords="offset points", xytext=(10, 0), va="center", fontsize=7)
    ax.axhspan(-abs(s23), abs(s23), color=GREY, alpha=0.2); ax.axhline(0, color=BLACK, lw=0.7)
    ax.text(6.4, abs(s23) + 0.03, f"observed seed difference |s2 - s3| = {abs(s23):.2f}", fontsize=6.5, color=GREY, ha="right", va="bottom")
    ax.set_xticks(range(len(eff))); ax.set_xticklabels([l for l, _, _ in eff], fontsize=6.8); ax.set_xlim(-0.5, 6.9)
    ax.set_ylabel("chrF++ difference (paired bootstrap)")
    ax.set_title("(b) Effect sizes against the seed spread (one pair of seeds, step 100;\nB1 and mixed rows are the stored stage models on the same 20 languages)", loc="left", fontsize=8.5)
    fig.suptitle("Seed spread (E-04) and component ablations (E-11), 20 languages (yor hau ibo swh amh zul xho kin som wol lug sna twi ewe lin tir orm nso tsn afr), FLORES+ / NTREX / MAFAND, greedy, 768 tokens", fontsize=8.2, y=1.03)
    save(fig, "fig16_e04_e11_seeds_ablations")


# --------------------------------------------------------------------------------------------------------------------
# Figure 2: RL training curves per stage
def load_metrics(path):
    rows = {}
    train, dev = {}, {}
    nlang = None
    with open(path) as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            st = r.get("step")
            if st is None:
                continue
            if "kl" in r and "reward" in r:
                train[st] = r  # restarts rewrite a step: keep the last record
            if "bench/chrf_macro" in r:
                dev[st] = r
                nlang = sum(1 for k in r if k.startswith("bench/finished/"))
    return train, dev, nlang


def smooth(x, w=10):
    x = np.asarray(x, float)
    if len(x) < 3:
        return x
    k = min(w, len(x))
    c = np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    return c


def fig2():
    stages = [
        ("RL stage A", p(ARCH, "runs_385m", "A"), "400 steps x 64 prompts"),
        ("RL B1", p(ARCH, "runs_385m", "B1"), "135 steps x 48"),
        ("RL mixed", p(ARCH, "runs_385m", "mixed"), "60 steps x 42"),
        ("weak-group RL", p(ARCH, "clean_rerun_misc", "runs_385m_weak", "mixed"), "85 steps, 8 languages"),
        ("science RL", p(ARCH, "clean_rerun_misc", "runs_385m_sci", "final"), "160 steps x 42"),
        ("long-document RL", p(ARCH, "clean_rerun_misc", "runs_385m_long", "final"), "80 steps x 32 prompts, group 8"),
    ]
    stages = [s for s in stages if os.path.exists(p(s[1], "metrics.jsonl"))]
    if not stages:
        print("SKIP fig2 (no metrics.jsonl)")
        return
    fig, axes = plt.subplots(4, len(stages), figsize=(2.55 * len(stages), 8.6), sharex="col")
    rowlab = ["training reward", "KL (k3) to start model", "greedy finished rate", "dev chrF++ (greedy)"]
    for ci, (name, d, sub) in enumerate(stages):
        train, dev, nl = load_metrics(p(d, "metrics.jsonl"))
        st = np.array(sorted(train))
        for ri, key in enumerate(["reward", "kl", "greedy_finished_rate"]):
            ax = axes[ri, ci]
            v = np.array([train[s].get(key, np.nan) for s in st], float)
            ax.plot(st, v, color=BLUE, alpha=0.25, lw=0.7)
            ax.plot(st, smooth(v), color=BLUE, lw=1.4)
            if ri == 2:
                ax.set_ylim(0.8, 1.005)
            if ci == 0:
                ax.set_ylabel(rowlab[ri])
        ax = axes[3, ci]
        ds = np.array(sorted(dev))
        if len(ds):
            ax.plot(ds, [dev[s]["bench/chrf_macro"] for s in ds], marker="o", ms=3, color=VERM, lw=1.2)
        if ci == 0:
            ax.set_ylabel(rowlab[3])
        if len(ds):
            dv = [dev[s]["bench/chrf_macro"] for s in ds]; lo_, hi_ = min(dv), max(dv)
            if hi_ - lo_ < 1.0:  # dev evaluation noise is sd 0.09-0.10 chrF++: do not magnify it
                mid = (lo_ + hi_) / 2; ax.set_ylim(mid - 0.5, mid + 0.5)
        ax.set_xlabel("step")
        nsent = next((int(dev[s]["bench/n"]) for s in ds if "bench/n" in dev[s]), None) if len(ds) else None
        axes[0, ci].set_title(f"{name}\n{sub}\ndev: {nl} languages, n = {nsent}" if nsent else f"{name}\n{sub}", fontsize=8)
        if name == "RL stage A":
            for ri in range(4):
                axes[ri, ci].axvline(41, color=GREY, ls=":", lw=1)
            ev = p(d, "kl_guard_events.jsonl")
            cuts = []
            if os.path.exists(ev):
                cuts = [json.loads(l)["step"] for l in open(ev) if '"event": "cut"' in l]
            for c in cuts:
                for ri in range(4):
                    axes[ri, ci].axvline(c, color=VERM, ls="--", lw=0.8)
            axes[0, ci].text(41, axes[0, ci].get_ylim()[0], " features\n on", fontsize=6, color=GREY, va="bottom")
            axes[1, ci].text(cuts[0] if cuts else 226, axes[1, ci].get_ylim()[1], "KL-guard\ncuts ", fontsize=6, color=VERM, va="top", ha="right")
        if name == "long-document RL" and int(st.max()) < 80:
            axes[0, ci].text(0.5, 0.06, f"in progress:\nlatest step {int(st.max())}", transform=axes[0, ci].transAxes,
                             fontsize=7, color=VERM, ha="center")
    fig.suptitle("RL training records per stage (1 seed per stage; thin = per step, thick = 10-step mean; dev sets differ per stage; dev panels span at least 1 chrF++ because the dev evaluation noise is sd 0.09-0.10)\n"
                 "Stage A, B1, mixed, weak: FLORES+ dev held-out half; science and long-document RL: AfriScience-MT dev. Reward definitions differ across stages.",
                 fontsize=8.5, y=1.0)
    fig.tight_layout()
    save(fig, "fig2_rl_training_curves")


# --------------------------------------------------------------------------------------------------------------------
# Figure 3: decomposition of the RL gain, incl. the decode-time guard (E-01) and the guard grid
def fig3():
    ev = p(DATA, "rl_vs_sft_evidence_2026-11-04", "evidence.json")
    e1 = p(DATA, "e01_loop_guard_2026-11-04", "evidence_e01.json")
    gd = p(DATA, "e01_loop_guard_2026-11-04", "guard_dev.json")
    if not need(ev, e1, gd):
        return
    E, G, D = jload(ev), jload(e1), jload(gd)
    cells = lambda d, k: d["macro"][k]["all_cells"]
    sft, rl = cells(E, "all")["pp"], cells(E, "all")["pp"]
    sft_v, rl_v = sft[0], sft[1]
    cut_v = cells(E, "sft_trunc")["pp"][0]
    g = cells(G, "all")["pp"]  # sft_guard, rl_guard
    sg_v, rg_v = g[0], g[1]
    ac = [cells(E, "all")["com"][0], None, cells(G, "all")["com"][0], cells(E, "all")["com"][1], cells(G, "all")["com"][1]]
    labels = ["SFT\ngreedy", "SFT cut at\nfirst loop\n(post hoc)", "SFT +\nguard\n(freq 0.10)", "RL\ngreedy", "RL +\nguard"]
    pp = [sft_v, cut_v, sg_v, rl_v, rg_v]
    cols = [GREY, SKY, ORANGE, BLUE, GREEN]
    fig = plt.figure(figsize=(14, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 1.3], wspace=0.38)
    ax = fig.add_subplot(gs[0])
    ax.bar(range(5), pp, color=cols, width=0.7)
    for i, v in enumerate(pp):
        ax.text(i, v + 0.3, f"{v:.2f}", ha="center", fontsize=8)
    d_sft_rl = E["macro"]["all"]["all_cells"]["pp"]
    d_cut = E["macro"]["sft_trunc"]["all_cells"]["pp"]
    for (x0, x1, yy, txt) in ((0, 3, 37.2, f"RL - SFT = {d_sft_rl[2]:+.2f} [{d_sft_rl[3]:+.2f}, {d_sft_rl[4]:+.2f}]"),
                              (1, 3, 39.0, f"RL - loop-cut SFT = {d_cut[2]:+.2f} [{d_cut[3]:+.2f}, {d_cut[4]:+.2f}]"),
                              (2, 4, 40.8, f"RL+guard - SFT+guard = {g[2]:+.2f} [{g[3]:+.2f}, {g[4]:+.2f}]")):
        ax.annotate("", xy=(x1, yy), xytext=(x0, yy), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text((x0 + x1) / 2, yy + 0.25, txt, ha="center", fontsize=6.8)
    ax.set_ylim(20, 43); ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=6.3)
    ax.set_ylabel("chrF++, macro over 93 benchmark-language cells")
    ax.set_title("(a) chrF++: what is left of the RL gain\n93 cells: FLORES+ 54, NTREX 24, MAFAND 15", loc="left", fontsize=8.5)
    ax = fig.add_subplot(gs[1])
    av = [ac[0], None, ac[2], ac[3], ac[4]]
    for i, v in enumerate(av):
        if v is None:
            ax.text(i, 0.52, "not\nscorable", ha="center", fontsize=7, color=GREY)
            continue
        ax.bar(i, v, color=cols[i], width=0.7); ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    gc = cells(G, "all")["com"]; ec = cells(E, "all")["com"]
    ax.set_ylim(0.50, 0.68); ax.set_xticks(range(5)); ax.set_xticklabels(["SFT", "loop-\ncut", "SFT\n+guard", "RL", "RL\n+guard"], fontsize=7)
    ax.set_ylabel("held-out AfriCOMET-STL, macro over 76 cells")
    ax.set_title(f"(b) AfriCOMET (held out), 76 cells\nRL - SFT {ec[2]:+.4f}; with guards {gc[2]:+.4f}", loc="left", fontsize=8.5)
    ax = fig.add_subplot(gs[2])
    order = ["greedy", "freq002", "freq005", "freq010", "rep102", "rep105", "freq005_rep102"]
    lab = {"greedy": "greedy", "freq002": "freq 0.02", "freq005": "freq 0.05", "freq010": "freq 0.10\n(chosen)", "rep102": "rep 1.02",
           "rep105": "rep 1.05", "freq005_rep102": "f 0.05\n+ r 1.02"}
    x = np.arange(len(order)); w = 0.38
    s_ = [D["sft_grid"][k]["chrfpp_macro"] for k in order]; r_ = [D["rl_grid"][k]["chrfpp_macro"] for k in order]
    ax.bar(x - w / 2, s_, w, color=GREY, label="SFT"); ax.bar(x + w / 2, r_, w, color=BLUE, label="RL (mixed best)")
    for i, k in enumerate(order):
        ax.text(i - w / 2, s_[i] + 0.2, f"{s_[i]:.1f}", ha="center", fontsize=6)
        ax.text(i - w / 2, 27.3, f"{D['sft_grid'][k]['loop']:.2f}", ha="center", fontsize=5.5, color="white")
    ax.set_ylim(27, 43); ax.set_xticks(x); ax.set_xticklabels([lab[k].replace("freq ", "f ").replace("rep ", "r ").replace(" (chosen)", "\n(chosen)") for k in order], fontsize=6.3)
    ax.set_ylabel("dev chrF++ (720 sentences, 18 languages)")
    ax.set_title("(c) Guard grid on the FLORES+ dev held-out half\n(white digits: SFT loop flag; rule fixed in advance; 1 seed)", loc="left", fontsize=8.5)
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    fig.suptitle("Decomposition of the RL gain (one training seed per model; intervals: 95% paired bootstrap over test sentences)", y=1.02, fontsize=9.5)
    save(fig, "fig3_decomposition_guard")


# --------------------------------------------------------------------------------------------------------------------
# Figure 4: NLLB comparison on the 43 shared languages
def fig4():
    d = p(DATA, "nllb_comparison_rl_models_2026-11-04")
    if not need(p(d, "table1_macro_43_languages.csv"), p(d, "table2_per_language_43.csv"), p(d, "table3_by_nllb_tertile.csv")):
        return
    t2 = rows_csv(p(d, "table2_per_language_43.csv")); t3 = rows_csv(p(d, "table3_by_nllb_tertile.csv"))
    f = lambda r, k: float(r[k])
    reward_nllb = {"kea", "kbp", "nus", "bam", "kab"}
    fig = plt.figure(figsize=(12.5, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1, 1.15], wspace=0.28, hspace=0.45)
    # (a) macro
    ax = fig.add_subplot(gs[0, 0])
    mac = {k: np.mean([f(r, k) for r in t2]) for k in ["nllb_3p3b", "nllb_1p3b", "nllb_distilled_1p3b", "nllb_moe_54p5b", "sft", "stageA", "b1best", "mixedbest"]}
    names = [("SFT", "sft", GREY), ("stage\nA", "stageA", SKY), ("B1\nbest", "b1best", ORANGE), ("mixed\nbest", "mixedbest", BLUE),
             ("NLLB\n1.3B", "nllb_1p3b", "#cccccc"), ("NLLB\ndist.", "nllb_distilled_1p3b", "#cccccc"),
             ("NLLB\n3.3B", "nllb_3p3b", BLACK), ("NLLB\nMoE", "nllb_moe_54p5b", "#cccccc")]
    for i, (lab, k, col) in enumerate(names):
        ax.bar(i, mac[k], color=col, width=0.7)
        ax.text(i, mac[k] + 0.3, f"{mac[k]:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n[0] for n in names], fontsize=7, rotation=0)
    ax.set_ylim(15, 43); ax.set_ylabel("chrF++, macro over 43 languages")
    r = mac["mixedbest"] / mac["nllb_3p3b"]; r0 = mac["sft"] / mac["nllb_3p3b"]
    ax.set_title(f"(a) Mixed best = {r*100:.0f}% of NLLB-3.3B (SFT {r0*100:.0f}%)\n43 FLORES+ languages shared with the NLLB table", loc="left", fontsize=8.5)
    # (b) tertiles
    ax = fig.add_subplot(gs[1, 0])
    tl = [r_["tertile"] for r_ in t3 if not r_["tertile"].startswith("all")]
    series = [("NLLB 3.3B", "nllb_3p3b", BLACK), ("SFT", "sft", GREY), ("stage A", "stageA", SKY), ("mixed best", "mixedbest", BLUE)]
    w = 0.2
    for si, (lab, k, col) in enumerate(series):
        v = [float(r_[k]) for r_ in t3 if not r_["tertile"].startswith("all")]
        ax.bar(np.arange(3) + (si - 1.5) * w, v, w * 0.92, color=col, label=lab)
        if k == "mixedbest":
            for i, r_ in enumerate([r_ for r_ in t3 if not r_["tertile"].startswith("all")]):
                ax.text(i + (si - 1.5) * w, v[i] + 0.5, f"{float(r_['mixedbest_pct_of_nllb3p3b']):.0f}%", ha="center", fontsize=7, color=BLUE)
    ax.set_xticks(range(3)); ax.set_xticklabels([t.replace(" (", "\n(") + " languages" for t in tl], fontsize=7.5)
    ax.set_xlabel("tertile of NLLB-3.3B chrF++ (14 / 14 / 15 languages)"); ax.set_ylabel("chrF++")
    ax.legend(ncol=2, fontsize=7); ax.set_title("(b) By NLLB resource tertile (label: mixed best as % of NLLB-3.3B)", loc="left", fontsize=8.5)
    # (c) per language
    ax = fig.add_subplot(gs[:, 1])
    rr = sorted(t2, key=lambda r_: f(r_, "mixed_minus_nllb3p3b"))
    y = np.arange(len(rr))
    diff_m = np.array([f(r_, "mixed_minus_nllb3p3b") for r_ in rr]); diff_s = np.array([f(r_, "sft") - f(r_, "nllb_3p3b") for r_ in rr])
    ax.hlines(y, diff_s, diff_m, color="#bbbbbb", lw=1.0)
    ax.scatter(diff_s, y, s=18, facecolor="white", edgecolor=GREY, label="SFT - NLLB-3.3B", zorder=3)
    for yi, r_, dm in zip(y, rr, diff_m):
        iso = r_["iso"]; rw = iso in reward_nllb
        ax.scatter(dm, yi, s=30, marker="D" if rw else "o", color=VERM if rw else BLUE, zorder=4)
    ax.scatter([], [], s=30, color=BLUE, label="mixed best - NLLB-3.3B")
    ax.scatter([], [], s=30, marker="D", color=VERM, label="same, language with an NLLB pseudo-reference term in the RL reward")
    ax.axvline(0, color=BLACK, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([r_["iso"] + ("*" if r_["iso"] == "yor" else "") for r_ in rr], fontsize=6.5)
    ax.set_xlabel("chrF++ difference to the published NLLB-3.3B (FLORES-200 devtest, beam) per language")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title("(c) Per language: 7 of 43 at or above NLLB-3.3B, all in the lowest tertile\n* yor: the published NLLB value is anomalously low (ledger N6)", loc="left", fontsize=8.5)
    fig.suptitle("RL models against NLLB-200 (ours: greedy, FLORES+ devtest, 1 seed; theirs: published, beam, FLORES-200; languages with no published NLLB row excluded)",
                 fontsize=9, y=0.995)
    save(fig, "fig4_nllb_43")


# --------------------------------------------------------------------------------------------------------------------
# Figure 5a: science, sentence level
def fig5a():
    sp = p(DATA, "eval_science_rl_2026-11-04", "sci_pairs.json")
    tb = p(DATA, "eval_science_sft_2026-11-04", "tables.md")
    if not need(sp):
        return
    P_ = jload(sp)["pairs"]
    def get(pair, m):
        v = P_[pair][m]
        return v["a"], v["b"], v["diff"], v["lo"], v["hi"]
    models = ["mixed\nbest\n(a=0)", "science\nSFT\n(a=0.5)", "sci RL\nbest\n(step 75)", "sci RL\nfinal\n(step 160)"]
    metrics = [("chrf", "chrF (plain), independent", GREEN), ("afri", "AfriCOMET-STL, held out", BLUE), ("qe", "SSA-COMET-QE, reward family", VERM)]
    absv = {}
    for m, _, _ in metrics:
        a0 = get("alpha0->alpha05/test", m)[0]; a05 = get("alpha0->alpha05/test", m)[1]
        rb = get("alpha05->rl_best/test", m)[1]; rf = get("alpha05->rl_final/test", m)[1]
        absv[m] = [a0, a05, rb, rf]
    old = None
    if os.path.exists(tb):
        t2 = parse_md_table(tb, "## Table 2.")
        if t2:
            for r in t2:
                if r[0].startswith("old 270M RL-final"):
                    old = num(r[-1])
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.25], "wspace": 0.38})
    for ax, (m, lab, col) in zip(axes[:3], metrics):
        v = absv[m]
        ax.plot(range(4), v, marker="o", color=col, lw=1.2)
        for i, x in enumerate(v):
            ax.annotate(f"{x:.4g}" if m != "chrf" else f"{x:.2f}", (i, x), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        if m == "chrf" and old:
            ax.axhline(old, color=GREY, ls="--", lw=1); ax.text(3.1, old, f"old 270M RL-final\n({old:.1f}, other code state)", fontsize=6.5, color=GREY, va="center", ha="right")
        ax.set_xticks(range(4)); ax.set_xticklabels(models, fontsize=6.8); ax.set_xlim(-0.4, 3.4)
        ax.set_title(lab, fontsize=8.5)
        rng = max(v) - min(v); ax.set_ylim(min(v) - 0.6 * rng - (0.5 if m == "chrf" else 0), max(v) + 0.5 * rng + (1.2 if m == "chrf" and old and old > max(v) else 0))
        if m == "chrf" and old:
            ax.set_ylim(min(v) - 0.6, max(old, max(v)) + 0.6)
    axes[0].set_ylabel("AfriScience-MT TEST, sentence level,\nmacro over 6 languages (970 sentences each)")
    ax = axes[3]
    pairs = [("alpha0->alpha05/test", "science SFT\n- alpha 0", PURPLE), ("alpha05->rl_best/test", "RL best\n- SFT(0.5)", SKY), ("alpha05->rl_final/test", "RL final\n- SFT(0.5)", BLUE)]
    scale = {"chrf": 1.0, "afri": 100.0, "qe": 100.0}
    ylab = {"chrf": "chrF", "afri": "AfriCOMET x100", "qe": "SSA-QE x100 (reward family)"}
    w = 0.26
    for pi, (pk, plab, col) in enumerate(pairs):
        for mi, (m, _, _) in enumerate(metrics):
            a, b, d, lo, hi = get(pk, m)
            s = scale[m]
            ax.bar(mi + (pi - 1) * w, d * s, w * 0.92, color=col, label=plab.replace("\n", " ") if mi == 0 else None,
                   yerr=[[(d - lo) * s], [(hi - d) * s]], capsize=2, error_kw={"lw": 0.8})
    ax.axhline(0, color=BLACK, lw=0.7)
    ax.set_xticks(range(3)); ax.set_xticklabels(["chrF", "AfriCOMET\nx100", "SSA-QE x100\n(reward family)"], fontsize=7)
    ax.set_ylabel("paired difference (95% bootstrap over sentences)"); ax.legend(fontsize=6.5, loc="upper right")
    ax.set_title("Paired differences, TEST", fontsize=8.5)
    fig.suptitle("Science stage, sentence level (1 seed; 6 languages: amh hau lug nso yor zul). SSA-COMET-QE is in the reward family of the science RL: read it next to chrF and held-out AfriCOMET.", fontsize=8.5, y=1.03)
    save(fig, "fig5a_science_sentence")


# Figure 5b: science, document level
def fig5b():
    dt = p(DATA, "eval_science_rl_2026-11-04", "doc_tables.md")
    if not need(dt):
        return
    tab = parse_md_table(dt, "## Table 1.")
    if not tab:
        print("SKIP fig5b (table not found)"); return
    hdr = tab[0]; rows = tab[1:]
    ix = {h.replace("*", ""): i for i, h in enumerate(hdr)}
    sel = [("385M start (alpha 0, no science SFT)", "start\n(a=0)"), ("385M science SFT (alpha 0.5)", "science\nSFT"),
           ("385M science RL (best)", "sci RL\nbest"), ("385M science RL (final)", "sci RL\nfinal"),
           ("old 270M RL-final (rl2c, first research)", "old 270M\nRL-final")]
    cols = [GREY, PURPLE, SKY, BLUE, ORANGE]
    def val(model, mode, col):
        for r in rows:
            if r[0] == model and r[1] == mode:
                return num(r[ix[col]]) if r[ix[col]] not in ("-", "") else float("nan")
        return float("nan")
    panels = [("whole", "finished", "(a) Whole document: finished share", (0, 1.05)), ("whole", "chrF", "(b) Whole document: chrF", None),
              ("chunked", "SSA-COMET-QE window", "(c) Chunked: SSA-COMET-QE windows\n(reward family; axis not at zero)", None), ("chunked", "chrF", "(d) Chunked: chrF (axis not at zero)", None)]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.7), gridspec_kw={"wspace": 0.3})
    for ax, (mode, col, title, yl) in zip(axes, panels):
        v = [val(m, mode, col) for m, _ in sel]
        if mode == "chunked":  # non-zero baseline: dots, not bars
            for i, x in enumerate(v):
                ax.vlines(i, min(v) - 0.1 * (max(v) - min(v)), x, color=cols[i], lw=1.5); ax.scatter(i, x, color=cols[i], s=40, zorder=3)
                ax.annotate(f"{x:.3f}" if x < 2 else f"{x:.1f}", (i, x), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        else:
            ax.bar(range(len(sel)), v, color=cols, width=0.7)
            for i, x in enumerate(v):
                ax.text(i, x, f"{x:.3f}" if x < 2 else f"{x:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(sel))); ax.set_xticklabels([s[1] for s in sel], fontsize=7, rotation=0)
        ax.set_title(title, fontsize=8.5, loc="left")
        if yl: ax.set_ylim(*yl)
        else:
            lo = min(v); hi = max(v)
            ax.set_ylim(0, hi * 1.12) if mode == "whole" else ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo) + 0.001)
    fig.suptitle("Science stage, document level: 168 test documents (28 papers x 6 languages), 1 seed, greedy; whole = 4,096 new tokens in one generation, chunked = sentence by sentence + best-of-4 repair\n"
                 "No bootstrap (28 papers per language). Concatenated-document COMET sees only 512 tokens and is not shown. Old 270M = first research's RL-final in the same harness.", fontsize=8, y=1.06)
    save(fig, "fig5b_science_document")


# --------------------------------------------------------------------------------------------------------------------
# Figure 6: weak-group run
def fig6():
    bl = p(DATA, "eval_weak_group_2026-11-03", "scores", "bootstrap_by_language.csv")
    mt = p(ARCH, "clean_rerun_misc", "runs_385m_weak", "mixed", "metrics.jsonl")
    if not need(bl):
        return
    targets = ["mos", "fon", "kbp", "bam", "wol", "ewe", "twi", "dyu"]
    rows = [r for r in rows_csv(bl) if r["pair"] == "weakfinal:mixedbest" and r["benchmark"] == "flores_plus" and r["subset"] == "all"]
    rows.sort(key=lambda r: float(r["d_chrf"]))
    fig = plt.figure(figsize=(11.5, 8.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.32)
    ax = fig.add_subplot(gs[0]); y = np.arange(len(rows))
    for yi, r in zip(y, rows):
        t = r["iso"] in targets; col = VERM if t else GREY
        d, lo, hi = float(r["d_chrf"]), float(r["d_chrf_lo"]), float(r["d_chrf_hi"])
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.2); ax.scatter(d, yi, s=22, color=col, zorder=3, marker="D" if t else "o")
    ax.axvline(0, color=BLACK, lw=0.8); ax.axvline(1.0, color=GREEN, ls="--", lw=1)
    ax.text(1.02, 3, "pre-registered rule:\n+1.0 on the 8 targets\n(mean of targets, CI > 0)", fontsize=7, color=GREEN, va="center")
    ax.set_yticks(y); ax.set_yticklabels([r["iso"] for r in rows], fontsize=6.5)
    ax.set_xlabel("chrF++ change, weak-group final (step 85) minus its start (mixed best), FLORES+ devtest, 95% bootstrap over sentences")
    ax.set_title("(a) Per language (n = 54; red diamonds = the 8 targets; mean on targets +0.03 [-0.06, +0.12],\nother languages -0.09 [-0.13, -0.05]; 1 seed, no matched control)", loc="left", fontsize=8.5)
    ax2 = fig.add_subplot(gs[1])
    if os.path.exists(mt):
        dev = {}
        for l in open(mt):
            r = json.loads(l)
            if "bench_aux/weak_dev/chrf_macro" in r:
                dev[r["step"]] = r["bench_aux/weak_dev/chrf_macro"]
        s = sorted(dev)
        ax2.plot(s, [dev[k] for k in s], marker="o", color=VERM)
        ax2.set_xlabel("step of the weak-group run"); ax2.set_ylabel("dev chrF++ on the 8 targets' dev sentences")
        ax2.set_title("(b) Weak dev series (4 languages, axis span 1 chrF++:\nno gain beyond noise); plateau stop at step 75", loc="left", fontsize=8.5)
        ax2.axvline(75, color=GREY, ls=":", lw=1)
        mid = (min(dev.values()) + max(dev.values())) / 2; ax2.set_ylim(mid - 0.5, mid + 0.5)  # dev noise is sd 0.09-0.10: do not magnify
    save(fig, "fig6_weak_group")


# --------------------------------------------------------------------------------------------------------------------
# Figure 7: finished / loop rates, reward flag vs independent detector; English-output share
def fig7():
    ev = p(DATA, "rl_vs_sft_evidence_2026-11-04", "evidence.json")
    dg = p(DATA, "degeneration_audit_independent_2026-11-04")
    if not need(ev, p(dg, "bootstrap_rates.csv"), p(dg, "arabic_tifinagh_english_leak.csv")):
        return
    E = jload(ev)
    br = {(r["model"], r["benchmark"]): r for r in rows_csv(p(dg, "bootstrap_rates.csv"))}
    sysm = [("SFT", "sft", "sft"), ("stage A", "stageA", "stageA"), ("B1 best", "b1", "b1best"), ("mixed best", "mixed", "mixedbest")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), gridspec_kw={"width_ratios": [1.0, 1.35, 1.0], "wspace": 0.4})
    ax = axes[0]
    for i, (lab, ek, bk) in enumerate(sysm):
        fin = E["flags"][ek]["flores_plus"]["finished"]
        ax.bar(i, fin, color=SYS_COL[lab], width=0.65); ax.text(i, fin + 0.004, f"{fin:.3f}", ha="center", fontsize=7.5)
    ax.set_ylim(0.75, 1.03); ax.set_xticks(range(4)); ax.set_xticklabels([s[0] for s in sysm], fontsize=7.5)
    ax.set_ylabel("finished rate (reward's own guard)"); ax.set_title("(a) FLORES+ finished rate (54 languages; axis\nfrom 0.75; optimised by construction)", loc="left", fontsize=8.5)
    ax = axes[1]
    w = 0.38
    for bi, (b, bl) in enumerate([("flores_plus", "FLORES+"), ("ntrex", "NTREX"), ("mafand", "MAFAND")]):
        pass
    x = np.arange(4)
    for k, (flag, col, lab) in enumerate([("rw", ORANGE, "reward's loop flag"), ("ind", PURPLE, "independent detector")]):
        v = [float(br[(s[2], "flores_plus")][f"{flag}_loop"]) for s in sysm]
        lo = [float(br[(s[2], "flores_plus")][f"{flag}_lo" if flag == "rw" else "ind_lo"]) for s in sysm]
        hi = [float(br[(s[2], "flores_plus")][f"{flag}_hi" if flag == "rw" else "ind_hi"]) for s in sysm]
        ax.bar(x + (k - 0.5) * w, v, w * 0.92, color=col, label=lab, yerr=[np.array(v) - np.array(lo), np.array(hi) - np.array(v)], capsize=2, error_kw={"lw": 0.8})
        for xi, vi in zip(x, v):
            ax.text(xi + (k - 0.5) * w, vi + 0.012, f"{vi:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in sysm], fontsize=7.5); ax.set_ylabel("share of outputs flagged as looping / degenerate")
    r = br[("mixedbest", "flores_plus")]
    ax.set_title(f"(b) FLORES+ loop rate; mixed best: independent / reward\n= {float(r['ratio']):.2f} [{float(r['ratio_lo']):.2f}, {float(r['ratio_hi']):.2f}]; 95% bootstrap over sentences", loc="left", fontsize=8.5)
    ax.legend(fontsize=7.5)
    ax = axes[2]
    rows = rows_csv(p(dg, "arabic_tifinagh_english_leak.csv"))
    order = ["sft", "stageA", "b1best", "mixedbest"]
    labs = ["SFT", "stage A", "B1 best", "mixed best"]
    for iso, col in zip(["aeb", "ary", "arz", "zgh"], [VERM, ORANGE, BLUE, GREEN]):
        d = {r_["model"]: float(r_["nllb_eng"]) for r_ in rows if r_["iso"] == iso}
        ax.plot(range(4), [d.get(k, np.nan) for k in order], marker="o", color=col, label=iso)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=7.2); ax.set_ylabel("share of outputs identified as English")
    ax.set_title("(c) English output in aeb / ary / arz / zgh\n(not in the RL pool; NLLB-218 LID)", loc="left", fontsize=8.5); ax.legend(fontsize=7.5)
    fig.suptitle("Finishing, looping and English output (1 seed per model; FLORES+ devtest; reward flag = is_repetitive / is_self_repetitive; independent = lzma residual + repeated 8-grams, thresholds at 0.1% of human references)", fontsize=8.2, y=1.04)
    save(fig, "fig7_loops_finishing")


# --------------------------------------------------------------------------------------------------------------------
# Figure 8: engine parity (E-09)
def fig8():
    pb = p(DATA, "e09_engine_parity_2026-11-04", "parity_bootstrap.json"); ps = p(DATA, "e09_engine_parity_2026-11-04", "parity_summary.json")
    if not need(pb, ps):
        return
    B, S = jload(pb), jload(ps)
    eng = [("hf", "HF bf16"), ("vllm", "vLLM bf16"), ("hf32", "HF fp32"), ("vllm32", "vLLM fp32")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.2, 1.2, 1.0], "wspace": 0.6})
    ax = axes[0]; w = 0.38
    for k, (m, col) in enumerate([("sft", GREY), ("rl", BLUE)]):
        v = [B[m]["point"][e] for e, _ in eng]
        ax.plot(range(4), v, marker="o", color=col, lw=1.0, label="SFT" if m == "sft" else "RL (mixed best)")
        for i, x in enumerate(v):
            ax.annotate(f"{x:.2f}", (i, x), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5)
    ax.set_ylim(34.5, 41); ax.set_xticks(range(4)); ax.set_xticklabels([e[1] for e in eng], fontsize=7.5); ax.set_xlim(-0.4, 3.4)
    ax.set_ylabel("chrF++, language macro"); ax.legend(fontsize=7.5, loc="lower left")
    ax.set_title("(a) Quality by engine and precision\n(240 sentences, 10 languages, greedy, 256 tokens)", loc="left", fontsize=8.5)
    ax = axes[1]
    pairs = [("hf32 vs vllm32", "fp32: HF vs vLLM"), ("hf vs vllm", "bf16: HF vs vLLM"), ("hf vs hf32", "HF: bf16 vs fp32"), ("vllm vs vllm32", "vLLM: bf16 vs fp32")]
    for k, (m, col) in enumerate([("sft", GREY), ("rl", BLUE)]):
        v = [S[m]["identical"][pk] for pk, _ in pairs]
        ax.barh(np.arange(4) + (k - 0.5) * w, v, w * 0.92, color=col)
        for i, x in enumerate(v):
            ax.text(x + 0.01, i + (k - 0.5) * w, f"{x:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(range(4)); ax.set_yticklabels([p_[1] for p_ in pairs], fontsize=7.5); ax.invert_yaxis(); ax.set_xlim(0, 1.15)
    ax.set_xlabel("share of the 240 outputs that are identical"); ax.set_title("(b) Identical outputs: engines agree in fp32, not in bf16", loc="left", fontsize=8.5)
    ax = axes[2]
    items = [("vllm32 minus vllm", "vLLM"), ("hf32 minus hf", "HF")]
    for k, (m, col) in enumerate([("sft", GREY), ("rl", BLUE)]):
        for i, (key, lab) in enumerate(items):
            pt, lo, hi = B[m][key]
            ax.errorbar(i + (k - 0.5) * 0.25, pt, yerr=[[pt - lo], [hi - pt]], fmt="o", color=col, capsize=3)
    ax.axhline(0, color=BLACK, lw=0.7); ax.set_xticks(range(2)); ax.set_xticklabels(["vLLM", "HF"]); ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("fp32 minus bf16, chrF++ (95% paired bootstrap)")
    ax.set_title("(c) Precision effect: SFT (grey) vs RL (blue)\n240 sentences; full-set SFT in Figure 11 (E-09b)", loc="left", fontsize=8.5)
    fig.suptitle("Engine parity (E-09): headline numbers are vLLM bf16; 1 seed per model; 240-sentence set", fontsize=9, y=1.03)
    save(fig, "fig8_engine_parity")


# --------------------------------------------------------------------------------------------------------------------
# PENDING placeholders.  Each reads its data file as soon as it exists.
def placeholder(stem, title, note, files):
    fig, ax = plt.subplots(figsize=(7.5, 3.0)); ax.axis("off")
    ax.text(0.5, 0.78, "[PENDING]", ha="center", fontsize=26, color=VERM, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.55, title, ha="center", fontsize=10.5, transform=ax.transAxes)
    ax.text(0.5, 0.40, note, ha="center", fontsize=8.5, transform=ax.transAxes, color=GREY)
    ax.text(0.5, 0.05, "will read: " + "\n".join(files), ha="center", va="bottom", fontsize=6.8, family="monospace", transform=ax.transAxes)
    save(fig, stem)


def first(globpat):
    m = sorted(glob.glob(p(DATA, globpat)))
    return m[0] if m else None


def pending_e15():
    """E-15 long-document RL (data in since 2026-11-04): figure fig9_long_document_rl from
    reports/data/eval_long_rl_2026-11-04/doc_tables.md (Table 1) and doc_pairs.json (paired document bootstrap).
    If either file is missing the PENDING placeholder pending_e15_long_document is drawn instead."""
    d = p(DATA, "eval_long_rl_2026-11-04")
    dt, dp = p(d, "doc_tables.md"), p(d, "doc_pairs.json")
    pats = ["reports/data/eval_long_rl_2026-11-04/doc_tables.md", "reports/data/eval_long_rl_2026-11-04/doc_pairs.json"]
    if not (os.path.exists(dt) and os.path.exists(dp)):
        placeholder("pending_e15_long_document", "Long-document RL (E-15): whole-document finished / loop / chrF / tail coverage",
                    "evaluation files not found; figure is redrawn from the files below", pats)
        return
    tab = parse_md_table(dt, "## Table 1.")
    hdr = tab[0]; rows = tab[1:]
    ix = {h.replace("*", ""): i for i, h in enumerate(hdr)}
    sel = [("385M science RL (final)", "science RL\nfinal (start)", BLUE), ("385M long-document RL (best, step 50)", "long RL best\n(step 50)", GREEN),
           ("385M long-document RL (final, step 80)", "long RL final\n(step 80)", SKY), ("old 270M RL-final (rl2c, first research)", "old 270M\nRL-final", ORANGE)]
    def val(model, mode, col):
        for r in rows:
            if r[0] == model and r[1] == mode:
                return num(r[ix[col]]) if r[ix[col]] not in ("-", "") else float("nan")
        return float("nan")
    P_ = jload(dp)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), gridspec_kw={"hspace": 0.55, "wspace": 0.32})
    for ax, (col, lab, fmt) in zip(axes[0], [("finished", "finished share", "{:.2f}"), ("loop", "loop rate (reward's flag)", "{:.2f}"), ("chrF", "chrF", "{:.1f}")]):
        v = [val(m, "whole", col) for m, _, _ in sel]
        ax.bar(range(4), v, color=[c for _, _, c in sel], width=0.7)
        for i, x in enumerate(v):
            ax.text(i, x, fmt.format(x), ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l, _ in sel], fontsize=6.8)
        ax.set_title(f"({'abc'[list(axes[0]).index(ax)]}) Whole-document mode: {lab}", fontsize=8.5, loc="left"); ax.set_ylim(0, max(v) * 1.15)
    ax = axes[1][0]
    pairs = [("sci_final->long_best", "long best - start", GREEN), ("sci_final->long_final", "long final - start", SKY), ("long_best->long_final", "final - best", GREY)]
    mets = [("finished_whole", "finished share"), ("loop_whole", "loop rate")]
    w = 0.26
    for pi, (pk, pl, col) in enumerate(pairs):
        for mi, (mk, _) in enumerate(mets):
            r = P_[pk][mk]
            ax.errorbar(mi + (pi - 1) * w, r["diff"], yerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt="o", color=col, capsize=3, label=pl if mi == 0 else None)
    ax.axhline(0, color=BLACK, lw=0.7); ax.set_xticks(range(2)); ax.set_xticklabels([m[1] for m in mets]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("paired difference (share of documents)"); ax.legend(fontsize=6.8, loc="lower left")
    ax.set_title("(d) Whole mode: paired differences\n(95% bootstrap over 168 documents)", fontsize=8.5, loc="left")
    ax = axes[1][1]
    mets = [("chrf_whole", "whole"), ("chrf_chunked", "chunked")]
    for pi, (pk, pl, col) in enumerate(pairs):
        for mi, (mk, _) in enumerate(mets):
            r = P_[pk][mk]
            ax.errorbar(mi + (pi - 1) * w, r["diff"], yerr=[[r["diff"] - r["lo"]], [r["hi"] - r["diff"]]], fmt="o", color=col, capsize=3)
    ax.axhline(0, color=BLACK, lw=0.7); ax.set_xticks(range(2)); ax.set_xticklabels([m[1] for m in mets]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("chrF difference"); ax.set_title("(e) chrF: paired differences\n(best vs start: whole +3.4, chunked +0.13)", fontsize=8.5, loc="left")
    ax = axes[1][2]
    for si, (m, l, c) in enumerate(sel):
        tw, tc = val(m, "whole", "tail coverage"), val(m, "chunked", "tail coverage")
        ax.bar(si - 0.18, tw, 0.34, color=c); ax.bar(si + 0.18, tc, 0.34, color=c, alpha=0.45, hatch="//")
        ax.text(si - 0.18, tw, f"{tw:.2f}", ha="center", va="bottom", fontsize=6.8); ax.text(si + 0.18, tc, f"{tc:.2f}", ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l, _ in sel], fontsize=6.8); ax.set_ylim(0, 0.8)
    ax.set_title("(f) Tail coverage: whole (solid) vs chunked (hatched)\nshare of the last third of the paper that is translated", fontsize=8.5, loc="left")
    sq = [val(m, "chunked", "SSA-COMET-QE window") for m, _, _ in sel]
    fig.suptitle("Long-document RL (E-15), AfriScience-MT TEST, 168 documents, greedy, one seed. Whole = 4,096 new tokens in one generation. Chunked window SSA-COMET-QE (reward family): "
                 + " / ".join(f"{x:.3f}" for x in sq) + " (same order as the bars).", fontsize=8, y=0.97)
    save(fig, "fig9_long_document_rl")


def pending_e05():
    """E-05 (data in since 2026-11-04): fig12_ssa_all_stages from reports/data/e05_ssa_all_stages_2026-11-04/tables.md
    (first table, AfriScience-MT TEST). Placeholder pending_e05_ssa_qe_stages if the file is missing."""
    f = p(DATA, "e05_ssa_all_stages_2026-11-04", "tables.md")
    if not os.path.exists(f):
        placeholder("pending_e05_ssa_qe_stages", "SSA-COMET-QE, AfriCOMET and chrF for all stages (E-05)", "tables.md not found",
                    ["reports/data/e05_ssa_all_stages_2026-11-04/tables.md"])
        return
    tab = parse_md_table(f, "## AfriScience-MT TEST")
    hdr = [h.replace("*", "") for h in tab[0]]; rows = {r[0]: r for r in tab[1:]}
    chain = [("SFT (385M, start of RL)", "SFT"), ("RL stage A (final)", "RL\nA"), ("RL B1 best", "RL\nB1"), ("RL mixed best (= science SFT alpha 0)", "RL\nmixed"),
             ("science-SFT alpha 0.5 (science RL start)", "science\nSFT 0.5"), ("science RL best (step 75)", "sci RL\nbest"), ("science RL final (step 160)", "sci RL\nfinal"),
             ("long-document RL best (step 50)", "long RL\nbest"), ("long-document RL final (step 80)", "long RL\nfinal")]
    chain = [(k, l) for k, l in chain if k in rows]
    cols = [("chrF", "chrF (plain; independent of the reward,\nbut trained on the same domain via train pairs)", GREEN),
            ("AfriCOMET-STL (held out)", "AfriCOMET-STL (held out)", BLUE), ("SSA-COMET-QE", "SSA-COMET-QE (reward family\nfor every RL row)", VERM)]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), gridspec_kw={"wspace": 0.3})
    for ax, (c, title, col) in zip(axes, cols):
        v = [num(rows[k][hdr.index(c)]) for k, _ in chain]
        ax.plot(range(len(v)), v, marker="o", color=col, lw=1.2)
        for i, x in enumerate(v):
            ax.annotate(f"{x:.2f}" if c == "chrF" else f"{x:.4f}", (i, x), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6)
        ax.set_xticks(range(len(v))); ax.set_xticklabels([l for _, l in chain], fontsize=6.3); ax.set_title(title, fontsize=8.5)
        r = max(v) - min(v); ax.set_ylim(min(v) - 0.1 * r, max(v) + 0.15 * r)
    fig.suptitle("All stages on AfriScience-MT TEST, sentence level (970 sentences x 6 languages: amh hau lug nso yor zul; greedy; one seed per stage; no interval on absolute values, paired intervals are in Figure 5a). Weak-group run and the other alphas are in the report.", fontsize=7.8, y=1.04)
    save(fig, "fig12_ssa_all_stages")


def pending_e09b():
    """E-09b (data in since 2026-11-04): fig11_fp32_full_set from reports/data/e09b_fp32_2026-11-04/evidence_e09b.json
    (SFT in fp32 on the full headline set; RL fp32 not run). Placeholder pending_e09b_fp32_fullset if missing."""
    f = p(DATA, "e09b_fp32_2026-11-04", "evidence_e09b.json")
    if not os.path.exists(f):
        placeholder("pending_e09b_fp32_fullset", "fp32 against bf16 on the full evaluation set (E-09b)", "evidence_e09b.json not found",
                    ["reports/data/e09b_fp32_2026-11-04/evidence_e09b.json"])
        return
    D = jload(f)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"wspace": 0.3})
    ax = axes[0]
    keys = [("flores_plus", "FLORES+\n(54)"), ("ntrex", "NTREX\n(24)"), ("mafand", "MAFAND\n(15)"), ("all_cells", "all 93\ncells")]
    for i, (k, lab) in enumerate(keys):
        pp = D["macro"]["all"][k]["pp"]
        ax.errorbar(i, pp[2], yerr=[[pp[2] - pp[3]], [pp[4] - pp[2]]], fmt="o", color=BLACK, capsize=4)
        ax.annotate(f"{pp[2]:+.2f}", (i, pp[2]), textcoords="offset points", xytext=(10, 4), fontsize=7)
    ax.axhline(0, color=BLACK, lw=0.7); ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in keys], fontsize=7.5); ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("SFT: fp32 minus bf16, chrF++ (language macro)")
    ax.set_title("(a) fp32 does not help the SFT on the full set\n(122,585 rows, vLLM, greedy, 768 tokens; 95% paired bootstrap)", loc="left", fontsize=8.5)
    ax = axes[1]
    w = 0.36
    for k, (key, lab, col) in enumerate([("sft->mixed", "RL gain over the bf16 SFT", BLUE), ("sft_fp32->mixed", "RL gain over the fp32 SFT", ORANGE)]):
        for i, (b, bl) in enumerate(keys[:3]):
            pp = D["stages"][key][b]["pp"]
            ax.errorbar(i + (k - 0.5) * w, pp[2], yerr=[[pp[2] - pp[3]], [pp[4] - pp[2]]], fmt="o", color=col, capsize=4, label=lab if i == 0 else None)
            ax.annotate(f"{pp[2]:+.2f}", (i + (k - 0.5) * w, pp[2]), textcoords="offset points", xytext=(8, -3), fontsize=6.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in keys[:3]], fontsize=7.5); ax.set_xlim(-0.5, 2.5); ax.set_ylim(3.5, 10)
    ax.set_ylabel("RL (mixed best, bf16) minus SFT, chrF++"); ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("(b) RL gain over the bf16 and the fp32 SFT\n(RL itself was not re-run in fp32)", loc="left", fontsize=8.5)
    fig.suptitle("E-09b: SFT in fp32 on the full headline set, one seed; AfriCOMET of the fp32 run was not scored", fontsize=9, y=1.03)
    save(fig, "fig11_fp32_full_set")


def pending_e13():
    """E-13 (data in since 2026-11-04): fig10_old270m_same_harness from
    reports/data/e13_old270m_same_harness_2026-11-04/evidence_e13.json (paired bootstrap, rows clean for both generations).
    Placeholder pending_e13_old270m if the file is missing."""
    f = p(DATA, "e13_old270m_same_harness_2026-11-04", "evidence_e13.json")
    if not os.path.exists(f):
        placeholder("pending_e13_old270m", "Old 270M SFT and RL-final re-run in the 385M harness (E-13)",
                    "evidence_e13.json not found", ["reports/data/e13_old270m_same_harness_2026-11-04/evidence_e13.json"])
        return
    D = jload(f)
    B = [("flores_plus", "FLORES+\n(45 languages,\n48 sentences each)"), ("ntrex", "NTREX\n(24 languages)"), ("mafand", "MAFAND\n(15 languages)")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), gridspec_kw={"width_ratios": [1.5, 1.2, 1.2], "wspace": 0.32})
    ax = axes[0]
    series = [("old 270M SFT", lambda b: D["stages"]["old_sft->sft"][b]["pp"][0], GREY, "o"), ("385M SFT", lambda b: D["stages"]["old_sft->sft"][b]["pp"][1], BLACK, "s"),
              ("old 270M RL (rl2c)", lambda b: D["stages"]["old_rl2c->mixed"][b]["pp"][0], ORANGE, "o"), ("385M RL (mixed best)", lambda b: D["stages"]["old_rl2c->mixed"][b]["pp"][1], BLUE, "s")]
    for si, (lab, fn, col, mk) in enumerate(series):
        xs = np.arange(3) + (si - 1.5) * 0.16
        v = [fn(b) for b, _ in B]
        ax.scatter(xs, v, color=col, marker=mk, s=40, label=lab, zorder=3)
        for x, y in zip(xs, v):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=6.5)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in B], fontsize=7.5); ax.set_ylabel("chrF++, language macro")
    ax.legend(fontsize=7, loc="upper left"); ax.set_title("(a) chrF++ on rows clean for both generations\n(1 seed per model; 95% CIs in panel b)", loc="left", fontsize=8.5)
    ax = axes[1]
    for k, (key, lab, col) in enumerate([("old_sft->sft", "385M SFT - old SFT", BLACK), ("old_rl2c->mixed", "385M RL - old RL", BLUE)]):
        for bi, (b, _) in enumerate(B):
            pp = D["stages"][key][b]["pp"]
            ax.errorbar(bi + (k - 0.5) * 0.25, pp[2], yerr=[[pp[2] - pp[3]], [pp[4] - pp[2]]], fmt="o", color=col, capsize=3, label=lab if bi == 0 else None)
    ax.axhline(0, color=BLACK, lw=0.7); ax.set_xticks(range(3)); ax.set_xticklabels([b.split("\n")[0] for _, b in B], fontsize=7.5); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("chrF++ difference (95% paired bootstrap)"); ax.legend(fontsize=7, loc="upper right")
    ax.set_title("(b) Paired differences, chrF++", loc="left", fontsize=8.5)
    ax = axes[2]
    for k, (key, lab, col) in enumerate([("old_sft->sft", "385M SFT - old SFT", BLACK), ("old_rl2c->mixed", "385M RL - old RL", BLUE)]):
        for bi, (b, _) in enumerate(B):
            c = D["stages"][key][b]["com"]
            ax.errorbar(bi + (k - 0.5) * 0.25, c[2], yerr=[[c[2] - c[3]], [c[4] - c[2]]], fmt="o", color=col, capsize=3)
    ax.axhline(0, color=BLACK, lw=0.7); ax.set_xticks(range(3)); ax.set_xticklabels([b.split("\n")[0] for _, b in B], fontsize=7.5); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("held-out AfriCOMET-STL difference"); ax.set_title("(c) Paired differences, AfriCOMET\n(languages with a score: 40 / 21 / 15)", loc="left", fontsize=8.5)
    fig.suptitle("First research's 270M models re-run in the 385M harness (E-13): greedy, 768 tokens; only rows clean for both generations (FLORES+ keeps 2,160 of 45,540 rows)", fontsize=8.5, y=1.03)
    save(fig, "fig10_old270m_same_harness")


def main():
    only = set(sys.argv[1:])
    jobs = [("fig15", fig15), ("fig16", fig16), ("fig14", fig14), ("fig13", fig13), ("fig1", fig1), ("fig2", fig2), ("fig3", fig3), ("fig4", fig4), ("fig5a", fig5a), ("fig5b", fig5b), ("fig6", fig6), ("fig7", fig7),
            ("fig8", fig8), ("e15", pending_e15), ("e05", pending_e05), ("e09b", pending_e09b), ("e13", pending_e13)]
    for name, fn in jobs:
        if only and name not in only:
            continue
        try:
            fn()
        except Exception as ex:  # keep going: one broken input must not block the other figures
            print("FAILED", name, repr(ex))
            raise
    print("done:", ", ".join(MADE))


if __name__ == "__main__":
    main()
