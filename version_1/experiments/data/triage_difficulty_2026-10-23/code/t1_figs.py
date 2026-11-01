from t1load import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
A = pd.read_pickle(f"{DATA}/A.pkl")
P = A[(A.n_pair_seen == 0)].copy()
FIG = OUT + "/figures"; os.makedirs(FIG, exist_ok=True)
COL = {"rl": "#0072B2", "nllb3b3": "#D55E00", "sft": "#7f7f7f", "nllb600m": "#009E73", "a2w": "#56B4E9", "gap": "#CC79A7"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})
def dm(d, c): return d[c] - d.groupby(["b", "k", "iso"])[c].transform("mean")
for m in ["rl", "sft", "nllb3b3"]:
    for met in ("chrf", "comet"): P[f"res_{met}_{m}"] = dm(P, f"{met}_{m}")
for met in ("chrf", "comet"):
    P[f"gap_{met}"] = P[f"{met}_rl"] - P[f"{met}_nllb3b3"]; P[f"res_gap_{met}"] = dm(P, f"gap_{met}")
PE = P[~P.efi]
EDGES = [0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
def binned(d, col, edges=EDGES, minc=15):
    d = d.dropna(subset=[col]).copy(); d["sb"] = pd.cut(d.score, edges)
    out = []
    for sb, g in d.groupby("sb", observed=True):
        if g.ekey.nunique() < minc: continue
        e = boot_mean(g, col, B=300)
        out.append((sb.mid, *e[:3], g.ekey.nunique()))
    return pd.DataFrame(out, columns=["x", "m", "lo", "hi", "nch"])
def band_lines(ax, ymax=None):
    for v in (3.0, 4.0, 5.5): ax.axvline(v, color="k", lw=0.6, ls=":", alpha=0.6)
# F1 distribution -----------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
for ax, b in zip(axs.flat, BENCHES):
    for k, c in ((1, "#0072B2"), (3, "#E69F00"), (5, "#009E73")):
        u = A[(A.b == b) & (A.k == k)].drop_duplicates("ekey")
        if len(u): ax.hist(u.score, bins=np.arange(1, 6.01, 0.1), alpha=0.55, color=c, density=True, label=f"k={k} (n={len(u)})")
    band_lines(ax); ax.set_title(BNAME[b]); ax.legend(frameon=False, fontsize=7)
for ax in axs[1]: ax.set_xlabel("difficulty score (weighted mean of 5 ratings)")
fig.suptitle("Difficulty-score distribution of English chunks (dotted: owner's band edges 3.0 / 4.0 / 5.5; no chunk reaches 'very hard', two reach 'hard')", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_score_distribution.png"); plt.close(fig)
# F2 binned quality vs score ---------------------------------------------------------------------------------------------------------
sets = [("FLORES+ k=1", PE[(PE.b == "flores_plus") & (PE.k == 1)]), ("NTREX k=1", PE[(PE.b == "ntrex") & (PE.k == 1)]), ("MAFAND k=1", PE[(PE.b == "mafand") & (PE.k == 1)]), ("all benchmarks k=1", PE[PE.k == 1]), ("all benchmarks k=3", PE[PE.k == 3]), ("all benchmarks k=5", PE[PE.k == 5])]
for met, name, unit in (("comet", "AfriCOMET", "AfriCOMET relative to the language's own mean"), ("chrf", "chrF++", "chrF++ relative to the language's own mean")):
    fig, axs = plt.subplots(1, 6, figsize=(17, 3.4), sharey=True)
    for ax, (t, d) in zip(axs, sets):
        for m in ("rl", "nllb3b3", "sft"):
            bb = binned(d, f"res_{met}_{m}")
            ax.plot(bb.x, bb.m, "-o", ms=3, color=COL[m], label={"rl": "RL-final", "nllb3b3": "NLLB-3.3B", "sft": "SFT-final"}[m], lw=1.4)
            ax.fill_between(bb.x, bb.lo, bb.hi, color=COL[m], alpha=0.15, lw=0)
        band_lines(ax); ax.axhline(0, color="k", lw=0.5); ax.set_title(t, fontsize=9); ax.set_xlabel("difficulty score")
    axs[0].set_ylabel(unit, fontsize=8); axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_{name.replace("+","pp")}_vs_score.png"); plt.close(fig)
# F3 gap by group -----------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
for ax, (met, name) in zip(axs, (("chrf", "chrF++"), ("comet", "AfriCOMET"))):
    for g, c in (("strong", "#0072B2"), ("mid", "#E69F00"), ("weak", "#D55E00")):
        d = PE[(PE.k == 1) & (PE.group == g)]
        bb = binned(d, f"gap_{met}")
        ax.plot(bb.x, bb.m, "-o", ms=3, color=c, label=f"{g} languages"); ax.fill_between(bb.x, bb.lo, bb.hi, color=c, alpha=0.15, lw=0)
    band_lines(ax); ax.axhline(0, color="k", lw=0.6); ax.set_title(f"RL-final minus NLLB-3.3B, {name} (all benchmarks, k=1)"); ax.set_xlabel("difficulty score"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_gap_to_nllb3b3_by_group.png"); plt.close(fig)
# F4 threshold scans ---------------------------------------------------------------------------------------------------------------
S = pd.read_csv(OUT + "/04b_threshold_scans.csv")
fig, axs = plt.subplots(2, 4, figsize=(15, 6))
for r, met in enumerate(("AfriCOMET", "chrF++")):
    for c_, ds in enumerate(["FLORES+ devtest", "NTREX-128", "MAFAND-MT", "pooled (all benchmarks)"]):
        ax = axs[r, c_]
        for m in ("rl", "nllb3b3"):
            s = S[(S.dataset == ds) & (S.k == 1) & (S.group == "all") & (S.model == m) & (S.metric == met)]
            if len(s) == 0: continue
            ax.plot(s.threshold, s["diff"], color=COL[m], lw=1.4, label=m); ax.fill_between(s.threshold, s.lo_sup, s.hi_sup, color=COL[m], alpha=0.15, lw=0)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(f"{ds}, {met}", fontsize=9); ax.set_xlabel("threshold s"); 
        if c_ == 0: ax.set_ylabel("mean quality(score > s) - mean quality(score <= s)", fontsize=7)
axs[0, 0].legend(frameon=False, fontsize=7)
fig.suptitle("Threshold scan (language-relative quality, k=1, simultaneous 95% bands)", fontsize=9); fig.tight_layout(); fig.savefig(f"{FIG}/fig4_threshold_scan.png"); plt.close(fig)
# F7 risk by score bin ---------------------------------------------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
d = PE[PE.k == 1].copy()
for ax, (col, t) in zip(axs, (("unfin", "not finished (hit token limit)"), ("wronglang", "wrong language / script (GlotLID)"), ("loop", "repetition loop"))):
    for m, c in (("rl", COL["rl"]), ("sft", COL["sft"])):
        dd = d.dropna(subset=[f"fin_{m}"]).copy(); dd["unfin"] = 1 - dd[f"fin_{m}"]; dd["wronglang"] = 1 - dd[f"lid_{m}"]; dd["loop"] = dd[f"loop_{m}"]
        bb = binned(dd, col); ax.plot(bb.x, 100 * bb.m, "-o", ms=3, color=c, label=m); ax.fill_between(bb.x, 100 * bb.lo, 100 * bb.hi, color=c, alpha=0.15, lw=0)
    band_lines(ax); ax.set_title(t, fontsize=9); ax.set_xlabel("difficulty score"); ax.set_ylabel("% of chunks")
axs[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG}/fig7_failure_rates_vs_score.png"); plt.close(fig)
# F8 coverage / quality of the translated set ------------------------------------------------------------------------------------------
R = pd.read_csv(OUT + "/11_translate_when_score_leq_x.csv")
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
rr = R[(R.dataset == "pooled (all benchmarks)") & (R.k == 1) & (R.group == "all") & (R.translate_if_score_leq != "no limit")].copy(); rr["x"] = rr.translate_if_score_leq.astype(float)
for ax, (met, t) in zip(axs[:2], (("AfriCOMET", "AfriCOMET of translated set"), ("chrF++", "chrF++ of translated set"))):
    for m in ("rl", "nllb3b3"):
        s = rr[(rr.model == m) & (rr.metric == met)]; ax.plot(s.x, s.mean_quality, "-o", ms=3, color=COL[m], label=m); ax.fill_between(s.x, s.ci_lo, s.ci_hi, color=COL[m], alpha=0.15, lw=0)
    ax.set_title(t, fontsize=9); ax.set_xlabel("translate if score <= x")
s = rr[(rr.model == "rl") & (rr.metric == "chrF++")]; axs[2].plot(s.x, s.coverage_pct_of_chunks, "-o", ms=3, color="k"); axs[2].set_title("share of chunks translated (%)", fontsize=9); axs[2].set_xlabel("translate if score <= x")
axs[0].legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG}/fig8_translate_if_score_leq_x.png"); plt.close(fig)
print("figs ok")
