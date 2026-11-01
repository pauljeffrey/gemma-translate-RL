"""Section 1-2: distributions, quantile bands, band tables (n, mean chrF++/AfriCOMET per model, gap to NLLB-3.3B, win/tie/loss)."""
from t1load import *
import sys
A = pd.read_pickle(f"{DATA}/A.pkl")
os.makedirs(OUT, exist_ok=True)
P = A[A.n_pair_seen == 0].copy()          # primary: every sentence of the chunk is a `no_pair` sentence (reference pair unseen by SFT)
MODELS = {"flores_plus": MODELS_F, "ntrex": ["rl", "sft", "nllb3b3"], "mafand": ["rl", "sft", "nllb3b3"], "smol_sent": ["rl", "sft", "nllb3b3"], "glot_story": ["rl", "sft"], "salt": ["rl", "sft", "nllb3b3"]}
BIG = ["band_full"]

# ---------- 1. distributions -------------------------------------------------------------------------------------------------------------
rows = []
qs = [0.05, 0.10, 0.20, 0.25, 0.40, 0.50, 0.60, 0.75, 0.80, 0.90, 0.95]
for (b, k), d in A.groupby(["b", "k"]):
    u = d.drop_duplicates("ekey")                     # one row per distinct English chunk text (same text in several languages counted once)
    r = {"benchmark": BNAME[b], "k": k, "n_chunks_distinct_english": len(u), "n_chunk_language_rows": len(d), "mean_words": u.n_words.mean(),
         "mean": u.score.mean(), "sd": u.score.std(), "min": u.score.min(), "max": u.score.max(), "n_english_form": int(u.efi.sum()), "pct_english_form": 100 * u.efi.mean()}
    for q in qs: r[f"q{int(q*100):02d}"] = u.score.quantile(q)
    for bn in BAND_NAMES: r["n_" + bn.replace(" ", "_")] = int(((u.band == bn) & ~u.efi).sum())
    for bn in BAND_NAMES: r["pct_" + bn.replace(" ", "_")] = 100 * float(((u.band == bn) & ~u.efi).sum()) / len(u)
    rows.append(r)
dist = pd.DataFrame(rows); dist.to_csv(f"{OUT}/01_score_distribution_by_benchmark.csv", index=False, float_format="%.3f")
print(dist[["benchmark", "k", "n_chunks_distinct_english", "mean", "sd", "min", "max", "n_english_form"] + ["n_" + b.replace(" ", "_") for b in BAND_NAMES]].round(2).to_string())

# quantile-band proposal: cut points of the pooled (distinct English chunk) score distribution, k=1, all benchmarks, english-form excluded, next to the owner's bands
u1 = A[(A.k == 1) & ~A.efi].drop_duplicates(["b", "ekey"])
prop = []
for name, dd in [("pooled k=1 (all benchmarks)", u1)] + [(BNAME[b] + " k=1", u1[u1.b == b]) for b in BENCHES if b in u1.b.unique()]:
    for nq in (3, 4, 5, 10):
        cuts = [dd.score.quantile(i / nq) for i in range(1, nq)]
        prop.append({"distribution": name, "n_quantile_bands": nq, "n_chunks": len(dd), "cut_points": " | ".join(f"{c:.2f}" for c in cuts)})
pd.DataFrame(prop).to_csv(f"{OUT}/02_quantile_band_cutpoints.csv", index=False)
# fixed 0.5-wide bins counts
bins = [0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 7.0, 10]
u1b = u1.copy(); u1b["bin"] = pd.cut(u1b.score, bins)
(u1b.groupby(["b", "bin"]).size().unstack(0).fillna(0).astype(int)).to_csv(f"{OUT}/02b_counts_per_half_point_bin_k1.csv")

# ---------- 2. band tables ------------------------------------------------------------------------------------------------------------------
GROUPS = ["all", "strong", "mid", "weak"]
def sub(d, g): return d if g == "all" else d[d.group == g]
out = []
for b in BENCHES:
    if b not in P.b.unique(): continue
    for k in sorted(P[P.b == b].k.unique()):
        d0 = P[(P.b == b) & (P.k == k)]
        for g in GROUPS:
            d1 = sub(d0, g)
            if len(d1) == 0: continue
            for band in BAND_NAMES + ["english-form (kept in English)", "ALL (incl. english-form)"]:
                d2 = d1 if band.startswith("ALL") else d1[d1.band_full == band]
                if len(d2) < 20: continue
                for m in MODELS[b]:
                    for met in ("chrf", "comet"):
                        c = f"{met}_{m}"
                        if c not in d2 or d2[c].notna().sum() < 20: continue
                        est, lo, hi, n, nc = boot_mean(d2, c, B=500)
                        r = {"benchmark": BNAME[b], "k": k, "group": g, "band": band, "model": m, "metric": "chrF++" if met == "chrf" else "AfriCOMET", "n_rows": n, "n_chunks": nc,
                             "mean": est, "ci_lo": lo, "ci_hi": hi}
                        if m != "nllb3b3" and "chrf_nllb3b3" in d2:
                            cn = f"{met}_nllb3b3"; thr = 1.0 if met == "chrf" else 0.01
                            dd = d2.dropna(subset=[c, cn]).copy()
                            if len(dd) >= 20:
                                dd["_gap"] = dd[c] - dd[cn]
                                e2, l2, h2, n2, _ = boot_mean(dd, "_gap", B=500)
                                dd["_w"] = (dd._gap > thr).astype(float); dd["_l"] = (dd._gap < -thr).astype(float); dd["_t"] = 1 - dd._w - dd._l
                                r.update({"gap_vs_nllb3b3": e2, "gap_lo": l2, "gap_hi": h2, "n_paired": n2, "win_pct": 100 * dd._w.mean(), "tie_pct": 100 * dd._t.mean(), "loss_pct": 100 * dd._l.mean()})
                                _, wl, wh, _, _ = boot_mean(dd, "_w", B=300); r["win_lo"], r["win_hi"] = 100 * wl, 100 * wh
                        out.append(r)
bt = pd.DataFrame(out); bt.to_csv(f"{OUT}/03_band_quality_gap_winloss.csv", index=False, float_format="%.4g")
print(len(bt))
