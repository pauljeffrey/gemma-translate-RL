"""Section 3: relation of quality to the difficulty score, turning points.
Outcome = quality relative to the language's own average on that benchmark and chunk size (language-demeaned), so that language strength does not confound score.
Per dataset (benchmark, k) x language group x model x metric (and the RL - NLLB-3.3B gap):
  * Spearman correlation of score with quality (English-chunk level, mean over languages) + cluster bootstrap CI
  * linear slope per +1.0 score point (weighted LS, cluster bootstrap CI)
  * one-breakpoint hinge model (breakpoint, slopes before/after, bootstrap CI), decreasing isotonic fit
  * threshold scan: mean(quality | score > s) - mean(quality | score <= s), pointwise and simultaneous (sup-t) 95% CIs; first s with a significant drop."""
from t1load import *
from scipy import stats
A = pd.read_pickle(f"{DATA}/A.pkl")
P = A[(A.n_pair_seen == 0) & ~A.efi].copy()
B = 500
GRID = np.round(np.arange(2.5, 5.51, 0.25), 2)
def dm(d, c):
    return d[c] - d.groupby(["b", "k", "iso"])[c].transform("mean")
for m in MODELS_F:
    for met in ("chrf", "comet"):
        c = f"{met}_{m}"
        if c in P: P["res_" + c] = dm(P, c)
for met in ("chrf", "comet"):
    for m in MODELS_F:
        if m in ("nllb3b3",): continue
        c = f"{met}_{m}"; cn = f"{met}_nllb3b3"
        if c in P and cn in P:
            P["gap_" + c] = P[c] - P[cn]
            P["res_gap_" + c] = dm(P, "gap_" + c)
def chunk_tab(d, col):
    dd = d.dropna(subset=[col])
    g = dd.groupby("ekey").agg(x=("score", "first"), S=(col, "sum"), N=(col, "size"), lw=("log_words", "first"))
    return g
def boot_stats(g, B=B, seed=1):
    x = g.x.values; S = g.S.values; N = g.N.values; C = len(x)
    uv, vi = np.unique(x, return_inverse=True)
    rng = np.random.default_rng(seed)
    thetas = np.round(np.arange(np.quantile(x, 0.15), np.quantile(x, 0.85) + 1e-9, 0.1), 2)
    def fit(cnt):
        Sv = np.bincount(vi, weights=cnt * S, minlength=len(uv)); Nv = np.bincount(vi, weights=cnt * N, minlength=len(uv))
        ok = Nv > 0; xv, yv, wv = uv[ok], Sv[ok] / Nv[ok], Nv[ok]
        if len(xv) < 6: return None
        best, sse0, b0 = hinge_fit(xv, yv, wv, thetas)
        slope = b0[1]
        return slope, best, xv, yv, wv
    cnt0 = np.ones(C)
    r0 = fit(cnt0)
    if r0 is None: return None
    slope0, best0, xv, yv, wv = r0
    sl, th, b1, b2 = [], [], [], []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, C, C), minlength=C).astype(float)
        r = fit(cnt)
        if r is None: continue
        sl.append(r[0]); th.append(r[1][0]); b1.append(r[1][2]); b2.append(r[1][2] + r[1][3])
    q = lambda a: np.quantile(a, [0.025, 0.975])
    ir = iso_fit(xv, yv, wv)
    iso_curve = ir.predict(GRID)
    rho = stats.spearmanr(x, S / N)[0]
    rb = []
    for _ in range(B):
        i = rng.integers(0, C, C); rb.append(stats.spearmanr(x[i], (S / N)[i])[0])
    return {"n_chunks": C, "n_rows": int(N.sum()), "x_p05": np.quantile(x, 0.05), "x_p95": np.quantile(x, 0.95), "spearman": rho, "spearman_lo": q(rb)[0], "spearman_hi": q(rb)[1],
            "slope_per_point": slope0, "slope_lo": q(sl)[0], "slope_hi": q(sl)[1],
            "hinge_theta": best0[0], "hinge_theta_lo": q(th)[0], "hinge_theta_hi": q(th)[1], "slope_below": best0[2], "slope_below_lo": q(b1)[0], "slope_below_hi": q(b1)[1],
            "slope_above": best0[2] + best0[3], "slope_above_lo": q(b2)[0], "slope_above_hi": q(b2)[1],
            "iso_drop_2.5_to_max": float(iso_curve[0] - iso_curve[-1]), "iso_drop_3.0_to_5.0": float(ir.predict([3.0])[0] - ir.predict([5.0])[0])}
DATASETS = [("flores_plus", 1), ("flores_plus", 3), ("flores_plus", 5), ("ntrex", 1), ("ntrex", 3), ("ntrex", 5), ("mafand", 1), ("mafand", 3), ("mafand", 5), ("smol_sent", 1), ("salt", 1), ("glot_story", 1),
            ("pooled", 1), ("pooled", 3), ("pooled", 5)]
def get(b, k):
    if b == "pooled": return P[P.k == k]
    return P[(P.b == b) & (P.k == k)]
rows, scans = [], []
for (b, k) in DATASETS:
    d0 = get(b, k)
    if len(d0) == 0: continue
    for g in ["all", "strong", "mid", "weak"]:
        d1 = d0 if g == "all" else d0[d0.group == g]
        if d1.ekey.nunique() < 60: continue
        outcomes = []
        for met in ("chrf", "comet"):
            for m in ["rl", "nllb3b3", "sft"] + (["a2w", "nllb600m"] if b in ("flores_plus", "pooled") else []):
                col = f"res_{met}_{m}"
                if col in d1: outcomes.append((m, met, col))
            col = f"res_gap_{met}_rl"
            if col in d1: outcomes.append(("rl_minus_nllb3b3", met, col))
        for (m, met, col) in outcomes:
            gt = chunk_tab(d1, col)
            if len(gt) < 60: continue
            r = boot_stats(gt)
            if r is None: continue
            r.update({"dataset": BNAME.get(b, "pooled (all benchmarks)"), "k": k, "group": g, "model": m, "metric": "chrF++" if met == "chrf" else "AfriCOMET"})
            sc_ = threshold_scan(d1, col, GRID, B=300)
            r["first_sig_drop_pointwise"] = first_sig_drop(sc_, "hi"); r["first_sig_drop_simultaneous"] = first_sig_drop(sc_, "hi_sup")
            r["first_sig_rise_pointwise"] = float(sc_[sc_.ok & (sc_.lo > 0)].threshold.min()) if (sc_.ok & (sc_.lo > 0)).any() else np.nan
            r["n_scan_points_ok"] = int(sc_.ok.sum())
            rows.append(r)
            sc_.insert(0, "dataset", r["dataset"]); sc_.insert(1, "k", k); sc_.insert(2, "group", g); sc_.insert(3, "model", m); sc_.insert(4, "metric", r["metric"])
            scans.append(sc_)
        print(b, k, g, len(rows), flush=True)
tp = pd.DataFrame(rows)
cols = ["dataset", "k", "group", "model", "metric"] + [c for c in tp.columns if c not in ("dataset", "k", "group", "model", "metric")]
tp[cols].to_csv(f"{OUT}/04_turning_points.csv", index=False, float_format="%.4g")
pd.concat(scans).to_csv(f"{OUT}/04b_threshold_scans.csv", index=False, float_format="%.4g")
print(tp[["dataset", "k", "group", "model", "metric", "n_chunks", "spearman", "spearman_lo", "spearman_hi", "slope_per_point", "slope_lo", "slope_hi", "first_sig_drop_pointwise", "first_sig_drop_simultaneous"]].round(3).to_string())
