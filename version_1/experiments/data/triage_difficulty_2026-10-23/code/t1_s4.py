"""Section 4-5: confounders (length, dimensions), partial correlations, failure prediction by dimension; agreement with NLLB-3.3B."""
from t1load import *
from scipy import stats
import statsmodels.api as sm
A = pd.read_pickle(f"{DATA}/A.pkl")
P = A[(A.n_pair_seen == 0) & ~A.efi].copy()
def dm(d, c): return d[c] - d.groupby(["b", "k", "iso"])[c].transform("mean")
for m in ["rl", "sft", "nllb3b3"]:
    for met in ("chrf", "comet"):
        c = f"{met}_{m}"; P["res_" + c] = dm(P, c)
for met in ("chrf", "comet"):
    P[f"gap_{met}"] = P[f"{met}_rl"] - P[f"{met}_nllb3b3"]; P[f"res_gap_{met}"] = dm(P, f"gap_{met}")
DIMF = [d + "_f" for d in DIMS]
DATASETS = [("flores_plus", 1), ("flores_plus", 3), ("flores_plus", 5), ("ntrex", 1), ("ntrex", 3), ("ntrex", 5), ("mafand", 1), ("mafand", 3), ("mafand", 5), ("smol_sent", 1), ("pooled", 1), ("pooled", 3), ("pooled", 5)]
def get(b, k): return P[P.k == k] if b == "pooled" else P[(P.b == b) & (P.k == k)]
def bci(vals, B=1000, seed=2):
    return np.quantile(vals, [0.025, 0.975])
def chunk_frame(d, col):
    dd = d.dropna(subset=[col])
    g = dd.groupby("ekey").agg(y=(col, "mean"), n=(col, "size"), score=("score", "first"), lw=("log_words", "first"), **{k: (k, "first") for k in DIMF}, efi_prob=("efi_prob", "first"))
    return g
def resid(y, X):
    X1 = np.c_[np.ones(len(y)), X]; return y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
def partial_r(a, b, ctrl):
    return stats.pearsonr(resid(a, ctrl), resid(b, ctrl))[0] if ctrl is not None and ctrl.size else stats.pearsonr(a, b)[0]
rng = np.random.default_rng(3)
# ------------------------------------------------------------------ 4a length confound: score ~ length, quality ~ length, quality ~ score with / without length
rows = []
for (b, k) in DATASETS:
    d0 = get(b, k)
    if len(d0) == 0: continue
    for met in ("chrf", "comet"):
        for m, col in (("rl", f"res_{met}_rl"), ("nllb3b3", f"res_{met}_nllb3b3"), ("rl_minus_nllb3b3", f"res_gap_{met}")):
            g = chunk_frame(d0, col)
            if len(g) < 60: continue
            C = len(g); y = g.y.values; s = g.score.values; lw = g.lw.values; w = g.n.values
            r = {"dataset": BNAME.get(b, "pooled (all benchmarks)"), "k": k, "metric": "chrF++" if met == "chrf" else "AfriCOMET", "model": m, "n_chunks": C,
                 "rho_score_length": stats.spearmanr(s, lw)[0], "rho_quality_length": stats.spearmanr(y, lw)[0], "rho_quality_score": stats.spearmanr(y, s)[0]}
            def coefs(idx):
                Xs = np.c_[np.ones(len(idx)), s[idx]]; Xl = np.c_[np.ones(len(idx)), s[idx], lw[idx]]
                sw = np.sqrt(w[idx])
                b1 = np.linalg.lstsq(Xs * sw[:, None], y[idx] * sw, rcond=None)[0][1]
                b2 = np.linalg.lstsq(Xl * sw[:, None], y[idx] * sw, rcond=None)[0][1]
                b3 = np.linalg.lstsq(Xl * sw[:, None], y[idx] * sw, rcond=None)[0][2]
                return b1, b2, b3
            c0 = coefs(np.arange(C)); bs = np.array([coefs(rng.integers(0, C, C)) for _ in range(400)])
            for nm, v, j in (("slope_score_alone", c0[0], 0), ("slope_score_given_length", c0[1], 1), ("slope_loglength_given_score", c0[2], 2)):
                r[nm] = v; r[nm + "_lo"], r[nm + "_hi"] = bci(bs[:, j])
            r["partial_r_score_quality_given_length"] = partial_r(s, y, lw[:, None])
            rows.append(r)
pd.DataFrame(rows).to_csv(f"{OUT}/05_length_confound.csv", index=False, float_format="%.4g")
# ------------------------------------------------------------------ 4b stratify by length tertile
rows = []
for (b, k) in DATASETS:
    d0 = get(b, k)
    if len(d0) == 0: continue
    u = d0.drop_duplicates("ekey")
    qs_ = u.n_words.quantile([1 / 3, 2 / 3]).values
    for met in ("chrf", "comet"):
        col = f"res_{met}_rl"
        for si, (lo, hi) in enumerate([(-np.inf, qs_[0]), (qs_[0], qs_[1]), (qs_[1], np.inf)]):
            d1 = d0[(d0.n_words > lo) & (d0.n_words <= hi)]
            g = chunk_frame(d1, col)
            if len(g) < 40: continue
            x = g.score.values; y = g.y.values; w = g.n.values; C = len(g)
            def sl(idx):
                X = np.c_[np.ones(len(idx)), x[idx]]; sw = np.sqrt(w[idx]); return np.linalg.lstsq(X * sw[:, None], y[idx] * sw, rcond=None)[0][1]
            bs = [sl(rng.integers(0, C, C)) for _ in range(400)]
            rows.append({"dataset": BNAME.get(b, "pooled (all benchmarks)"), "k": k, "metric": "chrF++" if met == "chrf" else "AfriCOMET", "model": "rl", "length_tertile": ["short", "middle", "long"][si],
                         "words_range": f"{d1.n_words.min():.0f}-{d1.n_words.max():.0f}", "n_chunks": C, "score_range_p05_p95": f"{np.quantile(x,.05):.2f}-{np.quantile(x,.95):.2f}", "slope_per_point": sl(np.arange(C)),
                         "slope_lo": np.quantile(bs, .025), "slope_hi": np.quantile(bs, .975)})
pd.DataFrame(rows).to_csv(f"{OUT}/05b_stratified_by_length_tertile.csv", index=False, float_format="%.4g")
# ------------------------------------------------------------------ 4c dimensions: zero-order, partial (given length), unique (joint OLS), failure AUC
def auc(score, label):
    r = stats.rankdata(score); n1 = label.sum(); n0 = len(label) - n1
    return (r[label.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
P["fail_flag_rl"] = ((P.fin_rl == 0) | (P.loop_rl == 1) | (P.lid_rl == 0)).astype(float)
P.loc[P.fin_rl.isna(), "fail_flag_rl"] = np.nan
P["fail_bad_chrf_rl"] = np.nan; P["fail_bad_comet_rl"] = np.nan
for key, col in (("chrf", "res_chrf_rl"), ("comet", "res_comet_rl")):
    q10 = P.groupby(["b", "k", "iso"])[col].transform(lambda x: x.quantile(0.10))       # worst decile of the language's own chunks (relative failure)
    P["fail_bad_%s_rl" % key] = np.where(P[col].notna(), (P[col] <= q10).astype(float), np.nan)
rows, arows = [], []
for (b, k) in DATASETS:
    if k != 1 and b not in ("flores_plus", "ntrex", "mafand", "pooled"): continue
    d0 = get(b, k)
    if len(d0) == 0: continue
    for met in ("chrf", "comet"):
        for m, col in (("rl", f"res_{met}_rl"), ("nllb3b3", f"res_{met}_nllb3b3"), ("rl_minus_nllb3b3", f"res_gap_{met}")):
            g = chunk_frame(d0, col)
            if len(g) < 100: continue
            y = g.y.values; lw = g.lw.values[:, None]; C = len(g)
            Xall = g[DIMF].values
            for j, dname in enumerate(DIMS + ["score(weighted mean)", "efi_prob", "log_words"]):
                v = g[DIMF[j]].values if j < 5 else (g.score.values if j == 5 else (g.efi_prob.values if j == 6 else g.lw.values))
                if np.nanstd(v) < 1e-6: continue
                oth = np.delete(Xall, j, axis=1) if j < 5 else None
                def stat(idx):
                    zo = stats.spearmanr(v[idx], y[idx])[0]
                    pl = partial_r(v[idx], y[idx], lw[idx]) if j != 7 + 0 and j < 7 else np.nan
                    if j < 5:
                        ctrl = np.c_[oth[idx], lw[idx]]; pu = partial_r(v[idx], y[idx], ctrl)
                    else: pu = np.nan
                    return zo, pl, pu
                e0 = stat(np.arange(C)); bs = np.array([stat(rng.integers(0, C, C)) for _ in range(300)])
                r = {"dataset": BNAME.get(b, "pooled (all benchmarks)"), "k": k, "metric": "chrF++" if met == "chrf" else "AfriCOMET", "model": m, "dimension": dname, "n_chunks": C,
                     "sd_of_dimension": float(np.std(v))}
                for i_, nm in enumerate(("spearman_zero_order", "partial_r_given_length", "partial_r_given_length_and_other_four_dims")):
                    r[nm] = e0[i_]; r[nm + "_lo"], r[nm + "_hi"] = (np.nanquantile(bs[:, i_], .025), np.nanquantile(bs[:, i_], .975)) if not np.all(np.isnan(bs[:, i_])) else (np.nan, np.nan)
                rows.append(r)
    # failure AUC (row level, cluster bootstrap over English chunks)
    if m is not None:
        for fl in ("fail_flag_rl", "fail_bad_chrf_rl", "fail_bad_comet_rl"):
            dd = d0.dropna(subset=[fl])
            if dd[fl].sum() < 30: continue
            clusters = dd.ekey.values; uc = np.unique(clusters); idx_by = {c: np.where(clusters == c)[0] for c in uc}
            for j, dname in enumerate(DIMS + ["score(weighted mean)", "n_words"]):
                v = dd[DIMF[j]].values if j < 5 else (dd.score.values if j == 5 else dd.n_words.values)
                lab = dd[fl].values
                a0 = auc(v, lab)
                bsa = []
                for _ in range(100):
                    cs = rng.choice(uc, len(uc)); ii = np.concatenate([idx_by[c] for c in cs]); bsa.append(auc(v[ii], lab[ii]))
                arows.append({"dataset": BNAME.get(b, "pooled (all benchmarks)"), "k": k, "failure": fl, "base_rate": lab.mean(), "n_rows": len(dd), "dimension": dname, "auc": a0,
                              "auc_lo": np.nanquantile(bsa, .025), "auc_hi": np.nanquantile(bsa, .975)})
    print(b, k, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/06_dimension_partial_correlations.csv", index=False, float_format="%.4g")
pd.DataFrame(arows).to_csv(f"{OUT}/06b_failure_auc_by_dimension.csv", index=False, float_format="%.4g")
