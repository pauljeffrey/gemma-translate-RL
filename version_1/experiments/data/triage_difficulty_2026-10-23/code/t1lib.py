import numpy as np, pandas as pd, json, os
W = {"vocabulary_load": .25, "grammar_complexity": .20, "discourse_complexity": .20, "cultural_load": .20, "format_structure": .15}
DIMS = list(W)
BANDS = [("very easy", -np.inf, 3.0), ("easy", 3.0, 4.0), ("moderately difficult", 4.0, 5.5), ("hard", 5.5, 7.0), ("very hard", 7.0, np.inf)]
BAND_NAMES = [b[0] for b in BANDS]
def band_of(score, efi=None):
    s = np.asarray(score, float)
    out = np.select([s <= 3.0, s <= 4.0, s <= 5.5, s <= 7.0], BAND_NAMES[:4], BAND_NAMES[4]).astype(object)
    if efi is not None: out = np.where(np.asarray(efi, bool), "english-form (kept in English)", out)
    return out
DATA = "/tmp/scratch"
OUT = "/path/to/repo/reports/data/triage_difficulty_2026-10-23"
MODELS_F = ["rl", "sft", "a2w", "live250", "nllb3b3", "nllb600m", "tpsy08", "tpsy2b"]
LABEL = {"rl": "RL-final (step 1433)", "sft": "SFT-final", "a2w": "science-stage start (A2W 0.7)", "live250": "science run step 250", "nllb3b3": "NLLB-3.3B", "nllb600m": "NLLB-600M",
         "tpsy08": "TranslatePsy 0.8B", "tpsy2b": "TranslatePsy 2B"}

def load_cls():
    c = pd.read_json(f"{DATA}/cls_all.jsonl", lines=True)
    c["score"] = sum(w * c[k + "_int"] for k, w in W.items()).round(2)          # official: weighted mean of the integer ratings
    c["score_f"] = sum(w * c[k + "_f"] for k, w in W.items())                  # continuous version of the same weighted mean
    c["score_model_check"] = c["score_model"]
    c["band"] = band_of(c["score"])
    c["band_full"] = band_of(c["score"], c["efi"])
    return c

def load_bench(b, cls):
    df = pd.read_csv(f"{DATA}/chunks_{b}.csv.gz")
    return df.merge(cls, on="ekey", how="left")

def cluster_boot(S, N, B=1000, seed=0):
    """S, N: per-cluster sums and counts -> bootstrap distribution of sum(S)/sum(N) over cluster resampling."""
    rng = np.random.default_rng(seed); C = len(S)
    idx = rng.integers(0, C, size=(B, C))
    return S[idx].sum(1) / np.maximum(N[idx].sum(1), 1)

def boot_mean(df, col, cluster="ekey", B=1000, seed=0, w=None):
    d = df[[cluster, col]].dropna()
    if len(d) == 0: return (np.nan, np.nan, np.nan, 0, 0)
    g = d.groupby(cluster)[col].agg(["sum", "count"])
    S, N = g["sum"].values, g["count"].values
    est = S.sum() / N.sum()
    if len(S) < 2: return (est, np.nan, np.nan, int(N.sum()), len(S))
    bs = cluster_boot(S, N, B, seed)
    return (est, *np.quantile(bs, [0.025, 0.975]), int(N.sum()), len(S))

def boot_paired(df, a, b, cluster="ekey", B=1000, seed=0):
    d = df[[cluster, a, b]].dropna()
    d = d.assign(_d=d[a] - d[b])
    return boot_mean(d, "_d", cluster, B, seed)

# ------------------------------------------------------------------------------------------------ turning-point tools
def chunk_level(d, ycol, xcol="score", cluster="ekey"):
    """aggregate rows (chunk x language) to English-chunk level: x, mean y over languages, number of rows."""
    g = d.dropna(subset=[ycol]).groupby(cluster).agg(x=(xcol, "first"), y=(ycol, "mean"), n=(ycol, "size"), S=(ycol, "sum"))
    return g.reset_index()

def iso_fit(x, y, w):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x, y, sample_weight=w)
    return ir

def hinge_fit(x, y, w, thetas):
    """continuous piecewise-linear with one breakpoint: y = a + b1*x + b2*max(x-theta,0); weighted LS; returns best (theta, a, b1, b2, sse) and the no-break linear sse."""
    sw = np.sqrt(w)
    best = None
    for th in thetas:
        X = np.c_[np.ones_like(x), x, np.maximum(x - th, 0)] * sw[:, None]
        beta, res, *_ = np.linalg.lstsq(X, y * sw, rcond=None)
        sse = float(((X @ beta - y * sw) ** 2).sum())
        if best is None or sse < best[4]: best = (th, *beta, sse)
    X = np.c_[np.ones_like(x), x] * sw[:, None]
    b0, *_ = np.linalg.lstsq(X, y * sw, rcond=None)
    sse0 = float(((X @ b0 - y * sw) ** 2).sum())
    return best, sse0, b0

def threshold_scan(d, ycol, grid, B=500, seed=0, cluster="ekey", xcol="score", min_side=30):
    """For each candidate threshold s: mean(y | x > s) - mean(y | x <= s), cluster bootstrap over English chunks, pointwise and sup-t simultaneous 95% CIs."""
    dd = d.dropna(subset=[ycol])
    g = dd.groupby(cluster).agg(x=(xcol, "first"), S=(ycol, "sum"), N=(ycol, "size"))
    x, S, N = g.x.values, g.S.values, g.N.values
    J = len(grid)
    Sa = np.stack([S * (x > s) for s in grid], 1); Na = np.stack([N * (x > s) for s in grid], 1)
    Sb = np.stack([S * (x <= s) for s in grid], 1); Nb = np.stack([N * (x <= s) for s in grid], 1)
    def est(idx=None):
        if idx is None: sa, na, sb, nb = Sa.sum(0), Na.sum(0), Sb.sum(0), Nb.sum(0)
        else: sa, na, sb, nb = Sa[idx].sum(0), Na[idx].sum(0), Sb[idx].sum(0), Nb[idx].sum(0)
        with np.errstate(divide="ignore", invalid="ignore"): return sa / na - sb / nb, na, nb
    e0, na0, nb0 = est()
    rng = np.random.default_rng(seed); C = len(x)
    bs = np.empty((B, J))
    for b in range(B): bs[b] = est(rng.integers(0, C, C))[0]
    se = np.nanstd(bs, 0)
    z = np.nanmax(np.abs(bs - np.nanmean(bs, 0)) / np.where(se > 0, se, np.nan), 1)
    zc = np.nanquantile(z, 0.95)
    out = pd.DataFrame({"threshold": grid, "diff": e0, "se": se, "lo": np.nanquantile(bs, 0.025, 0), "hi": np.nanquantile(bs, 0.975, 0),
                        "lo_sup": e0 - zc * se, "hi_sup": e0 + zc * se, "n_above_rows": na0, "n_below_rows": nb0,
                        "n_above_chunks": [(x > s).sum() for s in grid], "n_below_chunks": [(x <= s).sum() for s in grid]})
    out["ok"] = (out.n_above_chunks >= min_side) & (out.n_below_chunks >= min_side)
    out.loc[~out.ok, ["diff", "se", "lo", "hi", "lo_sup", "hi_sup"]] = np.nan
    return out

def first_sig_drop(scan, kind="hi"):
    s = scan[scan.ok & (scan[kind] < 0)]
    return float(s.threshold.min()) if len(s) else np.nan
