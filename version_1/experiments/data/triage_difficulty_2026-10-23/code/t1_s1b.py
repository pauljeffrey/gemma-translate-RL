from t1load import *
A = pd.read_pickle(f"{DATA}/A.pkl"); P = A[A.n_pair_seen == 0].copy()
def dm(d, c): return d[c] - d.groupby(["b", "k", "iso"])[c].transform("mean")
for m in ["rl", "sft", "nllb3b3"]:
    for met in ("chrf", "comet"): P[f"res_{met}_{m}"] = dm(P, f"{met}_{m}")
rows = []
for name, sel in [("FLORES+ devtest", lambda d: d[d.b == "flores_plus"]), ("NTREX-128", lambda d: d[d.b == "ntrex"]), ("MAFAND-MT", lambda d: d[d.b == "mafand"]), ("pooled (all benchmarks)", lambda d: d)]:
    for k in (1, 3, 5):
        d0 = sel(P[P.k == k])
        for g in ["all", "strong", "mid", "weak"]:
            d1 = d0 if g == "all" else d0[d0.group == g]
            for band in BAND_NAMES + ["english-form (kept in English)"]:
                d2 = d1[d1.band_full == band]
                if d2.ekey.nunique() < 20: continue
                for m in ["rl", "nllb3b3", "sft"]:
                    for met in ("chrf", "comet"):
                        e = boot_mean(d2, f"res_{met}_{m}", B=500)
                        rows.append({"dataset": name, "k": k, "group": g, "band": band, "model": m, "metric": "chrF++" if met == "chrf" else "AfriCOMET", "n_rows": e[3], "n_chunks": e[4], "mean_relative_to_language_mean": e[0], "ci_lo": e[1], "ci_hi": e[2]})
pd.DataFrame(rows).to_csv(f"{OUT}/03b_band_quality_language_relative.csv", index=False, float_format="%.4g")
