"""Section 5-8: agreement with NLLB-3.3B output, english-form chunks, risk of failures, practical translate-when rules."""
from t1load import *
from scipy import stats
A = pd.read_pickle(f"{DATA}/A.pkl")
P = A[A.n_pair_seen == 0].copy()
def dm(d, c): return d[c] - d.groupby(["b", "k", "iso"])[c].transform("mean")
for m in ["rl", "sft", "nllb3b3", "nllb600m", "a2w", "live250", "tpsy08"]:
    for met in ("chrf", "comet"):
        c = f"{met}_{m}"
        if c in P: P["res_" + c] = dm(P, c)
    if f"agree_{m}" in P: P[f"res_agree_{m}"] = dm(P, f"agree_{m}")
DATASETS = [("flores_plus", 1), ("flores_plus", 3), ("flores_plus", 5), ("ntrex", 1), ("ntrex", 3), ("ntrex", 5), ("mafand", 1), ("mafand", 3), ("mafand", 5), ("smol_sent", 1), ("pooled", 1), ("pooled", 3), ("pooled", 5)]
def get(b, k):
    return P[P.k == k] if b == "pooled" else P[(P.b == b) & (P.k == k)]
nm = lambda b: BNAME.get(b, "pooled (all benchmarks)")
GROUPS = ["all", "strong", "mid", "weak"]
def sub(d, g): return d if g == "all" else d[d.group == g]
def auc(score, label):
    r = stats.rankdata(score); n1 = label.sum(); n0 = len(label) - n1
    return (r[label.astype(bool)].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan
def auc_boot(d, sc, lab, B=100, seed=5):
    d = d.dropna(subset=[sc, lab]); v = d[sc].values; y = d[lab].values.astype(int)
    if y.sum() < 30 or (1 - y).sum() < 30: return (np.nan,) * 3
    cl = d.ekey.values; uc, inv = np.unique(cl, return_inverse=True); order = np.argsort(inv); starts = np.r_[0, np.cumsum(np.bincount(inv))]
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        cs = rng.integers(0, len(uc), len(uc)); ii = np.concatenate([order[starts[c]:starts[c + 1]] for c in cs]); out.append(auc(v[ii], y[ii]))
    return auc(v, y), np.nanquantile(out, .025), np.nanquantile(out, .975)
# ---------------------------------------------------------------- agreement by band: pseudo-reference chrF++ (vs NLLB-3.3B output) next to human-reference chrF++
rows = []
for (b, k) in DATASETS:
    d0 = get(b, k)
    for g in GROUPS:
        d1 = sub(d0, g)
        for band in BAND_NAMES + ["english-form (kept in English)", "ALL (incl. english-form)"]:
            d2 = d1 if band.startswith("ALL") else d1[d1.band_full == band]
            if d2.ekey.nunique() < 20: continue
            for m in ["rl", "sft", "a2w", "live250", "nllb600m", "tpsy08"]:
                if f"agree_{m}" not in d2: continue
                dd = d2.dropna(subset=[f"agree_{m}", f"chrf_{m}", "chrf_nllb3b3"]).copy()
                if len(dd) < 20: continue
                dd["_inf"] = dd[f"agree_{m}"] - dd[f"chrf_{m}"]
                dd["_h3"] = dd["chrf_nllb3b3"]
                e1 = boot_mean(dd, f"agree_{m}", B=300); e2 = boot_mean(dd, f"chrf_{m}", B=300); e3 = boot_mean(dd, "_inf", B=300)
                rows.append({"dataset": nm(b), "k": k, "group": g, "band": band, "system": m, "n_rows": e1[3], "n_chunks": e1[4],
                             "chrf_vs_nllb3b3_output(pseudo_ref)": e1[0], "pseudo_lo": e1[1], "pseudo_hi": e1[2],
                             "chrf_vs_human_ref": e2[0], "human_lo": e2[1], "human_hi": e2[2], "inflation_pseudo_minus_human": e3[0], "infl_lo": e3[1], "infl_hi": e3[2],
                             "share_agree_ge_50_pct": 100 * (dd[f"agree_{m}"] >= 50).mean()})
pd.DataFrame(rows).to_csv(f"{OUT}/08_agreement_with_nllb3b3_by_band.csv", index=False, float_format="%.4g")
# ---------------------------------------------------------------- is agreement a usable quality signal? AUC for detecting RL's worst quartile (language-relative)
rows = []
for (b, k) in DATASETS:
    d0 = get(b, k)
    for g in GROUPS:
        d1 = sub(d0, g)
        if d1.ekey.nunique() < 100: continue
        for met, lab in (("chrf", "worst quartile of RL chrF++ (language-relative)"), ("comet", "worst quartile of RL AfriCOMET (language-relative)")):
            col = f"res_{met}_rl"; d1 = d1.copy()
            q25 = d1.groupby("iso")[col].transform(lambda x: x.quantile(0.25)); d1["_bad"] = np.where(d1[col].notna(), (d1[col] <= q25).astype(float), np.nan)
            d1["_neg_agree"] = -d1["res_agree_rl"]; d1["_score"] = d1["score"]; d1["_nwords"] = d1["n_words"]
            for sig, nme in (("_neg_agree", "low agreement with NLLB-3.3B (neg. language-relative chrF++ vs NLLB output)"), ("_score", "difficulty score"), ("_nwords", "source length (words)")):
                a, lo, hi = auc_boot(d1, sig, "_bad")
                rows.append({"dataset": nm(b), "k": k, "group": g, "target": lab, "signal": nme, "auc": a, "auc_lo": lo, "auc_hi": hi, "n_rows": int(d1["_bad"].notna().sum())})
    print(b, k, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/08b_agreement_vs_score_as_quality_signal.csv", index=False, float_format="%.4g")
# ---------------------------------------------------------------- english-form chunks
rows = []
for (b, k) in DATASETS:
    d0 = get(b, k)
    for g in ["all"]:
        e = d0[d0.efi]; ne = d0[~d0.efi]
        if e.ekey.nunique() == 0: continue
        for m in ["rl", "nllb3b3", "sft"]:
            for met in ("chrf", "comet"):
                c = f"{met}_{m}"
                if c not in d0: continue
                ee = boot_mean(e, c, B=300); nn = boot_mean(ne, c, B=300)
                rows.append({"dataset": nm(b), "k": k, "model": m, "metric": "chrF++" if met == "chrf" else "AfriCOMET", "n_efi_chunks": e.ekey.nunique(), "pct_of_chunks": 100 * e.ekey.nunique() / d0.ekey.nunique(),
                             "efi_mean_score": e.drop_duplicates("ekey").score.mean(), "nonefi_mean_score": ne.drop_duplicates("ekey").score.mean(),
                             "efi_quality": ee[0], "efi_lo": ee[1], "efi_hi": ee[2], "nonefi_quality": nn[0], "nonefi_lo": nn[1], "nonefi_hi": nn[2]})
pd.DataFrame(rows).to_csv(f"{OUT}/09_english_form_dependent.csv", index=False, float_format="%.4g")
# ---------------------------------------------------------------- risk of failures by band (RL, SFT)
rows = []
for (b, k) in DATASETS:
    d0 = get(b, k)
    for g in GROUPS:
        d1 = sub(d0, g)
        for band in BAND_NAMES + ["english-form (kept in English)", "ALL (incl. english-form)"]:
            d2 = d1 if band.startswith("ALL") else d1[d1.band_full == band]
            if d2.ekey.nunique() < 20: continue
            for m in ["rl", "sft"]:
                dd = d2.dropna(subset=[f"fin_{m}"]).copy()
                if len(dd) < 50: continue
                dd["unfinished"] = 1 - dd[f"fin_{m}"]; dd["wrong_language"] = 1 - dd[f"lid_{m}"]; dd["loop"] = dd[f"loop_{m}"]; dd["rep3"] = dd[f"rep3_{m}"]
                dd["any_failure"] = ((dd.unfinished == 1) | (dd.loop == 1) | (dd.wrong_language == 1)).astype(float)
                dd["poor_comet_lt_0.5"] = np.where(dd[f"comet_{m}"].notna(), (dd[f"comet_{m}"] < 0.5).astype(float), np.nan)
                dd["poor_chrf_lt_20"] = (dd[f"chrf_{m}"] < 20).astype(float)
                r = {"dataset": nm(b), "k": k, "group": g, "band": band, "model": m, "n_rows": len(dd), "n_chunks": dd.ekey.nunique()}
                for c in ["unfinished", "loop", "wrong_language", "rep3", "any_failure", "poor_comet_lt_0.5", "poor_chrf_lt_20"]:
                    e = boot_mean(dd, c, B=300); r[c + "_pct"] = 100 * e[0]; r[c + "_lo"] = 100 * e[1]; r[c + "_hi"] = 100 * e[2]
                rows.append(r)
pd.DataFrame(rows).to_csv(f"{OUT}/10_risk_by_band.csv", index=False, float_format="%.4g")
# ---------------------------------------------------------------- practical rule: translate when score <= x
rows = []
XS = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 99]
for (b, k) in DATASETS:
    d0 = get(b, k); d0 = d0[~d0.efi]
    for g in GROUPS:
        d1 = sub(d0, g)
        if d1.ekey.nunique() < 50: continue
        for x in XS:
            d2 = d1[d1.score <= x]; d3 = d1[d1.score > x]
            if d2.ekey.nunique() < 20: continue
            for m in ["rl", "nllb3b3", "sft"] + (["a2w"] if b in ("flores_plus", "pooled") else []):
                for met in ("chrf", "comet"):
                    c = f"{met}_{m}"
                    if c not in d2 or d2[c].notna().sum() < 30: continue
                    e = boot_mean(d2, c, B=300)
                    q = d2[c].dropna()
                    r = {"dataset": nm(b), "k": k, "group": g, "translate_if_score_leq": x if x < 99 else "no limit", "model": m, "metric": "chrF++" if met == "chrf" else "AfriCOMET",
                         "coverage_pct_of_chunks": 100 * d2.ekey.nunique() / d1.ekey.nunique(), "n_rows": e[3], "mean_quality": e[0], "ci_lo": e[1], "ci_hi": e[2],
                         "p10_row_quality": q.quantile(.10), "p25_row_quality": q.quantile(.25)}
                    if len(d3) >= 20 and d3[c].notna().sum() >= 20:
                        e2 = boot_mean(d3, c, B=300); r["mean_quality_of_chunks_above"] = e2[0]
                    if m in ("rl", "sft"):
                        dd = d2.dropna(subset=[f"fin_{m}"]); r["failure_pct"] = 100 * (((dd[f"fin_{m}"] == 0) | (dd[f"loop_{m}"] == 1) | (dd[f"lid_{m}"] == 0)).mean()) if len(dd) else np.nan
                    if m != "nllb3b3" and f"{met}_nllb3b3" in d2:
                        dd = d2.dropna(subset=[c, f"{met}_nllb3b3"]).assign(_g=lambda t: t[c] - t[f"{met}_nllb3b3"])
                        if len(dd) > 30:
                            e3 = boot_mean(dd, "_g", B=300); r["gap_vs_nllb3b3"], r["gap_lo"], r["gap_hi"] = e3[:3]
                    rows.append(r)
    print("rules", b, k, flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/11_translate_when_score_leq_x.csv", index=False, float_format="%.4g")
