"""Section 9: long documents (lc1: SmolDoc human-reference documents and AfriScience-MT dev/test documents), document-level chrF++ vs human references by difficulty score."""
from t1load import *
from scipy import stats
c = pd.read_json(f"{DATA}/cls_ld.jsonl", lines=True)
c["score"] = sum(w * c[k + "_int"] for k, w in W.items()).round(2); c["score_f"] = sum(w * c[k + "_f"] for k, w in W.items())
c["band"] = band_of(c.score, c.efi)
L = pd.read_csv(f"{DATA}/longdoc_items.csv.gz").merge(c, left_on="doc_id", right_on="ekey")
L = L[L["mode"].isin(["sent", "whole"])]
docs = c.drop_duplicates("ekey")
dist = c.merge(L[["doc_id", "set", "bin", "src_tokens"]].drop_duplicates("doc_id"), left_on="ekey", right_on="doc_id")
print(dist.groupby(["set", "bin"]).agg(n_docs=("ekey", "size"), tokens=("src_tokens", "mean"), score_mean=("score", "mean"), score_min=("score", "min"), score_max=("score", "max"), efi=("efi", "sum")).round(2))
dist.groupby(["set", "bin"]).agg(n_docs=("ekey", "size"), mean_src_tokens=("src_tokens", "mean"), score_mean=("score", "mean"), score_sd=("score", "std"), score_min=("score", "min"), score_max=("score", "max"), n_english_form=("efi", "sum")).round(3).reset_index().to_csv(f"{OUT}/12_longdoc_score_distribution.csv", index=False)
rng = np.random.default_rng(7)
rows = []
L["lw"] = np.log(L.src_tokens)
for (model, mode, st), d in L.groupby(["model", "mode", "set"]):
    # doc-level: average over languages -> one value per document (cluster = document)
    d = d.copy(); d["res"] = d.chrf - d.groupby("iso").chrf.transform("mean")
    g = d.groupby("doc_id").agg(res=("res", "mean"), score=("score", "first"), lw=("lw", "first"), chrf=("chrf", "mean"), n=("chrf", "size"), finished=("finished", "mean"), loop=("loop", "mean"))
    if len(g) < 20: continue
    x, y, lw = g.score.values, g.res.values, g.lw.values; C = len(g)
    def stat(i):
        X1 = np.c_[np.ones(len(i)), x[i]]; X2 = np.c_[np.ones(len(i)), x[i], lw[i]]
        return np.linalg.lstsq(X1, y[i], rcond=None)[0][1], np.linalg.lstsq(X2, y[i], rcond=None)[0][1], stats.spearmanr(x[i], y[i])[0]
    e0 = stat(np.arange(C)); bs = np.array([stat(rng.integers(0, C, C)) for _ in range(500)])
    rows.append({"model": model, "mode": mode, "set": st, "n_docs": C, "n_items": int(g.n.sum()), "score_range": f"{x.min():.2f}-{x.max():.2f}", "mean_chrf": g.chrf.mean(),
                 "spearman_score_quality": e0[2], "sp_lo": np.quantile(bs[:, 2], .025), "sp_hi": np.quantile(bs[:, 2], .975),
                 "slope_per_point": e0[0], "slope_lo": np.quantile(bs[:, 0], .025), "slope_hi": np.quantile(bs[:, 0], .975),
                 "slope_given_log_tokens": e0[1], "slope_len_lo": np.quantile(bs[:, 1], .025), "slope_len_hi": np.quantile(bs[:, 1], .975)})
pd.DataFrame(rows).to_csv(f"{OUT}/12b_longdoc_score_vs_chrf.csv", index=False, float_format="%.4g")
print(pd.DataFrame(rows)[["model", "mode", "set", "n_docs", "score_range", "mean_chrf", "spearman_score_quality", "sp_lo", "sp_hi", "slope_per_point", "slope_lo", "slope_hi", "slope_given_log_tokens", "slope_len_lo", "slope_len_hi"]].round(2).to_string())
# bands: mean chrF++ per band (documents), models sent mode
d = L[L["mode"] == "sent"].copy()
d["sbin"] = pd.cut(d.score, [0, 3.0, 3.5, 4.0, 4.5, 5.5, 9], labels=["<=3.0", "3.0-3.5", "3.5-4.0", "4.0-4.5", "4.5-5.5", ">5.5"])
tab = []
for (model, st, sb), g in d.groupby(["model", "set", "sbin"], observed=True):
    gg = g.groupby("doc_id").chrf.mean()
    if len(gg) < 5: continue
    e = boot_mean(g.assign(ekey=g.doc_id), "chrf", B=500)
    tab.append({"model": model, "set": st, "score_bin": sb, "n_docs": g.doc_id.nunique(), "n_items": len(g), "mean_chrf": e[0], "ci_lo": e[1], "ci_hi": e[2], "finished_pct": 100 * g.finished.mean(), "loop_pct": 100 * g["loop"].mean(),
                "lid_doc_ok_pct": 100 * g.lid_doc_ok.mean()})
pd.DataFrame(tab).to_csv(f"{OUT}/12c_longdoc_chrf_by_score_bin_sentence_mode.csv", index=False, float_format="%.4g")
print(pd.DataFrame(tab).query("model in ['rl','nllb3b3','sft']").round(1).to_string())
