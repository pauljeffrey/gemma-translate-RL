"""E-08 step 3: extra tables (needs analyse.py outputs: thresholds.json, features_*.parquet, refs_features.parquet).
 - rate of both detectors with and without the 7 near-failed languages (taq knc zgh dik aeb arz ary)
 - threshold sensitivity of the independent detector (reference p99.9 vs p99.99), sub-rule decomposition, word-count of independent-only rows
 - language-matched sentence chrF++ of independent-only flagged rows against unflagged rows (sentence chrF++ per row, cached)
 - reference-side controls for tail, copy and length ratio
Usage: LID218=... python analyse2.py
"""
import json, os, sys
import numpy as np, pandas as pd
from sacrebleu.metrics import CHRF

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]]
exec(open(os.path.join(HERE, "analyse.py")).read().split("# ------------------------------------------------------------------ per model tables")[0])  # reuses loading, thresholds, add_flags, lid helpers
sys.path.insert(0, HERE)
import compute_features as CF

MAIN = ["sft", "stageA", "b1best", "mixedbest", "weakfinal"]
SEVEN = ["taq", "knc", "zgh", "dik", "aeb", "arz", "ary"]
D = {}
for m in MAIN:
    d = add_flags(F[m], R)
    D[m] = d

# ---------------------------------------------------------------- per-row sentence chrF++ (cached)
chrf = CHRF(word_order=2)
for m in MAIN:
    p = os.path.join(HERE, f"sentchrf_{m}.parquet")
    if os.path.exists(p):
        s = pd.read_parquet(p)
    else:
        rows = CF.load(CF.FILES[m])
        s = pd.DataFrame({"benchmark": [r["benchmark"] for r in rows], "iso": [r["iso"] for r in rows], "id": [str(r["id"]) for r in rows],
                          "chrfpp": [chrf.sentence_score(CF.nfc(r["hyp"]), [CF.nfc(r["reference"])]).score for r in rows]})
        s.to_parquet(p)
    D[m] = D[m].merge(s, on=KEY, how="left")

out = {}
# ---------------------------------------------------------------- 1. with and without the 7 near-failed languages
rows = []
for m in MAIN:
    d = D[m]
    for b in BENCH:
        g = d[d.benchmark == b]
        for tag, gg in [("all", g), ("without_7", g[~g.iso.isin(SEVEN)]), ("only_7", g[g.iso.isin(SEVEN)])]:
            if len(gg) == 0:
                continue
            rows.append(dict(model=m, benchmark=b, subset=tag, n_lang=gg.iso.nunique(), rw_loop=gg.groupby("iso").rw_loop.mean().mean(), ind_loop=gg.groupby("iso").ind_loop.mean().mean(),
                             rw_loop_pooled=gg.rw_loop.mean(), ind_loop_pooled=gg.ind_loop.mean(), unfinished=gg.groupby("iso").unfinished.mean().mean()))
pd.DataFrame(rows).to_csv(os.path.join(HERE, "rates_with_without_7_languages.csv"), index=False)
sh = []
for m in MAIN:
    g = D[m][D[m].benchmark == "flores_plus"]
    sh.append(dict(model=m, share_of_rw_flags_in_7=float(g[g.rw_loop].iso.isin(SEVEN).mean()), share_of_ind_flags_in_7=float(g[g.ind_loop].iso.isin(SEVEN).mean()),
                   share_of_ind_only_in_7=float(g[g.ind_loop & ~g.rw_loop].iso.isin(SEVEN).mean())))
pd.DataFrame(sh).to_csv(os.path.join(HERE, "share_flags_in_7_languages_flores.csv"), index=False)

# ---------------------------------------------------------------- 2. sensitivity and decomposition
sens = []
alt = {"p99.9": (0.1, 99.9), "p99.99": (0.01, 99.99), "p99": (1.0, 99.0)}
for tag, (lo, hi) in alt.items():
    t = {"z": float(np.nanpercentile(R["r_z"], lo)), "c8": float(np.nanpercentile(R["r_c8_rep"], hi)), "w3": float(np.nanpercentile(R["r_w3_rep"], hi))}
    refflag = ((R["r_z"] < t["z"]) | (R["r_c8_rep"] > t["c8"]) | (R["r_w3_rep"] > t["w3"]))
    for m in MAIN:
        d = D[m]
        fl = (d.h_z < t["z"]) | (d.h_c8_rep > t["c8"]) | (d.h_w3_rep > t["w3"])
        for b in BENCH:
            g = d[d.benchmark == b]; f = fl[g.index]
            sens.append(dict(threshold=tag, z=t["z"], c8=t["c8"], w3=t["w3"], model=m, benchmark=b, ind_rate_macro=f.groupby(g.iso).mean().mean(), rw_rate_macro=g.groupby("iso").rw_loop.mean().mean(),
                             ref_rate_macro=refflag[R.benchmark == b].groupby(R.iso[R.benchmark == b]).mean().mean()))
pd.DataFrame(sens).to_csv(os.path.join(HERE, "threshold_sensitivity.csv"), index=False)

dec = []
for m in MAIN:
    g = D[m]; io = g[g.ind_loop & ~g.rw_loop]
    dec.append(dict(model=m, n_ind_only=len(io), share_words_lt24=float((io.h_words < 24).mean()), median_words=float(io.h_words.median()),
                    share_z=float(io.ind_z.mean()), share_c8=float(io.ind_c8.mean()), share_w3=float(io.ind_w3.mean()),
                    share_finished=float(io.finished.mean()), median_hyp_over_ref_chars=float(io.ratio_ref_chars.median()),
                    share_hyp_over_2x_ref=float((io.ratio_ref_chars > 2).mean()), share_in_7=float(io.iso.isin(SEVEN).mean())))
pd.DataFrame(dec).to_csv(os.path.join(HERE, "independent_only_decomposition.csv"), index=False)

# ---------------------------------------------------------------- 3. language-matched quality of independent-only rows
lm = []
for m in MAIN:
    g = D[m]
    for b in BENCH + ["all3"]:
        gg = g if b == "all3" else g[g.benchmark == b]
        t = gg.groupby(["iso"]).apply(lambda x: pd.Series({
            "n_io": int((x.ind_loop & ~x.rw_loop).sum()), "n_both": int((x.ind_loop & x.rw_loop).sum()), "n_neither": int((~x.ind_loop & ~x.rw_loop).sum()),
            "chrf_io": x[x.ind_loop & ~x.rw_loop].chrfpp.mean(), "chrf_both": x[x.ind_loop & x.rw_loop].chrfpp.mean(), "chrf_neither": x[~x.ind_loop & ~x.rw_loop].chrfpp.mean()}), include_groups=False)
        t = t[(t.n_io >= 10) & (t.n_neither >= 10)]
        diff = (t.chrf_io - t.chrf_neither)
        lm.append(dict(model=m, benchmark=b, n_lang=len(t), mean_chrf_ind_only=t.chrf_io.mean(), mean_chrf_neither=t.chrf_neither.mean(), mean_diff=diff.mean(),
                       n_lang_ind_only_worse=int((diff < 0).sum()), sd_diff=diff.std(),
                       se=diff.std() / np.sqrt(max(1, len(diff)))))
pd.DataFrame(lm).to_csv(os.path.join(HERE, "language_matched_quality_of_independent_only.csv"), index=False)

# how much chrF++ is lost on rows the reward detector does not flag but the independent one does (macro over languages: share x gap)
gain = []
for m in MAIN:
    g = D[m]
    for b in BENCH:
        gg = g[g.benchmark == b]
        per = []
        for iso, x in gg.groupby("iso"):
            n = len(x); io = x[x.ind_loop & ~x.rw_loop]; ne = x[~x.ind_loop & ~x.rw_loop]
            if len(io) and len(ne):
                per.append(len(io) / n * (ne.chrfpp.mean() - io.chrfpp.mean()))
            else:
                per.append(0.0)
        gain.append(dict(model=m, benchmark=b, macro_loss_if_ind_only_rows_were_average_chrfpp=float(np.mean(per)), macro_chrfpp_all_rows=float(gg.groupby("iso").chrfpp.mean().mean())))
pd.DataFrame(gain).to_csv(os.path.join(HERE, "chrf_loss_attributable_to_independent_only_rows.csv"), index=False)

# ---------------------------------------------------------------- 4. reference-side controls (tail, copy)
ctrl = []
for b in BENCH:
    g = R[R.benchmark == b]
    ctrl.append(dict(benchmark=b, ref_tail_unterminated=float(((g.e_ends_final) & g.r_ends_alpha).groupby(g.iso).mean().mean()),
                     ref_copy_like=float(((g.r_en_overlap >= 0.6) & (g.r_n_long_words >= 4)).groupby(g.iso).mean().mean()),
                     ref_rw_loop=float(g.r_rw_loop.groupby(g.iso).mean().mean()),
                     ref_len_over_eng_median=float((g.r_chars / g.e_chars.clip(lower=1)).median())))
pd.DataFrame(ctrl).to_csv(os.path.join(HERE, "reference_controls.csv"), index=False)

# ---------------------------------------------------------------- 5. >2x length rows: what are they
l2 = []
for m in MAIN:
    g = D[m]; x = g[g.ratio_ref_chars > 2]
    l2.append(dict(model=m, n_gt2=len(x), rate=len(x) / len(g), share_unfinished=float(x.unfinished.mean()), share_rw_loop=float(x.rw_loop.mean()), share_ind_loop=float(x.ind_loop.mean()),
                   share_neither=float((~x.rw_loop & ~x.ind_loop & x.finished).mean())))
pd.DataFrame(l2).to_csv(os.path.join(HERE, "length_gt2_rows_decomposition.csv"), index=False)

# ---------------------------------------------------------------- 6. worst languages (mixedbest FLORES+)
g = D["mixedbest"]; g = g[g.benchmark == "flores_plus"]
w = g.groupby("iso").agg(rw_loop=("rw_loop", "mean"), ind_loop=("ind_loop", "mean"), unfinished=("unfinished", "mean"), chrfpp=("chrfpp", "mean")).sort_values("ind_loop", ascending=False)
w.head(25).to_csv(os.path.join(HERE, "worst_languages_mixedbest_flores.csv"))
print("ok")
