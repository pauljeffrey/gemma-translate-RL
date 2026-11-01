"""Run 2 analysis: FLORES+ devtest, RL-final minus SFT-final, judge = gemini-3.8-flash (Figure 8, reference-based, human FLORES+ reference).

    python judge_j2_run2_analyze.py --raw <raw_run2a.jsonl> <raw_run2b.jsonl> --sample <run2_sample.csv> --out <dir>
Paired by sentence (the same FLORES+ sentence under both systems); paired bootstrap over sentences (4,000 resamples, seed 2026, languages resampled
independently, macro = unweighted mean over languages). Failed parses (none expected) are dropped pairwise; the count is reported.
Same-sentence comparisons with the stored sentence-level chrF++ (sacrebleu CHRF word_order 2), AfriCOMET-STL-1.1 (reference-based) and SSA-COMET-MTL
(reference-based) of the paper's evaluation (box e1 / e2); AfriCOMET / SSA-COMET were NOT recomputed (no GPU) and SSA-COMET exists only for part of
the Hausa / Yoruba / Igbo sentences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from judge_pilot_common import parse_score

LANGS = "amh hau kin yor swh ibo som xho".split()
NB = 4000
rng = np.random.default_rng(2026)


def boot_mean_diff(d_by_lang, nb=NB):
    """d_by_lang: dict lang -> 1-D array of paired differences. Returns (per-lang mean, per-lang CI, macro mean, macro CI, pooled mean, pooled CI)."""
    idx = {l: rng.integers(0, len(v), size=(nb, len(v))) for l, v in d_by_lang.items()}
    per = {l: d_by_lang[l][idx[l]].mean(1) for l in d_by_lang}
    macro = np.mean([per[l] for l in d_by_lang], axis=0)
    tot = sum(d_by_lang[l][idx[l]].sum(1) for l in d_by_lang)
    n = sum(len(v) for v in d_by_lang.values())
    pooled = tot / n
    ci = lambda x: tuple(np.percentile(x, [2.5, 97.5]))
    return ({l: ci(per[l]) for l in per}, ci(macro), ci(pooled))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    raw = pd.concat([pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")]) for p in a.raw])
    raw["score"] = raw["text"].map(parse_score)
    parts = raw["id"].str.split(":", expand=True)  # flores:lang:system:sid
    raw["lang"], raw["system"], raw["sid"] = parts[1], parts[2], parts[3].astype(int)
    fails = int(raw["score"].isna().sum())
    nonstop = int((~raw["finish"].astype(str).str.contains("STOP")).sum())
    smp = pd.read_csv(a.sample)
    sc = raw.pivot(index=["lang", "sid"], columns="system", values="score").reset_index().rename(columns={"sft": "j_sft", "rl": "j_rl"})
    df = smp.merge(sc, on=["lang", "sid"], how="left")
    assert len(df) == len(smp) and len(raw) == 2 * len(smp), (len(df), len(raw), len(smp))
    df["pair_ok"] = df["j_sft"].notna() & df["j_rl"].notna()
    df["j_diff"] = df["j_rl"] - df["j_sft"]
    df["chrf_diff"] = df["rl_chrf"] - df["sft_chrf"]
    df["afri_diff"] = (df["rl_afri"] - df["sft_afri"]) * 100  # AfriCOMET on a 0-100 scale for readability
    df["ssa_diff"] = (df["rl_ssa"] - df["sft_ssa"]) * 100
    df.to_csv(out / "run2_judge_scores_by_sentence.csv", index=False)
    ok = df[df.pair_ok]

    d = {l: ok[ok.lang == l]["j_diff"].to_numpy(float) for l in LANGS}
    cis_lang, ci_macro, ci_pool = boot_mean_diff(d)
    rows = []
    for l in LANGS:
        g = ok[ok.lang == l]
        x = g["j_diff"].to_numpy(float)
        rows.append(dict(lang=l, n=len(g), sft_mean=g["j_sft"].mean(), rl_mean=g["j_rl"].mean(), diff=x.mean(), ci_lo=cis_lang[l][0], ci_hi=cis_lang[l][1],
                         win=int((x > 0).sum()), tie=int((x == 0).sum()), loss=int((x < 0).sum()),
                         win_m5=int((x > 5).sum()), tie_m5=int((np.abs(x) <= 5).sum()), loss_m5=int((x < -5).sum()),
                         sft_median=g["j_sft"].median(), rl_median=g["j_rl"].median()))
    T = pd.DataFrame(rows)
    macro = dict(lang="MACRO", n=int(T.n.sum()), sft_mean=T.sft_mean.mean(), rl_mean=T.rl_mean.mean(), diff=T["diff"].mean(), ci_lo=ci_macro[0], ci_hi=ci_macro[1],
                 win=int(T.win.sum()), tie=int(T.tie.sum()), loss=int(T.loss.sum()), win_m5=int(T.win_m5.sum()), tie_m5=int(T.tie_m5.sum()), loss_m5=int(T.loss_m5.sum()))
    allx = ok["j_diff"].to_numpy(float)
    pooled = dict(lang="POOLED_SENTENCES", n=len(ok), sft_mean=ok.j_sft.mean(), rl_mean=ok.j_rl.mean(), diff=allx.mean(), ci_lo=ci_pool[0], ci_hi=ci_pool[1],
                  win=int((allx > 0).sum()), tie=int((allx == 0).sum()), loss=int((allx < 0).sum()), win_m5=int((allx > 5).sum()), tie_m5=int((np.abs(allx) <= 5).sum()), loss_m5=int((allx < -5).sum()))
    T = pd.concat([T, pd.DataFrame([macro, pooled])], ignore_index=True)
    T.to_csv(out / "run2_judge_by_language.csv", index=False)

    # same sentences, other metrics (paired bootstrap with the same machinery; sentences with the metric available for both systems)
    other = []
    for name, col in (("chrF++ (sentence)", "chrf_diff"), ("AfriCOMET-STL-1.1 (ref) x100", "afri_diff"), ("SSA-COMET-MTL (ref) x100", "ssa_diff"), ("judge", "j_diff")):
        sub = ok[ok[col].notna()]
        dd = {l: sub[sub.lang == l][col].to_numpy(float) for l in LANGS if (sub.lang == l).sum() >= 10}
        cl, cm, cp = boot_mean_diff(dd)
        for l in dd:
            x = dd[l]
            other.append(dict(metric=name, scope=l, n=len(x), mean_diff=x.mean(), ci_lo=cl[l][0], ci_hi=cl[l][1], win=int((x > 0).sum()), tie=int((x == 0).sum()), loss=int((x < 0).sum())))
        other.append(dict(metric=name, scope="MACRO", n=sum(len(v) for v in dd.values()), mean_diff=np.mean([v.mean() for v in dd.values()]), ci_lo=cm[0], ci_hi=cm[1],
                          win=sum(int((v > 0).sum()) for v in dd.values()), tie=sum(int((v == 0).sum()) for v in dd.values()), loss=sum(int((v < 0).sum()) for v in dd.values())))
    O = pd.DataFrame(other)
    O.to_csv(out / "run2_same_sentences_metric_differences.csv", index=False)

    # agreement of the paired differences (sentence level) and of the language-level differences
    agr = []
    for col in ("chrf_diff", "afri_diff", "ssa_diff"):
        sub = ok[ok[col].notna()]
        r = spearmanr(sub["j_diff"], sub[col])[0]
        sign = float(((np.sign(sub["j_diff"]) == np.sign(sub[col])) & (sub["j_diff"] != 0) & (sub[col] != 0)).sum() / max(((sub["j_diff"] != 0) & (sub[col] != 0)).sum(), 1))
        agr.append(dict(metric=col, n=len(sub), spearman_of_paired_differences=r, sign_agreement_when_both_nonzero=sign))
    for col in ("chrf_diff", "afri_diff", "ssa_diff"):
        m = O[(O.metric.str.startswith({"chrf_diff": "chrF", "afri_diff": "AfriCOMET", "ssa_diff": "SSA"}[col])) & (O.scope != "MACRO")].set_index("scope")["mean_diff"]
        jm = T[T.lang.isin(m.index)].set_index("lang")["diff"]
        agr.append(dict(metric=col + " (language-level means)", n=len(m), spearman_of_paired_differences=spearmanr(jm.loc[m.index], m)[0], sign_agreement_when_both_nonzero=float((np.sign(jm.loc[m.index]) == np.sign(m)).mean())))
    pd.DataFrame(agr).to_csv(out / "run2_agreement_with_other_metrics.csv", index=False)

    # clean subset (English side never seen): pooled and per language, small n
    cl = ok[~ok.contaminated.astype(bool)]
    clean = []
    for l in LANGS:
        x = cl[cl.lang == l]["j_diff"].to_numpy(float)
        clean.append(dict(scope=l, n=len(x), judge_diff=x.mean() if len(x) else np.nan, chrf_diff=cl[cl.lang == l]["chrf_diff"].mean()))
    xs = cl["j_diff"].to_numpy(float)
    bs = [rng.choice(xs, len(xs)).mean() for _ in range(NB)]
    clean.append(dict(scope="POOLED_CLEAN", n=len(xs), judge_diff=xs.mean(), chrf_diff=cl["chrf_diff"].mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    pd.DataFrame(clean).to_csv(out / "run2_clean_subset.csv", index=False)

    diag = dict(n_requests=len(raw), parse_failures=fails, finish_not_stop=nonstop, n_pairs_ok=int(df.pair_ok.sum()), mean_thought_tokens=float(raw.thought_tokens.mean()), max_thought_tokens=int(raw.thought_tokens.max()))
    json.dump(diag, open(out / "run2_diagnostics.json", "w"), indent=1)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(T.round(2).to_string())
    print(O.round(2).to_string())
    print(pd.DataFrame(agr).round(3).to_string())
    print(pd.DataFrame(clean).round(2).to_string())
    print(diag)


if __name__ == "__main__":
    main()
