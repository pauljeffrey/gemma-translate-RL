"""Run 3 analysis: long documents judged on aligned units with the QE prompt (Figure 9), gemini-3.8-flash.

    python judge_j2_run3_analyze.py --raw <raw_run3a.jsonl> <raw_run3b.jsonl> --dir <reports/data/judge_j2_2026-10-25>

Statistics (all CIs: document-cluster bootstrap, 4,000 resamples, seed 2026; the 40 documents of a language are resampled with replacement, languages
independently; macro = unweighted mean over the languages that the system / both systems of a pair have):
 (a) mean judge score on JUDGED UNITS (a block covering several sampled sentences is counted once) per system and language;
 (b) omission rate = share of sampled source sentences without a judged unit (omitted, copied-from-source or empty, the LC1 coverage rule), and the
     OMISSION-PENALISED mean = mean over ALL sampled sentences, an omitted sentence counting 0 and a sentence in a judged unit taking that unit's score;
 (c)/(d) paired differences on the same sampled source sentences: "both covered" = sentences with a judged unit under both systems; "penalised" = all
     sampled sentences, omission = 0;
 (e) share of sampled units / sentences in documents that LC1 flagged as loop or unfinished (stored LC1 scores), and the mean score inside / outside them.
A failed parse (not a bare number in [0,100]) removes the unit from (a) and the paired "both covered" sets and counts 0.5-style 50 nowhere: the number is reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from judge_pilot_common import parse_score

NB = 4000
rng = np.random.default_rng(2026)
LANGS = "hau yor swh amh ibo kin".split()
SYS = ["final2_whole", "final2_sentfix", "rl_whole", "nllb3b3_sent", "tpsy08_sent"]
PAIRS = [("final2_whole", "final2_sentfix"), ("final2_whole", "rl_whole"), ("final2_whole", "nllb3b3_sent"), ("final2_sentfix", "nllb3b3_sent"),
         ("final2_whole", "tpsy08_sent"), ("final2_sentfix", "tpsy08_sent"), ("rl_whole", "nllb3b3_sent")]
ci = lambda x: tuple(np.percentile(x, [2.5, 97.5]))


class Boot:
    """Document-cluster bootstrap for ratio-of-sums statistics; draws are shared across all statistics of a language (so differences are paired)."""

    def __init__(self, docs_by_lang):
        self.docs = {l: sorted(v) for l, v in docs_by_lang.items()}
        self.W = {l: rng.multinomial(len(v), np.full(len(v), 1 / len(v)), size=NB).astype(float) for l, v in self.docs.items()}

    def vec(self, lang, df, col_num, col_den=None):
        g = df.groupby("doc_id")
        num = g[col_num].sum().reindex(self.docs[lang]).fillna(0).to_numpy(float)
        den = (g[col_den].sum() if col_den else g[col_num].count()).reindex(self.docs[lang]).fillna(0).to_numpy(float)
        return num, den

    def ratio(self, lang, num, den):
        n, d = self.W[lang] @ num, self.W[lang] @ den
        with np.errstate(invalid="ignore", divide="ignore"):
            b = n / d
        return (num.sum() / den.sum() if den.sum() > 0 else np.nan), b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", required=True)
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    D = Path(a.dir)
    raw = pd.concat([pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")]) for p in a.raw])
    raw["score"] = raw["text"].map(parse_score)
    units = pd.read_csv(D / "run3_units.csv").merge(raw[["id", "score", "finish", "thought_tokens"]], on="id", how="left")
    assert units["finish"].notna().all(), "units without a response"
    cov = pd.read_csv(D / "run3_coverage_by_sampled_sentence.csv")
    cov["score"] = cov["unit_id"].map(units.set_index("id")["score"])
    units.to_csv(D / "run3_units_scored.csv", index=False)
    cov.to_csv(D / "run3_sentence_level_scores.csv", index=False)
    fails = int(units["score"].isna().sum())
    docs_by_lang = {l: set(cov[cov.lang == l].doc_id) for l in LANGS}
    # the 40 documents per language are the cluster set (every sampled document is in the sample of every system)
    B = Boot(docs_by_lang)

    # ---------------- (a) judged-unit means, (b) omission
    rowsA, rowsB = [], []
    for s in SYS:
        per_unit, per_pen, per_om = {}, {}, {}
        for l in LANGS:
            u = units[(units.system == s) & (units.lang == l) & units.score.notna()]
            c = cov[(cov.system == s) & (cov.lang == l)]
            if c.empty:
                continue
            nu, du = B.vec(l, u, "score")
            v, b = B.ratio(l, nu, du)
            per_unit[l] = (v, b)
            c = c.assign(pen=c["score"].fillna(0.0).where(c.covered == 1, 0.0), om=1 - c.covered, sw=c["score"].where(c.covered == 1))
            c["swn"] = c["sw"].notna().astype(float)
            c["sw0"] = c["sw"].fillna(0.0)
            n_, d_ = B.vec(l, c, "pen", None)
            per_pen[l] = B.ratio(l, n_, d_)
            n_, d_ = B.vec(l, c, "om", None)
            per_om[l] = B.ratio(l, n_, d_)
            n_, d_ = B.vec(l, c, "sw0", "swn")
            sw = B.ratio(l, n_, d_)
            lo, hi = ci(b)
            rowsA.append(dict(system=s, lang=l, n_sampled_sentences=len(c), n_units_judged=len(u), mean_block_sents=float(u.block_sents.mean()), unit_mean=v, ci_lo=lo, ci_hi=hi,
                              sentence_weighted_mean_covered=sw[0], parse_failures=int(units[(units.system == s) & (units.lang == l)].score.isna().sum())))
            lo2, hi2 = ci(per_pen[l][1])
            lo3, hi3 = ci(per_om[l][1])
            rowsB.append(dict(system=s, lang=l, omission_rate=per_om[l][0], om_ci_lo=lo3, om_ci_hi=hi3, penalised_mean=per_pen[l][0], pen_ci_lo=lo2, pen_ci_hi=hi2))
        for rows, store, keys in ((rowsA, per_unit, ("unit_mean", "ci_lo", "ci_hi")), (rowsB, per_pen, ("penalised_mean", "pen_ci_lo", "pen_ci_hi"))):
            ls = list(store)
            for tag, sel in (("MACRO(all langs of system)", ls), ("MACRO(5 common langs, no kin)", [l for l in ls if l != "kin"])):
                m = np.mean([store[l][0] for l in sel])
                lo, hi = ci(np.mean([store[l][1] for l in sel], axis=0))
                base = dict(system=s, lang=tag)
                if rows is rowsA:
                    rows.append({**base, keys[0]: m, keys[1]: lo, keys[2]: hi, "n_langs": len(sel)})
                else:
                    omm = np.mean([per_om[l][0] for l in sel])
                    olo, ohi = ci(np.mean([per_om[l][1] for l in sel], axis=0))
                    rows.append({**base, "omission_rate": omm, "om_ci_lo": olo, "om_ci_hi": ohi, keys[0]: m, keys[1]: lo, keys[2]: hi, "n_langs": len(sel)})
    TA, TB = pd.DataFrame(rowsA), pd.DataFrame(rowsB)
    TA.to_csv(D / "run3_table_a_unit_means.csv", index=False)
    TB.to_csv(D / "run3_table_b_omission.csv", index=False)

    # ---------------- (c), (d) paired
    rowsP = []
    for sa, sb in PAIRS:
        per = {"both": {}, "pen": {}}
        for l in LANGS:
            ca, cb = (cov[(cov.system == s) & (cov.lang == l)].set_index(["doc_id", "sent_idx"]) for s in (sa, sb))
            if ca.empty or cb.empty:
                continue
            j = ca[["score", "covered"]].join(cb[["score", "covered"]], lsuffix="_a", rsuffix="_b", how="inner").reset_index()
            both = j[j.score_a.notna() & j.score_b.notna()].copy()
            both["d"] = both.score_a - both.score_b
            j["pa"] = j.score_a.where(j.covered_a == 1, 0.0).fillna(0.0)
            j["pb"] = j.score_b.where(j.covered_b == 1, 0.0).fillna(0.0)
            j["dp"] = j.pa - j.pb
            for kind, df_ in (("both", both), ("pen", j)):
                col = "d" if kind == "both" else "dp"
                nn, dd = B.vec(l, df_, col)
                per[kind][l] = B.ratio(l, nn, dd) + (len(df_),)
                v, b, n = per[kind][l]
                lo, hi = ci(b)
                x = df_[col].to_numpy(float)
                rowsP.append(dict(A=sa, B=sb, lang=l, set=("both covered" if kind == "both" else "omission-penalised (all sampled)"), n_sentences=n, mean_A_minus_B=v, ci_lo=lo, ci_hi=hi,
                                  A_better=int((x > 0).sum()), tie=int((x == 0).sum()), B_better=int((x < 0).sum())))
        for kind in ("both", "pen"):
            ls = list(per[kind])
            for tag, sel in (("MACRO(langs of both)", ls), ("MACRO(5 common langs, no kin)", [l for l in ls if l != "kin"])):
                m = np.mean([per[kind][l][0] for l in sel])
                lo, hi = ci(np.mean([per[kind][l][1] for l in sel], axis=0))
                rowsP.append(dict(A=sa, B=sb, lang=tag, set=("both covered" if kind == "both" else "omission-penalised (all sampled)"), n_sentences=sum(per[kind][l][2] for l in sel),
                                  mean_A_minus_B=m, ci_lo=lo, ci_hi=hi))
    TP = pd.DataFrame(rowsP)
    TP.to_csv(D / "run3_table_c_d_paired.csv", index=False)

    # ---------------- (e) loops / unfinished documents, block sizes
    rowsE = []
    for s in SYS:
        c = cov[cov.system == s]
        u = units[units.system == s]
        flag_u = (u.doc_loop == 1) | (u.doc_finished == 0)
        flag_c = (c.doc_loop == 1) | (c.doc_finished == 0)
        rowsE.append(dict(system=s, sampled_sentences=len(c), sentences_in_loop_or_unfinished_docs=int(flag_c.sum()), share_sentences=float(flag_c.mean()),
                          units=len(u), units_in_flagged_docs=int(flag_u.sum()), share_units=float(flag_u.mean()),
                          sentences_in_loop_docs=int((c.doc_loop == 1).sum()), sentences_in_unfinished_docs=int((c.doc_finished == 0).sum()),
                          unit_mean_flagged=u[flag_u].score.mean(), unit_mean_unflagged=u[~flag_u].score.mean(),
                          omission_rate_flagged=float(1 - c[flag_c].covered.mean()) if flag_c.any() else np.nan, omission_rate_unflagged=float(1 - c[~flag_c].covered.mean())))
    pd.DataFrame(rowsE).to_csv(D / "run3_table_e_loops.csv", index=False)
    rowsK = []
    for s in SYS:
        u = units[units.system == s]
        for tag, m in (("1 sentence", u.block_sents == 1), ("2+ sentences", u.block_sents >= 2), ("shortfall bead", u.shortfall == 1)):
            if m.any():
                rowsK.append(dict(system=s, unit_type=tag, n=int(m.sum()), mean_score=u[m].score.mean()))
    pd.DataFrame(rowsK).to_csv(D / "run3_block_size_breakdown.csv", index=False)
    diag = dict(n_requests=len(units), parse_failures=fails, finish_not_stop=int((~units["finish"].astype(str).str.contains("STOP")).sum()),
                mean_thought_tokens=float(units.thought_tokens.mean()), max_thought_tokens=float(units.thought_tokens.max()))
    json.dump(diag, open(D / "run3_diagnostics.json", "w"), indent=1)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500)
    print(TA.round(3).to_string()); print(TB.round(3).to_string()); print(TP.round(3).to_string()); print(pd.DataFrame(rowsE).round(3).to_string()); print(pd.DataFrame(rowsK).round(2).to_string()); print(diag)


if __name__ == "__main__":
    main()
