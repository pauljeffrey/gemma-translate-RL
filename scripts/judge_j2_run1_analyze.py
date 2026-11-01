"""Run 1 analysis: Figure 9 (QE) Flash judge vs the human z-score, compared with the same judge on the reference-based prompt (pilot),
SSA-COMET-MTL-QE / SSA-COMET-QE and the reference-based SSA-COMET-MTL. Same statistics as judge_pilot_analyze.py (Spearman / Pearson per pair,
pooled, pooled with per-pair means removed, bootstrap over segments within pairs, 4,000 resamples, the SAME resamples for every scorer so that
differences are paired).

    python judge_j2_run1_analyze.py --items <scratch items_run1.jsonl> --qe-raw <raw_run1_qe.jsonl> --pilot-dir <reports/data/judge_pilot_2026-10-25> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import judge_pilot_analyze as P
from judge_pilot_common import parse_score

PAPER_QE_G25PRO_AVG = 0.610  # as published (Li et al., 2025), Gemini-2.5-Pro QE, average Spearman; not recomputed
LD6 = ["eng-amh", "eng-hau", "eng-kin", "eng-yor", "eng-swh", "eng-ibo"]
SCORERS = {
    "Flash QE (Fig.9, no reference)": "qe_judge",
    "Flash ref-based (Fig.8, pilot)": "ref_judge",
    "SSA-COMET-MTL-QE": "ssa_mtl_qe",
    "SSA-COMET-QE (ssa_qe)": "ssa_qe",
    "SSA-COMET-MTL (ref-based)": "ssa_mtl_ref",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--qe-raw", required=True)
    ap.add_argument("--pilot-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    items = pd.DataFrame([json.loads(l) for l in open(a.items, encoding="utf-8")]).drop(columns=["prompt"])
    qe = pd.DataFrame([json.loads(l) for l in open(a.qe_raw, encoding="utf-8")])
    qe["strict"] = qe["text"].map(parse_score)
    ref = pd.concat([pd.DataFrame([json.loads(l) for l in open(Path(a.pilot_dir) / f"raw_gemini-3.8-flash_{g}.jsonl", encoding="utf-8")]) for g in ("probe", "main")])
    ref["strict"] = ref["text"].map(parse_score)
    w = items.copy()
    w["qe_raw"] = w["id"].map(qe.set_index("id")["strict"])
    w["qe_fail"] = w["qe_raw"].isna()
    w["qe_judge"] = w["qe_raw"].fillna(50.0)  # paper rule: failure -> 0.5 on [0,1]
    w["ref_judge"] = w["id"].map(ref.set_index("id")["strict"]).fillna(50.0)
    w["qe_finish"] = w["id"].map(qe.set_index("id")["finish"])
    w["qe_thought_tokens"] = w["id"].map(qe.set_index("id")["thought_tokens"])
    w = w.drop(columns=["batch"]).rename(columns={"set": "set"})
    w.to_csv(out / "run1_judge_scores_by_segment.csv", index=False)
    assert len(w) == 300 and w["qe_judge"].notna().all()

    pairs = list(w["lp"].unique())
    scopes = pairs + ["POOLED", "POOLED_DEMEANED", "SSA4_POOLED", "AFRI4_POOLED", "LD6_POOLED", "LD6_DEMEANED"]

    def sub_of(p):
        if p in pairs:
            return w[w.lp == p].reset_index(drop=True), "raw"
        if p.startswith("POOLED"):
            return w.reset_index(drop=True), ("demeaned" if p == "POOLED_DEMEANED" else "raw")
        if p.startswith("LD6"):
            return w[w.lp.isin(LD6)].reset_index(drop=True), ("demeaned" if p == "LD6_DEMEANED" else "raw")
        return w[w["set"] == ("ssamte" if p.startswith("SSA") else "afrimte")].reset_index(drop=True), "raw"

    res, diffs = [], []
    for kind in ("spearman", "pearson"):
        for p in scopes:
            sub, mode = sub_of(p)
            if p in pairs:
                bi = P.rng.choice(len(sub), size=(P.NB, len(sub)))
            else:
                bi = P.boot_idx([np.where(sub["lp"].to_numpy() == q)[0] for q in sub["lp"].unique()])
            boots = {}
            for name, c in SCORERS.items():
                if p in pairs:
                    v = P.corr(sub[c], sub["z"], kind)
                    bs = np.array([P.corr(sub[c].to_numpy()[b], sub["z"].to_numpy()[b], kind) for b in bi])
                else:
                    v = P.stat_on(np.arange(len(sub)), sub, c, kind, mode)
                    bs = np.array([P.stat_on(b, sub, c, kind, mode) for b in bi])
                boots[name] = (v, bs)
                lo, hi = P.ci(bs)
                res.append(dict(kind=kind, scorer=name, scope=p, n=len(sub), r=v, ci_lo=lo, ci_hi=hi))
            base = boots["Flash QE (Fig.9, no reference)"]
            for other in ("Flash ref-based (Fig.8, pilot)", "SSA-COMET-MTL-QE", "SSA-COMET-QE (ssa_qe)"):
                d0 = base[0] - boots[other][0]
                ds = base[1] - boots[other][1]
                lo, hi = P.ci(ds)
                diffs.append(dict(kind=kind, scope=p, n=len(sub), contrast=f"Flash QE minus {other}", diff=d0, ci_lo=lo, ci_hi=hi))
    R = pd.DataFrame(res)
    R.to_csv(out / "run1_correlations.csv", index=False)
    pd.DataFrame(diffs).to_csv(out / "run1_paired_differences.csv", index=False)

    # usability rule: per language, the 95% CI of the QE judge's Spearman must exclude values below 0.2 (lower bound >= 0.2)
    q = R[(R.kind == "spearman") & (R.scorer == "Flash QE (Fig.9, no reference)")].set_index("scope")
    usable = pd.DataFrame({"n": q.loc[pairs, "n"], "spearman": q.loc[pairs, "r"], "ci_lo": q.loc[pairs, "ci_lo"], "ci_hi": q.loc[pairs, "ci_hi"]})
    usable["usable_ci_lo_ge_0.2"] = usable["ci_lo"] >= 0.2
    usable.to_csv(out / "run1_language_usability.csv")

    # score distribution diagnostics: the QE judge vs the ref judge on the same segments
    diag = dict(n=len(w), qe_parse_failures=int(w["qe_fail"].sum()), qe_finish_not_stop=int((~w["qe_finish"].astype(str).str.contains("STOP")).sum()),
                mean_qe=float(w["qe_judge"].mean()), mean_ref=float(w["ref_judge"].mean()), sd_qe=float(w["qe_judge"].std()), sd_ref=float(w["ref_judge"].std()),
                spearman_qe_vs_ref_judge=float(P.corr(w["qe_judge"], w["ref_judge"], "spearman")),
                mean_thought_tokens=float(w["qe_thought_tokens"].mean()), max_thought_tokens=float(w["qe_thought_tokens"].max()))
    json.dump(diag, open(out / "run1_diagnostics.json", "w"), indent=1)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(R[R.kind == "spearman"].pivot(index="scope", columns="scorer", values="r").round(3).loc[scopes])
    print(R[(R.kind == "spearman") & (R.scorer == "Flash QE (Fig.9, no reference)")][["scope", "n", "r", "ci_lo", "ci_hi"]].round(3).to_string())
    print(usable.round(3))
    print(pd.DataFrame(diffs).query("kind=='spearman'")[["scope", "contrast", "diff", "ci_lo", "ci_hi"]].round(3).to_string())
    print(diag)


if __name__ == "__main__":
    main()
