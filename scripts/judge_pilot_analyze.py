"""Analysis of the J1 judge pilot: judge score vs human z-score, per pair / pooled / language-demeaned, bootstrap CIs, paired difference.

    python judge_pilot_analyze.py --items items.jsonl --results <dir with raw_<model>_<group>.jsonl> --ledger ledger.json --out <repo csv dir> \
        --judges gemini-3.1-pro-preview gemini-3.8-flash

Failed outputs (not a bare number in [0,100], or no text) get 50 (=0.5 on the paper's [0,1] scale) in the "paper" variant; the "dropped"
variant removes them. Both are reported. A "lenient" parse (first number in a reply with commentary) is given as a sensitivity check.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from judge_pilot_common import parse_score

PAPER_G25PRO_MTE = {"eng-amh": 0.605, "eng-hau": 0.471, "eng-kik": 0.735, "eng-kin": 0.528, "eng-luo": 0.782, "eng-twi": 0.710, "eng-yor": 0.524}
NB = 4000
rng = np.random.default_rng(2026)


def corr(x, y, kind):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return (spearmanr(x, y)[0] if kind == "spearman" else pearsonr(x, y)[0])


def boot_idx(groups):
    """groups: list of index arrays (one per pair). Resample within each pair; return NB x N index matrix over the concatenation."""
    cols = [rng.choice(g, size=(NB, len(g)), replace=True) for g in groups]
    return np.concatenate(cols, axis=1)


def stat_on(idx_row, df, col, kind, mode):
    d = df.iloc[idx_row]
    x, y = d[col].to_numpy(float), d["z"].to_numpy(float)
    if mode == "demeaned":
        x = x - d.groupby("lp")[col].transform("mean").to_numpy()
        y = y - d.groupby("lp")["z"].transform("mean").to_numpy()
    return corr(x, y, kind)


def ci(v):
    v = np.asarray(v, float)
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judges", nargs="+", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    items = pd.DataFrame([json.loads(l) for l in open(a.items, encoding="utf-8")]).drop(columns=["prompt", "demos", "src", "mt", "ref"])
    rows = []
    for m in a.judges:
        for g in ("probe", "main"):
            p = Path(a.results) / f"raw_{m}_{g}.jsonl"
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                r["judge"] = m
                rows.append(r)
    raw = pd.DataFrame(rows)
    raw["strict"] = raw["text"].map(parse_score)
    raw["lenient"] = raw["text"].map(lambda t: (float(m.group()) if t and (m := re.search(r"\d+(?:\.\d+)?", t)) and 0 <= float(m.group()) <= 100 else None))
    wide = items.copy()
    for m in a.judges:
        r = raw[raw.judge == m].set_index("id")
        wide[f"{m}__raw"] = wide["id"].map(r["strict"])
        wide[f"{m}__fail"] = wide[f"{m}__raw"].isna()
        wide[f"{m}__paper"] = wide[f"{m}__raw"].fillna(50.0)  # paper: failed -> 0.5 on [0,1]
        wide[f"{m}__lenient"] = wide["id"].map(r["lenient"]).fillna(50.0)
        wide[f"{m}__finish"] = wide["id"].map(r["finish"])
        wide[f"{m}__out_tokens"] = wide["id"].map(r["candidate_tokens"] + r["thought_tokens"])
        wide[f"{m}__thought_tokens"] = wide["id"].map(r["thought_tokens"])
    wide = wide.rename(columns={"human_z": "z"}) if "human_z" in wide else wide
    wide.to_csv(out / "judge_scores_by_segment.csv", index=False)

    # ---------- failures / tokens / cost
    led = json.load(open(a.ledger))
    cost_rows = []
    for m in a.judges:
        e = [x for x in led if x["model"] == m and x["status"] == "done"]
        d = raw[raw.judge == m]
        cost_rows.append(dict(judge=m, n=len(d), parse_failures=int(wide[f"{m}__fail"].sum()), finish_not_stop=int((~d["finish"].astype(str).str.contains("STOP")).sum()),
                              input_tokens=sum(x["actual_input_tokens"] for x in e), output_tokens_incl_thinking=sum(x["actual_output_tokens"] for x in e),
                              mean_thought_tokens=d["thought_tokens"].mean(), max_thought_tokens=d["thought_tokens"].max(), cost_usd=sum(x["actual_usd"] for x in e),
                              caps="/".join(str(x["cap"]) for x in e)))
    pd.DataFrame(cost_rows).to_csv(out / "cost_and_failures.csv", index=False)

    # ---------- correlations
    pairs = list(wide["lp"].unique())
    cols = {m: f"{m}__paper" for m in a.judges}
    cols.update({"SSA-COMET-MTL(ref)": "ssa_mtl_ref", "AfriCOMET-1.1-STL(ref)": "africomet_stl11_ref", "AfriCOMET-1.0-MTL(ref)": "africomet_mtl_ref"})
    res = []
    groups_by_pair = [np.where(wide["lp"].to_numpy() == p)[0] for p in pairs]
    for kind in ("spearman", "pearson"):
        for name, c in cols.items():
            for p in pairs + ["POOLED", "POOLED_DEMEANED", "SSA4_POOLED", "AFRI4_POOLED"]:
                if p in pairs:
                    sub = wide[wide.lp == p].reset_index(drop=True)
                    bi = rng.choice(len(sub), size=(NB, len(sub)))
                    v = corr(sub[c], sub["z"], kind)
                    bs = [corr(sub[c].to_numpy()[b], sub["z"].to_numpy()[b], kind) for b in bi]
                else:
                    mode = "demeaned" if p == "POOLED_DEMEANED" else "raw"
                    sub = wide if p.startswith("POOLED") else wide[wide["set"] == ("ssamte" if p.startswith("SSA") else "afrimte")]
                    sub = sub.reset_index(drop=True)
                    g = [np.where(sub["lp"].to_numpy() == q)[0] for q in sub["lp"].unique()]
                    bi = boot_idx(g)
                    v = stat_on(np.arange(len(sub)), sub, c, kind, mode)
                    bs = [stat_on(b, sub, c, kind, mode) for b in bi]
                lo, hi = ci(bs)
                res.append(dict(kind=kind, scorer=name, scope=p, n=len(sub), r=v, ci_lo=lo, ci_hi=hi))
    R = pd.DataFrame(res)
    R.to_csv(out / "correlations.csv", index=False)

    # ---------- paired difference Pro - Flash (same resamples), plus judge vs COMET
    if len(a.judges) == 2:
        pro, fl = a.judges
        diffs = []
        for kind in ("spearman", "pearson"):
            for p in pairs + ["POOLED", "POOLED_DEMEANED", "SSA4_POOLED", "AFRI4_POOLED"]:
                if p in pairs:
                    sub = wide[wide.lp == p].reset_index(drop=True)
                    bi = rng.choice(len(sub), size=(NB, len(sub)))
                    mode = "raw"
                else:
                    mode = "demeaned" if p == "POOLED_DEMEANED" else "raw"
                    sub = wide if p.startswith("POOLED") else wide[wide["set"] == ("ssamte" if p.startswith("SSA") else "afrimte")]
                    sub = sub.reset_index(drop=True)
                    bi = boot_idx([np.where(sub["lp"].to_numpy() == q)[0] for q in sub["lp"].unique()])
                for other, oc in ((fl, cols[fl]), ("SSA-COMET-MTL(ref)", "ssa_mtl_ref")):
                    d0 = stat_on(np.arange(len(sub)), sub, cols[pro], kind, mode) - stat_on(np.arange(len(sub)), sub, oc, kind, mode)
                    ds = [stat_on(b, sub, cols[pro], kind, mode) - stat_on(b, sub, oc, kind, mode) for b in bi]
                    lo, hi = ci(ds)
                    diffs.append(dict(kind=kind, scope=p, n=len(sub), contrast=f"{pro} minus {other}", diff=d0, ci_lo=lo, ci_hi=hi, boot_se=np.nanstd(ds)))
        pd.DataFrame(diffs).to_csv(out / "paired_differences.csv", index=False)

    # ---------- reproduction check against the paper's published Gemini-2.5-Pro MTE Spearman
    rep = []
    for p, v in PAPER_G25PRO_MTE.items():
        row = dict(pair=p, paper_gemini25pro_mte_spearman=v)
        for m in a.judges:
            q = R[(R.kind == "spearman") & (R.scorer == m) & (R.scope == p)]
            row[f"{m}_pilot_spearman"] = q["r"].iloc[0] if len(q) else np.nan
            row[f"{m}_n"] = q["n"].iloc[0] if len(q) else np.nan
        rep.append(row)
    pd.DataFrame(rep).to_csv(out / "paper_reproduction_check.csv", index=False)

    # ---------- sensitivity: dropped failures and lenient parse, pooled spearman
    sens = []
    for m in a.judges:
        ok = wide[~wide[f"{m}__fail"]]
        sens.append(dict(judge=m, variant="paper (fail->0.5)", pooled_spearman=corr(wide[f"{m}__paper"], wide["z"], "spearman"), n=len(wide)))
        sens.append(dict(judge=m, variant="failures dropped", pooled_spearman=corr(ok[f"{m}__raw"], ok["z"], "spearman"), n=len(ok)))
        sens.append(dict(judge=m, variant="lenient parse", pooled_spearman=corr(wide[f"{m}__lenient"], wide["z"], "spearman"), n=len(wide)))
    pd.DataFrame(sens).to_csv(out / "sensitivity_parse.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(R[(R.kind == "spearman")].pivot(index="scope", columns="scorer", values="r").round(3))
    print(pd.DataFrame(cost_rows))


if __name__ == "__main__":
    main()
