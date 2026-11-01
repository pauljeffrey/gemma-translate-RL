#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""H1 step 5: do the two judges of the ensemble (AfriCOMET-QE, SSA-COMET-MTL) disagree more about final2's outputs than about RL-final's? What is stored decides what can be done:

  (A) per row, reference-based judges (AfriCOMET with reference, SSA-COMET with reference), all five systems on FLORES+ and SFT-final / RL-final / final2 on the other benchmarks.
      Each metric is z-scored per language over the pooled rows of all systems being compared (so the scale is common); gap = z(AfriCOMET) - z(SSA-COMET);
      reported: share of rows with |gap| > 1, mean gap, share where AfriCOMET rates higher / lower by more than 1 z. Paired difference to RL-final with a bootstrap over rows.
      Sources: box/jobs/t1/out/chunks_*.csv.gz (SFT-final, RL-final, A2W, step 250; E1/T1 scoring pipeline) and the Hub dataset (final2; R20 pipeline; on identical hypotheses the two
      pipelines differ by < 0.006, g1_identical_hypothesis_check.txt).
  (B) reference-free judges, cell level (benchmark x language): RL-final's and SFT-final's per-row QE scores are NOT in the archive (only the aggregates of the E1 sample in
      box/jobs/e1/summary/unit_stats.csv); final2's per-row QE scores are. The E1 sample (79,062 rows: every `clean` row + a hash-ordered fill to 400 per benchmark and language, seed 1)
      is rebuilt exactly (pairwise_neural.select_keys) and final2's mean on the SAME rows is compared with the stored RL-final mean: QE-minus-reference gap per metric family
      (does a judge that does not see the reference rate the output higher than the judge that does?), and the cross-judge difference.
  (C) final2 only, per row: AfriCOMET-QE and SSA-COMET-QE shaped with the final run's per-language curves (calibration shaping_flores_dev_v8.json: anchors, qe2.anchors) -> r1, r2,
      share of rows with |r1 - r2| > 1 (the branch where the reward takes the minimum), compared with the training log of the run (reward/qe_ens/min_share).
"""
import argparse
import hashlib
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardening_evidence_common as C  # noqa: E402
from gemma_mt.rl.reward_shaping import shape  # noqa: E402

warnings.filterwarnings("ignore")
T1 = os.path.join(C.JOBS, "t1", "out")
SYS_T1 = {"sft": ("comet_sft", "ssa_sft"), "rl": ("comet_rl", "ssa_rl"), "a2w": ("comet_a2w", "ssa_a2w"), "live250": ("comet_live250", "ssa_live250")}


def load_ref_scores(groups: pd.DataFrame) -> pd.DataFrame:
    ind = pd.read_parquet(os.path.join(C.WORK, "indicators.parquet"), columns=["system", "bench", "iso", "id", "pair_seen", "afri_ref", "ssa_ref", "afri_qe", "ssa_qe"])
    f2 = ind[ind.system == "final2"].drop_duplicates(["bench", "iso", "id"])
    parts = [f2[["system", "bench", "iso", "id", "pair_seen", "afri_ref", "ssa_ref"]]]
    for b in ("flores_plus", "ntrex", "smol_sent", "mafand", "glot_story", "salt"):
        t = pd.read_csv(os.path.join(T1, f"chunks_{b}.csv.gz"))
        t = t[t.k == 1].copy()
        t["id"] = t.uid.str.split("|").str[3]
        for s, (ca, cs) in SYS_T1.items():
            if ca not in t:
                continue
            x = t[["b", "iso", "id", ca, cs, "n_pair_seen"]].rename(columns={"b": "bench", ca: "afri_ref", cs: "ssa_ref"})
            x["system"] = s
            x["pair_seen"] = x["n_pair_seen"] > 0
            parts.append(x[["system", "bench", "iso", "id", "pair_seen", "afri_ref", "ssa_ref"]])
    d = pd.concat(parts, ignore_index=True)
    d = d[~d.pair_seen]
    d = d[d.iso.isin(groups.iso)]
    return d


def zscore_scope(d: pd.DataFrame, systems, benches):
    x = d[d.system.isin(systems) & d.bench.isin(benches)]
    w = x.pivot_table(index=["bench", "iso", "id"], columns="system", values=["afri_ref", "ssa_ref"])
    w = w.dropna()                                                # complete case: all systems have both metrics
    out = []
    for s in systems:
        out.append(pd.DataFrame({"bench": w.index.get_level_values(0), "iso": w.index.get_level_values(1), "id": w.index.get_level_values(2), "system": s,
                                 "afri": w[("afri_ref", s)].values, "ssa": w[("ssa_ref", s)].values}))
    z = pd.concat(out, ignore_index=True)
    for m in ("afri", "ssa"):
        g = z.groupby("iso")[m]
        z["z_" + m] = (z[m] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    z["gap"] = z["z_afri"] - z["z_ssa"]
    z["abs_gap_gt1"] = (z["gap"].abs() > 1).astype(float)
    z["afri_higher"] = (z["gap"] > 1).astype(float)
    z["ssa_higher"] = (z["gap"] < -1).astype(float)
    return z


def macro_boot(z: pd.DataFrame, sysA: str, sysB: str, cols, B=1000, seed=7):
    a = z[z.system == sysA].set_index(["bench", "iso", "id"])
    b = z[z.system == sysB].set_index(["bench", "iso", "id"])
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    isos = a.index.get_level_values(1).values
    res = {}
    rng = np.random.default_rng(seed)
    per = {}
    for iso in np.unique(isos):
        m = isos == iso
        D = (b[cols].values[m] - a[cols].values[m])
        per[iso] = D
    pt = np.mean([np.nanmean(D, 0) for D in per.values()], axis=0)
    acc = np.zeros((B, len(cols)))
    for iso, D in per.items():
        n = len(D)
        for s in range(0, B, 100):
            e = min(B, s + 100)
            ii = rng.integers(0, n, size=(e - s, n))
            acc[s:e] += np.nanmean(D[ii], axis=1)
    acc /= len(per)
    lo, hi = np.percentile(acc, [2.5, 97.5], axis=0)
    return pt, lo, hi, len(per), len(idx)


def part_A(groups):
    d = load_ref_scores(groups)
    rows = []
    cols = ["z_afri", "z_ssa", "gap", "abs_gap_gt1", "afri_higher", "ssa_higher"]
    scopes = {"FLORES+ (5 systems)": (("sft", "rl", "a2w", "live250", "final2"), ["flores_plus"]),
              "FLORES+ (3 systems)": (("sft", "rl", "final2"), ["flores_plus"]),
              "other benchmarks (3 systems)": (("sft", "rl", "final2"), ["ntrex", "mafand", "smol_sent", "salt", "glot_story"]),
              "all benchmarks (3 systems)": (("sft", "rl", "final2"), ["flores_plus", "ntrex", "mafand", "smol_sent", "salt", "glot_story"])}
    gm = groups.set_index("iso")
    for name, (systems, benches) in scopes.items():
        z = zscore_scope(d, systems, benches)
        z["tier"] = z.iso.map(gm["tier"])
        for tier in ["all"] + sorted(z.tier.dropna().unique()):
            zz = z if tier == "all" else z[z.tier == tier]
            if not len(zz):
                continue
            for s in systems:
                x = zz[zz.system == s]
                per = x.groupby("iso")[cols].mean()
                rows.append(dict(scope=name, tier=tier, system=s, kind="level", n_rows=len(x), n_lang=x.iso.nunique(), **{c: per[c].mean() for c in cols}))
            ref = "rl"
            for s in systems:
                if s == ref:
                    continue
                pt, lo, hi, nl, nr = macro_boot(zz, ref, s, cols)
                rec = dict(scope=name, tier=tier, system=s, kind=f"{s}_minus_{ref}", n_rows=nr, n_lang=nl)
                for k, c in enumerate(cols):
                    rec[c] = pt[k]
                    rec[c + "_lo"], rec[c + "_hi"] = lo[k], hi[k]
                rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.OUT, "judges_A_reference_based_row_level.csv"), index=False, float_format="%.5g")
    return out


def e1_sample_keys():
    p = os.path.join(C.WORK, "e1_neural_sample_keys.json")
    if os.path.exists(p):
        return {tuple(k) for k in json.load(open(p))}
    from gemma_mt.languages import LANGUAGES

    d = pd.read_parquet(os.path.join(C.WORK, "rows.parquet"), columns=["system", "bench", "iso", "id", "contaminated"])
    s = d[d.system == "sft"].drop_duplicates(["bench", "iso", "id"])
    r = d[d.system == "rl"].drop_duplicates(["bench", "iso", "id"])
    keys = set(zip(s.bench, s.iso, s.id)) & set(zip(r.bench, r.iso, r.id))
    cont = dict(zip(zip(s.bench, s.iso, s.id), s.contaminated))
    by = {}
    for k in keys:
        lg = LANGUAGES.get(k[1])
        if lg is None or not lg.qe_covered:
            continue
        by.setdefault((k[0], k[1]), []).append(k)
    sel = []
    for unit, ks in sorted(by.items()):
        ks.sort(key=lambda k: hashlib.sha1(f"1|{k[0]}|{k[1]}|{k[2]}".encode()).hexdigest())
        clean = [k for k in ks if not cont[k]]
        cs = set(clean)
        rest = [k for k in ks if k not in cs]
        sel += sorted(clean + rest[:max(0, 400 - len(clean))])
    json.dump([list(k) for k in sel], open(p, "w"))
    return set(sel)


def part_B(groups):
    sel = e1_sample_keys()
    ind = pd.read_parquet(os.path.join(C.WORK, "indicators.parquet"), columns=["system", "bench", "iso", "id", "pair_seen", "afri_ref", "ssa_ref", "afri_qe", "ssa_qe", "chrf"])
    f2 = ind[(ind.system == "final2") & (~ind.pair_seen)].drop_duplicates(["bench", "iso", "id"])
    f2 = f2[[(b, i, k) in sel for b, i, k in zip(f2.bench, f2.iso, f2.id)]]
    us = pd.read_csv(os.path.join(C.JOBS, "e1", "summary", "unit_stats.csv"))
    us = us[(us.subset == "no_pair") & (us.b != "ALL") & us.metric.isin(["afri_qe", "afri_ref", "ssa_qe", "ssa_ref"])]
    rows = []
    for (b, iso), x in f2.groupby(["bench", "iso"]):
        rec = dict(bench=b, iso=iso)
        for m in ("afri_qe", "afri_ref", "ssa_qe", "ssa_ref"):
            v = x[m].dropna()
            rec[f"final2_{m}"], rec[f"final2_{m}_n"] = (v.mean() if len(v) else np.nan), len(v)
            u = us[(us.b == b) & (us.iso == iso) & (us.metric == m)]
            if len(u):
                rec[f"rl_{m}"], rec[f"sft_{m}"], rec[f"e1_{m}_n"] = float(u.mean_b.iloc[0]), float(u.mean_a.iloc[0]), int(u.n.iloc[0])
        rows.append(rec)
    cell = pd.DataFrame(rows)
    cell["tier"] = cell.iso.map(groups.set_index("iso")["tier"])
    cell.to_csv(os.path.join(C.OUT, "judges_B_cell_level_E1_sample.csv"), index=False, float_format="%.6g")
    # sanity: row counts agree
    for m in ("afri_qe", "afri_ref", "ssa_qe", "ssa_ref"):
        ok = cell[[f"final2_{m}_n", f"e1_{m}_n"]].dropna()
        print("n final2 == n E1", m, int((ok.iloc[:, 0] == ok.iloc[:, 1]).sum()), "of", len(ok), "cells")
    # language-level quantities: mean over benchmarks of the cell value, then derived gaps
    out = []
    for sysn in ("sft", "rl", "final2"):
        for gname, f in (("AfriCOMET: QE - with reference", lambda c: c[f"{sysn}_afri_qe"] - c[f"{sysn}_afri_ref"]), ("SSA-COMET: QE - with reference", lambda c: c[f"{sysn}_ssa_qe"] - c[f"{sysn}_ssa_ref"]),
                         ("QE: AfriCOMET - SSA-COMET", lambda c: c[f"{sysn}_afri_qe"] - c[f"{sysn}_ssa_qe"]), ("with reference: AfriCOMET - SSA-COMET", lambda c: c[f"{sysn}_afri_ref"] - c[f"{sysn}_ssa_ref"])):
            cell[f"{sysn}|{gname}"] = f(cell)
    names = ["AfriCOMET: QE - with reference", "SSA-COMET: QE - with reference", "QE: AfriCOMET - SSA-COMET", "with reference: AfriCOMET - SSA-COMET"]
    rng = np.random.default_rng(11)
    for gname in names:
        sub = cell.dropna(subset=[f"{s}|{gname}" for s in ("rl", "final2")])
        lang = sub.groupby("iso")[[f"{s}|{gname}" for s in ("sft", "rl", "final2")]].mean()
        lang["tier"] = lang.index.map(groups.set_index("iso")["tier"])
        for tier in ["all"] + sorted(lang.tier.dropna().unique()):
            l = lang if tier == "all" else lang[lang.tier == tier]
            if len(l) < 3:
                continue
            d = (l[f"final2|{gname}"] - l[f"rl|{gname}"]).values
            idx = rng.integers(0, len(d), size=(4000, len(d)))
            lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
            out.append(dict(quantity=gname, tier=tier, n_lang=len(l), n_cells=int(sub[sub.iso.isin(l.index)].shape[0]), sft=l[f"sft|{gname}"].mean(), rl=l[f"rl|{gname}"].mean(),
                            final2=l[f"final2|{gname}"].mean(), final2_minus_rl=d.mean(), lo=lo, hi=hi, langs_final2_higher=int((d > 0).sum()), langs_final2_lower=int((d < 0).sum())))
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(C.OUT, "judges_B_qe_gap_final2_vs_rl.csv"), index=False, float_format="%.5g")
    return res


def part_C(groups):
    cal = json.load(open(os.path.join(C.CAL, "shaping_flores_dev_v8.json"), encoding="utf-8"))
    ind = pd.read_parquet(os.path.join(C.WORK, "indicators.parquet"), columns=["system", "bench", "iso", "id", "pair_seen", "afri_qe", "ssa_qe"])
    f2 = ind[(ind.system == "final2") & (~ind.pair_seen)].drop_duplicates(["bench", "iso", "id"]).dropna(subset=["afri_qe", "ssa_qe"])
    a1 = {k: v["anchors"] for k, v in cal.items() if isinstance(v, dict) and "anchors" in v}
    a2 = {k: (v.get("qe2") or {}).get("anchors") for k, v in cal.items() if isinstance(v, dict) and "qe2" in v}
    f2 = f2[f2.iso.map(lambda i: i in a1 and a2.get(i) is not None)].copy()
    f2["r1"] = [shape(c, a1[i]) for c, i in zip(f2.afri_qe, f2.iso)]
    f2["r2"] = [shape(c, a2[i]) for c, i in zip(f2.ssa_qe, f2.iso)]
    f2["d"] = f2.r1 - f2.r2
    f2["min_branch"] = (f2.d.abs() > 1.0).astype(float)
    f2["afri_high"] = (f2.d > 1.0).astype(float)
    f2["ssa_high"] = (f2.d < -1.0).astype(float)
    f2["abs_diff"] = f2.d.abs()
    rows = []
    for b, x in list(f2.groupby("bench")) + [("ALL", f2)]:
        per = x.groupby("iso")[["r1", "r2", "d", "min_branch", "afri_high", "ssa_high", "abs_diff"]].mean()
        rows.append(dict(bench=b, n_rows=len(x), n_lang=x.iso.nunique(), **{c: per[c].mean() for c in per.columns}, **{c + "_pooled": x[c].mean() for c in per.columns}))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.OUT, "judges_C_final2_shaped_disagreement.csv"), index=False, float_format="%.5g")
    return out


def part_D(groups):
    """final2 outputs only: what does the ensemble rule do to the outputs the anchors flag? R_ens (mean-or-min of the shaped judges) against r1 alone (AfriCOMET-QE only: the content
    signal of the RL-final reward) on the rows where each indicator fires."""
    cal = json.load(open(os.path.join(C.CAL, "shaping_flores_dev_v8.json"), encoding="utf-8"))
    ind = pd.read_parquet(os.path.join(C.WORK, "indicators.parquet"))
    f2 = ind[(ind.system == "final2") & (~ind.pair_seen)].drop_duplicates(["bench", "iso", "id"]).dropna(subset=["afri_qe", "ssa_qe"])
    a1 = {k: v["anchors"] for k, v in cal.items() if isinstance(v, dict) and "anchors" in v}
    a2 = {k: (v.get("qe2") or {}).get("anchors") for k, v in cal.items() if isinstance(v, dict) and "qe2" in v}
    f2 = f2[f2.iso.map(lambda i: i in a1 and a2.get(i) is not None)].copy()
    f2["r1"] = [shape(c, a1[i]) for c, i in zip(f2.afri_qe, f2.iso)]
    f2["r2"] = [shape(c, a2[i]) for c, i in zip(f2.ssa_qe, f2.iso)]
    d = f2.r1 - f2.r2
    f2["R"] = np.where(d.abs() <= 1.0, 0.5 * (f2.r1 + f2.r2), np.minimum(f2.r1, f2.r2))
    f2["min_branch"] = (d.abs() > 1.0).astype(float)
    f2["ens_minus_r1"] = f2.R - f2.r1
    f2["ens_lower_by_half"] = (f2.ens_minus_r1 < -0.5).astype(float)
    f2["ssa_lower_by_1"] = (d > 1.0).astype(float)
    f2["ssa_higher_by_1"] = (d < -1.0).astype(float)
    subsets = {"all rows": np.ones(len(f2), bool), "copy flag (anchors v3)": f2.copy_flag.fillna(0) > 0, "substring copy >= 25% of source": f2.copy_frac.fillna(0) >= 0.25,
               "Jaccard copy >= 0.9": f2.jacc_copy > 0, "padding flag": f2.pad_flag.fillna(0) > 0, "padding saturated": f2.pad_sat.fillna(0) > 0, "repetition flag": f2.rep_flag.fillna(0) > 0,
               "repeated 4-grams > 10%": f2.rep4 > 0.10, "stop-word flag": f2.func_flag.fillna(0) > 0, "entity penalty > 0.3": f2.ent2_pen.fillna(0) > 0.3,
               "wrong language (GlotLID)": f2.lid_wrong.fillna(0) > 0, "unfinished": f2.unfinished > 0, "no indicator fires": (f2.copy_flag.fillna(0) == 0) & (f2.pad_flag.fillna(0) == 0)
               & (f2.rep_flag.fillna(0) == 0) & (f2.func_flag.fillna(0) == 0) & (f2.lid_wrong.fillna(0) == 0) & (f2.jacc_copy == 0)}
    rows = []
    for name, m in subsets.items():
        x = f2[np.asarray(m)]
        if len(x) < 5:
            rows.append(dict(subset=name, n=len(x)))
            continue
        rows.append(dict(subset=name, n=len(x), n_lang=x.iso.nunique(), r1_afri=x.r1.mean(), r2_ssa=x.r2.mean(), R_ensemble=x.R.mean(), ens_minus_r1=x.ens_minus_r1.mean(),
                         min_branch_share=x.min_branch.mean(), ssa_lower_by_1=x.ssa_lower_by_1.mean(), ssa_higher_by_1=x.ssa_higher_by_1.mean(), ens_lower_than_r1_by_half=x.ens_lower_by_half.mean()))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(C.OUT, "judges_D_final2_ensemble_on_flagged_outputs.csv"), index=False, float_format="%.4g")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABCD")
    a = ap.parse_args()
    groups = pd.read_csv(os.path.join(C.OUT, "language_groups.csv"))
    if "A" in a.parts:
        print(part_A(groups).query("kind=='level' and tier=='all'")[["scope", "system", "n_lang", "z_afri", "z_ssa", "gap", "abs_gap_gt1", "afri_higher", "ssa_higher"]].round(4).to_string())
    if "B" in a.parts:
        print(part_B(groups).round(4).to_string())
    if "C" in a.parts:
        print(part_C(groups).round(4).to_string())
    if "D" in a.parts:
        print(part_D(groups).round(4).to_string())


if __name__ == "__main__":
    main()
