#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Markdown / csv tables of the new-language report (D-47) from the metrics of newlang_probe_metrics.py and the validation of newlang_confidence_validate.py.

    python scripts/newlang_report_tables.py --metrics DIR --validate DIR --qe-refs csv --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import newlang_registry as R  # noqa: E402


def md(df, floatfmt="{:.1f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("-" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--validate", required=True)
    ap.add_argument("--qe-refs", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    summ = pd.DataFrame(json.load(open(os.path.join(a.metrics, "summary.json"))))
    L = pd.read_csv(os.path.join(a.validate, "language_features.csv"))
    val = json.load(open(os.path.join(a.validate, "validation.json")))
    prim = {k: v["sets"][0].replace(":", "_").replace("/", "_") for k, v in {**R.NEW, **R.YARDSTICK}.items()}
    qe_out = {r["key"]: r for r in val.get("qe_outputs", [])}
    qe_ref = {}
    if a.qe_refs and os.path.exists(a.qe_refs):
        Q = pd.read_csv(a.qe_refs)
        for (k, s), g in Q.groupby(["key", "set"]):
            if s == prim.get(k):
                h, m = g[g.kind == "human"].qe.values, g[g.kind == "mismatch"].qe.values
                qe_ref[k] = dict(human=h.mean(), mismatch=m.mean(), auc=float((np.sum(h[:, None] > m[None, :]) + 0.5 * np.sum(h[:, None] == m[None, :])) / (len(h) * len(m))))
    # ---- T1: main probe table
    rows = []
    for _, r in L.iterrows():
        k = r["key"]
        sm = summ[(summ.label == "rl2c") & (summ.key == k) & (summ.set == prim[k]) & (summ.cond == ("trained" if r["trained"] else "prefill"))].iloc[0]
        top = sm["greedy_lid_top"]
        top = ", ".join(f"{a_.split('_')[0]} {b}" for a_, b in list(top.items())[:2])
        rows.append(dict(language=k, kind=("trained" if r["trained"] else "new " + str(r["group"])), set=prim[k].replace("flores_", "FLORES+ ").replace("smol_", "SmolSent ").replace("ntrex_", "NTREX ").replace("local_", ""),
                         greedy_chrF=r["greedy_chrf"], sample_chrF=r["sample_chrf"], floor=r["floor"], excess=r["excess_all"], GlotLID_right_T1=100 * r["lid_right_T1"], valid_pct=100 * r["valid_rate"],
                         finished_pct=100 * r["finished"], copy_pct=100 * r["copy"], mean_logp=r["logp_sample"], consensus_nolid=r["cons_nolid"], greedy_top_LID=top,
                         QE_human=qe_ref.get(k, {}).get("human", np.nan), QE_output=qe_out.get(k, {}).get("qe_output_mean", np.nan), QE_AUC=qe_ref.get(k, {}).get("auc", np.nan)))
    T1 = pd.DataFrame(rows)
    T1 = pd.concat([T1[T1.kind == "trained"].sort_values("excess", ascending=False), T1[T1.kind != "trained"].sort_values("excess", ascending=False)])
    T1.to_csv(os.path.join(a.out, "probe_main.csv"), index=False)
    open(os.path.join(a.out, "T1_probe_main.md"), "w").write(md(T1.drop(columns=["QE_human", "QE_output", "QE_AUC"]), "{:.2f}"))
    # ---- T2: new languages, other conditions
    rows = []
    for k in R.NEW:
        def g(label, cond, col="greedy_chrf"):
            x = summ[(summ.label == label) & (summ.key == k) & (summ.set == prim[k]) & (summ.cond == cond)]
            return None if x.empty else x.iloc[0]
        pre, fr, bs = g("rl2c", "prefill"), g("rl2c", "free"), g("base", "base")
        d = dict(language=k, prefill_chrF=pre["greedy_chrf"] if pre is not None else np.nan, floor=pre["mismatch_floor"] if pre is not None else np.nan)
        if fr is not None:
            d["free_tag_written"] = ", ".join(f"{a_} {b}" for a_, b in list(fr["tags"].items())[:2])
            d["free_chrF"] = fr["greedy_chrf"]
        for c in R.NEW[k]["controls"]:
            x = g("rl2c", f"ctrl:{c}")
            if x is not None:
                d[f"ctrl_{c}_chrF"] = x["greedy_chrf"]
                d[f"ctrl_{c}_floor"] = x["mismatch_floor"]
        if bs is not None:
            d["base_chrF"], d["base_copy_pct"] = bs["greedy_chrf"], 100 * bs["greedy_copy"]
        rows.append(d)
    T2 = pd.DataFrame(rows)
    T2.to_csv(os.path.join(a.out, "probe_conditions_new_languages.csv"), index=False)
    # ---- T3: the validation summary
    pooled = val["within_prompt_pooled"]
    cv = pd.DataFrame(val["content_validity"])
    cv["kind"] = ["trained" if k in R.YARDSTICK else "new" for k in cv["key"]]
    cv.to_csv(os.path.join(a.out, "within_prompt_content_validity.csv"), index=False)
    L.to_csv(os.path.join(a.out, "language_features.csv"), index=False)
    json.dump(val, open(os.path.join(a.out, "validation.json"), "w"), indent=1)
    print(open(os.path.join(a.out, "T1_probe_main.md")).read())
    print(T2.round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
