#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""H1 step 4b: tables for the long-document indicators of step 4. Writes in $HE_OUT:
  longdoc_lc1_by_system.csv         macro (over languages) mean of every indicator per system, mode and length group
  longdoc_lc1_paired.csv            paired differences (B minus A; documents resampled within language, macro over languages, 1,000 resamples) for the pairs RL-final -> final2,
                                    step 250 -> final2, A2W -> step 250, A2W -> final2, SFT-final -> RL-final, per mode and length group
  longdoc_afs_papers_by_system.csv / longdoc_afs_papers_paired.csv   the same for AfriScience-MT whole papers
  longdoc_hub77_by_language.csv, longdoc_hub77_by_group.csv          absolute indicators of final2 on the 77-language set (sentence-by-sentence wrapper)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardening_evidence_common as C  # noqa: E402

warnings.filterwarnings("ignore")
IND = [("unfinished", 0.005), ("empty", 0.001), ("rep_flag", 0.01), ("rep_severe", 0.005), ("rep_degenerate", 0.005), ("rep_sev", 0.005), ("rep_rep4", 0.005), ("cover_failed", 0.005),
       ("pad_flag", 0.01), ("pad_def", 0.005), ("func_def", 0.005), ("copy_flag", 0.01), ("copy_frac", 0.002), ("num_loss", 0.005), ("lid_doc_wrong", 0.01), ("lid_sent_wrong_share", 0.005)]
COLS = [c for c, _ in IND]
MARG = dict(IND)


def fill(d):
    d = d.copy()
    for c in COLS:
        if c not in ("num_loss",):
            d[c] = d[c].fillna(0.0)
    return d


def by_system(d, keys):
    rows = []
    for k, x in d.groupby(keys):
        per = x.groupby("iso")[COLS].mean()
        rec = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        rec.update(n_docs=len(x), n_lang=x.iso.nunique())
        for c in COLS:
            rec[c] = per[c].mean()
            rec[c + "__pooled"] = x[c].mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def paired(d, a, b, scope_cols, key, B=1000, seed=5):
    rows = []
    for scope, x in d.groupby(scope_cols):
        xa = x[x.system == a].set_index(key)
        xb = x[x.system == b].set_index(key)
        idx = xa.index.intersection(xb.index)
        if len(idx) < 10:
            continue
        xa, xb = xa.loc[idx], xb.loc[idx]
        isos = xa.iso.values
        diffs = {c: {} for c in COLS}
        for iso in np.unique(isos):
            m = isos == iso
            for c in COLS:
                v = xb[c].values[m] - xa[c].values[m]
                diffs[c][iso] = v[~np.isnan(v)]
        for c in COLS:
            dd = {l: v for l, v in diffs[c].items() if len(v)}
            if not dd:
                continue
            pt, lo, hi = C.macro_paired_bootstrap(dd, B=B, seed=seed)
            lp = np.array([v.mean() for v in dd.values()])
            _, llo, lhi = C.language_bootstrap(lp)
            rows.append(dict(a=a, b=b, scope=scope if isinstance(scope, str) else "|".join(map(str, scope)), indicator=c, n_docs=len(idx), n_lang=len(dd),
                             mean_a=float(np.mean([xa[c].values[isos == l][~np.isnan(xa[c].values[isos == l])].mean() for l in dd])), mean_b=float(np.mean([xb[c].values[isos == l][~np.isnan(xb[c].values[isos == l])].mean() for l in dd])),
                             diff_b_minus_a=pt, ci_lo=lo, ci_hi=hi, lang_ci_lo=llo, lang_ci_hi=lhi, margin=MARG[c], verdict=C.label_effect(lo, hi, MARG[c]), negligible=bool(abs(pt) < MARG[c])))
    return pd.DataFrame(rows)


def main():
    # ---------------- lc1
    d = fill(pd.read_parquet(os.path.join(C.WORK, "longdoc_lc1.parquet")))
    d["all"] = "all documents"
    lens = d.copy()
    lens["length_group"] = "all documents"
    d2 = pd.concat([d, lens], ignore_index=True)
    bs = by_system(d2, ["system", "mode", "length_group"])
    bs.to_csv(os.path.join(C.OUT, "longdoc_lc1_by_system.csv"), index=False, float_format="%.5g")
    out = []
    for a, b in (("rl", "final2"), ("live", "final2"), ("a2w", "live"), ("a2w", "final2"), ("sft", "rl"), ("rl", "a2w")):
        for mode in ("whole", "sentfix"):
            x = d2[d2["mode"] == mode]
            p = paired(x, a, b, ["length_group"], ["iid"])
            p["mode"] = mode
            out.append(p)
        print("lc1 pair", a, b, flush=True)
    pd.concat(out, ignore_index=True).to_csv(os.path.join(C.OUT, "longdoc_lc1_paired.csv"), index=False, float_format="%.5g")
    # ---------------- AfriScience whole papers
    p = os.path.join(C.WORK, "longdoc_afs_papers.parquet")
    if os.path.exists(p):
        a_ = fill(pd.read_parquet(p))
        a_["key"] = a_["split"] + "|" + a_["paper"].astype(str) + "|" + a_["iso"]
        a_["scope"] = a_["split"]
        both = a_.copy()
        both["scope"] = "dev+test"
        a2 = pd.concat([a_, both], ignore_index=True)
        by_system(a2, ["system", "scope"]).to_csv(os.path.join(C.OUT, "longdoc_afs_papers_by_system.csv"), index=False, float_format="%.5g")
        out = []
        for x, y in (("rl", "final2"), ("a2w", "final2"), ("rl", "a2w")):
            out.append(paired(a2.rename(columns={"scope": "scope_"}), x, y, ["scope_"], ["key"]))
        pd.concat(out, ignore_index=True).to_csv(os.path.join(C.OUT, "longdoc_afs_papers_paired.csv"), index=False, float_format="%.5g")
    # ---------------- hub 77
    p = os.path.join(C.WORK, "longdoc_hub77.parquet")
    if os.path.exists(p):
        h = pd.read_parquet(p)
        groups = pd.read_csv(os.path.join(C.OUT, "language_groups.csv")).set_index("iso")
        modes = C.content_modes()
        h["content_mode"] = h.iso.map(modes)
        h["tier"] = h.iso.map(groups["tier"]).fillna(h["content_mode"].map({"consensus": "consensus-only"})).fillna("unassigned")
        h["hub_loop_share"] = h["hub_n_looping_sentences"] / h["hub_n_sentences"].clip(lower=1)
        h["hub_copied_share"] = h["hub_n_copied_sentences"] / h["hub_n_sentences"].clip(lower=1)
        h["hub_unfinished"] = (h["hub_n_unfinished_units"] > 0).astype(float)
        h["hub_any_loop"] = (h["hub_n_looping_sentences"] > 0).astype(float)
        h["unfinished"] = h["hub_unfinished"]
        cols = COLS + ["hub_loop_share", "hub_copied_share", "hub_unfinished", "hub_any_loop"]
        h[cols] = h[cols].astype(float)
        per = h.groupby("iso")[cols].mean()
        per["content_mode"] = per.index.map(modes)
        per["tier"] = per.index.map(lambda i: h.loc[h.iso == i, "tier"].iloc[0])
        per["n_docs"] = h.groupby("iso").size()
        per.to_csv(os.path.join(C.OUT, "longdoc_hub77_by_language.csv"), float_format="%.5g")
        g = per.groupby("content_mode")[cols].mean()
        g["n_lang"] = per.groupby("content_mode").size()
        g.to_csv(os.path.join(C.OUT, "longdoc_hub77_by_group.csv"), float_format="%.5g")
        print(g.round(4).T.to_string())


if __name__ == "__main__":
    main()
