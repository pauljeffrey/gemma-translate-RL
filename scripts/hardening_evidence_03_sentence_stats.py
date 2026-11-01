#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""H1 step 3: sentence-level failure / hack indicators per system, benchmark and language group, and paired comparisons (paired bootstrap over sentences, macro over languages).

Input $HE_WORK/indicators.parquet, ref_indicators.parquet (step 2). Output (in $HE_OUT):
  sentence_indicators_by_system.csv   macro mean (over languages) and pooled mean of every indicator, per system and benchmark, plus the human references
  paired_<A>_vs_<B>.csv               B - A per indicator, benchmark and language group: macro difference, 95% interval (sentences resampled within language, 1,000 resamples),
                                      the interval that resamples LANGUAGES as well, and the verdict
  language_groups.csv                 the group of each language (reward content mode, tertile of SFT-final chrF++)
Everything is a 'no_pair' analysis (rows whose (English, reference) pair was not in the SFT text) as in the paper; AfriScience rows are all used.
Direction: every indicator is 'higher = worse'. Not judged (detector off for the language, text too short) counts as 0 (the reward charges nothing then), except the
entity-loss rates, which are computed over the rows whose source has the entity (same rows for both systems).
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hardening_evidence_common as C  # noqa: E402

warnings.filterwarnings("ignore")

# (column, label, kind, margin) ; kind 'rate' (0/1 or share, in percentage points), 'unit' (deficit 0..1 or share)
INDICATORS = [
    ("unfinished", "unfinished (no end token in budget)", "rate", 0.005),
    ("empty", "empty output", "rate", 0.001),
    ("lid_wrong", "wrong language (GlotLID, where the guard is on)", "rate", 0.005),
    ("jacc_copy", "copy of source, Jaccard >= 0.9 (hard class of both rewards)", "rate", 0.001),
    ("degenerate", "degenerate (v1 repetition rule, hard class of both rewards)", "rate", 0.005),
    ("len_bad", "length ratio outside [0.2, 5] (hard class of both rewards)", "rate", 0.005),
    ("copy_flag", "substring copy of the source above the human threshold (anchors v3)", "rate", 0.005),
    ("copy_frac", "copied fraction of the source (verbatim runs, mean)", "unit", 0.002),
    ("num_loss", "numbers of the source missing in the output (share)", "unit", 0.005),
    ("name_loss", "names of the source missing in the output (share)", "unit", 0.005),
    ("ent2_pen", "graded entity loss (anchors v3 entity2 units; rows with entities)", "unit", 0.005),
    ("func_flag", "stop-word profile deviates (deficit > 0)", "rate", 0.005),
    ("func_def", "stop-word profile deficit (0..1)", "unit", 0.002),
    ("func_sat", "stop-word profile saturated (deficit >= 0.9)", "rate", 0.001),
    ("pad_flag", "padding / low type-token ratio (deficit > 0)", "rate", 0.005),
    ("pad_def", "padding deficit (0..1)", "unit", 0.002),
    ("pad_sat", "padding saturated (deficit >= 0.9)", "rate", 0.001),
    ("rep_flag", "repetition above the human limit (deficit > 0)", "rate", 0.005),
    ("rep_def", "repetition severity (0..1)", "unit", 0.002),
    ("rep4", "share of repeated word 4-grams (mean)", "unit", 0.002),
    ("pen_total", "penalty the final reward would subtract (reward units)", "unit", 0.005),
]
COLS = [c for c, *_ in INDICATORS]
ENTITY_ROWS = {"num_loss": "num_n", "name_loss": "name_n", "ent2_pen": "ent2_judged"}


def prep(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["num_loss"] = np.where(d["num_n"] > 0, 1.0 - d["num_hit"] / d["num_n"].clip(lower=1), np.nan)
    d["name_loss"] = np.where(d["name_n"] > 0, 1.0 - d["name_hit"] / d["name_n"].clip(lower=1), np.nan)
    d["lid_wrong"] = d["lid_wrong"].where(d["lid_judged"] > 0, 0.0)
    for c in COLS:
        if c in ENTITY_ROWS:
            continue
        d[c] = d[c].fillna(0.0)
    d["ent2_pen"] = d["ent2_pen"]                       # NaN where nothing was judged: stays NaN
    return d


def language_groups(ind: pd.DataFrame) -> pd.DataFrame:
    """group_mode = reward content mode of the final run (qe / blend / consensus); tier = consensus-only | strong | mid | weak (tertiles of the SFT-final sentence chrF++
    over the languages that are not consensus-only, `no_pair` rows of FLORES+, NTREX, MAFAND, SmolSent, SALT, GlotStoryBook)."""
    s = ind[(ind.system == "sft") & (~ind.pair_seen) & ind.bench.isin(["flores_plus", "ntrex", "mafand", "smol_sent", "salt", "glot_story"])]
    chrf = s.groupby("iso")["chrf"].mean().rename("sft_chrf")
    g = pd.DataFrame({"iso": sorted(ind.iso.unique())}).set_index("iso")
    g["content_mode"] = [ind.loc[ind.iso == i, "content_mode"].iloc[0] for i in g.index]
    g = g.join(chrf)
    non = g[g.content_mode != "consensus"]["sft_chrf"].dropna()
    q1, q2 = non.quantile([1 / 3, 2 / 3])
    def tier(r):
        if r.content_mode == "consensus":
            return "consensus-only"
        if np.isnan(r.sft_chrf):
            return "unassigned"
        return "weak" if r.sft_chrf <= q1 else ("mid" if r.sft_chrf <= q2 else "strong")
    g["tier"] = [tier(r) for r in g.itertuples()]
    g["tertile_cut_lo"], g["tertile_cut_hi"] = q1, q2
    return g.reset_index()


def boot_rows(X: np.ndarray, B: int, rng: np.random.Generator, chunk: int = 50) -> np.ndarray:
    """X (n, K) -> (B, K) bootstrap means over rows. NaN-aware: rows with NaN in a column are excluded from that column."""
    n, K = X.shape
    out = np.empty((B, K))
    ok = ~np.isnan(X)
    X0 = np.where(ok, X, 0.0)
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        idx = rng.integers(0, n, size=(e - s, n))
        num = X0[idx].sum(1)
        den = ok[idx].sum(1)
        out[s:e] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return out


def paired(ind: pd.DataFrame, groups: pd.DataFrame, a: str, b: str, B: int, benches=None, seed=1) -> pd.DataFrame:
    """b - a. Returns rows for (bench in benches + 'ALL'), (scheme, group), indicator."""
    d = ind[(~ind.pair_seen) & ind.system.isin([a, b])]
    ka = d[d.system == a].set_index(["bench", "iso", "id"])
    kb = d[d.system == b].set_index(["bench", "iso", "id"])
    common = ka.index.intersection(kb.index)
    ka, kb = ka.loc[common], kb.loc[common]
    diffs = (kb[COLS].values - ka[COLS].values)
    meta = common.to_frame(index=False)
    if benches is None:
        benches = sorted(meta.bench.unique())
    gm = groups.set_index("iso")
    rng = np.random.default_rng(seed)
    rows = []
    scopes = [(bn, meta.bench.values == bn) for bn in benches] + [("ALL", np.ones(len(meta), bool))]
    for scope, mask in scopes:
        # per-language point estimates and bootstrap matrices
        langs = sorted(meta.loc[mask, "iso"].unique())
        L = {}
        for iso in langs:
            m = mask & (meta.iso.values == iso)
            X = diffs[m]
            if len(X) < 20:
                continue
            bm = boot_rows(X, B, rng)
            with np.errstate(all="ignore"):
                pt = np.nanmean(X, axis=0)
            L[iso] = (pt, bm, len(X), np.nanmean(ka[COLS].values[m], axis=0), np.nanmean(kb[COLS].values[m], axis=0))
        for scheme in ("all_languages", "tier", "content_mode"):
            if scheme == "all_languages":
                gl = {"all": list(L)}
            else:
                col = "tier" if scheme == "tier" else "content_mode"
                gl = {}
                for iso in L:
                    gl.setdefault(gm.loc[iso, col], []).append(iso)
            for gname, members in gl.items():
                if not members or gname == "unassigned":
                    continue
                pts = np.array([L[i][0] for i in members])
                bms = np.stack([L[i][1] for i in members])                  # (L, B, K)
                ma = np.array([L[i][3] for i in members])
                mb = np.array([L[i][4] for i in members])
                nrows = sum(L[i][2] for i in members)
                with np.errstate(all="ignore"):
                    macro = np.nanmean(pts, 0)
                    mb_boot = np.nanmean(bms, 0)                           # (B, K)
                    lo, hi = np.nanpercentile(mb_boot, [2.5, 97.5], axis=0)
                    mean_a, mean_b = np.nanmean(ma, 0), np.nanmean(mb, 0)
                for k, (col, lab, kind, margin) in enumerate(INDICATORS):
                    # interval including language sampling: resample languages' point estimates
                    x = pts[:, k]
                    x = x[~np.isnan(x)]
                    if len(x) >= 3:
                        idx = rng.integers(0, len(x), size=(2000, len(x)))
                        lm = x[idx].mean(1)
                        llo, lhi = np.percentile(lm, [2.5, 97.5])
                    else:
                        llo = lhi = np.nan
                    verdict = C.label_effect(lo[k], hi[k], margin)
                    rows.append(dict(a=a, b=b, bench=scope, scheme=scheme, group=gname, n_lang=len(x), n_rows=nrows, indicator=col, label=lab, mean_a=mean_a[k], mean_b=mean_b[k],
                                     diff_b_minus_a=macro[k], ci_lo=lo[k], ci_hi=hi[k], lang_ci_lo=llo, lang_ci_hi=lhi, margin=margin, verdict=verdict,
                                     negligible=bool(abs(macro[k]) < margin) if not np.isnan(macro[k]) else False,
                                     langs_b_better=int((x < 0).sum()), langs_b_worse=int((x > 0).sum())))
    return pd.DataFrame(rows)


def by_system(ind: pd.DataFrame, ref: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    d = ind[~ind.pair_seen]
    gm = groups.set_index("iso")
    rows = []
    r = prep(ref.assign(system="human reference", unfinished=0.0))
    r = r[~r.pair_seen]
    for sysn, dd in list(d.groupby("system")) + [("human reference", r)]:
        for bench, db in list(dd.groupby("bench")) + [("ALL", dd)]:
            for scheme in ("all_languages", "tier"):
                if scheme == "all_languages":
                    parts = {"all": db}
                else:
                    t = db["iso"].map(gm["tier"])
                    parts = {g: db[t == g] for g in sorted(t.dropna().unique()) if g != "unassigned"}
                for g, x in parts.items():
                    if not len(x):
                        continue
                    per_lang = x.groupby("iso")[COLS].mean()
                    rec = dict(system=sysn, bench=bench, scheme=scheme, group=g, n_rows=len(x), n_lang=x.iso.nunique())
                    for c in COLS:
                        rec[c + "__macro"] = per_lang[c].mean()
                        rec[c + "__pooled"] = x[c].mean()
                    rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--pairs", default="rl:final2,a2w:final2,sft:rl,live250:final2,a2w:live250,rl:a2w")
    a = ap.parse_args()
    ind = pd.read_parquet(os.path.join(C.WORK, "indicators.parquet"))
    ref = pd.read_parquet(os.path.join(C.WORK, "ref_indicators.parquet"))
    groups = language_groups(ind)
    groups.to_csv(os.path.join(C.OUT, "language_groups.csv"), index=False)
    print(groups.tier.value_counts().to_dict())
    ind = prep(ind)
    bs = by_system(ind, ref, groups)
    bs.to_csv(os.path.join(C.OUT, "sentence_indicators_by_system.csv"), index=False, float_format="%.6g")
    for p in a.pairs.split(","):
        x, y = p.split(":")
        df = paired(ind, groups, x, y, a.B)
        df.to_csv(os.path.join(C.OUT, f"paired_{x}_vs_{y}.csv"), index=False, float_format="%.6g")
        print("paired", x, y, len(df), flush=True)


if __name__ == "__main__":
    main()
