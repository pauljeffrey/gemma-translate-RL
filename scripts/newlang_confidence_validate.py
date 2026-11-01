#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validation of the reference-free confidence features against the human references (D-47).

Input : the metrics of newlang_probe_metrics.py (samples.csv, groups.csv, summary.json) of the RL checkpoint, optionally qe.csv of newlang_qe.py.
Output: language_features.csv (one row per language x condition), validation.json (every number quoted in the report), tables for the report.

What is measured (all against the HUMAN reference; the features themselves never see it):
  language level   Spearman across languages between each reference-free feature and the language's quality (mean chrF++ of ALL its completions; excess over the unrelated-text floor)
  within prompt    Spearman between a completion's reference-free score (consensus, mean log-prob, GlotLID probability) and its chrF++, inside its group of 5 (greedy + 4 samples), averaged over
                   the groups with >= 3 valid members (the unit GRPO compares: D-43 reported 0.27-0.38 for efi / ach / nyn / ibb at G = 8)
  group gate       reliability of a group-level gate: within-prompt rho and mean true chrF++ of the valid members, for groups that pass / fail the gate
  weight           leave-one-language-out linear model chrF++_lang ~ reference-free features (the capability estimate behind the advantage weight)

    python scripts/newlang_confidence_validate.py --metrics DIR --out DIR [--qe qe.csv] [--label rl2c]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import newlang_registry as R  # noqa: E402


def sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 4 or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
        return None, None, int(m.sum())
    r, p = stats.spearmanr(x[m], y[m])
    return float(r), float(p), int(m.sum())


def within(df, a, b, min_valid=3):
    """Mean / median / se of the per-group Spearman between columns a and b over the VALID members of each group (groups with >= min_valid valid members and varying values)."""
    rhos = []
    for _, g in df.groupby(["key", "set", "cond", "label", "pi"]):
        g = g[(g["valid"] == 1)]
        if len(g) < min_valid or g[a].isna().any() or g[a].nunique() < 2 or g[b].nunique() < 2:
            continue
        r, _ = stats.spearmanr(g[a], g[b])
        if not np.isnan(r):
            rhos.append(r)
    if not rhos:
        return dict(n=0, mean=None, median=None, se=None, pos=None)
    return dict(n=len(rhos), mean=float(np.mean(rhos)), median=float(np.median(rhos)), se=float(np.std(rhos) / np.sqrt(len(rhos))), pos=float(np.mean(np.asarray(rhos) > 0)))


def within_nolid(df, a, b, min_valid=3):
    d2 = df.copy()
    d2["valid"] = d2["valid_nolid"]
    return within(d2, a, b, min_valid)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))


def extra_analyses(S, G, L, a):
    """margins, label dispersion, capability model (fit on trained, applied to new), content-only within-prompt validity per language, QE."""
    out = {}
    # ---- GlotLID label dispersion per group: share of the modal top-1 label among the checked members (label-free: needs no accepted set)
    disp = {}
    for (key, pi), g in S[S["lid_checked"] == 1].groupby(["key", "pi"]):
        disp[(key, pi)] = g["lid1"].value_counts(normalize=True).iloc[0]
    G = G.copy()
    G["lid_modal"] = [disp.get((k, p), np.nan) for k, p in zip(G["key"], G["pi"])]
    L = L.copy()
    L["lid_modal"] = [G.loc[G["key"] == k, "lid_modal"].mean() for k in L["key"]]
    # ---- margins between the trained yardsticks and the new languages (nde is reported separately: it writes a Nguni language that GlotLID calls nbl / zul)
    tr, nw = L[L["trained"] == 1], L[(L["trained"] == 0) & (L["key"] != "nde")]
    feats = ["lid_right_T1", "p_accept", "valid_rate", "groups_ge3", "n_valid_mean", "lid_modal", "logp_sample", "logp_greedy", "cons_nolid", "finished", "distinct"]
    marg = {}
    for f in feats:
        t, n = tr[f].dropna(), nw[f].dropna()
        if len(t) and len(n):
            marg[f] = dict(trained_min=float(t.min()), trained_max=float(t.max()), new_min=float(n.min()), new_max=float(n.max()), auc_trained_above_new=auc(t, n),
                           gap_trained_min_minus_new_max=float(t.min() - n.max()))
    out["margins"] = marg
    # ---- capability model: excess chrF ~ a + b1 cons + b2 logp, fit on the trained languages, leave-one-out inside them, applied to the new languages (out of distribution)
    cols = ["cons_nolid", "logp_sample"]
    X, y = tr[cols].values, tr["excess_all"].values

    def fit(X, y):
        return np.linalg.lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)[0]

    loo = np.array([np.r_[1, X[i]] @ fit(np.delete(X, i, 0), np.delete(y, i)) for i in range(len(y))])
    w = fit(X, y)
    Xn = nw[cols].values
    pred_new = np.c_[np.ones(len(Xn)), Xn] @ w
    out["capability_model"] = dict(features=cols, coef=[float(c) for c in w], loo_spearman=float(stats.spearmanr(loo, y)[0]), loo_mae=float(np.abs(loo - y).mean()), sd_excess=float(y.std()),
                                   new_pred=dict(zip(nw["key"], [float(x) for x in pred_new])), new_actual=dict(zip(nw["key"], [float(x) for x in nw["excess_all"]])),
                                   new_mae=float(np.abs(pred_new - nw["excess_all"].values).mean()))
    L["q_hat"] = np.c_[np.ones(len(L)), L[cols].values] @ w
    # ---- content-only within-prompt validity of the consensus (consensus vs chrF++ minus the placebo chrF++), per language, without the LID guard
    S2 = S.copy()
    S2["chrf_content"] = S2["chrf"] - S2["chrf_placebo"]
    rows = []
    for key in L["key"]:
        s = S2[S2["key"] == key]
        wc = within_nolid(s, "consensus_nolid", "chrf_content")
        wt = within_nolid(s, "consensus_nolid", "chrf")
        wpl = within_nolid(s, "consensus_nolid", "chrf_placebo")
        rows.append(dict(key=key, n_groups=wc["n"], rho_true=wt["mean"], rho_placebo=wpl["mean"], rho_content=wc["mean"], rho_content_se=wc["se"]))
    out["content_validity"] = rows
    # ---- the "top-consensus sample" in junk languages: is it the sample of the dominant (wrong) language?
    att = []
    for key in nw["key"]:
        s = S[(S["key"] == key) & (S["valid_nolid"] == 1) & S["consensus_nolid"].notna()]
        if s.empty:
            continue
        dom = s["lid1"].value_counts().index[0]
        top = s.loc[s.groupby("pi")["consensus_nolid"].idxmax()]
        att.append(dict(key=key, dominant_label=dom, dominant_share_all=float((s["lid1"] == dom).mean()), dominant_share_in_top_consensus=float((top["lid1"] == dom).mean()),
                        top_vs_mean_len_ratio=float(top["len_ratio"].mean() / s["len_ratio"].mean())))
    out["attractor"] = att
    # ---- across prompts inside one language: does a group's reference-free summary rank the group's true quality? (Spearman per language, then the mean over the trained languages)
    gl = []
    for key, g in G.groupby("key"):
        g = g.copy()
        g["cm"] = pd.to_numeric(g["cons_mean"], errors="coerce")
        r = {"key": key}
        for f in ("cm", "logp_mean", "len_cv", "lid_modal"):
            rr, pp, nn = sp(g[f], g["chrf_all_mean"])
            r[f"rho_{f}"], r[f"n_{f}"] = rr, nn
        gl.append(r)
    out["across_prompts"] = gl
    # ---- group gate with the default parameters: which share of the groups passes
    from gemma_mt.rl import confidence as C

    P = C.ConfidenceParams()
    gate = {}
    for key, g in S.groupby("key"):
        res = {"ok": 0, "few_valid": 0, "wrong_language": 0, "collapsed": 0}
        for pi, gg in g.groupby("pi"):
            f = C.group_features([bool(v) for v in gg["valid"]], [None if c == 0 else bool(i) for c, i in zip(gg["lid_checked"], gg["lid_in"])], list(gg["text"].fillna("")), list(gg["mean_logp_body"]))
            res[C.group_gate(f, P)[1]] += 1
        tot = sum(res.values())
        gate[key] = {k: v / tot for k, v in res.items()}
    out["gate_shares"] = gate
    # ---- QE
    if a.qe_refs and os.path.exists(a.qe_refs):
        Q = pd.read_csv(a.qe_refs)
        qr = []
        for (key, sname), g in Q.groupby(["key", "set"]):
            h, m = g[g.kind == "human"].qe.values, g[g.kind == "mismatch"].qe.values
            qr.append(dict(key=key, set=sname, n=len(h), qe_human=float(h.mean()), qe_mismatch=float(m.mean()), auc_human_vs_mismatch=auc(h, m)))
        out["qe_validity"] = qr
    if a.qe_outputs and os.path.exists(a.qe_outputs):
        Q = pd.read_csv(a.qe_outputs)
        S3 = S.merge(Q[Q.kind.isin(["greedy", "sample"])][["key", "set", "cond", "kind", "pi", "k", "qe"]], on=["key", "set", "cond", "kind", "pi", "k"], how="inner")
        S3["chrf_content"] = S3["chrf"] - S3["chrf_placebo"]
        qo = []
        for key, g in S3.groupby("key"):
            wq = within_nolid(g, "qe", "chrf_content") if "qe" in g else None
            qo.append(dict(key=key, n=len(g), qe_output_mean=float(g["qe"].mean()), qe_output_greedy=float(g.loc[g.kind == "greedy", "qe"].mean()), rho_qe_content=wq["mean"] if wq else None,
                           rho_qe_content_n=wq["n"] if wq else 0))
        out["qe_outputs"] = qo
    L.to_csv(os.path.join(a.out, "language_features.csv"), index=False)
    G.to_csv(os.path.join(a.out, "groups_with_dispersion.csv"), index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qe-refs", default=None, help="qe_refs.csv (human / mismatch QE of the 150-sentence sets)")
    ap.add_argument("--qe-outputs", default=None, help="qe_outputs.csv (QE of the completions)")
    ap.add_argument("--label", default="rl2c")
    ap.add_argument("--exclude", nargs="*", default=["shi"], help="languages whose reference does not match the trained language (shi: NTREX is in Arabic script)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    S = pd.read_csv(os.path.join(a.metrics, "samples.csv"), keep_default_na=False, na_values=[""])
    G = pd.read_csv(os.path.join(a.metrics, "groups.csv"), keep_default_na=False, na_values=[""])
    S = S[S["label"] == a.label]
    G = G[G["label"] == a.label]
    prim = {k: (v["sets"][0].replace(":", "_").replace("/", "_")) for k, v in {**R.NEW, **R.YARDSTICK}.items()}
    main_cond = lambda k: "trained" if k in R.YARDSTICK else "prefill"              # noqa: E731
    S = S[[(r.cond == main_cond(r.key)) and (r.set == prim[r.key]) for r in S.itertuples()]]
    G = G[[(r.cond == main_cond(r.key)) and (r.set == prim[r.key]) for r in G.itertuples()]]
    S = S[~S["key"].isin(a.exclude)]
    G = G[~G["key"].isin(a.exclude)]
    S["consensus"] = pd.to_numeric(S["consensus"], errors="coerce")
    S["consensus_nolid"] = pd.to_numeric(S["consensus_nolid"], errors="coerce")
    out = {"label": a.label, "excluded": a.exclude}

    # ---------------------------------------------------------------- language level
    rows = []
    for (key,), s in S.groupby(["key"]):
        g = G[G["key"] == key]
        smp = s[s["kind"] == "sample"]
        gre = s[s["kind"] == "greedy"]
        floor = gre["chrf_mismatch"].mean()
        chk = smp[smp["lid_checked"] == 1]
        rows.append(dict(
            key=key, trained=int(key in R.YARDSTICK), group=("yard" if key in R.YARDSTICK else R.NEW[key]["group"]), n=int(gre.shape[0]),
            greedy_chrf=gre["chrf"].mean(), sample_chrf=smp["chrf"].mean(), all_chrf=s["chrf"].mean(), floor=floor, excess_all=s["chrf"].mean() - floor,
            sample_valid_chrf=smp.loc[smp["valid"] == 1, "chrf"].mean() if (smp["valid"] == 1).any() else np.nan,
            lid_right_T1=chk["lid_in"].mean() if len(chk) else np.nan, lid_right_greedy=gre.loc[gre["lid_checked"] == 1, "lid_in"].mean() if (gre["lid_checked"] == 1).any() else np.nan,
            p_accept=s["p_accept"].mean(), valid_rate=smp["valid"].mean(), valid_nolid_rate=smp["valid_nolid"].mean(), finished=smp["finished"].mean(), copy=smp["copy"].mean(),
            degenerate=np.maximum(smp["degenerate"], smp["selfrep"]).mean(), groups_ge3=float((g["n_valid"] >= 3).mean()), n_valid_mean=g["n_valid"].mean(),
            cons_valid=smp["consensus"].mean(), cons_nolid=smp["consensus_nolid"].mean(), distinct=g.loc[g["n_valid"] >= 2, "distinct"].mean() if (g["n_valid"] >= 2).any() else np.nan,
            len_cv=g.loc[g["n_valid"] >= 2, "len_cv"].mean() if (g["n_valid"] >= 2).any() else np.nan, logp_sample=smp["mean_logp_body"].mean(), logp_greedy=gre["mean_logp_body"].mean(),
            len_ratio=smp["len_ratio"].mean(), n_tok=smp["n_tok"].mean()))
    L = pd.DataFrame(rows).sort_values(["trained", "excess_all"], ascending=[False, False])
    # within-prompt validity per language
    for key in L["key"]:
        s = S[S["key"] == key]
        for nm, col in (("rho_cons", "consensus"), ("rho_logp", "mean_logp_body"), ("rho_pacc", "p_accept")):
            w = within(s, col, "chrf")
            L.loc[L["key"] == key, nm] = w["mean"]
            L.loc[L["key"] == key, nm + "_n"] = w["n"]
        w2 = within_nolid(s, "consensus_nolid", "chrf")
        wp = within_nolid(s, "consensus_nolid", "chrf_placebo")
        L.loc[L["key"] == key, "rho_cons_nolid_placebo"] = wp["mean"]
        L.loc[L["key"] == key, "rho_cons_nolid"] = w2["mean"]
        L.loc[L["key"] == key, "rho_cons_nolid_n"] = w2["n"]
    L.to_csv(os.path.join(a.out, "language_features.csv"), index=False)
    out["n_languages"] = int(len(L))

    # ---------------------------------------------------------------- cross-language correlations
    feats = ["lid_right_T1", "lid_right_greedy", "p_accept", "valid_rate", "valid_nolid_rate", "finished", "groups_ge3", "n_valid_mean", "cons_valid", "cons_nolid", "distinct", "len_cv",
             "logp_sample", "logp_greedy", "copy", "degenerate", "len_ratio"]
    cross = {}
    for subset, d in (("all", L), ("trained", L[L["trained"] == 1]), ("new", L[L["trained"] == 0])):
        cross[subset] = {}
        for f in feats:
            for tgt in ("sample_chrf", "excess_all"):
                r, p, n = sp(d[f], d[tgt])
                cross[subset][f"{f}~{tgt}"] = dict(rho=r, p=p, n=n)
    out["cross_language"] = cross

    # ---------------------------------------------------------------- within prompt, pooled
    pooled = {}
    for subset, keys in (("trained", [k for k in L["key"] if k in R.YARDSTICK]), ("new", [k for k in L["key"] if k in R.NEW])):
        s = S[S["key"].isin(keys)]
        pooled[subset] = {nm: within(s, col, "chrf") for nm, col in (("consensus", "consensus"), ("logp", "mean_logp_body"), ("p_accept", "p_accept"))}
        pooled[subset]["consensus_without_lid_guard"] = within_nolid(s, "consensus_nolid", "chrf")
        # placebo: the same consensus against the chrF++ of the SAME text with the reference of an UNRELATED sentence (what surface statistics alone buy)
        pooled[subset]["placebo_consensus_without_lid_guard"] = within_nolid(s, "consensus_nolid", "chrf_placebo")
        s2 = s.copy()
        s2["chrf_content"] = s2["chrf"] - s2["chrf_placebo"]
        pooled[subset]["content_chrf_consensus_without_lid_guard"] = within_nolid(s2, "consensus_nolid", "chrf_content")
    out["within_prompt_pooled"] = pooled

    # ---------------------------------------------------------------- group level: does a group's reference-free summary predict the group's true quality
    G = G.copy()
    for c in ("cons_mean", "chrf_valid_mean", "cons_mean_nolid", "logp_valid"):
        G[c] = pd.to_numeric(G[c], errors="coerce")
    grp = {}
    for subset, d in (("trained", G[G["key"].isin(R.YARDSTICK)]), ("new", G[G["key"].isin(R.NEW)]), ("all", G)):
        grp[subset] = {}
        for f in ("cons_mean", "cons_mean_nolid", "lid_pass", "lid_p_accept", "n_valid", "logp_mean", "distinct", "len_cv"):
            r, p, n = sp(d[f], d["chrf_all_mean"])
            grp[subset][f"{f}~chrf_all_mean"] = dict(rho=r, p=p, n=n)
    out["group_level"] = grp
    out.update(extra_analyses(S, G, L, a))
    json.dump(out, open(os.path.join(a.out, "validation.json"), "w"), indent=1, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else o)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print(L[["key", "trained", "greedy_chrf", "sample_chrf", "floor", "lid_right_T1", "valid_rate", "cons_valid", "cons_nolid", "logp_sample", "rho_cons", "rho_logp"]].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
