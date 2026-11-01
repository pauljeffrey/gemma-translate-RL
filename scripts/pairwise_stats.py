# SPDX-License-Identifier: AGPL-3.0-or-later
"""E1 pairwise benchmark comparison, step 3 (CPU, box): sentence-level paired statistics of system B over system A.

    python pairwise_stats.py --dir /workspace/jobs/e1 --a sft --b rl --out summary [--neural neural] [--lang-groups lang_groups.json]

Inputs: <dir>/sys_<name>.jsonl.gz (pairwise_surface.py) and <dir>/<neural>/<metric>.jsonl (pairwise_neural.py; optional).
For every metric the sentence-level delta d = B - A decides a WIN of B (d > thr), a LOSS (d < -thr) or a TIE (|d| <= thr).
  thresholds: chrF++ 1.0 (0..100 scale), COMET-family 0.01 (scale ~0..1); binary surface metrics (finished, loop, lid, rep3): a win is a flip.
Units are (benchmark, language, subset) and (ALL benchmarks pooled, language, subset); subsets: all, no_pair (headline: reference pair unseen in the SFT text),
clean (English sentence unseen; the contamination-robust subset). Paired bootstrap: B=1000 resamples of the sentences of a unit (the same resampled sentences
for A and B). Macro rows (mean over languages, languages weigh equally) resample every language's sentences independently and average; micro rows pool the
sentences (strata = units, n-weighted). The sign test is the exact two-sided binomial test of wins vs losses (ties dropped); p-values assume independent
sentences, which pooled over languages is optimistic (FLORES+/NTREX/SmolSent translate the SAME English sentences into every language).
Output (CSV, in <dir>/<out>): unit_stats, macro, agreement, agreement_by_group, flags, threshold_sensitivity, corpus_chrf_units, meta.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import unicodedata
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from scipy import stats as sps

BOOT = 1000
FULL_METRICS = OrderedDict([  # name: (column prefix, threshold, sign (+1 higher is better), has_wtl)
    ("chrf", ("chrf", 1.0, 1)), ("fin", ("fin", 0.5, 1)), ("loop", ("loop", 0.5, -1)), ("rep3", ("rep3", 0.5, -1)),
    ("lid", ("lid", 0.5, 1)), ("lid_sib", ("lid_sib", 0.5, 1)), ("lratio", ("lratio", None, 1)), ("empty", ("empty", 0.5, -1)),
])
NN_METRICS = OrderedDict([("chrf_sub", ("chrf_sub", 1.0, 1)), ("afri_ref", ("afri_ref", 0.01, 1)), ("afri_qe", ("afri_qe", 0.01, 1)),
                          ("ssa_ref", ("ssa_ref", 0.01, 1)), ("ssa_qe", ("ssa_qe", 0.01, 1))])
REF_BASED = ["chrf_sub", "afri_ref", "ssa_ref"]
QE_BASED = ["afri_qe", "ssa_qe"]
SUBSETS = OrderedDict([("all", lambda df: np.ones(len(df), bool)), ("no_pair", lambda df: (df["pair_seen"] == False).to_numpy()),  # noqa: E712
                       ("clean", lambda df: (df["contaminated"] == False).to_numpy())])  # noqa: E712
BENCH = ["flores_plus", "ntrex", "mafand", "smol_sent", "ibom_mt", "flores_moz", "salt", "glot_story"]


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def load_surface(path, tag):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["hyp_" + tag] = r.pop("hyp")
            rows.append(r)
    return pd.DataFrame(rows)


def build_frame(d, A, B, neural_dir):
    da = load_surface(os.path.join(d, f"sys_{A}.jsonl.gz"), "a")
    db = load_surface(os.path.join(d, f"sys_{B}.jsonl.gz"), "b")
    keep = ["b", "iso", "id", "chrf", "fin", "loop", "rep3", "empty", "lid", "lid_sib", "lratio", "hyp_b"]
    db = db[keep].rename(columns={c: c + "_b" for c in keep if c not in ("b", "iso", "id", "hyp_b")})
    da = da.rename(columns={c: c + "_a" for c in ("chrf", "fin", "loop", "rep3", "empty", "lid", "lid_sib", "lratio")})
    df = da.merge(db, on=["b", "iso", "id"], how="inner")
    df["same"] = (df["hyp_a"] == df["hyp_b"]).astype(float)
    df["same_a"] = 0.0
    df["same_b"] = df["same"]
    df = df.drop(columns=["hyp_a", "hyp_b", "en"])
    for c in ("fin", "loop", "rep3", "empty"):
        for s in ("a", "b"):
            df[f"{c}_{s}"] = df[f"{c}_{s}"].astype(float)
    for c in ("lid", "lid_sib"):
        for s in ("a", "b"):
            df[f"{c}_{s}"] = df[f"{c}_{s}"].map(lambda v: np.nan if v is None or (isinstance(v, float) and v != v) else float(v)).astype(float)
    for s in ("a", "b"):
        df[f"lratio_{s}"] = df[f"lratio_{s}"].clip(0, 5)
    # neural metrics
    for m in ("afri_ref", "afri_qe", "ssa_ref", "ssa_qe"):
        df[f"{m}_a"] = np.nan
        df[f"{m}_b"] = np.nan
        p = os.path.join(d, neural_dir, f"{m}.jsonl")
        if not os.path.isfile(p):
            continue
        look = {}
        for line in open(p, encoding="utf-8"):
            j = json.loads(line)
            b, iso = j["chunk"].split("|")
            for i, sc in j["scores"].items():
                if A in sc and B in sc:
                    look[(b, iso, i)] = (sc[A], sc[B])
        idx = pd.MultiIndex.from_frame(df[["b", "iso", "id"]])
        va = np.full(len(df), np.nan)
        vb = np.full(len(df), np.nan)
        pos = {k: n for n, k in enumerate(zip(df["b"], df["iso"], df["id"]))}
        for k, (x, y) in look.items():
            n = pos.get(k)
            if n is not None:
                va[n], vb[n] = x, y
        df[f"{m}_a"], df[f"{m}_b"] = va, vb
        del idx
    # sentence-level labels of the ensembles (ref-based majority, QE agreement)
    def lab(m):
        xa, xb = df[f"{m}_a"].to_numpy(), df[f"{m}_b"].to_numpy()
        thr = NN_METRICS[m][1]
        dd = xb - xa
        return np.where(np.isnan(dd), np.nan, np.where(dd > thr, 1.0, np.where(dd < -thr, -1.0, 0.0)))

    df["chrf_sub_a"], df["chrf_sub_b"] = df["chrf_a"], df["chrf_b"]
    ok = ~np.isnan(df["afri_ref_a"].to_numpy() + df["afri_ref_b"].to_numpy())
    for m in ("chrf_sub", "afri_ref", "afri_qe", "ssa_ref", "ssa_qe"):
        df.loc[~ok, f"{m}_a"] = np.nan if m != "chrf_sub" else np.nan
        df.loc[~ok, f"{m}_b"] = np.nan
    L = {m: lab(m) for m in NN_METRICS}
    vs = np.vstack([L[m] for m in REF_BASED])
    s = np.nansum(vs, 0)
    nn = np.isnan(vs).any(0)
    ens = np.where(nn, np.nan, np.sign(s))
    unan = np.where(nn, np.nan, np.where((vs == 1).all(0), 1.0, np.where((vs == -1).all(0), -1.0, 0.0)))
    qs = np.vstack([L[m] for m in QE_BASED])
    qagree = np.where(np.isnan(qs).any(0), np.nan, np.where((qs == 1).all(0), 1.0, np.where((qs == -1).all(0), -1.0, 0.0)))
    for name, v in (("ens_ref", ens), ("ens_unan", unan), ("ens_qe", qagree)):
        df[f"{name}_a"] = np.where(np.isnan(v), np.nan, 0.0)
        df[f"{name}_b"] = v
    return df


def pstat(w, l):
    n = int(w + l)
    return float(sps.binomtest(int(w), n, 0.5).pvalue) if n > 0 else 1.0


def unit_boot(xa, xb, thr, sign, idx):
    """point values + bootstrap vectors for one (unit, metric); xa, xb already NaN-free."""
    d = xb - xa
    if thr is None:
        w = np.zeros_like(d)
        l = np.zeros_like(d)
    else:
        w = (sign * d > thr).astype(np.float32)
        l = (sign * d < -thr).astype(np.float32)
    M = np.stack([xa, xb, d, w, l], 1).astype(np.float32)
    n = len(xa)
    pt = M.mean(0)
    bt = M[idx].mean(1) if idx is not None else np.tile(pt, (1, 1))
    return n, pt, bt


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty_like(p)
    m = len(p)
    run = 1.0
    for rank, i in enumerate(o[::-1]):
        k = m - rank
        run = min(run, p[i] * m / k)
        q[i] = run
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="summary")
    ap.add_argument("--neural", default="neural")
    ap.add_argument("--lang-groups", required=True)
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--benchmarks", default="")
    a = ap.parse_args()
    out = os.path.join(a.dir, a.out)
    os.makedirs(out, exist_ok=True)
    groups = json.load(open(a.lang_groups))
    df = build_frame(a.dir, a.a, a.b, a.neural)
    if a.benchmarks:
        df = df[df["b"].isin(a.benchmarks.split(","))].reset_index(drop=True)
    print(len(df), "paired rows", flush=True)
    metrics = OrderedDict(list(FULL_METRICS.items()) + [("same", ("same", 0.5, 1))] + list(NN_METRICS.items()) +
                          [("ens_ref", ("ens_ref", 0.5, 1)), ("ens_unan", ("ens_unan", 0.5, 1)), ("ens_qe", ("ens_qe", 0.5, 1))])
    has_nn = df["afri_ref_a"].notna().any()
    rows, vecs = [], {}
    rng_base = 12345
    cols = {m: (df[f"{pre}_a"].to_numpy(float), df[f"{pre}_b"].to_numpy(float)) for m, (pre, thr, sg) in metrics.items()
            if f"{pre}_a" in df}
    bcol, icol = df["b"].to_numpy(), df["iso"].to_numpy()
    bench_list = sorted(set(bcol), key=lambda x: (BENCH.index(x) if x in BENCH else 99, x))
    for sub, fsub in SUBSETS.items():
        msk0 = fsub(df)
        for bench in ["ALL"] + bench_list:
            mb = msk0 & ((bcol == bench) if bench != "ALL" else np.ones(len(df), bool))
            if not mb.any():
                continue
            for iso in sorted(set(icol[mb])):
                mu = mb & (icol == iso)
                ridx = np.flatnonzero(mu)
                for grp_name, mnames in (("full", [m for m in metrics if m in FULL_METRICS or m == "same"]), ("nn", [m for m in metrics if m not in FULL_METRICS and m != "same"])):
                    # rows of this metric group
                    for m in mnames:
                        if m not in cols:
                            continue
                        xa_all, xb_all = cols[m]
                        ok = ~(np.isnan(xa_all[ridx]) | np.isnan(xb_all[ridx]))
                        if m == "same" or not ok.any():
                            if m != "same":
                                continue
                        sel = ridx[ok]
                        if len(sel) < 5:
                            continue
                        seed = abs(hash((sub, bench, iso, grp_name))) % (2 ** 31) if False else (rng_base + sum(map(ord, f"{sub}|{bench}|{iso}|{grp_name}")) * 7919) % (2 ** 31)
                        rng = np.random.default_rng(seed)
                        idx = rng.integers(0, len(sel), size=(BOOT, len(sel)))
                        pre, thr, sg = metrics[m]
                        n, pt, bt = unit_boot(xa_all[sel], xb_all[sel], thr, sg, idx)
                        w, l = pt[3], pt[4]
                        wn, ln = int(round(w * n)), int(round(l * n))
                        rec = dict(subset=sub, b=bench, iso=iso, metric=m, n=n, mean_a=pt[0], mean_b=pt[1], delta=pt[2],
                                   delta_lo=ci(bt[:, 2])[0], delta_hi=ci(bt[:, 2])[1],
                                   win=w, tie=1 - w - l, loss=l, win_lo=ci(bt[:, 3])[0], win_hi=ci(bt[:, 3])[1], loss_lo=ci(bt[:, 4])[0], loss_hi=ci(bt[:, 4])[1],
                                   p_sign=pstat(wn, ln) if thr is not None else np.nan, wins=wn, losses=ln,
                                   pref=(w / (w + l)) if (w + l) > 0 else np.nan)
                        rows.append(rec)
                        vecs[(sub, bench, iso, m)] = (n, pt, bt)
    us = pd.DataFrame(rows)
    # BH q-values across languages within (subset, bench, metric)
    us["q_sign"] = np.nan
    for _, g in us.groupby(["subset", "b", "metric"]):
        if g["p_sign"].notna().all() and len(g) > 1:
            us.loc[g.index, "q_sign"] = bh(g["p_sign"].to_numpy())
    us.insert(4, "group", us["iso"].map(lambda i: groups[i]["group"]))
    us.insert(5, "family", us["iso"].map(lambda i: groups[i]["family"]))
    us.to_csv(os.path.join(out, "unit_stats.csv"), index=False, float_format="%.5g")
    print("unit_stats", len(us), flush=True)

    # ------------------------------------------------------------------ macro / micro by scope
    scopes = OrderedDict()
    scopes["all_langs"] = lambda i: True
    scopes["rl77"] = lambda i: groups[i]["group"] != "sft_only"
    for g in ("qe_validated", "blend", "consensus_only", "sft_only"):
        scopes["group:" + g] = (lambda g: lambda i: groups[i]["group"] == g)(g)
    for fam in sorted(set(v["family"] for v in groups.values())):
        scopes["family:" + fam] = (lambda fam: lambda i: groups[i]["family"] == fam)(fam)
    scopes["script:Ge'ez (amh, tir)"] = lambda i: i in ("amh", "tir")
    for t in ("strong", "medium", "weak"):
        scopes["tier:" + t] = (lambda t: lambda i: groups[i].get("tier") == t)(t)
    mrows = []
    for (sub, bench, m) in sorted({(k[0], k[1], k[3]) for k in vecs}):
        units = [(k[2], v) for k, v in vecs.items() if k[0] == sub and k[1] == bench and k[3] == m and v[0] >= a.min_n]
        if not units:
            continue
        for sname, f in scopes.items():
            us_ = [(iso, v) for iso, v in units if f(iso)]
            if not us_:
                continue
            P = np.stack([v[1] for _, v in us_])          # (L,5)
            Bv = np.stack([v[2] for _, v in us_])         # (L,B,5)
            nn_ = np.array([v[0] for _, v in us_], float)
            mac_pt, mac_bt = P.mean(0), Bv.mean(0)
            mic_pt = (P * nn_[:, None]).sum(0) / nn_.sum()
            mic_bt = (Bv * nn_[:, None, None]).sum(0) / nn_.sum()
            thr = metrics[m][1]
            rec = dict(subset=sub, b=bench, scope=sname, metric=m, n_langs=len(us_), n_sent=int(nn_.sum()),
                       mean_a=mac_pt[0], mean_b=mac_pt[1], delta=mac_pt[2], delta_lo=ci(mac_bt[:, 2])[0], delta_hi=ci(mac_bt[:, 2])[1],
                       macro_win=mac_pt[3], macro_tie=1 - mac_pt[3] - mac_pt[4], macro_loss=mac_pt[4],
                       macro_win_lo=ci(mac_bt[:, 3])[0], macro_win_hi=ci(mac_bt[:, 3])[1], macro_loss_lo=ci(mac_bt[:, 4])[0], macro_loss_hi=ci(mac_bt[:, 4])[1],
                       micro_delta=mic_pt[2], micro_win=mic_pt[3], micro_tie=1 - mic_pt[3] - mic_pt[4], micro_loss=mic_pt[4],
                       micro_win_lo=ci(mic_bt[:, 3])[0], micro_win_hi=ci(mic_bt[:, 3])[1], micro_loss_lo=ci(mic_bt[:, 4])[0], micro_loss_hi=ci(mic_bt[:, 4])[1])
            if thr is not None:
                wn, ln = int(round((mic_pt[3] * nn_.sum()))), int(round((mic_pt[4] * nn_.sum())))
                rec["micro_p_sign"] = pstat(wn, ln)
                rec["micro_pref"] = mic_pt[3] / (mic_pt[3] + mic_pt[4]) if (mic_pt[3] + mic_pt[4]) > 0 else np.nan
                pr_bt = mic_bt[:, 3] / np.maximum(mic_bt[:, 3] + mic_bt[:, 4], 1e-9)
                rec["micro_pref_lo"], rec["micro_pref_hi"] = ci(pr_bt)
                # languages where B wins significantly / loses significantly (CI of the mean delta excludes 0)
                nsig_w = nsig_l = 0
                for iso, v in us_:
                    lo, hi = ci(v[2][:, 2])
                    nsig_w += lo > 0
                    nsig_l += hi < 0
                rec["langs_b_better_sig"], rec["langs_b_worse_sig"] = int(nsig_w), int(nsig_l)
            mrows.append(rec)
    macro = pd.DataFrame(mrows)
    macro.to_csv(os.path.join(out, "macro.csv"), index=False, float_format="%.5g")
    print("macro", len(macro), flush=True)

    # ------------------------------------------------------------------ flags: QE wins without a reference-based win
    fl = []
    pv = us.pivot_table(index=["subset", "b", "iso"], columns="metric", values=["delta", "delta_lo", "delta_hi", "n"], aggfunc="first")
    for (sub, bench, iso), r in pv.iterrows():
        def sig(m):
            try:
                lo, hi, d = r[("delta_lo", m)], r[("delta_hi", m)], r[("delta", m)]
            except KeyError:
                return None
            if np.isnan(d):
                return None
            return "+" if lo > 0 else ("-" if hi < 0 else "0")
        s = {m: sig(m) for m in ["chrf_sub", "afri_ref", "ssa_ref", "afri_qe", "ssa_qe", "chrf"]}
        if s["afri_qe"] is None:
            continue
        qe_up = s["afri_qe"] == "+" or s["ssa_qe"] == "+"
        ref_up = [s[m] == "+" for m in ("chrf_sub", "afri_ref", "ssa_ref") if s[m] is not None]
        ref_dn = [s[m] == "-" for m in ("chrf_sub", "afri_ref", "ssa_ref") if s[m] is not None]
        flag = ""
        if qe_up and not any(ref_up):
            flag = "QE up, no reference-based metric up"
        elif qe_up and not all(ref_up):
            flag = "QE up, not all reference-based metrics up"
        if qe_up and any(ref_dn):
            flag = "QE up, a reference-based metric DOWN"
        fl.append(dict(subset=sub, b=bench, iso=iso, group=groups[iso]["group"], n_nn=int(r[("n", "afri_qe")]) if not np.isnan(r[("n", "afri_qe")]) else 0,
                       **{f"sig_{m}": v for m, v in s.items()}, **{f"d_{m}": r[("delta", m)] for m in ("chrf_sub", "afri_ref", "ssa_ref", "afri_qe", "ssa_qe")}, flag=flag))
    pd.DataFrame(fl).to_csv(os.path.join(out, "flags.csv"), index=False, float_format="%.5g")

    # ------------------------------------------------------------------ agreement between metrics (sentence level)
    arows, grows = [], []
    if has_nn:
        for sub, fsub in SUBSETS.items():
            msk = fsub(df)
            ok = msk & ~np.isnan(cols["afri_ref"][0] + cols["afri_ref"][1]) & ~np.isnan(cols["ssa_qe"][0] + cols["ssa_qe"][1]) & ~np.isnan(cols["ssa_ref"][0] + cols["ssa_ref"][1])
            if not ok.any():
                continue
            lab = {}
            dlt = {}
            for m in NN_METRICS:
                xa, xb = cols[m]
                dd = (xb - xa)[ok]
                dlt[m] = dd
                th = NN_METRICS[m][1]
                lab[m] = np.where(dd > th, 1, np.where(dd < -th, -1, 0))
            ms = list(NN_METRICS)
            for i, m1 in enumerate(ms):
                for m2 in ms[i + 1:]:
                    l1, l2 = lab[m1], lab[m2]
                    both = (l1 != 0) & (l2 != 0)
                    agree_nt = float((l1[both] == l2[both]).mean()) if both.any() else np.nan
                    agree3 = float((l1 == l2).mean())
                    pe = sum(((l1 == c).mean() * (l2 == c).mean()) for c in (-1, 0, 1))
                    kappa = (agree3 - pe) / (1 - pe) if pe < 1 else np.nan
                    rho = float(sps.spearmanr(dlt[m1], dlt[m2]).statistic)
                    arows.append(dict(subset=sub, scope="all_langs", metric_1=m1, metric_2=m2, n=int(ok.sum()), n_both_nontie=int(both.sum()),
                                      agree_nontie=agree_nt, agree_3class=agree3, kappa_3class=kappa, spearman_delta=rho))
            # per group: agreement of chrF++ with each neural metric + pooled win rates per metric
            for g in ("qe_validated", "blend", "consensus_only", "sft_only"):
                gm = ok & np.array([groups[i]["group"] == g for i in icol])
                if gm.sum() < 50:
                    continue
                for m in ms:
                    pass
                for m2 in ("afri_ref", "ssa_ref", "afri_qe", "ssa_qe"):
                    l1 = np.where((cols["chrf"][1] - cols["chrf"][0])[gm] > 1, 1, np.where((cols["chrf"][1] - cols["chrf"][0])[gm] < -1, -1, 0))
                    dd = (cols[m2][1] - cols[m2][0])[gm]
                    l2 = np.where(dd > 0.01, 1, np.where(dd < -0.01, -1, 0))
                    both = (l1 != 0) & (l2 != 0)
                    grows.append(dict(subset=sub, group=g, metric_2=m2, n=int(gm.sum()), n_both_nontie=int(both.sum()),
                                      agree_nontie_with_chrf=float((l1[both] == l2[both]).mean()) if both.any() else np.nan))
            # who wins the disagreements: B wins by QE but ref-ensemble says otherwise
            e = {}
            ensr = cols["ens_ref"][1][ok]
            q = cols["ens_qe"][1][ok]
            arows.append(dict(subset=sub, scope="all_langs", metric_1="ens_qe(B wins both QE)", metric_2="ens_ref",
                              n=int(ok.sum()), n_both_nontie=int(((q == 1)).sum()),
                              agree_nontie=float((ensr[q == 1] == 1).mean()) if (q == 1).any() else np.nan,
                              agree_3class=float((ensr[q == 1] == 0).mean()) if (q == 1).any() else np.nan,
                              kappa_3class=float((ensr[q == 1] == -1).mean()) if (q == 1).any() else np.nan, spearman_delta=np.nan))
    pd.DataFrame(arows).to_csv(os.path.join(out, "agreement.csv"), index=False, float_format="%.5g")
    pd.DataFrame(grows).to_csv(os.path.join(out, "agreement_by_group.csv"), index=False, float_format="%.5g")

    # ------------------------------------------------------------------ tie-threshold sensitivity (pooled over all sentences of the subset)
    srows = []
    for sub, fsub in SUBSETS.items():
        msk = fsub(df)
        for m, ths in (("chrf", [0, 0.5, 1, 2, 5]), ("afri_ref", [0, .005, .01, .02, .05]), ("afri_qe", [0, .005, .01, .02, .05]),
                       ("ssa_ref", [0, .005, .01, .02, .05]), ("ssa_qe", [0, .005, .01, .02, .05])):
            if m not in cols:
                continue
            xa, xb = cols[m]
            ok = msk & ~np.isnan(xa + xb)
            if not ok.any():
                continue
            dd = (xb - xa)[ok]
            for th in ths:
                w, l = float((dd > th).mean()), float((dd < -th).mean())
                srows.append(dict(subset=sub, metric=m, threshold=th, n=int(ok.sum()), win=w, tie=1 - w - l, loss=l, pref=w / (w + l) if w + l else np.nan,
                                  p_sign=pstat(int((dd > th).sum()), int((dd < -th).sum()))))
    pd.DataFrame(srows).to_csv(os.path.join(out, "threshold_sensitivity.csv"), index=False, float_format="%.5g")

    # ------------------------------------------------------------------ corpus-level chrF++ per unit (cross-check with the R5 tables)
    sys.path.insert(0, "/workspace/sft_build/pylibs")
    try:
        from sacrebleu.metrics import CHRF
        ch = CHRF(word_order=2)
        crow = []
        raw = {}
        for tag, name in (("a", a.a), ("b", a.b)):
            raw[tag] = {}
            with gzip.open(os.path.join(a.dir, f"sys_{name}.jsonl.gz"), "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    raw[tag][(r["b"], r["iso"], r["id"])] = (r["hyp"], r["ref"], r["pair_seen"], r["contaminated"])
        grp = defaultdict(list)
        for k in raw["a"]:
            if k in raw["b"]:
                grp[(k[0], k[1])].append(k)
        for (bench, iso), ks in sorted(grp.items()):
            for sub, f in (("all", lambda t: True), ("no_pair", lambda t: t[2] is False), ("clean", lambda t: t[3] is False)):
                kk = [k for k in ks if f(raw["a"][k])]
                if not kk:
                    continue
                refs = [[nfc(raw["a"][k][1]) for k in kk]]
                crow.append(dict(subset=sub, b=bench, iso=iso, n=len(kk), chrf_a=ch.corpus_score([nfc(raw["a"][k][0]) for k in kk], refs).score,
                                 chrf_b=ch.corpus_score([nfc(raw["b"][k][0]) for k in kk], refs).score))
        pd.DataFrame(crow).to_csv(os.path.join(out, "corpus_chrf_units.csv"), index=False, float_format="%.5g")
    except Exception as e:  # noqa: BLE001
        print("corpus chrF check skipped:", e)
    json.dump(dict(a=a.a, b=a.b, boot=BOOT, min_n=a.min_n, thresholds={m: v[1] for m, v in metrics.items()}, n_rows=int(len(df)),
                   n_neural_rows=int(df["afri_ref_a"].notna().sum())), open(os.path.join(out, "meta.json"), "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
