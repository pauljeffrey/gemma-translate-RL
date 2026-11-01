"""R14 widening study: is the NLLB pseudo-reference (signal A) better aligned with human judgement than the CURRENT reward components, per language, with CIs?
Evidence: (1) policy samples (human chrF++ vs the FLORES reference; cluster bootstrap over prompts), (2) a5 corruption rows (AUC human vs mismatched / copies / padded),
(3) SSA-MTE human direct-assessment scores (amh kik kin luo; not chrF-based). Writes CSVs + nllb_reward_calibration_v2.json.
usage: w_analyze.py OUT_DIR"""
import collections, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import rankdata
from gemma_mt.rl import consensus as cons
from gemma_mt.rl.nllb_signal import calibrate_scores, chrf01, shape01, forced_to_unit
from gemma_mt.rl.qe_ensemble import combine_pair

R = "/workspace/jobs/r14"; S3 = R + "/s3"; A5 = "/workspace/jobs/a5"; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(23)
jl = lambda p: [json.loads(l) for l in open(p)]
shp = json.load(open(R + "/code/shaping_flores_dev_v8.json")); DELTA = float(shp["_meta_v8"]["delta"])
dec = {r["iso"]: r for r in csv.DictReader(open(R + "/code/nllb600m_decision_table.csv"))}
cal1 = json.load(open(R + "/nllb_reward_calibration_v1.json"))
teacher_sft = {l["language"] for l in json.load(open(R + "/manifest.json"))["languages"]}
def mode_of(iso): return (shp.get(iso, {}).get("content") or {}).get("mode", "qe")
def curve(c, anchors): return shape01(c, anchors)          # same curve, floor 0 (fail_floor: 0.0 in the final yaml)
def qe_ens(iso, a, s):
    b = shp.get(iso) or {}; q2 = b.get("qe2") or {}
    r1 = curve(a, b["anchors"]) if b.get("anchors") else None
    r2 = curve(s, q2["anchors"]) if q2.get("anchors") and q2.get("usable", True) else None
    if r1 is not None and r2 is not None: return combine_pair(r1, r2, DELTA)[0]
    return r1 if r1 is not None else r2
LO, HI = -2.0, 2.25
def cons_value(c): return LO + (HI - LO) * min(max(c, 0), 100) / 100.0
def rk(x): return rankdata(x)
def sp(x, y):
    if len(x) < 4: return np.nan
    a, b = rk(x), rk(y)
    if a.std() == 0 or b.std() == 0: return np.nan
    return float(np.corrcoef(a, b)[0, 1])
def resid(y, Z):
    Z1 = np.column_stack([np.ones(len(y))] + [rk(z) for z in Z]); yy = rk(y)
    beta, *_ = np.linalg.lstsq(Z1, yy, rcond=None); return yy - Z1 @ beta
def partial(x, y, Z):
    if not all(np.isfinite(z).all() for z in Z) or not np.isfinite(x).all(): return np.nan
    try: ex, ey = resid(x, Z), resid(y, Z)
    except np.linalg.LinAlgError: return np.nan
    if ex.std() == 0 or ey.std() == 0: return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])
def ci(v): v = np.asarray(v, float); v = v[~np.isnan(v)]; return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)

# ------------------------------------------------------------------ 1. policy samples
pol = jl(A5 + "/policy_samples.jsonl")
sig = json.load(open(f"{S3}/sig_policy_3b3.json"))
ps_a = json.load(open(A5 + "/policy_scores/africomet_qe_stl11.json")); ps_s = json.load(open(A5 + "/policy_scores/ssa_mtl_qe.json"))
valid = lambda r: bool(r.get("finished", True)) and r.get("prefix_status") == "ok" and bool(r["hyp"].strip()) and r["hyp"].strip() != r["src"].strip()
idx = collections.defaultdict(list)
for i, r in enumerate(pol): idx[r["iso"]].append(i)
prow = []
B = 300
for iso, ii in sorted(idx.items()):
    if iso not in cal1 and iso not in dec or sig["pr"][ii[0]] is None: continue
    h = np.array([chrf01(pol[i]["hyp"] or ".", pol[i]["ref"]) for i in ii]); v = np.array([valid(pol[i]) for i in ii])
    A = np.array([sig["pr"][i] for i in ii]); Q = np.array([qe_ens(iso, ps_a[i], ps_s[i]) for i in ii], float)
    gid = np.array([pol[i]["id"] for i in ii]); cv = np.full(len(ii), np.nan)
    grp = collections.defaultdict(list)
    for loc, g in enumerate(gid): grp[g].append(loc)
    for g, locs in grp.items():
        o = cons.group_consensus([cons.Profile(pol[ii[l]]["hyp"]) if v[l] else None for l in locs], 3)
        for l, x in zip(locs, o): cv[l] = np.nan if x is None else x
    ok = ~np.isnan(cv) & v
    cvv = np.array([cons_value(x) if x == x else np.nan for x in cv])
    if np.isnan(Q).any(): Q = np.where(np.isnan(Q), cvv, Q)          # consensus-only languages have no QE: the consensus value stands in for it
    md = mode_of(iso)
    cur = Q.copy() if md == "qe" else (0.5 * Q + 0.5 * cvv if md == "blend" else cvv)       # the content reward the policy gets today (blend beta 0.5; consensus-only: consensus value)
    gl = [np.array([l for l in locs if ok[l]]) for locs in grp.values()]; gl = [g for g in gl if len(g) >= 4]
    def stats(sel_groups):
        locs = np.concatenate([gl[k] for k in sel_groups]); hh = h[locs]
        o = {"sp_A": sp(A[locs], hh), "sp_cur": sp(cur[locs], hh), "sp_Q": sp(Q[locs], hh), "sp_cons": sp(cv[locs], hh)}
        o["d_A_cur"] = o["sp_A"] - o["sp_cur"]; o["d_A_Q"] = o["sp_A"] - o["sp_Q"]
        wA = [sp(A[gl[k]], h[gl[k]]) for k in sel_groups]; wC = [sp(cur[gl[k]], h[gl[k]]) for k in sel_groups]
        o["wg_A"], o["wg_cur"] = float(np.nanmean(wA)), float(np.nanmean(wC)); o["wg_d"] = o["wg_A"] - o["wg_cur"]
        o["partial_A_given_cur"] = partial(A[locs], hh, [cur[locs]])
        o["partial_A_given_Q_cons"] = partial(A[locs], hh, [Q[locs], cvv[locs]])
        o["partial_cur_given_A"] = partial(cur[locs], hh, [A[locs]])
        return o
    pt = stats(range(len(gl))); bs = collections.defaultdict(list)
    for _ in range(B):
        o = stats(rng.integers(0, len(gl), len(gl)))
        for k, x in o.items(): bs[k].append(x)
    rec = {"iso": iso, "mode": md, "n_samples": int(ok.sum()), "n_groups": len(gl)}
    for k, x in pt.items():
        lo, hi = ci(bs[k]); rec[k] = x; rec[k + "_lo"] = lo; rec[k + "_hi"] = hi
    prow.append(rec)
    print("policy", iso, round(rec["sp_A"], 2), round(rec["sp_cur"], 2), flush=True)
def wcsv(path, rows):
    if not rows: return
    f = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("iso", "mode"), k))
    w = csv.DictWriter(open(path, "w", newline=""), fieldnames=f); w.writeheader()
    for r in rows: w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
wcsv(f"{OUT}/nllb_widening_policy_per_language.csv", prow)

# ------------------------------------------------------------------ 2. corruption rows: AUC human vs class
hack = jl(A5 + "/hack_rows.jsonl"); hs = {"a": json.load(open(A5 + "/hack_scores/africomet_qe_stl11.json")), "s": json.load(open(A5 + "/hack_scores/ssa_mtl_qe.json"))}
sh = json.load(open(f"{S3}/sig_hack_3b3.json"))
hi_ = collections.defaultdict(dict)
for i, r in enumerate(hack): hi_[(r["iso"], r["cls"])][r["j"]] = i
CL = {"mismatch": "mismatch", "copy": ("copy_source", "copy_noisy"), "padded": ("append_unrelated",), "drop_half": ("drop_half",), "partial_copy": ("mix_half_src", "append_source", "swap25_src")}
def auc(h, c):
    d = h[:, None] - c[None, :]; return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)
hrow = []
for iso in sorted({r["iso"] for r in hack}):
    if sh["pr"][hi_[(iso, "human")][0]] is None: continue
    n = len(hi_[(iso, "human")])
    def col(cls, which):
        ii = [hi_[(iso, cls)][j] for j in range(n)]
        return np.array([sh["pr"][i] for i in ii]) if which == "A" else np.array([qe_ens(iso, hs["a"][i], hs["s"][i]) for i in ii], float)
    rec = {"iso": iso, "mode": mode_of(iso)}
    for which in ("A", "Q"):
        H = col("human", which)
        for name, cls in CL.items():
            cls = (cls,) if isinstance(cls, str) else cls
            C = np.concatenate([col(c, which) for c in cls])
            if name == "mismatch": a = auc(H, C)
            else: a = float(np.mean([auc(H, col(c, which)) for c in cls]))
            bsv = []
            for _ in range(200):
                d = rng.integers(0, n, n)
                if name == "mismatch": bsv.append(auc(H[d], col("mismatch", which)[d]))
                else: bsv.append(float(np.mean([auc(H[d], col(c, which)[d]) for c in cls])))
            lo, hi = ci(bsv); rec[f"auc_{name}_{which}"] = a; rec[f"auc_{name}_{which}_lo"] = lo; rec[f"auc_{name}_{which}_hi"] = hi
    hrow.append(rec)
wcsv(f"{OUT}/nllb_widening_hack_auc_per_language.csv", hrow)
print("hack done", flush=True)

# ------------------------------------------------------------------ 3. SSA-MTE human DA (not chrF based)
srow = []
if os.path.exists(f"{S3}/ssamte_refs_3b3.json"):
    rows = jl(A5 + "/ssamte_rows.jsonl"); refs = json.load(open(f"{S3}/ssamte_refs_3b3.json"))
    sc = {"africomet_qe": json.load(open(A5 + "/ssamte_scores/africomet_qe_stl11.json")), "ssa_mtl_qe": json.load(open(A5 + "/ssamte_scores/ssa_mtl_qe.json"))}
    assert all(len(v) == len(rows) for v in sc.values()), "ssamte score arrays are not aligned"
    for lp in sorted({r["lp"] for r in rows if r["lp"].startswith("eng-")}):
        iso = lp[4:]
        if iso not in refs: continue
        ii = [i for i, r in enumerate(rows) if r["lp"] == lp and r["system"] != "NLLB" and r["mt"].strip()]       # NLLB's own output would be circular
        z = np.array([rows[i]["z"] for i in ii]); A = np.array([chrf01(rows[i]["mt"], refs[iso][rows[i]["src"]]) for i in ii])
        a, s = np.array([sc["africomet_qe"][i] for i in ii]), np.array([sc["ssa_mtl_qe"][i] for i in ii])
        Q = np.array([qe_ens(iso, x, y) if shp.get(iso, {}).get("anchors") else np.nan for x, y in zip(a, s)], float)
        srcs = np.array([rows[i]["src"] for i in ii]); us = sorted(set(srcs)); sidx = [np.flatnonzero(srcs == u) for u in us]
        def st(sel):
            loc = np.concatenate([sidx[k] for k in sel]); o = {"sp_A": sp(A[loc], z[loc]), "sp_africomet": sp(a[loc], z[loc]), "sp_ssa_mtl": sp(s[loc], z[loc])}
            if not np.isnan(Q).all(): o["sp_QEens"] = sp(Q[loc], z[loc])
            for nm, x in (("A", A), ("africomet", a), ("ssa_mtl", s)):
                o["wsrc_" + nm] = float(np.nanmean([sp(x[sidx[k]], z[sidx[k]]) for k in sel]))
            return o
        pt = st(range(len(us))); bs = collections.defaultdict(list)
        for _ in range(500):
            for k, x in st(rng.integers(0, len(us), len(us))).items(): bs[k].append(x)
        rec = {"iso": iso, "n_judgements": len(ii), "n_sources": len(us)}
        for k, x in pt.items(): lo, hi = ci(bs[k]); rec[k] = x; rec[k + "_lo"] = lo; rec[k + "_hi"] = hi
        dd = np.array(bs["sp_A"]) - np.array(bs["sp_ssa_mtl"]); rec["d_A_minus_ssa_lo"], rec["d_A_minus_ssa_hi"] = ci(dd)
        dd = np.array(bs["sp_A"]) - np.array(bs["sp_africomet"]); rec["d_A_minus_africomet_lo"], rec["d_A_minus_africomet_hi"] = ci(dd)
        srow.append(rec); print("ssamte", iso, round(rec["sp_A"], 2), round(rec["sp_ssa_mtl"], 2), flush=True)
    wcsv(f"{OUT}/nllb_widening_ssamte_human_da.csv", srow)

# ------------------------------------------------------------------ 4. calibration v2 (weights)
devcal = json.load(open(f"{S3}/sig_devcal_3b3.json")); pr_by = {r["iso"]: r for r in prow}; hk_by = {r["iso"]: r for r in hrow}
cal2 = {}
for iso, d in devcal.items():
    pr = calibrate_scores(d["human_pr"], d["mism_pr"]); fd = calibrate_scores([forced_to_unit(x) for x in d["human_fd"]], [forced_to_unit(x) for x in d["mism_fd"]])
    r = dec.get(iso) or {}; p = pr_by.get(iso); h = hk_by.get(iso)
    delta = float(r["d33"]) if r.get("d33") else None; dlo = float(r["d33_lo"]) if r.get("d33_lo") else None
    old = (cal1.get(iso) or {}).get("nllb", {}).get("pseudoref", {})
    fh = None
    # hard-class failure from the v1 study (same-source failure rate)
    fh = old.get("hack_fail_hard")
    gates = {"teacher": delta is not None and delta >= 2 and (dlo or -1) > 0, "auc_dev": pr["auc_vs_mismatched"] >= 0.80, "hack": (fh if fh is not None else 1) <= 0.25,
             "within_group": p is not None and p["wg_A"] >= 0.20}
    if p is None: align = 0.0
    elif p["d_A_cur_lo"] > 0: align = 1.0
    elif p["d_A_cur"] > 0: align = 0.7
    elif p["d_A_cur_hi"] > 0: align = 0.4
    else: align = 0.0
    pf = 1.0 if (p is not None and p["partial_A_given_cur_lo"] > 0) else 0.8
    wf_prev = float(min(1.0, max(0.4, (delta or 0) / 6.0)) * min(1.0, max(0.0, (0.25 - (fh if fh is not None else 1.0)) / 0.15)))
    use = all(gates.values()) and align > 0
    factor = wf_prev * align * pf
    pr.update({"weight_factor": factor if use else 0.0, "weight_factor_v1": wf_prev, "alignment_factor": align, "partial_factor": pf,
               "policy_spearman_A": p and p["sp_A"], "policy_spearman_current_reward": p and p["sp_cur"], "policy_spearman_A_minus_current": p and p["d_A_cur"],
               "policy_spearman_A_minus_current_ci": p and [p["d_A_cur_lo"], p["d_A_cur_hi"]], "policy_within_group_A": p and p["wg_A"], "policy_within_group_current": p and p["wg_cur"],
               "policy_partial_A_given_current": p and p["partial_A_given_cur"], "policy_partial_A_given_current_ci": p and [p["partial_A_given_cur_lo"], p["partial_A_given_cur_hi"]],
               "hack_fail_hard": fh, "hack_auc_mismatch_A": h and h["auc_mismatch_A"], "hack_auc_mismatch_Q": h and h["auc_mismatch_Q"],
               "hack_auc_copy_A": h and h["auc_copy_A"], "hack_auc_padded_A": h and h["auc_padded_A"], "hack_auc_padded_Q": h and h["auc_padded_Q"], "gates": gates})
    w = round(min(0.5, 0.35 * pr["reliability"] * pr["weight_factor"]), 3) if use else 0.0
    cal2[iso] = {"nllb": {"pseudoref": pr, "forced": fd, "teacher_vs_student": {"delta": delta, "lo": dlo, "hi": float(r["d33_hi"]) if r.get("d33_hi") else None, "benchmark": r.get("benchmark"), "teacher": "nllb-200-3.3B"},
                          "use": bool(use), "weight": w, "weight_if_teacher_sft_in_mixture": round(0.5 * w, 3), "teacher_data_in_teacher_v1": iso in teacher_sft, "licence_class": "reward_only_nc"}}
json.dump(cal2, open(f"{OUT}/nllb_reward_calibration_v2.json", "w"), indent=1)
print("use", sorted(k for k, v in cal2.items() if v["nllb"]["use"]))
