"""R14 stage 3: evaluate the NLLB signals against human references (a5 corruption rows, policy samples, R3 stuffing attack) and write the calibration file.
usage: s3_analyze.py OUT_DIR   (reads /workspace/jobs/r14/s3/sig_*, a5 scores, the stage-1 decision table copied to /workspace/jobs/r14/code/nllb600m_decision_table.csv)"""
import collections, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gemma_mt.rl import consensus as cons
from gemma_mt.rl.nllb_signal import calibrate_scores, chrf01
from gemma_mt.rl.reward_shaping import auc as _auc

R = "/workspace/jobs/r14"; S3 = R + "/s3"; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
jl = lambda p: [json.loads(l) for l in open(p)]
A5 = "/workspace/jobs/a5"
hack, pol = jl(A5 + "/hack_rows.jsonl"), jl(A5 + "/policy_samples.jsonl")
shp = json.load(open(R + "/code/shaping_flores_dev_v8.json"))
mode = {k: (v.get("content") or {}).get("mode", "qe") for k, v in shp.items() if isinstance(v, dict)}
HARD = ("mismatch", "neighbour", "drop_half", "append_unrelated", "replace_words")
PARTIAL = ("mix_half_src", "append_source", "swap25_src")
CLASSES = ("human", "mismatch", "neighbour", "copy_source", "copy_noisy", "drop_half", "shuffle", "append_unrelated", "replace_words") + PARTIAL
dec = {r["iso"]: r for r in csv.DictReader(open(R + "/code/nllb600m_decision_table.csv"))}
sig = {}
FAST600 = {"kea", "amh", "tir", "nus", "kbp", "bam", "kab", "ewe", "kin", "zul", "hau", "yor", "wol", "dyu"}      # the 600M run covered only these languages (rows filtered); expand back to full length
for m in ("3b3", "600m"):
    for name, rws in (("hack", hack), ("policy", pol)):
        p = f"{S3}/sig_{name}_{m}.json"
        if os.path.exists(p):
            d = json.load(open(p))
            if m == "600m":
                ii = [i for i, r in enumerate(rws) if r["iso"] in FAST600]
                full = {"pr": [None] * len(rws), "fd": [None] * len(rws)}
                for k in ("pr", "fd"):
                    for i, v in zip(ii, d[k]): full[k][i] = v
                d = full
            sig[(name, m)] = d
def sc_of(d, f): return json.load(open(f"{A5}/{d}/{f}.json"))
hs = {"africomet_qe": sc_of("hack_scores", "africomet_qe_stl11"), "ssa_mtl_qe": sc_of("hack_scores", "ssa_mtl_qe")}
ps = {"africomet_qe": sc_of("policy_scores", "africomet_qe_stl11"), "ssa_mtl_qe": sc_of("policy_scores", "ssa_mtl_qe")}
from gemma_mt.rl.nllb_signal import forced_to_unit, shape01
_dc = json.load(open(f"{S3}/sig_devcal_3b3.json")) if os.path.exists(f"{S3}/sig_devcal_3b3.json") else {}
_anch = {iso: (calibrate_scores(d["human_pr"], d["mism_pr"])["anchors"], calibrate_scores([forced_to_unit(x) for x in d["human_fd"]], [forced_to_unit(x) for x in d["mism_fd"]])["anchors"]) for iso, d in _dc.items()}
def combos(rows_, sg):
    mn, mean = [None] * len(rows_), [None] * len(rows_)
    for i, r in enumerate(rows_):
        a = _anch.get(r["iso"])
        if a is None or sg["pr"][i] is None or sg["fd"][i] is None: continue
        p_, f_ = shape01(sg["pr"][i], a[0]), shape01(forced_to_unit(sg["fd"][i]), a[1])
        mn[i], mean[i] = min(p_, f_), 0.5 * (p_ + f_)
    return mn, mean
metrics_h = dict(hs)
for m in ("3b3", "600m"):
    if ("hack", m) in sig:
        metrics_h[f"nllb{m}_pseudoref"] = sig[("hack", m)]["pr"]; metrics_h[f"nllb{m}_forced"] = sig[("hack", m)]["fd"]
        if m == "3b3":
            metrics_h["nllb3b3_combo_min"], metrics_h["nllb3b3_combo_mean"] = combos(hack, sig[("hack", m)])
langs = sorted({r["iso"] for r in hack if metrics_h.get("nllb3b3_pseudoref") and metrics_h["nllb3b3_pseudoref"][[i for i, x in enumerate(hack) if x["iso"] == r["iso"]][0]] is not None})
idx = collections.defaultdict(dict)
for i, r in enumerate(hack): idx[(r["iso"], r["cls"])][r["j"]] = i
rng = np.random.default_rng(14); B = 300
def arr(m, iso, c): return np.array([metrics_h[m][idx[(iso, c)][j]] for j in range(len(idx[(iso, "human")]))], float)
def auc_(h, mm):
    d = h[..., :, None] - mm[..., None, :]
    return ((d > 0).sum((-1, -2)) + 0.5 * (d == 0).sum((-1, -2))) / (d.shape[-1] * d.shape[-2])
fail = lambda h, c: (c >= h).mean(-1)
rows = []
for m in metrics_h:
    for iso in langs:
        h = arr(m, iso, "human"); n = len(h)
        if np.isnan(h).any(): continue
        draws = rng.integers(0, n, size=(B, n))
        rec = {"metric": m, "iso": iso, "mode": mode.get(iso, "qe"), "n": n, "auc_mismatch": float(auc_(h, arr(m, iso, "mismatch")))}
        rec["auc_mismatch_lo"], rec["auc_mismatch_hi"] = np.percentile(auc_(h[draws], arr(m, iso, "mismatch")[draws]), [5, 95])
        for c in CLASSES[1:]: rec["fail_" + c] = float(fail(h, arr(m, iso, c)))
        rec["fail_hard"] = float(np.mean([rec["fail_" + c] for c in HARD])); rec["fail_partial"] = float(np.mean([rec["fail_" + c] for c in PARTIAL]))
        rows.append(rec)
fields = list(rows[0]) if rows else []
for r in rows:
    for k in r:
        if k not in fields: fields.append(k)
with open(f"{OUT}/nllb600m_hack_per_language.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); [w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}) for r in rows]
# aggregate over language sets
def agg(metric, sel):
    rs = [r for r in rows if r["metric"] == metric and sel(r)]
    if not rs: return None
    o = {"n_langs": len(rs)}
    for k in ("auc_mismatch", "fail_hard", "fail_copy_source", "fail_copy_noisy", "fail_partial", "fail_drop_half", "fail_append_unrelated", "fail_shuffle", "fail_replace_words", "fail_neighbour"):
        o[k] = float(np.mean([r[k] for r in rs]))
    return o
sets = {"all covered": lambda r: True, "mode=qe": lambda r: r["mode"] == "qe", "mode=blend": lambda r: r["mode"] == "blend", "mode=consensus": lambda r: r["mode"] == "consensus",
        "teacher better (3.3B delta>=2)": lambda r: float((dec.get(r["iso"]) or {}).get("d33") or -99) >= 2, "teacher not better": lambda r: float((dec.get(r["iso"]) or {}).get("d33") or -99) < 2}
agg_rows = []
for sname, sf in sets.items():
    for m in metrics_h:
        a = agg(m, sf)
        if a: agg_rows.append({"set": sname, "metric": m, **{k: round(v, 3) if isinstance(v, float) else v for k, v in a.items()}})
with open(f"{OUT}/nllb600m_hack_aggregate.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(agg_rows[0])); w.writeheader(); w.writerows(agg_rows)

# ---------------------------------------------------------------- policy samples
def rankz(x):
    r = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float)
    return (r - r.mean()) / (r.std() + 1e-9)
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0: return float("nan")
    return float(np.corrcoef(rankz(a), rankz(b))[0, 1])
def partial(x, y, z):  # rank partial correlation of x,y given z
    rx, ry, rz = rankz(x), rankz(y), rankz(z)
    bx = np.polyfit(rz, rx, 1); by = np.polyfit(rz, ry, 1)
    ex, ey = rx - np.polyval(bx, rz), ry - np.polyval(by, rz)
    return float(np.corrcoef(ex, ey)[0, 1])
def valid(r): return bool(r.get("finished", True)) and r.get("prefix_status") == "ok" and bool(r["hyp"].strip()) and r["hyp"].strip() != r["src"].strip()
pidx = collections.defaultdict(list)
for i, r in enumerate(pol): pidx[r["iso"]].append(i)
pm = dict(ps)
for m in ("3b3", "600m"):
    if ("policy", m) in sig:
        pm[f"nllb{m}_pseudoref"] = sig[("policy", m)]["pr"]; pm[f"nllb{m}_forced"] = sig[("policy", m)]["fd"]
        if m == "3b3":
            pm["nllb3b3_combo_min"], pm["nllb3b3_combo_mean"] = combos(pol, sig[("policy", m)])
prow = []
for iso in langs:
    ii = pidx[iso]
    hy = np.array([chrf01(pol[i]["hyp"] or ".", pol[i]["ref"]) for i in ii])
    v = np.array([valid(pol[i]) for i in ii])
    grp = collections.defaultdict(list)
    for loc, i in enumerate(ii): grp[(pol[i]["id"])].append(loc)
    cvals = np.full(len(ii), np.nan)
    for g, locs in grp.items():
        profs = [cons.Profile(pol[ii[l]]["hyp"]) if v[l] else None for l in locs]
        out = cons.group_consensus(profs, 3)
        for l, o in zip(locs, out): cvals[l] = np.nan if o is None else o / 100.0
    ok = ~np.isnan(cvals) & v
    rec = {"iso": iso, "mode": mode.get(iso, "qe"), "n": int(ok.sum()), "spearman_consensus": spearman(cvals[ok], hy[ok])}
    for m, arr_ in pm.items():
        s = np.array([np.nan if arr_[i] is None else arr_[i] for i in ii])
        if np.isnan(s).any(): continue
        rec[f"spearman_{m}"] = spearman(s[ok], hy[ok])
        wg = []
        for g, locs in grp.items():
            l2 = [l for l in locs if ok[l]]
            if len(l2) >= 4: wg.append(spearman(s[l2], hy[l2]))
        rec[f"within_group_{m}"] = float(np.nanmean(wg)) if wg else float("nan")
        if "nllb" in m:
            rec[f"partial_{m}_given_consensus"] = partial(s[ok], hy[ok], cvals[ok])
            rec[f"partial_consensus_given_{m}"] = partial(cvals[ok], hy[ok], s[ok])
            if "africomet_qe" in pm and "nllb" in m:
                q = np.array(pm["africomet_qe"], float)[ii]
                rec[f"partial_{m}_given_qe"] = partial(s[ok], hy[ok], q[ok])
    prow.append(rec)
pf = sorted({k for r in prow for k in r})
with open(f"{OUT}/nllb600m_policy_per_language.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["iso", "mode", "n"] + [k for k in pf if k not in ("iso", "mode", "n")]); w.writeheader()
    [w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}) for r in prow]
# stuffing attack
att = json.load(open(f"{S3}/sig_attack_3b3.json")) if os.path.exists(f"{S3}/sig_attack_3b3.json") else {}
arows = []
for iso, d in att.items():
    if iso.startswith("_"): continue
    for sgn in ("pr", "fd"):
        h = np.array(d[f"human_{sgn}"])
        arows.append({"iso": iso, "signal": sgn, "fail_stuffed": float((np.array(d[f"stuffed_{sgn}"]) >= h).mean()), "fail_padded": float((np.array(d[f"padded_{sgn}"]) >= h).mean())})
if arows:
    with open(f"{OUT}/nllb600m_stuffing_attack.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(arows[0])); w.writeheader(); w.writerows(arows)
# ---------------------------------------------------------------- calibration file (dev, human-validated)
cal = {}
polmap = {r["iso"]: r for r in prow}
hk = {(r["metric"], r["iso"]): r for r in rows}
for m in ("3b3",):
    p = f"{S3}/sig_devcal_{m}.json"
    if not os.path.exists(p): continue
    dc = json.load(open(p))
    for iso, d in dc.items():
        pr = calibrate_scores(d["human_pr"], d["mism_pr"])
        from gemma_mt.rl.nllb_signal import forced_to_unit
        fd = calibrate_scores([forced_to_unit(x) for x in d["human_fd"]], [forced_to_unit(x) for x in d["mism_fd"]])
        dr = dec.get(iso) or {}
        tv = {"delta": float(dr["d33"]) if dr.get("d33") else None, "lo": float(dr["d33_lo"]) if dr.get("d33_lo") else None, "hi": float(dr["d33_hi"]) if dr.get("d33_hi") else None,
              "benchmark": dr.get("benchmark"), "teacher": "nllb-200-3.3B"}
        hkr = hk.get(("nllb3b3_pseudoref", iso)) or {}
        pol_r = polmap.get(iso) or {}
        pr.update({"hack_auc_mismatch": hkr.get("auc_mismatch"), "hack_fail_hard": hkr.get("fail_hard"), "hack_fail_partial": hkr.get("fail_partial"),
                   "policy_within_group_spearman": pol_r.get("within_group_nllb3b3_pseudoref"), "policy_partial_given_consensus": pol_r.get("partial_nllb3b3_pseudoref_given_consensus")})
        fh = hkr.get("fail_hard") if hkr.get("fail_hard") is not None else 1.0
        use = (tv["delta"] is not None and tv["delta"] >= 2 and (tv["lo"] or -1) > 0 and pr["auc_vs_mismatched"] >= 0.80 and fh <= 0.25
               and (pol_r.get("within_group_nllb3b3_pseudoref") or 0) >= 0.2)
        # weight factor: the further the teacher is above the student the more it can teach (delta/6, floor 0.4, 1 at +6); the hack-study failure rate (corrupted >= human) scales it down: 1 up to 0.10, 0 at 0.25
        pr["weight_factor"] = float(min(1.0, max(0.4, (tv["delta"] or 0) / 6.0)) * min(1.0, max(0.0, (0.25 - fh) / 0.15)))
        cal[iso] = {"nllb": {"pseudoref": pr, "forced": fd, "teacher_vs_student": tv, "use": bool(use), "licence_class": "reward_only_nc"}}
json.dump(cal, open(f"{OUT}/nllb_calibration_v1.json", "w"), indent=1)
print("languages", len(langs), "calibrated", len(cal), "use", sum(1 for v in cal.values() if v["nllb"]["use"]))
