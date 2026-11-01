"""R17 widening: does mixing the shaped pseudo-reference chrF++ (A) into the current QE ensemble improve the alignment with HUMAN DIRECT-ASSESSMENT (SSA-MTE z-scores)?
(1-w)*QEens + w*shaped(A) vs QEens, w in {0.15, 0.25, 0.35, 0.5}; Spearman pooled and within-source, cluster bootstrap over sources. usage: w_blend_da.py OUT.csv"""
import collections, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import rankdata
from gemma_mt.rl.nllb_signal import calibrate_scores, chrf01, shape01
from gemma_mt.rl.qe_ensemble import combine_pair
R = "/workspace/jobs/r14"; S3 = R + "/s3"; A5 = "/workspace/jobs/a5"
rng = np.random.default_rng(29); jl = lambda p: [json.loads(l) for l in open(p)]
shp = json.load(open(R + "/code/shaping_flores_dev_v8.json")); DELTA = float(shp["_meta_v8"]["delta"])
def qe_ens(iso, a, s):
    b = shp.get(iso) or {}; q2 = b.get("qe2") or {}
    r1 = shape01(a, b["anchors"]) if b.get("anchors") else None
    r2 = shape01(s, q2["anchors"]) if q2.get("anchors") and q2.get("usable", True) else None
    if r1 is not None and r2 is not None: return combine_pair(r1, r2, DELTA)[0]
    return r1 if r1 is not None else r2
def sp(x, y):
    if len(x) < 4: return np.nan
    a, b = rankdata(x), rankdata(y)
    return np.nan if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
devcal = json.load(open(f"{S3}/sig_devcal_3b3.json")); rows = jl(A5 + "/ssamte_rows.jsonl"); refs = json.load(open(f"{S3}/ssamte_refs_3b3.json"))
sc = {"a": json.load(open(A5 + "/ssamte_scores/africomet_qe_stl11.json")), "s": json.load(open(A5 + "/ssamte_scores/ssa_mtl_qe.json"))}
W = [0.0, 0.15, 0.25, 0.35, 0.5]; out = []
for iso in sorted(refs):
    if iso not in devcal or not (shp.get(iso) or {}).get("anchors"): continue
    anc = calibrate_scores(devcal[iso]["human_pr"], devcal[iso]["mism_pr"])["anchors"]
    ii = [i for i, r in enumerate(rows) if r["lp"] == "eng-" + iso and r["system"] != "NLLB" and r["mt"].strip()]
    z = np.array([rows[i]["z"] for i in ii]); srcs = np.array([rows[i]["src"] for i in ii])
    Av = np.array([shape01(chrf01(rows[i]["mt"], refs[iso][rows[i]["src"]]), anc) for i in ii]); Q = np.array([qe_ens(iso, sc["a"][i], sc["s"][i]) for i in ii], float)
    us = sorted(set(srcs)); sidx = [np.flatnonzero(srcs == u) for u in us]
    def st(sel):
        loc = np.concatenate([sidx[k] for k in sel]); o = {}
        for w in W:
            m = (1 - w) * Q + w * Av
            o[f"sp_{w}"] = sp(m[loc], z[loc]); o[f"wsrc_{w}"] = float(np.nanmean([sp(m[sidx[k]], z[sidx[k]]) for k in sel]))
        return o
    pt = st(range(len(us))); bs = collections.defaultdict(list)
    for _ in range(500):
        for k, v in st(rng.integers(0, len(us), len(us))).items(): bs[k].append(v)
    rec = {"iso": iso, "n_judgements": len(ii)}
    for w in W:
        rec[f"sp_w{w}"] = pt[f"sp_{w}"]; rec[f"wsrc_w{w}"] = pt[f"wsrc_{w}"]
        if w > 0:
            d = np.array(bs[f"sp_{w}"]) - np.array(bs["sp_0.0"]); rec[f"d_sp_w{w}_lo"], rec[f"d_sp_w{w}_hi"] = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            d = np.array(bs[f"wsrc_{w}"]) - np.array(bs["wsrc_0.0"]); rec[f"d_wsrc_w{w}_lo"], rec[f"d_wsrc_w{w}_hi"] = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    out.append(rec); print(iso, {k: round(v, 3) for k, v in rec.items() if k.startswith("sp_w")}, flush=True)
f = sorted({k for r in out for k in r}, key=lambda k: (k != "iso", k)); w_ = csv.DictWriter(open(sys.argv[1], "w", newline=""), fieldnames=f); w_.writeheader()
[w_.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}) for r in out]
