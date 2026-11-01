"""R14 stage 3 (reference row): how much do the existing QE scores add beyond the group consensus on the same policy samples? Same definitions as s3_analyze.py."""
import collections, json, sys
sys.path.insert(0, "/workspace/jobs/r14/code")
import numpy as np
from gemma_mt.rl import consensus as cons
from gemma_mt.rl.nllb_signal import chrf01
A5 = "/workspace/jobs/a5"
pol = [json.loads(l) for l in open(A5 + "/policy_samples.jsonl")]
S = {"africomet_qe": json.load(open(A5 + "/policy_scores/africomet_qe_stl11.json")), "ssa_mtl_qe": json.load(open(A5 + "/policy_scores/ssa_mtl_qe.json"))}
def rankz(x):
    r = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float); return (r - r.mean()) / (r.std() + 1e-9)
def partial(x, y, z):
    rx, ry, rz = rankz(x), rankz(y), rankz(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])
valid = lambda r: bool(r.get("finished", True)) and r.get("prefix_status") == "ok" and bool(r["hyp"].strip()) and r["hyp"].strip() != r["src"].strip()
idx = collections.defaultdict(list)
for i, r in enumerate(pol): idx[r["iso"]].append(i)
out = collections.defaultdict(list)
for iso, ii in idx.items():
    hy = np.array([chrf01(pol[i]["hyp"] or ".", pol[i]["ref"]) for i in ii]); v = np.array([valid(pol[i]) for i in ii])
    grp = collections.defaultdict(list)
    for loc, i in enumerate(ii): grp[pol[i]["id"]].append(loc)
    cv = np.full(len(ii), np.nan)
    for g, locs in grp.items():
        o = cons.group_consensus([cons.Profile(pol[ii[l]]["hyp"]) if v[l] else None for l in locs], 3)
        for l, x in zip(locs, o): cv[l] = np.nan if x is None else x / 100
    ok = ~np.isnan(cv) & v
    for m, sc in S.items():
        s = np.array([sc[i] for i in ii], float)
        out[m + "_given_consensus"].append(partial(s[ok], hy[ok], cv[ok])); out["consensus_given_" + m].append(partial(cv[ok], hy[ok], s[ok]))
for k, v in out.items(): print(k, round(float(np.nanmean(v)), 3), len(v))
