#!/usr/bin/env python
"""Paired sentence-level comparison on AfriScience-MT dev/test (6 languages): science RL final (the long-document RL's start) against the long-document RL best / final.
Per sentence: AfriCOMET-STL-1.1 (reference-based, held out of every reward), SSA-COMET-QE (reference-free; REWARD FAMILY of the science RL, reported only as the paper's headline with that label),
plain chrF and chrF++ statistics. Paired Poisson bootstrap (1000 resamples, independent weights per language, shared by the two systems), macro over the 6 languages.
    python sci_pairs.py OUT.json [cap_gb]   (venv_comet python, PYTHONPATH=pylibs_none:gemma_mt_tree)"""
import glob, json, sys, unicodedata
E = "/workspace/eval_385m"
sys.path.insert(0, f"{E}/gemma_mt_tree")
import numpy as np, torch
torch.cuda.set_per_process_memory_fraction(float(sys.argv[2] if len(sys.argv) > 2 else 8) / 79.0)
import pyarrow.parquet as pq
from sacrebleu.metrics import CHRF
from gemma_mt.rl.comet import CometScorer, REF_MODEL_ID
SYS = {"sci_best": "sci_rl_best", "sci_final": "sci_rl_final", "long_best": "long_rl_best", "long_final": "long_rl_final"}
PAIRS = [("sci_final", "long_best"), ("sci_final", "long_final"), ("long_best", "long_final"), ("sci_best", "long_best")]
snap = glob.glob("/workspace/clean_sft/hf_home/hub/datasets--dsfsi--afriscience_mt/snapshots/*/corpus")[0]
src = {sp: {} for sp in ("dev", "test")}
for sp in src:
    for r in pq.read_table(f"{snap}/{sp}-00000.parquet").to_pylist(): src[sp].setdefault(r["target_lang"], []).append(r["source"])
nfc = lambda x: unicodedata.normalize("NFC", x)
ch = CHRF(word_order=2)
ac = CometScorer(REF_MODEL_ID, batch_size=32, reference_based=True)
qe = CometScorer("McGill-NLP/ssa-comet-qe", batch_size=32, reference_based=False)
D = {}
for name, d in SYS.items():
    for sp in ("dev", "test"):
        P = json.load(open(f"{E}/out_afsci_{d}/predictions_{sp}.json"))["sentence"]
        for iso, rows in P.items():
            h = [r["hyp"] for r in rows]; rf = [r["ref"] for r in rows]; s = src[sp][iso]
            st = np.array(ch._extract_corpus_statistics([nfc(x) for x in h], [[nfc(x) for x in rf]]), dtype=np.float64)
            D[(name, sp, iso)] = dict(stats=st, afri=np.array([float(x) for x in ac(s, h, rf)]), qe=np.array([float(x) for x in qe(s, h)]), n=len(rows),
                                      fin=np.array([bool(r.get("finished", True)) for r in rows]))
        print(name, sp, flush=True)
def fscore(S, k):
    S = np.atleast_2d(S)[:, :3 * k]; h, r, m = S[:, 0::3], S[:, 1::3], S[:, 2::3]
    ok = (h > 0) & (r > 0); p = np.where(ok, m / np.maximum(h, 1), 0); rc = np.where(ok, m / np.maximum(r, 1), 0); e = ok.sum(1)
    ap = np.where(e > 0, p.sum(1) / np.maximum(e, 1), 0); ar = np.where(e > 0, rc.sum(1) / np.maximum(e, 1), 0); den = 4 * ap + ar
    return np.where(den > 0, 100 * 5 * ap * ar / np.where(den > 0, den, 1), 0)
rng = np.random.default_rng(20261104); B = 1000; out = {"pairs": {}, "per_system": {}}
isos = sorted({k[2] for k in D})
W = {(sp, iso): rng.poisson(1.0, size=(B, D[("sci_final", sp, iso)]["n"])).astype(np.float64) for sp in ("dev", "test") for iso in isos}
def cell(sysname, sp, iso, w):
    x = D[(sysname, sp, iso)]; bt = w @ x["stats"]; tot = x["stats"].sum(0)
    return {"chrf": (fscore(tot, 6)[0], fscore(bt[:, :18], 6)), "chrfpp": (fscore(tot, 8)[0], fscore(bt, 8)), "afri": (x["afri"].mean(), (w @ x["afri"]) / w.sum(1)), "qe": (x["qe"].mean(), (w @ x["qe"]) / w.sum(1)),
            "fin": (x["fin"].mean(), None)}
for sp in ("dev", "test"):
    for a, b in PAIRS:
        res = {}
        for m in ("chrf", "chrfpp", "afri", "qe"):
            pa, pb, da = [], [], []
            per = {}
            for iso in isos:
                ca, cb = cell(a, sp, iso, W[(sp, iso)]), cell(b, sp, iso, W[(sp, iso)])
                pa.append(ca[m][0]); pb.append(cb[m][0]); da.append(cb[m][1] - ca[m][1])
                lo, hi = np.percentile(da[-1], [2.5, 97.5]); per[iso] = dict(a=float(ca[m][0]), b=float(cb[m][0]), diff=float(cb[m][0] - ca[m][0]), lo=float(lo), hi=float(hi))
            bd = np.mean(da, axis=0); lo, hi = np.percentile(bd, [2.5, 97.5])
            res[m] = dict(a=float(np.mean(pa)), b=float(np.mean(pb)), diff=float(np.mean(pb) - np.mean(pa)), lo=float(lo), hi=float(hi), per_language=per)
        out["pairs"][f"{a}->{b}/{sp}"] = res
    for name in SYS:
        out["per_system"][f"{name}/{sp}"] = {iso: dict(finished=float(D[(name, sp, iso)]["fin"].mean())) for iso in isos}
json.dump(out, open(sys.argv[1], "w"), indent=1); print("DONE")
