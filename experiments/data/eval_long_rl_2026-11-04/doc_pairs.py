#!/usr/bin/env python
"""Paired bootstrap over the 168 test documents (28 papers x 6 languages; resampling documents; Poisson(1) weights, 1000 resamples, macro over the 6 languages): science RL final (the long RL's start)
against the long-document RL best / final: chrF (word order 0) of whole and chunked mode, finished share and loop share of whole mode (loop = rl/guards detector). CPU only.
    python doc_pairs.py OUT.json   (venv_comet python)"""
import importlib.util, json, sys, unicodedata
import numpy as np
from sacrebleu.metrics import CHRF
E = "/workspace/eval_385m"
spec = importlib.util.spec_from_file_location("g", "/workspace/clean_rerun/tree/gemma_mt/gemma_mt/rl/guards.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
S = {"sci_final": "sci_rl_final", "long_best": "long_rl_best", "long_final": "long_rl_final"}
P = {k: json.load(open(f"{E}/out_doc_{v}/documents_predictions.json")) for k, v in S.items()}
nfc = lambda x: unicodedata.normalize("NFC", x)
ch = CHRF(word_order=0)
isos = sorted(P["sci_final"]["whole"])
st, fin, loop = {}, {}, {}
for k in S:
    for mode in ("whole", "chunked"):
        for iso in isos:
            rows = P[k][mode][iso]
            st[(k, mode, iso)] = np.array(ch._extract_corpus_statistics([nfc(r["hyp"]) for r in rows], [[nfc(r["ref"]) for r in rows]]), dtype=float)
            if mode == "whole":
                fin[(k, iso)] = np.array([str(r["finished"]) == "True" for r in rows], dtype=float)
                loop[(k, iso)] = np.array([bool(g.is_repetitive(r["hyp"]) or g.is_self_repetitive(r["hyp"])) for r in rows], dtype=float)
rng = np.random.default_rng(20261104); B = 1000
W = {iso: rng.poisson(1.0, size=(B, len(P["sci_final"]["whole"][iso]))).astype(float) for iso in isos}
def f18(S_):
    S_ = np.atleast_2d(S_)[:, :18]; h, r, m = S_[:, 0::3], S_[:, 1::3], S_[:, 2::3]; ok = (h > 0) & (r > 0)
    p = np.where(ok, m / np.maximum(h, 1), 0); rc = np.where(ok, m / np.maximum(r, 1), 0); e = ok.sum(1)
    ap = np.where(e > 0, p.sum(1) / np.maximum(e, 1), 0); ar = np.where(e > 0, rc.sum(1) / np.maximum(e, 1), 0); den = 4 * ap + ar
    return np.where(den > 0, 100 * 5 * ap * ar / np.where(den > 0, den, 1), 0)
out = {}
for a, b in (("sci_final", "long_best"), ("sci_final", "long_final"), ("long_best", "long_final")):
    res = {}
    for name, fn in (("chrf_whole", lambda k, iso, w: (f18(st[(k, "whole", iso)].sum(0))[0], f18(w @ st[(k, "whole", iso)]))),
                     ("chrf_chunked", lambda k, iso, w: (f18(st[(k, "chunked", iso)].sum(0))[0], f18(w @ st[(k, "chunked", iso)]))),
                     ("finished_whole", lambda k, iso, w: (fin[(k, iso)].mean(), (w @ fin[(k, iso)]) / w.sum(1))),
                     ("loop_whole", lambda k, iso, w: (loop[(k, iso)].mean(), (w @ loop[(k, iso)]) / w.sum(1)))):
        da, pa, pb = [], [], []
        for iso in isos:
            ca, cb = fn(a, iso, W[iso]), fn(b, iso, W[iso]); pa.append(ca[0]); pb.append(cb[0]); da.append(cb[1] - ca[1])
        bd = np.mean(da, axis=0); lo, hi = np.percentile(bd, [2.5, 97.5])
        res[name] = dict(a=float(np.mean(pa)), b=float(np.mean(pb)), diff=float(np.mean(pb) - np.mean(pa)), lo=float(lo), hi=float(hi))
    out[f"{a}->{b}"] = res
    print(a, b, {k: (round(v["diff"], 3), round(v["lo"], 3), round(v["hi"], 3)) for k, v in res.items()}, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
