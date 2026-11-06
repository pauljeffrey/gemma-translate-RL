#!/usr/bin/env python
"""Paired comparison of the in-training dev predictions (dev_predictions/step_N.jsonl) of runs that share the dev sentences (same trainer seed): corpus chrF++ per language, macro over the languages of the
`bench` set (and `weak_dev`), Poisson(1) weights per sentence shared by the runs, 1000 resamples, seed 20261104.
    python dev_compare.py STEP NAME=RUN_DIR ... --pair A:B ...  (venv_comet python)"""
import json, sys, unicodedata
import numpy as np
from sacrebleu.metrics import CHRF
step = sys.argv[1]; sysargs = [a for a in sys.argv[2:] if "=" in a]; pairs = sys.argv[sys.argv.index("--pair") + 1:] if "--pair" in sys.argv else []
chrf = CHRF(word_order=2); nfc = lambda s: unicodedata.normalize("NFC", s)
D = {}
for a in sysargs:
    n, d = a.split("=")
    D[n] = {}
    for l in open(f"{d}/dev_predictions/step_{step}.jsonl"):
        r = json.loads(l); D[n][(r["set"], r["iso"], r["id"])] = r
keys = sorted(set.intersection(*[set(v) for v in D.values()])); rng = np.random.default_rng(20261104); B = 1000
res = {}
for st in ("bench", "weak_dev"):
    ks = [k for k in keys if k[0] == st]
    if not ks: continue
    isos = sorted(set(k[1] for k in ks)); W = rng.poisson(1.0, size=(B, len(ks))).astype(np.float64)
    S = {n: np.array(chrf._extract_corpus_statistics([nfc(D[n][k]["hyp"]) for k in ks], [[nfc(D[n][k]["ref"]) for k in ks]]), dtype=np.float64) for n in D}
    def fs(St):
        h, r, m = St[..., 0:24:3], St[..., 1:24:3], St[..., 2:24:3]; ok = (h > 0) & (r > 0)
        p = np.where(ok, m / np.maximum(h, 1), 0); rc = np.where(ok, m / np.maximum(r, 1), 0); e = ok.sum(-1)
        ap = np.where(e > 0, p.sum(-1) / np.maximum(e, 1), 0); ar = np.where(e > 0, rc.sum(-1) / np.maximum(e, 1), 0); den = 4 * ap + ar
        return np.where(den > 0, 100 * 5 * ap * ar / np.where(den > 0, den, 1), 0)
    M = {}
    for n in D:
        pt, bt = [], []
        for iso in isos:
            ix = [j for j, k in enumerate(ks) if k[1] == iso]; s = S[n][ix]; pt.append(fs(s.sum(0)[None])[0]); bt.append(fs(W[:, ix] @ s))
        M[n] = (np.mean(pt), np.mean(bt, 0))
    print(f"== {st}: {len(ks)} sentences, {len(isos)} languages, step {step}")
    for n in D: print(f"  {n}: chrF++ macro {M[n][0]:.2f} [{np.percentile(M[n][1],2.5):.2f}, {np.percentile(M[n][1],97.5):.2f}]")
    for p in pairs:
        a, b = p.split(":"); d = M[a][1] - M[b][1]
        print(f"  {a} - {b}: {M[a][0]-M[b][0]:+.2f} [{np.percentile(d,2.5):+.2f}, {np.percentile(d,97.5):+.2f}]")
