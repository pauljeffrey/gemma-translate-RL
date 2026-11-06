#!/usr/bin/env python
"""Add the independent degeneration flag of E-08 (`iloop`) to a harness npz (ev_stats.py layout): lzma residual (z < -0.2536) against a log-log fit per (benchmark, language) cell on the cell's human references,
repeated char 8-gram share > 0.2345, repeated word 3-gram share > 0.1429 (reports/data/degeneration_audit_independent_2026-11-04/thresholds.json).
    python e_iloop.py NPZ_IN NPZ_OUT PRED.jsonl [PRED2.jsonl ...] [--langs a,b,c]"""
import json, lzma, sys, unicodedata
import numpy as np
THR = {"z": -0.2535888440397546, "c8": 0.23451316190228919, "w3": 0.1428571428571429}
_LZ = [{"id": lzma.FILTER_LZMA2, "preset": 9, "dict_size": 1 << 17}]   # dict_size 128 KB: identical size for texts < 128 KB, 30x faster set-up than the 64 MB default of preset 9
nfc = lambda s: unicodedata.normalize("NFC", s or "")
def rep(items): return 0.0 if not items else 1.0 - len(set(items)) / len(items)
def feats(t):
    t = nfc(t); b = t.encode("utf-8"); s = " ".join(t.lower().split()); w = s.split()
    return len(b), (len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=_LZ)) if b else 0), rep([s[i:i + 8] for i in range(len(s) - 7)]), rep([tuple(w[i:i + 3]) for i in range(len(w) - 2)])
args = [a for a in sys.argv[1:] if not a.startswith("--")]; langs = None
if "--langs" in sys.argv: langs = set(sys.argv[sys.argv.index("--langs") + 1].split(",")); args = [a for a in args if a != sys.argv[sys.argv.index("--langs") + 1]]
npz_in, npz_out, preds = args[0], args[1], args[2:]
z = dict(np.load(npz_in)); want = {(b, i, str(r)): k for k, (b, i, r) in enumerate(zip(z["bench"], z["iso"], z["rid"]))}
rows = {}
for p in preds:
    for l in open(p):
        r = json.loads(l); k = (r["benchmark"], r["iso"], str(r["id"]))
        if k in want and k not in rows: rows[k] = r
cells = {}
for k, r in rows.items(): cells.setdefault((k[0], k[1]), []).append(k)
iloop = np.zeros(len(z["bench"]), bool); cnt = 0
for cell, ks in cells.items():
    fr = [feats(rows[k]["reference"]) for k in ks]; ok = [f for f in fr if f[0] >= 20]
    lb = np.log([f[0] for f in ok]); lc = np.log([max(f[1], 1) for f in ok])
    c = np.polyfit(lb, lc, 1) if len(ok) >= 100 and lb.std() > 0 else None
    for k in ks:
        nb, lz, c8, w3 = feats(rows[k]["hyp"]); zz = np.nan
        if nb >= 20 and c is not None: zz = np.log(max(lz, 1)) - (c[0] * np.log(nb) + c[1])
        iloop[want[k]] = bool((np.isfinite(zz) and zz < THR["z"]) or c8 > THR["c8"] or w3 > THR["w3"]); cnt += 1
z["iloop"] = iloop; np.savez_compressed(npz_out, **z); print(npz_out, "rows", cnt, "of", len(iloop), "iloop rate", round(float(iloop[[want[k] for k in rows]].mean()), 4))
