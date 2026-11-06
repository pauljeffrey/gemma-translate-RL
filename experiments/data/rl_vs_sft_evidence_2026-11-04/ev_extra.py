#!/usr/bin/env python
"""Per-cell retention of the paired 'both finished, no loop' subset, SFT / RL finish and loop rates, and the loop detector's rate on the HUMAN
REFERENCES (its false-positive floor). CPU only.   python ev_extra.py EVID_DIR MIXED_PREDICTIONS.jsonl OUT.json"""
import importlib.util
import json
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("g", "/workspace/clean_rerun/tree/gemma_mt/gemma_mt/rl/guards.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
evid, pred, out = sys.argv[1:4]
Z = {s: np.load(f"{evid}/{s}.npz") for s in ("sft", "mixed")}
key = lambda z: {(b, i, r): n for n, (b, i, r) in enumerate(zip(z["bench"], z["iso"], z["rid"]))}
K = {s: key(Z[s]) for s in Z}
ref_flag = {}
seen = set()
for line in open(pred):
    r = json.loads(line)
    k = (r["benchmark"], r["iso"], str(r["id"]))
    if k in seen:
        continue
    seen.add(k)
    ref_flag[k] = bool(g.is_repetitive(r["reference"]) or g.is_self_repetitive(r["reference"]))
cells = {}
for k, n in K["mixed"].items():
    if k not in K["sft"] or k[0] not in ("flores_plus", "ntrex", "mafand"):
        continue
    m = K["sft"][k]
    c = cells.setdefault(f"{k[0]}:{k[1]}", dict(n=0, good=0, sft_fin=0, sft_loop=0, rl_fin=0, rl_loop=0, ref_flag=0))
    sf, sl = bool(Z["sft"]["finished"][m]), bool(Z["sft"]["loop"][m])
    rf, rl = bool(Z["mixed"]["finished"][n]), bool(Z["mixed"]["loop"][n])
    c["n"] += 1; c["good"] += int(sf and not sl and rf and not rl)
    c["sft_fin"] += int(sf); c["sft_loop"] += int(sl); c["rl_fin"] += int(rf); c["rl_loop"] += int(rl); c["ref_flag"] += int(ref_flag[k])
json.dump(cells, open(out, "w"), indent=1)
tot = {}
for name, c in cells.items():
    b = name.split(":")[0]
    t = tot.setdefault(b, dict(n=0, good=0, sft_fin=0, sft_loop=0, rl_fin=0, rl_loop=0, ref_flag=0))
    for k in t: t[k] += c[k]
print({b: {k: round(v / t["n"], 4) for k, v in t.items() if k != "n"} for b, t in tot.items()})
