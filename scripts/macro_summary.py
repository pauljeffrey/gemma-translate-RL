#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Macro bootstrap summary of RL - SFT from the per-language replicate arrays of the box job macro_boot.py.
usage: macro_summary.py macro_boot.pkl reports/data/final_model_runs/macro_boot.json   (needs numpy)"""
import json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gemma_mt"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_final_per_language import BENCH_ORDER, FAMILY, group_of, L  # noqa: E402

D = pickle.load(open(sys.argv[1], "rb"))
isos = list(L.LANGUAGES)
SUBS = ["all", "no_pair", "clean"]
out = {}
def summ(cells, key):
    """cells: list of (n, res) -> dict for metric key"""
    a = np.mean([c[1][key + "_sft"] for c in cells], axis=0); b = np.mean([c[1][key + "_rl"] for c in cells], axis=0); d = b - a
    pt = float(np.mean([c[1][key + "_rl_pt"] - c[1][key + "_sft_pt"] for c in cells]))
    return {"d": pt, "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}
def groups_for(members):
    g = {"all": members, "rl77": [i for i in members if group_of(i) != "sft_only"]}
    for name in ("qe_validated", "blend", "consensus_only", "sft_only"):
        g[name] = [i for i in members if group_of(i) == name]
    for fam in set(FAMILY.values()):
        g["fam:" + fam] = [i for i in members if FAMILY[i] == fam]
    return g
for b in BENCH_ORDER:
    for s in SUBS:
        members = [i for i in isos if (b, i, s) in D]
        for g, mem in groups_for(members).items():
            if mem:
                out[f"{b}|{s}|{g}"] = {m: summ([D[(b, i, s)] for i in mem], m) for m in ("chrf", "spbleu")}
prim = {}
for i in isos:
    for b in BENCH_ORDER:
        if (b, i, "no_pair") in D:
            prim[i] = b; break
members = list(prim)
for g, mem in groups_for(members).items():
    if mem:
        out[f"primary|no_pair|{g}"] = {m: summ([D[(prim[i], i, "no_pair")] for i in mem], m) for m in ("chrf", "spbleu")}
json.dump(out, open(sys.argv[2], "w"), indent=0)
print(len(out), "entries")
