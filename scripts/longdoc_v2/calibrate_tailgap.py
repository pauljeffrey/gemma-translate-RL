# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adds `tail_gap` (per length bucket: median over HUMAN documents of mean chunk reward - CVaR25 of the chunk rewards) to calibration/longdoc_v2.json.
python calibrate_tailgap.py rw_v1_human.jsonl calibration/longdoc_v2.json [--q 0.25] [--pct 0.5]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from gemma_mt.rl import longdoc as LD  # noqa: E402

rows = [json.loads(l) for l in open(sys.argv[1])]
path = sys.argv[2]
q = float(sys.argv[sys.argv.index("--q") + 1]) if "--q" in sys.argv else 0.25
pct = float(sys.argv[sys.argv.index("--pct") + 1]) if "--pct" in sys.argv else 0.5
cal = json.load(open(path))
edges = cal["edges"]
import bisect
gaps = {}
for r in rows:
    ch = r.get("chunks")
    if not ch or len(ch) < 8 or r["status"] != "ok":
        continue
    v = [c[0] for c in ch]; w = [c[1] for c in ch]
    m = sum(a * b for a, b in zip(v, w)) / sum(w)
    gaps.setdefault(bisect.bisect_left(edges, C.BIN_WORDS[r["bin"]]), []).append(m - LD.weighted_tail_mean(v, w, q))
tg = []
for b in range(len(edges) + 1):
    g = sorted(gaps.get(b) or gaps[min(gaps, key=lambda k: abs(k - b))])
    tg.append(round(g[min(len(g) - 1, int(pct * len(g)))], 3))
cal["tail_gap"] = tg
cal.setdefault("meta", {})["tail_gap"] = {"q": q, "pct": pct, "n": {str(k): len(v) for k, v in sorted(gaps.items())}, "source": "human SmolDoc references stitched, reward = qe_ensemble chunk rewards of the phase-2c reward block (reward_run.py)"}
json.dump(cal, open(path, "w"), indent=1)
print(tg, {k: len(v) for k, v in gaps.items()})
