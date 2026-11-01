# SPDX-License-Identifier: AGPL-3.0-or-later
"""Flags-off equivalence: rewards of the study items with the PRISTINE tree (before D-60) vs the patched tree with every longdoc flag at its default.
  PYTHONPATH=<tree> python flags_off_check.py --tree-name pristine --out a.jsonl   /   ... --tree-name patched --out b.jsonl   then   python flags_off_check.py --compare a.jsonl b.jsonl"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--items", default="/workspace/jobs/r8/items_core.jsonl")
ap.add_argument("--out")
ap.add_argument("--n", type=int, default=900)
ap.add_argument("--compare", nargs=2)
ap.add_argument("--calibration", default="/workspace/jobs/r8/gemma_mt_r8/calibration/shaping_flores_dev_v8.json")
ap.add_argument("--config", default="/workspace/jobs/r8/gemma_mt_r8/configs/rl_rtxpro_phase2c.yaml")
a = ap.parse_args()
if a.compare:
    A = [json.loads(l) for l in open(a.compare[0])]
    B = [json.loads(l) for l in open(a.compare[1])]
    bad = [(x, y) for x, y in zip(A, B) if x["reward"] != y["reward"] or x["status"] != y["status"] or x["qe"] != y["qe"]]
    print(len(A), len(B), "items; differing:", len(bad), "max |d reward|", max([abs(x["reward"] - y["reward"]) for x, y in zip(A, B)] or [0]))
    sys.exit(0)
import common as C
import qe_cache
from gemma_mt.eval.longdoc_reward import build_reward, completion_ids, stop_ids_of
from gemma_mt.rl.reward import Sample
import gemma_mt
print("tree:", os.path.dirname(gemma_mt.__file__))
docs = C.core_docs()
items = C.rd(a.items)
import random
random.Random(0).shuffle(items)
items = [it for it in items if it["cond"].startswith(("sf", "nllb", "W:t10", "W:greedy", "C:"))][: a.n]
items.sort(key=lambda r: (r["iso"], r["doc_id"]))
sc = qe_cache.CachedPairScorer(None, "/workspace/jobs/r8/qe_cache")
reward, tok, _ = build_reward(a.config, overrides={"qe_ensemble": True, "lid_enabled": False, "calibration_path": a.calibration, "lid_model_path": None}, scorer=sc,
                              tokenizer_json="/workspace/sft_build/runs/final_1433/tokenizer.json")
stop = stop_ids_of(tok)[0]
out = []
for i in range(0, len(items), 16):
    b = items[i:i + 16]
    res = reward([Sample(completion_ids(tok, r["iso"], r["out"], r.get("finished", True), stop), r["iso"], docs[r["doc_id"]]["text"]) for r in b])
    for r, rw, st, q in zip(b, res.rewards, res.status, res.qe):
        out.append({"doc_id": r["doc_id"], "iso": r["iso"], "cond": r["cond"], "reward": float(rw), "status": st, "qe": q})
C.wr(a.out, out)
print("wrote", len(out))
