"""Add GlotLID top-1/top-3 to a predictions file (hyp and reference). python lid_run.py IN OUT MODEL_BIN"""
import sys, json
sys.path.insert(0, "/workspace/rlcheck")
from gemma_mt.rl.lid import GlotLIDClassifier
inp, out, mp = sys.argv[1:4]
lid = GlotLIDClassifier(path=mp)
rows = [json.loads(l) for l in open(inp, encoding="utf-8")]
hyp = [r.get("hyp") or "" for r in rows]
top = lid.top(hyp, 3)
reft = lid.top([r.get("reference") or "" for r in rows], 3) if any(r.get("reference") for r in rows) else [None] * len(rows)
with open(out, "w", encoding="utf-8") as f:
    for r, t, rt in zip(rows, top, reft):
        r["lid_hyp"] = t
        if rt is not None and r.get("reference"): r["lid_ref"] = rt
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("ok", len(rows))
