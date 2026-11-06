#!/usr/bin/env python
"""AfriCOMET-STL-1.1 (masakhane/africomet-stl-1.1, reference-based, held out of every reward) per sentence for every output of e02_decode.py json files.
    PYTHONPATH=$E/pylibs_none:$E/gemma_mt_tree $E/venv_comet/bin/python e02_comet.py IN1.json [IN2.json ...] [--cap-gb 20]    -> IN.comet.json = {system: {comet: [per sentence]}}
Scores all 20 languages; the analysis restricts to the languages the harness scores (comet-langs-from)."""
import json, sys
E = "/workspace/eval_385m"; sys.path.insert(0, f"{E}/gemma_mt_tree")
import torch
cap = float(sys.argv[sys.argv.index("--cap-gb") + 1]) if "--cap-gb" in sys.argv else 20.0
torch.cuda.set_per_process_memory_fraction(min(0.95, cap / 79.0))
from gemma_mt.rl.comet import CometScorer, REF_MODEL_ID
args = [a for a in sys.argv[1:] if a != "--cap-gb" and not a.replace(".", "").isdigit()]
ins = args
sc = CometScorer(REF_MODEL_ID, batch_size=32, reference_based=True)
for p in ins:
    d = json.load(open(p)); items = d["items"]; r = {}
    for sysn, hs in d["hyp"].items():
        v = [float(x) for x in sc([i[1] for i in items], [h["hyp"] for h in hs], [i[2] for i in items])]
        r[sysn] = {"comet": v}; print(p, sysn, round(sum(v) / len(v), 4), flush=True)
    json.dump(r, open(p.replace(".json", "") + ".comet.json", "w"))
print("DONE")
