"""R14 stage 2: sentence-level AfriCOMET-QE (reference-free) of the NLLB translations, for the QE-validated languages. Writes s2/qe_<iso>.jsonl {english, qe}.
usage: s2_qe.py iso1 iso2 ...   (needs the transformers-4 env: PYTHONPATH=pylibs_tf4:pylibs ahead of everything)"""
import json, os, sys
from gemma_mt.rl.comet import CometScorer
S2 = "/workspace/jobs/r14/s2"
sc = CometScorer("masakhane/africomet-qe-stl-1.1", batch_size=64)
for iso in sys.argv[1:]:
    out = f"{S2}/qe_{iso}.jsonl"
    if not os.path.exists(f"{S2}/trans_{iso}.jsonl"): continue
    done = {json.loads(l)["english"] for l in open(out)} if os.path.exists(out) else set()          # incremental: only sentences without a score yet
    rows = [json.loads(l) for l in open(f"{S2}/trans_{iso}.jsonl")]
    rows = [r for r in rows if r["hyp"].strip() and r["english"] not in done]
    if not rows: continue
    s = sc([r["english"] for r in rows], [r["hyp"] for r in rows])
    with open(out, "a") as f:
        for r, v in zip(rows, s): f.write(json.dumps({"english": r["english"], "qe": v}) + "\n")
    print(iso, len(rows), "scored", flush=True)
