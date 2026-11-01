"""T1: long-document sets (lc1): English document texts for classification + per-document chrF++ vs HUMAN references (sets human = SmolDoc, sci = AfriScience-MT dev/test)
for every model/mode that was scored. Output: /workspace/jobs/t1/ld_docs.jsonl (ekey=doc_id, text) and /workspace/jobs/t1/out/longdoc_items.csv.gz"""
import sys, os, json, pickle, glob
sys.path[:0] = ["/workspace/jobs/lc1", "/workspace/sft_build/pylibs_tf4", "/workspace/sft_build/pylibs", "/workspace/jobs/e1/code2c"]
import numpy as np, pandas as pd
from sacrebleu.metrics import CHRF
chrf = CHRF(word_order=2)
docs = [json.loads(l) for l in open("/workspace/jobs/lc1/work/docs.jsonl")]
with open("/workspace/jobs/t1/ld_docs.jsonl", "w") as f:
    for d in docs: f.write(json.dumps({"ekey": d["doc_id"], "text": d["text"]}, ensure_ascii=False) + "\n")
rows = []
for p in sorted(glob.glob("/workspace/jobs/lc1/sc/*.pkl")):
    name = os.path.basename(p)[:-4]; model, mode = name.split("__")
    for r in pickle.load(open(p, "rb")):
        if r["set"] == "core": continue
        st = r["stats"].get("human")
        if st is None: continue
        sc = chrf._compute_score_from_stats([float(x) for x in st[0]]).score
        rows.append({"model": model, "mode": mode, "iid": r["iid"], "set": r["set"], "bin": r["bin"], "iso": r["iso"], "doc_id": r["doc_id"], "src_tokens": r["src_tokens"],
                     "chrf": sc, "finished": r.get("finished"), "loop": r.get("loop"), "length_ratio": r.get("length_ratio"), "cov_sents": r.get("cov_sents"),
                     "lid_doc_ok": r.get("lid_doc_ok"), "lid_sent": r.get("lid_sent")})
df = pd.DataFrame(rows); df.to_csv("/workspace/jobs/t1/out/longdoc_items.csv.gz", index=False)
print(df.groupby(["model", "mode"]).size())
