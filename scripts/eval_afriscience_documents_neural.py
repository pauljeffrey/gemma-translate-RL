#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference-based neural scores of the document-level AfriScience-MT predictions (scripts/eval_afriscience_documents.py): AfriCOMET-STL (masakhane/africomet-stl-1.1) and SSA-COMET-MTL (McGill-NLP/ssa-comet-mtl,
src + mt + ref, mean of the three views of rl/comet.CometScorer). Two variants per mode:
  concat   the whole document (src, hyp, ref concatenated) in one call: the COMET encoders silently truncate at their window (512 tokens), so only the START of a ~1,000-token document is judged (the PAPER's
           handling of documents longer than the encoder window is not stated in the notes we have; this is the straightforward reading of 'metrics on the concatenated document translations')
  window   (chunked mode only, where the hypothesis is sentence-aligned) windows of --window consecutive sentences scored with their own source / reference; the document score is the mean of its windows
    python scripts/eval_afriscience_documents_neural.py --pred A.json B.json --names a b --out neural.json [--cap-gb 5.5]
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser(); ap.add_argument("--pred", nargs="+", required=True); ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--window", type=int, default=4); ap.add_argument("--cap-gb", type=float, default=5.5); ap.add_argument("--tree", default="/workspace/eval_385m/gemma_mt_tree"); a = ap.parse_args()
sys.path.insert(0, a.tree)
import torch
if torch.cuda.is_available() and a.cap_gb > 0:
    torch.cuda.set_per_process_memory_fraction(a.cap_gb / (torch.cuda.get_device_properties(0).total_memory / 2 ** 30))
from gemma_mt.rl.comet import CometScorer, REF_MODEL_ID, SSA_MTL_MODEL_ID
preds = {n: json.load(open(p)) for n, p in zip(a.names, a.pred)}
res = json.load(open(a.out)) if os.path.exists(a.out) else {}
for metric, mid, bs in (("africomet", REF_MODEL_ID, 8), ("ssa_comet_mtl", SSA_MTL_MODEL_ID, 4)):
    sc = CometScorer(mid, batch_size=bs, reference_based=True)
    for n, P in preds.items():
        for mode in ("whole", "chunked"):
            for iso, rows in P[mode].items():
                key = f"{n}/{mode}/{iso}"; e = res.setdefault(key, {})
                if f"{metric}_concat" not in e:
                    e[f"{metric}_concat"] = sum(float(x) for x in sc([r["src"] for r in rows], [r["hyp"] for r in rows], [r["ref"] for r in rows])) / len(rows)
                if mode == "chunked" and f"{metric}_window" not in e:
                    S, H, R, owner = [], [], [], []
                    for di, r in enumerate(rows):
                        for i in range(0, len(r["src_sents"]), a.window):
                            S.append(" ".join(r["src_sents"][i:i + a.window])); H.append(" ".join(r["sent_hyps"][i:i + a.window])); R.append(" ".join(r["ref_sents"][i:i + a.window])); owner.append(di)
                    v = [float(x) for x in sc(S, H, R)]
                    per_doc = {}
                    for o, x in zip(owner, v): per_doc.setdefault(o, []).append(x)
                    e[f"{metric}_window"] = sum(sum(x) / len(x) for x in per_doc.values()) / len(per_doc)
                e["n_docs"] = len(rows)
            json.dump(res, open(a.out, "w"), indent=1); print(metric, n, mode, flush=True)
    del sc; torch.cuda.empty_cache()
print("NEURAL_DOCS_DONE", flush=True)
