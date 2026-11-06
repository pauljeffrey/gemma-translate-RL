#!/usr/bin/env python
"""E-05: SSA-COMET-QE (reference-free, the AfriScience-MT paper's headline), AfriCOMET-STL-1.1 (reference-based, held out of every reward) and plain chrF / chrF++ per language on the
AfriScience-MT dev and test SENTENCE predictions of every stage that has them.   python sent_neural_all.py OUT.json [cap_gb]
Every row of an RL stage is labelled in the report as reward family (ssa-comet-qe shares its encoder family with ssa-comet-mtl, which is in the RL reward)."""
import glob, json, os, sys
E = "/workspace/eval_385m"
sys.path.insert(0, f"{E}/gemma_mt_tree")
import torch
cap = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
torch.cuda.set_per_process_memory_fraction(cap / 79.0)
import pyarrow.parquet as pq, sacrebleu, unicodedata
from gemma_mt.rl.comet import CometScorer, REF_MODEL_ID, SSA_MTL_MODEL_ID
NAMES = [("sft", "m385"), ("stageA", "stageA"), ("b1best", "b1best"), ("mixedbest", "wk_mixedbest"), ("weakfinal", "wk_weakfinal"),
         ("sci_a0", "sci_a0"), ("sci_a0.3", "sci_a0.3"), ("sci_a0.5", "sci_a0.5"), ("sci_a0.7", "sci_a0.7"), ("sci_a1.0", "sci_a1.0"),
         ("sci_rl_best", "sci_rl_best"), ("sci_rl_final", "sci_rl_final"), ("long_rl_best", "long_rl_best"), ("long_rl_final", "long_rl_final")]
snap = glob.glob("/workspace/clean_sft/hf_home/hub/datasets--dsfsi--afriscience_mt/snapshots/*/corpus")[0]
src = {sp: {} for sp in ("dev", "test")}
for sp in src:
    for r in pq.read_table(f"{snap}/{sp}-00000.parquet").to_pylist(): src[sp].setdefault(r["target_lang"], []).append(r["source"])
preds = {}
for name, d in NAMES:
    ok = all(os.path.exists(f"{E}/out_afsci_{d}/predictions_{sp}.json") for sp in ("dev", "test"))
    if ok: preds[name] = {sp: json.load(open(f"{E}/out_afsci_{d}/predictions_{sp}.json"))["sentence"] for sp in ("dev", "test")}
    else: print("skip (no predictions)", name, flush=True)
nfc = lambda x: unicodedata.normalize("NFC", x)
res = {n: {sp: {} for sp in ("dev", "test")} for n in preds}
for n in preds:
    for sp in ("dev", "test"):
        for iso, rows in preds[n][sp].items():
            h = [nfc(r["hyp"]) for r in rows]; rf = [nfc(r["ref"]) for r in rows]
            res[n][sp][iso] = {"n": len(rows), "chrf": sacrebleu.corpus_chrf(h, [rf], word_order=0).score, "chrfpp": sacrebleu.corpus_chrf(h, [rf], word_order=2).score}
for name, mid, bs, ref in (("ssa_qe", "McGill-NLP/ssa-comet-qe", 32, False), ("africomet_stl", REF_MODEL_ID, 32, True), ("ssa_mtl_ref", SSA_MTL_MODEL_ID, 16, True)):
    sc = CometScorer(mid, batch_size=bs, reference_based=ref)
    for n in preds:
        for sp in ("dev", "test"):
            for iso, rows in preds[n][sp].items():
                s = src[sp][iso]; h = [r["hyp"] for r in rows]; rf = [r["ref"] for r in rows]
                v = [float(x) for x in (sc(s, h, rf) if ref else sc(s, h))]
                res[n][sp][iso][name] = sum(v) / len(v)
        print(name, n, flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    del sc; import gc; gc.collect(); torch.cuda.empty_cache()
print("DONE", flush=True)
