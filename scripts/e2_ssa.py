"""E2: SSA-COMET-MTL (McGill-NLP/ssa-comet-mtl), reference-based (mean of the three views src/mt/ref as in the model card), per sentence on the no_pair subset of FLORES+ devtest,
for the languages of sets afrique21 + tpsy19. usage: e2_ssa.py NAME [NAME ...] -> scores/NAME.ssa.pkl {iso: {id: score}}.  Scorer = copy of gemma_mt/rl/comet.py (D-55 version) in lib/."""
import sys, os, pickle, glob, time, json
sys.path[:0] = ["/workspace/jobs/e2/lib", "/workspace/sft_build/pylibs_tf4", "/workspace/sft_build/pylibs"]
from gemma_mt.rl.comet import CometScorer
sets = json.load(open("/workspace/jobs/e2/sets.json"))
langs = set(sets["afrique21"]) | set(sets["tpsy19"])
ck = glob.glob("/workspace/sft_build/hf/hub/models--McGill-NLP--ssa-comet-mtl/snapshots/*/checkpoints/model.ckpt")[0]
sc = None
for name in sys.argv[1:]:
    S = pickle.load(open(f"/workspace/jobs/e2/scores/{name}.pkl", "rb"))
    outp = f"/workspace/jobs/e2/scores/{name}.ssa.pkl"
    R = pickle.load(open(outp, "rb")) if os.path.exists(outp) else {}
    for iso, d in sorted(S.items()):
        if iso not in langs or iso in R: continue
        idx = [i for i, p in enumerate(d["pair_seen"]) if p is False]
        if sc is None: sc = CometScorer(ck, batch_size=16, reference_based=True, max_batch_chars=16 * 700)
        for attempt in range(6):
            try:
                s = sc([d["src"][i] for i in idx], [d["hyp"][i] for i in idx], [d["ref"][i] for i in idx]); break
            except RuntimeError as e:
                if "out of memory" not in str(e) or attempt == 5: raise
                import torch; torch.cuda.empty_cache(); time.sleep(60); sc.batch_size = max(2, sc.batch_size // 2); sc.max_batch_chars = sc.batch_size * 700
        R[iso] = {d["ids"][i]: float(x) for i, x in zip(idx, s)}
        print(name, iso, len(idx), round(sum(s) / len(s), 4), flush=True)
        pickle.dump(R, open(outp, "wb"))
print("done", sys.argv[1:])
