"""E2: AfriCOMET reference-based (masakhane/africomet-stl-1.1) per sentence for the no_pair subset of FLORES+ devtest, languages the COMET card lists (africomet_code set).
usage: e2_comet.py NAME [NAME ...]  -> scores/NAME.comet.pkl {iso: {id: score}}.  Resumable per language; retries on OOM."""
import sys, os, pickle, glob, time
sys.path[:0] = ["/workspace/sft_build/gemma_mt_2c", "/workspace/sft_build/pylibs_tf4", "/workspace/sft_build/pylibs"]
from gemma_mt.languages import LANGUAGES
from gemma_mt.rl.comet import CometScorer
ck = glob.glob("/workspace/sft_build/hf/hub/models--masakhane--africomet-stl-1.1/snapshots/*/checkpoints/model.ckpt")[0]
sc = None
for name in sys.argv[1:]:
    S = pickle.load(open(f"/workspace/jobs/e2/scores/{name}.pkl", "rb"))
    outp = f"/workspace/jobs/e2/scores/{name}.comet.pkl"
    R = pickle.load(open(outp, "rb")) if os.path.exists(outp) else {}
    for iso, d in sorted(S.items()):
        if LANGUAGES[iso].africomet_code is None or iso in R: continue
        idx = [i for i, p in enumerate(d["pair_seen"]) if p is False]
        if sc is None: sc = CometScorer(ck, batch_size=32, reference_based=True)
        for attempt in range(5):
            try:
                s = sc([d["src"][i] for i in idx], [d["hyp"][i] for i in idx], [d["ref"][i] for i in idx]); break
            except RuntimeError as e:
                if "out of memory" not in str(e) or attempt == 4: raise
                import torch; torch.cuda.empty_cache(); time.sleep(60); sc.batch_size = max(4, sc.batch_size // 2)
        R[iso] = {d["ids"][i]: float(x) for i, x in zip(idx, s)}
        print(name, iso, len(idx), round(sum(s) / len(s), 4), flush=True)
        pickle.dump(R, open(outp, "wb"))
print("done", sys.argv[1:])
