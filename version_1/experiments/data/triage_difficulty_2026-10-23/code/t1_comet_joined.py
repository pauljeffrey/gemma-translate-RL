"""T1 spot check: AfriCOMET on the JOINED text of FLORES+ chunks (k=3,5) vs the mean of the per-sentence AfriCOMET scores.
languages hau yor swh zul amh ibo; models rl, nllb3b3; only chunks whose sentences are all no_pair. Output /workspace/jobs/t1/out/comet_joined_flores.csv"""
import sys, os, gzip, json, glob, pickle, unicodedata, time
sys.path[:0] = ["/workspace/sft_build/gemma_mt_2c", "/workspace/sft_build/pylibs_tf4", "/workspace/sft_build/pylibs"]
import torch
torch.cuda.set_per_process_memory_fraction(3.5 / 95.0)
from gemma_mt.rl.comet import CometScorer
nfc = lambda s: unicodedata.normalize("NFC", s)
LANGS = "hau yor swh zul amh ibo".split(); MODELS = ["rl", "nllb3b3"]
units = [json.loads(l) for l in gzip.open("/workspace/jobs/t1/units.jsonl.gz", "rt")]
units = [u for u in units if u["b"] == "flores_plus" and u["k"] in (3, 5) and u["iso"] in LANGS]
ck = glob.glob("/workspace/sft_build/hf/hub/models--masakhane--africomet-stl-1.1/snapshots/*/checkpoints/model.ckpt")[0]
sc = CometScorer(ck, batch_size=8, reference_based=True)
S = {m: pickle.load(open(f"/workspace/jobs/e2/scores/{m}.pkl", "rb")) for m in MODELS}
rows = []
t0 = time.time()
for iso in LANGS:
    d0 = S["rl"][iso]; ps = dict(zip(d0["ids"], d0["pair_seen"])); ref = dict(zip(d0["ids"], d0["ref"])); src = dict(zip(d0["ids"], d0["src"]))
    for m in MODELS:
        hyp = dict(zip(S[m][iso]["ids"], S[m][iso]["hyp"]))
        U = [u for u in units if u["iso"] == iso and not any(ps[i] for i in u["ids"])]
        s = sc([" ".join(src[i] for i in u["ids"]) for u in U], [nfc(" ".join(hyp[i] for i in u["ids"])) for u in U], [nfc(" ".join(ref[i] for i in u["ids"])) for u in U])
        rows += [{"iso": iso, "k": u["k"], "uid": u["uid"], "model": m, "joined_comet": float(x)} for u, x in zip(U, s)]
    print(iso, len(rows), round(time.time() - t0), "s", flush=True)
import pandas as pd
pd.DataFrame(rows).to_csv("/workspace/jobs/t1/out/comet_joined_flores.csv", index=False)
