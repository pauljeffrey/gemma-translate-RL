"""a1 decoding study, step 4: AfriCOMET (ref) and AfriCOMET-QE on the AfriCOMET-validated languages of the study, per config; paired bootstrap vs greedy."""
import json, sys
import numpy as np
sys.path.insert(0, "/workspace/sft_build/pylibs")
from gemma_mt.eval.runtime import CometPair
from gemma_mt.languages import QE_VALIDATED
D = "/workspace/jobs/a1/study"
S = json.load(open(D + "/study_set.json")); H = json.load(open(D + "/hyps.json"))
rows = S["rows"]
isos = [i for i in dict.fromkeys(r["iso"] for r in rows) if i in QE_VALIDATED]
ix = [k for k, r in enumerate(rows) if r["iso"] in isos]
pair = CometPair(python="/workspace/venv_gemma/bin/python", pythonpath="/workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs", batch_size=64, cache_dir=D + "/comet_cache")
srcs = [rows[k]["english"] for k in ix]; refs = [rows[k]["reference"] for k in ix]
sc = {}
for c, hyps in H.items():
    hs = [hyps[k] for k in ix]
    sc[c] = {"qe": np.array(pair.qe(srcs, hs)), "ref": np.array(pair.reference(srcs, hs, refs))}
    print(c, round(float(sc[c]["qe"].mean()), 4), round(float(sc[c]["ref"].mean()), 4), flush=True)
pair.close()
lang = np.array([rows[k]["iso"] for k in ix])
li = {i: np.where(lang == i)[0] for i in isos}
rng = np.random.default_rng(0); B = 1000
samp = {i: li[i][rng.integers(0, len(li[i]), size=(B, len(li[i])))] for i in isos}
out = {"isos": isos, "n": len(ix), "configs": {}}
def macro_boot(arr):   # (B,) macro over languages of the sentence mean
    return np.mean([arr[samp[i]].mean(1) for i in isos], axis=0)
for m in ("qe", "ref"):
    base = macro_boot(sc["greedy"][m])
    for c in H:
        b = macro_boot(sc[c][m]); d = b - base
        pl = {i: float(sc[c][m][li[i]].mean()) for i in isos}
        out["configs"].setdefault(c, {})[m] = {"macro": float(np.mean(list(pl.values()))), "lo": float(np.percentile(b, 2.5)), "hi": float(np.percentile(b, 97.5)),
            "d": float(d.mean()), "d_lo": float(np.percentile(d, 2.5)), "d_hi": float(np.percentile(d, 97.5)), "per_lang": pl}
json.dump(out, open(D + "/qe_results.json", "w"))
print("done")
