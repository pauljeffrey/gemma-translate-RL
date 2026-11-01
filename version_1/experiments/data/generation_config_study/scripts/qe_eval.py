"""R12: AfriCOMET (reference) and AfriCOMET-QE of the hypotheses of any scored configuration (cache/row_*.pkl) on the 18 validated languages; paired bootstrap vs greedy.
usage: qe_eval.py name [name ...] ; results res/qe.json (incremental). Scores are cached in comet_cache (seeded with R5's cache)."""
import json, os, sys
import numpy as np
import sl
from gemma_mt.eval.runtime import CometPair
from gemma_mt.languages import QE_VALIDATED
isos = [i for i in sl.ISOS if i in QE_VALIDATED]
ix = [k for k, r in enumerate(sl.rows) if r["iso"] in isos]
srcs = [sl.rows[k]["english"] for k in ix]; refs = [sl.rows[k]["reference"] for k in ix]
out_p = sl.R + "/res/qe.json"; res = json.load(open(out_p)) if os.path.exists(out_p) else {}
names = [n for n in sys.argv[1:]]
if "greedy" not in names and "greedy" not in res: names = ["greedy"] + names
pair = CometPair(python="/workspace/venv_gemma/bin/python", pythonpath="/workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs", batch_size=64, cache_dir=sl.R + "/comet_cache")
lang = np.array([sl.rows[k]["iso"] for k in ix]); li = {i: np.where(lang == i)[0] for i in isos}
rng = np.random.default_rng(0); B = 1000
samp = {i: li[i][rng.integers(0, len(li[i]), size=(B, len(li[i])))] for i in isos}
SC = {}
def macro_boot(arr): return np.mean([arr[samp[i]].mean(1) for i in isos], axis=0)
store = sl.R + "/res/qe_scores.npz"; saved = dict(np.load(store)) if os.path.exists(store) else {}
for n in names:
    if f"{n}__qe" in saved and f"{n}__ref" in saved: SC[n] = {"qe": saved[f"{n}__qe"], "ref": saved[f"{n}__ref"]}; continue
    H = sl.find_row(n)["hyp"]; hs = [H[k] for k in ix]
    SC[n] = {"qe": np.array(pair.qe(srcs, hs)), "ref": np.array(pair.reference(srcs, hs, refs))}
    saved[f"{n}__qe"], saved[f"{n}__ref"] = SC[n]["qe"], SC[n]["ref"]; np.savez(store, **saved)
    print(n, round(float(SC[n]["qe"].mean()), 4), round(float(SC[n]["ref"].mean()), 4), flush=True)
pair.close()
if "greedy" not in SC: SC["greedy"] = {"qe": saved["greedy__qe"], "ref": saved["greedy__ref"]}
for m in ("qe", "ref"):
    base = macro_boot(SC["greedy"][m])
    for n in SC:
        b = macro_boot(SC[n][m]); d = b - base
        res.setdefault(n, {})[m] = {"macro": float(np.mean([SC[n][m][li[i]].mean() for i in isos])), "d": float(d.mean()), "d_lo": float(np.percentile(d, 2.5)), "d_hi": float(np.percentile(d, 97.5))}
json.dump(res, open(out_p, "w")); print("done")
