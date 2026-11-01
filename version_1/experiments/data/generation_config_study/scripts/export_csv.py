"""R12: export the sentence-level results to CSV (grid, pools, seeds, per-language, QE)."""
import csv, json, os, re
import numpy as np
R = "/workspace/jobs/r12"; O = R + "/export"; os.makedirs(O, exist_ok=True)
g = json.load(open(R + "/res/grid.json"))
keys = ["name", "engine", "chrf_all", "d_chrf_all", "lo_chrf_all", "hi_chrf_all", "bleu_all", "d_bleu_all", "lo_bleu_all", "hi_bleu_all"] + [f"{m}_{x}" for x in ("strong", "medium", "weak") for m in ("chrf", "d_chrf", "lo_chrf", "hi_chrf")] + \
       ["finished", "loop", "loop_weak", "rep3", "dist2", "empty", "lid", "num_kept", "name_kept", "len_ratio", "mean_tokens", "seconds", "tokens"]
def parse(n):
    n0 = re.sub(r"^hf_", "", n); n0 = re.sub(r"_n16$|_s\d$", "", n0)
    d = {"T": "", "top_p": "", "top_k": "", "min_p": "", "extra": ""}
    for i, part in enumerate(n0.split("_")):
        if i == 0 and part.startswith("T"): d["T"] = part[1:]
        elif part.startswith("mp"): d["min_p"] = part[2:]
        elif part.startswith("p"): d["top_p"] = part[1:]
        elif part.startswith("k"): d["top_k"] = part[1:]
        else: d["extra"] += part
    return d
with open(O + "/grid_single_sample.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(keys + ["T", "top_p", "top_k", "min_p", "extra"])
    for n, v in sorted(g.items()):
        p = parse(n); w.writerow([n] + [("%.5g" % v[k] if isinstance(v.get(k), float) else v.get(k)) for k in keys[1:]] + [p[k] for k in ("T", "top_p", "top_k", "min_p", "extra")])
isos = list(next(iter(g.values()))["per_lang_chrf"])
with open(O + "/per_language_chrf.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["config"] + isos)
    for n, v in sorted(g.items()): w.writerow([n] + ["%.2f" % v["per_lang_chrf"][i] for i in isos])
if os.path.exists(R + "/res/pools.json"):
    P = json.load(open(R + "/res/pools.json"))
    pk = ["name", "pool", "method", "n", "chrf_all", "d_chrf_all", "lo_chrf_all", "hi_chrf_all", "bleu_all", "d_bleu_all", "d_chrf_strong", "d_chrf_medium", "d_chrf_weak", "loop", "finished", "lid", "num_kept", "tok_x_greedy"]
    with open(O + "/pools_mbr_oracle.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(pk)
        for n, v in sorted(P.items()): w.writerow([("%.5g" % v[k] if isinstance(v.get(k), float) else v.get(k)) for k in pk])
if os.path.exists(R + "/res/qe.json"):
    Q = json.load(open(R + "/res/qe.json"))
    with open(O + "/africomet.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "qe", "d_qe", "qe_lo", "qe_hi", "ref", "d_ref", "ref_lo", "ref_hi"])
        for n, v in sorted(Q.items()): w.writerow([n, "%.4f" % v["qe"]["macro"], "%.4f" % v["qe"]["d"], "%.4f" % v["qe"]["d_lo"], "%.4f" % v["qe"]["d_hi"], "%.4f" % v["ref"]["macro"], "%.4f" % v["ref"]["d"], "%.4f" % v["ref"]["d_lo"], "%.4f" % v["ref"]["d_hi"]])
print("exported", os.listdir(O))
