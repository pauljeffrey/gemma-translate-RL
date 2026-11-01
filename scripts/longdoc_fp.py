import json, collections, numpy as np, sys
from gemma_mt.rl import anchors_v2 as V, textstats as T
cal = json.load(open(sys.argv[1]))
A = V.Anchors2(cal, v2_detectors=("func", "pad", "copy"), bigram_path="/workspace/jobs/r9/assets/calibration/anchors_v2_bigrams.npz")
edges = [(0, 100), (100, 200), (200, 400), (400, 800), (800, 10**9)]
agg = collections.defaultdict(list)
for l in open("/workspace/jobs/b2/human_units.jsonl"):
    r = json.loads(l)
    if not (r["set"] == "smol_doc" or r["set"].endswith("_doc.jsonl")): continue
    iso = r["iso"]; ow = T.out_tokens(r["ref"]); sw = T.out_tokens(r["src"])
    n = len(ow)
    b = [i for i, (lo, hi) in enumerate(edges) if lo <= n < hi][0]
    f = A.func_deficit(iso, sw, ow); p = A.pad_deficit(iso, ow)
    g = A.pad_features(iso, ow)
    if f is not None: agg[(b, "func")].append(f)
    if p is not None:
        agg[(b, "pad")].append(p)
        if g: agg[(b, "ttr")].append(g["ttr"])
for b, (lo, hi) in enumerate(edges):
    row = []
    for k in ("func", "pad"):
        v = np.asarray(agg[(b, k)]); row.append(f"{k}: n={len(v)} >0 {np.mean(v>0):.3f} >=0.9 {np.mean(v>=0.9):.3f}" if len(v) else f"{k}: -")
    t = agg[(b, "ttr")]
    print(f"words {lo}-{hi}: " + " | ".join(row) + (f" | mean ttr {np.mean(t):.3f}" if t else ""))
