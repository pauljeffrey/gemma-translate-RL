import sl, numpy as np, json
ref = np.array([sl._rep3(r) for r in sl.REFS]); d2 = np.mean([sl._dist2(r) for r in sl.REFS])
g = sl.get_row("greedy"); hy = np.array([sl._rep3(h) for h in g["hyp"]])
print("ref rep3 share", ref.mean(), "ref dist2", d2, "greedy hyp rep3 (any)", hy.mean(), "greedy rep3 not in ref", g["rep3"].mean())
by = {i: (float(ref[sl.IDX[i]].mean()), float(g["rep3"][sl.IDX[i]].mean())) for i in sl.ISOS}
print(by)
json.dump({"ref_rep3": float(ref.mean()), "ref_dist2": float(d2), "greedy_rep3_any": float(hy.mean())}, open(sl.R + "/res/refrep.json", "w"))
