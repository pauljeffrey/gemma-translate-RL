import json, numpy as np
g = json.load(open("/workspace/jobs/r12/res/grid.json")); out = {}
for c in ["T0.3", "T0.5", "T0.7", "T1.0", "T1.0_p0.9", "T1.0_mp0.05", "T1.0_mp0.1", "T0.7_mp0.05"]:
    names = [c] + [f"{c}_s{k}" for k in (1, 2, 3, 4)]
    if not all(n in g for n in names): continue
    for grp in ("all", "strong", "medium", "weak"):
        pass
    v = np.array([g[n]["chrf_all"] for n in names]); d = np.array([g[n]["d_chrf_all"] for n in names]); b = np.array([g[n]["bleu_all"] for n in names]); l = np.array([g[n]["loop"] for n in names])
    out[c] = {"seeds_chrf": v.round(2).tolist(), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max()), "mean_d": float(d.mean()), "spbleu_mean": float(b.mean()), "spbleu_sd": float(b.std(ddof=1)),
              "weak_mean": float(np.mean([g[n]["chrf_weak"] for n in names])), "strong_mean": float(np.mean([g[n]["chrf_strong"] for n in names])), "medium_mean": float(np.mean([g[n]["chrf_medium"] for n in names])), "loop_mean": float(l.mean())}
    print(c, out[c]["seeds_chrf"], f"mean {out[c]['mean']:.2f} sd {out[c]['sd']:.2f} d {out[c]['mean_d']:+.2f}")
json.dump(out, open("/workspace/jobs/r12/res/seeds.json", "w"), indent=1)
