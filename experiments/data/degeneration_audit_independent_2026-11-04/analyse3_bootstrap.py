"""E-08 step 5: bootstrap CIs (resampling source sentence ids, shared across the parallel languages; 1,000 resamples) of the macro rates of the reward-family loop flag,
the independent flag, their difference and ratio, per model and benchmark. Needs analyse.py outputs. Writes bootstrap_rates.csv."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0]]
exec(open(os.path.join(HERE, "analyse.py")).read().split("# ------------------------------------------------------------------ per model tables")[0])
rng = np.random.default_rng(1)
rows = []
for m in ["sft", "stageA", "b1best", "mixedbest", "weakfinal"]:
    d = add_flags(F[m], R)
    for b in BENCH:
        g = d[d.benchmark == b]
        A = g.pivot(index="iso", columns="id", values="rw_loop").astype(float)
        B = g.pivot(index="iso", columns="id", values="ind_loop").astype(float)
        a, bb = A.values, B.values
        n = a.shape[1]
        def macro(x, idx):
            s = np.nanmean(x[:, idx], axis=1)
            return float(np.nanmean(s))
        full = (macro(a, np.arange(n)), macro(bb, np.arange(n)))
        bs = []
        for _ in range(1000):
            idx = rng.integers(0, n, n)
            r1, r2 = macro(a, idx), macro(bb, idx)
            bs.append((r1, r2, r2 - r1, r2 / r1 if r1 > 0 else np.nan))
        bs = np.array(bs)
        q = lambda k: (float(np.nanpercentile(bs[:, k], 2.5)), float(np.nanpercentile(bs[:, k], 97.5)))
        rows.append(dict(model=m, benchmark=b, rw_loop=full[0], rw_lo=q(0)[0], rw_hi=q(0)[1], ind_loop=full[1], ind_lo=q(1)[0], ind_hi=q(1)[1],
                         diff=full[1] - full[0], diff_lo=q(2)[0], diff_hi=q(2)[1], ratio=full[1] / full[0] if full[0] > 0 else np.nan, ratio_lo=q(3)[0], ratio_hi=q(3)[1]))
pd.DataFrame(rows).to_csv(os.path.join(HERE, "bootstrap_rates.csv"), index=False)
print(pd.DataFrame(rows).round(4).to_string())
