"""E1 robustness: does the RL preference survive when sentences where either model loops / does not finish are removed (no_pair subset)?"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/workspace/jobs/e1")
import pairwise_stats as S
df = S.build_frame("/workspace/jobs/e1", "sft", "rl", "neural")
df = df[df["pair_seen"] == False]
rows = []
def summ(d, tag):
    for k, thr in (("chrf", 1.0), ("chrf_sub", 1.0), ("afri_ref", .01), ("ssa_ref", .01), ("afri_qe", .01), ("ssa_qe", .01)):
        x = d[[k + "_a", k + "_b"]].dropna()
        dd = (x[k + "_b"] - x[k + "_a"]).to_numpy()
        w, l = (dd > thr).sum(), (dd < -thr).sum()
        g = d.loc[x.index].assign(dd=dd).groupby("iso")["dd"].mean()
        rows.append(dict(subset=tag, metric=k, n=len(x), delta_micro=dd.mean(), delta_macro_lang=g.mean(), win=w / len(x), tie=1 - (w + l) / len(x), loss=l / len(x), pref=w / (w + l)))
summ(df, "all rows")
ok = (df["loop_a"] == 0) & (df["loop_b"] == 0) & (df["fin_a"] == 1) & (df["fin_b"] == 1)
summ(df[ok], "neither model loops / unfinished")
ok2 = ok & (df["lratio_a"].between(0.5, 2.0)) & (df["lratio_b"].between(0.5, 2.0))
summ(df[ok2], "neither loops, both length ratio in [0.5, 2]")
r = pd.DataFrame(rows)
print(r.round(4).to_string())
print("share of sentences kept:", ok.mean(), ok2.mean())
r.to_csv("/workspace/jobs/e1/summary/robustness_no_loop.csv", index=False, float_format="%.5g")
