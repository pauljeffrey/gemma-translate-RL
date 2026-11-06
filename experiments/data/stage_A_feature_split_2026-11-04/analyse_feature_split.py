"""E-11a: stage A, steps 1-40 without the four features, steps 41-400 with them.
Reads research_archive_2026-11-03/runs_385m/A/{metrics.jsonl,kl_guard_events.jsonl,group_stats.jsonl}.
Writes slopes.csv, its.csv, per10.csv, summary.json, fig_feature_split.png next to this script.
Run:  python analyse_feature_split.py   (repo root = 3 levels up)
"""
import json, os, sys, numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RUN = os.path.join(ROOT, "research_archive_2026-11-03", "runs_385m", "A")
rows = [json.loads(l) for l in open(os.path.join(RUN, "metrics.jsonl"))]
tr = pd.DataFrame([r for r in rows if "kl" in r])
tr = tr.set_index("step").sort_index()
METRICS = {
    "kl": "KL (k3) to reference",
    "reward": "training reward (all members)",
    "greedy_reward_mean": "greedy-member reward",
    "finished_rate": "finished rate (sampled)",
    "greedy_finished_rate": "finished rate (greedy member)",
    "greedy_loop_rate": "loop rate (greedy member)",
    "hard_fail_frac": "hard-fail share",
    "grad_norm": "gradient norm",
    "completion_len": "completion length (tokens)",
}
SPLIT = 41
rng = np.random.default_rng(0)


def ols_slope(x, y):
    X = np.column_stack([np.ones_like(x), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b[1], b[0]


def block_boot_slope(x, y, block=5, B=2000):
    """moving-block bootstrap of (x,y) pairs is wrong for trends; resample residual blocks."""
    s, a = ols_slope(x, y); res = y - (a + s * x); n = len(y)
    nb = int(np.ceil(n / block)); out = []
    for _ in range(B):
        st = rng.integers(0, n - block + 1, nb)
        r = np.concatenate([res[i:i + block] for i in st])[:n]
        out.append(ols_slope(x, a + s * x + r)[0])
    return np.percentile(out, [2.5, 97.5])


def ar1(res):
    r = res - res.mean(); return float((r[1:] * r[:-1]).sum() / (r * r).sum())


SEGS = [("pre 2-40", 2, 40), ("pre 11-40 (after LR warm-up)", 11, 40), ("post 41-100", 41, 100),
        ("post 41-225 (before KL cuts)", 41, 225), ("post 101-225", 101, 225),
        ("post 248-400 (after 3 KL cuts, LR 2.5e-7)", 248, 400), ("all 2-400", 2, 400)]
srows = []
for m in METRICS:
    for name, a, b in SEGS:
        d = tr.loc[a:b, m].dropna(); x = d.index.values.astype(float); y = d.values
        s, c = ols_slope(x, y); lo, hi = block_boot_slope(x, y)
        srows.append(dict(metric=m, segment=name, n=len(d), slope_per_step=s, ci_lo=lo, ci_hi=hi,
                          mean=y.mean(), ar1_resid=ar1(y - (c + s * x))))
slopes = pd.DataFrame(srows); slopes.to_csv(os.path.join(HERE, "slopes.csv"), index=False)

# Interrupted time series on steps 2..100 and 2..225:
#   y = a + b t + c [t>=41] + d (t-41)[t>=41],  HAC (Newey-West, 5 lags) and residual-block bootstrap CI
its = []
for m in METRICS:
    for hi_step in (100, 225):
        d = tr.loc[2:hi_step, m].dropna(); t = d.index.values.astype(float); y = d.values
        post = (t >= SPLIT).astype(float)
        X = np.column_stack([np.ones_like(t), t, post, (t - SPLIT) * post])
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        beta = fit.params; res = y - X @ beta; n = len(y); nb = int(np.ceil(n / 5)); bs = []
        for _ in range(2000):
            st = rng.integers(0, n - 5 + 1, nb)
            r = np.concatenate([res[i:i + 5] for i in st])[:n]
            bs.append(np.linalg.lstsq(X, X @ beta + r, rcond=None)[0])
        bs = np.array(bs)
        its.append(dict(metric=m, window=f"2-{hi_step}", pre_slope=beta[1], level_shift_at_41=beta[2],
                        level_shift_ci_lo=np.percentile(bs[:, 2], 2.5), level_shift_ci_hi=np.percentile(bs[:, 2], 97.5),
                        level_shift_p_hac=fit.pvalues[2], slope_change=beta[3],
                        slope_change_ci_lo=np.percentile(bs[:, 3], 2.5), slope_change_ci_hi=np.percentile(bs[:, 3], 97.5),
                        slope_change_p_hac=fit.pvalues[3], ar1_resid=ar1(res)))
its = pd.DataFrame(its); its.to_csv(os.path.join(HERE, "its.csv"), index=False)

# per-10-step means / medians
per10 = []
for k in range(0, 40):
    lo, hi = k * 10 + 1, k * 10 + 10
    d = tr.loc[lo:hi]
    per10.append(dict(steps=f"{lo}-{hi}", **{m + "_median": d[m].median() for m in METRICS},
                      **{m + "_mean": d[m].mean() for m in METRICS}, lr_end=d["lr"].iloc[-1]))
per10 = pd.DataFrame(per10); per10.to_csv(os.path.join(HERE, "per10.csv"), index=False)

# group-std / guide stats only exist from step 41 (group_stats.jsonl); report presence.
gs_steps = set()
for l in open(os.path.join(RUN, "group_stats.jsonl")):
    gs_steps.add(json.loads(l)["step"])
bench = [(r["step"], r["bench/chrf_macro"], r["bench/finished_rate"]) for r in rows if "bench/chrf_macro" in r]
guide_cols = [c for c in tr.columns if c.startswith("guide/")]
ref_cols = [c for c in tr.columns if c.startswith("reward/ref")]
g = {c: float(tr.loc[41:100, c].median()) for c in ["guide/adv_share", "guide/reward", "guide/sampled_reward", "guide/better_frac", "guide/best_frac", "guide/overhead_frac"] if c in tr}
summary = dict(
    n_train_rows=int(len(tr)), first_step=int(tr.index.min()), last_step=int(tr.index.max()),
    group_stats_first_step=int(min(gs_steps)), group_stats_last_step=int(max(gs_steps)),
    guide_columns_first_step=int(tr[guide_cols[0]].dropna().index.min()) if guide_cols else None,
    ref_columns_present_pre41=bool(tr.loc[:40, ref_cols].notna().any().any()) if ref_cols else None,
    lr_steps=[(int(s), float(tr.loc[s, "lr"])) for s in (1, 2, 5, 10, 11, 40, 41, 225, 226, 236, 247, 248, 400)],
    bench_dev=bench, guide_medians_41_100=g,
    kl_per10_median={r.steps: r.kl_median for r in per10.itertuples()},
)
json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)


# --- extra: local window comparison and counterfactual KL extrapolation -------------------------------
loc = []
for m in METRICS:
    a = tr.loc[31:40, m]; b = tr.loc[41:50, m]
    loc.append(dict(metric=m, mean_31_40=a.mean(), mean_41_50=b.mean(), diff=b.mean() - a.mean(),
                    sd_31_40=a.std(), sd_41_50=b.std()))
pd.DataFrame(loc).to_csv(os.path.join(HERE, "local_window_31_40_vs_41_50.csv"), index=False)

# KL as a function of cumulative learning rate (LR warm-up 1-10, then 2e-6 constant until step 225)
cum = tr["lr"].cumsum()
cf = {}
for lo_fit in (2, 11):
    d = tr.loc[lo_fit:40]
    # power law KL = a * cumLR^p fitted in log space on the pre-feature steps
    p, la = np.polyfit(np.log(cum.loc[lo_fit:40].values), np.log(d["kl"].values), 1)
    pred = np.exp(la) * cum.loc[41:100].values ** p
    # also KL = a * t^p
    p2, la2 = np.polyfit(np.log(d.index.values.astype(float)), np.log(d["kl"].values), 1)
    pred2 = np.exp(la2) * np.arange(41, 101, dtype=float) ** p2
    cf[f"fit_{lo_fit}_40"] = dict(
        power_in_cumLR=float(p), power_in_step=float(p2),
        predicted_kl_step_100_cumLR=float(pred[-1]), predicted_kl_step_100_step=float(pred2[-1]),
        observed_kl_step_100=float(tr.loc[100, "kl"]), observed_median_kl_96_100=float(tr.loc[96:100, "kl"].median()),
        pred_slope_41_100_cumLR=float(np.polyfit(np.arange(41, 101), pred, 1)[0]),
        pred_slope_41_100_step=float(np.polyfit(np.arange(41, 101), pred2, 1)[0]),
        observed_slope_41_100=float(ols_slope(np.arange(41, 101, dtype=float), tr.loc[41:100, "kl"].values)[0]))
json.dump(cf, open(os.path.join(HERE, "kl_counterfactual.json"), "w"), indent=1)

# KL slope in consecutive 15-step windows (is the acceleration already present before step 41?)
win = []
for lo in range(11, 131, 15):
    hi = lo + 14
    d = tr.loc[lo:hi]; x = d.index.values.astype(float)
    row = dict(window=f"{lo}-{hi}")
    for m in ("kl", "reward", "greedy_loop_rate"):
        sl, c = ols_slope(x, d[m].values); row[m + "_slope"] = sl
    win.append(row)
pd.DataFrame(win).to_csv(os.path.join(HERE, "window15_slopes.csv"), index=False)

# figure
fig, axs = plt.subplots(3, 3, figsize=(14, 9), sharex=True)
for ax, m in zip(axs.ravel(), METRICS):
    ax.plot(tr.index, tr[m], lw=0.6, color="0.6")
    ax.plot(tr.index, tr[m].rolling(9, center=True, min_periods=3).median(), lw=1.4, color="C0")
    ax.axvline(40.5, color="C3", ls="--", lw=1)
    for s in (226, 236, 247): ax.axvline(s, color="C1", ls=":", lw=0.8)
    ax.set_title(METRICS[m], fontsize=9)
for ax in axs[-1]: ax.set_xlabel("step")
fig.suptitle("RL stage A: red dashed = features start (step 41); orange dotted = KL-guard LR cuts (226, 236, 247)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_feature_split.png"), dpi=130)
print("ok")
