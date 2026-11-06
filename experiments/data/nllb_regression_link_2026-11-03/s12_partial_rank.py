#!/usr/bin/env python3
"""Step 12: partial rank correlation of NLLB-mined share with the regression, controlling for resource level (log10 non-NLLB stage-B tokens), the old model's level (ceiling /
regression to the mean) and both.  Rank-transform, residualise on the controls (OLS), correlate the residuals; bootstrap over languages (4,000) for the 95% CI; permutation p (5,000)."""
import os
import numpy as np, pandas as pd
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(99)
L = pd.read_csv(os.path.join(H, "link_language_table_full.csv"))


def resid(y, X):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return y - X1 @ b


def pr(d, x, y, ctrl):
    r = lambda v: stats.rankdata(v)
    xr, yr = r(d[x].values), r(d[y].values)
    if ctrl:
        C = np.column_stack([r(d[c].values) for c in ctrl])
        xr, yr = resid(xr, C), resid(yr, C)
    return np.corrcoef(xr, yr)[0, 1]

rows = []
for y in ["R_sft_big", "R_sft_all4", "R_rl_big", "R_rl_all3", "dB_pool", "dB_flores_plus"]:
    for ctrl, nm in [([], "none"), (["log_nonnllb"], "resource (log non-NLLB tokens)"), (["old_sft_level"], "old SFT level"), (["a0_level"], "A0 level"), (["log_nonnllb", "old_sft_level"], "resource + old level"),
                     (["log_nonnllb", "a0_level"], "resource + A0 level")]:
        cols = [y, "share_nllb_mined"] + ctrl
        d = L[cols].dropna().reset_index(drop=True)
        if len(d) < 12:
            continue
        rho = pr(d, "share_nllb_mined", y, ctrl)
        bs = [pr(d.iloc[i], "share_nllb_mined", y, ctrl) for i in (rng.integers(0, len(d), len(d)) for _ in range(4000))]
        perm = []
        for _ in range(3000):
            dd = d.copy(); dd[y] = rng.permutation(dd[y].values)
            perm.append(pr(dd, "share_nllb_mined", y, ctrl))
        rows.append(dict(target=y, control=nm, n=len(d), partial_rho=rho, lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5), p_perm=(np.sum(np.abs(perm) >= abs(rho)) + 1) / 3001))
r = pd.DataFrame(rows)
r.to_csv(os.path.join(H, "link_partial_rank.csv"), index=False)
pd.set_option("display.width", 200)
print(r.round(3).to_string())
d = L[["share_nllb_mined", "log_nonnllb", "old_sft_level", "R_sft_big"]].dropna()
print("share vs log non-NLLB tokens (rho):", stats.spearmanr(d.share_nllb_mined, d.log_nonnllb).statistic, " R_sft_big vs log non-NLLB:", stats.spearmanr(d.R_sft_big, d.log_nonnllb).statistic,
      " R_sft_big vs old level:", stats.spearmanr(d.R_sft_big, d.old_sft_level).statistic, " share vs old level:", stats.spearmanr(d.share_nllb_mined, d.old_sft_level).statistic, "n", len(d))
