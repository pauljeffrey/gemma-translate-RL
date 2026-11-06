#!/usr/bin/env python3
"""Step 8: (a) two-source dose-response of the A0 -> 385M-SFT change on FLORES devtest (both models clean), (b) matched contrasts, (c) leave-one-out / leave-block-out, (d) figures.

(a) dB_l = chrF++(385M SFT) - chrF++(A0) on FLORES+ devtest, all rows (neither model saw FLORES devtest). Regressors: log10(1 + non-NLLB new translation tokens / A0 tokens),
    log10(1 + NLLB tokens / A0 tokens), instruction share of stage B, A0 level (headroom). A0 tokens per language = A0_trained_Mtok of reports/data/token_table_provisional_2026-11-01.csv.
    If NLLB tokens were as useful as other new tokens the two slopes would be similar; a clearly smaller (or negative) NLLB slope says NLLB tokens add less/hurt.
(b) matched contrasts: treated = NLLB share >= 0.30; controls = share <= 0.10 within +-0.35 dex of non-NLLB stage-B tokens and +-8 chrF of A0 level.
"""
import os, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__))
D = os.path.abspath(os.path.join(H, ".."))
rng = np.random.default_rng(7)
L = pd.read_csv(os.path.join(H, "link_language_table.csv"))
tt = pd.read_csv(os.path.join(D, "token_table_provisional_2026-11-01.csv"))[["iso", "A0_trained_Mtok"]]
S = pd.read_csv(os.path.join(H, "nllb_share_by_language.csv"))[["iso", "tr_tokens_est", "nllb_mined_tokens", "nllb_seed_tokens", "aya_aligned"]]
L = L.merge(tt, on="iso", how="left").merge(S[["iso", "tr_tokens_est", "nllb_seed_tokens"]], on="iso", how="left", suffixes=("", "_s"))
L["a0_tok"] = L.A0_trained_Mtok * 1e6
L["new_nonnllb_tr"] = (L.tr_tokens_est - L.nllb_mined_tokens - L.nllb_seed_tokens - L.a0_tok).clip(lower=0)
L["x_clean"] = np.log10(1 + L.new_nonnllb_tr / L.a0_tok.clip(lower=1e6))
L["x_nllb"] = np.log10(1 + L.nllb_mined_tokens / L.a0_tok.clip(lower=1e6))
L["x_instr"] = L.instr_share

rows = []
for y, nm in [("dB_flores_plus", "FLORES devtest"), ("dB_pool", "pooled benchmarks"), ("dB_ntrex", "NTREX"), ("dB_mafand", "MAFAND")]:
    for cols, lab in [(["x_clean", "x_nllb"], "two sources"), (["x_clean", "x_nllb", "x_instr"], "+ instruction share"), (["x_clean", "x_nllb", "x_instr", "a0_level"], "+ A0 level")]:
        d = L[[y] + cols + ["iso"]].dropna()
        d = d[d.a0_tok.notna()] if "a0_tok" in d else d
        if len(d) < len(cols) + 8:
            continue
        mod = sm.OLS(d[y], sm.add_constant(d[cols])).fit(cov_type="HC3")
        bs = []
        for _ in range(3000):
            i = rng.integers(0, len(d), len(d)); dd = d.iloc[i]
            try: bs.append(sm.OLS(dd[y], sm.add_constant(dd[cols])).fit().params[["x_clean", "x_nllb"]].values)
            except Exception: pass
        bs = np.array(bs)
        rows.append(dict(target=nm, spec=lab, n=len(d), b_clean=mod.params["x_clean"], b_clean_se=mod.bse["x_clean"], b_nllb=mod.params["x_nllb"], b_nllb_se=mod.bse["x_nllb"], p_nllb=mod.pvalues["x_nllb"],
                         nllb_lo=np.percentile(bs[:, 1], 2.5), nllb_hi=np.percentile(bs[:, 1], 97.5), diff_nllb_minus_clean=mod.params["x_nllb"] - mod.params["x_clean"],
                         diff_lo=np.percentile(bs[:, 1] - bs[:, 0], 2.5), diff_hi=np.percentile(bs[:, 1] - bs[:, 0], 97.5), r2=mod.rsquared))
dr = pd.DataFrame(rows)
dr.to_csv(os.path.join(H, "link_dose_response.csv"), index=False)
pd.set_option("display.width", 250)
print(dr.round(2).to_string())

# ---- (b) matched contrasts
for yc in ["R_sft_all4", "R_sft_big", "dB_pool", "R_rl_all3"]:
    pass
m_rows = []
tr = L[(L.share_nllb_mined >= 0.30) & L.a0_level.notna()]
ct = L[(L.share_nllb_mined <= 0.10) & L.a0_level.notna()]
for t in tr.itertuples():
    c = ct[(abs(ct.log_nonnllb - t.log_nonnllb) <= 0.35) & (abs(ct.a0_level - t.a0_level) <= 8) & (ct.iso != t.iso)]
    for yc in ["R_sft_all4", "dB_pool", "R_rl_all3"]:
        tv = getattr(t, yc)
        cv = c[yc].mean() if len(c) else np.nan
        m_rows.append(dict(treated=t.iso, share=t.share_nllb_mined, outcome=yc, treated_value=tv, n_controls=len(c), controls=" ".join(c.iso.tolist()), control_mean=cv, diff=tv - cv if len(c) else np.nan))
mt = pd.DataFrame(m_rows)
mt.to_csv(os.path.join(H, "link_matched.csv"), index=False)
sm_ = mt.dropna(subset=["diff"]).groupby("outcome").agg(n_treated=("treated", "nunique"), mean_diff=("diff", "mean"), median_diff=("diff", "median"), frac_negative=("diff", lambda v: (v < 0).mean()))
print(sm_.round(2))
sm_.to_csv(os.path.join(H, "link_matched_summary.csv"))

# ---- (c) leave-one-out and leave-block-out of the headline Spearman (NTREX+MAFAND SFT regression and pooled stage-B change)
out = []
for yc in ["R_sft_big", "R_sft_all4", "dB_pool", "R_rl_all3"]:
    d = L[["iso", "block", "share_nllb_mined", yc]].dropna()
    full = stats.spearmanr(d.share_nllb_mined, d[yc]).statistic
    loo = [stats.spearmanr(d.drop(i).share_nllb_mined, d.drop(i)[yc]).statistic for i in d.index]
    out.append(dict(target=yc, kind="full", drop="-", rho=full))
    out.append(dict(target=yc, kind="leave-one-out range", drop="-", rho=f"{min(loo):.2f}..{max(loo):.2f}"))
    j = d.index[int(np.argmin(loo))]
    out.append(dict(target=yc, kind="LOO most negative", drop=d.loc[j, "iso"], rho=min(loo)))
    j = d.index[int(np.argmax(loo))]
    out.append(dict(target=yc, kind="LOO least negative", drop=d.loc[j, "iso"], rho=max(loo)))
    for b in d.block.unique():
        dd = d[d.block != b]
        if len(dd) > 8:
            out.append(dict(target=yc, kind="drop block", drop=b, rho=stats.spearmanr(dd.share_nllb_mined, dd[yc]).statistic))
    # also without the three highest-share languages (>0.9) and without NLLB=0 languages
    dd = d[d.share_nllb_mined < 0.9]
    out.append(dict(target=yc, kind="drop share>0.9", drop="-", rho=stats.spearmanr(dd.share_nllb_mined, dd[yc]).statistic))
    dd = d[d.share_nllb_mined > 0]
    out.append(dict(target=yc, kind="NLLB>0 only", drop="-", rho=stats.spearmanr(dd.share_nllb_mined, dd[yc]).statistic))
pd.DataFrame(out).to_csv(os.path.join(H, "link_robustness.csv"), index=False)
print(pd.DataFrame(out).to_string())

# ---- (d) figures
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
cols = {"Sotho-Tswana-Tsonga-Swati": "#c0392b", "Great-Lakes Bantu": "#e67e22", "Nguni_no_NLLB": "#2980b9", "Other Bantu": "#7f8c8d", "West African": "#27ae60", "Afroasiatic/Cushitic": "#8e44ad", "other": "#bdc3c7"}
for a_, (yc, tl) in zip(ax, [("R_sft_all4", "385M SFT minus old SFT (old-unseen sentences, mean of NTREX/MAFAND/SmolSent/FLORES-48)"), ("R_old2a0", "A0 (no NLLB) minus old SFT"), ("dB_pool", "385M SFT minus A0 (stage A+B, NLLB enters here)")]):
    d = L.dropna(subset=[yc])
    for b, g in d.groupby("block"):
        a_.scatter(g.share_nllb_mined, g[yc], c=cols.get(b, "#999"), label=b, s=28)
    for r in d[(d[yc].abs() > 9) | d.iso.isin(["kea", "nso", "lug", "ssw", "hau"])].itertuples():
        a_.annotate(r.iso, (r.share_nllb_mined, getattr(r, yc)), fontsize=7, xytext=(2, 2), textcoords="offset points")
    a_.axhline(0, c="k", lw=0.5); a_.set_xlabel("NLLB-mined share of the language's stage-B tokens"); a_.set_title(tl, fontsize=8.5); a_.set_ylabel("chrF++ difference")
ax[0].legend(fontsize=6.5, loc="lower left")
plt.tight_layout(); plt.savefig(os.path.join(H, "fig_nllb_share_vs_change.png"), dpi=140)
