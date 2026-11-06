#!/usr/bin/env python3
"""Step 5: is the per-language regression of the 385M (vs old 270M SFT / RL, and vs A0) linked to the language's NLLB-mined share of stage B?

Inputs (this directory): nllb_share_by_language.csv (s03), eval_by_language_benchmark.csv (s02), a0_m385_rescored_old_unseen.csv (s04),
../sft_clean_2026-10-28/comparison_old_vs_rebuilt.csv (old SFT target tokens per language).
Outputs: link_language_table.csv, link_spearman.csv, link_ols.csv, link_blocks.csv, link_matched.csv, fig_*.png
"""
import os, itertools, warnings
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
warnings.filterwarnings("ignore")
H = os.path.dirname(os.path.abspath(__file__))
D = os.path.abspath(os.path.join(H, ".."))
rng = np.random.default_rng(20261103)

share = pd.read_csv(os.path.join(H, "nllb_share_by_language.csv"))
ev = pd.read_csv(os.path.join(H, "eval_by_language_benchmark.csv"))
a0c = pd.read_csv(os.path.join(H, "a0_m385_rescored_old_unseen.csv"))
oldcmp = pd.read_csv(os.path.join(D, "sft_clean_2026-10-28", "comparison_old_vs_rebuilt.csv"))[["iso", "old_target_tokens", "est_tokens_after", "rebuilt_after_as_pct_of_old_tokens"]]

BLOCK = {**{i: "Sotho-Tswana-Tsonga-Swati" for i in "nso tsn sot tso ssw".split()}, **{i: "Nguni_no_NLLB" for i in "zul xho nde nbl".split()},
         **{i: "Great-Lakes Bantu" for i in "lug kin run".split()}, **{i: "Other Bantu" for i in "sna nya swh bem tum lin kon kik kam umb kmb lua ven".split()},
         **{i: "West African" for i in "yor ibo ewe twi fon wol bam kbp mos dyu efi pcm".split()}, **{i: "Afroasiatic/Cushitic" for i in "hau orm som amh tir kab shi".split()},
         **{i: "other" for i in "afr mlg kea sag luo nus ach".split()}}
NONLATIN = {"amh", "tir", "shi"}

# ---- language table -----------------------------------------------------------------------
L = share[["iso", "tokens_B", "nllb_mined_tokens", "rows_nllb_mined", "share_nllb_mined", "share_nllb_strict", "share_nllb_incl_tum", "share_nllb_mined_of_translation", "instr_tokens",
           "non_nllb_tokens_B_strict", "tr_tokens_est", "aya_aligned", "tumbuka_mt"]].copy()
L["instr_share"] = L.instr_tokens / L.tokens_B
L["log_nonnllb"] = np.log10(L.non_nllb_tokens_B_strict.clip(lower=1e5))
L["log_nllb"] = np.log10(L.nllb_mined_tokens.clip(lower=1e4))
L = L.merge(oldcmp, on="iso", how="left")
L["log_data_vs_old"] = np.log10(L.rebuilt_after_as_pct_of_old_tokens / 100.0)
L["block"] = L.iso.map(BLOCK).fillna("other")
L["nonlatin"] = L.iso.isin(NONLATIN).astype(int)

e = ev.copy()
e["bm_ok"] = e.bm.isin(["ntrex", "mafand", "smol_sent", "flores_plus"])
# per-benchmark regression values (already differences), then language index = equal-weight mean over available benchmarks
def idx(df, col, bms, name):
    d = df[df.bm.isin(bms)].dropna(subset=[col]).groupby("iso")[col].agg(["mean", "count"]).rename(columns={"mean": name, "count": name + "_nb"})
    return d
parts = [
    idx(e, "R_sft", ["ntrex", "mafand"], "R_sft_big"), idx(e, "R_sft", ["ntrex", "mafand", "smol_sent", "flores_plus"], "R_sft_all4"),
    idx(e, "R_rl_est", ["ntrex", "mafand"], "R_rl_big"), idx(e, "R_rl_est", ["ntrex", "mafand", "flores_plus"], "R_rl_all3"),
    idx(e, "R_A_est", ["ntrex", "mafand"], "R_A_big"),
    idx(e, "R_sft", ["ntrex"], "R_sft_ntrex"), idx(e, "R_sft", ["mafand"], "R_sft_mafand"), idx(e, "R_sft", ["flores_plus"], "R_sft_flores48"), idx(e, "R_sft", ["smol_sent"], "R_sft_smol"),
    idx(e, "R_rl_est", ["ntrex"], "R_rl_ntrex"), idx(e, "R_rl_est", ["mafand"], "R_rl_mafand"),
]
Y = pd.concat(parts, axis=1).reset_index().rename(columns={"index": "iso"})
L = L.merge(Y, on="iso", how="left")

# stage-B step (A0 -> 385M SFT) on ALL rows, both models clean (FLORES devtest: 0 exposure for either), pooled over the benchmarks the language has
st = e.dropna(subset=["a0_all", "m385_sft_all"]).copy()
st["dB"] = st.m385_sft_all - st.a0_all
for bm in ["flores_plus", "ntrex", "mafand"]:
    pass
g = pd.concat([st[st.bm == b].set_index("iso").dB.rename("dB_" + b) for b in ["flores_plus", "ntrex", "mafand", "smol_sent"]], axis=1)
g["dB_pool"] = g.mean(axis=1)
L = L.merge(g.reset_index(), on="iso", how="left")
# old SFT -> A0 (pre-NLLB regression) on old-unseen sentences
c = a0c[(a0c.subset == "clean")].pivot_table(index=["bm", "iso"], columns="system", values="chrf").reset_index()
c = c.merge(e[["bm", "iso", "old_sft", "old_rl"]], on=["bm", "iso"], how="inner")
c["R_old2a0"] = c.a0 - c.old_sft
c["R_a0_2sft"] = c.m385 - c.a0
cc = c[c.bm.isin(["ntrex", "mafand", "smol_sent", "flores_plus"])].groupby("iso")[["R_old2a0", "R_a0_2sft"]].mean().reset_index()
cc["R_old2a0_big"] = c[c.bm.isin(["ntrex", "mafand"])].groupby("iso").R_old2a0.mean().reindex(cc.iso).values
cc["R_a0_2sft_big"] = c[c.bm.isin(["ntrex", "mafand"])].groupby("iso").R_a0_2sft.mean().reindex(cc.iso).values
L = L.merge(cc, on="iso", how="left")
# level covariates
lv = e[e.bm.isin(["ntrex", "mafand", "smol_sent", "flores_plus"])].groupby("iso")[["old_sft", "a0_all"]].mean().rename(columns={"old_sft": "old_sft_level", "a0_all": "a0_level"}).reset_index()
L = L.merge(lv, on="iso", how="left")
# RL gain (B1 best - SFT, all rows) on FLORES / NTREX / MAFAND
rg = e.assign(gain=e.b1_all - e.m385_sft_all).dropna(subset=["gain"]).groupby("iso").gain.mean().rename("rl_gain").reset_index()
L = L.merge(rg, on="iso", how="left")
L = L.replace([np.inf, -np.inf], np.nan)
L.to_csv(os.path.join(H, "link_language_table.csv"), index=False)

# ---- Spearman with bootstrap CI + permutation p ------------------------------------------------------
def spear(x, y, B=5000):
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]; n = len(x)
    if n < 6: return dict(n=n, rho=np.nan, lo=np.nan, hi=np.nan, p_perm=np.nan)
    rho = stats.spearmanr(x, y).statistic
    bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(set(x[i])) < 2 or len(set(y[i])) < 2: continue
        bs.append(stats.spearmanr(x[i], y[i]).statistic)
    perm = np.array([stats.spearmanr(x, rng.permutation(y)).statistic for _ in range(B)])
    return dict(n=n, rho=rho, lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5), p_perm=(np.sum(np.abs(perm) >= abs(rho)) + 1) / (B + 1))

rows = []
targets = [("R_sft_big", "SFT vs old SFT, NTREX+MAFAND clean"), ("R_sft_all4", "SFT vs old SFT, NTREX+MAFAND+SmolSent+FLORES48 clean"), ("R_sft_ntrex", "SFT vs old SFT, NTREX clean"),
           ("R_sft_mafand", "SFT vs old SFT, MAFAND clean"), ("R_sft_flores48", "SFT vs old SFT, FLORES old-unseen 48"), ("R_sft_smol", "SFT vs old SFT, SmolSent clean"),
           ("R_rl_big", "B1 vs old RL, NTREX+MAFAND clean (est)"), ("R_rl_all3", "B1 vs old RL, NTREX+MAFAND+FLORES48 (est)"),
           ("R_A_big", "stage A vs old RL, NTREX+MAFAND clean (est)"),
           ("R_old2a0", "A0 vs old SFT (pre-NLLB), 4 clean benchmarks"), ("R_old2a0_big", "A0 vs old SFT (pre-NLLB), NTREX+MAFAND"),
           ("R_a0_2sft", "385M SFT vs A0, old-unseen sentences, 4 benchmarks"), ("dB_flores_plus", "385M SFT - A0, FLORES devtest all rows (both clean)"), ("dB_ntrex", "385M SFT - A0, NTREX all rows"),
           ("dB_mafand", "385M SFT - A0, MAFAND all rows"), ("dB_pool", "385M SFT - A0, pooled"), ("rl_gain", "RL gain (B1 - SFT), pooled")]
xs = [("share_nllb_mined", "NLLB-mined share of stage-B tokens"), ("share_nllb_strict", "NLLB-derived strict share (mined+seed+Aya)"), ("log_nllb", "log10 NLLB-mined tokens"),
      ("share_nllb_mined_of_translation", "NLLB share of translation tokens")]
for (yc, yn), (xc, xn) in itertools.product(targets, xs):
    s = spear(L[xc].values.astype(float), L[yc].values.astype(float))
    rows.append(dict(target=yc, target_desc=yn, predictor=xc, **s))
    # restricted to languages with NLLB > 0
    m = L.rows_nllb_mined > 0
    s2 = spear(L.loc[m, xc].values.astype(float), L.loc[m, yc].values.astype(float))
    rows.append(dict(target=yc, target_desc=yn + " | NLLB>0 only", predictor=xc, **s2))
sp = pd.DataFrame(rows)
sp.to_csv(os.path.join(H, "link_spearman.csv"), index=False)

# ---- OLS ------------------------------------------------------------------------------------------------
def z(s): return (s - s.mean()) / s.std()
specs = {
    "M1 share": ["share_nllb_mined"],
    "M2 + log non-NLLB tokens": ["share_nllb_mined", "log_nonnllb"],
    "M3 + data vs old build": ["share_nllb_mined", "log_nonnllb", "log_data_vs_old"],
    "M4 + old level, script, instruction share": ["share_nllb_mined", "log_nonnllb", "log_data_vs_old", "old_sft_level", "nonlatin", "instr_share"],
}
orows = []
for yc in ["R_sft_big", "R_sft_all4", "R_rl_big", "R_rl_all3", "R_old2a0", "R_a0_2sft", "dB_pool", "dB_flores_plus"]:
    for nm, xs_ in specs.items():
        d = L[[yc] + xs_].dropna()
        if len(d) < len(xs_) + 6: continue
        X = sm.add_constant(d[xs_]); mod = sm.OLS(d[yc], X).fit(cov_type="HC3")
        b = []
        for _ in range(2000):
            i = rng.integers(0, len(d), len(d)); dd = d.iloc[i]
            try: b.append(sm.OLS(dd[yc], sm.add_constant(dd[xs_])).fit().params["share_nllb_mined"])
            except Exception: pass
        orows.append(dict(target=yc, model=nm, n=len(d), coef_share=mod.params["share_nllb_mined"], se_hc3=mod.bse["share_nllb_mined"], p=mod.pvalues["share_nllb_mined"],
                          boot_lo=np.percentile(b, 2.5), boot_hi=np.percentile(b, 97.5), r2=mod.rsquared, per_10pct_pts=mod.params["share_nllb_mined"] * 0.1,
                          other={k: round(float(v), 3) for k, v in mod.params.items() if k not in ("const", "share_nllb_mined")}))
pd.DataFrame(orows).to_csv(os.path.join(H, "link_ols.csv"), index=False)

# ---- blocks -------------------------------------------------------------------------------------------------
bl = L.groupby("block").agg(n=("iso", "count"), share=("share_nllb_mined", "mean"), R_sft_all4=("R_sft_all4", "mean"), R_sft_big=("R_sft_big", "mean"), R_rl_all3=("R_rl_all3", "mean"), R_old2a0=("R_old2a0", "mean"),
                            R_a0_2sft=("R_a0_2sft", "mean"), dB_pool=("dB_pool", "mean"), rl_gain=("rl_gain", "mean")).reset_index()
bl.to_csv(os.path.join(H, "link_blocks.csv"), index=False)
print(bl.round(2).to_string())
print(sp[sp.target_desc.str.contains("NLLB>0") == False][sp.predictor == "share_nllb_mined"][["target", "n", "rho", "lo", "hi", "p_perm"]].round(3).to_string())
print(pd.DataFrame(orows)[["target", "model", "n", "coef_share", "se_hc3", "p", "boot_lo", "boot_hi", "r2"]].round(3).to_string())
