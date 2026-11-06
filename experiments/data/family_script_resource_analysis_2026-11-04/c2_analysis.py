#!/usr/bin/env python
"""E-12 step 2: regress RL gain (SFT -> mixed best, chrF++) and SFT level on token budget, Glottolog family, script,
NLLB resource proxy, NLLB-mined share, AfriCOMET availability. Observational. All inputs are stored files.
Run after c1_build_registry.py:  python c2_analysis.py
"""
import json, itertools, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
ROOT = "/path/to/repo"
D = ROOT + "/reports/data"
OUT = D + "/family_script_resource_analysis_2026-11-04"
rng = np.random.default_rng(20261104)
NB = 4000

R = pd.read_csv(OUT + "/registry_125.csv")
ev = json.load(open(D + "/rl_vs_sft_evidence_2026-11-04/evidence.json"))
cells = ev["cells"]["all"]  # clean rows; FLORES+ has 0 removed rows
rows = []
for k, v in cells.items():
    b, iso = k.split(":")
    rows.append(dict(benchmark=b, iso=iso, n=v["n"], sft=v["pp"]["sft"], rl=v["pp"]["rl"], gain=v["pp"]["diff"],
                     gain_lo=v["pp"]["lo"], gain_hi=v["pp"]["hi"]))
CELLS = pd.DataFrame(rows)
CELLS.to_csv(OUT + "/cells_gain_sft_rl_chrfpp.csv", index=False)

# stage models on FLORES+ (all systems from the weak-group score table; identical to EVID numbers for sft/mixed)
M = pd.read_csv(D + "/eval_weak_group_2026-11-03/scores/metrics_by_language.csv")
M = M[(M.benchmark == "flores_plus") & (M.subset == "all")]
W = M.pivot(index="iso", columns="system", values="chrf")
Fin = M.pivot(index="iso", columns="system", values="finished")
Rep = M.pivot(index="iso", columns="system", values="repeat")
fl = CELLS[CELLS.benchmark == "flores_plus"].set_index("iso")
chk = (fl.sft - W.sft.reindex(fl.index)).abs().max(), (fl.rl - W.mixedbest.reindex(fl.index)).abs().max()
print("check EVID vs weak-group tables FLORES+ (max abs diff sft, rl):", chk)

D1 = R.set_index("iso").loc[fl.index].copy()
D1["sft"] = fl.sft
D1["mixed"] = fl.rl
D1["gain"] = fl.gain
D1["gain_stageA"] = W.stageA - W.sft
D1["sft_finished"] = Fin.sft
D1["sft_repeat"] = Rep.sft
D1["log_tok"] = np.log10(D1.sft_tokens_M.astype(float))
D1["script_nonlatin"] = (D1.script != "LATIN").astype(int)
D1["africomet"] = D1.africomet_available_flores
D1["nllb_share"] = D1.share_nllb_mined  # fraction of stage-B tokens
D1["nllb_chrf"] = D1.nllb_3p3b_published_chrfpp
D1["fam"] = D1.family_reg.str.replace(r"[^A-Za-z]", "_", regex=True)
D1["our_group"] = D1.our_group_evid_table4
D1 = D1.reset_index()
D1.to_csv(OUT + "/flores54_regression_table.csv", index=False)
print("n FLORES languages", len(D1), " missing tokens:", D1.log_tok.isna().sum(), " missing nllb_share:", D1.nllb_share.isna().sum(),
      " nllb_chrf n:", D1.nllb_chrf.notna().sum())

res = []


def vif_table(df, cols):
    X = sm.add_constant(df[cols].astype(float))
    return {c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns) if c != "const"}


def boot_ols(formula, df, y, nb=NB):
    base = smf.ols(formula, df).fit()
    names = list(base.params.index)
    bs = []
    n = len(df)
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        d = df.iloc[idx]
        try:
            m = smf.ols(formula, d).fit()
            bs.append(m.params.reindex(names).values)
        except Exception:
            pass
    bs = np.array(bs, dtype=float)
    lo = np.nanpercentile(bs, 2.5, axis=0)
    hi = np.nanpercentile(bs, 97.5, axis=0)
    robust = smf.ols(formula, df).fit(cov_type="HC3")
    t = pd.DataFrame({"coef": base.params, "se_HC3": robust.bse, "p_HC3": robust.pvalues, "boot_lo": lo, "boot_hi": hi})
    return base, t


def run(name, df, y, rhs_terms, note=""):
    d = df.dropna(subset=[y] + [t for t in rhs_terms if "(" not in t and t != "C(fam)"]).copy()
    f = f"{y} ~ " + " + ".join(rhs_terms)
    base, t = boot_ols(f, d, y)
    t.insert(0, "model", name)
    t.insert(1, "y", y)
    t["n"] = len(d)
    t["R2"] = base.rsquared
    t["adjR2"] = base.rsquared_adj
    t["note"] = note
    res.append(t.reset_index().rename(columns={"index": "term"}))
    return base, t, d


numeric = ["log_tok", "script_nonlatin", "africomet", "nllb_share"]
FAM = "C(family_reg, Treatment('Atlantic-Congo: Narrow Bantu'))"
models = {}
for y in ["gain", "sft"]:
    # M_A: full model on FLORES+ (54 languages, minus any with missing covariates)
    models[(y, "A")] = run(f"A_full_{y}", D1, y, ["log_tok", "script_nonlatin", "africomet", "nllb_share", FAM],
                           "reference family: Narrow Bantu; script_nonlatin: Ethiopic/Arabic/Tifinagh vs Latin")
    # M_B: no family
    models[(y, "B")] = run(f"B_nofamily_{y}", D1, y, ["log_tok", "script_nonlatin", "africomet", "nllb_share"], "without family terms")
    # M_C: with the NLLB-3.3B chrF++ resource proxy (published, 43 languages)
    models[(y, "C")] = run(f"C_nllbproxy_{y}", D1, y, ["log_tok", "script_nonlatin", "africomet", "nllb_share", "nllb_chrf", FAM],
                           "adds NLLB-3.3B published chrF++ (n=43)")
    # M_D: tokens only
    models[(y, "D")] = run(f"D_tokens_only_{y}", D1, y, ["log_tok"], "single predictor")
# gain additionally given the SFT level
models[("gain", "E")] = run("E_gain_given_sft", D1, "gain", ["sft", "log_tok", "africomet", FAM], "adds SFT chrF++ level as covariate")
allres = pd.concat(res)
allres.to_csv(OUT + "/regression_coefficients.csv", index=False)

# VIF (numeric covariates, then with family dummies)
vifs = {}
d = D1.dropna(subset=["log_tok", "nllb_share"])
vifs["numeric (log_tok, nonlatin, africomet, nllb_share)"] = vif_table(d, numeric)
dd = pd.get_dummies(d.family_reg, drop_first=False).astype(float)
dd = dd.drop(columns=[c for c in dd.columns if c.startswith("Atlantic-Congo: Narrow Bantu")])
X = pd.concat([d[numeric].astype(float), dd], axis=1)
Xc = sm.add_constant(X)
vifs["with family dummies"] = {c: variance_inflation_factor(Xc.values, i) for i, c in enumerate(Xc.columns) if c != "const"}
d3 = D1.dropna(subset=["log_tok", "nllb_share", "nllb_chrf"])
vifs["with NLLB chrF++ proxy (n=%d, no family)" % len(d3)] = vif_table(d3, numeric + ["nllb_chrf"])
json.dump(vifs, open(OUT + "/vif.json", "w"), indent=1)

# Spearman
sp = []
for y in ["gain", "sft"]:
    for x in ["log_tok", "nllb_share", "nllb_chrf", "africomet", "script_nonlatin", "sft", "sft_finished", "sft_repeat"]:
        if x == y:
            continue
        dd_ = D1.dropna(subset=[y, x])
        if dd_[x].nunique() < 2:
            continue
        rho, p = stats.spearmanr(dd_[x], dd_[y])
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(dd_), len(dd_))
            r_ = stats.spearmanr(dd_[x].values[i], dd_[y].values[i])[0]
            bs.append(r_)
        sp.append(dict(y=y, x=x, n=len(dd_), rho=rho, p=p, boot_lo=np.nanpercentile(bs, 2.5), boot_hi=np.nanpercentile(bs, 97.5)))
sp = pd.DataFrame(sp)
sp.to_csv(OUT + "/spearman.csv", index=False)

# mixed model: random intercept by Glottolog top family (family_reg w/o Bantu split) and by branch
mx = []
for y in ["gain", "sft"]:
    for grpcol in ["family_reg", "family_branch", "glottolog_top_family"]:
        d = D1.dropna(subset=["log_tok", "nllb_share", y]).copy()
        try:
            m = smf.mixedlm(f"{y} ~ log_tok + script_nonlatin + africomet + nllb_share", d, groups=d[grpcol]).fit(reml=True)
            vr = float(m.cov_re.iloc[0, 0])
            icc = vr / (vr + m.scale)
            mx.append(dict(y=y, group=grpcol, n=len(d), n_groups=d[grpcol].nunique(), var_group=vr, var_resid=m.scale, ICC=icc,
                           b_log_tok=m.params["log_tok"], p_log_tok=m.pvalues["log_tok"], converged=m.converged))
        except Exception as e:
            mx.append(dict(y=y, group=grpcol, error=str(e)[:100]))
    # null model ICC
    for grpcol in ["family_reg", "family_branch", "glottolog_top_family"]:
        d = D1.dropna(subset=[y]).copy()
        try:
            m = smf.mixedlm(f"{y} ~ 1", d, groups=d[grpcol]).fit(reml=True)
            vr = float(m.cov_re.iloc[0, 0])
            mx.append(dict(y=y, group=grpcol + " (null model)", n=len(d), n_groups=d[grpcol].nunique(), var_group=vr, var_resid=m.scale,
                           ICC=vr / (vr + m.scale), converged=m.converged))
        except Exception as e:
            mx.append(dict(y=y, group=grpcol + " (null)", error=str(e)[:100]))
pd.DataFrame(mx).to_csv(OUT + "/mixed_models.csv", index=False)

# leave-one-family-out: effect of log_tok on gain / sft
lofo = []
for y in ["gain", "sft"]:
    for fam in sorted(D1.family_reg.unique()):
        d = D1[(D1.family_reg != fam)].dropna(subset=["log_tok", y])
        m = smf.ols(f"{y} ~ log_tok", d).fit(cov_type="HC3")
        lofo.append(dict(y=y, dropped=fam, n=len(d), b_log_tok=m.params["log_tok"], se=m.bse["log_tok"]))
pd.DataFrame(lofo).to_csv(OUT + "/leave_one_family_out.csv", index=False)


# group tables with language-level bootstrap CIs
def group_table(col, df=D1, y="gain"):
    out = []
    for g, d in df.groupby(col):
        v = d[y].values
        bs = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(2000)] if len(v) > 1 else [np.nan]
        out.append(dict(group_col=col, group=g, n_lang=len(v), mean_sft=d.sft.mean(), mean_mixed=d.mixed.mean(),
                        mean_gain=v.mean(), median_gain=np.median(v), boot_lo=np.nanpercentile(bs, 2.5), boot_hi=np.nanpercentile(bs, 97.5),
                        min_gain=v.min(), max_gain=v.max(), languages=" ".join(sorted(d.iso))))
    return pd.DataFrame(out).sort_values("mean_gain", ascending=False)


GT = pd.concat([group_table(c) for c in ["glottolog_top_family", "family_reg", "family_branch", "script", "our_group", "africomet"]])
GT.to_csv(OUT + "/gain_by_group_flores54.csv", index=False)
# the same for the SFT level
GS = pd.concat([group_table(c, y="sft") for c in ["glottolog_top_family", "family_reg", "family_branch", "script"]])
GS = GS.rename(columns={"mean_gain": "mean_sft_level_check"})
GS.to_csv(OUT + "/sft_level_by_group_flores54.csv", index=False)

# 8 unlabelled languages: Glottolog family and our group
UNL = ["cjk", "dik", "mfe", "taq", "arz", "ary", "zgh", "aeb"]
U = D1[D1.iso.isin(UNL)][["iso", "language", "our_group", "glottocode", "glottolog_name", "glottolog_top_family", "family_branch",
                          "glottolog_path_first4", "script", "sft", "mixed", "gain"]]
U.to_csv(OUT + "/unlabelled8_labels.csv", index=False)

# pooled-benchmark robustness: gain over FLORES+/NTREX/MAFAND cells, benchmark fixed effects, SE clustered by language
P = CELLS.merge(R[["iso", "family_reg", "script", "sft_tokens_M", "share_nllb_mined", "africomet_available_flores"]], on="iso")
P["log_tok"] = np.log10(P.sft_tokens_M.astype(float))
P["script_nonlatin"] = (P.script != "LATIN").astype(int)
P = P.dropna(subset=["log_tok", "share_nllb_mined"])
pool = []
for y in ["gain", "sft"]:
    m = smf.ols(f"{y} ~ log_tok + script_nonlatin + share_nllb_mined + C(benchmark) + {FAM}", P).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(P.iso)[0]})
    for t in m.params.index:
        pool.append(dict(y=y, term=t, coef=m.params[t], se_cluster=m.bse[t], p=m.pvalues[t], n_cells=int(m.nobs), n_lang=P.iso.nunique()))
pd.DataFrame(pool).to_csv(OUT + "/pooled_cells_regression.csv", index=False)

# summary print
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
print(allres[allres.model.isin(["A_full_gain", "A_full_sft", "D_tokens_only_gain", "D_tokens_only_sft", "C_nllbproxy_gain", "C_nllbproxy_sft"])].round(3).to_string())
print(sp.round(3).to_string())
print(pd.DataFrame(mx).round(3).to_string())
print(GT[["group_col", "group", "n_lang", "mean_sft", "mean_gain", "boot_lo", "boot_hi"]].round(2).to_string())
print(U.round(2).to_string())
print(json.dumps(vifs, indent=1))
print(pd.DataFrame(pool).round(3).to_string())
