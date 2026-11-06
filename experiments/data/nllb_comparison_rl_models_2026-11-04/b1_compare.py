#!/usr/bin/env python
"""E-06: RL models vs published NLLB-200 (3.3B, 1.3B, distilled-1.3B, MoE-54.5B) and other published systems on the FLORES+ devtest
languages shared with the published tables; plus the same-sentence NLLB-3.3B-vs-RL(beam-4) check from the post-editor run.
Only stored files are read.  python b1_compare.py
"""
import json, warnings
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = "/path/to/repo"
D = ROOT + "/reports/data"
OUT = D + "/nllb_comparison_rl_models_2026-11-04"
rng = np.random.default_rng(20261104)
NB = 10000

# ---- ours: FLORES+ devtest chrF++ per language, all systems (weak-group score table: identical to EVID/POSTB1 for sft, stageA, b1best, mixedbest)
M = pd.read_csv(D + "/eval_weak_group_2026-11-03/scores/metrics_by_language.csv")
M = M[(M.benchmark == "flores_plus") & (M.subset == "all")]
ours = M.pivot(index="iso", columns="system", values="chrf")
fin = M.pivot(index="iso", columns="system", values="finished")
rep = M.pivot(index="iso", columns="system", values="repeat")
ours = ours[["sft", "stageA", "b1best", "mixedbest", "weakfinal"]]
SYS = {"sft": "SFT", "stageA": "RL stage A (step 400)", "b1best": "B1 best (step 50)", "mixedbest": "mixed best",
       "weakfinal": "weak final (step 85)"}

# ---- published NLLB (per language, 43 shared with our FLORES+ 54)
pub = pd.read_csv(D + "/eval_385m_results_2026-11-03/published_nllb_flores_per_language.csv").set_index("iso")
# consistency check against our stored SFT numbers in that file
chk = (pub.chrf_m385 - ours.sft.reindex(pub.index)).abs().max()
print("max |published-file SFT - weak-group SFT| =", chk)
NL = {"NLLB-3.3B": "nllb_3p3b", "NLLB-1.3B": "nllb_1p3b", "NLLB-distilled-1.3B": "nllb_distilled_1p3b", "NLLB-MoE-54.5B": "nllb_moe_54p5b"}
L43 = list(pub.index)
assert len(L43) == 43

def boot_diff(a, b, nb=NB):
    d = np.asarray(a) - np.asarray(b)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(nb)])
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

# ---- Table 1: macro over the 43 shared languages
rows = []
for s, lab in SYS.items():
    for nm, col in NL.items():
        a = ours.loc[L43, s].values
        b = pub.loc[L43, col].values
        m, lo, hi = boot_diff(a, b)
        ratio = a / b
        rows.append(dict(model=lab, n_languages=43, ours_macro=a.mean(), system=nm, system_macro=b.mean(), diff=m, diff_lo=lo, diff_hi=hi,
                         macro_ratio=a.mean() / b.mean(), median_lang_ratio=np.median(ratio),
                         n_ours_ge=int((a >= b).sum()), n_ours_ge_minus1=int((a >= b - 1).sum()),
                         n_ours_ge_80pct=int((ratio >= 0.8).sum()), n_ours_ge_50pct=int((ratio >= 0.5).sum())))
T1 = pd.DataFrame(rows)
T1.to_csv(OUT + "/table1_macro_43_languages.csv", index=False)

# ---- Table 2: per language, 43 rows
T2 = pd.DataFrame({"iso": L43, "language": pub.language.values, "nllb_3p3b": pub.nllb_3p3b.values, "nllb_1p3b": pub.nllb_1p3b.values,
                   "nllb_distilled_1p3b": pub.nllb_distilled_1p3b.values, "nllb_moe_54p5b": pub.nllb_moe_54p5b.values})
for s in SYS:
    T2[s] = ours.loc[L43, s].values
T2["mixed_minus_nllb3p3b"] = T2.mixedbest - T2.nllb_3p3b
T2["mixed_over_nllb3p3b"] = T2.mixedbest / T2.nllb_3p3b
T2["sft_over_nllb3p3b"] = T2.sft / T2.nllb_3p3b
T2["mixed_finished"] = fin.mixedbest.reindex(L43).values
T2["mixed_repeat"] = rep.mixedbest.reindex(L43).values
T2["sft_finished"] = fin.sft.reindex(L43).values
T2["sft_repeat"] = rep.sft.reindex(L43).values
# tertiles of NLLB-3.3B (high 14, mid 14, tail 15, same convention as EVAL: rank on NLLB-3.3B chrF++)
o = T2.sort_values("nllb_3p3b", ascending=False).reset_index(drop=True)
tier = {}
for i, r in o.iterrows():
    tier[r.iso] = "high (14)" if i < 14 else ("mid (14)" if i < 28 else "tail (15)")
T2["nllb_tertile"] = T2.iso.map(tier)
T2 = T2.sort_values("mixed_minus_nllb3p3b", ascending=False)
T2.to_csv(OUT + "/table2_per_language_43.csv", index=False)

# ---- Table 3: tertiles
rows = []
for t in ["high (14)", "mid (14)", "tail (15)"]:
    d = T2[T2.nllb_tertile == t]
    r = dict(tertile=t, n=len(d), nllb_3p3b=d.nllb_3p3b.mean(), nllb_moe_54p5b=d.nllb_moe_54p5b.mean())
    for s in SYS:
        r[s] = d[s].mean()
        r[s + "_pct_of_nllb3p3b"] = 100 * d[s].mean() / d.nllb_3p3b.mean()
    rows.append(r)
d = T2
r = dict(tertile="all 43", n=43, nllb_3p3b=d.nllb_3p3b.mean(), nllb_moe_54p5b=d.nllb_moe_54p5b.mean())
for s in SYS:
    r[s] = d[s].mean(); r[s + "_pct_of_nllb3p3b"] = 100 * d[s].mean() / d.nllb_3p3b.mean()
rows.append(r)
T3 = pd.DataFrame(rows)
T3.to_csv(OUT + "/table3_by_nllb_tertile.csv", index=False)

# ---- failure-adjusted views of the same comparison: restrict to languages where the model finished >= 0.99 and loops <= 0.02
ok = T2[(T2.mixed_finished >= 0.99) & (T2.mixed_repeat <= 0.02)]
rows = []
for s in ["sft", "mixedbest"]:
    f = fin[s].reindex(L43); rp = rep[s].reindex(L43)
    sel = [i for i in L43 if f[i] >= 0.99 and rp[i] <= 0.02]
    a = ours.loc[sel, s].values; b = pub.loc[sel, "nllb_3p3b"].values
    m, lo, hi = boot_diff(a, b)
    rows.append(dict(model=SYS[s], rule="languages with finished>=0.99 and repeat<=0.02 (own model)", n_languages=len(sel), ours=a.mean(),
                     nllb_3p3b=b.mean(), diff=m, lo=lo, hi=hi, ratio=a.mean() / b.mean()))
# fixed language set = those the mixed model is clean on, for all models
sel = list(ok.iso)
for s in SYS:
    a = ours.loc[sel, s].values; b = pub.loc[sel, "nllb_3p3b"].values
    m, lo, hi = boot_diff(a, b)
    rows.append(dict(model=SYS[s], rule="fixed language set = languages where the mixed model is clean", n_languages=len(sel), ours=a.mean(),
                     nllb_3p3b=b.mean(), diff=m, lo=lo, hi=hi, ratio=a.mean() / b.mean()))
T4 = pd.DataFrame(rows)
T4.to_csv(OUT + "/table4_clean_language_subsets.csv", index=False)

# ---- other published systems on shared languages (Afrique 5-shot, AfroBench-Lite harness, en->xx per language)
pb = pd.read_csv(D + "/published_baselines.csv")
af = pb[(pb.direction == "en->xx") & pb.benchmark.str.contains("AfroBench-Lite") & pb.iso.isin(ours.index)]
rows = []
for model, d in af.groupby("model"):
    d = d.drop_duplicates("iso").set_index("iso")
    langs = [i for i in d.index if i in ours.index]
    r = dict(system=model, n_languages=len(langs), system_macro=d.loc[langs, "chrF++"].astype(float).mean())
    for s in SYS:
        r[s] = ours.loc[langs, s].mean()
    r["mixed_minus_system"] = r["mixedbest"] - r["system_macro"]
    r["n_lang_mixed_ge_system"] = int((ours.loc[langs, "mixedbest"].values >= d.loc[langs, "chrF++"].astype(float).values).sum())
    rows.append(r)
T5 = pd.DataFrame(rows).sort_values("system_macro", ascending=False)
T5.to_csv(OUT + "/table5_afrique_21_languages.csv", index=False)
afl = sorted(af.iso.unique())
print("Afrique languages (en->xx per-language, AfroBench-Lite):", afl)

# ---- same-sentence check from the post-editor run: NLLB-3.3B (our own run, FLORES+ references, 100 sentences per language) vs RL stage A beam-4
PE = D + "/post_editor_2026-11-03/e1/eval_e1b"
nl = pd.read_csv(PE + "/nllb3.3b_greedy/metrics_by_language.csv")
nl = nl[(nl.benchmark == "flores_plus") & (nl.system == "draft") & (nl.subset == "all")].set_index("iso").chrf
bm = pd.read_csv(PE + "/beam4/metrics_by_language.csv")
bm = bm[(bm.benchmark == "flores_plus") & (bm.system == "draft") & (bm.subset == "all")].set_index("iso")
bmn = bm.n
bm = bm.chrf
md = pd.read_csv(PE + "/madlad3b_greedy_x/metrics_by_language.csv")
md = md[(md.benchmark == "flores_plus") & (md.system == "draft") & (md.subset == "all")].set_index("iso").chrf
g250 = pd.read_csv(D + "/post_editor_2026-11-03/e1/eval_e1s/metrics_by_language.csv")
g250 = g250[(g250.benchmark == "flores_plus") & (g250.system == "draft") & (g250.subset == "all")].set_index("iso").chrf
common = sorted(set(nl.index) & set(bm.index))
S = pd.DataFrame({"nllb3p3b_own_run_100": nl.reindex(common), "rl_stageA_beam4_100": bm.reindex(common),
                  "rl_stageA_greedy_250": g250.reindex(common), "madlad3b_100": md.reindex(common),
                  "published_nllb3p3b_1012": pub.nllb_3p3b.reindex(common), "sft_greedy_1012": ours.sft.reindex(common),
                  "mixedbest_greedy_1012": ours.mixedbest.reindex(common), "stageA_greedy_1012": ours.stageA.reindex(common)})
S["n_beam_sentences"] = bmn.reindex(common)
S.to_csv(OUT + "/table6_same_sentence_postedit_run.csv")
rows = []
def addrow(a, b, la, lb):
    sub = S[[a, b]].dropna()
    m, lo, hi = boot_diff(sub[a].values, sub[b].values)
    rows.append(dict(a=la, b=lb, n_languages=len(sub), a_macro=sub[a].mean(), b_macro=sub[b].mean(), diff=m, lo=lo, hi=hi, n_a_ge_b=int((sub[a] >= sub[b]).sum())))
addrow("rl_stageA_beam4_100", "nllb3p3b_own_run_100", "RL stage A beam-4 (100 same sentences)", "NLLB-3.3B greedy, our run (100 same sentences)")
addrow("rl_stageA_greedy_250", "nllb3p3b_own_run_100", "RL stage A greedy (250 sentences, superset not verified)", "NLLB-3.3B greedy, our run (100)")
addrow("nllb3p3b_own_run_100", "published_nllb3p3b_1012", "NLLB-3.3B, our run (100 sentences)", "NLLB-3.3B published (1012, beam)")
addrow("rl_stageA_greedy_250", "stageA_greedy_1012", "RL stage A greedy 250-sentence subsample", "RL stage A greedy 1012 sentences (sample-size check)")
addrow("mixedbest_greedy_1012", "published_nllb3p3b_1012", "mixed best greedy 1012", "NLLB-3.3B published 1012 (43 shared languages)")
M6 = pd.DataFrame(rows)
M6.to_csv(OUT + "/table6b_same_sentence_summary.csv", index=False)


# restricted to the 43 languages shared with the published NLLB table
S43 = S.loc[[i for i in S.index if i in pub.index]]
rows = []
for a, b, la, lb in [("rl_stageA_beam4_100", "nllb3p3b_own_run_100", "RL stage A beam-4", "NLLB-3.3B our run"),
                     ("rl_stageA_greedy_250", "nllb3p3b_own_run_100", "RL stage A greedy (250)", "NLLB-3.3B our run"),
                     ("rl_stageA_beam4_100", "rl_stageA_greedy_250", "RL stage A beam-4 (100)", "RL stage A greedy (250)")]:
    sub = S43[[a, b]].dropna()
    m, lo, hi = boot_diff(sub[a].values, sub[b].values)
    rows.append(dict(a=la, b=lb, n_languages=len(sub), a_macro=sub[a].mean(), b_macro=sub[b].mean(), diff=m, lo=lo, hi=hi, n_a_ge_b=int((sub[a] >= sub[b]).sum())))
M6b = pd.DataFrame(rows)
M6b.to_csv(OUT + "/table6c_same_sentence_43_languages.csv", index=False)
# AfriCOMET (held-out reference-based) per language on the same 100 sentences, beam-4 RL stage-A draft vs NLLB-3.3B draft
cm = pd.read_csv(D + "/post_editor_2026-11-03/e1/comet_e1c/comet_by_language.csv")
cm = cm[(cm.benchmark == "flores_plus") & (cm.subset == "all")]
cb = cm[cm.system == "beam4"].set_index("iso").comet_ref
cn = cm[cm.system == "nllb"].set_index("iso").comet_ref
cmad = cm[cm.system == "madlad"].set_index("iso").comet_ref
com = sorted(set(cb.index) & set(cn.index))
m, lo, hi = boot_diff(cb.reindex(com).values, cn.reindex(com).values)
C7 = pd.DataFrame([dict(a="RL stage A beam-4 drafts AfriCOMET", b="NLLB-3.3B drafts AfriCOMET", n_languages=len(com), a_macro=cb.reindex(com).mean(),
                        b_macro=cn.reindex(com).mean(), diff=m, lo=lo, hi=hi, n_a_ge_b=int((cb.reindex(com).values >= cn.reindex(com).values).sum()))])
C7.to_csv(OUT + "/table7_africomet_same_100_sentences.csv", index=False)
print(M6b.round(2).to_string()); print(C7.round(4).to_string())

# madlad vs ours
sub = S[["madlad3b_100", "rl_stageA_beam4_100", "rl_stageA_greedy_250"]].dropna()
print("MADLAD langs:", len(sub), sub.mean().round(2).to_dict())

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 100)
print(T1.round(2).to_string())
print(T3.round(1).to_string())
print(T4.round(2).to_string())
print(T5.round(2).to_string())
print(M6.round(2).to_string())
print(T2[["iso", "nllb_3p3b", "sft", "mixedbest", "mixed_minus_nllb3p3b", "mixed_over_nllb3p3b", "mixed_finished", "mixed_repeat", "nllb_tertile"]].round(2).to_string())
