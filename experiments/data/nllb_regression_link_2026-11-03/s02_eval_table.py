#!/usr/bin/env python3
"""Step 2: per language x benchmark table of the 385M systems vs the old 270M models, on the sentences the old models had NOT seen.

Definitions (chrF++ throughout, greedy, 768 tokens, one reference):
  old_sft, old_rl       old 270M SFT-final / RL-final, scored on the rows whose English AND pair were not in the old SFT text ('clean' = old-unseen sentences)
  m385_sft_clean        385M SFT-final scored on exactly those sentences (reports/data/eval_385m_results_2026-11-03/vs_old270m_same_sentences_per_language.csv)
  m385_sft_all          385M SFT on all rows of the benchmark; stageA_all / b1_all = 385M after RL on ALL rows (no per-sentence output of the RL runs is archived on the laptop)
  R_sft  = m385_sft_clean - old_sft                       (regression vs old SFT, same sentences)
  R_rl_est = (b1_all + (m385_sft_clean - m385_sft_all)) - old_rl
            the second term moves the all-rows RL score onto the old-unseen sentence set by the SAME offset the SFT model shows between the two sets.  ESTIMATE, not measured.
  R_rl_all = b1_all - old_rl_all_rows (reported only for FLORES where the old model saw 95% of the English: contaminated, shown as a flagged column).
"""
import os, sys
import pandas as pd, numpy as np

H = os.path.dirname(os.path.abspath(__file__))
REP = os.path.abspath(os.path.join(H, "..", ".."))        # reports/
D = os.path.join(REP, "data")
E = os.path.join(D, "eval_385m_results_2026-11-03")
P = os.path.join(D, "eval_post_B1_2026-11-03", "results")
BM = {"NTREX-128": "ntrex", "FLORES+ devtest": "flores_plus", "MAFAND-MT test": "mafand", "SmolSent": "smol_sent", "SALT": "salt", "GlotStoryBook": "glot_story",
      "IbomMT": "ibom_mt", "FLORES-MOZ": "flores_moz"}

ss = pd.read_csv(os.path.join(E, "vs_old270m_same_sentences_per_language.csv"))
ss = ss[ss.subset == "clean"].copy()
ss["bm"] = ss.benchmark.map(BM)
ss = ss.rename(columns={"chrf_m385_oldsubset": "m385_sft_clean", "sft_chrf": "old_sft", "rl_chrf": "old_rl", "n_m385": "n_clean"})[["bm", "iso", "group", "n_clean", "m385_sft_clean", "old_sft", "old_rl"]]


def res(fn):
    r = pd.read_csv(os.path.join(P, fn))
    return r[r.subset == "all"][["benchmark", "iso", "n", "chrf", "finished_rate"]]


sft = res("results_sft.csv").rename(columns={"benchmark": "bm", "chrf": "m385_sft_all", "n": "n_all", "finished_rate": "fin_sft"})
A = pd.concat([res("results_stageA.csv"), res("results_stageA_x.csv")]).rename(columns={"benchmark": "bm", "chrf": "stageA_all", "finished_rate": "fin_A"}).drop(columns="n")
B = pd.concat([res("results_b1best.csv"), res("results_b1best_x.csv")]).rename(columns={"benchmark": "bm", "chrf": "b1_all", "finished_rate": "fin_B1"}).drop(columns="n")
a0 = pd.read_csv(os.path.join(E, "greedy_per_language_all_benchmarks.csv"))
a0 = a0[a0.subset == "all"][["benchmark", "iso", "chrf_a0", "chrf_m385", "n_a0"]].rename(columns={"benchmark": "bm", "chrf_a0": "a0_all"})
old_all = pd.read_csv(os.path.join(D, "eval_385m_plan_2026-11-02", "own_270m_per_language.csv"))
old_all = old_all[old_all.subset == "all"][["benchmark", "iso", "sft_chrf", "rl_chrf"]].rename(columns={"benchmark": "bm", "sft_chrf": "old_sft_allrows", "rl_chrf": "old_rl_allrows"})
# the plan file names benchmarks like the harness
old_all["bm"] = old_all["bm"].replace({v: v for v in BM.values()})

t = ss.merge(sft, on=["bm", "iso"], how="left").merge(A, on=["bm", "iso"], how="left").merge(B, on=["bm", "iso"], how="left")
t = t.merge(a0[["bm", "iso", "a0_all", "n_a0"]], on=["bm", "iso"], how="left").merge(old_all, on=["bm", "iso"], how="left")
off = t.m385_sft_clean - t.m385_sft_all
t["offset_clean_minus_all"] = off
t["R_sft"] = t.m385_sft_clean - t.old_sft
t["R_sft_vs_oldrl"] = t.m385_sft_clean - t.old_rl
t["stageA_clean_est"] = t.stageA_all + off
t["b1_clean_est"] = t.b1_all + off
t["R_A_est"] = t.stageA_clean_est - t.old_rl
t["R_rl_est"] = t.b1_clean_est - t.old_rl
t["R_rl_allrows"] = t.b1_all - t.old_rl_allrows
t["n_clean_ok"] = t.n_clean >= 70
t.to_csv(os.path.join(H, "eval_by_language_benchmark.csv"), index=False)
print(t.groupby("bm").agg(n=("iso", "count"), R_sft=("R_sft", "mean"), R_rl=("R_rl_est", "mean"), off=("offset_clean_minus_all", "mean")))
print(t[t.bm == "ntrex"][["iso", "n_clean", "m385_sft_clean", "old_sft", "R_sft", "R_rl_est", "a0_all", "m385_sft_all"]].round(1).to_string())
