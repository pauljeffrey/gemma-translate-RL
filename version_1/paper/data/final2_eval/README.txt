Evaluation of the final checkpoint (final2, step 401, Hub <HF_ORG_LEGACY>/gemma-science-translate tag final2-step401), 2026-10-23, agent P5.
macro_bench.csv / paired_bench.csv / per_language_bench.csv / sentence_wtl_final2_vs_rl.csv : FLORES+ / NTREX-128 / MAFAND-MT / SmolSent, SFT-final vs RL-final vs final2 (scripts/box_p5_bench_compare.py)
afs_macro.csv / afs_paired.csv : AfriScience-MT dev and test, sentence and paper level (scripts/box_p5_afs_compare.py). Systems: rl_a4 = RL-final through eval_afriscience.py (a4/eval_final1433), rl_r7 / a2w = stage evaluation of paper Section 6.6 (r7/eval/final_1433*, A2W*), final2 = a4/eval_final2_step401.
afriscience_paper_length_ratio.csv : character ratio hyp/ref of the paper-level outputs (computed on the box with a 15-line script, numbers copied here).
e2_flores_final2_report.md / e2_other_final2_report.md : reports written by the evaluation script; the system label in them is 'SFT' (the script's generic label), the model is final2.
afriscience_eval_final2_step401.{md,json}, afriscience_eval_final1433.md : output of eval_afriscience.py.
The FLORES+ AfriCOMET / SSA-COMET aggregation is in reports/data/model_comparison_computed/*_with_final2.csv (and the same files in paper/data/model_comparison_computed/).
