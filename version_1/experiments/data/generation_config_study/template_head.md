# Generation-config study for the final model: temperature, top-p, top-k, min-p, typical / epsilon sampling, best-of-n and MBR, at sentence level, document level and in RL rollouts (2026-10-22)

Task R12 / D-64. Question from the owner: which generation configuration should be used for short text, long documents, RL rollouts and as the fast config, given that greedy decoding is "very repetitive". Model: `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c@latest` (RL final, step 1433). Builds on `reports/final_model_decoding_study_2026-10-21.md` (R5: sentence level, greedy / beam / repetition penalty / n-gram blocking / MBR) and `reports/longdoc_eval_final_model_2026-10-21.md` (R6: documents). Data, box scripts and JSON: `reports/data/generation_config_study/` (`export/*.csv`, `json/*.json`, `scripts/`); renderer `gemma_mt/scripts/build_generation_config_report.py`. Nothing was pushed; scratch on the box was deleted at the end.

## 0. Answer

{{answer}}

{{decision_table}}

