# experiments/

Results of the clean-rerun generation (everything from 2026-10-26 onwards). The experiments of the first generation are in `../version_1/experiments/`.

| Folder | Content |
|---|---|
| `reports/` | written reports: contamination audit, pre-registration, clean rebuild, catalogue audit, instruction intake, NLLB and parallel-pair intake, clean calibration, clean RL and science-stage plans, lean schedule; from 2026-11-02 the 385M reports: RL stage plan and preflight, stage A and B1 results, evaluation plan and results on FLORES+, NTREX, MAFAND, SMOL and AfriScience-MT, contamination audit and follow-ups, weak-language, family/script and NLLB analyses, the post-editor experiments, the science and long-document stages, and the controls E-01 to E-14 (decode-time loop guard, beam and MBR, RL seeds and ablations, engine parity, fp32, SFT control, old models in the same harness, science RL without science SFT) |
| `data/` | result tables behind the reports (CSV and JSON) and the small scripts that produced them: token tables per language, NLLB usage, GlotLID precision/recall before and after the label fixes, fastText per-language results, fluency-model validation, tag token map, protected-set statistics; one folder per 385M report (`eval_385m_results_2026-11-03/`, `e03_control_execution_2026-11-04/`, ...). Per-sentence prediction dumps, model files, parquet and archives are not included |
| `growth1/` | training logs of the SFT/growth run: `metrics_a0u_up6.jsonl` (270M-class run), `metrics_g1a.jsonl` (stage A), `metrics_g1b_s1.jsonl`, `metrics_g1b_s2.jsonl` (stage B shards), `alerts.txt` (launcher events) |

Each metrics file has one JSON object per logged step; evaluation rows carry `val_loss`, `val_loss_macro` and `val_loss/<iso>` per language.
