# experiments/

Results of the clean-rerun generation (everything from 2026-10-26 onwards). The experiments of the first generation are in `../version_1/experiments/`.

| Folder | Content |
|---|---|
| `reports/` | written reports: contamination audit, pre-registration, clean rebuild, catalogue audit, instruction intake, NLLB and parallel-pair intake, clean calibration, clean RL and science-stage plans, lean schedule |
| `data/` | result tables behind the reports (CSV and JSON): token tables per language, NLLB usage, GlotLID precision/recall before and after the label fixes, fastText per-language results, fluency-model validation, tag token map, protected-set statistics |
| `growth1/` | training logs of the SFT/growth run: `metrics_a0u_up6.jsonl` (270M-class run), `metrics_g1a.jsonl` (stage A), `metrics_g1b_s1.jsonl`, `metrics_g1b_s2.jsonl` (stage B shards), `alerts.txt` (launcher events) |

Each metrics file has one JSON object per logged step; evaluation rows carry `val_loss`, `val_loss_macro` and `val_loss/<iso>` per language.
