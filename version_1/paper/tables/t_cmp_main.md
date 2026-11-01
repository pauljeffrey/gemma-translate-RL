| Model | params | chrF++ NLLB-43 | AfriCOMET 42 lang. | chrF++ TPsy-19 | AfriCOMET TPsy-19 | chrF++ Afrique-21 | SSA-COMET Afrique-21 (ours-scored) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| SabiYarn-270M RL-final (GRPO step 1433) | 270M | 34.7 | 0.620 | 41.2 | 0.657 | 39.1 | 52.1 |
| SabiYarn-270M SFT-final (step 27638) | 270M | 29.5 | 0.559 | 38.2 | 0.613 | 36.1 | 47.6 |
| SabiYarn-270M science-stage start (A2W alpha 0.7) | 270M | 34.8 | 0.615 | 41.6 | 0.653 | 39.5 | 51.8 |
| SabiYarn-270M live science run (step 250, 2026-10-23) | 270M | 35.0 | 0.619 | 41.6 | 0.654 | 39.5 | 52.0 |
| SabiYarn-270M final run (final2 step 401) | 270M | 34.9 | 0.618 | 41.6 | 0.654 | 39.5 | 52.0 |
| NLLB-200-distilled-600M (re-run: CT2 fp16, beam 4) | 615M | 36.4 | 0.654 | 42.6 | 0.688 | 40.7 | 56.7 |
| NLLB-200-3.3B (re-run: CT2 int8, beam 4) | 3.3B | 38.4 | 0.669 | 44.7 | 0.701 | 42.8 | 59.3 |
| TranslatePsy-AfriSLM-0.8B (re-run: zero-shot, greedy) | 800M | n/a (19/43) | n/a (19/42) | 45.3 | 0.717 | n/a (18/21) | n/a (18/21) |
| NLLB-200-3.3B (published, whole devtest) | 3.3B | 38.8 | n/a | 45.2 | n/a | 43.3 | n/a |
| AfriqueQwen3.5-4B-50Langs (published, whole devtest) | 4B | n/a (21/43) | n/a | n/a (18/19) | n/a | 41.7 | n/a |
| AfriqueGemma-4B (published, whole devtest) | 4.3B | n/a (21/43) | n/a | n/a (18/19) | n/a | 31.1 | n/a |
| AfriqueLlama-8B (published, whole devtest) | 8B | n/a (21/43) | n/a | n/a (18/19) | n/a | 36.7 | n/a |
| AfriqueGemma-12B (published, whole devtest) | 12B | n/a (21/43) | n/a | n/a (18/19) | n/a | 38.6 | n/a |
| AfriqueQwen-14B (published, whole devtest) | 14.8B | n/a (21/43) | n/a | n/a (18/19) | n/a | 36.7 | n/a |
| NLLB-200-MoE-54.5B (published, whole devtest) | 54.5B | 39.1 | n/a | 45.5 | n/a | 43.7 | n/a |
