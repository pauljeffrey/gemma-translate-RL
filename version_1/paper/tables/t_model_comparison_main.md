### Table M1. One table: chrF++ and AfriCOMET, English -> X, FLORES+ devtest

| Model | Params | chrF++ NLLB-43 | chrF++ Afrique-21 | chrF++ TPsy-19 | AfriCOMET (ours) 42 lang. | AfriCOMET (ours) TPsy-19 | SSA-COMET Afrique-21, ours-scored x100 | SSA-COMET Afrique-21, paper's | Source |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| SabiYarn-270M RL-final (GRPO step 1433) | 270M | 34.7 [34.6, 34.8] | 39.1 [39.0, 39.3] | 41.2 [41.1, 41.4] | 0.620 [0.619, 0.621] | 0.657 [0.655, 0.659] | 52.1 [52.0, 52.3] | n/a | computed (same sentences, no_pair) |
| SabiYarn-270M SFT-final (step 27638) | 270M | 29.5 [29.3, 29.6] | 36.1 [35.8, 36.4] | 38.2 [38.0, 38.5] | 0.559 [0.558, 0.560] | 0.613 [0.611, 0.615] | 47.6 [47.4, 47.8] | n/a | computed (same sentences, no_pair) |
| SabiYarn-270M science-stage start (A2W alpha 0.7) | 270M | 34.8 [34.7, 34.9] | 39.5 [39.4, 39.7] | 41.6 [41.4, 41.8] | 0.615 [0.613, 0.616] | 0.653 [0.651, 0.654] | 51.8 [51.6, 52.0] | n/a | computed (same sentences, no_pair) |
| SabiYarn-270M live science run (step 250, 2026-10-23) | 270M | 35.0 [34.9, 35.1] | 39.5 [39.4, 39.7] | 41.6 [41.5, 41.8] | 0.619 [0.618, 0.620] | 0.654 [0.652, 0.655] | 52.0 [51.8, 52.2] | n/a | computed (same sentences, no_pair) |
| SabiYarn-270M final run (final2 step 401) | 270M | 34.9 [34.8, 35.0] | 39.5 [39.4, 39.7] | 41.6 [41.4, 41.8] | 0.618 [0.617, 0.619] | 0.654 [0.652, 0.655] | 52.0 [51.8, 52.1] | n/a | computed (same sentences, no_pair) |
| NLLB-200-distilled-600M (re-run: CT2 fp16, beam 4) | 615M | 36.4 [36.3, 36.5] | 40.7 [40.5, 40.8] | 42.6 [42.4, 42.8] | 0.654 [0.653, 0.655] | 0.688 [0.686, 0.690] | 56.7 [56.5, 56.9] | n/a | computed (same sentences, no_pair) |
| NLLB-200-3.3B (re-run: CT2 int8, beam 4) | 3.3B | 38.4 [38.3, 38.5] | 42.8 [42.7, 43.0] | 44.7 [44.4, 44.9] | 0.669 [0.668, 0.671] | 0.701 [0.699, 0.702] | 59.3 [59.1, 59.4] | n/a | computed (same sentences, no_pair) |
| TranslatePsy-AfriSLM-0.8B (re-run: zero-shot, greedy) | 800M | n/a (19/43) | n/a (18/21) | 45.3 [45.1, 45.5] | n/a (19/42) | 0.717 [0.715, 0.718] | n/a (18/21) | n/a | computed (same sentences, no_pair) |
| TranslatePsy-AfriSLM-2B (re-run: zero-shot, greedy) | 2B | n/a (9/43) | n/a (8/21) | n/a (9/19) | n/a (9/42) | n/a (9/19) | n/a (8/21) | n/a | computed (same sentences, no_pair) |
| NLLB-200-1.3B | 1.3B | 38.1 | 42.3 | 44.2 | n/a | n/a | n/a | n/a | published (paper's own pipeline, whole devtest) |
| NLLB-200-distilled-1.3B | 1.3B | 38.3 | 42.6 | 44.5 | n/a | n/a | n/a | n/a | published (paper's own pipeline, whole devtest) |
| NLLB-200-3.3B | 3.3B | 38.8 | 43.3 | 45.2 | n/a | n/a | n/a | n/a | published (paper's own pipeline, whole devtest) |
| Qwen3-4B | 4B | n/a (21/43) | 9.9 | n/a (18/19) | n/a | n/a | n/a | 15.8 | published (paper's own pipeline, whole devtest) |
| AfriqueQwen-4B | 4B | n/a (21/43) | 33.1 | n/a (18/19) | n/a | n/a | n/a | 51.8 | published (paper's own pipeline, whole devtest) |
| Qwen3.5-4B | 4B | n/a (21/43) | 18.8 | n/a (18/19) | n/a | n/a | n/a | 26.2 | published (paper's own pipeline, whole devtest) |
| AfriqueQwen3.5-4B | 4B | n/a (21/43) | 36.6 | n/a (18/19) | n/a | n/a | n/a | 55.2 | published (paper's own pipeline, whole devtest) |
| AfriqueQwen3.5-4B-ExtendedCM | 4B | n/a (21/43) | 36.7 | n/a (18/19) | n/a | n/a | n/a | 55.2 | published (paper's own pipeline, whole devtest) |
| AfriqueQwen3.5-4B-50Langs | 4B | n/a (21/43) | 41.7 | n/a (18/19) | n/a | n/a | n/a | 62.9 | published (paper's own pipeline, whole devtest) |
| Gemma3-4B | 4.3B | n/a (21/43) | 14.8 | n/a (18/19) | n/a | n/a | n/a | 32.7 | published (paper's own pipeline, whole devtest) |
| AfriqueGemma-4B | 4.3B | n/a (21/43) | 31.1 | n/a (18/19) | n/a | n/a | n/a | 51.3 | published (paper's own pipeline, whole devtest) |
| Llama3.1-8B | 8B | n/a (21/43) | 16.4 | n/a (18/19) | n/a | n/a | n/a | 21.6 | published (paper's own pipeline, whole devtest) |
| Lugha-Llama-8B-wura | 8B | n/a (21/43) | 27.4 | n/a (18/19) | n/a | n/a | n/a | 43.5 | published (paper's own pipeline, whole devtest) |
| AfriqueLlama-8B | 8B | n/a (21/43) | 36.7 | n/a (18/19) | n/a | n/a | n/a | 55.4 | published (paper's own pipeline, whole devtest) |
| Qwen3-8B | 8.2B | n/a (21/43) | 11.3 | n/a (18/19) | n/a | n/a | n/a | 16.0 | published (paper's own pipeline, whole devtest) |
| AfriqueQwen-8B | 8.2B | n/a (21/43) | 35.1 | n/a (18/19) | n/a | n/a | n/a | 53.9 | published (paper's own pipeline, whole devtest) |
| Gemma3-12B | 12B | n/a (21/43) | 18.8 | n/a (18/19) | n/a | n/a | n/a | 40.5 | published (paper's own pipeline, whole devtest) |
| AfriqueGemma-12B | 12B | n/a (21/43) | 38.6 | n/a (18/19) | n/a | n/a | n/a | 58.6 | published (paper's own pipeline, whole devtest) |
| Qwen3-14B | 14.8B | n/a (21/43) | 14.6 | n/a (18/19) | n/a | n/a | n/a | 18.9 | published (paper's own pipeline, whole devtest) |
| AfriqueQwen-14B | 14.8B | n/a (21/43) | 36.7 | n/a (18/19) | n/a | n/a | n/a | 55.7 | published (paper's own pipeline, whole devtest) |
| NLLB-200-MoE-54.5B | 54.5B | 39.1 | 43.7 | 45.5 | n/a | n/a | n/a | n/a | published (paper's own pipeline, whole devtest) |

Rows 'computed': our own scoring on identical sentences (FLORES+ devtest `no_pair` subset: the sentences whose reference pair our SFT never saw; macro over the language set, 95 % bootstrap CI over sentences, no language-sampling noise; AfriCOMET = reference-based `masakhane/africomet-stl-1.1`, scale 0-1). Rows 'published': the paper's own chrF++ on the whole FLORES-200 devtest (5-shot base models and encoder-decoder MT; different prompts, decoding and, for the Afrique tables, an unspecified split), SSA-COMET = a different metric (paper's checkpoint unnamed; 0-100), so published and computed cells are never mixed in one column of a macro. 'n/a (k/K)': the system does not cover all K languages of the set (k covered); we do not average over a different language set. NLLB-43 = the 43 FLORES+ languages with a published NLLB number; Afrique-21 = the 21 languages of the AfriqueLLM tables that are in FLORES+; TPsy-19 = the 19 TranslatePsy-AfriSLM languages. AfriCOMET n/a for kea, knc, vmw (not on the AfriCOMET-STL-1.1 card), so 42 of the 45 FLORES+ languages are scored.
