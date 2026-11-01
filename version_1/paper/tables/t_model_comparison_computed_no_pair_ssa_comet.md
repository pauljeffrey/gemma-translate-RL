### Table M2 (no_pair). SSA-COMET-MTL (ref-based, McGill-NLP/ssa-comet-mtl, 0-1; only the 22 languages of Afrique-21 and TPsy-19 were scored), macro over language sets, computed on identical sentences (no_pair)

| Model | all45 | nllb43 | afrique21 | tpsy19 | tpsy2b9 | comet_scored42 | africomet_encoder16 | afrimte9 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SabiYarn-270M RL-final (GRPO step 1433) | n/a (22/45) | n/a (22/43) | 0.521 [0.520, 0.523] | 0.528 [0.526, 0.530] | 0.503 [0.500, 0.506] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| SabiYarn-270M SFT-final (step 27638) | n/a (22/45) | n/a (22/43) | 0.476 [0.474, 0.478] | 0.488 [0.486, 0.490] | 0.462 [0.459, 0.465] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| SabiYarn-270M science-stage start (A2W alpha 0.7) | n/a (22/45) | n/a (22/43) | 0.518 [0.516, 0.520] | 0.525 [0.523, 0.527] | 0.500 [0.497, 0.503] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| SabiYarn-270M live science run (step 250, 2026-10-23) | n/a (22/45) | n/a (22/43) | 0.520 [0.518, 0.522] | 0.527 [0.525, 0.528] | 0.502 [0.499, 0.505] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| SabiYarn-270M final run (final2 step 401) | n/a (22/45) | n/a (22/43) | 0.520 [0.518, 0.521] | 0.526 [0.525, 0.528] | 0.501 [0.498, 0.504] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| NLLB-200-distilled-600M (re-run: CT2 fp16, beam 4) | n/a (22/45) | n/a (22/43) | 0.567 [0.565, 0.569] | 0.567 [0.565, 0.569] | 0.569 [0.566, 0.572] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| NLLB-200-3.3B (re-run: CT2 int8, beam 4) | n/a (22/45) | n/a (22/43) | 0.593 [0.591, 0.594] | 0.591 [0.590, 0.593] | 0.589 [0.586, 0.592] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| TranslatePsy-AfriSLM-0.8B (re-run: zero-shot, greedy) | n/a (19/45) | n/a (19/43) | n/a (18/21) | 0.608 [0.606, 0.609] | 0.603 [0.601, 0.606] | n/a (19/42) | n/a (15/16) | n/a (6/9) |
| TranslatePsy-AfriSLM-2B (re-run: zero-shot, greedy) | n/a (9/45) | n/a (9/43) | n/a (8/21) | n/a (9/19) | 0.604 [0.602, 0.607] | n/a (9/42) | n/a (7/16) | n/a (2/9) |

set sizes: all45=45, nllb43=43, afrique21=21, tpsy19=19, tpsy2b9=9, comet_scored42=42, africomet_encoder16=16, afrimte9=9
