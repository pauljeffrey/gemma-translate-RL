### Table M2 (clean). SSA-COMET-MTL (ref-based, McGill-NLP/ssa-comet-mtl, 0-1; only the 22 languages of Afrique-21 and TPsy-19 were scored), macro over language sets, computed on identical sentences (clean)

| Model | all45 | nllb43 | afrique21 | tpsy19 | tpsy2b9 | comet_scored42 | africomet_encoder16 | afrimte9 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SabiYarn-270M RL-final (GRPO step 1433) | n/a (22/45) | n/a (22/43) | 0.445 [0.440, 0.451] | 0.448 [0.443, 0.455] | 0.429 [0.422, 0.438] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| SabiYarn-270M SFT-final (step 27638) | n/a (22/45) | n/a (22/43) | 0.396 [0.389, 0.403] | 0.406 [0.399, 0.413] | 0.380 [0.369, 0.391] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| SabiYarn-270M science-stage start (A2W alpha 0.7) | n/a (22/45) | n/a (22/43) | 0.439 [0.434, 0.445] | 0.445 [0.439, 0.451] | 0.427 [0.418, 0.435] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| SabiYarn-270M live science run (step 250, 2026-10-23) | n/a (22/45) | n/a (22/43) | 0.444 [0.438, 0.450] | 0.449 [0.443, 0.455] | 0.430 [0.421, 0.439] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| SabiYarn-270M final run (final2 step 401) | n/a (22/45) | n/a (22/43) | 0.442 [0.437, 0.448] | 0.448 [0.442, 0.453] | 0.429 [0.420, 0.438] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| NLLB-200-distilled-600M (re-run: CT2 fp16, beam 4) | n/a (22/45) | n/a (22/43) | 0.482 [0.475, 0.489] | 0.484 [0.478, 0.491] | 0.487 [0.478, 0.498] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| NLLB-200-3.3B (re-run: CT2 int8, beam 4) | n/a (22/45) | n/a (22/43) | 0.520 [0.513, 0.526] | 0.520 [0.513, 0.526] | 0.521 [0.511, 0.531] | n/a (22/42) | n/a (15/16) | n/a (7/9) |
| TranslatePsy-AfriSLM-0.8B (re-run: zero-shot, greedy) | n/a (19/45) | n/a (19/43) | n/a (18/21) | 0.535 [0.529, 0.541] | 0.535 [0.526, 0.545] | n/a (19/42) | n/a (15/16) | n/a (6/9) |
| TranslatePsy-AfriSLM-2B (re-run: zero-shot, greedy) | n/a (9/45) | n/a (9/43) | n/a (8/21) | n/a (9/19) | 0.550 [0.542, 0.558] | n/a (9/42) | n/a (7/16) | n/a (2/9) |

set sizes: all45=45, nllb43=43, afrique21=21, tpsy19=19, tpsy2b9=9, comet_scored42=42, africomet_encoder16=16, afrimte9=9
