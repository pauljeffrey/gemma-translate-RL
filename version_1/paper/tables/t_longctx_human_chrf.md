### Table LC-H. chrF++ against HUMAN references, macro over the common languages (6 SmolDoc languages hau yor swh amh zul ibo; 5 AfriScience languages amh hau lug yor zul), 95 % bootstrap CI over documents

| model | mode | SmolDoc short | stitched ~512 | stitched ~1,024 | stitched ~2,048 | AfriScience dev | AfriScience test |
|:---|---:|---:|---:|---:|---:|---:|---:|
| SabiYarn-270M SFT-final | whole document | 43.1 [41.2, 45.3] | 40.4 [38.1, 42.7] | 14.7 [13.0, 16.4] | 12.3 [11.2, 13.4] | 19.4 [17.0, 21.8] | 16.0 [13.8, 18.4] |
| SabiYarn-270M SFT-final | sentence-by-sentence + repair | 46.1 [45.1, 47.0] | 46.9 [46.0, 47.6] | 49.6 [49.0, 50.1] | 52.3 [51.8, 52.8] | 53.0 [51.7, 54.1] | 53.0 [51.9, 54.0] |
| SabiYarn-270M RL-final (2c) | whole document | 48.2 [47.1, 49.3] | 47.0 [45.6, 48.4] | 25.6 [23.9, 27.4] | 17.9 [16.6, 19.1] | 33.8 [30.5, 36.7] | 31.5 [29.2, 34.3] |
| SabiYarn-270M RL-final (2c) | sentence-by-sentence + repair | 49.0 [47.9, 49.9] | 49.7 [48.8, 50.6] | 52.1 [51.5, 52.6] | 55.1 [54.8, 55.5] | 53.3 [52.1, 54.3] | 53.0 [51.9, 54.1] |
| SabiYarn-270M science start (A2W a0.7) | whole document | 48.8 [47.8, 49.9] | 47.8 [46.5, 49.0] | 39.4 [38.2, 40.5] | 28.8 [26.7, 31.0] | 43.7 [41.3, 45.7] | 41.6 [39.3, 43.7] |
| SabiYarn-270M science start (A2W a0.7) | sentence-by-sentence + repair | 49.1 [48.1, 50.1] | 49.8 [49.0, 50.7] | 52.5 [51.9, 53.0] | 55.4 [54.9, 55.9] | 55.7 [54.4, 56.9] | 55.4 [54.1, 56.5] |
| SabiYarn-270M live run (step 250) | whole document | 49.0 [47.9, 50.0] | 47.9 [46.9, 49.0] | 39.6 [38.3, 40.8] | 28.2 [26.6, 29.9] | 43.8 [41.7, 45.8] | 41.5 [39.3, 43.6] |
| SabiYarn-270M live run (step 250) | sentence-by-sentence + repair | 49.3 [48.3, 50.3] | 50.0 [49.2, 50.9] | 52.6 [52.1, 53.1] | 55.5 [55.1, 56.0] | 55.6 [54.3, 56.8] | 55.4 [54.1, 56.5] |
| SabiYarn-270M final run (final2, step 401) | whole document | 49.1 [48.0, 50.1] | 47.8 [46.7, 48.9] | 39.6 [38.3, 40.7] | 27.7 [25.8, 29.7] | 43.2 [40.7, 45.3] | 41.3 [39.2, 43.3] |
| SabiYarn-270M final run (final2, step 401) | sentence-by-sentence + repair | 49.3 [48.2, 50.3] | 50.0 [49.1, 50.9] | 52.6 [52.1, 53.2] | 55.5 [55.0, 56.0] | 55.6 [54.3, 56.8] | 55.4 [54.1, 56.6] |
| NLLB-200-distilled-600M | whole document | 34.8 [31.4, 38.4] | 23.0 [21.8, 24.4] | 12.4 [11.4, 13.5] | 8.5 [7.8, 9.1] | 11.5 [10.4, 12.7] | 10.8 [9.9, 11.7] |
| NLLB-200-distilled-600M | sentence-by-sentence | 50.5 [49.3, 51.6] | 51.6 [50.6, 52.6] | 53.9 [53.2, 54.5] | 56.6 [56.2, 57.0] | 55.1 [53.8, 56.5] | 54.6 [53.1, 55.8] |
| NLLB-200-3.3B (CT2 int8) | whole document | 39.1 [35.2, 43.5] | 23.5 [21.7, 25.6] | 9.7 [9.0, 10.5] | 5.9 [5.1, 6.6] | 9.8 [8.8, 10.9] | 8.4 [7.4, 9.4] |
| NLLB-200-3.3B (CT2 int8) | sentence-by-sentence | 51.7 [50.5, 52.9] | 53.1 [52.1, 54.0] | 55.0 [54.4, 55.7] | 57.6 [57.2, 58.0] | 57.1 [55.8, 58.4] | 56.9 [55.5, 58.1] |
| TranslatePsy-AfriSLM-0.8B | whole document | 35.5 [31.5, 40.5] | 22.9 [20.2, 25.7] | 11.7 [9.9, 13.3] | 8.1 [6.2, 9.5] | 18.3 [16.9, 19.7] | 17.9 [16.9, 19.0] |
| TranslatePsy-AfriSLM-0.8B | sentence-by-sentence | 53.2 [52.1, 54.3] | 54.2 [53.1, 55.2] | 56.0 [55.3, 56.6] | 58.7 [58.3, 59.0] | 57.9 [56.5, 59.2] | 57.6 [56.2, 58.8] |
| TranslatePsy-AfriSLM-2B | whole document | n/a | n/a | 14.9 [12.0, 17.3] | 7.5 [5.1, 9.6] | 23.3 [20.7, 25.7] | n/a |
| TranslatePsy-AfriSLM-2B | sentence-by-sentence | n/a | n/a | 55.7 [55.0, 56.5] | 58.3 [57.9, 58.7] | 58.8 [57.4, 60.2] | n/a |
| Qwen3.5-0.8B (zero-shot) | whole document | n/a | n/a | n/a | n/a | n/a | n/a |
| Qwen3.5-0.8B (zero-shot) | sentence-by-sentence | 10.4 [9.9, 10.9] | 11.2 [10.8, 11.5] | 12.6 [12.4, 12.8] | 13.7 [13.4, 14.0] | 10.4 [10.0, 10.8] | 10.0 [9.7, 10.3] |
| Gemma-3-1B-it (zero-shot) | whole document | n/a | n/a | n/a | n/a | n/a | n/a |
| Gemma-3-1B-it (zero-shot) | sentence-by-sentence | n/a | n/a | 21.3 [21.1, 21.5] | 23.3 [23.1, 23.5] | 14.3 [14.0, 14.8] | n/a |

SmolDoc bins: 60 / 38 / 33 / 16 English documents; AfriScience dev 25 and test 28 papers. Stitched = unrelated clean SmolDoc documents concatenated (tests length, not discourse). `n/a` = the model was not run on that bin. AfriScience test was scored for all five SabiYarn checkpoints including the live-run step 250 (an intermediate checkpoint, no selection used it) and the final checkpoint (step 401, the last step of the same run, no selection used it): if the final test read must stay pristine, treat the live row's test column as dev-like. A2W and the live run were trained on the AfriScience TRAIN split (in-domain advantage); the live run selects checkpoints on dev. Document-level pooled chrF++ (about 10 points above the sentence-level numbers of the earlier AfriScience reports).
