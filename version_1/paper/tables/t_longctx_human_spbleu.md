### Table LC-H. spBLEU (flores200) against HUMAN references, macro over the common languages (6 SmolDoc languages hau yor swh amh zul ibo; 5 AfriScience languages amh hau lug yor zul), 95 % bootstrap CI over documents

| model | mode | SmolDoc short | stitched ~512 | stitched ~1,024 | stitched ~2,048 | AfriScience dev | AfriScience test |
|:---|---:|---:|---:|---:|---:|---:|---:|
| SabiYarn-270M SFT-final | whole document | 19.8 [17.9, 21.7] | 17.1 [15.1, 19.1] | 2.0 [1.5, 2.7] | 0.8 [0.6, 1.0] | 4.7 [3.5, 6.1] | 3.1 [2.2, 4.3] |
| SabiYarn-270M SFT-final | sentence-by-sentence + repair | 21.4 [20.4, 22.4] | 21.3 [20.1, 22.2] | 22.1 [21.5, 22.8] | 23.2 [22.4, 23.8] | 29.4 [28.0, 30.7] | 28.5 [27.4, 29.6] |
| SabiYarn-270M RL-final (2c) | whole document | 23.6 [22.3, 25.1] | 21.2 [19.8, 22.6] | 7.0 [5.7, 7.9] | 2.3 [1.9, 2.7] | 13.3 [9.8, 16.4] | 10.7 [8.8, 13.0] |
| SabiYarn-270M RL-final (2c) | sentence-by-sentence + repair | 24.2 [23.0, 25.4] | 24.1 [22.8, 25.1] | 24.8 [24.1, 25.4] | 26.2 [25.5, 26.8] | 29.2 [27.9, 30.5] | 28.0 [26.6, 29.1] |
| SabiYarn-270M science start (A2W a0.7) | whole document | 24.4 [23.1, 25.7] | 22.3 [20.8, 23.7] | 15.0 [14.1, 15.6] | 7.5 [5.9, 8.6] | 21.9 [19.8, 24.0] | 19.5 [17.2, 21.2] |
| SabiYarn-270M science start (A2W a0.7) | sentence-by-sentence + repair | 24.5 [23.5, 25.8] | 24.3 [23.1, 25.4] | 25.3 [24.5, 25.9] | 26.6 [25.7, 27.3] | 32.9 [31.3, 34.4] | 31.7 [30.2, 32.8] |
| SabiYarn-270M live run (step 250) | whole document | 24.4 [23.2, 25.6] | 22.4 [21.2, 23.6] | 14.6 [13.6, 15.5] | 6.7 [5.5, 7.6] | 21.6 [19.5, 23.9] | 18.9 [16.9, 20.9] |
| SabiYarn-270M live run (step 250) | sentence-by-sentence + repair | 24.8 [23.7, 26.0] | 24.5 [23.3, 25.6] | 25.4 [24.7, 26.1] | 26.7 [25.9, 27.4] | 32.8 [31.3, 34.3] | 31.7 [30.3, 32.9] |
| SabiYarn-270M final run (final2, step 401) | whole document | 24.5 [23.4, 25.7] | 22.1 [20.9, 23.3] | 14.9 [14.0, 15.6] | 6.7 [5.4, 7.8] | 21.2 [19.0, 23.7] | 18.8 [16.9, 20.6] |
| SabiYarn-270M final run (final2, step 401) | sentence-by-sentence + repair | 24.7 [23.6, 25.9] | 24.5 [23.3, 25.6] | 25.5 [24.7, 26.2] | 26.7 [25.8, 27.4] | 32.7 [31.2, 34.3] | 31.7 [30.3, 33.0] |
| NLLB-200-distilled-600M | whole document | 15.5 [12.8, 18.1] | 7.4 [6.4, 8.2] | 2.2 [1.7, 2.7] | 0.5 [0.3, 0.6] | 3.3 [2.8, 3.7] | 2.8 [2.2, 3.3] |
| NLLB-200-distilled-600M | sentence-by-sentence | 26.8 [25.5, 28.3] | 27.3 [26.2, 28.5] | 28.1 [27.1, 28.8] | 29.1 [28.6, 29.7] | 31.5 [29.8, 33.0] | 30.0 [27.9, 31.7] |
| NLLB-200-3.3B (CT2 int8) | whole document | 18.7 [15.1, 22.7] | 6.2 [4.9, 7.8] | 0.5 [0.4, 0.7] | 0.1 [0.0, 0.2] | 1.2 [1.0, 1.6] | 1.0 [0.7, 1.3] |
| NLLB-200-3.3B (CT2 int8) | sentence-by-sentence | 29.0 [27.4, 30.8] | 29.8 [28.4, 31.3] | 30.0 [29.0, 31.0] | 30.9 [30.0, 31.6] | 34.8 [33.2, 36.1] | 33.8 [31.9, 35.4] |
| TranslatePsy-AfriSLM-0.8B | whole document | 17.2 [13.6, 20.7] | 7.5 [5.6, 9.2] | 3.1 [1.9, 3.9] | 1.6 [1.0, 1.9] | 4.1 [3.4, 5.0] | 3.7 [3.1, 4.4] |
| TranslatePsy-AfriSLM-0.8B | sentence-by-sentence | 30.5 [29.1, 31.9] | 30.8 [29.4, 32.1] | 30.9 [30.0, 31.7] | 32.0 [31.3, 32.7] | 35.9 [34.2, 37.4] | 34.7 [33.3, 36.1] |
| TranslatePsy-AfriSLM-2B | whole document | n/a | n/a | 5.0 [3.2, 5.8] | 1.5 [0.4, 2.2] | 9.2 [7.1, 12.1] | n/a |
| TranslatePsy-AfriSLM-2B | sentence-by-sentence | n/a | n/a | 31.0 [29.9, 32.0] | 31.9 [30.9, 32.8] | 38.6 [36.8, 40.1] | n/a |
| Qwen3.5-0.8B (zero-shot) | whole document | n/a | n/a | n/a | n/a | n/a | n/a |
| Qwen3.5-0.8B (zero-shot) | sentence-by-sentence | 0.5 [0.4, 0.6] | 0.5 [0.4, 0.5] | 0.6 [0.5, 0.7] | 0.7 [0.5, 0.8] | 1.5 [1.1, 2.0] | 1.1 [0.9, 1.5] |
| Gemma-3-1B-it (zero-shot) | whole document | n/a | n/a | n/a | n/a | n/a | n/a |
| Gemma-3-1B-it (zero-shot) | sentence-by-sentence | n/a | n/a | 1.7 [1.5, 1.9] | 1.9 [1.7, 2.0] | 1.0 [0.6, 1.4] | n/a |

SmolDoc bins: 60 / 38 / 33 / 16 English documents; AfriScience dev 25 and test 28 papers. Stitched = unrelated clean SmolDoc documents concatenated (tests length, not discourse). `n/a` = the model was not run on that bin. AfriScience test was scored for all five SabiYarn checkpoints including the live-run step 250 (an intermediate checkpoint, no selection used it) and the final checkpoint (step 401, the last step of the same run, no selection used it): if the final test read must stay pristine, treat the live row's test column as dev-like. A2W and the live run were trained on the AfriScience TRAIN split (in-domain advantage); the live run selects checkpoints on dev. Document-level pooled chrF++ (about 10 points above the sentence-level numbers of the earlier AfriScience reports).
