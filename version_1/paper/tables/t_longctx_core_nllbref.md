### Table LC-N. chrF++ of AfriLongDoc core documents against the NLLB-3.3B sentence-by-sentence pseudo-reference (6 languages hau yor swh amh zul ibo), 95 % bootstrap CI over documents

| model | mode | NLLB-lineage of the model | b256 (control) | b512 | b1024 | b2048 | b4096 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| SabiYarn-270M SFT-final | whole document | possible | 49.5 [47.0, 52.4] | 41.9 [39.3, 44.7] | 18.3 [16.8, 20.2] | 11.3 [10.4, 12.6] | 7.3 [6.5, 8.2] |
| SabiYarn-270M SFT-final | sentence-by-sentence + repair | possible | 52.2 [49.8, 54.8] | 56.6 [55.5, 57.9] | 58.6 [57.3, 59.8] | 60.7 [59.8, 61.5] | 62.2 [60.7, 63.7] |
| SabiYarn-270M RL-final (2c) | whole document | possible | 54.1 [52.3, 55.8] | 55.1 [53.7, 56.5] | 34.3 [31.4, 37.1] | 18.9 [17.8, 20.5] | 11.7 [10.2, 12.9] |
| SabiYarn-270M RL-final (2c) | sentence-by-sentence + repair | possible | 54.7 [53.1, 56.4] | 59.2 [58.1, 60.3] | 59.5 [58.4, 60.6] | 62.4 [61.5, 63.5] | 64.4 [63.3, 65.7] |
| SabiYarn-270M science start (A2W a0.7) | whole document | possible | 54.4 [52.3, 56.3] | 55.4 [53.9, 57.0] | 45.6 [43.9, 47.1] | 33.0 [31.5, 35.1] | 18.8 [17.2, 20.6] |
| SabiYarn-270M science start (A2W a0.7) | sentence-by-sentence + repair | possible | 55.0 [53.2, 56.7] | 59.3 [58.1, 60.4] | 60.0 [58.8, 61.2] | 63.0 [62.2, 63.9] | 64.9 [63.7, 66.2] |
| SabiYarn-270M live run (step 250) | whole document | possible | 54.8 [52.9, 56.7] | 56.5 [54.9, 58.0] | 46.1 [44.2, 47.8] | 32.1 [31.0, 33.4] | 16.0 [15.0, 16.9] |
| SabiYarn-270M live run (step 250) | sentence-by-sentence + repair | possible | 55.2 [53.4, 56.8] | 59.5 [58.3, 60.7] | 60.1 [58.9, 61.4] | 63.0 [62.2, 63.9] | 65.0 [63.8, 66.3] |
| SabiYarn-270M final run (final2, step 401) | whole document | possible | 54.6 [52.9, 56.7] | 56.3 [55.2, 57.8] | 45.7 [43.9, 47.3] | 32.4 [30.7, 34.5] | 16.0 [14.2, 18.1] |
| SabiYarn-270M final run (final2, step 401) | sentence-by-sentence + repair | possible | 55.0 [53.3, 56.7] | 59.5 [58.3, 60.7] | 60.2 [59.0, 61.4] | 63.1 [62.3, 64.0] | 65.0 [63.8, 66.3] |
| NLLB-200-distilled-600M | whole document | same family | 54.4 [51.0, 58.1] | 24.6 [22.8, 26.4] | 14.5 [13.4, 15.7] | 8.9 [7.7, 9.8] | n/a |
| NLLB-200-distilled-600M | sentence-by-sentence | same family | 65.1 [63.2, 67.2] | 69.3 [68.0, 70.6] | 70.3 [69.2, 71.5] | 72.4 [71.6, 73.4] | 73.8 [72.6, 75.0] |
| TranslatePsy-AfriSLM-0.8B | whole document | distilled from NLLB-3.3B | 52.3 [45.5, 58.9] | 23.4 [20.2, 26.3] | 13.3 [11.8, 14.8] | 8.3 [6.8, 9.6] | 5.2 [3.6, 7.0] |
| TranslatePsy-AfriSLM-0.8B | sentence-by-sentence | distilled from NLLB-3.3B | 68.0 [66.1, 70.1] | 72.0 [70.6, 73.3] | 71.7 [70.1, 73.1] | 74.6 [73.6, 75.9] | 75.7 [73.8, 77.2] |
| TranslatePsy-AfriSLM-2B | whole document | distilled from NLLB-3.3B | n/a | n/a | n/a | 8.1 [5.8, 10.3] | n/a |
| Qwen3.5-0.8B (zero-shot) | whole document | no | n/a | n/a | n/a | n/a | n/a |
| Qwen3.5-0.8B (zero-shot) | sentence-by-sentence | no | 9.2 [8.6, 9.8] | 12.3 [11.7, 13.2] | 13.6 [12.8, 14.6] | 14.6 [14.2, 15.1] | n/a |
| Gemma-3-1B-it (zero-shot) | whole document | no | n/a | n/a | n/a | n/a | n/a |

BIASED reference: NLLB-3.3B is scored against itself nowhere (row omitted); NLLB-600M is the same model family, TranslatePsy-AfriSLM was distilled from NLLB-3.3B translations, and the SabiYarn models may share NLLB-derived training data (R6 found NLLB-referenced scores 1-16 chrF++ above human-referenced ones). Use this table for the ORDER of systems/modes, not for absolute quality. Documents are 20 per bin (10 at b4096), the same English documents for every language and model; NLLB whole-document runs have no b4096.
