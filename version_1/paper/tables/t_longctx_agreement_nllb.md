### Table LC-A. Agreement with NLLB-3.3B versus quality against humans (same documents, sentence-by-sentence mode)

| model | mode | stitched ~1,024: vs human | vs NLLB-3.3B | gap | AfriScience dev: vs human | vs NLLB-3.3B | gap |
|:---|---:|---:|---:|---:|---:|---:|---:|
| SabiYarn-270M SFT-final | sentence-by-sentence + repair | 49.6 | 56.1 | +6.6 | 53.0 | 56.8 | +3.8 |
| SabiYarn-270M RL-final (2c) | sentence-by-sentence + repair | 52.1 | 60.7 | +8.6 | 53.3 | 57.8 | +4.5 |
| SabiYarn-270M science start (A2W a0.7) | sentence-by-sentence + repair | 52.5 | 60.3 | +7.8 | 55.7 | 57.8 | +2.2 |
| SabiYarn-270M live run (step 250) | sentence-by-sentence + repair | 52.6 | 60.6 | +8.0 | 55.6 | 57.9 | +2.3 |
| SabiYarn-270M final run (final2, step 401) | sentence-by-sentence + repair | 52.6 | 60.6 | +8.0 | 55.6 | 57.9 | +2.3 |
| NLLB-200-distilled-600M | sentence-by-sentence | 53.9 | 69.4 | +15.6 | 55.1 | 67.9 | +12.8 |
| TranslatePsy-AfriSLM-0.8B | sentence-by-sentence | 56.0 | 73.4 | +17.5 | 57.9 | 73.2 | +15.3 |
| TranslatePsy-AfriSLM-2B | sentence-by-sentence | 55.7 | 74.8 | +19.1 | 58.8 | 74.5 | +15.6 |
| Qwen3.5-0.8B (zero-shot) | sentence-by-sentence | 12.6 | 12.7 | +0.1 | 10.4 | 10.8 | +0.4 |
| Gemma-3-1B-it (zero-shot) | sentence-by-sentence | 21.3 | 21.8 | +0.4 | 14.3 | 14.6 | +0.3 |

gap = chrF++ against the NLLB-3.3B outputs minus chrF++ against the human references. A large positive gap = the system resembles NLLB more than it resembles a human translator.
