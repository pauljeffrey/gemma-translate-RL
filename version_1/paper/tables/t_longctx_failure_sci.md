### Table LC-F-sci. Failure metrics, macro over languages (sci set)

finished = stopped at its own end token / before the length limit; loop = word-cycle repetition absent from the reference; coverage = share of source sentences with an aligned, non-copied output sentence (alignment plan of the long-document reward v2); len. ratio = hypothesis / reference characters; LID = share of output sentences (>= 25 characters) that GlotLID labels as the target language.

| model | mode | AfriScience dev: finished / loop / coverage / len. ratio / LID | AfriScience test: finished / loop / coverage / len. ratio / LID |
|:---|---:|---:|---:|
| SabiYarn-270M SFT-final | whole document | 0.17 / 0.79 / 0.76 / 1.45 / 0.90 | 0.11 / 0.89 / 0.69 / 1.50 / 0.90 |
| SabiYarn-270M SFT-final | sentence-by-sentence + repair | 0.96 / 0.02 / 0.97 / 0.99 / 0.99 | 0.95 / 0.01 / 0.96 / 1.00 / 0.99 |
| SabiYarn-270M RL-final (2c) | whole document | 0.67 / 0.42 / 0.93 / 1.03 / 0.99 | 0.71 / 0.47 / 0.92 / 0.99 / 0.99 |
| SabiYarn-270M RL-final (2c) | sentence-by-sentence + repair | 0.94 / 0.00 / 1.00 / 1.00 / 0.99 | 0.96 / 0.01 / 1.00 / 1.00 / 0.99 |
| SabiYarn-270M science start (A2W a0.7) | whole document | 0.90 / 0.11 / 0.88 / 0.88 / 0.99 | 0.86 / 0.14 / 0.87 / 0.89 / 0.99 |
| SabiYarn-270M science start (A2W a0.7) | sentence-by-sentence + repair | 0.96 / 0.02 / 0.99 / 0.99 / 0.99 | 0.94 / 0.01 / 0.99 / 0.99 / 0.99 |
| SabiYarn-270M live run (step 250) | whole document | 0.96 / 0.04 / 0.89 / 0.86 / 0.98 | 0.95 / 0.07 / 0.89 / 0.87 / 0.99 |
| SabiYarn-270M live run (step 250) | sentence-by-sentence + repair | 0.95 / 0.02 / 0.99 / 0.99 / 0.98 | 0.95 / 0.01 / 1.00 / 0.99 / 0.99 |
| SabiYarn-270M final run (final2, step 401) | whole document | 0.94 / 0.10 / 0.91 / 0.87 / 0.98 | 0.95 / 0.05 / 0.89 / 0.87 / 0.99 |
| SabiYarn-270M final run (final2, step 401) | sentence-by-sentence + repair | 0.96 / 0.02 / 1.00 / 0.99 / 0.99 | 0.96 / 0.01 / 0.99 / 0.99 / 0.99 |
| NLLB-200-distilled-600M | whole document | 0.36 / 0.78 / 0.51 / 0.47 / 0.80 | 0.34 / 0.73 / 0.52 / 0.46 / 0.81 |
| NLLB-200-distilled-600M | sentence-by-sentence | 0.84 / 0.25 / 1.00 / 1.04 / 0.99 | 0.84 / 0.33 / 1.00 / 1.04 / 0.99 |
| NLLB-200-3.3B (CT2 int8) | whole document | 0.83 / 0.58 / 0.60 / 0.30 / 0.86 | 0.74 / 0.69 / 0.67 / 0.33 / 0.80 |
| NLLB-200-3.3B (CT2 int8) | sentence-by-sentence | 0.96 / 0.16 / 1.00 / 1.02 / 0.99 | 0.94 / 0.18 / 1.00 / 1.02 / 0.99 |
| TranslatePsy-AfriSLM-0.8B | whole document | 0.21 / 0.82 / 0.79 / 2.02 / 0.97 | 0.21 / 0.79 / 0.79 / 1.95 / 0.97 |
| TranslatePsy-AfriSLM-0.8B | sentence-by-sentence | 0.97 / 0.04 / 1.00 / 1.01 / 0.99 | 0.96 / 0.04 / 1.00 / 1.02 / 0.99 |
| TranslatePsy-AfriSLM-2B | whole document | 0.53 / 0.34 / 0.65 / 1.40 / 0.95 | n/a |
| TranslatePsy-AfriSLM-2B | sentence-by-sentence | 0.99 / 0.01 / 1.00 / 1.03 / 0.99 | n/a |
| Qwen3.5-0.8B (zero-shot) | whole document | n/a | n/a |
| Qwen3.5-0.8B (zero-shot) | sentence-by-sentence | 0.14 / 0.85 / 0.65 / 2.48 / 0.14 | 0.13 / 0.85 / 0.64 / 2.47 / 0.14 |
| Gemma-3-1B-it (zero-shot) | whole document | n/a | n/a |
| Gemma-3-1B-it (zero-shot) | sentence-by-sentence | 0.00 / 1.00 / 0.88 / 2.11 / 0.36 | n/a |
