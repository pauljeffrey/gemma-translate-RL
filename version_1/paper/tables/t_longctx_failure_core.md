### Table LC-F-core. Failure metrics, macro over languages (core set)

finished = stopped at its own end token / before the length limit; loop = word-cycle repetition absent from the reference; coverage = share of source sentences with an aligned, non-copied output sentence (alignment plan of the long-document reward v2); len. ratio = hypothesis / reference characters; LID = share of output sentences (>= 25 characters) that GlotLID labels as the target language.

| model | mode | b512: finished / loop / coverage / len. ratio / LID | b1024: finished / loop / coverage / len. ratio / LID | b2048: finished / loop / coverage / len. ratio / LID |
|:---|---:|---:|---:|---:|
| SabiYarn-270M SFT-final | whole document | 0.61 / 0.42 / 0.87 / 1.25 / 0.91 | 0.12 / 0.79 / 0.70 / 1.76 / 0.91 | 0.03 / 0.88 / 0.62 / 1.79 / 0.88 |
| SabiYarn-270M SFT-final | sentence-by-sentence + repair | 0.99 / 0.00 / 0.99 / 0.98 / 0.98 | 0.99 / 0.00 / 0.98 / 0.99 / 0.97 | 0.99 / 0.04 / 0.98 / 0.98 / 0.97 |
| SabiYarn-270M RL-final (2c) | whole document | 0.99 / 0.01 / 0.98 / 0.97 / 0.98 | 0.77 / 0.43 / 0.93 / 1.01 / 0.95 | 0.41 / 0.83 / 0.95 / 1.26 / 0.89 |
| SabiYarn-270M RL-final (2c) | sentence-by-sentence + repair | 0.97 / 0.01 / 1.00 / 1.00 / 0.98 | 0.97 / 0.00 / 0.99 / 1.01 / 0.97 | 0.94 / 0.00 / 1.00 / 1.00 / 0.97 |
| SabiYarn-270M science start (A2W a0.7) | whole document | 0.96 / 0.06 / 0.98 / 1.00 / 0.97 | 0.90 / 0.15 / 0.88 / 0.91 / 0.95 | 0.72 / 0.33 / 0.80 / 0.87 / 0.96 |
| SabiYarn-270M science start (A2W a0.7) | sentence-by-sentence + repair | 0.98 / 0.03 / 1.00 / 0.99 / 0.98 | 0.99 / 0.01 / 0.99 / 1.00 / 0.96 | 0.97 / 0.03 / 1.00 / 0.99 / 0.97 |
| SabiYarn-270M live run (step 250) | whole document | 1.00 / 0.03 / 0.99 / 0.98 / 0.98 | 0.99 / 0.05 / 0.90 / 0.84 / 0.96 | 0.94 / 0.12 / 0.70 / 0.58 / 0.95 |
| SabiYarn-270M live run (step 250) | sentence-by-sentence + repair | 0.98 / 0.02 / 1.00 / 0.99 / 0.98 | 0.97 / 0.00 / 0.99 / 1.00 / 0.97 | 0.97 / 0.00 / 1.00 / 0.99 / 0.97 |
| SabiYarn-270M final run (final2, step 401) | whole document | 1.00 / 0.02 / 0.98 / 0.98 / 0.98 | 0.98 / 0.06 / 0.89 / 0.88 / 0.96 | 0.91 / 0.25 / 0.74 / 0.66 / 0.95 |
| SabiYarn-270M final run (final2, step 401) | sentence-by-sentence + repair | 0.97 / 0.01 / 1.00 / 0.99 / 0.98 | 0.97 / 0.02 / 0.99 / 1.00 / 0.97 | 0.95 / 0.00 / 1.00 / 0.99 / 0.97 |
| NLLB-200-distilled-600M | whole document | 0.73 / 0.28 / 0.54 / 0.55 / 0.94 | 0.43 / 0.54 / 0.38 / 0.39 / 0.89 | 0.41 / 0.54 / 0.30 / 0.22 / 0.87 |
| NLLB-200-distilled-600M | sentence-by-sentence | 0.99 / 0.07 / 1.00 / 1.01 / 0.97 | 0.96 / 0.09 / 0.99 / 1.00 / 0.97 | 0.88 / 0.17 / 1.00 / 1.01 / 0.97 |
| NLLB-200-3.3B (CT2 int8) | whole document | 0.93 / 0.18 / 0.54 / 0.44 / 0.95 | 0.85 / 0.33 / 0.36 / 0.27 / 0.87 | 0.83 / 0.52 / 0.38 / 0.16 / 0.85 |
| TranslatePsy-AfriSLM-0.8B | whole document | 0.88 / 0.09 / 0.50 / 0.56 / 0.98 | 0.70 / 0.27 / 0.47 / 0.89 / 0.97 | 0.70 / 0.27 / 0.44 / 0.77 / 0.94 |
| TranslatePsy-AfriSLM-0.8B | sentence-by-sentence | 0.99 / 0.01 / 1.00 / 1.01 / 0.98 | 0.98 / 0.01 / 0.99 / 1.02 / 0.98 | 0.96 / 0.02 / 1.00 / 1.01 / 0.98 |
| TranslatePsy-AfriSLM-2B | whole document | n/a | n/a | 0.77 / 0.14 / 0.44 / 0.64 / 0.88 |
| Qwen3.5-0.8B (zero-shot) | whole document | n/a | n/a | n/a |
| Qwen3.5-0.8B (zero-shot) | sentence-by-sentence | 0.14 / 0.81 / 0.69 / 2.56 / 0.21 | 0.08 / 0.87 / 0.68 / 2.60 / 0.21 | 0.03 / 0.83 / 0.67 / 2.50 / 0.23 |
| Gemma-3-1B-it (zero-shot) | whole document | n/a | n/a | n/a |
