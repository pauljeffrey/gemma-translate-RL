### Table LC-F-human. Failure metrics, macro over languages (human set)

finished = stopped at its own end token / before the length limit; loop = word-cycle repetition absent from the reference; coverage = share of source sentences with an aligned, non-copied output sentence (alignment plan of the long-document reward v2); len. ratio = hypothesis / reference characters; LID = share of output sentences (>= 25 characters) that GlotLID labels as the target language.

| model | mode | stitched ~1,024: finished / loop / coverage / len. ratio / LID | stitched ~2,048: finished / loop / coverage / len. ratio / LID |
|:---|---:|---:|---:|
| SabiYarn-270M SFT-final | whole document | 0.07 / 0.90 / 0.78 / 1.80 / 0.90 | 0.03 / 1.00 / 0.74 / 1.80 / 0.96 |
| SabiYarn-270M SFT-final | sentence-by-sentence + repair | 1.00 / 0.02 / 0.98 / 0.98 / 0.96 | 0.98 / 0.02 / 0.99 / 0.99 / 0.96 |
| SabiYarn-270M RL-final (2c) | whole document | 0.51 / 0.67 / 0.95 / 1.29 / 0.93 | 0.39 / 0.90 / 0.94 / 1.27 / 0.89 |
| SabiYarn-270M RL-final (2c) | sentence-by-sentence + repair | 0.99 / 0.00 / 1.00 / 1.00 / 0.95 | 0.95 / 0.00 / 1.00 / 1.01 / 0.95 |
| SabiYarn-270M science start (A2W a0.7) | whole document | 0.98 / 0.15 / 0.87 / 0.80 / 0.95 | 0.80 / 0.30 / 0.72 / 0.75 / 0.95 |
| SabiYarn-270M science start (A2W a0.7) | sentence-by-sentence + repair | 0.98 / 0.03 / 1.00 / 0.99 / 0.96 | 0.97 / 0.02 / 1.00 / 1.00 / 0.95 |
| SabiYarn-270M live run (step 250) | whole document | 1.00 / 0.04 / 0.87 / 0.78 / 0.95 | 0.95 / 0.10 / 0.70 / 0.56 / 0.94 |
| SabiYarn-270M live run (step 250) | sentence-by-sentence + repair | 0.99 / 0.02 / 1.00 / 0.99 / 0.96 | 0.98 / 0.02 / 1.00 / 1.00 / 0.95 |
| SabiYarn-270M final run (final2, step 401) | whole document | 0.98 / 0.08 / 0.88 / 0.81 / 0.95 | 0.92 / 0.25 / 0.73 / 0.59 / 0.95 |
| SabiYarn-270M final run (final2, step 401) | sentence-by-sentence + repair | 1.00 / 0.01 / 1.00 / 0.99 / 0.95 | 0.98 / 0.03 / 1.00 / 1.00 / 0.95 |
| NLLB-200-distilled-600M | whole document | 0.55 / 0.64 / 0.41 / 0.41 / 0.86 | 0.45 / 0.69 / 0.31 / 0.23 / 0.90 |
| NLLB-200-distilled-600M | sentence-by-sentence | 0.96 / 0.09 / 0.99 / 0.99 / 0.95 | 0.95 / 0.20 / 1.00 / 0.99 / 0.94 |
| NLLB-200-3.3B (CT2 int8) | whole document | 0.87 / 0.53 / 0.39 / 0.25 / 0.90 | 0.85 / 0.49 / 0.33 / 0.16 / 0.93 |
| NLLB-200-3.3B (CT2 int8) | sentence-by-sentence | 1.00 / 0.01 / 0.99 / 0.99 / 0.95 | 1.00 / 0.02 / 0.99 / 0.99 / 0.95 |
| TranslatePsy-AfriSLM-0.8B | whole document | 0.77 / 0.23 / 0.46 / 0.66 / 0.93 | 0.61 / 0.38 / 0.53 / 0.94 / 0.91 |
| TranslatePsy-AfriSLM-0.8B | sentence-by-sentence | 0.98 / 0.02 / 1.00 / 1.01 / 0.95 | 0.98 / 0.02 / 1.00 / 1.01 / 0.95 |
| TranslatePsy-AfriSLM-2B | whole document | 0.75 / 0.15 / 0.55 / 0.79 / 0.94 | 0.76 / 0.18 / 0.54 / 0.64 / 0.94 |
| TranslatePsy-AfriSLM-2B | sentence-by-sentence | 1.00 / 0.00 / 1.00 / 1.02 / 0.94 | 1.00 / 0.00 / 1.00 / 1.03 / 0.94 |
| Qwen3.5-0.8B (zero-shot) | whole document | n/a | n/a |
| Qwen3.5-0.8B (zero-shot) | sentence-by-sentence | 0.02 / 0.98 / 0.75 / 2.78 / 0.23 | 0.00 / 0.98 / 0.77 / 2.80 / 0.22 |
| Gemma-3-1B-it (zero-shot) | whole document | n/a | n/a |
| Gemma-3-1B-it (zero-shot) | sentence-by-sentence | 0.10 / 0.92 / 0.98 / 1.61 / 0.48 | 0.00 / 1.00 / 0.98 / 1.65 / 0.49 |
