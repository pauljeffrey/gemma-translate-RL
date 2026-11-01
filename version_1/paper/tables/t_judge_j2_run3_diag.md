| system | sampled sentences in documents flagged loop or unfinished | judged-unit mean, flagged vs unflagged documents | mean at ~1,024 / ~2,048 tokens (covered sentences) | omitted at ~1,024 / ~2,048 tokens | units possibly misaligned (alignment check) | Spearman, judge vs own-block chrF-like score |
|:--|--:|--:|--:|--:|--:|--:|
| Final checkpoint (step 401), whole document | 105 / 600 | 8.9 vs 27.3 | 33.8 / 12.1 | 11% / 23% | 4.5% | 0.74 |
| Final checkpoint, sentence-by-sentence wrapper | 27 / 600 | 56.3 vs 65.8 | 66.3 / 64.5 | 0% / 0% | 0.0% | 0.65 |
| RL-final, whole document | 407 / 600 | 4.2 vs 28.6 | 19.1 / 3.9 | 9% / 5% | 2.2% | 0.57 |
| NLLB-200 3.3B, sentence by sentence | 0 / 600 | n/a vs 85.3 | 85.7 / 84.8 | 0% / 0% | 0.0% | n/a |
| TranslatePsy-AfriSLM-0.8B, sentence by sentence | 19 / 500 | 83.8 vs 86.6 | 86.9 / 86.0 | 0% / 0% | 0.2% | 0.46 |
