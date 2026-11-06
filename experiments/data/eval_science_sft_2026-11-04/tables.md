Start checkpoint: **mixed** (highest common-language dev chrF++); `/workspace/clean_rerun/runs_385m/mixed/best`. Candidates: mixed: dev chrF++ 29.20, AfriCOMET 0.6355 (step 50); weak: dev chrF++ 29.05, AfriCOMET 0.6349 (step 25)

## Table 1. Interpolation sweep: theta(alpha) = (1 - alpha) theta_start + alpha theta_science-SFT (alpha 0 = the start checkpoint)

| alpha | AfriScience dev chrF (macro 6) | AfriScience test chrF | NTREX macro chrF++ | d vs alpha 0 | MAFAND macro chrF++ | d vs alpha 0 | passes non-regression (>= start - 0.5) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 45.74 | 45.40 | 37.16 | +0.00 | 40.78 | +0.00 | yes |
| 0.3 | 46.53 | 46.28 | 37.13 | -0.04 | 41.06 | +0.28 | yes |
| 0.5 (chosen) | 47.17 | 46.63 | 36.90 | -0.26 | 41.05 | +0.27 | yes |
| 0.7 | 46.96 | 46.56 | 36.43 | -0.73 | 40.83 | +0.05 | no |
| 1.0 | 46.02 | 44.92 | 34.71 | -2.45 | 39.58 | -1.20 | no |

Chosen alpha: **0.5** (rule: best AfriScience dev chrF subject to NTREX and MAFAND not below the start by more than 0.5).

## Table 2. AfriScience-MT test, sentence level, plain chrF (word order 0), greedy: now a FAIR comparison with the old 270M (both have had a science stage)

| system | amh | hau | lug | nso | yor | zul | macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M alpha 0 (start, no science SFT) | 29.1 | 55.2 | 38.1 | 52.1 | 37.7 | 60.2 | 45.4 |
| 385M alpha 0.5 (chosen) | 29.9 | 56.1 | 39.3 | 54.7 | 38.1 | 61.7 | 46.6 |
| old 270M RL-final (1,433) | 25.3 | 59.4 | 46.2 | 55.9 | 37.4 | 63.4 | 47.9 |
| old 270M science-stage start (A2W) | 26.2 | 62.0 | 48.9 | 58.3 | 39.8 | 64.7 | 50.0 |
| old 270M final2 (step 401) | 26.6 | 61.9 | 49.3 | 58.5 | 40.1 | 64.7 | 50.2 |
| NLLB-200-1.3B fine-tuned (published) | 45.5 | 67.7 | 51.5 | 64.3 | 17.7 | 67.2 | 52.3 |
| NLLB-200-distilled-600M fine-tuned (published) | 44.4 | 66.9 | 50.6 | 64.2 | 17.6 | 66.3 | 51.7 |
| M2M100-1.2B fine-tuned (published) | 39.0 | 63.1 | 49.5 | 62.4 | 16.6 | 58.4 | 48.2 |

## Table 3. NTREX / MAFAND non-regression on the clean rows of the old models (chrF++, languages both cover)

| benchmark | languages | old SFT (clean) | old RL (clean) | 385M start (alpha 0) | 385M chosen alpha |
|---|---:|---:|---:|---:|---:|
| NTREX | 24 | 31.5 | 35.6 | 37.2 | 36.9 |
| MAFAND | 15 | 38.2 | 41.8 | 40.8 | 41.1 |

## SFT validation curve (AfriScience dev loss = early-stopping split; replay loss = forgetting monitor)

```
/workspace/science_sft/sft_attempt1_oom.log:[gemma_mt] eval[val_replay] @ 0: val_loss 1.4949 macro 1.5146 (76 languages, 487230 target tokens)
/workspace/science_sft/sft.log:[gemma_mt] eval[val_replay] @ 0: val_loss 1.4949 macro 1.5146 (76 languages, 487230 target tokens)
/workspace/science_sft/sft.log:[gemma_mt] eval[val_replay] @ 20: val_loss 1.3557 macro 1.3743 (76 languages, 487230 target tokens)
/workspace/science_sft/sft.log:[gemma_mt] eval[val_replay] @ 100: val_loss 1.3384 macro 1.3573 (76 languages, 487230 target tokens)
/workspace/science_sft/sft.log:[gemma_mt] eval[val_replay] @ 200: val_loss 1.3365 macro 1.3552 (76 languages, 487230 target tokens)
/workspace/science_sft/sft.log:[gemma_mt] eval[val_replay] @ 283: val_loss 1.3359 macro 1.3546 (76 languages, 487230 target tokens)
```

