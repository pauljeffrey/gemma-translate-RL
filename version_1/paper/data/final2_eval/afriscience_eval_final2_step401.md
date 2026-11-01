### /dev/shm/lc1_models/final2 - dev - sentence level (greedy unless stated)

| metric | amh | hau | lug | nso | yor | zul | macro |
|---|---|---|---|---|---|---|---|
| chrf | 27.43 | 64.14 | 49.50 | 61.49 | 40.64 | 62.41 | 50.94 |
| chrf_pp | 25.12 | 62.21 | 45.38 | 59.59 | 39.58 | 56.81 | 48.12 |
| bleu | 6.67 | 40.07 | 18.86 | 37.41 | 17.93 | 25.32 | 24.38 |
| spbleu | 14.83 | 41.98 | 24.47 | 40.84 | 24.01 | 36.17 | 30.38 |
| chrf_sentence_mean | 27.94 | 64.14 | 49.25 | 61.86 | 40.74 | 62.67 | 51.10 |
| finished_rate | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| length_ratio | 0.99 | 0.98 | 0.96 | 0.98 | 1.00 | 1.01 | 0.98 |

### /dev/shm/lc1_models/final2 - dev - document level (greedy unless stated)

| metric | amh | hau | lug | nso | yor | zul | macro |
|---|---|---|---|---|---|---|---|
| chrf | 27.07 | 65.30 | 50.24 | 58.33 | 34.36 | 60.21 | 49.25 |
| chrf_pp | 24.09 | 62.70 | 43.89 | 55.83 | 32.86 | 52.89 | 45.38 |
| bleu | 3.26 | 35.00 | 9.58 | 27.41 | 10.04 | 16.79 | 17.01 |
| spbleu | 10.38 | 36.87 | 17.62 | 29.91 | 15.15 | 27.98 | 22.99 |
| finished_rate | 1.00 | 1.00 | 0.92 | 1.00 | 1.00 | 0.96 | 0.98 |
| length_ratio | 0.81 | 0.93 | 1.01 | 0.83 | 0.90 | 0.89 | 0.90 |

### /dev/shm/lc1_models/final2 - test - sentence level (greedy unless stated)

| metric | amh | hau | lug | nso | yor | zul | macro |
|---|---|---|---|---|---|---|---|
| chrf | 26.60 | 61.86 | 49.33 | 58.45 | 40.06 | 64.68 | 50.16 |
| chrf_pp | 24.32 | 59.91 | 44.94 | 56.50 | 38.95 | 59.17 | 47.30 |
| bleu | 6.04 | 36.90 | 17.05 | 33.38 | 17.44 | 25.98 | 22.80 |
| spbleu | 13.19 | 38.03 | 23.42 | 35.91 | 22.61 | 37.73 | 28.48 |
| chrf_sentence_mean | 27.20 | 61.61 | 49.48 | 58.82 | 39.73 | 64.69 | 50.26 |
| finished_rate | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| length_ratio | 0.98 | 0.97 | 0.99 | 0.98 | 0.97 | 1.02 | 0.99 |
| published chrF: NLLB-200-1.3B fine-tuned | 45.5 | 67.7 | 51.5 | 64.3 | 17.7 | 67.2 | 52.3 |
| published chrF: NLLB-200-distilled-600M fine-tuned | 44.4 | 66.9 | 50.6 | 64.2 | 17.6 | 66.3 | 51.7 |
| published chrF: M2M100-1.2B fine-tuned | 39.0 | 63.1 | 49.5 | 62.4 | 16.6 | 58.4 | 48.2 |
| published chrF: NLLB-200-1.3B zero-shot | 37.4 | 59.4 | 43.2 | 51.4 | 13.0 | 60.1 | 44.1 |
| published chrF: Gemma2-9B LoRA r64 fine-tuned | 39.7 | 58.8 | 50.1 | 61.6 | 41.6 | 50.2 | 50.3 |
| published chrF: GPT-5.4 10-shot | 43.8 | 63.0 | 56.4 | 59.5 | 28.1 | 68.9 | 53.3 |
| published chrF: Gemini-3.1-Flash-Lite 10-shot | 41.6 | 61.1 | 49.5 | 55.9 | 22.9 | 61.8 | 48.8 |

### /dev/shm/lc1_models/final2 - test - document level (greedy unless stated)

| metric | amh | hau | lug | nso | yor | zul | macro |
|---|---|---|---|---|---|---|---|
| chrf | 25.24 | 59.66 | 45.51 | 54.83 | 32.41 | 58.63 | 46.05 |
| chrf_pp | 22.15 | 57.10 | 39.37 | 51.93 | 30.84 | 51.58 | 42.16 |
| bleu | 2.90 | 28.65 | 7.97 | 21.62 | 9.34 | 16.01 | 14.41 |
| spbleu | 8.85 | 29.79 | 14.51 | 23.53 | 13.21 | 25.38 | 19.21 |
| finished_rate | 0.96 | 1.00 | 0.96 | 1.00 | 0.96 | 1.00 | 0.98 |
| length_ratio | 0.90 | 0.83 | 0.92 | 0.86 | 0.88 | 0.82 | 0.87 |
| published chrF: doc: GPT-5.4 1-shot | 42.8 | 66.8 | 59.5 | 64.9 | 28.7 | 68.6 | 55.2 |
| published chrF: doc: GPT-5.4 0-shot | 26.0 | 66.1 | 59.0 | 63.3 | 19.8 | 66.1 | 50.0 |
| published chrF: doc: Gemini-3.1-Flash-Lite 1-shot | 45.1 | 64.3 | 50.8 | 55.8 | 17.5 | 62.9 | 49.4 |
| published chrF: doc: TranslateGemma-12B 1-shot | 20.4 | 33.3 | 15.6 | 21.5 | 7.0 | 32.7 | 21.8 |
| published chrF: doc: Gemma2-9B-IT 1-shot | 3.8 | 30.9 | 8.2 | 10.4 | 2.7 | 12.6 | 11.4 |
