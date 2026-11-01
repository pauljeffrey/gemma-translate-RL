### gemma-3-270m-translate-rl2c@latest (step 1433) - dev - sentence level (greedy unless stated)

| metric | amh | hau | lug | nso | yor | zul | macro |
|---|---|---|---|---|---|---|---|
| chrf | 26.07 | 61.45 | 46.54 | 58.10 | 38.13 | 61.23 | 48.59 |
| chrf_pp | 23.63 | 59.26 | 42.55 | 55.98 | 37.03 | 55.45 | 45.65 |
| bleu | 5.90 | 35.61 | 16.96 | 32.50 | 15.19 | 22.97 | 21.52 |
| spbleu | 13.09 | 37.46 | 20.39 | 35.93 | 20.98 | 33.92 | 26.96 |
| chrf_sentence_mean | 26.26 | 61.46 | 46.42 | 58.39 | 38.29 | 61.47 | 48.71 |
| finished_rate | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| length_ratio | 1.00 | 0.97 | 0.97 | 0.97 | 1.02 | 1.02 | 0.99 |

### gemma-3-270m-translate-rl2c@latest (step 1433) - dev - document level (greedy unless stated)

| metric | amh | hau | lug | nso | yor | zul | macro |
|---|---|---|---|---|---|---|---|
| chrf | 23.47 | 49.42 | 33.46 | 39.81 | 22.54 | 47.28 | 36.00 |
| chrf_pp | 20.51 | 46.16 | 27.77 | 36.86 | 20.92 | 40.32 | 32.09 |
| bleu | 2.50 | 13.13 | 1.63 | 11.12 | 2.63 | 6.10 | 6.19 |
| spbleu | 8.49 | 15.75 | 3.80 | 13.41 | 5.69 | 12.62 | 9.96 |
| finished_rate | 0.92 | 0.76 | 0.48 | 0.88 | 0.60 | 0.84 | 0.75 |
| length_ratio | 1.03 | 1.63 | 1.91 | 1.17 | 1.67 | 1.33 | 1.46 |

### gemma-3-270m-translate-rl2c@latest (step 1433) - test - sentence level (greedy unless stated)

| metric | amh | hau | lug | nso | yor | zul | macro |
|---|---|---|---|---|---|---|---|
| chrf | 25.28 | 59.43 | 46.19 | 55.65 | 37.41 | 63.39 | 47.89 |
| chrf_pp | 22.80 | 57.23 | 41.88 | 53.40 | 36.22 | 57.66 | 44.86 |
| bleu | 4.91 | 33.13 | 14.88 | 29.12 | 14.46 | 23.61 | 20.02 |
| spbleu | 11.58 | 34.08 | 18.91 | 31.71 | 19.52 | 35.01 | 25.13 |
| chrf_sentence_mean | 25.72 | 59.19 | 46.45 | 55.97 | 37.08 | 63.45 | 47.98 |
| finished_rate | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| length_ratio | 1.01 | 0.97 | 1.00 | 0.97 | 1.00 | 1.04 | 1.00 |
| published chrF: NLLB-200-1.3B fine-tuned | 45.5 | 67.7 | 51.5 | 64.3 | 17.7 | 67.2 | 52.3 |
| published chrF: NLLB-200-distilled-600M fine-tuned | 44.4 | 66.9 | 50.6 | 64.2 | 17.6 | 66.3 | 51.7 |
| published chrF: M2M100-1.2B fine-tuned | 39.0 | 63.1 | 49.5 | 62.4 | 16.6 | 58.4 | 48.2 |
| published chrF: NLLB-200-1.3B zero-shot | 37.4 | 59.4 | 43.2 | 51.4 | 13.0 | 60.1 | 44.1 |
| published chrF: Gemma2-9B LoRA r64 fine-tuned | 39.7 | 58.8 | 50.1 | 61.6 | 41.6 | 50.2 | 50.3 |
| published chrF: GPT-5.4 10-shot | 43.8 | 63.0 | 56.4 | 59.5 | 28.1 | 68.9 | 53.3 |
| published chrF: Gemini-3.1-Flash-Lite 10-shot | 41.6 | 61.1 | 49.5 | 55.9 | 22.9 | 61.8 | 48.8 |

### gemma-3-270m-translate-rl2c@latest (step 1433) - test - document level (greedy unless stated)

| metric | amh | hau | lug | nso | yor | zul | macro |
|---|---|---|---|---|---|---|---|
| chrf | 22.09 | 49.10 | 30.82 | 34.21 | 21.78 | 40.75 | 33.13 |
| chrf_pp | 19.11 | 45.82 | 25.30 | 31.09 | 20.04 | 34.14 | 29.25 |
| bleu | 1.96 | 14.45 | 1.14 | 6.57 | 2.39 | 3.56 | 5.01 |
| spbleu | 6.68 | 16.45 | 2.74 | 8.02 | 4.69 | 7.70 | 7.71 |
| finished_rate | 0.96 | 0.86 | 0.43 | 0.82 | 0.54 | 0.79 | 0.73 |
| length_ratio | 0.88 | 1.31 | 1.96 | 1.23 | 1.58 | 1.62 | 1.43 |
| published chrF: doc: GPT-5.4 1-shot | 42.8 | 66.8 | 59.5 | 64.9 | 28.7 | 68.6 | 55.2 |
| published chrF: doc: GPT-5.4 0-shot | 26.0 | 66.1 | 59.0 | 63.3 | 19.8 | 66.1 | 50.0 |
| published chrF: doc: Gemini-3.1-Flash-Lite 1-shot | 45.1 | 64.3 | 50.8 | 55.8 | 17.5 | 62.9 | 49.4 |
| published chrF: doc: TranslateGemma-12B 1-shot | 20.4 | 33.3 | 15.6 | 21.5 | 7.0 | 32.7 | 21.8 |
| published chrF: doc: Gemma2-9B-IT 1-shot | 3.8 | 30.9 | 8.2 | 10.4 | 2.7 | 12.6 | 11.4 |
