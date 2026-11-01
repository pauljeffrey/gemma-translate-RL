### Table A-M5a. Published AGGREGATES on FLORES-200 devtest, 19 African languages (TranslatePsy-AfriSLM paper, Tables 24/25; direction not stated in the captions, both directions averaged by inference). NOT comparable with the en->xx per-language numbers of Table M1; our own en->xx macros over the same 19 languages are in Table M2.

| model | params | chrF++ | spBLEU | SSA-COMET (0-1) | COMET-22 (0-1) | MetricX-24 (lower better) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| AfriNLLB-600M | 548M | 46.74 | 25.76 | 0.5729 | 0.7315 | 5.916 |
| NLLB-600M | 600M | 45.73 | 24.48 | 0.5718 | 0.7340 | 5.752 |
| TranslatePsy-AfriSLM-0.8B | 800M | 48.3 | 27.0 | 0.5944 | 0.7432 | 5.197 |
| Qwen3.5-0.8B | 800M | 14.06 | 3.27 | 0.2967 | 0.3656 | 15.974 |
| NLLB-200-1.3B | 1.3B | 47.77 | 26.8 | 0.5878 | 0.7471 | 5.159 |
| TranslatePsy-AfriSLM-2B | 2B | 50.0 | 28.95 | 0.6070 | 0.7552 | 4.637 |
| Qwen3.5-2B | 2B | 20.95 | 5.87 | 0.2885 | 0.4724 | 14.344 |
| NLLB-200-3.3B | 3.3B | 48.89 | 28.06 | 0.5944 | 0.7525 | 4.913 |
| TranslatePsy-AfriSLM-4B | 4B | 51.13 | 30.28 | 0.6143 | 0.7629 | 4.317 |
| Qwen3.5-4B | 4B | 30.58 | 11.35 | 0.3863 | 0.6021 | 11.416 |
| Qwen3-4B | 4B | 17.35 | 5.03 | 0.2426 | 0.4046 | 15.427 |
| AfriqueGemma-4B | 4.3B | 44.46 | 24.98 | 0.5462 | 0.7203 | 6.091 |
| TranslateGemma-4B | 4.3B | 24.78 | 10.21 | 0.3913 | 0.5487 | 11.086 |
| Hunyuan-MT-7B | 7B | 23.49 | 6.03 | 0.4450 | 0.5056 | 10.779 |
| AfriqueLlama-8B | 8B | 45.04 | 25.57 | 0.5513 | 0.7245 | 5.921 |
| Apertus-8B | 8B | 30.45 | 12.22 | 0.3814 | 0.5883 | 11.471 |
| AfriqueQwen-8B | 8.2B | 44.41 | 25.01 | 0.5448 | 0.7202 | 6.109 |
| Qwen3-8B | 8.2B | 20.63 | 6.9 | 0.2655 | 0.4566 | 15.327 |
| Qwen3.5-9B | 9B | 36.74 | 15.52 | 0.4523 | 0.6688 | 9.758 |
| AfriqueGemma-12B | 12B | 47.84 | 28.48 | 0.5655 | 0.7389 | 5.486 |
| TranslateGemma-12B | 12B | 39.48 | 16.66 | 0.5037 | 0.7125 | 7.718 |
| AfriqueQwen-14B | 14.8B | 46.16 | 26.82 | 0.5569 | 0.7303 | 5.702 |
| TranslateGemma-27B | 27B | 43.33 | 21.13 | 0.5455 | 0.7335 | 6.142 |
| Qwen3.5-27B | 27B | 41.78 | 19.96 | 0.5132 | 0.7144 | 7.250 |
| Qwen3.5-122B-A10B | 122B | 44.42 | 21.89 | 0.5505 | 0.7373 | 5.920 |

### Table A-M5b. AfroBench pooled MT chrF (Flores-200 + MAFAND + NTREX-128, en/fr -> many African languages, best of 3 prompts, 0-shot; AfroBench paper Table 4)

| model | params | chrF (pooled) |
| :--- | ---: | ---: |
| Gemini 1.5 pro | closed | 37.6 |
| GPT-4o (Aug) | closed | 35.1 |
| NLLB 3.3B | 3.3B | 40.4 |
| Aya-101 13B | 13B | 23.4 |
| Gemma 2 27B | 27B | 27.9 |
| LLaMa 3.1 70B | 70B | 25.1 |
