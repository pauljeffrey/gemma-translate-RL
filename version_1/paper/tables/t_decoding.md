| configuration | chrF++ | $\Delta$ chrF++ [95% CI] | spBLEU | AfriCOMET (18 lang.) | loop % | LID % | cost vs same-engine greedy |
|:--|--:|--:|--:|--:|--:|--:|--:|
| greedy (vLLM) = reference | 32.52 | 0 | 12.15 | 0.6230 | 0.1 | 99.9 | 1.0x |
| greedy + repetition\_penalty 1.05 (vLLM, prompt + output) | 32.51 | -0.01 [-0.13, +0.12] | 11.62 | 0.6316 | 0.0 | 99.9 | 1.1x |
| greedy + repetition\_penalty 1.1 (vLLM, prompt + output) | 31.03 | -1.48 [-1.64, -1.33]* | 9.58 | 0.6116 | 0.0 | 99.9 | 1.1x |
| greedy + repetition penalty 1.05, generated tokens only (HF) | 32.33 | -0.19 [-0.30, -0.08]* | 11.77 | 0.6293 | 0.0 | 99.9 | 0.8x |
| greedy + repetition penalty 1.1, generated tokens only (HF) | 31.11 | -1.40 [-1.54, -1.27]* | 10.55 | 0.6155 | 0.0 | 99.8 | 0.9x |
| greedy + repetition\_penalty 1.1 (HF, prompt + output) | 30.98 | -1.53 [-1.69, -1.38]* | 9.58 | 0.6098 | 0.0 | 99.9 | 1.0x |
| greedy + no\_repeat\_ngram\_size 3 (HF) | 31.45 | -1.06 [-1.18, -0.95]* | 10.34 | 0.6182 | 0.0 | 100.0 | 1.1x |
| greedy + no\_repeat\_ngram\_size 4 (HF) | 31.95 | -0.56 [-0.67, -0.46]* | 11.06 | 0.6215 | 0.0 | 100.0 | 1.3x |
| sampling T 0.3 (vLLM, 1 sample) | 32.39 | -0.13 [-0.23, -0.02]* | 12.03 | 0.6238 | 0.1 | 99.9 | 1.3x |
| sampling T 0.5 (vLLM, 1 sample) | 32.17 | -0.35 [-0.48, -0.22]* | 11.78 | 0.6215 | 0.0 | 99.9 | 1.2x |
| sampling T 0.7 (vLLM, 1 sample) | 31.73 | -0.79 [-0.93, -0.65]* | 11.36 | 0.6144 | 0.0 | 99.9 | 1.0x |
| beam search 2 (HF) | 33.00 | +0.49 [+0.38, +0.60]* | 12.62 | 0.6302 | 0.4 | 99.9 | 2.3x |
| beam search 4 (HF) | 33.18 | +0.66 [+0.55, +0.78]* | 12.84 | 0.6338 | 0.6 | 99.9 | 4.6x |
| beam search 8 (HF) | 33.26 | +0.74 [+0.62, +0.86]* | 12.88 | 0.6344 | 1.0 | 99.9 | 5.6x |
| MBR: 8 samples T 0.7, pick the candidate with the highest mean chrF++ to the other 7 | 33.03 | +0.51 [+0.39, +0.63]* | 12.05 | 0.6249 | 0.0 | 99.9 | 10.0x |
| MBR: same pool + the greedy output (9 candidates) | 33.05 | +0.53 [+0.44, +0.61]* | 12.17 | 0.6267 | 0.1 | 99.9 | 12.6x |
| MBR: 8 samples T 0.5 | 33.06 | +0.54 [+0.44, +0.65]* | 12.22 | 0.6285 | 0.0 | 99.9 | 15.6x |
| MBR: 8 samples T 0.5 + greedy | 32.99 | +0.47 [+0.39, +0.55]* | 12.20 | 0.6287 | 0.0 | 99.9 | 18.3x |
