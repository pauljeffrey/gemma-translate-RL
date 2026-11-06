## Table S1a. AfriScience-MT TEST, sentence level, greedy, macro over the 6 languages (amh hau lug nso yor zul)

| model | chrF (word order 0) | chrF++ | BLEU | finished | length ratio | AfriCOMET-STL (ref, held out) | SSA-COMET-QE (reward family) | SSA-COMET-MTL ref (reward family) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 385M mixed best = science-SFT alpha 0 (science RL's grandparent) | 45.40 | 42.44 | 17.72 | 0.9978 | 1.010 | 0.6828 | 0.6130 | 0.5123 |
| 385M science SFT, alpha 0.5 (**the science RL's start**) | 46.63 | 43.80 | 19.27 | 0.9957 | 1.039 | 0.6879 | 0.6118 | 0.5150 |
| 385M science RL, best/ (step 75) | 46.79 | 43.93 | 19.20 | 0.9978 | 1.021 | 0.6893 | 0.6135 | 0.5185 |
| 385M science RL, final/ (step 160) | 47.10 | 44.22 | 19.53 | 0.9990 | 1.004 | 0.6932 | 0.6158 | 0.5224 |

## Table S1b. AfriScience-MT DEV, sentence level, greedy, macro over the 6 languages (amh hau lug nso yor zul)

| model | chrF (word order 0) | chrF++ | BLEU | finished | length ratio | AfriCOMET-STL (ref, held out) | SSA-COMET-QE (reward family) | SSA-COMET-MTL ref (reward family) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 385M mixed best = science-SFT alpha 0 (science RL's grandparent) | 45.74 | 42.81 | 18.68 | 0.9984 | 0.996 | 0.6988 | 0.6242 | 0.5249 |
| 385M science SFT, alpha 0.5 (**the science RL's start**) | 47.17 | 44.35 | 20.34 | 0.9970 | 1.023 | 0.7044 | 0.6235 | 0.5293 |
| 385M science RL, best/ (step 75) | 47.38 | 44.51 | 20.53 | 0.9986 | 1.002 | 0.7067 | 0.6255 | 0.5326 |
| 385M science RL, final/ (step 160) | 47.62 | 44.74 | 20.64 | 0.9994 | 0.994 | 0.7098 | 0.6277 | 0.5360 |

## Table S2. Paired differences, AfriScience-MT sentence level (paired Poisson bootstrap over sentences, 1000 resamples, macro over the 6 languages; 95 % CI)

| comparison | split | chrF | AfriCOMET-STL (ref, held out) | SSA-COMET-QE (reward family) |
|---|---|---|---|---|
| science RL best minus science SFT alpha 0.5 | test | +0.16 [-0.04, +0.37] | +0.0013 [-0.0005, +0.0031] | +0.0018 [+0.0006, +0.0030] |
| science RL best minus science SFT alpha 0.5 | dev | +0.21 [+0.01, +0.43] | +0.0023 [+0.0005, +0.0041] | +0.0021 [+0.0009, +0.0032] |
| science RL final minus science SFT alpha 0.5 | test | +0.46 [+0.27, +0.65] | +0.0053 [+0.0036, +0.0071] | +0.0041 [+0.0029, +0.0053] |
| science RL final minus science SFT alpha 0.5 | dev | +0.45 [+0.23, +0.66] | +0.0054 [+0.0034, +0.0072] | +0.0043 [+0.0030, +0.0055] |
| science RL best minus alpha 0 (mixed best) | test | +1.39 [+1.17, +1.63] | +0.0065 [+0.0046, +0.0084] | +0.0005 [-0.0008, +0.0017] |
| science RL best minus alpha 0 (mixed best) | dev | +1.64 [+1.43, +1.85] | +0.0079 [+0.0061, +0.0099] | +0.0013 [+0.0001, +0.0026] |
| science SFT alpha 0.5 minus alpha 0 (for scale) | test | +1.23 [+1.00, +1.47] | +0.0052 [+0.0031, +0.0070] | -0.0012 [-0.0025, -0.0000] |
| science SFT alpha 0.5 minus alpha 0 (for scale) | dev | +1.43 [+1.19, +1.67] | +0.0056 [+0.0036, +0.0078] | -0.0007 [-0.0020, +0.0007] |
| science RL final minus best | test | +0.30 [+0.11, +0.49] | +0.0039 [+0.0025, +0.0055] | +0.0023 [+0.0012, +0.0034] |
| science RL final minus best | dev | +0.24 [+0.07, +0.41] | +0.0031 [+0.0016, +0.0047] | +0.0022 [+0.0011, +0.0033] |

## Table S3. Per language, TEST, science RL final minus science SFT alpha 0.5

| iso | chrF alpha 0.5 | chrF RL final | diff [CI] | AfriCOMET alpha 0.5 | AfriCOMET RL final | diff [CI] |
|---|---:|---:|---|---:|---:|---|
| amh | 29.90 | 30.69 | +0.79 [+0.34, +1.22] | 0.6061 | 0.6154 | +0.0093 [+0.0038, +0.0147] |
| hau | 56.10 | 56.74 | +0.64 [+0.01, +1.28] | 0.6927 | 0.6941 | +0.0015 [-0.0026, +0.0055] |
| lug | 39.27 | 39.87 | +0.60 [+0.01, +1.20] | 0.6312 | 0.6378 | +0.0066 [+0.0015, +0.0120] |
| nso | 54.72 | 54.69 | -0.03 [-0.34, +0.34] | 0.7156 | 0.7205 | +0.0049 [+0.0016, +0.0085] |
| yor | 38.11 | 38.62 | +0.51 [+0.21, +0.83] | 0.7594 | 0.7647 | +0.0053 [+0.0025, +0.0082] |
| zul | 61.71 | 61.97 | +0.26 [-0.09, +0.60] | 0.7227 | 0.7268 | +0.0042 [+0.0010, +0.0073] |

