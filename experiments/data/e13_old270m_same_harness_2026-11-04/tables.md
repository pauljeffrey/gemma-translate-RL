## Table 1. Language-macro chrF++ on rows that are clean for both generations (paired bootstrap, 1000 resamples, 95% CI)

| benchmark (languages) | old 270M SFT | 385M SFT | 385M SFT - old SFT [CI] | old 270M RL (rl2c) | 385M RL (mixed best) | 385M RL - old RL [CI] |
|---|---:|---:|---|---:|---:|---|
| FLORES+ devtest (45) | 24.69 | 22.34 | -2.35 [-3.25, -1.38] | 31.68 | 32.03 | +0.35 [+0.08, +0.62] |
| NTREX (24) | 31.46 | 27.33 | -4.13 [-4.40, -3.85] | 35.52 | 35.55 | +0.02 [-0.07, +0.11] |
| MAFAND (15) | 38.26 | 34.71 | -3.54 [-4.22, -2.75] | 41.80 | 40.25 | -1.55 [-1.72, -1.39] |

## Table 2. Held-out AfriCOMET-STL (language macro; languages with a score in parentheses)

| benchmark | old SFT | 385M SFT | diff [CI] | old RL | 385M RL | diff [CI] |
|---|---:|---:|---|---:|---:|---|
| FLORES+ devtest (40) | 0.4798 | 0.4685 | -0.0113 [-0.0189, -0.0037] | 0.5488 | 0.5437 | -0.0051 [-0.0104, -0.0001] |
| NTREX (21) | 0.5455 | 0.5205 | -0.0250 [-0.0273, -0.0229] | 0.5973 | 0.5844 | -0.0129 [-0.0144, -0.0115] |
| MAFAND (15) | 0.6204 | 0.6046 | -0.0158 [-0.0188, -0.0130] | 0.6610 | 0.6407 | -0.0203 [-0.0222, -0.0182] |

## Table 3. 385M RL (mixed best) minus 385M SFT on the same rows, chrF++

| benchmark | SFT | RL | diff [CI] |
|---|---:|---:|---|
| FLORES+ devtest (45) | 22.34 | 32.03 | +9.69 [+8.51, +10.69] |
| NTREX (24) | 27.33 | 35.55 | +8.22 [+7.95, +8.50] |
| MAFAND (15) | 34.71 | 40.25 | +5.54 [+4.86, +6.10] |

## Table 4. Old SFT -> old RL (rl2c), the first research's own RL gain on these rows, chrF++ (macro entry 'all')

| benchmark | old SFT | old RL | diff [CI] |
|---|---:|---:|---|
| FLORES+ devtest (45) | 24.69 | 31.68 | +6.99 [+6.08, +7.69] |
| NTREX (24) | 31.46 | 35.52 | +4.06 [+3.83, +4.30] |
| MAFAND (15) | 38.26 | 41.80 | +3.54 [+2.91, +4.09] |
| all cells (84) | 29.05 | 34.58 | +5.54 [+5.03, +5.91] |

## Table 5. Finished rate / loop rate (reward's flag) / hypothesis-to-reference length, rows clean for both generations

| system | FLORES+ (n rows) | NTREX | MAFAND |
|---|---|---|---|
| old 270M SFT | 0.850 / 0.171 / 2.53 (n = 2160) | 0.922 / 0.083 / 1.78 (n = 28680) | 0.974 / 0.029 / 1.38 (n = 15498) |
| 385M SFT | 0.852 / 0.196 / 2.61 (n = 2160) | 0.876 / 0.137 / 2.50 (n = 28680) | 0.961 / 0.044 / 1.59 (n = 15498) |
| old 270M RL (rl2c) | 1.000 / 0.002 / 1.06 (n = 2160) | 0.999 / 0.002 / 1.04 (n = 28680) | 1.000 / 0.000 / 1.00 (n = 15498) |
| 385M RL (mixed best) | 1.000 / 0.017 / 1.04 (n = 2160) | 0.999 / 0.007 / 1.02 (n = 28680) | 1.000 / 0.001 / 0.99 (n = 15498) |
