#### T1. FLORES+ devtest, no_pair rows, NLLB-covered languages (headline)

Row set `np270`, 42680 sentences, 44 languages (AfriCOMET: 40). 

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 25.59 [25.21, 25.98] | 27.91 | 7.14 | 0.5502 [0.5459, 0.5548] | 9.30 [9.00, 9.55] | 0.0732 [0.0711, 0.0752] | 44 |
| 385M SFT-only control (E-03, +TranslatePsy) | 28.56 [28.23, 28.89] | 31.19 | 9.34 | 0.5858 [0.5815, 0.5901] | 6.32 [6.11, 6.52] | 0.0377 [0.0360, 0.0394] | 34 |
| 385M RL stage A | 34.16 [33.87, 34.46] | 37.13 | 12.85 | 0.6196 [0.6156, 0.6236] | 0.72 [0.65, 0.81] | 0.0039 [0.0031, 0.0046] | 38 |
| 385M RL B1 best | 34.54 [34.25, 34.84] | 37.56 | 13.06 | 0.6211 [0.6173, 0.6251] | 0.34 [0.29, 0.41] | 0.0024 [0.0017, 0.0030] | 34 |
| 385M RL (mixed best = Hub main) | 34.88 [34.60, 35.18] | 37.95 | 13.20 | 0.6234 [0.6195, 0.6273] |  |  |  |
| NLLB-200-distilled-600M | 35.94 [35.59, 36.27] | 39.04 | 15.06 | 0.6615 [0.6575, 0.6653] | -1.05 [-1.24, -0.86] | -0.0381 [-0.0399, -0.0362] | 15 |
| NLLB-200-3.3B | 38.09 [37.76, 38.40] | 41.35 | 17.01 | 0.6770 [0.6734, 0.6806] | -3.21 [-3.38, -3.02] | -0.0535 [-0.0553, -0.0518] | 7 |

#### T2. FLORES+ devtest, no_pair rows, TranslatePsy-0.8B language set

Row set `np270`, 17380 sentences, 19 languages (AfriCOMET: 19). 

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 34.00 [33.49, 34.49] | 36.83 | 12.53 | 0.5886 [0.5834, 0.5938] | 7.04 [6.64, 7.41] | 0.0536 [0.0513, 0.0559] | 19 |
| 385M SFT-only control (E-03, +TranslatePsy) | 40.96 [40.58, 41.37] | 44.44 | 17.66 | 0.6648 [0.6599, 0.6695] | 0.08 [-0.13, 0.30] | -0.0226 [-0.0242, -0.0208] | 9 |
| 385M RL stage A | 40.75 [40.41, 41.10] | 44.14 | 17.85 | 0.6414 [0.6368, 0.6462] | 0.29 [0.14, 0.43] | 0.0008 [-0.0004, 0.0018] | 13 |
| 385M RL B1 best | 40.90 [40.54, 41.24] | 44.32 | 18.00 | 0.6412 [0.6367, 0.6457] | 0.14 [0.04, 0.26] | 0.0010 [-0.0000, 0.0020] | 11 |
| 385M RL (mixed best = Hub main) | 41.04 [40.70, 41.38] | 44.50 | 18.06 | 0.6422 [0.6375, 0.6467] |  |  |  |
| NLLB-200-distilled-600M | 42.58 [42.17, 42.96] | 46.14 | 20.35 | 0.6878 [0.6832, 0.6921] | -1.54 [-1.78, -1.28] | -0.0456 [-0.0480, -0.0434] | 5 |
| NLLB-200-3.3B | 44.66 [44.27, 45.01] | 48.34 | 22.43 | 0.7007 [0.6966, 0.7046] | -3.62 [-3.84, -3.39] | -0.0586 [-0.0609, -0.0563] | 2 |
| TranslatePsy-AfriSLM-0.8B | 45.27 [44.92, 45.60] | 48.94 | 22.39 | 0.7167 [0.7125, 0.7204] | -4.23 [-4.39, -4.07] | -0.0745 [-0.0765, -0.0727] | 1 |

#### T2b. FLORES+ devtest, no_pair rows, TranslatePsy-2B language set (9 languages)

Row set `np270`, 7705 sentences, 9 languages (AfriCOMET: 9). 

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 32.95 [32.26, 33.65] | 35.39 | 11.71 | 0.5731 [0.5671, 0.5788] | 8.60 [8.01, 9.21] | 0.0581 [0.0545, 0.0617] | 9 |
| 385M SFT-only control (E-03, +TranslatePsy) | 41.65 [41.18, 42.10] | 44.84 | 18.15 | 0.6568 [0.6513, 0.6619] | -0.09 [-0.40, 0.18] | -0.0257 [-0.0280, -0.0232] | 4 |
| 385M RL stage A | 41.02 [40.55, 41.48] | 44.10 | 18.01 | 0.6298 [0.6247, 0.6350] | 0.54 [0.27, 0.83] | 0.0014 [-0.0005, 0.0032] | 8 |
| 385M RL B1 best | 41.26 [40.82, 41.66] | 44.38 | 18.39 | 0.6293 [0.6242, 0.6344] | 0.30 [0.09, 0.53] | 0.0019 [0.0002, 0.0035] | 7 |
| 385M RL (mixed best = Hub main) | 41.56 [41.17, 41.95] | 44.72 | 18.59 | 0.6312 [0.6258, 0.6362] |  |  |  |
| NLLB-200-distilled-600M | 44.18 [43.68, 44.62] | 47.52 | 21.92 | 0.6902 [0.6852, 0.6948] | -2.62 [-2.96, -2.27] | -0.0591 [-0.0624, -0.0560] | 2 |
| NLLB-200-3.3B | 46.29 [45.79, 46.72] | 49.72 | 24.06 | 0.7010 [0.6962, 0.7054] | -4.73 [-5.04, -4.42] | -0.0698 [-0.0731, -0.0666] | 0 |
| TranslatePsy-AfriSLM-0.8B | 46.61 [46.17, 46.99] | 50.02 | 23.55 | 0.7151 [0.7104, 0.7194] | -5.05 [-5.29, -4.82] | -0.0840 [-0.0866, -0.0813] | 0 |
| TranslatePsy-AfriSLM-2B | 46.44 [46.02, 46.81] | 49.94 | 23.27 | 0.7156 [0.7116, 0.7196] | -4.88 [-5.15, -4.64] | -0.0845 [-0.0875, -0.0815] | 0 |

#### T3. NTREX-128, all rows

Row set `all`, 41937 sentences, 21 languages (AfriCOMET: 21). NLLB-3.3B: corpus-level per-language aggregate on the same rows (no per-sentence output, no AfriCOMET).

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 31.03 [30.69, 31.42] | 33.79 | 10.99 | 0.5508 [0.5464, 0.5548] | 7.97 [7.73, 8.21] | 0.0630 [0.0614, 0.0646] | 21 |
| 385M SFT-only control (E-03, +TranslatePsy) | 35.86 [35.54, 36.18] | 39.04 | 15.16 | 0.6088 [0.6046, 0.6125] | 3.14 [2.97, 3.32] | 0.0050 [0.0037, 0.0063] | 14 |
| 385M RL stage A | 38.59 [38.30, 38.87] | 41.96 | 17.30 | 0.6125 [0.6086, 0.6163] | 0.41 [0.34, 0.48] | 0.0013 [0.0006, 0.0020] | 17 |
| 385M RL B1 best | 38.84 [38.56, 39.11] | 42.24 | 17.41 | 0.6134 [0.6093, 0.6172] | 0.16 [0.10, 0.23] | 0.0004 [-0.0002, 0.0010] | 15 |
| 385M RL (mixed best = Hub main) | 39.00 [38.73, 39.25] | 42.43 | 17.42 | 0.6138 [0.6099, 0.6177] |  |  |  |
| NLLB-200-3.3B | 43.26 | -- | 21.90 | -- | -4.26 [-5.22, -3.24] (language bootstrap) |  | 1 |

#### T4. MAFAND-MT test, all rows

Row set `all`, 18445 sentences, 14 languages (AfriCOMET: 14). NLLB-3.3B: corpus-level per-language aggregate on the same rows.

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 34.50 [34.23, 34.76] | 37.71 | 13.23 | 0.6143 [0.6120, 0.6164] | 5.39 [5.15, 5.63] | 0.0395 [0.0377, 0.0413] | 14 |
| 385M SFT-only control (E-03, +TranslatePsy) | 39.55 [39.31, 39.75] | 43.06 | 17.96 | 0.6696 [0.6677, 0.6714] | 0.34 [0.16, 0.52] | -0.0158 [-0.0174, -0.0143] | 6 |
| 385M RL stage A | 39.93 [39.74, 40.10] | 43.48 | 18.28 | 0.6563 [0.6547, 0.6582] | -0.05 [-0.13, 0.05] | -0.0025 [-0.0035, -0.0015] | 6 |
| 385M RL B1 best | 39.93 [39.75, 40.08] | 43.50 | 18.27 | 0.6550 [0.6532, 0.6569] | -0.04 [-0.11, 0.03] | -0.0012 [-0.0021, -0.0003] | 7 |
| 385M RL (mixed best = Hub main) | 39.88 [39.71, 40.03] | 43.49 | 18.09 | 0.6537 [0.6521, 0.6555] |  |  |  |
| NLLB-200-3.3B | 44.61 | -- | 23.89 | -- | -4.73 [-6.56, -3.12] (language bootstrap) |  | 1 |

#### T5. FLORES+ devtest, rows clean for the earlier 270M too

Row set `clean270`, 2112 sentences, 44 languages (AfriCOMET: 40). 

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 22.54 [20.87, 24.40] | 24.62 | 5.62 | 0.4685 [0.4502, 0.4850] | 9.61 [8.34, 10.62] | 0.0752 [0.0670, 0.0836] | 42 |
| 385M SFT-only control (E-03, +TranslatePsy) | 25.27 [23.79, 26.85] | 27.69 | 7.13 | 0.5021 [0.4834, 0.5202] | 6.87 [5.98, 7.61] | 0.0416 [0.0335, 0.0495] | 37 |
| 385M RL stage A | 31.24 [29.89, 32.47] | 34.03 | 10.27 | 0.5382 [0.5219, 0.5540] | 0.90 [0.58, 1.20] | 0.0055 [0.0020, 0.0093] | 33 |
| 385M RL B1 best | 31.81 [30.51, 33.05] | 34.69 | 10.54 | 0.5411 [0.5244, 0.5570] | 0.33 [0.10, 0.58] | 0.0026 [-0.0006, 0.0057] | 30 |
| 385M RL (mixed best = Hub main) | 32.14 [30.82, 33.35] | 35.06 | 10.69 | 0.5437 [0.5277, 0.5583] |  |  |  |
| 270M SFT (earlier, E-13 rerun) | 24.94 [23.52, 26.50] | 27.22 | 7.04 | 0.4798 [0.4625, 0.4968] | 7.20 [6.31, 7.94] | 0.0639 [0.0574, 0.0709] | 41 |
| 270M RL-final (earlier, E-13 rerun) | 31.80 [30.40, 33.05] | 34.74 | 10.46 | 0.5488 [0.5332, 0.5638] | 0.35 [0.08, 0.63] | -0.0052 [-0.0102, -0.0003] | 20 |
| NLLB-200-distilled-600M | 32.53 [30.86, 34.03] | 35.39 | 12.36 | 0.5790 [0.5612, 0.5960] | -0.39 [-1.13, 0.51] | -0.0353 [-0.0423, -0.0275] | 20 |
| NLLB-200-3.3B | 34.99 [33.35, 36.50] | 38.03 | 14.34 | 0.6013 [0.5851, 0.6170] | -2.85 [-3.57, -1.99] | -0.0576 [-0.0655, -0.0503] | 9 |

#### T5b. FLORES+ clean-for-270M rows, TranslatePsy-0.8B language set

Row set `clean270`, 912 sentences, 19 languages (AfriCOMET: 19). 

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 30.37 [28.50, 32.38] | 32.91 | 10.26 | 0.4975 [0.4768, 0.5177] | 7.79 [6.32, 9.05] | 0.0559 [0.0464, 0.0664] | 17 |
| 385M SFT-only control (E-03, +TranslatePsy) | 36.93 [35.23, 38.48] | 40.23 | 13.75 | 0.5680 [0.5462, 0.5892] | 1.22 [0.27, 2.13] | -0.0146 [-0.0234, -0.0048] | 12 |
| 385M RL stage A | 37.63 [36.25, 38.91] | 40.83 | 14.86 | 0.5508 [0.5311, 0.5694] | 0.52 [0.01, 1.04] | 0.0027 [-0.0029, 0.0082] | 11 |
| 385M RL B1 best | 37.97 [36.53, 39.26] | 41.25 | 14.97 | 0.5530 [0.5336, 0.5704] | 0.19 [-0.20, 0.59] | 0.0005 [-0.0034, 0.0044] | 10 |
| 385M RL (mixed best = Hub main) | 38.16 [36.72, 39.41] | 41.45 | 15.08 | 0.5535 [0.5351, 0.5700] |  |  |  |
| 270M SFT (earlier, E-13 rerun) | 33.87 [32.11, 35.79] | 36.68 | 12.17 | 0.5186 [0.4977, 0.5390] | 4.29 [3.03, 5.41] | 0.0349 [0.0246, 0.0460] | 16 |
| 270M RL-final (earlier, E-13 rerun) | 38.18 [36.73, 39.50] | 41.53 | 15.25 | 0.5663 [0.5484, 0.5838] | -0.02 [-0.39, 0.33] | -0.0129 [-0.0197, -0.0057] | 8 |
| NLLB-200-distilled-600M | 39.09 [37.18, 40.77] | 42.39 | 17.31 | 0.5993 [0.5811, 0.6176] | -0.94 [-1.77, 0.12] | -0.0459 [-0.0543, -0.0371] | 7 |
| NLLB-200-3.3B | 41.50 [39.77, 42.96] | 44.95 | 19.53 | 0.6217 [0.6028, 0.6391] | -3.34 [-4.28, -2.37] | -0.0682 [-0.0785, -0.0582] | 3 |
| TranslatePsy-AfriSLM-0.8B | 41.91 [40.35, 43.32] | 45.47 | 18.58 | 0.6349 [0.6157, 0.6528] | -3.75 [-4.34, -3.17] | -0.0815 [-0.0896, -0.0734] | 2 |

#### T6. NTREX-128, rows clean for the earlier 270M

Row set `clean270`, 25221 sentences, 21 languages (AfriCOMET: 21). 

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 29.03 [28.60, 29.47] | 31.77 | 9.45 | 0.5209 [0.5149, 0.5259] | 8.28 [7.97, 8.59] | 0.0639 [0.0619, 0.0659] | 21 |
| 385M SFT-only control (E-03, +TranslatePsy) | 33.78 [33.40, 34.18] | 36.97 | 13.24 | 0.5787 [0.5732, 0.5838] | 3.52 [3.29, 3.74] | 0.0061 [0.0043, 0.0079] | 16 |
| 385M RL stage A | 36.84 [36.50, 37.17] | 40.25 | 15.58 | 0.5830 [0.5777, 0.5878] | 0.47 [0.38, 0.57] | 0.0018 [0.0008, 0.0027] | 18 |
| 385M RL B1 best | 37.15 [36.83, 37.49] | 40.60 | 15.75 | 0.5841 [0.5788, 0.5889] | 0.16 [0.09, 0.23] | 0.0007 [-0.0001, 0.0015] | 15 |
| 385M RL (mixed best = Hub main) | 37.31 [36.99, 37.64] | 40.80 | 15.71 | 0.5848 [0.5796, 0.5897] |  |  |  |
| 270M SFT (earlier, E-13 rerun) | 33.18 [32.80, 33.58] | 36.27 | 12.18 | 0.5458 [0.5402, 0.5507] | 4.13 [3.88, 4.39] | 0.0389 [0.0372, 0.0407] | 21 |
| 270M RL-final (earlier, E-13 rerun) | 37.35 [37.04, 37.68] | 40.88 | 15.62 | 0.5977 [0.5928, 0.6022] | -0.04 [-0.13, 0.04] | -0.0129 [-0.0144, -0.0116] | 6 |
| NLLB-200-3.3B | 41.47 | -- | 20.03 | -- | -4.16 [-5.07, -3.17] (language bootstrap) |  | 2 |

#### T7. MAFAND-MT, rows clean for the earlier 270M

Row set `clean270`, 14276 sentences, 14 languages (AfriCOMET: 14). 

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 33.44 [32.78, 34.20] | 36.64 | 12.12 | 0.6034 [0.5998, 0.6070] | 5.93 [5.21, 6.53] | 0.0383 [0.0357, 0.0409] | 14 |
| 385M SFT-only control (E-03, +TranslatePsy) | 39.08 [38.67, 39.48] | 42.63 | 17.55 | 0.6592 [0.6560, 0.6625] | 0.30 [0.08, 0.50] | -0.0175 [-0.0197, -0.0152] | 5 |
| 385M RL stage A | 39.39 [39.01, 39.74] | 43.01 | 17.71 | 0.6439 [0.6407, 0.6469] | -0.02 [-0.13, 0.10] | -0.0022 [-0.0036, -0.0008] | 6 |
| 385M RL B1 best | 39.26 [38.81, 39.67] | 42.87 | 17.51 | 0.6429 [0.6395, 0.6460] | 0.11 [-0.09, 0.40] | -0.0012 [-0.0023, 0.0000] | 8 |
| 385M RL (mixed best = Hub main) | 39.37 [38.99, 39.72] | 43.04 | 17.60 | 0.6417 [0.6385, 0.6448] |  |  |  |
| 270M SFT (earlier, E-13 rerun) | 36.64 [36.02, 37.34] | 39.99 | 15.04 | 0.6179 [0.6144, 0.6214] | 2.73 [2.11, 3.29] | 0.0238 [0.0212, 0.0263] | 10 |
| 270M RL-final (earlier, E-13 rerun) | 40.52 [40.17, 40.90] | 44.21 | 18.70 | 0.6603 [0.6573, 0.6632] | -1.14 [-1.31, -0.98] | -0.0186 [-0.0206, -0.0165] | 1 |
| NLLB-200-3.3B | 43.99 | -- | 23.34 | -- | -4.62 [-6.47, -3.02] (language bootstrap) |  | 1 |

#### T8. FLORES+ devtest, all 1,012 rows (chrF-family only; AfriCOMET of NLLB exists only on the no_pair rows)

Row set `all`, 44528 sentences, 44 languages (AfriCOMET: 0). 

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 25.79 [25.42, 26.17] | 28.12 | 7.30 | -- | 9.27 [9.00, 9.50] |  | 44 |
| 385M SFT-only control (E-03, +TranslatePsy) | 28.75 [28.43, 29.07] | 31.38 | 9.52 | -- | 6.30 [6.09, 6.50] |  | 34 |
| 385M RL stage A | 34.34 [34.06, 34.64] | 37.31 | 13.05 | -- | 0.71 [0.64, 0.78] |  | 38 |
| 385M RL B1 best | 34.74 [34.45, 35.03] | 37.75 | 13.24 | -- | 0.32 [0.27, 0.36] |  | 33 |
| 385M RL (mixed best = Hub main) | 35.05 [34.77, 35.34] | 38.12 | 13.36 | -- |  |  |  |
| NLLB-200-distilled-600M | 36.12 [35.77, 36.44] | 39.21 | 15.23 | -- | -1.06 [-1.25, -0.87] |  | 15 |
| NLLB-200-3.3B | 38.31 [37.99, 38.62] | 41.56 | 17.22 | -- | -3.25 [-3.41, -3.08] |  | 7 |

#### T9. SmolSent, all rows

Row set `all`, 30203 sentences, 35 languages (AfriCOMET: 33). 

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 21.68 [21.47, 21.91] | 23.64 | 4.25 | 0.5254 [0.5236, 0.5271] |  |  |  |
| NLLB-200-3.3B | 36.78 | -- | 15.48 | -- |  |  |  |

#### T10. SALT, all rows

Row set `all`, 1500 sentences, 3 languages (AfriCOMET: 3). 

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 39.49 [38.38, 40.71] | 42.85 | 17.32 | 0.6828 [0.6766, 0.6902] |  |  |  |
| NLLB-200-3.3B | 51.81 | -- | 30.18 | -- |  |  |  |

#### T11a. GlotStoryBook (385M SFT only)

Row set `all`, 3426 sentences, 8 languages (AfriCOMET: 0). 

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 20.22 [19.61, 20.81] | 23.03 | 3.51 | -- |  |  |  |

#### T11b. IbomMT (385M SFT only)

Row set `all`, 2024 sentences, 2 languages (AfriCOMET: 1). 

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 12.04 [11.47, 12.58] | 13.37 | 1.80 | 0.5193 [0.5099, 0.5278] |  |  |  |

#### T11c. FLORES-MOZ (385M SFT only)

Row set `all`, 1012 sentences, 1 languages (AfriCOMET: 0). 

| system | chrF++ [95% CI] | chrF | spBLEU | AfriCOMET [95% CI] | RL minus system, chrF++ [CI] | RL minus system, AfriCOMET [CI] | langs RL above |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M SFT | 21.54 [20.50, 22.48] | 24.64 | 2.30 | -- |  |  |  |

#### T12. Published whole-devtest block (FLORES-200 devtest chrF++ of the NLLB release; our systems rescored on FLORES+ all 1,012 rows; not sentence-paired)

| system | what | languages | chrF++ macro | RL minus system | 95% CI (language bootstrap, 10,000) | languages RL above |
|---|---|---:|---:|---:|---|---:|
| 385M SFT | our rescoring, FLORES+ all 1,012 rows | 43 | 26.30 | 9.24 | [7.74, 10.75] | 43 |
| 385M SFT-only control (E-03, +TranslatePsy) | our rescoring, FLORES+ all 1,012 rows | 43 | 29.35 | 6.19 | [4.28, 8.16] | 33 |
| 385M RL stage A | our rescoring, FLORES+ all 1,012 rows | 43 | 34.91 | 0.63 | [0.44, 0.82] | 37 |
| 385M RL B1 best | our rescoring, FLORES+ all 1,012 rows | 43 | 35.25 | 0.29 | [0.19, 0.39] | 32 |
| 385M RL (mixed best = Hub main) | our rescoring, FLORES+ all 1,012 rows | 43 | 35.54 | -- |  |  |
| NLLB-200-distilled-600M | our rescoring, FLORES+ all 1,012 rows | 43 | 36.62 | -1.08 | [-1.98, -0.20] | 15 |
| NLLB-200-3.3B | our rescoring, FLORES+ all 1,012 rows | 43 | 38.63 | -3.09 | [-4.13, -2.05] | 7 |
| NLLB-200-3.3B (published) | published FLORES-200 devtest chrF++ (NLLB release metrics file, beam, 1 decimal) | 43 | 38.80 | -3.26 | [-4.30, -2.21] | 7 |
| NLLB-200-1.3B (published) | published FLORES-200 devtest chrF++ (NLLB release metrics file, beam, 1 decimal) | 43 | 38.09 | -2.55 | [-3.46, -1.62] | 7 |
| NLLB-200-distilled-1.3B (published) | published FLORES-200 devtest chrF++ (NLLB release metrics file, beam, 1 decimal) | 43 | 38.34 | -2.80 | [-3.72, -1.87] | 7 |
| NLLB-200-MoE-54.5B (published) | published FLORES-200 devtest chrF++ (NLLB release metrics file, beam, 1 decimal) | 43 | 39.08 | -3.54 | [-4.59, -2.52] | 6 |

#### T13. SSA-COMET-MTL reference-based x100 (REWARD FAMILY, NOT INDEPENDENT; FLORES+ no_pair rows; none exists for the 385M models)

| system | languages | macro on its own languages | macro on the 19 TranslatePsy-0.8B languages | macro on the 9 TranslatePsy-2B languages |
|---|---:|---:|---:|---:|
| NLLB-200-distilled-600M | 22 | 56.71 | 56.72 | 56.91 |
| NLLB-200-3.3B | 22 | 59.28 | 59.14 | 58.90 |
| TranslatePsy-AfriSLM-0.8B | 19 | 60.75 | 60.75 | 60.31 |
| TranslatePsy-AfriSLM-2B | 9 | 60.43 | -- | 60.43 |
| 270M RL-final (first research's own run, earlier code state; NOT the E-13 rerun) | 22 | 51.92 | 52.81 | 50.29 |
| 270M SFT-final (first research's own run, earlier code state; NOT the E-13 rerun) | 22 | 47.54 | 48.78 | 46.19 |

#### T14. AfriScience-MT test sentences, 6 languages (amh hau lug nso yor zul), plain chrF (word order 0)

| system | kind | chrF macro | chrF++ | AfriCOMET-STL | SSA-COMET-QE (reward family) |
|---|---|---:|---:|---:|---:|
| 385M SFT | our run, same 5,820 test sentences, our scoring | 36.0 | 33.5 | 0.6064 | 0.5686 |
| 385M RL stage A | our run, same 5,820 test sentences, our scoring | 45.06 | 42.18 | 0.6822 | 0.6118 |
| 385M RL B1 best | our run, same 5,820 test sentences, our scoring | 45.3 | 42.4 | 0.6813 | 0.6113 |
| 385M RL (mixed best = Hub main) | our run, same 5,820 test sentences, our scoring | 45.4 | 42.44 | 0.6828 | 0.613 |
| 385M SFT-only control (E-03) | our run, same 5,820 test sentences, our scoring | 44.71 | 41.82 | 0.6947 | 0.6214 |
| RL-final (1,433) (old 270M) | first research's own run, stored value (earlier code state, not rerun) | 47.9 |  |  |  |
| science-stage start A2W (old 270M) | first research's own run, stored value (earlier code state, not rerun) | 50.0 |  |  |  |
| final2 step 401 (old 270M) | first research's own run, stored value (earlier code state, not rerun) | 50.2 |  |  |  |
| NLLB-200-1.3B fine-tuned (published) | published number (the AfriScience-MT paper's scoring) | 52.3 |  |  |  |
| NLLB-200-distilled-600M fine-tuned (published) | published number (the AfriScience-MT paper's scoring) | 51.7 |  |  |  |
| M2M100-1.2B fine-tuned (published) | published number (the AfriScience-MT paper's scoring) | 48.2 |  |  |  |
| NLLB-200-1.3B zero-shot (published) | published number (the AfriScience-MT paper's scoring) | 44.1 |  |  |  |
| Gemma2-9B LoRA r64 fine-tuned (published) | published number (the AfriScience-MT paper's scoring) | 50.3 |  |  |  |
| GPT-5.4 10-shot (published) | published number (the AfriScience-MT paper's scoring) | 53.3 |  |  |  |
| Gemini-3.1-Flash-Lite 10-shot (published) | published number (the AfriScience-MT paper's scoring) | 48.8 |  |  |  |

