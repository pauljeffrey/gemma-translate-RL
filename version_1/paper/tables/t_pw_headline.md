| metric | languages | SFT-final | RL-final | difference [95% CI] | RL wins / ties / loses % (macro) | RL preferred % of non-tied sentences [95% CI] | languages RL clearly better / worse | `clean` subset: RL preferred % [95% CI] |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| chrF++ (all rows) | 66 | 27.74 | 31.06 | +3.32 [+3.26, +3.38] | 54.8 / 20.0 / 25.1 | 63.7 [63.5, 64.0] | 65 / 0 | 61.6 [61.1, 62.1] |
| chrF++ (rows scored by the neural metrics) | 51 | 29.54 | 32.75 | +3.21 [+3.13, +3.28] | 54.2 / 19.1 / 26.7 | 62.1 [61.8, 62.5] | 50 / 0 | 60.7 [60.2, 61.2] |
| AfriCOMET, reference-based | 51 | 0.530 | 0.589 | +0.059 [+0.058, +0.060] | 64.1 / 12.4 / 23.5 | 70.0 [69.7, 70.4] | 51 / 0 | 69.0 [68.6, 69.4] |
| SSA-COMET-MTL, reference-based | 51 | 0.415 | 0.474 | +0.059 [+0.058, +0.060] | 64.2 / 12.0 / 23.8 | 70.4 [70.1, 70.7] | 51 / 0 | 69.6 [69.2, 70.0] |
| majority of the three reference-based metrics | 51 |  |  |  | 66.4 / 10.4 / 23.2 | 70.2 [69.8, 70.5] |  | 68.9 [68.5, 69.3] |
| AfriCOMET-QE (the RL reward model) | 51 | 0.478 | 0.547 | +0.069 [+0.068, +0.070] | 67.2 / 11.3 / 21.5 | 72.9 [72.6, 73.2] | 51 / 0 | 71.7 [71.3, 72.2] |
| SSA-COMET-MTL, reference-free | 51 | 0.425 | 0.494 | +0.069 [+0.068, +0.070] | 65.3 / 10.8 / 23.9 | 70.8 [70.4, 71.1] | 51 / 0 | 70.0 [69.6, 70.4] |
