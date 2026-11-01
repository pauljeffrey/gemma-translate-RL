| indicator (macro % of rows) | RL-final | final run (step 401) | difference, pp [95% CI] | human references |
|:--|--:|--:|--:|--:|
| no end token in the budget | 0.27 | 0.02 | -0.25 [-0.35, -0.16] | 0.00 |
| degenerate output (v1 hard class) | 0.69 | 0.20 | -0.49 [-0.64, -0.35] | 0.00 |
| wrong language (GlotLID) | 0.47 | 0.52 | +0.05 [+0.01, +0.10] | 3.74 |
| copy of the source (Jaccard $\ge$ 0.9) | 0.08 | 0.08 | +0.00 [-0.00, +0.01] (level) | 0.05 |
| substring copy of the source | 1.49 | 1.32 | -0.17 [-0.22, -0.13] | 0.42 |
| numbers lost | 6.44 | 5.66 | -0.78 [-1.27, -0.35] | 12.03 |
| names lost | 31.78 | 33.35 | +1.57 [+1.28, +1.91] | 45.95 |
| graded entity loss | 9.20 | 9.08 | -0.12 [-0.52, +0.26] (level) | 12.90 |
| stop-word profile flag | 4.01 | 4.02 | +0.01 [-0.12, +0.14] (level) | 0.91 |
| padding flag | 11.40 | 10.08 | -1.32 [-1.57, -1.10] | 0.85 |
| repetition flag | 14.41 | 12.76 | -1.65 [-1.94, -1.36] | 1.74 |
| penalty the final reward would charge (units of 0.01 reward) | 8.27 | 6.18 | -2.09 [-2.34, -1.88] | 1.18 |
| padding / repetition, 17 weak languages | 18.60 / 23.71 | 16.26 / 20.66 | -2.34 [-2.89, -1.86] / -3.05 [-3.66, -2.45] | 0.82 / 2.10 |
| padding / repetition, 11 consensus-only languages | 22.36 / 25.96 | 18.83 / 21.89 | -3.52 [-4.52, -2.52] / -4.06 [-5.24, -2.89] | 0.65 / 1.25 |
| padding / repetition, 17 strong languages | 1.95 / 3.34 | 2.27 / 3.56 | +0.32 [+0.21, +0.43] / +0.21 [+0.07, +0.35] | 1.06 / 1.79 |
| padding / repetition, GlotStoryBook only (8 languages) | 15.45 / 20.06 | 9.28 / 12.83 | -6.18 [-7.63, -4.87] / -7.23 [-8.82, -5.65] | n/a |
