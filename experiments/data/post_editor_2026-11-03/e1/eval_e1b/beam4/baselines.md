(no --decode-study-csv)

### Editor vs equal-compute baselines (same sentences; delta vs the draft, macro over languages, 95% paired bootstrap)

| system | benchmark | subset | chrF++ delta [CI] | spBLEU delta [CI] | langs better / ns / worse |
|---|---|---|---|---|---|
| editor (greedy) | flores_plus | all | -0.91 [-1.05, -0.76] | -0.62 [-0.76, -0.52] | 3 / 17 / 34 |
| editor (greedy) | flores_plus [mixed-register] | all | -0.93 [-1.10, -0.76] | -0.64 [-0.80, -0.51] | 2 / 13 / 28 |
| editor (greedy) | flores_plus [religious-mostly] | all | -0.71 [-1.09, -0.32] | -0.35 [-0.75, -0.04] | 0 / 3 / 1 |
| editor (greedy) | flores_plus [religious-only] | all | -1.78 [-2.14, -1.41] | -0.82 [-1.08, -0.57] | 0 / 0 / 5 |
| editor (greedy) | flores_plus | clean | -0.91 [-1.05, -0.76] | -0.62 [-0.76, -0.51] | 3 / 18 / 33 |
| editor (greedy) | flores_plus [mixed-register] | clean | -0.93 [-1.10, -0.76] | -0.64 [-0.80, -0.50] | 2 / 14 / 27 |
| editor (greedy) | flores_plus [religious-mostly] | clean | -0.71 [-1.09, -0.36] | -0.35 [-0.78, -0.02] | 0 / 3 / 1 |
| editor (greedy) | flores_plus [religious-only] | clean | -1.78 [-2.19, -1.39] | -0.82 [-1.10, -0.54] | 0 / 0 / 5 |
| editor (greedy) | ntrex | all | -1.25 [-1.46, -1.04] | -0.93 [-1.17, -0.72] | 0 / 8 / 16 |
| editor (greedy) | ntrex [mixed-register] | all | -1.28 [-1.51, -1.06] | -0.99 [-1.24, -0.76] | 0 / 7 / 15 |
| editor (greedy) | ntrex [religious-mostly] | all | -0.94 [-1.81, -0.31] | -0.29 [-0.71, +0.05] | 0 / 1 / 1 |
| editor (greedy) | ntrex | clean | -1.25 [-1.49, -1.04] | -0.93 [-1.17, -0.74] | 0 / 8 / 16 |
| editor (greedy) | ntrex [mixed-register] | clean | -1.28 [-1.52, -1.06] | -0.99 [-1.25, -0.78] | 0 / 7 / 15 |
| editor (greedy) | ntrex [religious-mostly] | clean | -0.94 [-1.83, -0.27] | -0.29 [-0.70, +0.07] | 0 / 1 / 1 |
| mbr16 | flores_plus | all | +0.57 [+0.43, +0.69] | -0.53 [-0.68, -0.40] | 12 / 36 / 6 |
| mbr16 | flores_plus [mixed-register] | all | +0.21 [+0.06, +0.34] | -0.56 [-0.72, -0.40] | 7 / 30 / 6 |
| mbr16 | flores_plus [religious-mostly] | all | -0.25 [-0.82, +0.30] | -1.36 [-2.01, -0.73] | 0 / 4 / 0 |
| mbr16 | flores_plus [religious-only] | all | +4.07 [+3.58, +4.56] | +0.17 [-0.09, +0.46] | 4 / 1 / 0 |
| mbr16 | flores_plus | clean | +0.57 [+0.44, +0.69] | -0.53 [-0.67, -0.40] | 12 / 36 / 6 |
| mbr16 | flores_plus [mixed-register] | clean | +0.21 [+0.07, +0.35] | -0.56 [-0.71, -0.42] | 7 / 30 / 6 |
| mbr16 | flores_plus [religious-mostly] | clean | -0.25 [-0.79, +0.28] | -1.36 [-2.06, -0.72] | 0 / 4 / 0 |
| mbr16 | flores_plus [religious-only] | clean | +4.07 [+3.59, +4.55] | +0.17 [-0.11, +0.45] | 4 / 1 / 0 |
| mbr16 | ntrex | all | -0.05 [-0.25, +0.16] | -0.56 [-0.78, -0.36] | 2 / 20 / 2 |
| mbr16 | ntrex [mixed-register] | all | -0.08 [-0.29, +0.14] | -0.58 [-0.82, -0.37] | 1 / 19 / 2 |
| mbr16 | ntrex [religious-mostly] | all | +0.22 [-0.39, +0.80] | -0.34 [-1.00, +0.25] | 1 / 1 / 0 |
| mbr16 | ntrex | clean | -0.05 [-0.25, +0.13] | -0.56 [-0.77, -0.34] | 1 / 21 / 2 |
| mbr16 | ntrex [mixed-register] | clean | -0.08 [-0.29, +0.12] | -0.58 [-0.81, -0.34] | 1 / 19 / 2 |
| mbr16 | ntrex [religious-mostly] | clean | +0.22 [-0.34, +0.85] | -0.34 [-1.02, +0.28] | 0 / 2 / 0 |
