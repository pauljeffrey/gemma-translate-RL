(no --decode-study-csv)

### Editor vs equal-compute baselines (same sentences; delta vs the draft, macro over languages, 95% paired bootstrap)

| system | benchmark | subset | chrF++ delta [CI] | spBLEU delta [CI] | langs better / ns / worse |
|---|---|---|---|---|---|
| editor (greedy) | flores_plus | all | -4.02 [-4.26, -3.76] | -2.98 [-3.23, -2.73] | 2 / 12 / 36 |
| editor (greedy) | flores_plus [mixed-register] | all | -2.50 [-2.77, -2.24] | -2.01 [-2.31, -1.79] | 2 / 11 / 27 |
| editor (greedy) | flores_plus [religious-mostly] | all | -2.70 [-3.67, -1.80] | -3.00 [-3.74, -2.09] | 0 / 1 / 2 |
| editor (greedy) | flores_plus [religious-only] | all | -9.95 [-10.80, -8.99] | -6.42 [-7.22, -5.47] | 0 / 0 / 5 |
| editor (greedy) | flores_plus | clean | -4.02 [-4.29, -3.77] | -2.98 [-3.25, -2.72] | 2 / 12 / 36 |
| editor (greedy) | flores_plus [mixed-register] | clean | -2.50 [-2.77, -2.25] | -2.01 [-2.32, -1.75] | 2 / 11 / 27 |
| editor (greedy) | flores_plus [religious-mostly] | clean | -2.70 [-3.75, -1.78] | -3.00 [-3.85, -2.05] | 0 / 1 / 2 |
| editor (greedy) | flores_plus [religious-only] | clean | -9.95 [-10.82, -9.00] | -6.42 [-7.20, -5.49] | 0 / 0 / 5 |
| editor (greedy) | ntrex | all | -2.93 [-3.26, -2.61] | -2.80 [-3.21, -2.40] | 0 / 3 / 19 |
| editor (greedy) | ntrex [mixed-register] | all | -2.99 [-3.34, -2.66] | -2.89 [-3.32, -2.47] | 0 / 3 / 18 |
| editor (greedy) | ntrex [religious-mostly] | all | -1.67 [-3.12, -0.28] | -0.89 [-1.77, +0.02] | 0 / 0 / 1 |
| editor (greedy) | ntrex | clean | -2.93 [-3.26, -2.58] | -2.80 [-3.19, -2.35] | 0 / 3 / 19 |
| editor (greedy) | ntrex [mixed-register] | clean | -2.99 [-3.32, -2.63] | -2.90 [-3.29, -2.42] | 0 / 3 / 18 |
| editor (greedy) | ntrex [religious-mostly] | clean | -1.67 [-3.05, -0.27] | -0.89 [-1.78, -0.06] | 0 / 0 / 1 |
| mbr16 | flores_plus | all | -2.98 [-3.24, -2.74] | -3.52 [-3.80, -3.24] | 7 / 15 / 28 |
| mbr16 | flores_plus [mixed-register] | all | -1.33 [-1.59, -1.05] | -2.60 [-2.89, -2.31] | 7 / 12 / 21 |
| mbr16 | flores_plus [religious-mostly] | all | -2.38 [-3.32, -1.43] | -3.43 [-4.35, -2.40] | 0 / 1 / 2 |
| mbr16 | flores_plus [religious-only] | all | -7.20 [-7.97, -6.43] | -6.23 [-7.08, -5.39] | 0 / 2 / 3 |
| mbr16 | flores_plus | clean | -2.98 [-3.24, -2.73] | -3.52 [-3.80, -3.25] | 7 / 15 / 28 |
| mbr16 | flores_plus [mixed-register] | clean | -1.33 [-1.61, -1.07] | -2.60 [-2.91, -2.28] | 7 / 12 / 21 |
| mbr16 | flores_plus [religious-mostly] | clean | -2.38 [-3.32, -1.45] | -3.43 [-4.26, -2.46] | 0 / 1 / 2 |
| mbr16 | flores_plus [religious-only] | clean | -7.20 [-7.96, -6.47] | -6.23 [-6.97, -5.40] | 0 / 2 / 3 |
| mbr16 | ntrex | all | -2.43 [-2.77, -2.13] | -3.02 [-3.43, -2.63] | 2 / 7 / 13 |
| mbr16 | ntrex [mixed-register] | all | -2.49 [-2.84, -2.17] | -3.12 [-3.54, -2.73] | 2 / 6 / 13 |
| mbr16 | ntrex [religious-mostly] | all | -1.05 [-2.68, +0.57] | -0.96 [-2.54, +0.67] | 0 / 1 / 0 |
| mbr16 | ntrex | clean | -2.43 [-2.77, -2.10] | -3.03 [-3.42, -2.60] | 2 / 7 / 13 |
| mbr16 | ntrex [mixed-register] | clean | -2.49 [-2.84, -2.16] | -3.12 [-3.54, -2.68] | 2 / 6 / 13 |
| mbr16 | ntrex [religious-mostly] | clean | -1.06 [-2.71, +0.53] | -0.96 [-2.70, +0.52] | 0 / 1 / 0 |
