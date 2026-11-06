# Post-editor evaluation (/workspace/pe/bench_drafts/predictions_greedy_full.jsonl, editor /workspace/pe/e1_greedy_edited.jsonl, greedy)

## Paired comparison edited - draft (macro over languages, equal weight; 95% paired sentence bootstrap)

| benchmark | subset | languages | chrF++ draft | chrF++ edited | delta [CI] | spBLEU delta [CI] | langs better / ns / worse |
|---|---|---:|---:|---:|---|---|---|
| flores_plus | all | 54 | 30.48 | 30.47 | -0.01 [-0.13, +0.09] | -0.14 [-0.24, -0.02] | 11 / 26 / 17 |
| flores_plus [mixed-register] | all | 43 | 33.21 | 32.97 | -0.24 [-0.35, -0.13] | -0.27 [-0.37, -0.13] | 6 / 20 / 17 |
| flores_plus [religious-mostly] | all | 4 | 33.18 | 33.35 | +0.17 [-0.39, +0.74] | +0.41 [-0.15, +0.99] | 0 / 4 / 0 |
| flores_plus [religious-only] | all | 5 | 15.01 | 16.04 | +1.02 [+0.64, +1.42] | +0.34 [+0.13, +0.55] | 3 / 2 / 0 |
| flores_plus | clean | 54 | 30.48 | 30.47 | -0.01 [-0.11, +0.10] | -0.14 [-0.23, -0.02] | 12 / 25 / 17 |
| flores_plus [mixed-register] | clean | 43 | 33.21 | 32.97 | -0.24 [-0.35, -0.12] | -0.27 [-0.36, -0.12] | 7 / 19 / 17 |
| flores_plus [religious-mostly] | clean | 4 | 33.18 | 33.35 | +0.17 [-0.39, +0.74] | +0.41 [-0.16, +1.05] | 0 / 4 / 0 |
| flores_plus [religious-only] | clean | 5 | 15.01 | 16.04 | +1.02 [+0.63, +1.41] | +0.34 [+0.13, +0.53] | 3 / 2 / 0 |
| mafand | all | 15 | 40.45 | 39.52 | -0.93 [-1.12, -0.74] | -0.86 [-1.08, -0.59] | 1 / 8 / 6 |
| mafand [mixed-register] | all | 15 | 40.45 | 39.52 | -0.93 [-1.12, -0.74] | -0.86 [-1.08, -0.59] | 1 / 8 / 6 |
| mafand | clean | 15 | 40.45 | 39.52 | -0.93 [-1.14, -0.74] | -0.86 [-1.08, -0.60] | 1 / 7 / 7 |
| mafand [mixed-register] | clean | 15 | 40.45 | 39.52 | -0.93 [-1.14, -0.74] | -0.86 [-1.08, -0.60] | 1 / 7 / 7 |
| ntrex | all | 24 | 36.53 | 35.95 | -0.58 [-0.75, -0.42] | -0.34 [-0.50, -0.13] | 1 / 12 / 11 |
| ntrex [mixed-register] | all | 22 | 36.98 | 36.32 | -0.66 [-0.83, -0.50] | -0.39 [-0.56, -0.19] | 1 / 10 / 11 |
| ntrex [religious-mostly] | all | 2 | 31.64 | 31.94 | +0.30 [-0.33, +0.99] | +0.18 [-0.34, +1.14] | 0 / 2 / 0 |
| ntrex | clean | 24 | 36.54 | 35.95 | -0.58 [-0.75, -0.41] | -0.34 [-0.49, -0.10] | 1 / 11 / 12 |
| ntrex [mixed-register] | clean | 22 | 36.98 | 36.32 | -0.66 [-0.83, -0.49] | -0.39 [-0.54, -0.15] | 1 / 9 / 12 |
| ntrex [religious-mostly] | clean | 2 | 31.64 | 31.94 | +0.30 [-0.37, +0.93] | +0.18 [-0.34, +1.08] | 0 / 2 / 0 |
| smol_sent | all | 45 | 31.12 | 30.76 | -0.37 [-0.51, -0.23] | +0.02 [-0.17, +0.12] | 5 / 29 / 11 |
| smol_sent [mixed-register] | all | 37 | 32.31 | 31.92 | -0.39 [-0.53, -0.24] | +0.03 [-0.18, +0.16] | 4 / 24 / 9 |
| smol_sent [religious-mostly] | all | 3 | 25.54 | 25.83 | +0.29 [-0.18, +0.81] | -0.03 [-0.49, +0.36] | 0 / 3 / 0 |
| smol_sent [religious-only] | all | 5 | 25.67 | 25.07 | -0.60 [-1.19, +0.00] | -0.06 [-0.50, +0.37] | 1 / 2 / 2 |
| smol_sent | clean | 45 | 31.13 | 30.76 | -0.37 [-0.52, -0.24] | +0.02 [-0.18, +0.13] | 5 / 29 / 11 |
| smol_sent [mixed-register] | clean | 37 | 32.32 | 31.93 | -0.39 [-0.55, -0.26] | +0.03 [-0.18, +0.16] | 4 / 24 / 9 |
| smol_sent [religious-mostly] | clean | 3 | 25.54 | 25.83 | +0.29 [-0.19, +0.79] | -0.03 [-0.50, +0.37] | 0 / 3 / 0 |
| smol_sent [religious-only] | clean | 5 | 25.68 | 25.08 | -0.60 [-1.15, -0.02] | -0.06 [-0.50, +0.36] | 1 / 2 / 2 |

## What the editor did (mean over benchmark x language cells)

| unchanged | edit_rate_mean | empty_before | empty_after | loop_before | loop_after | copy_before | copy_after | wrong_lang_before | wrong_lang_after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.497 | 0.184 | 0.000 | 0.000 | 0.039 | 0.048 | 0.002 | 0.002 | 0.062 | 0.064 |

Sentence-level (chrF++): edited better 7803, worse 9165, tie 17532.

## Baselines

(no --decode-study-csv)

### Editor vs equal-compute baselines (same sentences; delta vs the draft, macro over languages, 95% paired bootstrap)

| system | benchmark | subset | chrF++ delta [CI] | spBLEU delta [CI] | langs better / ns / worse |
|---|---|---|---|---|---|
| editor (greedy) | flores_plus | all | -0.01 [-0.13, +0.09] | -0.14 [-0.24, -0.02] | 11 / 26 / 17 |
| editor (greedy) | flores_plus [mixed-register] | all | -0.24 [-0.35, -0.13] | -0.27 [-0.37, -0.13] | 6 / 20 / 17 |
| editor (greedy) | flores_plus [religious-mostly] | all | +0.17 [-0.39, +0.74] | +0.41 [-0.15, +0.99] | 0 / 4 / 0 |
| editor (greedy) | flores_plus [religious-only] | all | +1.02 [+0.64, +1.42] | +0.34 [+0.13, +0.55] | 3 / 2 / 0 |
| editor (greedy) | flores_plus | clean | -0.01 [-0.11, +0.10] | -0.14 [-0.23, -0.02] | 12 / 25 / 17 |
| editor (greedy) | flores_plus [mixed-register] | clean | -0.24 [-0.35, -0.12] | -0.27 [-0.36, -0.12] | 7 / 19 / 17 |
| editor (greedy) | flores_plus [religious-mostly] | clean | +0.17 [-0.39, +0.74] | +0.41 [-0.16, +1.05] | 0 / 4 / 0 |
| editor (greedy) | flores_plus [religious-only] | clean | +1.02 [+0.63, +1.41] | +0.34 [+0.13, +0.53] | 3 / 2 / 0 |
| editor (greedy) | mafand | all | -0.93 [-1.12, -0.74] | -0.86 [-1.08, -0.59] | 1 / 8 / 6 |
| editor (greedy) | mafand [mixed-register] | all | -0.93 [-1.12, -0.74] | -0.86 [-1.08, -0.59] | 1 / 8 / 6 |
| editor (greedy) | mafand | clean | -0.93 [-1.14, -0.74] | -0.86 [-1.08, -0.60] | 1 / 7 / 7 |
| editor (greedy) | mafand [mixed-register] | clean | -0.93 [-1.14, -0.74] | -0.86 [-1.08, -0.60] | 1 / 7 / 7 |
| editor (greedy) | ntrex | all | -0.58 [-0.75, -0.42] | -0.34 [-0.50, -0.13] | 1 / 12 / 11 |
| editor (greedy) | ntrex [mixed-register] | all | -0.66 [-0.83, -0.50] | -0.39 [-0.56, -0.19] | 1 / 10 / 11 |
| editor (greedy) | ntrex [religious-mostly] | all | +0.30 [-0.33, +0.99] | +0.18 [-0.34, +1.14] | 0 / 2 / 0 |
| editor (greedy) | ntrex | clean | -0.58 [-0.75, -0.41] | -0.34 [-0.49, -0.10] | 1 / 11 / 12 |
| editor (greedy) | ntrex [mixed-register] | clean | -0.66 [-0.83, -0.49] | -0.39 [-0.54, -0.15] | 1 / 9 / 12 |
| editor (greedy) | ntrex [religious-mostly] | clean | +0.30 [-0.37, +0.93] | +0.18 [-0.34, +1.08] | 0 / 2 / 0 |
| editor (greedy) | smol_sent | all | -0.37 [-0.51, -0.23] | +0.02 [-0.17, +0.12] | 5 / 29 / 11 |
| editor (greedy) | smol_sent [mixed-register] | all | -0.39 [-0.53, -0.24] | +0.03 [-0.18, +0.16] | 4 / 24 / 9 |
| editor (greedy) | smol_sent [religious-mostly] | all | +0.29 [-0.18, +0.81] | -0.03 [-0.49, +0.36] | 0 / 3 / 0 |
| editor (greedy) | smol_sent [religious-only] | all | -0.60 [-1.19, +0.00] | -0.06 [-0.50, +0.37] | 1 / 2 / 2 |
| editor (greedy) | smol_sent | clean | -0.37 [-0.52, -0.24] | +0.02 [-0.18, +0.13] | 5 / 29 / 11 |
| editor (greedy) | smol_sent [mixed-register] | clean | -0.39 [-0.55, -0.26] | +0.03 [-0.18, +0.16] | 4 / 24 / 9 |
| editor (greedy) | smol_sent [religious-mostly] | clean | +0.29 [-0.19, +0.79] | -0.03 [-0.50, +0.37] | 0 / 3 / 0 |
| editor (greedy) | smol_sent [religious-only] | clean | -0.60 [-1.15, -0.02] | -0.06 [-0.50, +0.36] | 1 / 2 / 2 |
| beam4 | flores_plus | all | +1.56 [+1.37, +1.73] | +1.13 [+0.99, +1.30] | 35 / 19 / 0 |
| beam4 | flores_plus [mixed-register] | all | +1.45 [+1.26, +1.61] | +1.04 [+0.88, +1.23] | 28 / 15 / 0 |
| beam4 | flores_plus [religious-mostly] | all | +2.34 [+1.45, +3.27] | +2.08 [+1.24, +3.03] | 3 / 1 / 0 |
| beam4 | flores_plus [religious-only] | all | +2.31 [+1.50, +3.02] | +1.28 [+0.85, +1.66] | 3 / 2 / 0 |
| beam4 | flores_plus | clean | +1.56 [+1.37, +1.74] | +1.13 [+0.99, +1.29] | 34 / 20 / 0 |
| beam4 | flores_plus [mixed-register] | clean | +1.45 [+1.27, +1.62] | +1.04 [+0.90, +1.23] | 27 / 16 / 0 |
| beam4 | flores_plus [religious-mostly] | clean | +2.34 [+1.41, +3.34] | +2.08 [+1.25, +3.12] | 3 / 1 / 0 |
| beam4 | flores_plus [religious-only] | clean | +2.31 [+1.54, +3.04] | +1.28 [+0.85, +1.65] | 3 / 2 / 0 |
| beam4 | ntrex | all | +1.69 [+1.41, +1.95] | +1.45 [+1.18, +1.76] | 20 / 4 / 0 |
| beam4 | ntrex [mixed-register] | all | +1.58 [+1.29, +1.84] | +1.44 [+1.15, +1.76] | 18 / 4 / 0 |
| beam4 | ntrex [religious-mostly] | all | +2.93 [+1.39, +4.52] | +1.48 [+0.59, +2.99] | 2 / 0 / 0 |
| beam4 | ntrex | clean | +1.69 [+1.42, +1.96] | +1.45 [+1.18, +1.77] | 20 / 4 / 0 |
| beam4 | ntrex [mixed-register] | clean | +1.58 [+1.32, +1.84] | +1.44 [+1.16, +1.74] | 18 / 4 / 0 |
| beam4 | ntrex [religious-mostly] | clean | +2.93 [+1.33, +4.68] | +1.48 [+0.53, +3.07] | 2 / 0 / 0 |
| mbr16 | flores_plus | all | +2.12 [+1.96, +2.27] | +0.60 [+0.47, +0.75] | 44 / 10 / 0 |
| mbr16 | flores_plus [mixed-register] | all | +1.65 [+1.49, +1.81] | +0.48 [+0.33, +0.66] | 33 / 10 / 0 |
| mbr16 | flores_plus [religious-mostly] | all | +2.10 [+1.27, +2.98] | +0.72 [+0.01, +1.59] | 4 / 0 / 0 |
| mbr16 | flores_plus [religious-only] | all | +6.38 [+5.61, +7.08] | +1.45 [+1.04, +1.80] | 5 / 0 / 0 |
| mbr16 | flores_plus | clean | +2.12 [+1.96, +2.28] | +0.60 [+0.46, +0.75] | 44 / 10 / 0 |
| mbr16 | flores_plus [mixed-register] | clean | +1.65 [+1.49, +1.83] | +0.48 [+0.34, +0.65] | 33 / 10 / 0 |
| mbr16 | flores_plus [religious-mostly] | clean | +2.10 [+1.28, +2.95] | +0.72 [+0.01, +1.66] | 4 / 0 / 0 |
| mbr16 | flores_plus [religious-only] | clean | +6.38 [+5.63, +7.08] | +1.45 [+1.03, +1.80] | 5 / 0 / 0 |
| mbr16 | ntrex | all | +1.64 [+1.37, +1.89] | +0.89 [+0.65, +1.16] | 20 / 4 / 0 |
| mbr16 | ntrex [mixed-register] | all | +1.50 [+1.24, +1.75] | +0.87 [+0.61, +1.14] | 18 / 4 / 0 |
| mbr16 | ntrex [religious-mostly] | all | +3.16 [+1.60, +4.81] | +1.14 [+0.23, +2.67] | 2 / 0 / 0 |
| mbr16 | ntrex | clean | +1.64 [+1.36, +1.91] | +0.89 [+0.64, +1.17] | 20 / 4 / 0 |
| mbr16 | ntrex [mixed-register] | clean | +1.50 [+1.23, +1.75] | +0.87 [+0.61, +1.13] | 18 / 4 / 0 |
| mbr16 | ntrex [religious-mostly] | clean | +3.15 [+1.61, +4.85] | +1.14 [+0.22, +2.71] | 2 / 0 / 0 |
