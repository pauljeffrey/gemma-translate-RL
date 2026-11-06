# Post-editor evaluation (/workspace/pe/bench_drafts/predictions_greedy_full.jsonl, editor /workspace/pe/e1_gated.jsonl, greedy)

## Paired comparison edited - draft (macro over languages, equal weight; 95% paired sentence bootstrap)

| benchmark | subset | languages | chrF++ draft | chrF++ edited | delta [CI] | spBLEU delta [CI] | langs better / ns / worse |
|---|---|---:|---:|---:|---|---|---|
| flores_plus | all | 54 | 30.48 | 31.18 | +0.70 [+0.62, +0.78] | +0.31 [+0.25, +0.41] | 18 / 35 / 1 |
| flores_plus [mixed-register] | all | 43 | 33.21 | 33.71 | +0.51 [+0.44, +0.57] | +0.23 [+0.17, +0.34] | 9 / 33 / 1 |
| flores_plus [religious-mostly] | all | 4 | 33.18 | 33.92 | +0.74 [+0.30, +1.25] | +0.79 [+0.31, +1.30] | 2 / 2 / 0 |
| flores_plus [religious-only] | all | 5 | 15.01 | 16.95 | +1.94 [+1.59, +2.27] | +0.64 [+0.45, +0.84] | 5 / 0 / 0 |
| flores_plus | clean | 54 | 30.48 | 31.18 | +0.70 [+0.63, +0.78] | +0.31 [+0.26, +0.42] | 18 / 35 / 1 |
| flores_plus [mixed-register] | clean | 43 | 33.21 | 33.71 | +0.51 [+0.44, +0.58] | +0.23 [+0.18, +0.35] | 9 / 33 / 1 |
| flores_plus [religious-mostly] | clean | 4 | 33.18 | 33.92 | +0.74 [+0.29, +1.27] | +0.79 [+0.34, +1.35] | 2 / 2 / 0 |
| flores_plus [religious-only] | clean | 5 | 15.01 | 16.95 | +1.94 [+1.60, +2.27] | +0.64 [+0.45, +0.82] | 5 / 0 / 0 |
| mafand | all | 15 | 40.45 | 40.60 | +0.15 [+0.06, +0.27] | +0.11 [+0.00, +0.33] | 1 / 14 / 0 |
| mafand [mixed-register] | all | 15 | 40.45 | 40.60 | +0.15 [+0.06, +0.27] | +0.11 [+0.00, +0.33] | 1 / 14 / 0 |
| mafand | clean | 15 | 40.45 | 40.60 | +0.15 [+0.06, +0.27] | +0.11 [+0.01, +0.32] | 1 / 14 / 0 |
| mafand [mixed-register] | clean | 15 | 40.45 | 40.60 | +0.15 [+0.06, +0.27] | +0.11 [+0.01, +0.32] | 1 / 14 / 0 |
| ntrex | all | 24 | 36.53 | 36.83 | +0.29 [+0.19, +0.39] | +0.23 [+0.13, +0.41] | 5 / 19 / 0 |
| ntrex [mixed-register] | all | 22 | 36.98 | 37.23 | +0.25 [+0.15, +0.34] | +0.21 [+0.11, +0.38] | 4 / 18 / 0 |
| ntrex [religious-mostly] | all | 2 | 31.64 | 32.42 | +0.78 [+0.28, +1.40] | +0.40 [-0.01, +1.40] | 1 / 1 / 0 |
| ntrex | clean | 24 | 36.54 | 36.83 | +0.29 [+0.19, +0.40] | +0.23 [+0.13, +0.44] | 5 / 19 / 0 |
| ntrex [mixed-register] | clean | 22 | 36.98 | 37.23 | +0.25 [+0.15, +0.36] | +0.21 [+0.11, +0.41] | 4 / 18 / 0 |
| ntrex [religious-mostly] | clean | 2 | 31.64 | 32.42 | +0.78 [+0.29, +1.33] | +0.40 [-0.01, +1.30] | 1 / 1 / 0 |
| smol_sent | all | 45 | 31.12 | 31.59 | +0.47 [+0.36, +0.56] | +0.53 [+0.37, +0.62] | 11 / 34 / 0 |
| smol_sent [mixed-register] | all | 37 | 32.31 | 32.69 | +0.38 [+0.28, +0.48] | +0.49 [+0.33, +0.60] | 7 / 30 / 0 |
| smol_sent [religious-mostly] | all | 3 | 25.54 | 26.10 | +0.57 [+0.23, +0.95] | +0.48 [+0.16, +0.75] | 2 / 1 / 0 |
| smol_sent [religious-only] | all | 5 | 25.67 | 26.75 | +1.08 [+0.57, +1.59] | +0.80 [+0.44, +1.17] | 2 / 3 / 0 |
| smol_sent | clean | 45 | 31.13 | 31.59 | +0.47 [+0.37, +0.57] | +0.53 [+0.37, +0.61] | 11 / 34 / 0 |
| smol_sent [mixed-register] | clean | 37 | 32.32 | 32.69 | +0.38 [+0.28, +0.48] | +0.49 [+0.32, +0.60] | 7 / 30 / 0 |
| smol_sent [religious-mostly] | clean | 3 | 25.54 | 26.11 | +0.57 [+0.23, +0.98] | +0.48 [+0.16, +0.78] | 2 / 1 / 0 |
| smol_sent [religious-only] | clean | 5 | 25.68 | 26.76 | +1.08 [+0.62, +1.61] | +0.80 [+0.45, +1.18] | 2 / 3 / 0 |

## What the editor did (mean over benchmark x language cells)

| unchanged | edit_rate_mean | empty_before | empty_after | loop_before | loop_after | copy_before | copy_after | wrong_lang_before | wrong_lang_after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.961 | 0.025 | 0.000 | 0.000 | 0.039 | 0.029 | 0.002 | 0.002 | 0.062 | 0.061 |

Sentence-level (chrF++): edited better 947, worse 313, tie 33240.

## Baselines

(no --decode-study-csv)
