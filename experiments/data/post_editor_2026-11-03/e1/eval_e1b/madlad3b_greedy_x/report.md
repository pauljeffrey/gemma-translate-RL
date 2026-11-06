# Post-editor evaluation (/workspace/pe/bench_mt/predictions_madlad3b_greedy.jsonl, editor /workspace/pe/run_e1/out/final, greedy)

## Paired comparison edited - draft (macro over languages, equal weight; 95% paired sentence bootstrap)

| benchmark | subset | languages | chrF++ draft | chrF++ edited | delta [CI] | spBLEU delta [CI] | langs better / ns / worse |
|---|---|---:|---:|---:|---|---|---|
| flores_plus | all | 37 | 28.35 | 31.39 | +3.04 [+2.62, +3.41] | +1.63 [+1.29, +1.95] | 23 / 9 / 5 |
| flores_plus [mixed-register] | all | 31 | 27.69 | 32.84 | +5.15 [+4.74, +5.56] | +3.15 [+2.79, +3.49] | 21 / 9 / 1 |
| flores_plus [religious-mostly] | all | 3 | 30.09 | 32.43 | +2.33 [+0.97, +3.71] | +0.85 [-0.39, +2.09] | 2 / 0 / 1 |
| flores_plus [religious-only] | all | 2 | 31.90 | 15.83 | -16.08 [-17.99, -14.00] | -11.90 [-13.59, -10.05] | 0 / 0 / 2 |
| flores_plus | clean | 37 | 28.35 | 31.39 | +3.04 [+2.66, +3.44] | +1.63 [+1.29, +1.99] | 23 / 10 / 4 |
| flores_plus [mixed-register] | clean | 31 | 27.69 | 32.84 | +5.15 [+4.72, +5.56] | +3.15 [+2.76, +3.50] | 21 / 9 / 1 |
| flores_plus [religious-mostly] | clean | 3 | 30.09 | 32.43 | +2.33 [+1.02, +3.74] | +0.85 [-0.32, +2.13] | 2 / 1 / 0 |
| flores_plus [religious-only] | clean | 2 | 31.90 | 15.83 | -16.08 [-18.15, -14.05] | -11.90 [-13.58, -10.06] | 0 / 0 / 2 |
| ntrex | all | 21 | 31.29 | 35.83 | +4.54 [+4.00, +5.04] | +2.81 [+2.32, +3.28] | 13 / 7 / 1 |
| ntrex [mixed-register] | all | 21 | 31.29 | 35.83 | +4.54 [+4.00, +5.04] | +2.81 [+2.32, +3.28] | 13 / 7 / 1 |
| ntrex | clean | 21 | 31.30 | 35.84 | +4.54 [+4.01, +5.05] | +2.81 [+2.33, +3.32] | 13 / 7 / 1 |
| ntrex [mixed-register] | clean | 21 | 31.30 | 35.84 | +4.54 [+4.01, +5.05] | +2.81 [+2.33, +3.32] | 13 / 7 / 1 |

## What the editor did (mean over benchmark x language cells)

| unchanged | edit_rate_mean | empty_before | empty_after | loop_before | loop_after | copy_before | copy_after | wrong_lang_before | wrong_lang_after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.135 | 0.594 | 0.000 | 0.000 | 0.083 | 0.060 | 0.001 | 0.000 | 0.143 | 0.070 |

Sentence-level (chrF++): edited better 3211, worse 1793, tie 796.

## Baselines

(no --decode-study-csv)
