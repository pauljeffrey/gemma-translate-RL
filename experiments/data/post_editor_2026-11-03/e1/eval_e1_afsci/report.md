# Post-editor evaluation (/workspace/pe/bench_afsci/predictions_greedy.jsonl, editor /workspace/pe/run_e1/out/final, greedy)

## Paired comparison edited - draft (macro over languages, equal weight; 95% paired sentence bootstrap)

| benchmark | subset | languages | chrF++ draft | chrF++ edited | delta [CI] | spBLEU delta [CI] | langs better / ns / worse |
|---|---|---:|---:|---:|---|---|---|
| afriscience_test | all | 6 | 41.37 | 41.11 | -0.26 [-0.56, +0.00] | +0.59 [+0.12, +1.09] | 3 / 2 / 1 |

## What the editor did (mean over benchmark x language cells)

| unchanged | edit_rate_mean | empty_before | empty_after | loop_before | loop_after | copy_before | copy_after | wrong_lang_before | wrong_lang_after |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.360 | 0.239 | 0.000 | 0.000 | 0.017 | 0.053 | 0.000 | 0.000 | 0.010 | 0.016 |

Sentence-level (chrF++): edited better 2075, worse 1585, tie 2160.

## Baselines

(no --decode-study-csv)
