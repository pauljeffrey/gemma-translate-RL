# Ledger snippet E-11a (source report: `reports/stage_A_feature_split_2026-11-04.md`, data `reports/data/stage_A_feature_split_2026-11-04/`; key SPLIT)

Add key to the report-key table: `SPLIT | reports/stage_A_feature_split_2026-11-04.md (data reports/data/stage_A_feature_split_2026-11-04/)`.

## Replace / amend M10 caveat (append)

"E-11a (SPLIT): the STA slopes reproduce (KL 0.00053 per step over steps 2-40 [0.00043, 0.00063], 0.00143 over steps 41-100 [0.00128, 0.00156]; block-bootstrap CIs), but the pre-feature segment contains the LR warm-up (steps 1-10): over steps 11-40 the slope is 0.00070 [0.00061, 0.00079], i.e. 2.0x not 2.7x, and the KL was already accelerating before step 41 (15-step windows: 11-25 0.00048, 26-40 0.00101, 41-55 0.00158). Per-step slopes of reward (+0.0065 pre, +0.0071 post), finished, loop and hard-fail do not speed up after step 41. The four features cannot be separated."

## New rows (section 4 or 5)

| ID | claim | exact value | source | Ind. | status | caveat |
|---|---|---|---|---|---|---|
| R17 | Stage A KL slope before and after the features (step 41): STA numbers reproduced from `metrics.jsonl` | KL per step: steps 2-40 0.00053 [0.00043, 0.00063]; steps 41-100 0.00143 [0.00128, 0.00156]; steps 11-40 (after LR warm-up) 0.00070 [0.00061, 0.00079]; steps 101-225 0.00011 [0.00007, 0.00015]; steps 248-400 0.00000 [-0.00002, 0.00002]; 15-step windows 11-25 / 26-40 / 41-55 / 56-70 / 71-85 / 86-100: 0.00048 / 0.00101 / 0.00158 / 0.00145 / 0.00184 / 0.00067; KL median 0.0429 (steps 31-40) vs 0.0566 (41-50) | SPLIT (`slopes.csv`, `window15_slopes.csv`, `summary.json`) | LOOP (training log, not a benchmark) | supported (as description) | block-bootstrap CIs (5-step blocks, residual lag-1 autocorrelation -0.4 to +0.9); KL rise precedes step 41 and saturates by step ~90 (0.124-0.129 until step ~160); a pre-trend extrapolation (power law in cumulative LR or in step, fits on steps 2-40 / 11-40) predicts a 41-100 slope of 0.00007-0.00036 and a step-100 KL of 0.042-0.065 versus observed 0.00143 and 0.131 (`kl_counterfactual.json`); does not identify the cause |
| R18 | Reward, finished, loop, hard-fail, gradient-norm per-step slopes do not speed up at step 41 | reward per step +0.0065 [0.0051, 0.0079] (steps 2-40) vs +0.0071 [0.0060, 0.0080] (41-100); greedy loop -0.0020 vs -0.0011; hard-fail -0.00138 vs -0.00040; greedy finished +0.0022 vs +0.0010; interrupted time series level shift at step 41 (window 2-100): reward +0.093 [0.041, 0.147], greedy loop -0.028 [-0.052, -0.004], hard-fail -0.014 [-0.026, -0.003], KL +0.0086 [0.0021, 0.0149]; finished rate and greedy finished shifts cover 0 | SPLIT (`slopes.csv`, `its.csv`, `local_window_31_40_vs_41_50.csv`) | LOOP | weak | cannot be attributed to any feature: all four started together at step 41 (and the run was stopped and resumed from `last/` at step 40); `reward` itself is not the same quantity before and after 41 (reference-anchored term on 15% of prompts, guide member in 25% of groups); dev chrF++ is only evaluated at steps 0, 50, ... so no dev point at step 40 |

## Status line for EXPERIMENTS_LEFT (E-11a)

E-11a | DONE 2026-11-04 (CPU, laptop): `reports/stage_A_feature_split_2026-11-04.md`. KL slopes reproduce exactly (0.00053 -> 0.00143 per step) but become 2.0x after excluding the LR warm-up and the KL was already accelerating before step 41; reward / finished / loop / hard-fail per-step slopes do not accelerate at step 41; the four features cannot be separated; E-11 stays required if any feature is called beneficial.
