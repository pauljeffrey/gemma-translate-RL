# E-11a: RL stage A before and after step 41 (the four features), from the run log

Date 2026-11-04. CPU only, laptop. Source: `research_archive_2026-11-03/runs_385m/A/metrics.jsonl` (400 training rows, steps 1-400, plus 9 dev rows at steps 0, 50, ..., 400), `kl_guard_events.jsonl`, `group_stats.jsonl` (starts at step 41), `summary.json`. Script, tables and figure: `reports/data/stage_A_feature_split_2026-11-04/` (`analyse_feature_split.py` -> `slopes.csv`, `its.csv`, `per10.csv`, `window15_slopes.csv`, `local_window_31_40_vs_41_50.csv`, `kl_counterfactual.json`, `summary.json`, `fig_feature_split.png`). Nothing was run on the A100.

What the run did (STA, `reports/rl_stage_A_385m_2026-11-03.md`, and the log): steps 1-40 without the four features (dynamic sampling, advantage std floor, beam-guided member, reference domain); at the step-40 checkpoint the trainer was stopped and resumed from `last/` with all four switched on at step 41. LR warm-up inside the log: lr 2e-7 at step 1, 4e-7 at step 2, 1e-6 at step 5, 2e-6 from step 10 on; KL-guard cuts at steps 226, 236, 247 (lr 1e-6, 5e-7, 2.5e-7 from step 248 on). The `kl_guard_events.jsonl` has a "restore" event at step 40 (the guard state was restored on resume).

## 1. The ledger number reproduces

OLS slope of the per-step KL (k3): **0.00053 per step over steps 2-40** (block-bootstrap 95% CI [0.00043, 0.00063], n = 39) and **0.00143 per step over steps 41-100** ([0.00128, 0.00156], n = 60). Both equal the numbers in STA / ledger M10 / EXPERIMENTS_LEFT E-11 exactly. The KL medians per 10 steps also match STA (0.028, 0.034, 0.043 before; 0.057, 0.074, 0.089, 0.107, 0.122 after).

But the "tripled at step 41" wording is not what the log shows:

* The pre-feature segment contains the LR warm-up (lr ramps over steps 1-10). Excluding it (steps 11-40) the pre slope is **0.00070 [0.00061, 0.00079]**, so the ratio is 2.0x, not 2.7x.
* KL was already accelerating before step 41. Slope in consecutive 15-step windows (`window15_slopes.csv`): steps 11-25 0.00048, **26-40 0.00101**, 41-55 0.00158, 56-70 0.00145, 71-85 0.00184, 86-100 0.00067, 101-115 0.00020, 116-130 -0.00012. The curve is smooth through step 41 (figure, KL panel); there is no kink at the switch. The KL level on steps 31-40 is 0.0429 (sd 0.0037) and on steps 41-50 0.0566 (sd 0.0053).
* A counterfactual from the pre-feature trend (power law in cumulative learning rate or in step, fitted on steps 2-40 or 11-40, extrapolated to 41-100; `kl_counterfactual.json`) predicts a slope over 41-100 of 0.00007 to 0.00036 and a step-100 KL of 0.042 to 0.065; observed slope 0.00143, KL 0.131 at step 100 (median over steps 96-100: 0.125). So the acceleration is faster than the extrapolated pre-trend under any of the four functional forms tried. This is a statement about extrapolating 30-40 noisy points with an assumed functional form, not an identification of a cause: a policy that has moved away from the reference accumulates KL faster than the early warm-up phase suggests, and the pre-trend already shows convexity.
* After step ~90 the KL saturates (flat 0.124-0.129 until step ~160), long before the first LR cut at step 226. Slopes: steps 101-225 0.00011 [0.00007, 0.00015]; steps 248-400 (LR 2.5e-7) 0.00000 [-0.00002, 0.00002] (median 0.149).

## 2. Other metrics before and after step 41

Per-step OLS slopes with residual-block bootstrap CIs (5-step blocks, 2,000 resamples; residual lag-1 autocorrelation is in `slopes.csv`: from -0.4 to +0.9 depending on metric and segment, so naive OLS intervals would be too narrow, hence the block bootstrap). Full table in `slopes.csv`.

| metric (per step) | pre 2-40 | pre 11-40 | post 41-100 | post 101-225 | post 248-400 |
|---|---:|---:|---:|---:|---:|
| KL | +0.00053 | +0.00070 | +0.00143 | +0.00011 | 0.00000 |
| training reward | +0.0065 [0.0051, 0.0079] | +0.0085 | +0.0071 [0.0060, 0.0080] | +0.0007 | -0.0002 (n.s.) |
| greedy-member reward | +0.0063 | +0.0082 | +0.0069 | +0.0006 | -0.0002 (n.s.) |
| finished rate (sampled) | +0.00014 | +0.00014 | +0.00006 | +0.00001 | 0 |
| finished rate (greedy member) | +0.0022 | +0.0023 | +0.0010 | +0.0002 | +0.00001 (n.s.) |
| loop rate (greedy member) | -0.0020 | -0.0023 | -0.0011 | -0.00014 | 0.00000 (n.s.) |
| hard-fail share | -0.00138 | -0.00195 | -0.00040 | -0.00006 | +0.00003 |
| gradient norm | -0.0078 | -0.0080 | -0.0043 | -0.0010 | -0.0007 |
| completion length | +0.045 (n.s.) | +0.104 | 0.000 (n.s.) | -0.0006 (n.s.) | +0.005 (n.s.) |

The reward slope per step is the same before and after step 41 (+0.0065 to +0.0085 versus +0.0071); the loop, finished and hard-fail improvement per step is, if anything, steeper BEFORE the features (loop -0.0020 versus -0.0011) because the early failure rates were high and the rates floor out. Nothing in the slopes shows a speed-up of reward, finishing or looping at step 41; the only metric that accelerates is KL, and it was already accelerating.

Local window, steps 31-40 versus 41-50 (`local_window_31_40_vs_41_50.csv`): reward -0.949 -> -0.833 (+0.116), greedy loop 0.139 -> 0.099, hard-fail 0.063 -> 0.040, greedy finished 0.881 -> 0.915, KL +0.014; these 10-step changes are of the size of the pre-feature trend (+0.0085 x 10 = +0.085 for the reward) plus a small residual.

Interrupted time series, `y = a + b t + c [t>=41] + d (t-41)[t>=41]` (`its.csv`; HAC Newey-West 5 lags and a residual-block bootstrap; window 2-100): level shift c at step 41 is KL +0.0086 [0.0021, 0.0149], reward +0.093 [0.041, 0.147], greedy-member reward +0.136 [0.052, 0.222], greedy loop rate -0.028 [-0.052, -0.004], hard-fail -0.014 [-0.026, -0.003], gradient norm +0.157 [0.028, 0.292], finished rate +0.0007 (CI covers 0), greedy finished +0.015 (CI covers 0); slope change d: KL +0.00090 [0.00065, 0.00114], reward +0.0005 (CI covers 0). Over window 2-225 the post-41 reward slope change is negative (-0.0048 [-0.0086, -0.0010]) because the curve saturates. Read these with care: (i) 39 pre points and a linear pre-trend extrapolated across a saturating curve; (ii) the reward and the finished/loop rates are not the same quantity before and after step 41: from step 41 15% of the prompts carry the reference-anchored reward term (STA: mean reward shift -0.21 on those rows), 25% of the groups contain the guide member (guide mean reward -0.19 versus -0.61 for sampled members in steps 41-100), and `reward`, `finished_rate`, `completion_len` may include the guide in those groups; (iii) the step-41 level shifts could equally be a resume artefact (checkpoint reload, RNG, a fresh prompt stream), which the log cannot exclude. The significant level shifts in KL, reward and loop rate therefore say "something changed at 41 on top of the trend", not which of the four features did it.

Dev curve (`bench/chrf_macro`, 720 sentences; only 9 evaluations, none between steps 0 and 50, so no dev point at 40): 36.81 (step 0), 39.04 (50), 40.14 (100), 40.13 (150), 40.47 (200), 40.43 (250), 40.46 (300), 40.47 (350), 40.50 (400); finished 0.933 -> 0.968 -> 0.988 -> ... -> 0.997. The dev data cannot split steps 1-40 from 41-50.

## 3. What can and cannot be concluded

1. The four features started together at step 41 and no per-feature quantity before step 41 exists, so the log cannot attribute any change to any one feature (STA says the same). `group_stats.jsonl` and all `guide/*`, `ds/*`, `reward/ref*` columns exist only from step 41; there is no pre-feature value to compare against.
2. Dynamic sampling and the std floor never acted (STA: flat fraction 0, group-std p10 0.128 versus floor 0.02), so the changes at 41 can only come from the beam guide and the reference domain (or from the resume itself).
3. What the log supports: KL growth is faster in steps 41-100 than in steps 2-40 (reproduced), but it was already accelerating in steps 26-40 and saturates by step ~90; reward, finished, loop and hard-fail improved at about the same or a slower per-step rate after 41 than before. "The KL slope tripled when the features began" overstates it: 2.0x excluding the warm-up, with an acceleration that precedes step 41.
4. The paper may describe the four features as design choices; it may state that the KL rose faster after they were added; it may not call any of them beneficial or the cause of the KL rise. E-11 (arms: no-feature baseline, no reference term, no guide) stays required if a feature is called beneficial. This free analysis does not replace it.

## 4. Files

* Script: `reports/data/stage_A_feature_split_2026-11-04/analyse_feature_split.py`
* Tables: `slopes.csv`, `its.csv`, `per10.csv`, `window15_slopes.csv`, `local_window_31_40_vs_41_50.csv`, `kl_counterfactual.json`, `summary.json`; figure `fig_feature_split.png`; ledger rows `ledger_snippet.md`.
