# Evaluation of the weak West-African group run (2026-11-03)

Question (owner): does continued RL on the weakest AfriCOMET-covered West-African languages (mos fon kbp bam wol ewe twi dyu) help? The run: 85 steps (stopped by the combined dev plateau rule at step 75, wind-down to 85) from the mixed-stage best checkpoint, prompt pool of those 8 languages only (general + news), MBR guide on for all 8 with the overhead cap removed, patch-3 monitor, KL guard 0.25, 2.1 GPU-hours. The evaluation is greedy, the same harness as the earlier reports (FLORES+ devtest per language, NTREX, MAFAND, AfriScience sentence level), data and scripts in `reports/data/eval_weak_group_2026-11-03/`.

## Verdict
**No: the weak-group run does not help.** On the 8 target languages the mean FLORES+ chrF++ change against its start point (the mixed best) is **+0.03 (95% CI -0.06 to +0.12)**, far below the pre-registered +1.0; held-out AfriCOMET changes by +0.0004 (7 languages with a score; CI -0.0011 to +0.0018); one language gains clearly (dyu +0.4), seven show no clear difference, none improves by more than 0.4. The other languages (46 in the paired test, 47 in the table below) change by -0.09 chrF++ (CI -0.13 to -0.05; three languages below -0.5: Moroccan Arabic -1.2, Tamazight -1.0, Tunisian Arabic -0.9, all outside the RL languages and untouched by this run, i.e. drift of the shared weights); NTREX is unchanged (37.16 -> 37.11), MAFAND -0.17 (40.78 -> 40.61), AfriScience test chrF -0.21 (45.40 -> 45.19, within-model series). The rule fails on its first condition (gain >= +1.0 with the CI above 0); AfriCOMET and the non-regression conditions are met.

Why it plateaued (from the run): the weak run's own dev chrF++ on the 8 languages' dev sentences was 29.02 at the start, 29.05 / 29.04 / 28.99 / 28.86 at steps 25 / 50 / 75 / 85; the combined stop fired at step 75 (two dev evaluations without a gain >= 0.15 chrF++ or AfriCOMET >= 0.004). The training reward rose a little (median -0.17 over steps 1-20 to -0.10 over steps 66-85; KL 0.033 -> 0.052), so the policy fitted the reward on these prompts without translating better on held-out sentences; these 8 languages had already stopped improving in the mixed stage. The weak run never produced a `best/` checkpoint (its dev macro never beat the start by the trainer's margin), so "weak best by the dev rule" is the start point itself and only the final checkpoint (step 85) was evaluated.

## MBR / beam guide, cost
Guide overhead median 0.44 of the step time (median step 89 s; the beam search is about 35 s of a 38 s guide time per step, the extra MBR samples about 3.6 s); the MBR guide ran on every step (skipped 0.00 of steps with the cap removed). The guide member was above the mean of the other group members in 0.82 of the groups (0.73-0.92 by language) and the best member in 0.32 (0.15-0.44); its advantage share of the step was 0.026. The guide's mean reward rose from 0.06 (steps 1-20) to 0.15 (steps 66-85) while the sampled members stayed at -0.17 / -0.16: the guide is stronger than the policy's samples, but this did not move the greedy dev scores. Per-language guide table below.

## Table 1. Weak-group run versus its start point (mixed best), FLORES+ devtest, greedy

| model vs start | group | languages | d chrF++ [95% CI, sentence bootstrap] | d spBLEU | d AfriCOMET (mean over languages with a score) [95% CI over languages] |
|---|---|---:|---|---:|---|
| weakfinal - mixedbest | target8 | 8 | +0.03 [-0.06, +0.12] | -0.01 | +0.0004 [-0.0011, +0.0018] (n=7) |
| weakfinal - mixedbest | other | 46 | -0.09 [-0.13, -0.05] | -0.05 | -0.0013 [-0.0021, -0.0006] (n=34) |

## Table 2. The 8 target languages, FLORES+ devtest (chrF++ / AfriCOMET)

| iso | start (mixed best) chrF++ | weak run (= final, step 85) | weak run (same) | change [CI] | change [CI] (same) | COMET start / weak / weak | old 270M SFT / RL (stored, FLORES contaminated) |
|---|---:|---:|---:|---|---|---|---|
| mos | 22.8 | 22.9 | 22.9 | +0.2 [-0.1, +0.4] | +0.2 [-0.1, +0.4] | 0.548 / 0.546 / 0.546 | 16.1 / 22.9 |
| fon | 22.2 | 22.3 | 22.3 | +0.1 [-0.0, +0.2] | +0.1 [-0.0, +0.2] | 0.603 / 0.604 / 0.604 | 19.4 / 21.6 |
| kbp | 24.9 | 24.9 | 24.9 | +0.0 [-0.3, +0.4] | +0.0 [-0.3, +0.4] | 0.661 / 0.662 / 0.662 | 7.5 / 21.7 |
| bam | 27.2 | 27.3 | 27.3 | +0.1 [-0.1, +0.3] | +0.1 [-0.1, +0.3] | 0.521 / 0.524 / 0.524 | 15.7 / 24.9 |
| wol | 30.6 | 30.4 | 30.4 | -0.1 [-0.4, +0.1] | -0.1 [-0.4, +0.1] | 0.653 / 0.655 / 0.655 | 27.1 / 29.6 |
| ewe | 31.2 | 31.1 | 31.1 | -0.1 [-0.3, +0.1] | -0.1 [-0.3, +0.1] | 0.608 / 0.606 / 0.606 | 29.3 / 31.0 |
| twi | 35.1 | 34.7 | 34.7 | -0.3 [-0.8, +0.0] | -0.3 [-0.8, +0.0] | 0.648 / 0.645 / 0.645 | 33.9 / 37.3 |
| dyu | 19.0 | 19.4 | 19.4 | +0.4 [+0.3, +0.6] | +0.4 [+0.3, +0.6] | - / - / - | 11.8 / 18.3 |

Per-language wins / ties / losses on the 8 targets (CI of the chrF++ difference excludes 0): weakfinal: 1 better, 7 no clear difference, 0 worse

## Table 3. Non-regression on the other languages (FLORES+, chrF++, per-language difference to the start)

| model | languages | mean change | languages worse than -0.5 | worst 5 |
|---|---:|---:|---:|---|
| weakfinal | 47 | -0.09 | 3 | ary -1.2, zgh -1.0, aeb -0.9, xho -0.5, dik -0.5 |

## Table 4. NTREX, MAFAND (macro chrF++ over each benchmark's languages; target languages inside them separately), AfriScience test chrF (within-model series only)

| system | NTREX all | NTREX target langs | MAFAND all | MAFAND target langs | AfriScience test chrF |
|---|---:|---:|---:|---:|---:|
| sft | 29.25 (25) | 24.78 (2) | 35.77 (16) | 28.58 (1) | 36.00 |
| stageA | 36.68 (25) | 30.74 (2) | 40.82 (16) | 33.85 (1) | 45.06 |
| b1best | 36.98 (25) | 30.80 (2) | 40.81 (16) | 33.99 (1) | 45.30 |
| mixedbest | 37.16 (25) | 30.97 (2) | 40.78 (16) | 34.14 (1) | 45.40 |
| weakfinal | 37.11 (25) | 31.10 (2) | 40.61 (16) | 34.09 (1) | 45.19 |

The old-270M columns are not shown for AfriScience (not comparable until the 385M's science stage).

## Table 5. The weak run's own dev series (8 languages' dev sentences)

| step | dev chrF++ macro | weak_dev chrF++ | held-out AfriCOMET | finished |
|---:|---:|---:|---:|---:|
| 0 | 29.02 | nan | nan | 1.000 |
| 25 | 29.05 | 25.77 | 0.6349 | 1.000 |
| 50 | 29.04 | 25.77 | 0.6339 | 1.000 |
| 75 | 28.99 | 25.62 | 0.6299 | 1.000 |
| 85 | 28.86 | 25.78 | 0.6341 | 1.000 |

Steps 85; guide overhead median 0.44; guide better than the other members in 0.82 of groups on average, best member in 0.32; guide advantage share mean 0.026; MBR skipped in 0.00 of steps.

Stop reason: dev_plateau; early_stop_dev: {"triggered": true, "evidence": {"fire": true, "blocked": null, "step": 75, "stage": "mixed", "steps_in_stage": 75, "min_steps": 75, "k": 2, "eps": 0.15, "stale_evals": 2, "anchor_macro": 29.048244932906783, "anchor_weak": 25.772351774130545, "anchor_comet": 0.6349395751953124, "require_comet": true; hacking alarm checks file: none.

Guide by language (MBR guide, every guided prompt): n, win rate (guide above the group mean), best rate, guide reward vs group mean

| iso | mode | n | win rate | best rate | reward guide | reward group mean | src counts |
|---|---|---:|---:|---:|---:|---:|---|
| mos | mbr | 114 | 0.81 | 0.42 | -0.06 | -0.32 | {'mbr_beam': 50, 'mbr_member->beam': 28, 'mbr_extra': 36} |
| fon | mbr | 99 | 0.83 | 0.38 | 0.41 | 0.18 | {'mbr_extra': 29, 'mbr_beam': 41, 'mbr_member->beam': 29} |
| kbp | mbr | 124 | 0.92 | 0.44 | -0.22 | -0.46 | {'mbr_member->beam': 24, 'mbr_extra': 59, 'mbr_beam': 41} |
| bam | mbr | 112 | 0.77 | 0.20 | -0.14 | -0.28 | {'mbr_member->beam': 34, 'mbr_extra': 45, 'mbr_beam': 33} |
| wol | mbr | 99 | 0.73 | 0.15 | 0.34 | 0.16 | {'mbr_member->beam': 22, 'mbr_extra': 27, 'mbr_beam': 50} |
| ewe | mbr | 111 | 0.84 | 0.23 | 0.36 | 0.02 | {'mbr_member->beam': 31, 'mbr_extra': 40, 'mbr_beam': 40} |
| twi | mbr | 35 | 0.86 | 0.34 | 0.90 | 0.46 | {'mbr_beam': 15, 'mbr_extra': 13, 'mbr_member->beam': 7} |
| dyu | mbr | 108 | 0.82 | 0.36 | 0.05 | -0.26 | {'mbr_extra': 45, 'mbr_member->beam': 24, 'mbr_beam': 38, 'mbr_few->beam': 1} |

## Verdict (pre-registered rule)

* **weakfinal vs mixedbest: DOES NOT MEET THE RULE**: target-8 mean chrF++ +0.03 (CI -0.06 to +0.12; needs >= +1.0 with the CI above 0: no); target AfriCOMET mean change +0.0004 (needs >= -0.005: yes); other-language mean chrF++ change -0.09 (needs >= -0.3: yes).


## Caveats
* One seed, greedy decoding; chrF++ intervals are paired sentence bootstraps, AfriCOMET intervals resample the language means (per-sentence AfriCOMET scores were not matched to the predictions), so they describe language-to-language spread, not sentence noise. dyu has no held-out AfriCOMET score.
* "Weak best" does not exist (see above); in the tables the columns "weak best" and "weak final" are the same checkpoint (step 85).
* Old-270M numbers (stored runs, FLORES+ contaminated for the old models) are shown only in Table 2; AfriScience is a within-model series until the 385M's science stage.
* The start point was chosen as the mixed best (dev 40.90 against B1 best 40.74); the other 46 languages were never in the weak run's prompts or dev set, so their change measures drift only.

## Decision implied
Continued RL on a weak group (8 languages, 2.1 GPU-h) gave no measurable gain over the mixed stage, so the next compute goes to the science stage (SFT then RL, already running) rather than a second weak-group round.
