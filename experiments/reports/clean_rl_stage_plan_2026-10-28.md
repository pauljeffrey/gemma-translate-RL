# RL stage plan and configs for the clean rerun (agent C1, 2026-10-28)

Nothing was run on a GPU. Labels: VERIFIED = computed or read from a named file; REPORTED = from a document; ESTIMATED = arithmetic or assumption. Configs are in `gemma_mt/configs/clean_rerun/` (specialist copies in `clean_rerun_specialist/`), tools in `gemma_mt/scripts/` (`clean_rerun_make_configs.py` generates the yamls: edit it, not the yamls; `clean_rerun_tools.py`: validate, manifest, gate, winddown, handoff, disclose; `launch_clean_rerun.sh`). CSVs: `reports/data/clean_rl_stage_plan_2026-10-28/`.

## 0. Result in brief
* Seven yamls, a launcher and a validator; all load strictly against the real `RLConfig`, `RewardConfig`, `validate_config`. Total 760,000 completions from stage A through the final run (preflight excluded).
* Caps are below M1's central schedule; RL max is about 37% below M1's high case. Stage A 400 steps (M1 640), b256 150, b512 80 (M1 120), b1024 60, mixed 60, final 150.
* Cost, central: 18.8 GPU-h for RL stages plus 0.47 h preflight, USD 25.2 with 10% margin. Max (hard caps on a contended card): 36.6 h / USD 43.5 with 20% margin.
* NOT tested against the live trainer. The launcher was tested only with a fake trainer; the manifest ran on the laptop and its pins matched.
* `kl_guard_restore` is true only for preflight and stage A, false from B1 on. `plateau_stop` is on only for A, B1 and the final run (other caps are shorter than the 100-step window; the validator needs `plateau_stop_min_steps` >= 100). `plateau_stop_ci_high_max` 0.06.

## 1. Stage table
All stages: group 16 (15 sampled + 1 greedy), `greedy_in_group` true, `greedy_stats` sampled, `greedy_adv_scale` 0.3, `greedy_adv_cap` 3, std-normalised advantages, k3c, grad clip 1.0, full reward including consensus (QE ensemble, anchors, longdoc_v2, LID guard). NLLB signal and science-anchor penalty only in the final run.

| stage | yaml | cap steps | prompts/step | completions at cap | lr, warm-up | KL coef | KL guard | plateau stop | pool | reference |
|---|---|---|---|---|---|---|---|---|---|---|
| preflight | rl_00_preflight | 49 | 64 | 50,176 | 2e-6, 10 | 0.05 | 0.15 | off | clean_v3 b64 | SFT-final of winning arm |
| A b64 | rl_A_b64 | 400 | 64 | 409,600 | 2e-6, 10 | 0.05 | 0.15 | on, min 150 | clean_v3 b64 | SFT-final |
| B1 b256 | rl_B1_b256 | 150 | 48 | 115,200 | 2e-6, 10 | 0.05 | 0.15 | on, min 100 | clean_v3 b256 | A final/ |
| B2 b512 | rl_B2_b512 | 80 | 42 | 53,760 | 2e-6, 10 | 0.05 | 0.15 | off | clean_v3 b512 | B1 final/ |
| B3 b1024 | rl_B3_b1024 | 60 | 42 | 40,320 | 2e-6, 10 | 0.05 | 0.15 | off | clean_v3 b1024 | B2 final/ |
| C mixed | rl_C_mixed | 60 | 42 | 40,320 | 2e-6, 10 | 0.05 | 0.15 | off | clean_v3, 4 bins uniform | B3 final/ |
| F final | rl_F_final | 150 | 42 | 100,800 | 1e-6, 20 | 0.1 | 0.25 | on, min 100 | clean_sci_v2, 70/30 | science-stage model (`SCI_MODEL`) |

Across stages:
* Each stage after preflight/A is a new run (new out_dir, fresh optimizer, `model_name` = previous stage's final model), as the 2c restart. The final run also uses the science-stage model as KL reference.
* KL guard: window 5, factor 0.5, floor 1/8, cooldown 10. `kl_guard_recover` false in stages, true in the final run (1.25 / 0.5 / 20).
* Plateau stop: window 100, block 20, min_finished 0.99, winddown 10, min_lr_scale 0.5, bootstrap 2000, ci_high_max 0.06. `greedy_share_warn` 0.35.
* The launcher's watcher writes `WINDDOWN 30` at step 120 of the final run; never overrides an existing WINDDOWN.
* `max_src_tokens` 1024 everywhere (source-side cap, Amendment 2; b2048 dropped). `max_new_tokens_by_bin`: b64 384, b256 1280, b512 2304, b1024 3584, mixed 3584. `vllm_max_model_len` 8192 (10240 in the final run).
* Mixed stage and final run: `curriculum_steps` {b64: 0} = mixed from step 0. `language_alpha` 0; the 25 dev-defined weak languages at x2.5, all others 1.0: tir dyu amh kbp fon nus mos kam bam umb kea kmb vmw kab seh yor wol ewe luo tum nyn ach kon pcm shi. Adaptive updater off.
* Final-run mix: 70/30 science/general; language floor 2 prompts per 20 steps; science bin weights b64 .11, b256 .22, b512 .32, b1024 .35 (b1024 keeps its old absolute share; the b2048 mass moved to shorter bins in old 1:2:3 proportions; plain renormalisation would give b1024 .54); general uniform .25 per bin.
* LibreTexts included (`source_dataset` `common-pile/libretexts`), English prompts only.
* Eval: `eval_every` 25 (128 prompts; 192 final), dev chrF++ every 50 steps, preflight at 25 and 49; weak_dev in all stages; final adds `afrisci_dev_doc` every 100, `general_dev`, AfriScience dev sentences as selection set. `keep_single_checkpoint` false so `final/` exists.

## 2. Evidence for each cap
* Stage A 400 (ESTIMATED from VERIFIED parts): dev macro saturated about 300k completions after SFT; 2b-b64 was flat from step 100 (42.01 -> 41.94). 293 steps at 1,024 completions/step; the new weight list has 114.5 total draw weight vs 91, so 369 steps for the same per-language dose, +8% margin = 400. At 400 steps a weak language gets 563 prompts, a weight-1 language 221 (old saturation dose 515 / 206). AB2 peaked at step 20 (320 prompts/lang) and fell 1.3 chrF++ by step 49 (784/lang), so more b64 is not free. M1's 640 would give 901 / 354.
* B1 b256 150 (VERIFIED): slope after step 150 +0.005/100 steps, CI [-0.011, 0.023]; after step 100 the CI [0.0007, 0.0241] still excludes 0; stage delta +0.25 chrF++.
* B2 b512 80: no trend over 446 steps (-0.0006 +- 0.0035/100), finished rate 1.0 from step 1 (VERIFIED); 80 is a choice (short exposure because long-document benefit by stage was never measured).
* B3 b1024 60: finished rate 0.9976 / 0.999 / 0.9995 over the first three 50-step blocks; no reward trend; blend-group stage delta -0.11 [-0.22, -0.01] (REPORTED).
* Mixed 60: block means -0.194, -0.170, -0.163; most gain in the first 50 steps; plateau not found within 149 steps, so 60 is the cheaper choice.
* Final 150: paper-level finished rate reached 1.0 within 50 steps, sentence dev flat from step 50 (sd 0.10); old run's lr was cut for 250 of 401 steps (confound).
* Prompts per step 64/48/42/42/42 are the 2c values (maximum, not tuned); no batch-size effect is identifiable. Group 16 and x2.5 weights: dose-response +1.38 chrF++ per doubling [-0.28, +3.11], weight class = sample count.
* Not supported by the record: the effect of the greedy member in stages A and B; real step times with the new reward. The preflight measures both.

## 3. Repetition of prompts (real `PromptSchedule`, seed 1234)
v3 has at least 4,773 prompts per language per bin, so pools never bind in the general stages.

| stage | draws per weak lang | draws per other lang | repeated-draw fraction weak / other |
|---|---|---|---|
| A 400 | 563 | 221 | 4.4% / 1.7% |
| A at 640 | 902 | 354 | 7.0% / 2.7% |
| B1 | 156 | 64 | 1.5% / 0.5% |
| B2 | 71 | 30 | 0.5% / 0.2% |
| B3 | 53 | 23 | 0.5% / 0.3% |
| mixed | 55 | 22 | 0% / 0% |

Final run on clean_sci_v2 (150 steps: 4,414 science + 1,886 general prompts; a language gets at least 45): science b64 10.3 draws vs pool 81 (5.8%/2.8%); b256 19.0 vs 120 (8.6%/5.2%); b512 28.9 vs 149 (8.2%/4.1%); b1024 31.1 vs 218 (7.7%/2.8%). Worst cell kon science b512: 42 draws from 150 rows (28%). Extending the final run to 250 steps raises the worst cell to 41%: do not extend past 250. About 1,037 of the 6,300 final-run prompts (16%) come from LibreTexts.

## 4. GPU-hours and USD ($1.19/h)
Assumptions: M1 step-time triples (quiet / median / median x1.4 contended); reward-overhead uplift x1.0/1.10/1.15 for stages A to mixed (ESTIMATED); eval/checkpoint overhead +5/8/12%; stage start 0.08/0.15/0.30 h; margin 5/10/20%. b64 17/19.7 s (VERIFIED, AB2 arm F); the no-b2048 mixed stage 100/105/166 s (ESTIMATED); b512 central 95.9 s. Low case: A 150, B1 100, B2 50, B3 40, mixed 40, final 100. Central: A 300, rest at caps.

| stage | steps low/central/max | GPU-h low/central/max | USD low/central/max |
|---|---|---|---|
| preflight | 49 | 0.32 / 0.47 / 0.78 | 0.38 / 0.56 / 0.93 |
| A | 150 / 300 / 400 | 0.82 / 2.10 / 4.24 | 0.98 / 2.50 / 5.04 |
| B1 | 100 / 150 / 150 | 0.93 / 1.63 / 2.54 | 1.10 / 1.93 / 3.02 |
| B2 | 50 / 80 / 80 | 1.03 / 2.68 / 4.14 | 1.22 / 3.19 / 4.92 |
| B3 | 40 / 60 / 60 | 1.60 / 2.81 / 4.33 | 1.90 / 3.34 / 5.15 |
| mixed | 40 / 60 / 60 | 1.25 / 2.23 / 3.87 | 1.48 / 2.65 / 4.60 |
| final | 100 / 150 / 150 | 3.87 / 7.35 / 10.57 | 4.61 / 8.75 / 12.57 |
| sum | | 9.82 / 19.27 / 30.47 | |
| with margin | | 10.3 / 21.2 / 36.6 | 12.3 / 25.2 / 43.5 |

Versus M1 (stages 2a-2e plus 4, before margin): low 10.99 vs 9.50 h; central 20.32 vs 18.80 h; max 47.23 vs 29.69 h (-37%). M1's production-track total becomes about 43.4 h (USD 52) central and 72.5 h (USD 86) max (arithmetic, SFT stages not recomputed). Specialist arm (S1 or S2), dose-matched: A 175, B1 65, B2 35, B3 25, mixed 25, final 65 steps (329,920 completions), roughly 0.45x the RL hours; not separately costed.

## 5. Preflight and stage gates (`clean_rerun_tools.py gate`)
Preflight purpose: check KL, grad norm, greedy loop rate and step time with the greedy member on (AB2 arm F took about 16 min). A fail stops the launcher.
* Preflight hard fails: any KL guard cut; median KL over the second half > 0.08 or max KL > 0.15; median grad norm > 3.0; greedy loop rate > 0.02; greedy finished rate < 0.97 or dev greedy finished rate < 0.95; dev chrF++ drops 0.5 or more.
* Reference values: AB2 arm F KL 0.12-0.14, grad norm 1.9 (4 languages); 2b-b64 KL about 0.045; D-67 greedy-stats recipe grad norm 0.60.
* Later stages fail on: 4 or more guard cuts, median grad norm > 3.0, greedy loop rate > 0.03, dev drop of 1.0 or more. The gate also flags the AB2 decline pattern, greedy-advantage share > 0.35, and reports GPU-h per 100 steps.

## 6. Open decisions
1. Stage A cap: 400 (USD 1.6 cheaper) or M1's 640. Weight-1 languages get 221 prompts vs the AB2 peak of 320; `STEPS_A=560` reaches 310 for +0.9 h / USD 1.1.
2. Hand-off: `HANDOFF=final` default (as 2c); `HANDOFF=best` available if dev chrF++ falls 0.3+ after its peak.
3. Greedy member in stages A and B follows the owner's instruction; the only divergence on record (2b) also had lr 3e-6, no std division and no guard. The preflight is the test; `greedy_in_group=false` is a one-line override.
4. Fresh optimizer in the final run (the old one carried 2c Adam moments, which made lr 1e-6 act about 3x faster). Say if moments should carry over from the mixed stage.
5. Weak-list quirks: kon is in via a GlotLID label-set artefact; 27 unmeasured languages get weight 1.0 with only the floor protecting them.
6. No language floor in stages A to mixed (the floor needs a `domain` column; v3 has none); P(fewer than 2 prompts per 20 steps) < 0.1% for a weight-1 language.
7. `HUB_PREFIX` is unset, so nothing is pushed by default (a box wipe would lose stage models).
8. Files other agents must produce (check fails until they exist): `assets/calibration`: shaping_clean_v1.json, anchors_clean.json, anchors_clean_bigrams.npz, longdoc_clean.json, science_anchor_clean.json, nllb_reward_calibration_clean.json; `assets/lexicon`; `assets/dev`: bench_dev_hsel.jsonl, bench_weak_hsel.jsonl, bench_afrisci_dev_sent.jsonl, bench_afrisci_dev_doc.jsonl; `assets/pseudorefs_reward_only_nc_v1`; venvs and pylibs at the old /workspace paths.
9. LibreTexts into the science SFT: the best-translation archive has no `source_dataset` column; use `launch_clean_rerun.sh disclose final` before the science SFT.
10. Specialist arm: AfriScience languages are amh hau lug nso yor zul, only amh and yor are in the specialist set; the specialist configs are generated but not otherwise checked.
11. Possible saving: b1024 steps cost about 134 s because some looping completions decode to the 3,584-token cap; a lower cap would turn those loops into failures (untested).
12. Trainer was still moving; re-run `validate` once it is final.

## 7. How it was checked
`clean_rerun_tools.py validate --stub-heavy` passes for both config folders and fails correctly on an unknown key, a wrong `greedy_adv_scale`, and a hand-edited yaml. The launcher dry run with a fake trainer ran preflight, A, B1, B2, B3, mixed, final in order with gates and hand-off paths; the WINDDOWN watcher was tested separately; bash 3.2 compatible. `tools manifest` matched all pins: rule file 7b86fc11…, Amendment 1 a5885cf2…, Amendment 2 955941c5…, weak-list rule 4472d711…, index .3 manifest 22172f30…, git commit recorded with 135 dirty entries. Not tested: the real trainer, `check --check-files` on a rental, the GPU-type pin, `--hash-pools`, `disclose` on real parquet, `A_FROM_PREFLIGHT`.
