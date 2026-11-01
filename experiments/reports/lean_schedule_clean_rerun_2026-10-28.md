# Lean schedule for the clean rerun: what the earlier experiments say we can cut (agent M1, 2026-10-28)

Read-only analysis of the existing record. No box was touched, nothing was trained or committed. Data and small CSVs: `reports/data/lean_schedule_2026-10-28/` (file list in section G). Pre-registration in force: `reports/preregistration_<S3_PREFIX>.md` (rule file sha256 `7b86fc11...c0f5`, Amendment 1).

Labels used on every number: **VERIFIED** = read or computed by me from a file that I name; **REPORTED** = stated in a document (paper, DECISIONS, report), not recomputed; **ESTIMATED** = my arithmetic or an assumption, with the basis stated.

## 0. Headline

1. **The record supports a much shorter RL.** Of the old RL lineage (about 1.76 M completions), the sentence-level dev gain and the whole gain of the weak-language groups were in place after phase 1, phase 2 and the b64 stage of phase 2b (0.88 M completions after phase 1). The 0.90 M completions of phase 2c (b256, b512, b1024, b2048, mixed; 40.5 h, about USD 48) moved the 19-language dev macro from 41.89 to 41.88 (range 41.57 to 41.98, sd 0.094, 31 evaluations, VERIFIED) and the weak-group means by +0.3 at most (stage-end checkpoints, REPORTED/VERIFIED below). The final run (401 steps, 28.4 h, about USD 34) moved the sentence dev by +0.13 and FLORES+ by +0.15 chrF++ over its start model A2W (REPORTED).
2. **Per-language sampling rate**: no dose-response is identifiable in the record (R13/D-65: +1.38 chrF++ per doubling, 95% CI [-0.28, +3.11]; the weight class is the sample count, corr 0.988). The data say "do not buy more"; they do not say "the x2.5 weighting is useless". Keep x2.5 on the dev-defined weak list, and spend fewer total samples.
3. **The learning-rate guard ("KL tracker") is the only adaptive-lr mechanism** (`train_rl.py` lines 878-887). Keep it, with the k3c clamp and the median window (section C). One real defect: its state is not saved in the checkpoint, and in the final run its threshold (0.15) sat inside the normal KL band, so 250 of 401 steps ran at 25-75% of the planned learning rate. The "no gain" of the final run is therefore confounded (section B.6).
4. **Recommended lean schedule (central)**: SFT A0 2.4 B tokens (not 3.0 B + stage A); RL stage A b64 640 steps; stage B b256 150, b512 120, b1024 60, mixed 60 (b2048 dropped, as pre-registered); science stage as before (318 steps) with its inputs regenerated from the new RL model; final run 150 steps. **Production track central 44.5 GPU-h = USD 53 (low 26.6 h / USD 32, high 92.6 h / USD 110) at USD 1.19/h, including a 10% margin; plus USD 11 central for the pre-registered S1 and S2 specialist arms** (total central 54.1 h = USD 64; low USD 38, high USD 128). The old pipeline cost about 105-115 h (USD 125-137, ESTIMATED because phases 1, 2, 2b and SFT hours were never recorded).
5. **Where the record cannot support a cut** (do not over-trust): the long bins (b512, b1024) cannot be shown useless for long-document behaviour, because no intermediate checkpoint was ever evaluated on AfriLongDoc (paper 6.4 says so). I keep them but short (120 and 60 steps) and offer a USD 3 check (section E.4).

## A. The old pipeline as it actually ran

Full machine-readable ledger: `old_pipeline_ledger.csv`. Hardware: one rented RTX PRO 6000 96 GB at about USD 1.19/h for RL and late SFT (rate stated by the project lead, no invoice in the record, paper Appendix F TODO), an RTX 5070 12 GB (own card) for the first part of SFT.

### A.1 SFT

| item | value | label |
|---|---|---|
| Stage A | AdamW 5e-5, warm-up 3%, cosine to 5e-6, 131,072 tokens/step, sqrt-sampling (floor 0.2%, retire after 4 passes); stopped at step 4,750 of 16,827 planned; RTX 5070 at 22 s/step (about 29 h) | config + paper 3.5 VERIFIED; hours ESTIMATED |
| Stage B | restart from stage A weights, Muon 4e-5 (match_rms_adamw) on the decoder matrices, AdamW on embeddings and norms, warm-up ratio 0.03, cosine, min_lr_ratio 0.1, wd 0.01, grad clip 1; UniMax budget 3.0 B tokens = 22,888 steps, every language 1-2 passes, the 20 translation-target languages weighted x4 (`gemma_mt/configs/sft_unimax.yaml`) | VERIFIED |
| SFT-final | **step 27,638 = 4,750 + 22,888 = the last step of the planned UniMax budget**, macro validation loss 1.2654 (1.2706 at step 25,750, still falling slowly). It was not chosen by a quality metric: checkpoints were selected by macro validation loss, early stopping was in "propose" mode, and the run simply completed. 3.62 B tokens in total | VERIFIED arithmetic; paper 3.5 REPORTED |
| Throughput | 5070: 21.4 s/step (5,822 tok/s) on 2026-10-17 at step 13,910; 13,728 steps remained at that date, i.e. up to 11.1 h on the PRO 6000 at the measured 2.9 s/step (section G.1). Which steps ran on which card is not recorded | REPORTED + ESTIMATED |
| Tail of SFT | steps 20,500 to 27,638 (last 26%, 0.94 B tokens): +0.44 chrF++ on all rows and **+0.05 on clean rows** (45 FLORES+ languages, 320-token budget); the first 2.69 B tokens gave +16.58 / +14.05 | REPORTED (paper `t_rq`) |

### A.2 Science SFT stage (A2W, the stage before the final run)

Start: RL-final (step 1,433). Mixture A2W: 20.88 M tokens, **318 steps x 65,536 tokens** (2 x 4 x 8,192), AdamW 3e-6 peak, warm-up 6%, cosine to 3e-7, wd 0, gradient checkpointing. Ratios (`reports/sft_mixture_v2_2026-10-22.md` table, REPORTED): science (AfriScience-MT train) 31.7%, human replay 43.0% (weak-language replay x0.5), RL best-translation archive 5.0%, long-document distillation 13.5%, self-distilled anchors 6.7%. Weight interpolation theta = (1-a) theta_RL + a theta_SFT-stage, sweep a in {0.3, 0.5, 0.7, 1.0}, **a = 0.7** chosen by guardrails (best AfriScience dev sentence chrF++ with general dev, weak-language groups and paper-level finished rate not worse). Wall-clock 46.8 min on a shared card (VERIFIED, `r7_runs/sft_A2W/metrics.jsonl`); the same recipe with 387 steps took 20.7 min on a quiet card (C3, VERIFIED); the 4-alpha sweep took 33 min (VERIFIED, log timestamps 07:25-07:58). Result: A2W a=0.7 sentence 47.97, paper-level chrF 47.81, finished 92.0%, weak-blend -0.36 [-0.58, -0.16] (REPORTED paper 6.6). Five variants were trained (A 330, A2W 318, B_v1 362, B_v2_ibom 514, C3 387 steps; 243 min in total, VERIFIED).

### A.3 RL phases (per phase and bin)

| phase | bins: steps | prompts/step | group | lr | KL | other | wall-clock |
|---|---|---|---|---|---|---|---|
| 1 | b64 40, b256 60, b512 60, b1024 60, mixed 80 (300) | 64/48/42/42/42 | 8 | 1e-6, warm-up 10 | 0.05, k3, no guard | std-normalised advantages, language alpha 0.5; 110,720 completions | not recorded; ESTIMATED 3.6 h |
| 2 | b64 98, b256 130, b512 75, b1024 45, mixed 149 (497 planned, log ends at 378) | 128/96/84/84/84 | 8 | 2e-6 | 0.05, k3, no guard | alpha 0, x2.5 on 17 weakest; 380,960 completions planned | not recorded; ESTIMATED 10 h |
| 2b | b64 488 (kept), b256 351 (diverged, stopped at 839 of 1,623) | 64/48 | 16 (15 + 1 greedy) | 3e-6 | 0.05, k3, no guard | advantages without std division | not recorded; ESTIMATED 2.7 h kept + 2.9 h discarded |
| 2c | b256 391, b512 446, b1024 149, b2048 298, mixed 149 (1,433) | 48/42/42/21/42 | 16, all sampled | 2e-6, warm-up 10 | 0.05; k3 to step 900, k3c from the restart at 900; guard 0.15, window 5, factor 0.5, floor 1/8, cooldown 10 | std-normalised; 900,384 completions | **40.48 h wall (2026-10-20 01:57 to 2026-10-21 18:26 UTC), VERIFIED** |
| science RL (D-52 `rl_science.yaml`, D-59 `rl_science_final.yaml`) | never ran as a production run: the v1 launcher was stopped after 11 min (`launch_FINAL.log`) | | | | | superseded by final2 | VERIFIED |
| final2 | mixed bins from step 0, 70% science / 30% general; 401 of 700 planned (wind-down 351-401) | 42 | 16 (15 + greedy-stats, scale 0.3, cap 3) | 1e-6, warm-up 20; effective 2.5e-7 to 1e-6 | 0.1, k3c, reference = start model A2W, guard 0.15 (median 5) | QE ensemble, anchors v3, longdoc v2, NLLB signal on kea kbp nus bam kab, floor 2 prompts/20 steps | **28.44 h first-to-last record, 26.24 h step time (197 s median in steps 1-150, 205 s in 151-250, 283 s in 251-401, shared card; 218 s median overall), VERIFIED** |

Per-bin step time and throughput in 2c (VERIFIED, computed from `paper/data/rl2c_metrics.jsonl`, unique steps; 49 duplicate steps from restarts excluded): b256 29.8 s/step (23.3 k generated tok/s), b512 95.9 s (11.2 k; 64-66 s on quiet blocks, 106 s on contended blocks), b1024 134.1 s (21.1 k), b2048 142.3 s (16.7 k), mixed 118.7 s (16.6 k). AB2 b64 (64 prompts x 16): 19.7 s/step (VERIFIED). Samples per language, exact replay of the schedule (REPORTED, D-65): a weight-1.0 language received 977 prompts (10.3 k completions) up to the end of the b64 stage of 2b and 1,526 prompts (19.0 k completions) in the whole lineage; a x2.5 language 2,123 (23.2 k) and 3,463 (44.6 k) (`per_language_dose_old_vs_lean.csv`).

Old total on the rented card (ESTIMATED where marked): SFT remainder <= 11 h + phases 1-2b about 19 h + 2c 40.5 h + science SFT 4.1 h + sweeps about 2.5 h + final2 28.4 h = **about 105 h, USD 125** (about 113 h if all SFT stage B had run on the PRO 6000), excluding the AB2 chain (9.7 h, USD 11.5 if all billed) and benchmark evaluations. Only the 2c and final2 figures (68.9 h, USD 82) are VERIFIED.

## B. What the evidence says about diminishing returns

### B.1 Dev curve by phase (19 languages, 760 sentences, chrF++)

| point | dev chrF++ | change | completions since previous point | label |
|---|---:|---:|---:|---|
| SFT-final | 39.65 | | | `rl1_evals.csv` VERIFIED |
| phase 1, step 50 / step 300 | 41.00 / 40.96 | +1.3 | about 24 k at step 50, 110.7 k at step 300 | VERIFIED |
| phase 2 start -> step 100 | 40.96 -> 41.77 (then 41.3-41.6 at 150-350, noisy) | +0.45 net | 100 steps = about 100 k | VERIFIED |
| phase 2b start (41.41) -> step 100 / 500 | 42.01 / 41.94 | +0.5, flat from step 100 | 100 steps = 102 k; 488 steps = 500 k | VERIFIED (`rl2b_evals.csv`) |
| phase 2c, 31 evaluations, steps 0-1433 | 41.89 -> 41.88; min 41.57, max 41.98, sd 0.094 | -0.01 | 900 k | VERIFIED (computed) |

So the dev macro saturates about 300 k completions after SFT (ESTIMATED sum of phase 1 + 100 steps of phase 2 + 100 steps of 2b; the three phases overlap in recipe) and nothing moves after that. The paper's per-million figure (REPORTED): +2.6 chrF++ per million completions for the first 0.86 M, -0.01 for the next 0.90 M.

### B.2 Weak-language groups at the stage-end checkpoints (REPORTED from `reports/rl_vs_sft_weak_signal_2026-10-21/results.json`; read into `stage_end_group_chrf_deltas.csv`)

Mean chrF++ over the languages of each group (all rows, 320-token budget); deltas with 95% bootstrap intervals over languages; "step" delta is against the previous checkpoint.

| checkpoint | consensus-only (13) delta vs SFT | step delta | blend (19) delta vs SFT | step delta | control (10) delta vs SFT | step delta |
|---|---|---|---|---|---|---|
| RL phase 1 | +1.64 [1.08, 2.26] | | +1.80 [1.39, 2.25] | | +1.76 [0.96, 2.67] | |
| 2b step 488 (end of b64) | +4.80 [3.56, 5.89] | **+3.16 [2.16, 4.16]** | +5.50 [4.51, 6.45] | **+3.70 [2.79, 4.61]** | +2.48 [1.26, 3.97] | +0.72 [0.15, 1.38] |
| 2c end of b256 | +5.06 | +0.26 [-0.04, 0.54] | +5.75 | +0.25 [0.04, 0.45] | +2.29 | -0.19 [-0.40, 0.00] |
| 2c end of b512 | +5.00 | -0.05 [-0.31, 0.17] | +5.73 | -0.02 [-0.16, 0.12] | +2.34 | +0.05 [-0.12, 0.20] |
| 2c end of b1024 | +4.93 | -0.07 [-0.22, 0.10] | +5.62 | -0.11 [-0.22, -0.01] | +2.34 | 0.00 [-0.15, 0.14] |
| 2c end of b2048 | +4.87 | -0.07 [-0.35, 0.14] | +5.65 | +0.03 [-0.11, 0.18] | +2.40 | +0.05 [-0.07, 0.18] |
| 2c final (mixed, step 1,433) | +5.07 | +0.20 [0.01, 0.44] | +5.85 | +0.21 [0.07, 0.36] | +2.39 | 0.00 [-0.17, 0.18] |

Reading: everything the weak groups gained was in place at the end of the b64 stage of 2b (0.88 M completions after phase 1); b256 and the final mixed stage each add about +0.2 (intervals just touching zero), b512, b1024 and b2048 add nothing (b1024 blend is -0.11 [-0.22, -0.01], marginally negative). **The same checkpoints show the finished rate of the 23 weak-signal languages rising from 82.6% (phase 1) to 99.1% (2b-b64)** and "degenerate" falling from 22.9% to 7.3%: much of the +3.2 to +3.7 is the model learning to stop (the AB2 ablation reaches the same conclusion, B.5). **Limit:** these sets are sentence-level (FLORES+, 320 tokens); they cannot show skills that need long windows, so they cannot say b512/b1024/b2048 were useless for long documents (B.4).

### B.3 Inside phase 2c, per bin (VERIFIED, computed from `rl2c_metrics.jsonl`; `rl2c_bin_trend.csv`, `rl2c_bin_plateau_test.csv`, `rl2c_bin_blocks_50steps.csv`)

Training reward of 2c (range -2 to +2.25) is the only fine-grained per-bin series. Slope = OLS of 20-step block means, bootstrap SE over blocks; "plateau" = first start step after which the 95% bootstrap interval of the slope contains 0.

| bin | steps | training reward first 50 -> last 50 | slope per 100 steps (all steps) | plateau after step (in bin) | eval reward (144 prompts, held out) | finished rate |
|---|---:|---|---|---|---|---|
| b256 | 391 | -0.010 -> +0.081 | +0.022 +- 0.007 | **150** (slope after 150: +0.005, CI [-0.011, 0.023]) | 0.536 -> 0.538 | 1.000 throughout |
| b512 | 446 | +0.026 -> +0.028 | -0.001 +- 0.004 | **0** (no trend at all) | 0.327 -> 0.314 | 1.000 |
| b1024 | 149 | -0.248 -> -0.233 | +0.015 +- 0.025 | **0** | -0.229 -> -0.147 (noise sd 0.04) | 0.998 -> 0.9995 |
| b2048 | 298 | -0.845 -> -0.753 | +0.036 +- 0.011 | **100** (slope after 100: +0.002, CI [-0.021, 0.033]) | -1.71 to -2.00 at all 22 evaluations | 0.991 -> 0.9996 (mostly in the first 100 steps) |
| mixed | 149 | -0.194 -> -0.165 | +0.055 +- 0.013 | not reached | n/a | 1.000 |

Cautions: the reward is the learned/QE reward, not chrF++ (the b256 reward rose +0.09 while the dev chrF++ did not move); steps 748-900 (last 90 of b512 and first 63 of b1024) ran at half the learning rate after the spurious guard cut (section C); the b2048 trend is confounded by the word-ratio reward change at step 1250 and the k3c restart at 900. The eval reward by bin is 144 prompts per evaluation (noise sd 0.03-0.08). Finishing at each length is learned in the first 50-100 steps of the bin.

### B.4 What the long bins did for long documents (the weak spot)

REPORTED (paper 6.4): RL-final extended whole-document translation from about 256 to about 512 source tokens (b512 chrF++ against NLLB 32.5 -> 47.7), but b1024/b2048/b4096 still fail (finished 65 / 42 / 31%, loops 53 / 73 / 73%). **No intermediate checkpoint was evaluated on AfriLongDoc**, so the gain from 256 to 512 cannot be attributed to b256, b512 or any other stage. The T = 1 rollouts of b1024 and b2048 have almost flat groups (within-group sd 0.18 and 0.11), so GRPO had little to learn there. In the final run, 70% science prompts skewed to b1024/b2048: the paper-level finished rate went 0.927 -> 1.000 within 50 steps and then drifted to 0.967, chrF++ +2.9 at step 50 not sustained (+0.35 at step 401, sd of the series 0.85); the b2048 eval reward stayed between -1.82 and -1.95 at all 17 evaluations (REPORTED paper 6.7). The paper-level gains of the science stage came from supervised long-document distillation, not from RL (REPORTED 6.6). **The record supports: finishing at a given length is learned within 50-100 steps; it does not support any claim that 150-446 steps per long bin were needed.**

### B.5 Matched-budget ablation AB2 (single run per arm, four languages yor kin orm lin)

REPORTED + VERIFIED (`box/jobs/ab2/runs/*/metrics.jsonl`, `paper/data/ab2_results/`): 49 RL steps of b64 (64 prompts x 16 = 50,176 completions, 784 prompts per language) from SFT-final gave FLORES+ `no_pair` gain_F = +5.72 [5.15, 6.30] chrF++, **79% of the +7.25 that the full 1.76 M-completion lineage gave on these languages from 2.9% of the completions** (paper 8.4). The dev curve of arm F (VERIFIED): 30.72 (step 0), 34.24 (10), **35.79 (20)**, 35.78 (30), 35.24 (40), 34.52 (49); arm W (weak SFT start): 19.76, 24.61, **25.32**, 25.29, 24.75, 24.55. Dev peaked at step 20 (20,480 completions, 320 prompts per language) and fell by 1.3 and 0.8 by step 49 with the KL at 0.129 (guard threshold 0.15). The endpoint (step 49, no selection) is what the gains above measure. From the weak start RL gained +7.36 [6.80, 7.85], against +0.32 [-0.18, 0.82] for 367 steps of continued SFT (equal GPU time) and -1.01 for 3,136 pair-matched SFT sentences; on sentences that every system finished the weak-start RL gain is +1.13 [0.71, 1.55] against +2.23 for the strong start, i.e. a large share of the weak-start gain is repair of unfinished outputs (finished rate 70.0% -> 88.7%). Consequences for the schedule: (i) the first about 5 k completions per language at b64 carry most of the benefit (consistent with B.1: about 300-400 k b64 completions over 77 languages); (ii) the dev decline after step 20-30 is a warning that more b64 at lr 2e-6 is not free; (iii) RL beats continued SFT at equal cost in this regime, so do not shift budget from RL to SFT.

### B.6 Final run (final2) and the learning-rate confound

Dev series (VERIFIED, `final2_metrics.jsonl`): AfriScience dev sentence chrF++ 48.63 (step 50), 48.72, 48.93, 48.85, 48.85, 48.82, 48.86, 48.91, 48.92 (step 401); general dev 42.08 -> 42.04; weak dev 25.16 -> 25.28; sd of the nine sentence evaluations 0.10. Start 48.79. All four stop criteria held at the evaluation of step 350 (REPORTED). Against the start A2W the final model is +0.15 [0.12, 0.17] chrF++ on FLORES+ (45 languages) and +0.17 [0.07, 0.27] on AfriScience test sentences; paper-level inside the noise (REPORTED 6.7). The five NLLB-signal languages on FLORES+: bam -0.16, kbp +0.57, kea -0.49, kab -0.10, nus +0.13 (REPORTED `t_final2_perlang`), mean about 0.

**Confound (VERIFIED, lr column and `launch_FINAL2.log`):** the guard cut the learning-rate scale to 0.5, 0.25, 0.125 at steps 121, 135, 145; the planned restart at step 150 reset it; the guard cut it to 0.5 again at 155; the operator wrote LR_SCALE 0.5 at 19:38 UTC on 10-22 and 0.75 at 03:13 on 10-23. Mean lr per 50 steps: 8.1e-7, 1.0e-6, 5.2e-7, 2.8e-7, 2.5e-7, 7.5e-7, 6.8e-7, 2.0e-7 (wind-down). **The final run spent steps 101-250 at 25-52% of its planned learning rate.** So the flatness after step 50-150 is both "no more to learn from this reward at this state" and "the lr was cut"; the record cannot separate them.

### B.7 Per-language sampling rate

Source: D-65 / `adaptive_language_weights_2026-10-22.md` (REPORTED, replay of the real schedule, regression over 64 languages with human-reference rows). Samples per language are the weight class (x2.5 class mean 1,340 prompts in 2c, weight-1.0 class 549; corr(log2 samples, class) 0.988; spread inside a class 6-13%), so dose-response is not identifiable. Gain SFT -> RL on `no_pair` rows ~ SFT level + reward mode + script + log2 samples: **+1.38 chrF++ per doubling [-0.28, +3.11]**; x2.5 indicator +2.00 [-0.26, +4.19]; low-baseline stratum (SFT < 22): 17 languages at x2.5 gained 8.8, 12 at x1.0 gained 7.9, difference +0.96 [-1.35, +3.46]; the SFT level dominates (-1.63 per +10 points [-2.66, -0.59]). Phase 2c itself is a plateau (no language moved by more than 1.4 on the dev). Saturation of afr swh zul xho sot som ibo: small gains are mostly the baseline effect; QE headroom is not chrF headroom. The adaptive updater ran in dry run only and never changed a weight. `low_resource_rl_data_ladder_proposal_2026-10-24.md` is a proposal with no results (its only numbers are the AB2 ones above). **Where evidence is weak:** weights were chosen by weakness (selection basis of the 17 is UNKNOWN or devtest-informed, audit 2.3), one run, confounded with reward mode.

### B.8 Where the evidence is weak (summary)

Single seed everywhere (AB2 and every phase; no run-to-run variance anywhere); dev monitors cannot see per-language changes below about 3 chrF++ (40/60/200 sentences: minimum detectable 6.5 / 5.3 / 2.9, D-65) but the 19-language macro has sd 0.09-0.10; stage-end evaluations exist only for 2b-b64 and the five 2c stage branches, not for phases 1 and 2 separately; no long-document evaluation by stage; hours of phases 1, 2, 2b and of SFT were never recorded; the final-run lr confound; the SFT dose-response between 0.46 B tokens (AB2 W0: 15.2 chrF++ on four languages) and 2.69 B is known only through validation loss; evaluation sets of the old run were partly contaminated (audit), so levels are inflated (about 3.6 chrF++ for SFT-final, 2.4 for RL-final on the headline subset) while differences between checkpoints are less affected because they share the exposure.

## C. The KL tracker / adaptive learning rate: what it is, what it did, what to keep

I searched for `kl_tracker`, `kl_target`, `adaptive_kl` and similar: nothing exists. The mechanism the owner remembers is the **KL guard**, plus two operator controls. There is no controller that targets a KL value.

**Mechanism (VERIFIED, `gemma_mt/gemma_mt/rl/train_rl.py`):** config fields lines 119-125 (`kl_guard_threshold` 0 = off, `kl_guard_factor` 0.5, `kl_guard_min_scale` 0.125, `kl_guard_window` 5, `kl_guard_cooldown` 10); state `lr_scale, last_cut = 1.0, -10**9` line 712; the learning rate of step s is `learning_rate * lr_scale * min(1, (s+1)/warmup) * ctl.lr_mult` (line 731); after each step the token-weighted mean KL of the step (`metrics_acc["kl"]/tok`, line 863) is appended to a window of the last 5 steps, and if the **median** of the window (line 880: sorted window, middle element) exceeds the threshold, the window is full, at least `cooldown` steps have passed since the last cut and `lr_scale` is above the floor (condition, lines 881-882), then `lr_scale = max(lr_scale * 0.5, 0.125)` and `last_cut = step` (lines 883-884, message 885-886); `kl_guard_lr_scale` is logged every step (line 887). Nothing is stopped or rolled back. The KL estimator is `k3c` (`grpo.py` lines 116-136): k3 = exp(d) - d - 1 with d = ref_logp - logp, clamped at d <= 5 so that one token whose probability the policy pushed far below the reference cannot produce an unbounded term; identical to k3 below the clamp and no KL gradient beyond it. `run_controls.py` (lines 1-12, `LR_SCALE_MAX` line 21, class line 46): a file `LR_SCALE` (float, clamp [0, 4]) multiplies the learning rate, a file `WINDDOWN` (N) decays it linearly to 0 over N steps and ends the run normally; both are read every step. The trainer also checks every 25 steps that vLLM's log-probabilities of its own samples match the trainer's (`vllm_check_every` 25, `vllm_logp_gap_tol` 0.25, line 88-89) and stops the run if not. Parameters in the executed configs: `rl_rtxpro_phase2c.yaml` lines 82-86 and `rl_science_final2.yaml` lines 110-114 (threshold 0.15; the yaml comments still say "mean", the code says median).

**State is not persisted (VERIFIED: `lr_scale` occurs only at the lines above, never in a checkpoint; D-45 addendum says so):** every restart re-arms full learning rate. Observed in the final run at steps 151 and 251.

| incident | what happened | what the guard did / would have done | label |
|---|---|---|---|
| Phase 2b divergence (lr 3e-6, no std division, greedy member in group) | b256 stage began at step 489; KL 0.045 -> 0.18 at step 530 -> 0.27 (541) -> 0.50 (560) -> 1.81 (661) -> 2.35 (700); finished rate 0.95 -> 0.56 in the eval prompts; dev chrF++ fell to 41.4 only at step 750; stopped at 839, model of step 488 kept | The guard did not exist yet. Retrospectively a 5-step median above 0.15 first occurs at **step 536**, 303 steps before the stop (computed). Whether it would have prevented the divergence was never tested; phase 2c changed three things at once (std-normalised advantages, lr 2e-6, guard) and dropped the greedy member | VERIFIED series; counterfactual untested |
| 2c outliers, steps 607 and 747 | KL 0.51 / **1,651**, grad norm 220 / **285,381**; every other step 0.02-0.05; cause: unbounded k3 on one token with log-ratio about 16 | gradient clip 1.0 effectively discarded those steps. The mean-window guard fired at 747 although nothing diverged: lr 2e-6 -> 1e-6 for steps 748-900 (153 steps; dev 41.64 at 750 -> 41.88 at 800) | VERIFIED |
| Fix at the step-900 restart | k3c clamp + median window; guard state reset | Steps 901-1433 (533 steps): max KL 0.030, max grad norm 1.87, no cut | VERIFIED (computed) |
| Final run | threshold 0.15 against the start model with kl_coef 0.1: median KL 0.096 (p10 0.049, p90 0.152, max 0.267); 45 of 401 steps above 0.15 | cuts at steps 121, 135, 145, 155, 341; reset at 151, 251; see B.6 | VERIFIED |

**Keep (all of it):** k3c with clamp 5; guard on the median of 5 with factor 0.5, floor 1/8, cooldown 10; grad clip 1.0; std-normalised advantages; no greedy member in stages A and B (final run: greedy statistics over the sampled members only, scale 0.3, cap 3); LR_SCALE and WINDDOWN files; the vLLM log-prob check; restarts at the curriculum switch from b64 to b256 with a fresh reference, exactly as 2c did. Threshold 0.15 is 3-7x the healthy KL of stages A and B (0.02-0.06 in 2b-b64 and 2c) and was reached 41 steps after the 2b divergence began; keep it there.

**Change (small, cheap; owner decision, section F):** (1) save `lr_scale` and `last_cut` in `trainer_state.pt` and restore them on resume, so that a restart cannot silently undo a cut; log it. (2) For the final run only, the reference is the start model and kl_coef is 0.1, which puts the healthy KL (0.10-0.14) next to 0.15. Either raise the final-run threshold (UNTESTED suggestion: 0.25; in 2b KL went from 0.18 to 0.27 in 11 steps, so 0.25 would have fired about 10 steps later) or keep 0.15 and treat a cut as information, not as a normal state; whatever is chosen, record the effective lr per step as a primary result next to the gains. (3) Write the four stop criteria of the final run into the launch manifest (they exist only in a message).

## D. Reuse table

One-line view; full reasons and evidence in `reuse_table.csv` (26 rows). Rule: AS IS = no test text and no output of a contaminated model; REFIT = same procedure, different (clean) input data; REBUILD = fitted on test text or built from outputs of the contaminated models, or the selection basis was test-informed (audit sections 2.2, 2.3, 5).

| verdict | assets |
|---|---|
| REUSABLE AS IS | GRPO trainer and all reward code (k3c, guard, run controls, chunk plan, penalties, consensus, ensemble combiner, anchors code, longdoc v2 code, language floor, LANG_WEIGHTS control); evaluation scripts and the contamination scanner; trainer configs (paths, budgets and b2048 change only); science-stage recipe and ratios (A2W) with its builder scripts, once the hold-out is extended to every protected set (audit 2.5); AfriCOMET-QE, SSA-COMET-MTL, GlotLID, NLLB-200 3.3B (final run only, NC, disclosed); clean SFT corpus and specialist set of D2; AfriLongDoc and its pseudo-references as evaluation material |
| REUSABLE AFTER REFIT ON CLEAN DATA | calibration scripts (point at H_cal, SSA-MTE train, IbomMT train; add the split-label assertion of audit 5.3(7)); GlotLID thresholds (`lid_measure`); lexicons and function-word tables (rebuild from clean SFT pairs, validate on H_cal); RL prompt pools v3 / sci_v2 (re-filter English sides against the protected index; 7 of 1,012 FLORES+ devtest sentences were in them); NLLB pseudo-references of the RL prompts (drop prompts that match; NC licence); AfriScience dev monitors (an S set) |
| MUST BE REBUILT | shaping calibration v7/v8 (FLORES+ dev + MAFAND dev; the Efik entry was fitted on 997 IbomMT **test** sentences; new H_cal has about 278 sentences per language against 997); constraint anchors v1-v3 and science anchor (fitted on the test splits of six benchmarks); longdoc_v2.json (SmolDoc and AfriLongDoc evaluation documents); NLLB language rule and its calibration (decided on FLORES+ devtest); the "17 weakest x2.5" draw list and the final-run weights (basis UNKNOWN or devtest-informed; use the dev-defined 25-language list); the 760-sentence general dev set and the 60-sentence weak dev (rebuild on H_sel); RL best-translation archive, long-document distillation and self-distilled anchors of the science mixture (outputs of RL-final, which was trained on contaminated SFT text) |

Implication for cost: all REBUILD items except the generation of the science-stage inputs are CPU work or minutes of GPU scoring (stage 0 in E: 0.5 h central). The science-stage inputs need the new RL model first, which fixes the order SFT -> RL -> inputs -> science SFT -> final run, as pre-registered.

## E. Recommended lean schedule

### E.1 Throughput assumptions (derived from logs; rate USD 1.19/h)

* **SFT**: 2.90 s per 131,072-token step on the RTX PRO 6000 (VERIFIED: AB2 weak SFT, 3,499 steps in 2.82 h including seven validation passes; `box/jobs/ab2/runs/sft_weak/metrics.jsonl`; AdamW, varlen attention, fused loss, no gradient checkpointing, micro-batch 16 x 2); 2.65 s/step in the S1 control (VERIFIED). The old plan's 6.4 s/step on this card (`gemma_rl_cost_estimate` section 8, repeated as "49.1 h" in `ge_ez_step0`) was an assumption (speed ratio 3.0 to the 5070) and is 2.2x too pessimistic. Muon on this card is unmeasured; the high case adds 10%.
* **RL**: per-bin step times of 2c on the same card (VERIFIED, section A.3), b64 from AB2. Contention on the shared card made b512 steps 64-113 s; low uses the quiet value, high adds 40% (the final run lost 44% when shared). No extrapolation from another card is used for RL. The final-run step time under "b2048 dropped" has no measurement: 130 / 160 / 220 s is ESTIMATED (final2 was 197 s with b2048 and 70% science).
* Overheads (evaluation cadence 25/50 steps, restarts, vLLM start-up): +5 / 8 / 12% on RL; margin on the total 5 / 10 / 20%.

### E.2 The table (central; low-high in brackets; full numbers in `lean_schedule_cost.csv`)

Samples per language assume a weak list of 25 languages at x2.5 (the dev-defined list) and 52 at x1.0, language floor ignored: weight-1.0 / weak, in prompts (completions = x16).

| stage | tokens / steps | samples per language (w1.0 / weak x2.5) | hyper-parameters carried over | GPU-h central [low-high] | USD central | evidence line for the cut |
|---|---|---|---|---|---|---|
| 0 calibration refit, prompt re-filter | CPU + short QE scoring | n/a | all penalties, anchors v3 design, lexicon design, LID guard (refit on H_cal / T2) | 0.5 [0.3-1.0] | 0.6 | rebuilds are CPU-bound (D) |
| 1 SFT A0 (77 languages) | **2.4 B tokens = 18,310 steps** [1.96 B = 1 pass, 14,950 - 3.0 B, 22,890] | per old UniMax: 1-2 passes, priority x4 | AdamW or Muon as pinned for A0/S1/S2 (identical across arms), 131,072 tokens/step, UniMax 1-2 passes, seed | 14.75 [11.0-20.4] | 17.6 | old steps 20,500-27,638 (0.94 B tokens, 26%) gave +0.05 chrF++ on clean rows, +0.44 on all rows; the single-stage clean corpus is only about 2.0 B tokens (ESTIMATED from 1.17 B target tokens x old 1.67 ratio), so 2.4 B = 1.2 passes. Weak: macro over 45 languages only, partly the cosine anneal; save checkpoints at 1.5 / 2.0 / 2.4 B and evaluate on H_sel |
| 1b SFT S1 (pre-registered) | 5,340 steps [3,430-6,870] (about 0.7 B tokens: 0.27 B specialist target tokens x 1.67 x A0 pass counts) | same passes as A0 | identical to A0 (prereg section 3) | 4.3 [2.5-6.1] | 5.1 | fixed by the pre-registration; tokens ESTIMATED |
| 1c SFT S2 (pre-registered) | as S1 + embeddings-only warm-up | same | as S1 | 4.4 [2.6-6.3] | 5.2 | fixed by the pre-registration |
| 2a RL stage A: b64 | **640 steps x 64 prompts x G16** [400-880] = 655 k completions | **358 / 894 prompts** (5.7 k / 14.3 k completions) | lr 2e-6, warm-up 10, std-normalised, group 16 all sampled, kl 0.05 k3c, guard 0.15 median-5, reference = SFT, full reward (QE ensemble, consensus, shaping, penalties, LID guard, anchors v3 refit) | 3.78 [2.0-7.5] | 4.5 | dev macro saturates about 300 k completions after SFT (B.1); weak groups complete at the end of 2b-b64 (B.2); AB2 dev peak at 20 steps = 320 prompts per language (B.5). Old dose to that point: w1.0 977 prompts / 10.3 k compl., weak 2,123 / 23.2 k, so the lean total (general RL, all bins) keeps 79% / 88% of those completions. The b64 step costs 20 s, so this is the cheap place to keep margin |
| 2b b256 | **150 steps x 48** [100-250] | 63 / 157 | same recipe, restart from the stage A model (reference refreshed, as 2c did) | 1.34 [0.85-3.2] | 1.6 | training reward plateau after step 150 (slope +0.005, CI [-0.011, 0.023]); stage delta +0.25 (just keeps the +0.2 the weak groups got) |
| 2c b512 | **120 steps x 42** [80-200] | 44 / 110 | same | 3.45 [1.5-8.3] | 4.1 | no trend over 446 steps (-0.001 +- 0.004 per 100 steps); stage delta -0.05; finishing learned within 50 steps |
| 2d b1024 | **60 steps x 42** [40-100]; **drop if the 1,024-token cap is target-side** (F.1) | 22 / 55 | same | 2.41 [1.5-5.8] | 2.9 | no trend; groups almost flat (sd 0.18); stage delta -0.07 (blend -0.11 [-0.22, -0.01]). Kept only as a short exposure because long-document benefit was never measured by stage (B.4) |
| 2e mixed | **60 steps x 42** [40-100] | 22 / 55 | same | 2.14 [1.3-5.2] | 2.5 | the stage delta was +0.20 [0.01, 0.44] / +0.21 [0.07, 0.36]: keep short |
| (b2048) | **dropped** (pre-registered) | | | 0 | 0 | eval reward -1.71 to -2.00 at all 22 evaluations; RL did not teach it; 11.9 h in 2c |
| 3a science inputs: RL archive, long-doc distillation (about 1,715 documents x 4 candidates, QE-ensemble choice), self-distilled anchors from the **new** RL model | generation + scoring | n/a | scripts `mix2_*.py` | 1.5 [0.8-3.0] | 1.8 | must be rebuilt (D); no timing in the record, ESTIMATED |
| 3b science SFT | **318 steps x 65,536 tokens** (A2W recipe, lr 3e-6, ratios 31.7 / 43.0 / 5.0 / 13.5 / 6.7) | n/a | unchanged | 0.35 [0.30-0.80] | 0.4 | A2W recipe won against four variants (A, B_v1, B_v2_ibom, C3 NLLB teacher +/-); cheapest stage with the clearest gain (+2.2 sentence, +12.7 paper-level over RL-final on test, REPORTED) |
| 3c alpha sweep + guardrails on S sets | alpha in {0.5, 0.7, 1.0} (+0.3 in high) | n/a | guardrail rule unchanged | 0.55 [0.40-0.80] | 0.7 | 8-10 min per alpha (VERIFIED log) |
| 4 final run (all reward components, NLLB signal on the language set decided on H_sel) | **150 steps x 42** [100-250] | 55 / 138 prompts | lr 1e-6, warm-up 20, kl 0.1, k3c, reference = start model, guard (section C), greedy statistics over sampled members, scale 0.3, cap 3, floor 2/20 steps, 70/30 science/general, WINDDOWN of the last 30 steps | 7.2 [3.8-17.1] | 8.6 | paper-level finishing was complete by step 50, sentence dev flat from step 50 (sd 0.10), all four stop criteria held at 350; gain over the start +0.15 / +0.17 (B.6); the lr confound says "do not extend without recording the effective lr" |
| 5 evaluations (A0, S1, S2, RL-final, final on E sets, once, after decisions are logged) | | n/a | | 2.5 [1.5-4.0] | 3.0 | ESTIMATED from AB2 evaluation times |
| **Production track subtotal** (0, 1, 2a-2e, 3a-3c, 4, 5) | | | | **40.5 [25.3-77.2]** | 48.2 | |
| margin 5 / 10 / 20% | | | | +4.0 | +4.8 | idle GPU, restarts, downloads |
| **Production track total** | | | | **44.5 [26.6-92.6]** | **53.0 [31.6-110.2]** | |
| S1 + S2 (pre-registered arms) incl. margin | | | | 9.6 [5.3-14.9] | 11.4 [6.4-17.7] | |
| **All stages total** | | | | **54.1 [31.9-107.5]** | **64.4 [38.0-127.9]** | |

Low = plateau-minimum steps, quiet card, 1-pass SFT. High = old-like budgets for every stage with a contended card; it is the sum of upper bounds and should be read as "the old pipeline with b2048 dropped", not as a likely outcome. Compared with the old pipeline (about 105-113 h, USD 125-137): RL phases 1-2c about 60 h -> 13.1 h central (b64 3.8 + b256-mixed 9.3), final run 28.4 h -> 7.2 h, SFT 18.4 h at the PRO 6000 rate -> 14.75 h. If the 1,024-token cap is target-side (F.1) subtract 2.4 h (USD 2.9).

### E.3 Stage rules that keep the safeguards and add cheap stopping

* Every stage ends at its step cap **or earlier** if both the H_sel macro dev chrF++ and the weak-dev macro gain less than 0.3 (3 sd of the 2c macro, sd 0.094) over the last 100 steps at two consecutive evaluations; extend toward the high cap only if the weak-dev still rises by 0.3 or more per 100 steps. The SFT is checkpointed at 1.5 / 2.0 / 2.4 B tokens for the same test.
* No safeguard is removed: k3c, median guard, grad clip, std-normalised advantages, no greedy member before the final run, LID guard, hard-failure floor, anchors, floor on language draws, restart at stage switches, LR_SCALE and WINDDOWN files, log-prob check. The curriculum order b64 -> b256 -> b512 -> b1024 -> mixed is kept because the only divergence in the record happened at a stage switch (2b, b64 to b256).
* The language weights stay at x2.5 for the dev-defined weak list (cost-neutral reallocation; the record neither proves nor refutes a benefit). Do not raise them beyond 2.5 and do not turn on the adaptive updater (it has never changed a weight and the dose-response is not identifiable).
* RL starts at lr 2e-6 from the SFT model: AB2 ran this recipe from SFT-final for 49 steps without a cut (KL 0.129 at step 49 with 16 prompts per language per step; the 77-language schedule delivers about 19x fewer prompts per language per step, so KL grows more slowly).

### E.4 Optional check before locking the long bins (about USD 3, ESTIMATED 2-2.5 h)

The stage branches of the old 2c run (`hub_stage_branches: true` in the yaml: end of the b256, b512, b1024, b2048 stages and `latest`; whether they are still on the private Hub repo was not checked) are the checkpoints behind B.2. Evaluating them on the AfriLongDoc core bins b256-b2048 (whole-document greedy, 8 languages) would settle whether b512/b1024/b2048 RL did anything for documents, which no file does now. The cost is a guess (AB2 evaluations took 3-5 min per four-language system; whole-document decoding is slower); if the answer is "b512 stage mattered for b512 documents", raise b512 to its high cap (200 steps, about +USD 3); if "nothing", drop b1024 to 40 steps.

## F. Open questions for the owner

1. **"Maximum 1,024 tokens"**: is the cap on the source window (b1024 = windows of up to 1,024 AfroXLM-R tokens, completions up to about 3,584 Gemma tokens, as in all old bins) or on the target side (completions of at most 1,024 Gemma tokens, which excludes b1024 and cuts b512)? The pre-registration says "target-side"; the old bins are named by source tokens. Costed with b1024 kept; dropping it saves USD 2.9 central.
2. **SFT dose**: accept 2.4 B tokens (about 1.2 passes of the clean corpus) instead of the old 3.0 B? A0's realised pass counts are inherited by S1/S2 (prereg), so this fixes the specialist arms too.
3. **Which arm gets the production RL, science stage and final run** (R5)? If a 20-language specialist wins, one RL run on 20 languages needs far fewer b64 steps (the dose is per language: about 130 steps for the same 320 prompts per language) but the product then covers 20 languages.
4. **The optional USD 3 AfriLongDoc check** of the old stage branches (E.4): yes or no?
5. **Guard changes**: persist the guard state in checkpoints, and the final-run threshold (0.15 as is, or 0.25)? Both are small code/config changes to a safeguard, so they are your call.
6. **Calibration size**: H_cal gives about 278 sentences per language (557 held-out ids split by parity) against 997 before, and AfriScience dev is an S set. Accept pooled priors where H_cal is too thin, and may the AfriScience-MT train references serve as the T2 source for the science anchors (the prereg lists only H_cal, SSA-MTE train and IbomMT train)?
7. **Rate and invoices**: USD 1.19/h is the lead's figure with no invoice in the record; the hours of phases 1, 2, 2b and SFT were never logged. Please confirm the rate and, if possible, the vast.ai invoice for those phases so the "old cost" in section 0 can be stated as verified.
8. **Seeds**: everything is one seed. Two extra S1 seeds cost about USD 10 central (2 x 4.3 h x 1.19). Worth it, or keep the money for the RL?

## G. Files, derivations and what I did not do

### G.1 Derivations that are not obvious
* Sum of unique-step times in 2c by bin; slopes with a block bootstrap (20/25-step blocks, 2,000 resamples, seed fixed): scripts were run in the session scratch directory; results are in the CSVs.
* SFT s/step: `(last time - first time) / steps` over `tokens_per_s` records of the AB2 `sft_weak` run (3,499 steps, 2.82 h = 2.90 s/step); S1: 350 steps in 0.257 h = 2.65 s/step.
* Phases 1, 2, 2b hours: completions per bin (from the yaml) x seconds per completion measured in 2c / AB2 (b64 19.2 ms, b256 38.8, b512 142.7, b1024 199.6, mixed 176.6): 3.6 + 10.2 + 2.7 h, plus 351 discarded b256 steps of 2b x 29.8 s = 2.9 h. Group size 8 may be slower per completion; treat as +-30%.
* Clean SFT tokens: 1.172 B estimated target tokens (D2 summary) / old 1.318 B target tokens x old 2.204 B pack tokens = about 1.96 B ESTIMATED; specialist target tokens 190 M (19 languages, `specialist/per_language_summary.csv`, VERIFIED) + yor 82 M estimated.
* 2b guard counterfactual: first index at which the median of five consecutive KL values from step 489 exceeds 0.15 in `rl2b_steps.csv` (step 536).

### G.2 CSV files in `reports/data/lean_schedule_2026-10-28/`
`old_pipeline_ledger.csv` (A), `dev_chrf_series_all_phases.csv` (B.1, B.5, B.6), `stage_end_group_chrf_deltas.csv` (B.2), `rl2c_bin_blocks_50steps.csv`, `rl2c_bin_trend.csv`, `rl2c_bin_plateau_test.csv`, `rl2c_eval_series.csv` (B.3), `per_language_dose_old_vs_lean.csv` (B.7, E), `kl_guard_and_instability_events.csv` (C), `reuse_table.csv` (D), `lean_schedule_cost.csv` (E).

### G.3 Not done
No GPU or box access, no new training, nothing published or committed. Not checked against an invoice. The pre-registered S1/S2 token counts and the time of the science-stage input generation are estimates, not measurements. No claim here is about a seed-to-seed variance because the record has none.
