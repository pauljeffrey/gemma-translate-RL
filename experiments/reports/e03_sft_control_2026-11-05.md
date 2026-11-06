# E-03 matched-compute continued-SFT control: execution record and result (2026-11-04 16:35 UTC to 2026-11-05)

Preparation: `reports/sft_control_prep_2026-11-04.md`. This file is the execution record; the final numbers and the verdict are added at the end. Progress rows (UTC, step, val) are in `reports/overnight_log_2026-11-04.md`.

## 0. Pre-registration (written 2026-11-04 17:00 UTC, before any control checkpoint exists, before any benchmark of the control was looked at)

Owner decisions of 2026-11-04 (via the coordinator): TranslatePsy en->xx data is authorised for this control only; control checkpoints are never pushed to the Hub or the bucket and never released; the final control checkpoint and all predictions are copied to the laptop archive. Arm C2 (end-of-turn x5) is dropped. The TranslatePsy half uses an EQUAL, explicitly fixed number of SAMPLES (source-target pairs) per language for the 19 languages (not just equal packs). The replay half stays as designed (UniMax over the last SFT shard + tail replay).

1. **Design.** Start `<HF_ORG>/gemma-0.4B-translate-sft` (SFT final, step 32,853). 6,000 steps x 32 packs x 4,096 tokens, Muon (fresh state), constant LR after a 100-step linear warm-up, seed 1234. Every step: 16 replay + 16 TranslatePsy packs (with at most +-1 pack moved between the halves in a few steps so that every TranslatePsy pack is used exactly once: see the table in section 1). No pack is drawn twice. Checkpoints for evaluation at steps 3,100 (= stage A's 4.42 h), 4,800 (= the compute behind mixed best, upper bound 6.83 h) and 6,000 (= the 8.66 h of the chain as spent).
2. **LR probe (pre-registered).** 300 steps each at LR 3e-6, 1e-5, 3e-5 on the first 300 steps of the same schedule (warm-up 100, constant after). Validation at steps 100/200/300 on `val/` (stage-B shard-6 validation split, macro over languages) and `val_tp/` (TranslatePsy validation, macro over the 19 languages). **Selection score = val macro + val_tp macro at step 300 (lower is better); candidates whose score is within 0.002 of the best are tied and the smallest LR among the tied candidates wins** (ties go to 3e-6). The choice is recorded before any benchmark is looked at. Because the LR is constant after warm-up, the winning probe IS the first 300 steps of C1 and C1 continues from its step-300 checkpoint (exact resume incl. Muon state); the other probes' checkpoints are deleted.
3. **Evaluation.** Same harness as the RL models: vLLM 0.32.0 bf16 greedy, 768 new tokens, FLORES+ devtest (54 languages), NTREX, MAFAND with held-out AfriCOMET-STL; AfriScience-MT dev/test sentence level, plain chrF (and SSA-COMET-QE if the pass is available); contamination flags attached as for the RL models (clean subset). Compared on identical sentences with stage A final, B1 best, mixed best and the SFT.
4. **Decision rule (primary endpoint: language-macro chrF++ over the 93 benchmark-language cells, FLORES+ 54 + NTREX 24 + MAFAND 15, clean rows; d = RL minus control; paired bootstrap with the Poisson-shared-per-source-sentence scheme of EVID, 1,000 resamples).**
   * RL is a better use of the compute: lower 95% bound of d > +1.0 chrF++ AND lower bound of the AfriCOMET difference > 0 AND d > 0 on the 19 TranslatePsy languages and on the other languages separately.
   * Comparable: the 95% interval of d lies inside (-1.0, +1.0) and the AfriCOMET interval contains 0.
   * The control is better: upper bound of d < -1.0.
   * Anything else: inconclusive.
   * Matching: stage A final vs control at step 3,100; mixed best vs control at step 4,800 (compute that produced it) and at step 6,000 (compute spent). The headline comparison is mixed best vs the step-6,000 control (as the chain is spent) and stage A vs step 3,100; checkpoints are never chosen by benchmark.
   * Reported next to it: the share of the SFT-to-RL gap the control closes, (C - SFT) / (RL - SFT), for chrF++, AfriCOMET, finished rate, loop rate; the 19 TranslatePsy languages vs the others; the 7 near-failed languages separately; plain chrF, spBLEU.
   * Caveats to state: one seed per arm; the TranslatePsy filters used AfriCOMET/SSA-COMET/MetricX, so neural metrics favour the control; TranslatePsy targets are NLLB-3.3B outputs and the benchmarks reward NLLB-like text; a differences below about 0.5 chrF++ is within seed noise (E-04 not done).

5. **Time rule (coordinator, recorded 2026-11-04 18:57 UTC, before any benchmark of the control was looked at).** If after 19:45 UTC the clean s/step projects the end of step 6,000 later than 06:00 UTC, the run is cut: C1 trains to the 4,800-step checkpoint (mixed-best-matched compute), which is exported and evaluated, and continues to 6,000 only if time allows. HARD STOP of training at 06:00 UTC wherever it is; the last registered checkpoint reached (3,100, 4,800 or 6,000) is exported and evaluated, and the report states which budget points were reached. Evaluation of 3,100/4,800 next to training only if the training s/step cost stays below 25%; otherwise after the stop. Final numbers and the copy to the laptop are due about 08:00 UTC.
6. **Micro-batch.** The first launch (micro-batch 8 x accum 4, as in the SFT) failed twice with CUDA OOM at step 2 (72-75 GB needed, side jobs held GPU memory); the control runs with micro-batch 4 x accum 8 (same 131,072 tokens per step, same gradient up to floating-point summation order). Failed attempt log: `reports/data/e03_control_execution_2026-11-04/` (copied at the end).
7. **TranslatePsy half, equal samples.** N = 277,900 source-target pairs per language for each of the 19 languages (5,280,100 samples, 96,138 packs, 393.8 M padded tokens), every pack used once; table in `reports/data/e03_control_execution_2026-11-04/tp_equal_samples_table.csv`.

8. **LR choice (made 2026-11-04 20:12 UTC, before any benchmark).** Probe results at step 300 (val macro / val_tp macro / score): 3e-6: 1.3535 / 0.6233 / 1.9768; 1e-5: 1.3620 / 0.5889 / 1.9509; 3e-5: 1.4088 / 0.5770 / 1.9858 (start: 1.3412 / 1.4501). Winner 1e-5 by the registered rule (no tie). The probe-3 checkpoint could not be written (ENOSPC) after its step-300 validation had been logged; its logged validation was used. C1 uses LR 1e-5 (constant after the 100-step warm-up), continued from the 1e-5 probe checkpoint at step 300.

## 1. Result in one paragraph

Continuing the SFT for the compute of the RL chain (6,000 steps of 131,072 tokens, 8.4 h of training at 5.05 s/step; RL chain 8.66 h) on a 50/50 mix of TranslatePsy en->xx (NLLB-3.3B outputs, 19 languages, an equal 277,900 pairs per language) and a replay slice of the last SFT shard, at LR 1e-5, gives language-macro chrF++ 29.97 on the 93 benchmark-language cells (FLORES+ 54, NTREX 24, MAFAND 15, clean rows), against 26.59 for the SFT and 34.55 for RL mixed best (33.94 for stage A). RL is ahead of the control by +4.58 chrF++ [+4.45, +4.70] and +0.0183 held-out AfriCOMET [+0.0172, +0.0192] overall. **The advantage is entirely outside the 19 TranslatePsy languages**: on the 47 cells of those languages RL minus control is -0.03 chrF++ [-0.13, +0.09] (stage A -0.20 [-0.31, -0.08]) and the control is better on AfriCOMET (-0.025), while on the other 46 cells RL is +9.29 [+9.09, +9.49] ahead because the control does almost nothing there (+0.15 chrF++ on FLORES+ over the SFT, finished rate 0.78 unchanged). **By the decision rule registered before the run the verdict is "inconclusive"**, because the registered condition "d > 0 on the 19 TranslatePsy languages" is not met (d = -0.03), although the other three conditions are. Read plainly: where a teacher-translated corpus exists (the NLLB-distilled TranslatePsy languages) continued SFT matches RL, finishing and looping included; RL's advantage is that it improves all languages from the model's own samples and a QE reward, including 35+ languages for which no such corpus exists. Only the 6,000-step point was evaluated; at the owner's request the 3,100-step (stage-A-matched) and 4,800-step (mixed-best-matched) points were NOT evaluated, so the compute-matched comparisons at those two budgets are untested (their bf16 exports are archived and can be evaluated later).

## 2. What was run (deviations from `reports/sft_control_prep_2026-11-04.md` marked)

* Data (box, CPU only, `nice 19`): TranslatePsy English->X rows streamed (2 shards x 160,000 rows per language), cleaned and scanned against the protected index on both sides (positive control 1012/1012; per-language drops in `reports/data/e03_control_execution_2026-11-04/reports_keep/`), then **an equal fixed N = 277,900 samples per language for all 19 languages** (5,280,100 samples, 96,138 packs, 393.8 M padded tokens; the first try N = 281,500 produced 97,383 packs and was re-packed), every pack used once; replay half 96,600 UniMax packs cut from stage-B shard 6 plus the tail replay (394.6 M tokens), 95,862 drawn. 138 of the 6,000 steps carry 17 TranslatePsy and 15 replay packs, the others 16 + 16 (so that each TranslatePsy pack is used exactly once). Table (n_samples, n_packs, tokens per language): `reports/data/e03_control_execution_2026-11-04/tp_equal_samples_table.csv`.

* Training: the SFT final `<HF_ORG>/gemma-0.4B-translate-sft`, Muon with a fresh state, 100-step warm-up then constant LR, weight decay 0.01, seed 1234. **Micro-batch 4 x accumulation 8** instead of 8 x 4 (the first launch ran out of GPU memory at step 2 while side jobs held memory; same 131,072 tokens per step). **LR probe (pre-registered) chose 1e-5** (not the 3e-6 of the prep): step-300 val macro / val_tp macro / score 3e-6: 1.3535 / 0.6233 / 1.9768; 1e-5: 1.3620 / 0.5889 / 1.9509; 3e-5: 1.4088 / 0.5770 / 1.9858; the probe-3 checkpoint write failed with ENOSPC after its validation had been logged (the logged numbers were used; /dev/shm was freed and the 1e-5 probe was continued as C1 from its step-300 checkpoint). Speed 5.05-5.15 s/step; 6,000 steps finished 04:30 UTC. Validation curve (78 languages, macro): 1.3412 at the start, 1.3694 at step 100, 1.3518 at 3,000, 1.3417 at 4,750, 1.3374 at 6,000; val_tp macro 1.4501 -> 0.4632. Per-step logs: `research_archive_2026-11-03/box_final_2026-11-04/e03_control/run_records_and_eval/c1_metrics.jsonl`.
* Compute actually used: 6,000 steps x 5.05 s = 8.4 h of C1 training (the chain: 8.66 h); the two losing probes cost 600 steps = 0.85 h more; the whole GPU session 18:45-04:30 UTC (9.75 h) plus 0.55 h evaluation and 0.2 h for the SSA-COMET pass. A concurrent evaluation of the 3,100 checkpoint slowed training by 57% and was aborted by the speed guard (rule: +25%).
* Evaluation: the same harness as the RL models (vLLM 0.32.0 bf16 greedy, 768 new tokens, FLORES+ devtest 54 languages, NTREX 24, MAFAND 15, held-out AfriCOMET-STL, AfriScience-MT dev/test sentence level), contamination flags attached as for the RL models (214 flagged rows removed from NTREX and MAFAND, none from FLORES+), paired bootstrap of EVID (Poisson weights shared per source sentence, 1,000 resamples) on identical sentences. Only the 6,000-step checkpoint (exported in fp32). The step-3,100 and 4,800 checkpoints were not evaluated (owner request).

## 3. Results (RL minus control; positive = RL better)

#### Headline (RL minus control, clean rows; chrF++ and AfriCOMET macro over cells; 95% paired bootstrap)

| RL model | cells | control chrF++ | RL chrF++ | RL - control chrF++ | RL - control chrF (plain) | AfriCOMET cells | control AfriCOMET | RL AfriCOMET | RL - control AfriCOMET |
|---|---|---|---|---|---|---|---|---|---|
| mixed best | 93 | 29.97 | 34.55 | +4.58 [+4.45, +4.70] | +4.91 [+4.78, +5.04] | 76 | 0.6101 | 0.6284 | +0.0183 [+0.0172, +0.0192] |
| stage A | 93 | 29.97 | 33.94 | +3.97 [+3.86, +4.08] | +4.22 [+4.10, +4.34] | 76 | 0.6101 | 0.6265 | +0.0164 [+0.0154, +0.0173] |
| B1 best | 93 | 29.97 | 34.33 | +4.36 [+4.24, +4.47] | +4.66 [+4.54, +4.78] | 76 | 0.6101 | 0.6274 | +0.0173 [+0.0163, +0.0182] |

#### By benchmark (mixed best minus control)

| benchmark | cells | control | RL | RL - control chrF++ | AfriCOMET control | RL | diff |
|---|---|---|---|---|---|---|---|
| flores_plus | 54 | 25.38 | 31.66 | +6.27 [+6.07, +6.45] | 0.5894 | 0.6267 | +0.0373 [+0.0355, +0.0389] |
| ntrex | 24 | 33.74 | 37.17 | +3.42 [+3.26, +3.59] | 0.6088 | 0.6139 | +0.0051 [+0.0037, +0.0064] |
| mafand | 15 | 40.44 | 40.78 | +0.34 [+0.17, +0.50] | 0.6672 | 0.6531 | -0.0141 [-0.0154, -0.0128] |

#### Language panels (the 19 TranslatePsy languages vs the others)

| RL model | panel | cells | control chrF++ | RL chrF++ | RL - control | AfriCOMET diff |
|---|---|---|---|---|---|---|
| mixed best | TP19 | 47 | 41.37 | 41.34 | -0.03 [-0.13, +0.09] | -0.0249 [-0.0257, -0.0241] (n=47) |
| mixed best | OTHERS | 46 | 18.32 | 27.61 | +9.29 [+9.09, +9.49] | +0.0883 [+0.0861, +0.0903] (n=29) |
| stage A | TP19 | 47 | 41.37 | 41.17 | -0.20 [-0.31, -0.08] | -0.0244 [-0.0252, -0.0235] (n=47) |
| stage A | OTHERS | 46 | 18.32 | 26.55 | +8.23 [+8.04, +8.41] | +0.0824 [+0.0803, +0.0843] (n=29) |

#### SFT to control and SFT to RL (language-macro chrF++ per benchmark)

| benchmark | SFT | control | control - SFT | mixed | mixed - SFT |
|---|---|---|---|---|---|
| flores_plus | 22.85 | 25.38 | +2.53 [+2.35, +2.71] | 31.66 | +8.81 [+8.59, +9.03] |
| ntrex | 29.25 | 33.74 | +4.49 [+4.29, +4.70] | 37.17 | +7.91 [+7.69, +8.12] |
| mafand | 35.77 | 40.44 | +4.67 [+4.39, +4.92] | 40.78 | +5.01 [+4.76, +5.23] |
| all 93 cells | 26.59 | 29.97 | +3.38 | 34.55 | +7.96 |

Share of the SFT-to-RL (mixed best) chrF++ gap closed by the control: 42.5% (arithmetic of the macros above; cell-count weighted).

#### Finished and loop rates (reward-family loop flag; clean rows; language macro)

| system | FLORES+ finished | FLORES+ loop | NTREX finished | NTREX loop | MAFAND finished | MAFAND loop | FLORES+ length ratio |
|---|---|---|---|---|---|---|---|
| SFT | 0.829 | 0.209 | 0.890 | 0.122 | 0.963 | 0.043 | 3.03 |
| control C1 (6,000) | 0.854 | 0.178 | 0.937 | 0.072 | 0.990 | 0.014 | 2.68 |
| stage A | 0.983 | 0.068 | 0.996 | 0.016 | 0.999 | 0.003 | 1.33 |
| B1 best | 0.993 | 0.054 | 0.999 | 0.009 | 1.000 | 0.001 | 1.17 |
| mixed best | 0.995 | 0.048 | 0.999 | 0.006 | 1.000 | 0.001 | 1.14 |

#### Pre-registered decision rule applied to mixed best vs C1 at 6,000 steps

| condition | value | met |
|---|---|---|
| lower 95% bound of d (chrF++, 93 cells) > +1.0 | d = +4.58 [+4.45, +4.70] | True |
| lower bound of the AfriCOMET difference > 0 | +0.0183 [+0.0172, +0.0192] | True |
| d > 0 on the 19 TranslatePsy languages | d = -0.03 [-0.13, +0.09] (AfriCOMET -0.0249 [-0.0257, -0.0241]) | False |
| d > 0 on the other languages | d = +9.29 [+9.09, +9.49] | True |

**Verdict by the rule as registered: inconclusive (registered rule: anything else).**


### 3.1 AfriScience-MT, sentence level, 6 languages (amh hau lug nso yor zul; test; plain chrF; SSA-COMET-QE and AfriCOMET reward-family / filter-family metrics)

| system | dev chrF | test chrF | test SSA-COMET-QE | test AfriCOMET-STL | test finished |
|---|---|---|---|---|---|
| SFT | 36.38 | 36.00 | 0.5686 | 0.6064 | 0.895 |
| stage A | 45.31 | 45.06 | 0.6118 | 0.6822 | 0.995 |
| B1 best | 45.46 | 45.30 | 0.6113 | 0.6813 | 0.997 |
| mixed best | 45.74 | 45.40 | 0.6130 | 0.6828 | 0.998 |
| control C1 (6,000) | 45.54 | 44.71 | 0.6214 | 0.6947 | 0.971 |

Per language, test chrF control vs mixed best: amh 31.05 vs 29.10, hau 56.25 vs 55.18, lug 44.06 vs 38.15, nso 40.64 vs 52.12, yor 34.51 vs 37.66, zul 61.78 vs 60.20 (`afriscience_sentence_chrf.json`, `sent_neural_c1.json`). nso is the one language without TranslatePsy data: the control stays at the SFT level (41.27 -> 40.64) while RL gains +10.9. The control is ahead of RL on SSA-COMET-QE and AfriCOMET on this set; both are in the family used to filter TranslatePsy (and are reward-family for RL), so they favour the control.

### 3.2 Finished / loop by language panel (reward-family loop flag; language macro, clean rows)

| panel | FLORES+ finished SFT / control / mixed | FLORES+ loop SFT / control / mixed | NTREX finished | NTREX loop | MAFAND finished | MAFAND loop |
|---|---|---|---|---|---|---|
| 19 TranslatePsy languages | 0.931 / 0.993 / 1.000 | 0.078 / 0.008 / 0.002 | 0.935 / 0.992 / 1.000 | 0.071 / 0.010 / 0.002 | 0.964 / 0.995 / 1.000 | 0.039 / 0.005 / 0.001 |
| other languages | 0.774 / 0.779 / 0.993 | 0.281 / 0.270 / 0.073 | 0.801 / 0.830 / 0.999 | 0.224 / 0.194 / 0.016 | 0.956 / 0.968 / 1.000 | 0.060 / 0.048 / 0.001 |

(`panel_finished_loop.json`.) The control fixes finishing and looping where it trained (TranslatePsy languages) and nowhere else.

### 3.3 Independent degeneration detector (E-08 detector: lzma residual, repeated char 8-grams, repeated word 3-grams; thresholds from the 0.1% tail of the 122,585 human references) next to the reward-family flag; language macro, all rows

| system | FLORES+ reward flag / independent / unfinished | NTREX reward flag / independent / unfinished | MAFAND reward flag / independent / unfinished |
|---|---|---|---|
| SFT | 0.209 / 0.311 / 0.171 | 0.122 / 0.166 / 0.110 | 0.043 / 0.075 / 0.037 |
| control C1 (6,000) | 0.178 / 0.268 / 0.146 | 0.071 / 0.101 / 0.062 | 0.014 / 0.033 / 0.010 |
| stage A | 0.068 / 0.121 / 0.017 | 0.016 / 0.031 / 0.004 | 0.003 / 0.009 / 0.001 |
| B1 best | 0.054 / 0.098 / 0.007 | 0.009 / 0.019 / 0.001 | 0.001 / 0.006 / 0.000 |
| mixed best | 0.048 / 0.085 / 0.005 | 0.006 / 0.014 / 0.001 | 0.001 / 0.004 / 0.000 |

FLORES+ ratio independent / reward flag: SFT 1.48, control 1.51, mixed best 1.77. Scripts: `reports/data/e03_control_execution_2026-11-04/c1_degeneration.py` (control; the other rows are the stored values of `reports/data/degeneration_audit_independent_2026-11-04/summary_by_model_benchmark.csv`). The control's loop count is also undercounted by the reward-family flag (ratio 1.51), as for the RL models; its unfinished rate is 0.146 on FLORES+ (mixed best 0.005).

## 4. How to read it, and what it does not show

* One seed, one LR (the probe's best of three; LR 3e-5 and 3e-6 are worse on the registered score), one budget point evaluated (6,000 steps). A different mix, a different LR schedule or a longer run could change the control. Differences below about 0.5 chrF++ are inside the unmeasured seed variation (E-04 not run).
* The control is not information-matched: it receives NLLB-3.3B translations (a teacher that scores 38.80 against 26.30 for the SFT on 43 languages, ledger N1) for 19 languages, which RL never sees. That RL matches this teacher-fed control on the 19 languages (chrF++ -0.03) without any teacher text in training is a favourable reading for RL; that RL beats it by +9.3 on the other 46 cells is a statement about breadth. A control with a teacher corpus for more languages does not exist in the data we hold.
* TranslatePsy was filtered with AfriCOMET, SSA-COMET and MetricX, so those metrics favour the control (AfriCOMET -0.025 on the 19 languages; SSA-COMET-QE on AfriScience 0.6214 vs 0.6130). chrF and chrF++ are the independent metrics (the headline); the registered rule used AfriCOMET as a second condition on the overall macro only.
* The registered verdict is "inconclusive" because of the panel condition. A reader who prefers the overall endpoint alone gets "RL is better" (d = +4.58, lower bound +4.45 > +1.0, AfriCOMET lower bound > 0); a reader who looks at the 19 TranslatePsy languages alone gets "comparable" (interval inside +-1.0 on chrF++; the AfriCOMET interval excludes 0 in favour of the control, so not "comparable" by the registered rule either). Neither re-reading is the registered verdict.
* The 3,100 and 4,800 checkpoints (stage-A-matched and mixed-best-matched compute) were NOT evaluated at the owner's request.
* The control checkpoint is trained on CC-BY-NC data and is not pushed anywhere; it is archived on the laptop only.

## 5. Ledger rows (to be pasted into `paper_385m/CLAIMS_LEDGER.md`; already applied by the E-03 agent: R14, CTL7-CTL11) and DISCLOSURES

See `paper_385m/CLAIMS_LEDGER.md` (R14 and section 12) and `paper_385m/DISCLOSURES.md` (TranslatePsy row and the process bullet). DISCLOSURES text: TranslatePsy English->X rows (CC-BY-NC-4.0, NLLB-3.3B outputs, filtered with AfriCOMET/SSA-COMET/MetricX) were used for the continued-SFT control only (277,900 pairs per language, 19 languages, 5.28 M pairs, 96,138 packs seen once), after scanning both sides against the protected benchmark index (all rows with a hit removed); the control checkpoint is a research artefact under CC-BY-NC-4.0 and NLLB terms, kept on the authors' storage, never released and never used in a released model.

## 6. Files and integrity

* Laptop archive: `research_archive_2026-11-03/box_final_2026-11-04/e03_control/`: `c1_s6000_final_checkpoint_fp32/` (final control weights; model.safetensors sha256 60d85aa823abbd94043f9626fd5f91915f4de3635c0491d56776d9564fe707d6, equal on box and laptop), `c1_s3100_export_bf16_NOT_EVALUATED/` and `c1_s4800_export_bf16_NOT_EVALUATED/` (sha256 in `SHA256SUMS.txt`), `run_records_and_eval/` (predictions of the 6,000-step control, metrics, probe logs, `lr_choice.json`, harness outputs, AfriScience predictions, AfriCOMET cache, paired-bootstrap JSON, trainer logs). Optimizer state of the final run (trainer_state.pt, 3.8 GB) was not archived.
* Scripts, tables and configs: `reports/data/e03_control_execution_2026-11-04/` (`stage_*.sh`, `gpu_phase.sh`, `eval_c1.sh`, `run_analysis.sh`, `make_tables.py`, `c1_degeneration.py`, `panel_finished_loop.py`, `verdict_tables.md/json`, `evidence_c1/`, `tp_equal_samples_table.csv`).
* Nothing was pushed to the Hub or the bucket (training ran with all tokens unset and `HF_HUB_OFFLINE=1`). Freed on the box afterwards: `/dev/shm/control/{runs,export,hf_home,sft_model,pack_e03,tmp}` (about 11.8 GB), `/workspace/eval_385m/out_c1_s3100` (aborted partial, 13 MB); kept: `/dev/shm/control/combined` (3.2 GB packed data, rebuildable in about 2 h), `sched_ctl`, logs, and the harness output `out_c1_s6000` (235 MB) on the root disk.
