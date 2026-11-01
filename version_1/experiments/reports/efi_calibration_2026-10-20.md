# efi (Efik): QE-free consensus calibration and validation (D-43), 2026-10-20

Status: calibration file, yaml and tests are ready. **The running trainer does not use them.** It was restarted at 05:05:53 UTC (resumed from the step-350 checkpoint) and loaded the config as it was at that time, i.e. with the other session's efi setting (its log reads qe=32, blend=22, consensus=23, so efi is in blend mode); see section 0.

## 0. Read this first: two efi calibrations exist

While this task ran, another session calibrated efi with the AfriCOMET-QE scorer and edited the same yaml (rental, 04:43 UTC):

| | calibration file | efi content mode | efi advantage weight | efi draw weight |
|---|---|---|---|---|
| other session (in effect now) | `calibration/shaping_flores_dev_v6.json` | `blend` (beta 0.5), QE AUC 0.731, reliability_same_source 0.898 | **0.898** | 1.0 (key removed from `language_weights`) |
| this task (prepared) | `calibration/shaping_flores_dev_v6_efi_consensus.json` | `consensus` (no QE call) | **0.5** | **2.5** |

`languages.py` lists efi as QE-validated (`africomet_code` efi, VERIFIED), which is why the other session could run QE on it. The brief for this task said efi is not on the card. Either way the QE score does carry some signal for efi (section 4), just less than the consensus signal.

`shaping_flores_dev_v6.json` (other session) was left untouched. The yaml now points to the new file (`v6_efi_consensus`). It takes effect only at the next trainer start (a crash restart, or the planned restart at a later stage boundary: b512 -> b1024 is at step 837; the b256 -> b512 boundary at 391 has already passed). To go back to the other session's setting: restore `configs/rl_rtxpro_phase2c.yaml` from `/workspace/sft_build/efi_calib/rl_rtxpro_phase2c.yaml.before_D43` (rental) or set `reward.calibration_path` to `.../shaping_flores_dev_v6.json` and delete `efi: 2.5` from `language_weights`.

## 1. Reference pairs and split

Sources (English -> efi, human): SmolSent 863, SmolDoc 130 documents = 1,609 sentence pairs (the `srcs`/`trgs` lists are sentence aligned), IbomMT test 1,012 (FLORES+ devtest English, paid linguists, CC-BY-NC-4.0). GATITOS (4,001 single words and phrases) is not used. NFC, de-duplication by English (only 3 duplicates): **3,481 pairs**.

Split: `sha1("calib-split-v1" + NFC English)` < 0.5 -> calibration, else hold-out (`calibrate_refs.split_rows`).

| | pairs | SmolSent | IbomMT | SmolDoc | English seen in SFT | reference pair seen in SFT |
|---|---|---|---|---|---|---|
| calibration | 1,791 | 432 | 545 | 814 | 1,533 (85.6%) | 0 |
| hold-out | 1,690 | 431 | 467 | 792 | 1,467 (86.8%) | 0 |

Contamination (flags from `bench/test_sets/_contamination/`): the English side of about 86% of the pairs occurs in the SFT windows (FLORES+ English and SMOL English reached the SFT data through other languages), but no efi reference pair was seen (pair flag 0/3,481). For the consensus check this matters little: the model has seen the English prompt, not the Efik reference. The greedy baseline below is split by the flag. The hold-out was never used for the entry.

## 2. Calibration entry (calibration half only, 1,791 pairs)

Shape of the other 23 consensus-only entries of v5 (`content {mode: consensus, beta: 1.0}`, `lid` block kept as measured: efi_Latn, recall 0.995, min_chars 30), plus the human-reference statistics. No `anchors`, no QE statistics (a consensus-mode language never calls QE; the shaper falls back to the default anchors it does not use).

| block | value |
|---|---|
| weight | `reliability` = `reliability_same_source` = 0.5, `usable: true` |
| len_ratio | p01 0.554, p02 0.593, p05 0.680, p50 1.000, p95 1.414, corpus 1.003, s2 4.58 |
| omission | cost 12 (false alarm 3.3% on untouched documents; the 2% target is not reached at any cost, largest cost used by the repo rule; false alarm by cost 4/5/6/8/12 = 5.3/4.8/4.5/3.9/3.3%); 947 windows; 16.8% of the windows have different English / Efik sentence counts |
| diacritics | applies; density mean 0.0665, p10 0.0244 (threshold), p50 0.0652, zero-mark sentences 3.0%; marks: U+0323 6,768, U+0304 2,854, U+0308 1,694, U+0303 1,030, U+0301 8, U+0307 4 |
| lexical | anchor recall p10 0.0 / p50 0.571 (n 644), anchor precision p10 0.0 / p50 1.0 (n 1,791): the p10 of 0 means the lexical penalty never fires for efi |

Caveat, orthography: SmolDoc writes Efik with a diaeresis (`ufök`, `nnyïn`), IbomMT and SmolSent with the under-dot (`ufọk`). U+0308 therefore enters the allowed inventory (1,694 marks, versus 4 in the IbomMT-only entry of the other session). The reward will not flag ö/ï as foreign marks for efi. If the SFT targets use only the under-dot, tighten this by removing `0308` from `diacritics.allowed.marks` and the `|0308` clusters in the entry.

The other session's entry (IbomMT only, 997 pairs, all of them used for calibration) has len_ratio p05 0.593 / p95 1.382 / corpus 0.953 and omission cost 12 too; mine is wider at the top and closer to 1.0 because SmolDoc and SmolSent are included.

## 3. Consensus validation

Model: the current 2c policy, exported from `runs/rl2c/last/trainer_state.pt` (the step-300 checkpoint, written 04:40 UTC, b256 stage) into HF format at `/workspace/sft_build/efi_calib/model_rl2c_latest` (`hub_stage_latest` does not exist in `runs/rl2c`; the push goes to the Hub branch only). CPU, fp32, HF `generate`, temperature 1.0, top_p 1, top_k 0, 8 samples per source, `build_prompt`, max_new_tokens 384, 150 hold-out sources of 5-45 English words (seed 0). Consensus = `TranslationReward` with the language forced to `consensus` mode and a scorer that raises if called (mean chrF++ against the valid siblings, as in training, including the LID, repetition and failure checks). Quality = sacrebleu chrF++ (word_order 2) of the reward's cleaned text against the human reference. Note the training group size is 16; the check uses 8.

Within-prompt Spearman = per source, rho(consensus, reference chrF++) over its valid samples, averaged over sources (this is what GRPO's group standardisation sees).

| language | sources | valid samples | within-prompt rho (mean +- se) | median | sources with rho > 0 | centered (all samples) | overall (pooled) | mean ref chrF++ |
|---|---|---|---|---|---|---|---|---|
| **efi** | 150 | 1,197 | **0.380 +- 0.033** | 0.452 | 81% | 0.419 | 0.611 | 26.8 |
| ach | 150 | 1,198 | 0.330 +- 0.030 | 0.357 | 82% | 0.375 | 0.644 | 27.6 |
| nyn (SALT) | 150 | 1,176 | 0.383 +- 0.032 | 0.464 | 81% | 0.426 | 0.656 | 27.2 |
| ibb | 150 | 1,199 | 0.266 +- 0.031 | 0.345 | 74% | 0.308 | 0.412 | 21.1 |

efi behaves like the two existing consensus-only languages (ach, nyn) and better than ibb (Ibibio, the closest relative; IbomMT only, lowest reference chrF++). Status counts for efi: 1,197 ok, 1 each degenerate / wrong_language / bad_prefix.

**Decision rule (>= 0.25 -> 0.5; 0.10-0.25 -> 0.3; < 0.10 -> 0): efi 0.380 -> advantage weight 0.5.** Even the lower end of the interval (0.38 - 2 se = 0.31) stays above 0.25.

### QE on the same samples (efi, 1,197 samples, 149 groups)

| signal | within-prompt rho | centered |
|---|---|---|
| consensus | 0.380 | 0.419 |
| AfriCOMET-QE alone | 0.220 | 0.221 |
| 50/50 blend (group z-scores) | 0.383 | 0.409 |

QE alone would land in the 0.3 band of the same rule; the blend is no better than consensus alone. QE vs consensus within prompt: rho 0.21. So for efi the other session's weight of 0.898 is not supported by this check, and the QE half adds nothing over consensus. (The 0.898 is the QE same-source accuracy against corrupted references, a different quantity from agreement between the model's own samples.)

### Greedy baseline (150 hold-out sources, same selection; not used for any decision)

| model | mean chrF++ | failures | chrF++ on 21 sources whose English was not in the SFT text | on the 129 seen |
|---|---|---|---|---|
| SFT final (`<HF_ORG>/gemma3-270m-translate` main, 376a94f) | 19.72 | 18 of 150 (11 no end of turn within 384 tokens, 5 degenerate, 1 wrong language, counted as 0) | 19.39 | 19.78 |
| current 2c policy (step 300 checkpoint) | 28.07 | 1 degenerate | 24.01 | 28.73 |

The gain from 19.7 to 28.1 is real in the aggregate, but 86% of the sources have English the model saw in training, and RL has had almost no efi samples, so most of it is likely general RL improvement (finishing the sentence, no loops). The clean subset is only 21 sources.

## 4. Draw weight and prompts

- Prompts: `/workspace/sft_build/gemma/rl_prompts_v2/efi.parquet` holds 90,000 prompts in each of b64, b256, b512 and b1024 (360,000); all 77 languages have every bin, so the mixed stage works.
- `PromptSchedule` on the real prompt set (5.94 M rows), config as edited: stages b256 0, b512 391, b1024 837, mixed 986. Simulated efi share of the draws: b512 2.9% (148 of 5,040 prompts over 120 steps), b1024 2.6%, mixed 2.9%; expected 2.4% (2.5 / sum of weights). Without any weight (uniform) it would be 1.0%; with the old 0.0001 it is 0.
- At the b512 batch sizes (42 prompts per step) efi gets about 1 prompt per step.

## 5. What changed

Laptop and rental (`/workspace/sft_build/gemma_mt_2c`):

| file | change |
|---|---|
| `gemma_mt/rl/calibrate_refs.py` (new) | pair loading, deterministic split, reference statistics, consensus entry, Spearman, decision rule |
| `scripts/build_reference_split.py`, `scripts/build_consensus_entry.py`, `scripts/validate_consensus.py`, `scripts/merge_validation.py`, `scripts/qe_vs_reference.py` (new) | the pipeline of this report |
| `scripts/dry_check_rl.py` | no longer hard-codes efi as the one language with advantage weight 0 / no anchors; it reports efi's mode and weight and requires anchors only for non-consensus QE languages (backwards compatible) |
| `tests/test_calibrate_refs.py` (new, 5 tests) | split determinism, de-duplication / contamination flags, Spearman and the decision rule, entry loads through `ContentShaper` as a usable consensus-only language, reward never calls QE for it |
| `calibration/shaping_flores_dev_v6_efi_consensus.json` (new) | v5 + the efi entry; v5 and the other session's v6 untouched; copy also in `/workspace/sft_build/gemma_mt/calibration/` |
| `configs/rl_rtxpro_phase2c.yaml` | `reward.calibration_path` -> the new file (path under `/workspace/sft_build/gemma_mt/calibration/`, like before); `language_weights` + `efi: 2.5` |
| `DECISIONS.md` | D-43 (laptop and rental) |

No existing module was changed (reward, shaping, consensus and calibrate_shaping read the entry as it is); `calibrate_shaping.py` still treats efi as "no data" by default because the QE path is unchanged.

Work files on the rental: `/workspace/sft_build/efi_calib/` (refs/, work/, logs/, `model_rl2c_latest`, `summary/`). Condensed results are in `reports/efi_calibration_2026-10-20/efi_calibration_results.json`.

## 6. Checks

- `scripts/dry_check_rl.py configs/rl_rtxpro_phase2c.yaml --real-lid` (CPU): DRY CHECK OK. 77 languages with a content mode {qe 32, blend 21, consensus 24}, no language with advantage weight 0, efi mode consensus weight 0.5, real GlotLID passes, the reward runs on scripted samples of all 77 languages, archive hook OK. The same check on the other session's v6 reports efi blend / 0.898.
- Tests on the rental (CPU, nice): 381 passed, 6 skipped, 0 failed with `test_stratified.py::test_pusher_is_disabled_without_repo_or_token` deselected. That test fails when run alone (pre-existing: a Hub token file exists on the box).
- The GPU was not used, the trainer was not touched, nothing was pushed.

## 7. Revert

1. `cp /workspace/sft_build/efi_calib/rl_rtxpro_phase2c.yaml.before_D43 /workspace/sft_build/gemma_mt_2c/configs/rl_rtxpro_phase2c.yaml` (and `git checkout`/restore the laptop copy), or set `reward.calibration_path` back to `.../shaping_flores_dev_v6.json` and drop `efi: 2.5`.
2. Delete `calibration/shaping_flores_dev_v6_efi_consensus.json` (nothing else refers to it).
