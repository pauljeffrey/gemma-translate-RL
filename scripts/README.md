# scripts/

Batch and one-off tools, grouped by purpose. Each script documents its own arguments in its header; `python scripts/<name>.py --help` where it has a CLI.

## Ablation 1 (weak-language SFT/RL ablation)

| Script | What |
|---|---|
| `ab1_analyze.py` | AB1 analysis: does RL keep its gain when it starts from a much weaker SFT checkpoint?  Paired sentence bootstrap on FLORES+ devtest (no_pair rows), pe |
| `ab1_build_weak_data.sh` | usage: URL=<presigned> bash build_weak_data.sh STEPS |
| `ab1_build_weak_sft_data.py` | AB1 (RL-vs-SFT-data ablation): build the data of the WEAK SFT run without unpacking the 8.8 GB packed SFT set. |
| `ab1_cost.py` | AB1 cost model (planning calculator, no GPU): GPU hours and dollars of the ablation = weak SFT + two RL arms + the six FLORES+ evaluations, at central |
| `ab1_make_configs.py` | AB1: write configs/rl_ablation_full.yaml and configs/rl_ablation_weak.yaml from the phase-2c b2048 config (the one that ran 1,433 stable steps), so th |
| `ab1_make_prompts.py` | AB1: the RL prompt set of the ablation = a seeded random subset of rl_prompts_v3, b64 and b256 rows only (both arms read the same files, so every step |
| `ab1_patch_tree.py` | AB1: make a gemma_mt tree fit for the RL-vs-SFT-data ablation. Idempotent text patches of `gemma_mt/rl/train_rl.py`; nothing else changes. |
| `ab1_presign.py` | AB1 helper (run on the LAPTOP, where the bucket keys live): print a time-limited presigned GET URL for one object of the object-storage bucket, so the GPU bo |

## Ablation 2 (specialist SFT arm vs RL)

| Script | What |
|---|---|
| `ab2_analyze.py` | AB2 analysis: RL from a strong vs a weak SFT start (arms F, W), and RL against two continued-SFT controls (S1, S1P) from the weak start.  CPU only. |
| `ab2_build_s1_data.py` | AB2: data of the CONTINUED-SFT controls, cut from the 8.8 GB packed SFT set `gemma/packed_fwd` (538,480 train packs x 4,096 tokens) as a STREAM (a `ta |
| `ab2_comet.py` | AfriCOMET scorers. Three interchangeable implementations of the same call: |
| `ab2_cost.py` | AB2 cost model (planning calculator, no GPU): GPU hours and dollars of the cheap ablation, central / high, at 1.19 USD/h, with the owner's hard cap of |
| `ab2_drawn_prompts.py` | AB2: which prompts does an RL arm actually draw?  Replays the trainer's own prompt loading, hold-out split and PromptSchedule (a pure function of seed |
| `ab2_eval.sh` | AB2 evaluation of ONE model on yor kin orm lin: FLORES+ devtest, NTREX, MAFAND, SMOL-sent (greedy, 768 new tokens, vLLM at 10% of the card), then per- |
| `ab2_joint_budget.py` | Joint-length budget of UnifiedMetric scorers (SSA-COMET-MTL), as PURE functions (D-55). |
| `ab2_make_configs.py` | AB2: write the RL arm configs (F, W) and the SFT configs (weak SFT, S1, S1P) of the cheap ablation. |
| `ab2_make_prompts.py` | AB2: the RL prompt pool of the cheap ablation = b64 windows of 4 languages (default yor kin orm lin) drawn at random from rl_prompts_v3, DECONTAMINATE |
| `ab2_neural.py` | E1 pairwise benchmark comparison, step 2 (GPU, box): neural MT metrics per sentence for several systems on the same benchmark rows. |
| `ab2_patch_sft.py` | AB2: idempotent patch of gemma_mt/train_sft.py in the AB2 tree (never the shared tree) for the CONTINUED-SFT controls. |
| `ab2_set_n.py` | AB2: choose N, the number of SFT steps of the GPU-time-matched control S1, from the MEASURED times (rule of DECISION_RULE.txt section 2: the S1 job's  |

## Benchmark contamination audit

| Script | What |
|---|---|
| `contam_audit_01_flag_tables.py` | Tables from the SFT contamination scan outputs (2026-10-26 contamination audit, agent C1, section 1). |
| `contam_audit_02_benchmark_tables.py` | Task 3 of the 2026-10-26 contamination audit: per-benchmark tables from reports/data/final_model_per_language.csv. |
| `contam_audit_03_public_source_scan.py` | Re-scan of PUBLIC Hub sources for FLORES+ dev/devtest text (2026-10-26 contamination audit, agent C1). |
| `contam_audit_04_mafand_in_afrolingu.py` | Is the MAFAND-MT TEST set inside UBC-NLP/AfroLingu-MT (an SFT source: converter `afrolingu_mt`, train split)? (2026-10-26 audit, agent C1) |
| `contam_audit_05_pool_sample.py` | Sample-based attribution of the FLORES+ exposure of the SFT text (2026-10-26 contamination audit, agent C1, section 1). |
| `contam_audit_06_dev_set_composition.py` | Composition of the RL development / monitoring sets (2026-10-26 contamination audit, agent C1, section 2). |
| `contam_audit_07_pool_all_benchmarks.py` | Same sample as contam_audit_05 (87,498 SFT windows, private Hub backup r10/pool), for EVERY benchmark of the paper (2026-10-26 audit, agent C1). |
| `contam_audit_08_luganda_source_match.py` | Which public dataset is the Luganda stream that carries FLORES+ English? (2026-10-26 audit, agent C1, section 1) |
| `contam_audit_09_candidate_overlap.py` | Overlap of candidate FRESH English->African test sets with what our models saw (2026-10-26 audit, agent C1, section 5). |
| `contam_audit_10_pool_rescan_with_original_scanner.py` | Run the ORIGINAL scanner (gemma_mt/eval/contamination.py: exact = sentence OR contiguous token run, pair = reference in the same-language target) on t |
| `contam_audit_11_pool_scaling.py` | Scale the SFT-window sample hits to the full SFT text PER LANGUAGE (the sample is ~780-2,600 windows per language, not proportional to size). |
| `contam_audit_12_decontaminate.py` | Decontamination filter for training / reward-reference text against protected evaluation sets (protocol of section 5 of the 2026-10-26 audit). |
| `contam_audit_13_ssamte_in_sft_sample.py` | Is the SSA-MTE train split (converter `ssa_mte` of the SFT build: English source + human reference) in the SFT windows? Test on the 87,498-window samp |
| `contam_audit_common.py` | Shared helpers of the 2026-10-26 contamination audit (agent C1). Read-only: no Hub writes, no secrets printed. |

## Benchmarks and speed

| Script | What |
|---|---|
| `bench_anchors_v2.py` | D-61: seconds per reward call (CPU, stub scorer) with the anchors off / v1 / v1 + v2, on a realistic training batch: 84 prompts x 16 samples, human do |
| `bench_attention.py` | Micro-benchmark of ONE attention call (fwd+bwd), Gemma-3-270m shapes: 4 q heads, 1 kv head, head_dim 256, bf16. |
| `bench_reward_b2048.py` | CPU cost of the reward code path for LONG sources (D-45: the b2048 bin). Nothing is trained, no GPU, no QE model (stub scorer). |
| `bench_step.py` | fwd+bwd time of one micro-batch of real packs for attention / loss implementations (the part of a training step that |

## Builders and report generators

| Script | What |
|---|---|
| `build_consensus_entry.py` | Write a calibration file = BASE + a QE-free consensus entry for one language built from the CALIBRATION half of its human reference sets (D-43). |
| `build_decoding_study_report.py` | Render reports/final_model_decoding_study_2026-10-21.md from reports/data/final_model_decoding_study/{results,qe_results,extra}.json (written by the b |
| `build_flores_dev_all.py` | Collect the FLORES+ DEV pairs (English -> language) of every RL language into ONE jsonl, for `calibrate_shaping --bench-file`. |
| `build_funcwords.py` | Per-language word statistics of the TARGET-language training text, for the stop-word profile of the anchors (D-61, rl/anchors_v2.py). |
| `build_generation_config_report.py` | Render the TABLES of reports/generation_config_study_2026-10-22.md from reports/data/generation_config_study/json/*.json (written by the box scripts k |
| `build_lexicons.py` | Build the English -> X lexicons of the dictionary-overlap signal (D-57, agent B3). CPU only (numpy), no network except the optional `gather`. |
| `build_reference_split.py` | Collect the English -> X human reference pairs of one language from the rare-language test sets (D-42), de-duplicate by English, split them |
| `build_rl_prompts_b2048.py` | Build the b2048 length bin of the Gemma RL prompts (DECISIONS.md D-45) and the prompt directory `rl_prompts_v3`. |
| `build_rl_prompts_new.py` | RL prompt files for the languages added to the consensus mode on 2026-11-01 (guw lue tll zne lub crs mfe arz). No reference translation is needed, so  |
| `build_science_anchor_calibration.py` | Thresholds of the science anchor (rl/sci_anchor.py, DECISIONS.md D-52) from the HUMAN references of AfriScience-MT TRAIN (sentence level, six language |
| `build_science_prompts.py` | Prompt set of the SCIENCE RL run (DECISIONS.md D-52, owner 2026-10-21): English scientific text windows for the 77 RL languages plus a 30 % replay sha |
| `build_sft_science_mix.py` | Build the data of the SHORT SCIENCE-SFT STAGE that runs between the general RL (step 1433) and the science RL (D-58). |
| `build_weak_dev.py` | Build the WEAK-LANGUAGE monitor set of the RL run (D-56): FLORES+ dev for every language whose reward has no (consensus) or a weak (blend) AfriCOMET s |
| `build_word_ratio_calibration.py` | Add a per-language WORD-ratio block to the shaping calibration (owner 2026-10-21): shaping_flores_dev_v6.json -> shaping_flores_dev_v7.json. |

## Clean rerun tooling

| Script | What |
|---|---|
| `clean_rerun_make_configs.py` | Writes the RL stage configs of the CLEAN RERUN (agent C1, 2026-10-28) under gemma_mt/configs/clean_rerun/ (arm A0, 77 languages) or |
| `clean_rerun_nllb_draws.py` | Which prompts do the RL stages DRAW for the NLLB-signal languages? (clean rerun, agent T2, 2026-10-29) |
| `clean_rerun_tools.py` | Helper commands of the clean-rerun RL launcher (agent C1, 2026-10-28). CPU only; nothing here trains. |

## Evaluation

| Script | What |
|---|---|
| `eval_afriscience.py` | Evaluate a checkpoint on AfriScience-MT (dsfsi/afriscience_mt, Apache-2.0; English -> amh hau lug nso yor zul), DECISIONS.md D-52. |
| `eval_anchors.py` | Measure the constraint penalties of rl/anchors.py (D-56): false positives on HUMAN references, detection of corruptions, and the rates on a model's ou |
| `eval_anchors_v2.py` | Measure the D-61 detectors (rl/anchors_v2.py) out of sample: false positives on held-out HUMAN references, AUC and detection against synthetic attacks |
| `eval_sft_benchmarks.sh` | Post-SFT benchmark evaluation (owner 2026-10-18): the SFT model and the base gemma-3-270m-it, English -> African, on FLORES+ devtest, NTREX-128 and |

## GPU box setup

| Script | What |
|---|---|
| `vast_calibrate.sh` | Per-language reward calibration (run ONCE before RL, on the DEV split of FLORES+). |
| `vast_eval.sh` | Evaluate one system on FLORES+ / NTREX-128 / MAFAND-MT; call once per system, then build the report. |
| `vast_prepare.sh` | Build the packed SFT dataset from raw parquet/jsonl (columns english, target, iso). |
| `vast_setup.sh` | One-time environment setup on a fresh Vast.ai instance (Linux, CUDA GPU, 11-24 GB). |
| `vast_train_rl.sh` | Stage 2: GRPO with the shaped AfriCOMET-QE reward. Single GPU only. |
| `vast_train_sft.sh` | Stage 1: SFT. Single GPU by default; NGPU=2 uses torchrun (DDP). |

## Hub / publishing

| Script | What |
|---|---|
| `archive_publish.py` | Merge the RL best-translation archives, apply a quality gate and (only if it passes) push them to a PRIVATE Hub dataset. |
| `push_sft_variant.py` | D-59: push one SFT-stage model (a trained variant, or an interpolated alpha of it) as a model folder `sft/<folder>/` of a Hub repo (owner-authorised 2 |

## Inter-African (X to Y) data and rewards

| Script | What |
|---|---|
| `afri2afri_dataset_card.py` | Generate README.md (the Hub dataset card) of the inter-African dataset from the builders' small outputs. Nothing is uploaded. |
| `afri2afri_fetch_mined.py` | Download the NLLB-mined `michsethowusu/<lang>-<lang>_sentence-pairs` repos (both languages among the 77) and convert them to parquet shards, one per l |
| `afri2afri_language_table.py` | One row per target language: what we hold for it (en->X pairs, pivot partners, direct rows, monolingual text, benchmark sets, reward family). numpy/js |
| `afri2afri_push_full.py` | Upload the `full_pairs/` tree to the dataset repo with the Hub's resumable large-folder uploader, and (optionally) mirror it to the object-storage bucket. DR |
| `afri2afri_push_hub.py` | Push the staged inter-African dataset to the Hub in two steps. DRY RUN unless --yes. The token is read from the environment (HF_WRITE_TOKEN, then HF_T |
| `afri2afri_report_tables.py` | Tables and heat-maps for the inter-African data report (D-46) from the small outputs of the builders (numpy + json only, runs on the laptop). |
| `afri2afri_reward_probe.py` | CPU probe of reference-free scorers for X -> Y translation (D-46, deliverable 4): is a score of (African source, African translation) usable as a rewa |
| `afri2afri_scout_sources.py` | List, for each of the 77 languages, which open monolingual corpora of the Hub have a folder for it (metadata calls only, no download) -> json. |
| `build_afri2afri_full.py` | The COMPLETE inter-African pair set as sharded parquet (D-46, owner 2026-10-20): every pivot-derived pair, every direct pair of the SFT corpus; the NL |
| `build_afri2afri_rl.py` | Inter-African RL prompt builder (D-46): monolingual SOURCE windows + target tag, in the `rl_prompts_v2` format family. CPU only. |
| `build_afri2afri_sft.py` | Inter-African SFT builder (D-46): X -> Y training windows from the data of the en->X build. CPU only, runs `nice`d on the old box. |

## Language ID and fluency

| Script | What |
|---|---|
| `fluency_lm.py` | Per-language CHARACTER n-gram fluency model for the RL reward (reference-free, CPU only, a few MB per language). |

## Long-document benchmark

| Script | What |
|---|---|
| `longdoc_bakeoff.py` | Translator bake-off on human sentence sets (D-44). Thin wrapper: see gemma_mt/eval/longdoc_bakeoff.py (`--help`). |
| `longdoc_build_hard.py` | Build the HARD / DOMAIN split (medicine, law, maths-science) of the long-document benchmark from the pool. |
| `longdoc_build_source_set.py` | English source set of the long-document benchmark (D-44). Thin wrapper: see gemma_mt/eval/longdoc_select.py (`--help`). |
| `longdoc_contam_fetch_negatives.py` | Fetches the NEGATIVE control texts of the contamination validation: 90 Wikinews articles of 2025-08..2026-05 through the MediaWiki API (UA with contac |
| `longdoc_contam_hard.sh` | Contamination scan of the HARD-split candidates (CPU, nice 10, 10 processes) in a PRIVATE pool directory: symlinks to every pool file plus the staged  |
| `longdoc_contam_ngram_study.py` | Token n-gram length study (rental paths): recovery of 300 known-positive SFT windows and false positives on 80 Gutenberg documents, for n = 6..32. Res |
| `longdoc_contam_ngram_study_excerpts.py` | Token n-gram length study, part 2: 92 word-boundary excerpts (60-300 words) of SFT windows re-tokenised alone -> share of their n-grams found in the s |
| `longdoc_contam_run.sh` | Incremental contamination scan of the benchmark pool on the rental box (CPU only, nice 10, 14 processes). Extra arguments are passed through |
| `longdoc_contam_scan.py` | Contamination / de-duplication scan of the candidate documents of the long-context translation benchmark (library: gemma_mt/eval/longdoc_contam.py). |
| `longdoc_contam_validate.py` | Validation of the long-document contamination scanner with positive / negative / partial / duplicate controls. |
| `longdoc_fetch_hard.py` | Fetchers of the HARD / DOMAIN split (medicine, law, maths-science) of the long-document benchmark.  They write pool files `hard_<source>__<class>.json |
| `longdoc_fetch_sources.py` | Fetch licence-clean English documents for the long-document translation benchmark (SOURCE POOL, 2026-10-20). |
| `longdoc_fp.py` |  |
| `longdoc_make_card.py` | Write README.md (dataset card) and DATASHEET.md of the benchmark repo from the selection outputs (D-44 addendum 2). |
| `longdoc_make_release.py` | Public records of the long-document benchmark (D-44 addendum). Thin wrapper: see gemma_mt/eval/longdoc_release.py (`--help`). |
| `longdoc_push_hub.py` | Push the staged benchmark repo (README.md, DATASHEET.md, core/, hard/) to a PRIVATE Hugging Face dataset repo and verify it (D-44 addendum 2). |
| `longdoc_score_reward.py` | Total RL reward of a system's document translations, with the real reward code (D-44 addendum). Thin wrapper: see gemma_mt/eval/longdoc_reward.py (`-- |
| `longdoc_smoke_cpu.py` | CPU smoke test of the whole long-document pipeline with a SMALL public model (D-44): stitched FLORES+ dev passages with human references are translate |
| `longdoc_stage_repo.py` | Stage the public benchmark repo (README.md, DATASHEET.md, core/, hard/) in a local directory for scripts/longdoc_push_hub.py (D-44 addendum 2). |
| `longdoc_translate.py` | Pseudo-reference runner (D-44): English source set -> NLLB-200 / MADLAD-400 sentence translations, resumable, with progress.log. Library: |

## Model growth

| Script | What |
|---|---|
| `export_trainer_state.py` | Export the weights of a trainer checkpoint (<run>/last/trainer_state.pt) to a plain Hugging Face directory (model.safetensors + config / tokenizer fil |
| `init_tag_rows.py` | Initialise the dedicated language-tag rows of a (grown) checkpoint: row of `<unused{i}>` := mean embedding of the sub-tokens of the old text tag. |
| `rename_tag_tokens.py` | Publishable tokenizer for the dedicated language-tag tokens (growth 1): rename the 125 `<unusedK>` slots of the Gemma-3 tokenizer to `<yor>`, `<hau>`  |
| `upcycle_check.py` | Proof that an upcycled checkpoint (scripts/upcycle_layers.py) is the base model at init, loads everywhere, and trains.  CPU only. |
| `upcycle_layers.py` | Depth-upcycle a Gemma-3 text checkpoint: insert new decoder layers that start as an EXACT identity (arm A2-up of the clean rerun). |
| `upcycle_s2_start.sh` | Builds the S2 START checkpoint of the clean rerun: google/gemma-3-270m-it + 3,000 Ge'ez tokens (ids 256001-259000 of the unused tail 256001-262143), e |
| `upcycle_s2_start_check.py` | Check of the S2 start checkpoint (google/gemma-3-270m-it + 3,000 Ge'ez tokens, scripts/upcycle_s2_start.sh). CPU only; exits non-zero on the first fai |

## Quality estimation

| Script | What |
|---|---|
| `qe_attack_rows.py` | D-61: rows (iso, cls, src, mt) for the QE check of the padded / stuffed attacks: held-out FLORES+ dev prompts of agent A5's policy samples; classes hu |
| `qe_attack_summary.py` | Summary of the QE check of the attacks: per class the mean score of each model and the share of attack rows scoring ABOVE the human reference of the s |
| `qe_choice_sample.py` | D-53 step 2: GRPO-like policy samples on FLORES+ DEV prompts. |
| `qe_choice_score.py` | Score a JSONL of (src, mt[, ref]) rows with one or more COMET models (D-53 QE-metric choice study). |
| `qe_choice_study.py` | D-53 QE-metric choice study: data builders and reports (the scoring itself is qe_choice_score.py, the sampling qe_choice_sample.py). |
| `qe_ensemble_gpu_check.py` | Real-GPU check of the QE ensemble reward (D-55): the OLD reward (AfriCOMET-QE alone) and the NEW one (ensemble) on the SAME samples. |
| `qe_vs_reference.py` | Companion of validate_consensus.py (D-43): add the AfriCOMET-QE score to the samples of a validation run and report the same within-prompt Spearman nu |

## Reports

| Script | What |
|---|---|
| `build_final_per_language.py` | Build the final-model per-language master table (D-49) from the benchmark runs of the SFT-final and RL-final models. |

## Reward anchors and calibration

| Script | What |
|---|---|
| `anchors_common.py` | Shared helpers of scripts/build_anchors_calibration.py and scripts/eval_anchors.py (D-56): the human reference sets, pseudo-documents and the syntheti |
| `anchors_v2_tables.py` | Per-language enable table of calibration/anchors_v2.json (D-61 report): entity weights, copy thresholds, stop-word / padding blocks, lexicon weights.  |
| `build_anchors_calibration.py` | Build calibration/anchors_v1.json: the thresholds of the constraint penalties of rl/anchors.py (D-56), fitted on HUMAN references. |
| `build_anchors_v2_calibration.py` | Build calibration/anchors_v2.json (D-61): the v1 anchors calibration + per-language blocks `entity2`, `copy`, `func`, `pad` fitted on HUMAN references |
| `build_anchors_v3_calibration.py` | Build calibration/anchors_v3.json (D-68): anchors_v2.json + domain-robust entries for the languages whose v2 'pivot_held' calibration is religious-tex |
| `build_calibration_v5.py` | Build calibration/shaping_flores_dev_v5.json = v4 + the added languages + the `lid` blocks + the `content` blocks (D-37, D-38). v4 is never modified. |
| `build_calibration_v6.py` | Build calibration/shaping_flores_dev_v6.json = v5 + the new consensus-only languages (owner 2026-11-01). v5 is never modified. |
| `calibrate_qe_ensemble.py` | Calibration of the QE ensemble (D-55): shaping_flores_dev_v7.json -> shaping_flores_dev_v8.json. |

## Reward hardening evidence

| Script | What |
|---|---|
| `hardening_evidence_01_assemble.py` | H1 step 1: assemble ONE table of stored sentence-level outputs: SFT-final, RL-final, final2 (step 401), final-run step 250 (FLORES+ only) and A2W alph |
| `hardening_evidence_02_indicators.py` | H1 step 2: run the final run's detectors (anchors v3 etc.) and GlotLID on every stored sentence-level output and on the human references. |
| `hardening_evidence_03_sentence_stats.py` | H1 step 3: sentence-level failure / hack indicators per system, benchmark and language group, and paired comparisons (paired bootstrap over sentences, |
| `hardening_evidence_03b_boilerplate.py` | H1 step 3b: cross-prompt boilerplate, the symptom of the consensus attractor (paper Section 7.6, anchors `boiler`): share of the word 5-grams of an ou |
| `hardening_evidence_04_longdoc.py` | H1 step 4: long-document indicators of the final run's reward (long-document reward v2, anchors v3 at document level) on STORED outputs. |
| `hardening_evidence_04b_longdoc_stats.py` | H1 step 4b: tables for the long-document indicators of step 4. Writes in $HE_OUT: |
| `hardening_evidence_05_judges.py` | H1 step 5: do the two judges of the ensemble (AfriCOMET-QE, SSA-COMET-MTL) disagree more about final2's outputs than about RL-final's? What is stored  |
| `hardening_evidence_06_training_logs.py` | H1 step 6: what the training logs say about the hardening terms (no model needed). Reads the step logs of the final run (box/other/r7_runs/final2/metr |
| `hardening_evidence_07_summary_figures.py` | H1 step 7: summary tables (verdict matrix, decomposition of RL-final -> final2 into the science stage and the hardened RL run, per-language table) and |
| `hardening_evidence_common.py` | H1 (2026-10-25): shared helpers of the "what did the reward hardening protect against?" analysis (scripts hardening_evidence_*.py). |
| `hardening_evidence_detectors.py` | H1: the detectors of the final run's reward applied to STORED outputs (no model, no GPU). Everything is the project's own code with the calibration th |

## SFT tooling

| Script | What |
|---|---|
| `sft_guardrails.py` | D-59: guardrail table of an SFT-stage sweep. Rule (lead, 2026-10-22): among the alphas, take the best sentence-level dev chrF++ (AfriScience-MT dev, 6 |
| `sft_lang_groups.py` | Per-language / group deltas of a candidate vs a baseline (weak suite chrF++, general suite chrF++, AfriScience dev sentence chrF++ and paper chrF++),  |
| `sft_stage_eval.py` | Evaluation harness of the short science-SFT stage (D-58): before / after tables, forgetting check, weight-interpolation sweep. |

## Vocabulary swap (Ge'ez and Fon)

| Script | What |
|---|---|
| `tokswap_commands_T1a_oldplan.sh` | NOTE (2026-10-26): this file documents the OTHER T1 session (T1a, old plan: re-used ids = leaf CJK tokens that are REMOVED from the vocabulary, N up t |
| `tokswap_eval.py` | tokswap_eval.py -- tokenizer-level tests of a swapped Gemma-3 tokenizer against the original (CPU, no model). |
| `tokswap_fetch_laptop.py` | tokswap step A (LAPTOP): download everything the box needs, with the READ token from the repo .env. |
| `tokswap_gguf_check.py` | tokswap_gguf_check.py -- read back the vocabulary of a GGUF produced by llama.cpp's converter from the swapped tokenizer.model and compare it with |
| `tokswap_hf_check.py` | tokswap_hf_check.py -- the swapped tokenizer loads through transformers AutoTokenizer (PreTrainedTokenizerFast), keeps the chat template and special-t |
| `tokswap_report_tables.py` | tokswap_report_tables.py -- turn the raw test outputs of the vocabulary-swap build into the markdown tables of reports/token_swap_build_2026-10-25.md. |
| `tokswap_run_all.sh` | tokswap_run_all.sh -- every command used to build the artefacts (on the box: /workspace/tokswap; CPU only).  Stages: A (tokenizers), B (checkpoints +  |
| `tokswap_sanity.py` | tokswap_sanity.py -- CPU sanity checks of a swapped (or control) 270M checkpoint against the original checkpoint. |
| `tokswap_sanity_fast.py` | tokswap_sanity.py -- CPU sanity checks of a swapped (or control) 270M checkpoint against the original checkpoint. |
| `tokswap_sanity_lane.sh` | usage: lane.sh <geez/fon>   (laptop CPU sanity lane) |
| `tokswap_spm_check.py` | tokswap_spm_check.py -- the edited SentencePiece tokenizer.model must give the same ids as the edited tokenizer.json. |
| `tokswap_swap_vocab.py` | swap_vocab.py -- Gemma-3 vocabulary swap (token-id reuse): free list, new Ge'ez / Fon BPE merges, swapped |
| `vocab_restrict_build_sets.py` | Vocabulary restriction study, step 1: build the allowed output-token set of every language and measure its coverage of the FLORES+ devtest REFERENCE t |
| `vocab_restrict_cover2.py` | Vocabulary restriction study: coverage of the FLORES+ devtest reference tokens by (static language set) and (static set + tokens of the SOURCE sentenc |
| `vocab_restrict_gen.py` | Vocabulary restriction study, steps 2 and 3b: greedy translation of the benchmark set with the output layer |
| `vocab_restrict_gen_train.py` | Vocabulary restriction study, step 1b: greedy generations of the FINAL model on English TRAIN-side prompts (RL prompt pool v3, FineWeb-edu English, ne |
| `vocab_restrict_keep.py` | Vocabulary restriction study, step 4: the token list of a per-language 'lite' model = allowed output set  U  tokens of English SOURCE text (train Engl |
| `vocab_restrict_llama.py` | Vocabulary restriction study, step 3c: llama.cpp with a PHYSICALLY pruned vocabulary. |
| `vocab_restrict_micro.py` | Vocabulary restriction study, step 3a: micro-benchmark of ONE decode step (batch B, KV length ~ L) in PyTorch CPU: transformer body vs output head, fu |
| `vocab_restrict_outcov.py` | Vocabulary restriction study: share of the tokens the UNRESTRICTED model itself writes (greedy, FLORES+ Q set) that lie inside the static set / the st |
| `vocab_restrict_score.py` | Vocabulary restriction study, step 2 scoring: per-language and macro chrF++ (sacrebleu word_order=2, NFC) of full / mask / slice hypotheses, paired bo |

## Weak-language data

| Script | What |
|---|---|
| `weak_select_data.py` | W1 (2026-10-26): build the DEV sample for the weak-language selection. Read-only from the Hub (token HF_TOKEN from the repo-root .env, never printed). |
| `weak_select_generate.py` | W1 (2026-10-26): greedy English -> X generation of the final-run Q8_0 GGUF on CPU for the weak-language dev sample (7,700 rows). |
| `weak_select_pairflags.py` | W1 (2026-10-26): dev contamination flags for the 100-sentence sample, from the archived scan of 2026-10-18 |
| `weak_select_report.py` | W1 (2026-10-26): render per_language.csv into the markdown ranking table (reports/data/weak_selection_2026-10-26/ranking_table.md). |
| `weak_select_score.py` | W1 (2026-10-26): score the dev generations and apply DECISION_RULE.txt (reports/data/weak_selection_2026-10-26/). Nothing here is tuned after generati |

## Other

| Script | What |
|---|---|
| `add_lexicon_calibration.py` | D-59: add the `lexicon` block (human low-quantile threshold of the GUARDED lexicon overlap) to an anchors calibration for the languages whose lexicon  |
| `attractor_study.py` | D-61: would a stuffed / padded output WIN a GRPO group in the consensus / blend languages, with and without the v2 penalties? |
| `compare_eval_results.py` | Compare the `results.json` of two runs of `scripts/eval_sft_benchmarks.sh` (e.g. the SFT model and an RL checkpoint) language by language. |
| `compare_runs.py` | Compare the loss curves / validation losses / gradient norms of several runs (metrics.jsonl) against a reference run. |
| `coverage_table.py` | D-61: per weak language (consensus / blend): which guards cover it. Markdown on stdout. |
| `dry_check_rl.py` | CPU-only dry check of an RL config before the run starts (D-37 .. D-40). Nothing is trained, no GPU, no QE model. |
| `e2_comet.py` | E2: AfriCOMET reference-based (masakhane/africomet-stl-1.1) per sentence for the no_pair subset of FLORES+ devtest, languages the COMET card lists (af |
| `e2_gen_chat.py` | E2: zero-shot en->xx translation of FLORES+ devtest with a chat model through vLLM (greedy). usage: e2_gen_chat.py NAME MODEL_DIR STYLE LANGS(comma is |
| `e2_model_comparison_aggregate.py` | E2 model comparison, step 3 (box, CPU): per-language and macro chrF++ / spBLEU / AfriCOMET with bootstrap CIs for every system scored by |
| `e2_refs.py` |  |
| `e2_run_ours_generation.sh` | E2: FLORES+ devtest generations of our four models with the same pipeline/decoding as the final table (greedy, vLLM, 768 new tokens) |
| `e2_run_queue.sh` | E2 sequential queue, GPU footprint <= 6 GB, nice 19 (coordinator rule 2026-10-23). SFT/RL predictions: read-only copies of e1 outputs (same pipeline,  |
| `e2_score.py` | E2: per-sentence chrF++ / spBLEU statistics for one system on FLORES+ devtest en->xx. |
| `e2_ssa.py` | E2: SSA-COMET-MTL (McGill-NLP/ssa-comet-mtl), reference-based (mean of the three views src/mt/ref as in the model card), per sentence on the no_pair s |
| `e2_units.py` | Rebuild per-row FLORES+ predictions from the generation units of a sft_benchmarks run (units = consecutive rows of the benchmark in loader order). usa |
| `entity2_report.py` | D-61 report numbers of the graded entity penalty: languages with a channel enabled (v1 strict vs v2 soft), the human mean penalty, and the penalty of  |
| `ge_ez_step0_common.py` | Shared helpers for the Ge'ez Step 0 scripts (2026-10-25). Read-only Hub access; the token is read inside this module and never printed. |
| `ge_ez_step0_cost.py` | Q4: compute-scaling estimate of the training cost of repeating (i) the reduced Ge'ez diagnostic and (ii) the full pipeline at Gemma-3-1B and 4B. |
| `ge_ez_step0_fertility.py` | Q1: tokenizer fertility of the Gemma-3 tokenizer vs NLLB's on FLORES+ devtest (human references). |
| `ge_ez_step0_fetch.py` | Download (read-only) the tokenizer files and FLORES+ dev/devtest text used by the Ge'ez Step 0 report. |
| `ge_ez_step0_fetch_train_text.py` | Fetch (read-only) public training-side text for the vocabulary-swap simulation: Wikipedia am/ti/fon (20231101) and FineTranslations amh/tir/fon (ODC-B |
| `ge_ez_step0_gemma_facts.py` | Q3: parameter split, disk sizes and CPU-compute ratio of Gemma-3 270M vs 1B (and 4B text part). Inputs: Hub metadata (gemma_hub_metadata.json, read-on |
| `ge_ez_step0_spm_check.py` | Check that the same N new tokens, written into the SentencePiece model (tokenizer.model, the file llama.cpp's converter reads for Gemma 3) as NORMAL p |
| `ge_ez_step0_vocab_census.py` | Q2a: census of the 262,144 Gemma-3 tokens by Unicode script, and how many could be freed without touching the 77 target languages. |
| `ge_ez_step0_vocab_swap_sim.py` | Q2b: simulate adding N new tokens (into freed vocabulary slots) for Ge'ez-script text (amh, tir) or Fon, and measure the fertility on HELD-OUT FLORES+ |
| `geez_check.py` | D-61: the Ge'ez-script languages (amh, tir) under the v2 detectors. Owner 2026-10-22: they must not be false-positived by the n-gram / stop-word / cop |
| `hardfail_study.py` | D-61: what does masking the hard-failed samples do to the policy gradient? A two-outcome softmax policy (fail with probability p, reward -2; otherwise |
| `interpolate_weights.py` | Weight-space interpolation between the model the science-SFT stage started from and the model it produced (WiSE-FT / model soup of two points). |
| `join_published_vs_ours.py` | Join OUR per-language scores with the PUBLISHED baselines (D-51) and write a comparison report. |
| `judge_j2_batch.py` | Gemini BATCH-API runner for the J2 judge runs (gemini-3.8-flash only), with a hard USD budget ledger. |
| `judge_j2_common.py` | Shared helpers for the J2 judge runs (QE validation, FLORES+ SFT vs RL, long documents). |
| `judge_j2_run1_analyze.py` | Run 1 analysis: Figure 9 (QE) Flash judge vs the human z-score, compared with the same judge on the reference-based prompt (pilot), |
| `judge_j2_run1_build.py` | Run 1 (QE-prompt validation): rebuild the 300 pilot segments with the Figure 9 (no reference) prompt. No API calls. |
| `judge_j2_run2_analyze.py` | Run 2 analysis: FLORES+ devtest, RL-final minus SFT-final, judge = gemini-3.8-flash (Figure 8, reference-based, human FLORES+ reference). |
| `judge_j2_run2_build.py` | Run 2 build (no API calls): FLORES+ devtest, SFT-final (step 27,638) vs RL-final (GRPO step 1,433), eight languages with pilot-validated demonstration |
| `judge_j2_run3_alignment_check.py` | Run 3 diagnostic (free, no API): is a low judge score of a whole-document unit a translation failure or a misalignment? |
| `judge_j2_run3_analyze.py` | Run 3 analysis: long documents judged on aligned units with the QE prompt (Figure 9), gemini-3.8-flash. |
| `judge_j2_run3_build.py` | Run 3 build (no API calls): long documents (LongCtx-MT core, real documents b1024 + b2048), judged by the QE prompt (Figure 9) on ALIGNED UNITS. |
| `judge_j2_summary.py` | J2 housekeeping: cost summary from the ledger and one file with the prompt templates and one example per prompt type.  python judge_j2_summary.py <dir |
| `judge_j2_test.py` | Unit tests for the Figure 9 (QE) prompt builder: python -I judge_j2_test.py  (or pytest). |
| `judge_pilot_analyze.py` | Analysis of the J1 judge pilot: judge score vs human z-score, per pair / pooled / language-demeaned, bootstrap CIs, paired difference. |
| `judge_pilot_batch.py` | Gemini BATCH-API runner for the J1 judge pilot, with a hard USD budget ledger. |
| `judge_pilot_build.py` | Build the J1 judge-pilot sample and prompts (no API calls). |
| `judge_pilot_common.py` | Shared helpers for the J1 judge pilot (Gemini-2.5-Pro vs Gemini-3.8-Flash against human DA, SSA-COMET protocol). |
| `lang_weight_evidence.py` | D-65 evidence: does the number of RL samples a language saw explain its SFT -> RL chrF++ gain? (observational; read-only on finished-run files) |
| `lang_weight_experiment_pairs.py` | D-65 experiment plan: matched language pairs for a within-run treated / control comparison, with a pre-registered balanced randomisation. |
| `lang_weight_experiment_power.py` | D-65 experiment plan arithmetic: what difference in chrF++ between a static-weight and an adaptive-weight language can a run actually detect? |
| `language_coverage.py` | D-59: how many prompts / sampled completions does every language get over a run? Simulates the real PromptSchedule of a yaml (CPU, ~10 s). |
| `launch_ab1.sh` | Launcher / supervisor of the AB1 ablation (RL from a WEAK vs a FULL SFT checkpoint, identical RL settings; paper section 8.4). Style and control files |
| `launch_ab2.sh` | Launcher / supervisor of the AB2 ablation (cheap RL-vs-SFT-start ablation + continued-SFT controls). Same control-file semantics as launch_ab1.sh / la |
| `launch_clean_rerun.sh` | Launcher / supervisor of the RL stages of the CLEAN RERUN (agent C1, 2026-10-28). Derived from launch_final2.sh (same OOM ladder, restart logic, pidfi |
| `launch_final.sh` | Launcher / supervisor of the FINAL science + general GRPO run (D-59). Same OOM ladder and restart logic as /workspace/sft_build/sup.sh, plus a pidfile |
| `launch_final2.sh` | Launcher / supervisor of the FINAL science + general GRPO run (D-59). Same OOM ladder and restart logic as /workspace/sft_build/sup.sh, plus a pidfile |
| `macro_summary.py` | Macro bootstrap summary of RL - SFT from the per-language replicate arrays of the box job macro_boot.py. |
| `measure_lid.py` | Measure GlotLID top-1 recall on HUMAN text of every RL language and decide the per-language guard (D-37). CPU only. |
| `merge_validation.py` | Merge the shards of validate_consensus.py runs of ONE language into a single summary file (D-43): python scripts/merge_validation.py --out val_efi.jso |
| `mix2_archive.py` | Mixture v2, component A: HIGH-CONFIDENCE rows of the RL best-translation archive as SFT windows. |
| `mix2_build.py` | Mixture v2 = mix_v1 (science + replay, UNCHANGED samples) + archive (A) + long-document distillation (B) + weak-language extras (C), re-packed with R4 |
| `mix2_common.py` | Shared helpers of the mixture-v2 builders (archive component A, long-document component B): the judges of the RL reward, outside the trainer. |
| `mix2_compare.py` | Compare mixture variants per language: tokens by component, share of the language's tokens that is NEW relative to mix_v1, human-reference share, self |
| `mix2_extras.py` | Mixture v2, variant B: choose the extra windows of R10 (human / real-text data by licence class) and R14 (NLLB teacher distillation) under per-languag |
| `mix2_extras2.py` | Variant B_v2: pick R10's extra windows per licence-class group under explicit per-language TOKEN BUDGETS (no window used twice, sources balanced insid |
| `mix2_gen_vllm.py` | Mixture v2, component B step 2: vLLM generation of the sentence-level requests (adapted from /workspace/jobs/a2/gen_vllm.py, same sampling settings). |
| `mix2_licence_table.py` | Per-language licence-class table of the model-output components (archive, longdoc, anchor) of a mixture and what a licence filter would remove. |
| `mix2_longdoc_score.py` | Mixture v2, component B step 3: decode the sentence-level generations, judge every candidate, pick one per sentence, assemble and score the documents. |
| `mix2_longdoc_select.py` | Mixture v2, component B step 4: choose the documents of the long-document distillation from the scored candidates (OUT/ld_scored.jsonl). |
| `mix2_mkcfg.py` | usage: mkcfg.py VARIANT DATA_DIR OUT_CFG |
| `mix2_nll.py` | How self-distilled is each component? Mean per-token NLL of the START model (final_1433) on the target tokens of a random sample of every source of mi |
| `mix2_run_job.sh` | usage: run_job.sh NAME JOBS UTIL |
| `mix2_select_longdoc.py` | Mixture v2, component B step 1: pick the SOURCE documents for the long-document distillation and write the sentence-level generation requests. |
| `mix2_smoke.py` |  |
| `mix2_valextra.py` | Build OUT/val_extra from the validation windows R10 packed with its extra data (<=100 samples per language and group), for the groups used in the mixt |
| `mk_phase2b.py` | completions per phase (owner 2026-10-19): b64 500k, b256 300k, b512 300k, b1024 100k, mixed 100k; group 16 (15 sampled + 1 greedy) |
| `newlang_audit_hub.py` | Hub / web audit of the new-language study (D-47): what text and parallel data exist for each candidate language. Read-only metadata queries, nothing d |
| `newlang_audit_table.py` | Assemble the per-language data table of the new-language study (D-47) from the raw audit files (scripts/newlang_audit_hub.py, newlang_lid_recall.py, t |
| `newlang_confidence_combined.py` | Combined validation of the reference-free confidence features over TWO checkpoints (D-47): the RL checkpoint (step 925 of phase 2c) and an early SFT c |
| `newlang_confidence_validate.py` | Validation of the reference-free confidence features against the human references (D-47). |
| `newlang_cpu_topup.py` | A MEASURED micro SFT top-up on CPU (D-47): how far do a few thousand human pairs of a language the RL model cannot write move it? |
| `newlang_lid_recall.py` | GlotLID recall of every reference set of the new-language study (D-47): the measurement that decides whether a GlotLID guard can be switched on for a  |
| `newlang_probe.py` | Zero-shot generation probe of the new-language study (D-47). CPU only, resumable, raw outputs (metrics are computed by newlang_probe_metrics.py). |
| `newlang_probe_metrics.py` | Metrics of the zero-shot probe (D-47). Reads the raw outputs of newlang_probe.py and writes ONE row per completion and ONE row per prompt group. |
| `newlang_probe_sets.py` | Build the English->X probe sets of the new-language study (D-47): up to N sentences per (language, reference set), seeded, English 5..40 words. |
| `newlang_qe.py` | AfriCOMET-QE of the probe outputs and of the human references (D-47), CPU. Run it with the COMET interpreter of the old box (transformers 4.x): |
| `newlang_registry.py` | Registry of the languages of the new-language viability study (D-47): where each language's human reference sets are, what the instruction calls it, |
| `newlang_report_tables.py` | Markdown / csv tables of the new-language report (D-47) from the metrics of newlang_probe_metrics.py and the validation of newlang_confidence_validate |
| `p3_rl_leakage_audit.py` | P3 leakage audit: how many of the ENGLISH test sentences of the benchmarks occur in the English prompts of the RL pools (CPU only). |
| `pairwise_fig_agreement.py` | Figure PW3 (E1): (A) agreement of the sentence-level winner between metrics, (B) what the reference-based metrics say about the sentences on which RL |
| `pairwise_fig_common.py` | Shared helpers of the three E1 pairwise-comparison figure scripts (gemma_mt/scripts/pairwise_fig_*.py). |
| `pairwise_fig_langs.py` | Figure PW2 (E1): per-language pairwise preference for RL-final over SFT-final (all benchmarks pooled per language, `no_pair` subset). |
| `pairwise_fig_wtl.py` | Figure PW1 (E1): win / tie / loss of the RL-final model over the SFT-final model, sentence level, per metric, benchmark and training group. |
| `pairwise_longdoc_prep.py` | E1 pairwise comparison, long-input check, step 0 (CPU, box): turn the R6 (A2) sentence-by-sentence document translations of the two models into the |
| `pairwise_make_tables.py` | E1 pairwise benchmark comparison: markdown tables of the report from the CSVs of reports/data/pairwise_benchmark_comparison/. |
| `pairwise_neural.py` | E1 pairwise benchmark comparison, step 2 (GPU, box): neural MT metrics per sentence for several systems on the same benchmark rows. |
| `pairwise_robust_noloop.py` | E1 robustness: does the RL preference survive when sentences where either model loops / does not finish are removed (no_pair subset)? |
| `pairwise_stats.py` | E1 pairwise benchmark comparison, step 3 (CPU, box): sentence-level paired statistics of system B over system A. |
| `pairwise_surface.py` | E1 pairwise benchmark comparison, step 1 (CPU, box): per-sentence surface metrics of one system from the eval pipeline's predictions.jsonl. |
| `policy_flags.py` | D-68: what do the v2 and v3 stop-word / padding detectors do to the MODEL's own outputs in the languages that failed in the live run? Uses the greedy  |
| `precompute_nllb_pseudorefs.py` | Precompute NLLB translations of the RL prompts for the reward pseudo-reference signal (R14). REWARD ONLY: output is CC-BY-NC, rows carry licence_class |
| `prepare_afriscience.py` | Fetch AfriScience-MT (dsfsi/afriscience_mt, Apache-2.0) and derive what the science RL run needs (DECISIONS.md D-52). |
| `prepare_initial_state.py` | D-59: build the STARTING trainer state of the final science+general GRPO run. |
| `probe_microbatch_memory.py` | GPU memory / time of ONE trainer micro-batch of the RL update at a given sequence length (D-45: is a 9.3k-token b2048 sequence safe?). |
| `refit_urh_bench.py` | D-68: re-run the anchors calibration of ONE language with a NEW native-verified human benchmark (jsonl rows {english, reference}; for Urhobo also acce |
| `replay_lang_weights.py` | Offline replay of the adaptive language-weight updater (D-65, rl/lang_weights.py) over a finished (or running: read-only) metrics.jsonl. |
| `replay_schedule_draws.py` | Replay the (stateless, deterministic) RL prompt schedules of the finished RL phases -> prompts and sampled completions per language (D-65 evidence). |
| `run_ab1_order.sh` | AB1 run order (RL-vs-SFT-data ablation). NOTHING starts until the file $AB/ORDER_GO exists (created by the owner / lead once the live run is wound dow |
| `run_ab2_order.sh` | AB2 run order (cheap RL-vs-SFT-start ablation + continued-SFT controls). The owner approved running it as soon as the final run ends: ONE command star |
| `run_sci_sft_stage.sh` | Driver of the short science-SFT stage (D-58) on the rental box. One phase per call; every phase is restartable. |
| `slim_checkpoint.py` | Strip the optimizer state from a trainer checkpoint (runs/<run>/last/trainer_state.pt): weights + step + RNG + stopper bookkeeping stay. |
| `summarize_attractor.py` | Markdown table from the JSON of scripts/attractor_study.py (D-61). |
| `summarize_eval_v2.py` | Markdown tables from the JSON of scripts/eval_anchors_v2.py (D-61 report). |
| `sup_science.sh` | Supervisor of the SCIENCE RL run (D-52). Same OOM ladder / restart logic as /workspace/sft_build/sup.sh, but: the run directory is runs/sci (not runs/ |
| `time_monitor_sets.py` | Time the monitor / selection dev sets of an RL run exactly as the trainer evaluates them (D-56): greedy decoding through the vLLM sampler, macro chrF+ |
| `train_sgd_lowmem.py` | train_sft with plain SGD (no optimizer state) instead of Muon / AdamW, for equivalence runs on a GPU that is shared and has |
| `urh_silver_check.py` | D-68: the Gemini-generated Urhobo silver set as a CHECK of the urh detectors (never as a source of thresholds, a stop-word profile or a loanword list) |
| `urh_silver_eval.py` | D-68: chrF++ of our models against the Gemini-generated Urhobo silver set (CPU, greedy). Reported ONLY as "agreement with Gemini-Urhobo": the referenc |
| `v3_data.py` | Human-style reference sets per language for the anchors v3 (D-68): religious / pivot-style text (the only text some languages have) and out-of-domain  |
| `v3_domain_study.py` | D-68: how large is the domain shift of a religious-text calibration, and does the domain-robust recipe (rl/anchors_calibrate_v3.py) remove it? |
| `validate_consensus.py` | Does the QE-free consensus signal (D-38) rank a model's own samples the way a human reference does? (D-43) |
| `validate_lexicons.py` | Validate the dictionary-overlap signal per language (D-57, agent B3): is `lexicon_overlap` a measure of translation quality? |
| `varlen_equivalence.py` | Numerical equivalence of the attention paths (explicit sdpa masks vs varlen vs flex_attention) and of the loss |
| `verify_b2048_prompts.py` | D-45 acceptance check of `rl_prompts_v3` against `rl_prompts_v2` with the REAL loader, hold-out and schedule of the trainer. CPU only (loads ~10 M pro |
| `vllm_smoke.py` | GPU smoke test of the vLLM rollout sampler (DECISIONS.md D-28). Needs a CUDA GPU, vLLM (in this interpreter for |
