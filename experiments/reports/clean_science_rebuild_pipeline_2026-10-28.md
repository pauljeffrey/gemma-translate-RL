# Clean science-stage rebuild pipeline and AfriLongDoc stage-check preparation (agent C2, 2026-10-28)

Labels: VERIFIED = measured or read; ESTIMATED = arithmetic or assumption; SMOKE = plumbing proof only, not a result. Nothing was run on a GPU, nothing committed, no Hub token printed. One Hub push to the private prompts repo, folder `science_stage/`.

## 1. What exists
All files in `data_pipeline/sft_clean/` (untracked), mirrored in Hub `<HF_ORG_LEGACY>/sabiyarn-clean-rl-prompts-2026-10-28/science_stage/` (code/, clean AfriScience windows, smoke and dry-run reports, stagecheck/).

| Part | File | Notes |
|---|---|---|
| shared helpers | s4_common.py | strict protected-index check (S3's Checker and decon.py, no exemption); licence class, LibreTexts and AfriScience licence notes; refuses NLLB/TranslatePsy/teacher sources; UniMax budgets |
| fetch / Hub push | s4_fetch.py | pulls pools, AfriScience train, the index; pushes to science_stage/ |
| (a) science windows | s4_afs_windows.py | A2W recipe on S3's filtered pairs; `--multi-parallel pair` (default) or `any`; `--gap-policy join` (default) or `split`; window-level index check |
| (b) replay | s4_replay.py | `--pack-dir` (D2 gated packs), optional `--windows-dir` for source and licence |
| (c) RL archive | s4_archive.py | reads archive.sqlite files (one per stage A, B1, B2, B3, mixed; merged, highest reward per source) |
| (d)+(e) long docs / anchors | s4_longdoc.py | subcommands select, gen, score, pick; delegates to the old unmodified mix2_longdoc_score.py, mix2_longdoc_select.py, mix2_gen_vllm.py, mix2_common.py |
| (f) mixer + packer | s4_mix.py | per-row provenance and licence; `--ratio-mode report|shrink` (shrink gives exact A2W shares 0.319/0.433/0.049/0.136/0.063) |
| final check | s4_decon_final.py | decodes packed training tokens and runs the strict check |
| (g) driver | s4_science_driver.sh | exact commands in order |
| smoke / tests | s4_smoke.sh, s4_smoke_archive.py, s4_make_sample.py, s4_tests.py | 10 tests passed |
| Job 2 | s4_afrilongdoc_stagecheck.py, s4_stagecheck_driver.sh | |

Replay: hold-out is the strict index on English and target; A2W recipe (UniMax levelling, science languages x2, weak languages halved). Anchors: 47 weak/Ge'ez languages, only hard failure filters, as A2W. LibreTexts rows are INCLUDED and tagged by source_dataset, licence label and `licence_note` (cc-by vs CC BY-NC-SA unresolved); `--exclude-source libretexts` removes them. The mixer writes train/provenance.parquet and sft_science_clean.yaml with the A2W hyper-parameters.

## 2. Commands in order (GPU session)
Environment: `NEW_RL_MODEL=<clean RL final HF dir>`, `PACK_DIR=<gated packs>`, `CAL`, `ANCHORS`, `SCORER_JSON`, `GLOTLID`, `HF_TOKEN`, `PY`, `PY_VLLM`. `ARCHIVES` defaults to /workspace/clean_rerun/runs/{A,B1,B2,B3,mixed}/archive/archive.sqlite.
```
bash data_pipeline/sft_clean/s4_science_driver.sh        # all steps; or name them:
  fetch afs replay archive longdoc anchors mix baseline train sweep report
```
fetch (pools, AfriScience train, index) -> afs (`--dev-from-hub`) -> replay (`--pack-dir $PACK_DIR`) -> archive (`--archive A=... B1=... B2=... B3=... mixed=...`) -> longdoc (select, gen --engine vllm, score, pick --kind longdoc) -> anchors (same four, --kind anchor) -> mix (s4_mix.py then s4_decon_final.py, must print HITS_samples 0) -> baseline, sweep, report (sft_stage_eval.py run/sweep/compare; alpha 0.3/0.5/0.7/1.0, A2W choice is 0.3 RL + 0.7 SFT) -> train (`python -m gemma_mt.train_sft --config $MIX/sft_science_clean.yaml`).

$3 check: `W=/workspace/stagecheck HF_TOKEN=... PY_VLLM=... bash data_pipeline/sft_clean/s4_stagecheck_driver.sh` = `s4_afrilongdoc_stagecheck.py fetch`, `prep`, `gen --engine vllm --delete-after`, `score`, `report --boot 400 --min-chrf 1.0 --min-fin 0.05`.

## 3. GPU time and cost (ESTIMATED, RTX PRO 6000, USD 1.19/h)
Basis: per-document counts measured on the real clean pools (sentence units / source tokens per document: b64 1.7/34, b256 4.4/128, b512 10.8/293, b1024 22.8/647).

| item | bottom-up | notes |
|---|---|---|
| (c)-(e) total | 1.0 h [0.7-2.0] | M1 plans 1.5 h [0.8-3.0]; budget 1.5 h |
| generation | 0.46 h at 25k tok/s [0.29 at 40k, 0.96 at 12k] | 14,428 candidate documents, ~186k sentence units, ~5.2M source tokens, x4 candidates x 2.0 = ~41.6M generated tokens |
| QE | 0.38 h at 1,100 pairs/s per model [0.28 at 1,500; 0.86 at 480] | <= 744k pairs x 2 models |
| archive re-judging, start-ups | 0.02 h + 0.1 h | |
| baseline eval | 0.15 h | |
| train, 319 steps | 0.35 h | |
| sweep | 0.55 h | |
| whole stage | 2.05 h = USD 2.4 | with M1's 1.5 h for (c)-(e): 2.55 h = USD 3.0 |

Long-document candidates are raised to 70 (b512) / 60 (b1024) per language because b2048 was 40% of the A2W long-document tokens; at the old 22/16 a shortfall (~0.6x) is expected. The mixer reports shortfalls.

$3 stage check (counts VERIFIED, time ESTIMATED): about 0.9 h [0.7-2.0] = USD 1.1 [0.8-2.4] (the E.4 guess was 2-2.5 h / USD 3). 2,560 items = 342 documents x 8 languages (687/719/676/478 items for b256/b512/b1024/b2048; invalid NLLB references skipped), 2.09M source tokens, ~4.2-5.7M generated tokens per checkpoint; ~4-9 min per checkpoint, 7 checkpoints ~0.5 h; downloads 15 s each.

## 4. Old checkpoint branches (VERIFIED, read-only Hub API, 2026-10-28; each has a 1,072,419,256-byte model.safetensors)
| name | repo @ revision | step | commit |
|---|---|---|---|
| sft_final | <HF_ORG>/gemma3-270m-translate @ step-27638 | 27,638 | 48ccde53 |
| rl2b_b64 | <HF_ORG>/gemma-3-270m-translate-rl2b @ stage-b64 | 488 | 1b0550ef |
| rl2c_b256 | <HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c @ stage-b256 | 391 | 02f8ab5d |
| rl2c_b512 | same repo @ stage-b512 | 837 | 7e372fb4 |
| rl2c_b1024 | same repo @ stage-b1024 | 986 | 69488f99 |
| rl2c_b2048 | same repo @ stage-b2048 | 1,284 | b16b24a4 |
| rl2c_mixed | same repo @ stage-mixed | 1,433 | 6b9535ba |

`latest` is also step 1,433 (5e5a56da). rl2c `main` holds NO weights. The stage-b64 branch is in the <HF_ORG> rl2b repo, not rl2c. Details: science_stage/stagecheck/old_checkpoint_branches_verified_2026-10-28.json.

Evaluation: whole-document greedy, languages hau yor swh amh zul wol kea sag, AfriLongDoc core b256-b2048. Metrics: finished rate, loop rate, chrF++ vs NLLB pseudo-references (old references are acceptable for evaluating old models), macro over languages, paired bootstrap over documents, stage deltas. Verdict rule fixed before the run and printed in the report: a stage "mattered" if its own-bin chrF++ delta >= 1.0 or finished delta >= 5 pp, with CI > 0. The lead should confirm this rule before the GPU run.

## 5. Smoke results (SMOKE)
`s4_smoke.sh`: whole chain on CPU, 2 cores, 468 s, 0 tracebacks; base model google/gemma-3-270m-it. Real: 8 AfriScience papers; a 38k-pair sample of the clean SFT pairs -> windows -> packs (ungated) -> replay; real clean pools for yor/hau/amh/kea. Not real: long-document and anchor generations came from a FAKE engine (`score` refuses them without `--allow-fake`); the QE scorer was fake; the calibration files were the OLD 2026-10-19 ones (clean calibration dir was empty). Components: science 271,997 tokens, replay 376,324, archive 33,362 (36 rows), long-document 9,537 (6 docs), anchors 5,133 (17 docs), 2,938 samples in 86 packs. LibreTexts documents were included and tagged (unresolved-licence tokens: anchor 220, archive 1,784); `--exclude-source libretexts` removed them in a separate run.

Real-data results (not smoke): AfriScience windows 28,933 windows, 6,451,304 tokens (A2W 6,619,550 / 29,451 windows, -2.5%); the window-level index check dropped 3 windows (8-grams spanning a sentence join). Per language (M tokens, A2W in brackets): amh 1.038 (1.044), hau 0.963 (0.996), lug 1.073 (1.084), nso 1.063 (1.129), yor 1.293 (1.338), zul 1.022 (1.029). 165 duplicate windows dropped in the mixer. Replay dry run at scale (490k-pair sample, ungated): 44,797 windows, 9.72M tokens (overshoot from the 600-example minimum per language on tiny sample languages; mixer trims to 8.89M); join to source and licence 61,929/61,929; re-encode check 0/300 mismatches; the strict check removed 1,800 of ~63.7k candidate windows (2.8%), 1,603 of them only dev halves (FLORES+ dev train half, SIB-200 dev); the mixer then dropped 815 NC-licensed windows (1.8%).

## 6. Decontamination (VERIFIED)
Index protected-2026-10-28.3, strict, no exemption, English and target of decoded packed tokens. Dry run at scale (28,768 real AfriScience windows + 43,982 replay + smoke-size archive/long-doc/anchors): 69,758 samples, 0 hits; controls 300/300 FLORES+ dev train-half sentences flagged. Smoke mixture 2,938 samples, 0 hits, controls 300/300. The real mixture cannot be built until the new RL model exists; `s4_decon_final.py` runs automatically in the driver. val_sci (AfriScience dev, a monitor) is protected by design, excluded from the check, and was not pushed.

## 7. Open items
1. Judges need the REFIT calibration (`--calibration`, `--anchors`, scorer JSON with AfriCOMET-QE + SSA-COMET-MTL); the real QE ensemble and vLLM engine were not run on CPU.
2. Weak lists: replay halving uses the A2W 45 weak-signal languages (anchors: A2W 47); the owner should confirm or replace (`--weak-langs`).
3. Strict rule: FLORES+ dev train-half windows D2 exempted for the 20 specialist languages are dropped here (2.5% of replay windows in the dry run).
4. Replay NC-licensed rows are dropped by default (`--exclude-classes nc,nc_teacher`); rows without a windows side table get licence class `unknown` (`WINDOWS_DIR` defaults to $PACK_DIR/../windows_sft).
5. The tokenizer of the packs must equal NEW_RL_MODEL's (the mixer refuses a mismatch); with the S2 extended tokenizer use the S2 packs.
6. Guardrail sets for baseline/sweep (`AFS_CORPUS`, `GENERAL_DEV`, `EVAL_POOL`) must be the clean-rerun sets; the old ones were partly contaminated.
7. LibreTexts (cc-by vs CC BY-NC-SA) and the AfriScience paper licence stay unresolved; tagged per row.
8. Replay overshoot and the A2W counts above come from a sample; real counts need the gated packs.
9. Compatibility with other agents' newer RL tree is untested on GPU.
10. Box 2: files under /workspace/s4 (~1 GB; heavy intermediates, the 2 downloaded old checkpoints, and the AfriLongDoc/NLLB reference copies were deleted).
