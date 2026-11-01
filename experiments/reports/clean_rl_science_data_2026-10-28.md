# Clean RL prompt pools and clean science-stage data (agent S3, 2026-10-28)

Nothing was trained; no GPU and no paid API were used. VERIFIED = measured here or read from the cited file. REPORTED = taken from an earlier report. ESTIMATED = an estimate.

Outputs: Hub repo (private) `<HF_ORG_LEGACY>/sabiyarn-clean-rl-prompts-2026-10-28` (1,126 files, 6.9 GB); tables in `reports/data/clean_rl_science_2026-10-28/`; scripts `data_pipeline/sft_clean/s3_*.py` (11 files); archive copies in `research_archive_2026-10-23/local/`. Nothing is committed to git. The box copy is deleted (only `code/` and `logs/` remain in `/workspace/clean_rl`).

Index: `protected-2026-10-28.2` was current when S3 started; D2 published `.3` meanwhile. The base index file is unchanged in `.3` (sha256 `f094fd2b…`, manifest diff VERIFIED), only the Yoruba exemption file differs. S3 loaded the index strictly without any exemption, so its output is valid for both. The `prov_index` column says `.2`.

`prompts_path` must be ONE pool directory (the loader walks recursively; never point it at the repo root). The repo's `load_rl_prompts` was run on `clean_v3/` and `clean_sci_v2/` and works.

## 1. Per-bin prompt counts after filtering (b2048 dropped)

Rows are prompts over 77 language files. `rl_prompts_v3` has only 1,406,812 distinct English windows, because the same window is a prompt of many languages.

| pool | b64 | b256 | b512 | b1024 | total | removed |
|---|---:|---:|---:|---:|---:|---|
| v3, before | 1,485,000 | 1,485,000 | 1,485,000 | 1,485,000 | 5,940,000 | |
| **v3, after** | **1,480,263** | **1,470,829** | **1,452,679** | **1,427,382** | **5,831,153** | 108,847 = 1.83% |
| sci_v2, before | 23,964 | 26,969 | 28,013 | 32,132 | 111,078 | |
| **sci_v2, after** | **23,868** | **26,746** | **27,622** | **31,415** | **109,651** | 1,427 = 1.28% |

* Per-bin removal rate in v3 is 0.32 / 0.95 / 2.18 / 3.88%. Long web windows lose the most.
* sci_v2 had 26,094 b2048 rows. v3's `zz_b2048/` (77 files) was not downloaded or scanned.
* Distinct windows: v3 1,406,812 -> 1,379,763 (1.92% removed); sci_v2 94,648 -> 93,356; 1,419,232 distinct kept windows in total.
* Per-language minimum after filtering: v3 4,972 / 4,939 / 4,872 / 4,773; sci_v2 198 / 234 / 262 / 326.
* The science part of sci_v2 is small per language: about 80 / 120 / 149 / 218 prompts at b64 / b256 / b512 / b1024.
* Lean-schedule needs (weak languages: b64 894, b256 157, b512 110, b1024 55, final run 138) are covered at least 5 times over.

## 2. Setup and rules
* Box 2, `taskset -c 24,26`, `nice -n 19`, `ionice -c3`. Memory at most 8.2 GB counting shared pages; disk peak about 10 GB; downloads one file at a time. `/workspace/tr`, `/workspace/chess` and others' tmux were not touched.
* One incident: a `kill $(pgrep -f run_prompts.sh)` sent to stop S3's own launcher matched S3's own ssh shell and ended that shell. Nobody else's process was hit and the python job continued.
* The Hub write token was read inside S3's own processes only.
* D2 changed its own venv (numpy 1.26) meanwhile, so S3 built its own venv.

## 3. The old science stage (VERIFIED from reports, configs and tables)

The model that fed the final run is mixture A2W, interpolated with the RL model: theta = 0.3 * theta_RL + 0.7 * theta_SFT-stage.
* Size: 20,880,766 tokens, 84,199 samples, 2,550 packs of 8,192 tokens. Schedule: 319 steps of 65,536 tokens, peak lr 3e-6.

| component | tokens | share | rows |
|---|---:|---:|---|
| AfriScience-MT train (6 languages, 2 passes) | 6,619,550 | 31.7% | 29,451 windows |
| replay of old SFT windows (77 languages, weak x0.5) | 8,985,911 | 43.0% | 47,141 |
| RL best-translation archive | 1,043,557 | 5.0% | 2,158 rows, 65 languages |
| long-document distillation | 2,825,610 | 13.5% | 1,715 documents, 71 languages |
| self-distilled anchors | 1,406,138 | 6.7% | 3,734 windows, 47 languages |
| R10 extras / NLLB teacher / TranslatePsy | 0 | 0 | not in A2W |

* The first science mixture (mix_v1) was 19,193,643 tokens: science 34.5% (6,631,094) and replay 65.5% (12,562,549), i.e. 35/65 by design.
* A2W science tokens per language: yor 1,337,976; hau 996,148; amh 1,043,738; lug 1,083,593; zul 1,029,350; nso 1,128,745.

**"70/30".** The only 70/30 in the record is the RL final run's prompt draw: `domain_weights {science 0.7, general 0.3}` in `configs/rl_science_final*.yaml`; paper 4.1 says the same. It is applied per prompt: domain, then bin, then language, then prompt. Science bin weights were b64 .05, b256 .10, b512 .15, b1024 .35, b2048 .35; general was uniform. It is independent of pool-file sizes (the weak-language sci_v2 files are about 70% general by row count). The final run was 401 steps x 42 prompts, group 16, lr 1e-6. If the SFT stage was meant, the ratios are the table above.

**Component verdicts.** Basis (REPORTED, audit): the old SFT text had FLORES+ devtest English in 95.3%; the science-stage hold-out did not list FLORES+ devtest, NTREX, MAFAND test, SmolSent, SALT or GlotStoryBook.

| component | verdict | reason / GPU cost |
|---|---|---|
| AfriScience-MT train | **REUSABLE** after the S3 filter (34,752 -> 34,007 pairs) | Only needs re-windowing on CPU with the protected index as hold-out. No GPU. |
| replay of old SFT windows | **EXCLUDED as built; REBUILD** from D2's clean pairs | CPU packing. Optional QE gating of about 47k windows: ESTIMATED < 0.3 GPU-h. |
| RL archive ("component A") | **EXCLUDED as built; REBUILD** | By-product of the new RL run's `archive.sqlite`. Re-judging about 6k rows: ESTIMATED < 0.1 GPU-h. Needs the new RL run first. |
| long-document distillation | **EXCLUDED as built; REBUILD** with the new model on the clean pools | 1.5 GPU-h central [0.8-3.0] (ESTIMATED, lean-schedule stage 3a). The old job took about 2.5 GPU-h on a shared card in 3 restarts (REPORTED). |
| anchors | **EXCLUDED as built; REBUILD** | Same pipeline on short windows; about 0.3 GPU-h, inside the 1.5 above. |
| R10 extras | **REUSABLE via D2's specialist set** (not in A2W) | Already filtered and licence-tagged; NC rows stay tagged. |
| NLLB-teacher (R14 / C3, 4.47 M tokens), TranslatePsy, NLLB pseudo-refs | **EXCLUDED** | NC / NLLB; owner rule. |

* The archive, long-doc and anchors are model outputs of the SFT-final / RL-final lineage on the old pools. The old pools have 2.2% (b512) and 3.9% (b1024) protected windows (measured here). "Contaminated model" is read as lineage. A filter-and-keep of the old rows is the owner's call; its yield could not be tested because the old row texts are not in the backup MANIFEST.
* Two calibration objects need refitting (not done): `science_anchor_v1.json` (refit on the clean train set) and `longdoc_v2.json` (fitted on AfriLongDoc-core NLLB references, now forbidden).

## 4. AfriScience-MT and the clean train set
* Dataset `dsfsi/afriscience_mt`, commit `5e7b8238d562…`. Licence Apache-2.0 on the Hub card (VERIFIED); the licence of the 230 underlying papers is not stated.
* 230 papers, 7,605 English sentences, 6 languages (amh hau lug nso yor zul), 11 domains.

| split | pairs | papers | English tokens | target tokens |
|---|---:|---:|---:|---|
| train | 5,792 | 177 | 153,933 | amh 322,737; hau 297,597; lug 342,932; nso 364,665; yor 573,261; zul 317,176 |
| dev | 843 | 25 | 21,937 | 41.5k-79.8k |
| test | 970 | 28 | 26,239 | 50.4k-101.5k |

Dev (5,058 pairs) and test (5,820 pairs) are protected; only counts and paper ids are listed (`afriscience_protected_papers.csv`).

Filter: both sides, strict, 35 split-level sets. Removed 745 of 34,752 pairs (2.14%); none solely because of the FLORES+ dev train half. Most removals are target-side matches with AfriScience dev/test themselves (short repeated phrases).

| iso | pairs after (removed) | English tokens | target tokens | median paper tokens |
|---|---|---:|---:|---:|
| amh | 5,759 (33) | 152,936 | 320,641 | 2,736 |
| hau | 5,634 (158) | 149,238 | 288,230 | 2,489 |
| lug | 5,741 (51) | 152,397 | 339,516 | 2,809 |
| nso | 5,481 (311) | 145,242 | 343,528 | 2,716 |
| yor | 5,631 (161) | 149,296 | 553,595 | 3,986 |
| zul | 5,761 (31) | 152,981 | 315,355 | 2,661 |

* All six languages keep 177 papers. One pass is 902,090 English plus 2,160,865 target tokens. Papers over 4,096 tokens: yor 81, other languages <= 6. None exceeds 8,192.
* Multi-parallel caveat: 486 English sentences have different outcomes across languages, 30 are removed in all six. The pair-level result was kept; the column `contiguous_with_previous_kept` and the removal log let a window builder drop gaps or drop those sentences in all languages.
* The alignment-shift issue (sentence-level alignments sometimes shifted by a clause) is unchanged.

## 5. Clean RL prompt pools
**Inputs.** v3: `sft_build/gemma/rl_prompts_v3/<iso>.parquet` (77 files); sci_v2: `jobs/r7/assets2/rl_prompts_sci_v2/<iso>.parquet` (77 files). Both from `sabiyarn-box-final-backup`; the sha256 of all 154 downloaded files (3.59 GB) match the backup MANIFEST. v3 is fineweb-edu and Wikipedia only; sci_v2 adds PMC, arXiv, LibreTexts, the AfriScience train English side, FineTranslations English, Tatoeba, OLDI seed, SALT text and Wikimedia.

**Method.** Unchanged `decon.py`: R1 exact sentence, R2 contained sentence, R3 any shared word 8-gram, both normal forms; English side against every protected set plus the held-out FLORES+ dev half; the dev train-half exemption was NOT attached. Distinct texts are checked once across a pool. b2048 rows are dropped without scanning. A prompt with any hit is removed, never repaired. Flag-only extra index: SSA-MTE train, 286,485 sentence hashes. Provenance columns: `prov_pool`, `prov_file`, `prov_row`, `prov_en_xxh64`, `prov_index`, `flag_ssa_mte_train`.

**Removed per protected set** (rows, any rule; full table in the CSV). v3:

| set | rows | % of scanned |
|---|---:|---:|
| afrilongdoc | 74,594 | 1.26 |
| smol | 54,983 | 0.93 |
| afridocmt:test | 17,690 | 0.30 |
| mafand:test | 8,883 | 0.15 |
| mafand:dev | 8,256 | 0.14 |
| tico19:test | 5,807 | 0.10 |
| afridocmt:dev | 3,699 | |
| glot_story_book | 3,094 | |
| tico19:dev | 2,194 | |
| wmt24pp | 1,413 | |
| ibom_efi:test | 1,265 | |
| flores_plus:dev | 778 | |
| flores_plus:devtest | 451 | |
| ntrex128 | 327 | |
| afriscience test / dev | 198 / 127 | |
| ssa_mte:test | 190 | |

sci_v2: afrilongdoc 812, smol 643, afridocmt:test 237, mafand:test 140, mafand:dev 134. Rule split for afrilongdoc in v3: R1 34,641, R2 1,502, R3 41,536 (most removals come from the 8-gram rule on web-style English). FLORES+ devtest: 20 rows exact (the 2026-10-23 audit found 7 of 1,012 devtest sentences, REPORTED). Removed only because of the FLORES+ dev train half: 513 rows in v3, 7 in sci_v2.

**By source domain.**

| source | before | after | removed |
|---|---:|---:|---:|
| v3 fineweb-edu | 4,216,089 | 4,116,968 | 2.35% |
| v3 Wikipedia | 1,723,911 | 1,714,185 | 0.56% |
| sci_v2 PMC | 13,942 | 13,870 | 0.52% |
| sci_v2 LibreTexts | 10,381 | 10,257 | 1.19% |
| sci_v2 arXiv papers | 5,205 | 5,195 | 0.19% |
| sci_v2 AfriScience train English | 9,807 | 9,709 | 1.0% |
| sci_v2 FineTranslations English | 8,761 | 8,590 | 1.95% |
| sci_v2 Tatoeba | 506 | 471 | 6.9% |

**Overlap with AfriScience dev/test, AfriLongDoc, SSA-MTE, AfriMTE.**

| set | removed (v3 + sci_v2) | after filtering |
|---|---:|---|
| AfriScience dev | 128 | 0 |
| AfriScience test | 198 | 0 |
| AfriLongDoc | 75,406 | 0 |
| AfriMTE-WMT2024 | 88 | 0 |
| SSA-MTE dev / test | 12 / 227 | 0 |
| SSA-MTE **train** | not removed (flag only) | **173 kept rows flagged** (v3 166 + sci_v2 7); 101 further flagged rows were removed for other sets |

AfriCOMET's own training data (AfriMTE annotations) is not on the Hub, so overlap with it could not be measured.

**Verification.**
* Independent pure-Python check (plain string sets, no hashing, no decon code) against FLORES+ eng dev and devtest, NTREX-128 English, AfriScience dev/test sentences plus whole papers, AfriLongDoc core and hard: **0 hits in all 1,419,232 distinct kept windows**. The other 14 protected sets are covered by the index only.
* The pipeline's own index re-run (a new Checker) on 142,484 kept windows: 0 hits.
* Planted controls: 300/300 caught by both checkers.
* First-run incident: the verifier reported 94 hits that were its own artefact (it split reference sentences at ". " and indexed fragments like "According to the U.S." and "The U.N."). After limiting splitting to document sets, the same windows give 0. The first result file is kept as `s3_verify_result_run1_verifier_fragment_artifacts.json`.

**Licence notes (per-row licence column).** fineweb-edu odc-by; Wikipedia share_alike; PMC / arXiv cc-by / cc0; AfriScience train apache-2.0. **LibreTexts conflict:** rows say `cc-by`, the backup README says CC BY-NC-SA. Unresolved; it is 10,257 sci_v2 prompts and can be switched off by `source_dataset`.

## 6. Open items for the owner
1. AfriScience-MT licence of the underlying papers: the card says Apache-2.0 but is silent on the 230 source papers.
2. LibreTexts licence conflict.
3. Whether a filter-and-keep of the old archive / long-doc / anchor rows is acceptable. S3 recommends the rebuild (needs the new model, about 1.5 GPU-h).
4. b2048 is gone, so `domain_bin_weights`, `prompts_per_step_by_bin` and `vllm_max_model_len` of the final-run yaml need new values. The science pool has only 213-220 b1024 prompts per language.
5. The cap is source-side (Amendment 2), so a b1024 English window can give far more than 1,024 target tokens; the decoding limit stays 768 for eval, check generation limits for RL.
6. Multi-parallel handling for AfriScience windows: pair-level (default) or drop English sentences removed in any language.
7. If the base index file ever changes, rerun `s3_*prompts.py` (resumable, about 35 min on 2 cores).

## 7. Not done
No rebuild of any science component (needs the new SFT/RL model); no refit of `science_anchor_v1.json` or `longdoc_v2.json`; no check against the old components' row texts; no measurement against AfriCOMET's training data; no semantic or paraphrase check; no Gemma-3 pretraining exposure audit.
