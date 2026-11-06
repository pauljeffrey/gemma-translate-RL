# Matched-compute continued-SFT control (E-03): preparation, CPU only (2026-11-04)

Question the control answers: is RL a better use of 4.4 to 8.7 A100-hours than more SFT? Nothing was run on a GPU, nothing was run on the A100 box (one read-only `df`/`ls` over ssh), nothing was pushed to the Hub or written to the bucket. The owner authorised TranslatePsy data for this control only (2026-11-04); the control checkpoints are research artefacts and are never released (section 1.3). Files: `reports/data/sft_control_prep_2026-11-04/` (scripts, configs, tables), downloads in `research_archive_2026-11-03/sft_control_2026-11-04_dl/` (about 1.0 GB, section 9). Ledger rows, the DISCLOSURES text and the EXPERIMENTS_LEFT text are in `reports/data/sft_control_prep_2026-11-04/ledger_snippet.md`.

## 1. The TranslatePsy data

### 1.1 What it is (dataset card `qvac/TranslatePsy-AfriSLM-Synthetic-Mix`, read 2026-11-04 from the Hub with the read token)

| item | value |
|---|---|
| repo | `qvac/TranslatePsy-AfriSLM-Synthetic-Mix` (public, not gated, last modified 2026-08-26); accompanies arXiv:2608.18655, Gritta, Lambert, Back, Nazir (Tether AI Research / QVAC), EMNLP 2026 |
| licence on the card | `cc-by-nc-4.0` ("Licensing Information: licensed by Tether Data under CC-BY-NC 4.0 ... generated using MADLAD-400 (CC-BY-4.0) and the NLLB-200-3.3B model (CC-BY-NC 4.0)") |
| origin | MADLAD-400 monolingual sentences, language-ID filtered (AfroLID for African, fastText for English), deduplicated, "removed content similar to the evaluation sets", **translated by `facebook/nllb-200-3.3B`** (CTranslate2 int8_float16, beam 3), scored in both directions with AfriCOMET, SSA-COMET and MetricX, robust z-score filter (threshold z >= 0.68 per ATTRIBUTIONS.md), bidirectional expansion. The targets are NLLB-3.3B output; the quality filters are in the reward family of our RL (AfriCOMET, SSA-COMET) |
| size | 215,653,192 rows, 364 arrow shards, about 182 GB (sum of file sizes from the Hub listing), about 32.37 B training tokens (card). Columns `source, target, messages, source_lang, target_lang, source_pair` |
| languages | English + 19: afr amh hau ibo kin lin lug mlg nya orm sna som sot swh tsn wol xho yor zul (all 19 are in our 125-language registry, all 19 have FLORES+ devtest rows; 16 have NTREX, 12 MAFAND, 3 SALT) |
| directions | each shard holds ONE direction of ONE pair (`tp_shard_map.json`, from the first-research scan R10: 197 en->xx shards, 167 xx->en); the corpus is mirrored, so xx->en rows add no new sentences for English->X (R10, `reports/weak_language_data_2026-10-22.md` section 3.3). en->xx shards per language: afr 4, amh 10, hau 14, ibo 11, kin 7, lin 9, lug 9, mlg 3, nya 6, orm 8, sna 5, som 8, sot 9, swh 17, tsn 6, wol 22, xho 9, yor 23, zul 17 |
| decontamination by its authors | against the evaluation sets of its own paper (Flores-200, BOUQuET, Smol; ATTRIBUTIONS.md). NOT against MAFAND, NTREX, SALT, GlotStoryBook, IbomMT, AfriScience-MT: our scan (section 5) finds rows of those |

### 1.2 How the first research used it

Analysed, not built, never in any model: `reports/weak_language_data_2026-10-22.md` section 3.3 (owner: SFT stays commercial-clean; 40k en->xx rows per language streamed, QE on 4,000, TranslatePsy above the human FLORES+ references on AfriCOMET-QE and SSA-COMET, 1,168 8-gram + 24 sentence hits against our hold-outs in 760k rows (0.16%)); `reports/clean_rl_science_data_2026-10-28.md` and `clean_science_rebuild_pipeline_2026-10-28.md` ("TranslatePsy ... EXCLUDED, NC / NLLB; owner rule"); `reports/sft_clean_rebuild_2026-10-28.md` (no source mentions TranslatePsy). The builder `tp_build.py` and the R10 scan files are in the private Hub dataset `<HF_ORG_LEGACY>/sabiyarn-box-backup-r10-weak-language-data/r10/` (`tp_build.py`, `tp_shard_map.json`, `translatepsy_analysis.csv`; `tp_shard_map.json` copied here). Memory notes: `feedback_licence_nllb_translatepsy_2026-10-23.md` (updated 2026-11-04: allowed for the matched-compute control, research only, must be disclosed), `project_final_science_run_2026-10-22.md`.

### 1.3 Terms of use of this control

TranslatePsy rows enter ONLY the control run. No TranslatePsy row is in the SFT, in any RL stage, in the science stage or in any released weight. Control checkpoints stay on the box (`hub_repo_id: null`, `HF_WRITE_TOKEN` unset in `run_control.sh`), are CC-BY-NC-4.0 + NLLB-derived and must not be pushed to a public repo or used in production. Text for DISCLOSURES.md is in `ledger_snippet.md`.

## 2. The 385M SFT recipe (what the control continues)

Sources: `research_archive_2026-11-03/clean_sft_small/run/configs/growth1/g1b_s6.yaml`, `.../runs/g1b_s6/config.json`, `summary.json`, `schedule_manifest.json`; `data_pipeline/sft_clean/fresh/launch_shards.sh`; `gemma_mt/gemma_mt/train_sft.py` (`SFTConfig`), `utils.lr_at`; Hub config.

| item | value |
|---|---|
| final weights | Hub `<HF_ORG>/gemma-0.4B-translate-sft` = `<HF_ORG>/gemma3-up6-g15-g1b-s6` (**public**, identical commit `31ebb30`, step 32,853, `training_state.json`: val macro 1.3412, micro 1.2846, 693,753,636 tokens seen in shard 6). The Hub tokenizer has the original `<unusedK>` tag layout, fingerprint `69f7f26e...` (checked on a pack built with it) |
| architecture (from the Hub `config.json`; answers ledger M1's open item) | `Gemma3ForCausalLM`, 39 layers, hidden size 640, 4 attention heads, 1 KV head, head_dim 256, intermediate 2048, sliding window 512 with a full-attention layer every 13th layer (3 full layers), vocab 262,144, rope_theta 1e6 / local 1e4, float32 file 1.54 GB |
| trainer | `python -m gemma_mt.train_sft --config <yaml> micro_batch_size=8 grad_accum_steps=4`, `attn_implementation varlen`, `mask_mode flash`, `loss_impl fused`, fp32 master weights with bf16 autocast, grad-clip 1.0 |
| batch | 8 x 4 = **32 packs of 4,096 tokens = 131,072 tokens per step** (padded; real tokens per step 99.8% of that in shard 6: 693,753,636 real tokens over 5,305 steps) |
| optimiser | **Muon** on 2-D decoder matrices, AdamW (betas 0.9/0.95, eps 1e-8) on embeddings and norms, one learning rate, `adjust_lr_fn match_rms_adamw`, momentum 0.95, 5 Newton-Schulz steps, weight decay 0.01 |
| LR schedule | shard 6 (the last of six): peak **3e-5**, 3% linear re-warm-up (fresh Muon state), cosine to `min_lr_ratio 0.1` = **3e-6 at the end** (shards 2-5 used min ratio 0.8). **Final SFT LR = 3e-6** |
| speed | **5.14 s per step** on the A100 (44.8 h for the 31,353 executed stage-B steps; `reports/nllb_regression_link_2026-11-03.md` line 112) = 25.5 k padded tokens/s. The 16.5 k tokens/s of `reports/rl_vs_sft_evidence_2026-11-04.md` section 6 comes from the science SFT at micro-batch 4 on a shared GPU and is the pessimistic rate |
| data of the last shard | 154,421 packs (stage-B shard 6 as uploaded) + 30,627 tail-language replay packs (the old A0 translation packs of the 40 languages with < 2,000 old packs, `make_replay.py`) appended in place, 15,328 replay draws = 9.0% of the draws; schedule 5,305 steps x 32 (169,760 draws, max 2 draws per pack); 105 languages; the 19 TranslatePsy languages are 97,099 of 154,432 draws of the uploaded packs (58% of the shard 6 draws including replay) |
| schedule | `schedule_file` (int64 [steps, 32]) + sidecar with sha256 / pack-language hash / pack-structure hash, checked by `train_sft.load_schedule`; produced by `data_pipeline/sft_clean/chain/sampler.py` with `unimax_schedule` |
| validation | the shard-6 validation split (404 packs, 78 languages with >= 2,000 target tokens in the macro): the 1.3412 / 1.2846 above |
| pack format | `prepare_sft --seq-len 4096 --max-sample-tokens 4096 --directions en2x --pack-by-language --val-fraction 0.002 --min-val-per-language 60`, tag mode `single_token`; prompt `Translate the text to {Language}:  {text}` (two spaces), model turn `<unusedK> {translation}<end_of_turn>`, loss on `<tag> {translation}<end_of_turn>` only (`gemma_mt/gemma_mt/format.py`) |

TranslatePsy rows are formatted exactly like every other translation row: `format.build_example(source, iso, target, tok)` with `GEMMA_MT_TAG_MODE=single_token`; the TranslatePsy `messages` column (system prompt "You are a professional English to Shona translator ...", user "Please translate the following English text into Shona: ...") is NOT used, only `source` and `target`. `verify_tag_mode.py`: 228 samples in the 19 languages tokenised in text mode and converted (tag pieces -> tag id) equal the single_token tokenisation in all 228; the tag ids and pieces equal the s6 `tag_map` for all 85 languages in it.

## 3. Design

### 3.1 Compute budget (training wall-clock only; evaluation excluded)

| budget point | GPU time | source | steps at 5.144 s/step | padded tokens | steps at 7.94 s/step (16.5 k tok/s) |
|---|---:|---|---:|---:|---:|
| stage-A matched | 15,920 s = 4.42 h | STA "Cost", EVID section 6 | 3,094 (**checkpoint at 3,100**) | 406 M | 2,003 |
| mixed-best matched (compute that produced the checkpoint) | 6.83 h | A 4.42 + B1 to its best step 50 (50 x 77.6 s = 1.08 h) + mixed <= 60 steps <= 1.33 h; derived from the step times in EVID section 6, the mixed best step is not stated, so this is an upper bound | 4,780 (**checkpoint at 4,800**) | 629 M | 3,095 |
| RL chain as spent | 8.66 h (A 15,920 s + B1 10,479 s + mixed 4,786 s) | EVID section 6, ledger R15 | 6,061 (**run length 6,000** = 8.57 h) | 786 M | 3,925 |

EXPERIMENTS_LEFT E-03 says "about 510 M tokens at 16.5 k tok/s"; at the measured SFT speed of the same GPU the same 8.66 h buy 786 M padded tokens. The matched quantity is GPU time, so the step budget is fixed by timing the first 50 steps (rule in the yaml header: steps = floor(31,176 s / measured s/step), checkpoints at 51.6% / 79.6% / 100%). At the pessimistic 7.94 s/step the run is 13.2 h.

### 3.2 Learning rate, schedule, one run for three budgets

Constant LR 3e-6 (= the final SFT LR, as instructed) after a 100-step linear warm-up of the fresh Muon state (the SFT's Muon state was not saved to the Hub); `min_lr_ratio 1.0`. Because the LR is constant and the schedule spreads every language evenly over the run (`unimax_schedule`: checked, at step 3,100 each language has done 50.0 to 53.1% of its draws, ideal 51.7%), the step-3,100 and step-4,800 checkpoints of the 6,000-step run ARE the smaller budgets: one run, three evaluation points (`stop_after_steps` segments, `export_trainer_state.py`, `run_control.sh`).

**Caveat the owner should decide before the run**: 3e-6 is a cooled LR (the SFT's own peak was 3e-5). A reviewer will say the control is weak if it moves little. A pre-registered, cheap check is a 300-step LR probe at {3e-6, 1e-5, 3e-5} (about 26 min each, 1.3 GPU-h; select by val macro on the shard-6 validation split and `val_tp`, ties go to 3e-6), then use the winner for the long run. I recommend it; it is not in the minimal budget.

### 3.3 Data mix

Per step: 16 replay packs + 16 TranslatePsy packs (**50% / 50%**), mixed inside each step. Why 50%: (a) TranslatePsy is en->xx, one domain (MADLAD web sentences), 19 languages only; the RL gained on all 54 FLORES+ languages, so the control must also keep the other ~90 languages training, otherwise it would measure forgetting; (b) a 786 M-token budget cannot be filled with 19 languages of new data without repetition (each TranslatePsy pack is drawn at most once; 96,000 packs = 5.3 M pairs, 281k pairs per language); (c) 50% keeps TranslatePsy-language content at 66.5% of the draws against 58.2% in the original last shard, so "more SFT in the same languages" is what is tested.

- **Replay half** (`cut_replay.py`): pool = stage-B shard 6 (154,421 packs) + the tail-language replay (30,627 packs) = 185,048 packs in 111 languages; 96,000 packs chosen by UniMax (min 0, max 1 pass, the repo's `unimax_allocation`): 78 languages are replayed in full (all their packs), the largest languages are cut to a common level of 1,682 packs (about 6.9 M tokens) each. The tokens are copied verbatim from the bucket (byte-identical to what shard 6 trained on). The 19 TranslatePsy languages get 31,596 of the 96,000 replay packs; the other 92 languages get 64,404. `replay_plan_96000.json` (per-language pool/selected), `control_language_plan_6000steps.csv` (replay + TranslatePsy draws per language, against the original shard-6 draws).
- **TranslatePsy half** (`tp_stream_sample.py` -> `tp_prepare.py` -> `prepare_sft`): UniMax over the 19 languages (min 0, max 1): 5,052 to 5,053 packs each. Per language 220k to 373k rows are needed (`tp_rows_needed.json`, from the sample pack: 43 to 74 samples per pack); the build streams 400k rows per language (2 shards x 200 batches). Equal packs per language = equal padded tokens (20.7 M per language), target tokens per sample differ (afr 23, yor 55).
- Totals at 6,000 steps: 786 M padded tokens; TranslatePsy half 393 M tokens (about 196 M target tokens, 49.8% target share in the sample pack), replay half 393 M (about 220 M target tokens at the s6 target share 56.2%: 389.6 M / 693.8 M). The TranslatePsy half is in the range of a language's WHOLE previous SFT (yor 30.6 M tokens in shard 6): it is a large intervention in 19 languages by construction of the matched budget.

### 3.4 End-of-turn emphasis arm

The trainer has no per-token weights (`gemma_mt/losses.py` sums NLL over labelled tokens). `train_sft_eot.py` (new file; gemma_mt is not edited) replaces `gemma_mt.losses.sft_nll_sum` before `make_loss_module` imports it: the NLL of every label equal to `<end_of_turn>` (id 106) is multiplied by **5**; the normalisation is unchanged (divided by the number of labelled tokens), validation (`evaluate` uses `sft_selected_logp`) stays unweighted. With weight 1.0 it is bit-identical to the original; test on a tiny random Gemma3 (`test_eot_loss.py`, .venv-comet, transformers 4.57.6): original 102.57, weighted 148.46 = manual sum with two weighted labels, counts unchanged. Why 5: the end token is 1 of about 36 loss tokens per TranslatePsy sample (2.7%); weight 5 makes it about 12% of the loss. Honest limit: in stage A the `no_end_of_turn` class was 0.87% of samples early on (ledger R10), the 17% unfinished FLORES+ rows are mostly loops (R4); end-of-turn emphasis can only fix a part of them. The arm is therefore exploratory; the headline comparison is C1.

| arm | LR | emphasis | training | evaluations |
|---|---|---|---|---|
| C1 | 3e-6 constant (100-step warm-up) | none | 6,000 steps, 8.57 h, checkpoints 3,100 / 4,800 / 6,000 | 3 |
| C2 | same | `<end_of_turn>` loss x5 | same | 2 (3,100, 6,000) |
| C3 (optional, only if the LR probe or C1 says the LR is too low) | probe winner | none | 6,000 steps | 1 |

## 4. Build (laptop proof at about 1/60 scale, full build commands for a machine with disk)

| step | proof on the laptop | full build |
|---|---|---|
| stream TranslatePsy en->xx | `tp_stream_sample.py`: 19 languages x 50,000 rows (2 shards x 25 batches), 950,000 rows in 1.6 min, 222 MB jsonl | `tp_stream_sample.py WORK 200 2`: 400k rows per language, 7.6 M rows, about 7 GB of Hub traffic, about 10 min |
| filters + protected-index scan | `tp_prepare.py`: 950,000 -> 939,870 rows (98.9%) in 153 s; positive control 1012/1012 FLORES+ devtest English sentences flagged | same, about 16 min single process |
| pack | `prepare_sft` on 5,000 rows per language: 93,860 samples, 1,689 packs, 6,864,164 tokens (73.1 tokens/sample, 3,416,470 target = 49.8%), packing efficiency 0.992, `audit_packed` OK, 3 decoded examples show `<unusedK>` tags | `prepare_sft` as in `pack.py prepare`, 4 workers, about 5.4 M samples, about 25 min |
| replay | `cut_replay.py --n-packs 600` (2.4 M tokens read from the bucket by ranged GET in 31 s) | 96,000 packs, about 1.6 GB of ranged reads |
| combine | `extend_packed.py`: replay 600 + TranslatePsy 1,689 packs = 2,289 packs, 107,159 samples | `--inplace` |
| schedule | `make_control_schedule.py --packed-dir ... --steps 30` on that combined set; **the repo's `load_schedule` accepted it with all six checks** (schedule_sha256, steps, per_step, n_packs, pack_lang_sha256, packed_structure_sha256); dry run at 6,000 steps on the real pack languages: 96,000 + 96,000 draws, no pack drawn twice | `build_control_data.sh` steps 7-8 |

`build_control_data.sh` and `run_control.sh` hold the exact commands with box paths (`/workspace/clean_sft/{run,code,venv_train}`, `/root/.bucket_env`, work dir `/dev/shm/control`). The replay audit of `audit_packed` flags instruction ("other") samples (no translation template) in the replay packs: expected (about 7% of the replay samples are instruction chat), it is only meaningful on the TranslatePsy part.

**Blocker / owner decision: disk.** On 2026-11-04 the A100 box root had 3.2 GB free (96% used); `/dev/shm` has 21 GB free (RAM-backed, lost on a restart; the long-document RL plans to put 8 GB of checkpoints there). The combined data are about 3.7 GB and each trainer checkpoint about 2.7 GB (Muon state included), so the run needs about 11 GB of `/dev/shm` or another disk. The build is CPU only: run it `nice -n 19 taskset` on few cores (the box showed load 5.6 of 80 cores), or on another machine, then copy; the laptop-to-box route is slow (memory note), and I may not write the bucket.

## 5. Contamination scan (protected index, the intake filter's rules)

Index: `research_archive_2026-11-03/databox_backup/work/_hub_protected/protected_index/protected_index.npz`, version `protected-2026-10-28.3`, sha256 `f094fd2b...` (verified at load; same file as `<HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28/protected_index/`), 2,991,392 sentence hashes + 42,696,306 8-gram hashes over 35 protected sets (FLORES+ dev/devtest, IbomMT (+efi), FLORES-MOZ, NTREX-128 (+African), SMOL, SALT, GlotStoryBook, AfriScience-MT dev/test, AfriLongDoc, TICO-19, SSA-MTE, AfriMTE-WMT24, WMT24++, SIB-200, XSIM++, AfriDocMT, NaijaS2ST, MAFAND dev/test). Rules = `decon.py` R1 exact (sentence or sentence part), R2 contiguous (protected sentence of >= 8 tokens inside the text), R3 any shared word 8-gram, both normal forms; **protected strings of fewer than 3 tokens are never indexed (the 3-word rule)**; the FLORES+ dev train half is exempt as in the SFT. This is the intake filter's (strict) machinery, not the scorer of the CONT audit (`reports/contamination_audit_385m_2026-11-03.md`, which counts one- and two-word strings and 8-gram near-duplicates separately); it is stricter on phrase overlap (R3).

**TranslatePsy sample** (950,000 en->xx rows, 19 languages x 50,000 from 2 shards each; `tp_prepare_report.json`, `tp_prepare_per_language.csv`):

| step | rows |
|---|---:|
| read | 950,000 |
| cheap filters: copy source=target 613, URL 1,275, repetition (3-gram share >= 0.30) 1,858, English-word share of target >= 0.45 50, length/ratio 4, duplicate 36 | -3,836 |
| after cheap filters | 946,164 |
| protected-index hits, source only 3,630, target only 2,531, both 133 (**0.67% of the rows, 6,294**) | -6,294 |
| **kept** | **939,870 (98.9%)** |

Rows with at least one hit, per protected set and side (a row can hit several sets: FLORES+ devtest, XSIM++ devtest, FLORES-MOZ and IbomMT share sentences):

| benchmark | source side (English) | target side (African) |
|---|---:|---:|
| FLORES+ devtest | 5 | 128 |
| FLORES+ dev, held-out half | 2 | 251 |
| NTREX-128 | 15 | 428 |
| MAFAND dev / test | 187 / 224 | 97 / 95 |
| SmolSent (`smol:all`) | 2,674 | 1,436 |
| SALT dev / test | 2 / 3 | 0 / 10 |
| GlotStoryBook | 201 | 129 |
| IbomMT (ibom-mt dev/test; efi) | 1 / 4; 20 / 20 | 0 |
| FLORES-MOZ dev / devtest | 1 / 4 | 0 |
| AfriScience-MT dev / test | 1 / 6 | 21 / 22 |
| AfriLongDoc | 2,739 | 1,313 |

By rule (rule-hits, a row can have several): source R1 1,897, R2 80, R3 1,888; target R1 196, R2 8, R3 2,469. So TranslatePsy does contain benchmark sentences (the authors decontaminated only against Flores-200, BOUQuET and Smol, and even for Flores 5 + 128 rows survive: mostly 8-gram phrase overlap in the NLLB output, some English sentences). **All such rows are dropped** (any hit on either side). At the full build (7.6 M streamed rows) the expected number of dropped rows is about 0.7% (about 50k), the same code removes them. This is why the evaluation sets can be used with the control: after the build `prepare_report.json` is the evidence, and the scan must be re-run on the real full data (the 1/8 sample here is a proof, not the audit of the final data).

**Replay half**: the text is the SFT text; the full 385M audit (CONT: s1-s6 and g0, 25,911,095 samples, no benchmark string of >= 3 words) covers it. Re-check with the intake rules on a stratified sample (`replay_sample_scan.py`: 522 packs from the bucket, 12,123 samples (11,284 translation, 839 instruction), 24,246 texts, prompt and target scanned separately): **0 rows with a hit** (`replay_sample_scan.json`).

## 6. Config and run

`control_c1.yaml`, `control_c2_eot.yaml` (both load into `SFTConfig` without unknown keys, checked with the repo's class): init `<HF_ORG>/gemma-0.4B-translate-sft` (public, 1.54 GB), data `/dev/shm/control/combined`, `schedule_file` 6,000 x 32, micro 8 x accum 4, Muon, LR 3e-6 constant (warm-up 100), seed 1234, eval every 250 steps on `val/` (shard-6 validation, macro) and `val_tp/`, resumable checkpoint every 100 steps, `hub_repo_id: null`. The trainer expects these files in the form given (same schema as `g1b_s6.yaml`, so the yaml files are also the copy in the trainer's format). Launch order and checkpoint exports: `run_control.sh` (smoke test of 20 steps, segments to 3,100 / 4,800 / 6,000, `export_trainer_state.py` to HF directories; ARM=c1 or c2_eot).

## 7. Evaluation plan (fixed before any control checkpoint exists)

**Systems compared on identical sentences**: SFT final (existing), stage A final, B1 best, mixed best (existing predictions), C1 at 3,100 / 4,800 / 6,000, C2 at 3,100 / 6,000. Stage-A final vs C@3,100; mixed best vs C@4,800 (compute that produced it) and vs C@6,000 (compute spent).

**Benchmarks and decoding** (identical to the RL models): vLLM bf16 greedy, 768 new tokens, `eval_sft_benchmarks.sh <model> --no-comet --skip-base --max-new-tokens 768` then the COMET pass (`run_m385.sh`, `run_m385_comet.sh` pattern), `GEMMA_MT_TAG_MODE=single_token`: FLORES+ devtest (54 languages), NTREX (24), MAFAND (15), SmolSent (45), SALT (5), GlotStoryBook (8), IbomMT (2), FLORES-MOZ (1); AfriScience-MT dev/test sentence level (`run_afsci.sh`, 6 languages) with plain chrF; contamination flags attached as for the RL models (`contam_audit_14_attach.py`, `clean` subset). Also the 720-sentence RL dev set for monitoring.

**Metrics.** Independent headline: chrF++, plain chrF, spBLEU (references; macro over languages, 95% paired bootstrap with EVID's Poisson-shared-per-source-sentence scheme `reports/data/rl_vs_sft_evidence_2026-11-04/ev_stats.py`), held-out reference-based `masakhane/africomet-stl-1.1`; on AfriScience also SSA-COMET-QE (`ssa_qe_pass.py`). LOOP-family (reported, not headline): finished rate, loop rate, length ratio, LID pass. Label SSA-COMET-QE and AfriCOMET honestly: the TranslatePsy rows were filtered by AfriCOMET, SSA-COMET and MetricX, so neural metrics favour the control (and the RL reward is in that family too).

**Reading the result (fixed now; the owner may change the margin before the run).** Primary endpoint: language-macro chrF++ over the 93 benchmark-language cells (FLORES+ 54, NTREX 24, MAFAND 15), clean rows, d = RL - control. Margin delta = 1.0 chrF++ (about a quarter of the lower end of the "beyond finishing" gain, +2.4 to +4.1, ledger R6).

| outcome | rule |
|---|---|
| RL is a better use of the compute | lower 95% bound of d > +1.0 on the primary endpoint AND lower bound of the AfriCOMET difference > 0 AND d > 0 on the 19 TranslatePsy languages and on the other languages separately |
| comparable | 95% interval of d inside (-1.0, +1.0) and the AfriCOMET interval includes 0 |
| the control is better | upper bound of d < -1.0 |
| anything else | inconclusive, stated as such |

Also reported: the share of the SFT -> RL gap that the control closes, (C - SFT) / (RL - SFT), for chrF++, AfriCOMET, finished rate and loop rate, with intervals; the two sub-panels (19 TranslatePsy languages with 19 / 16 / 12 / 3 languages on FLORES+ / NTREX / MAFAND / SALT, and the others), the 7 near-failed languages (taq knc zgh dik aeb arz ary) separately, and the C1 vs C2 contrast. Checkpoints are the three pre-specified steps; choice among them by the 720-sentence dev only (as for the RL stages), never by benchmark. One training seed per arm (as for every RL stage); the seed spread is E-04's job, so differences within about 0.5 chrF++ are not interpreted. Caveats to state in the paper: TranslatePsy outputs are NLLB-3.3B-like and the benchmarks reward NLLB-like text (NLLB-3.3B scores 38.80 against 26.30 for the SFT on 43 languages, ledger N1), so a good control result partly means "distilling NLLB", not "more of our own SFT"; the control sees TranslatePsy in 19 languages only; constant LR 3e-6 unless the LR probe says otherwise; the 8.66 h include B1's 85 steps after its best step.

## 8. GPU-hours (A100, estimated; nothing measured on the control itself)

| part | hours | assumption |
|---|---:|---|
| data build | 0 GPU | CPU, about 1.5 to 2 h wall on 4 cores |
| 20-step smoke test | 0.05 | |
| LR probe (optional) | 1.3 | 3 x 300 steps x 5.14 s |
| C1 training, 6,000 steps (includes the 3,100 and 4,800 checkpoints) | **8.6** (13.2 at 7.94 s/step) | 5.144 s/step measured in stage B, 20 x 250-step evaluations included |
| C2 training, 6,000 steps | 8.6 | or 4.4 if stopped at the stage-A budget |
| evaluation, per model | 0.8 | EVID option A, generation + AfriCOMET (+ SSA-COMET-QE on AfriScience) |
| evaluations: C1 x3, C2 x2 | 4.0 | |

| plan | training | evaluation | total GPU-h |
|---|---:|---:|---:|
| minimal: C1 only, checkpoints 3,100 and 6,000 | 8.6 | 1.6 | **10.2** |
| recommended: C1 full, C2 stopped at 3,100 | 13.0 | 3.2 | **16.2** |
| both arms full | 17.2 | 4.0 | **21.2** |
| plus LR probe | +1.3 | | +1.3 |

EXPERIMENTS_LEFT says 9.5 h for E-03 (one arm, one evaluation); the 'with and without end-of-turn emphasis' requirement doubles training. The cheaper alternative A' (rejection-sampling fine-tuning, about 5 h) is unchanged.

**Queue.** The A100 runs the science RL, then the long-document RL (5 to 8 h with its evaluation, `reports/rl_long_document_plan_2026-11-04.md`). E-03 starts after `POST_LONG_RL_DONE`. CPU-only items that can be done now: the data build on a machine with disk and `nice`, then the 20-step smoke test as the first GPU use after the long-document evaluation (2 minutes). Suggested position in the experiment list: E-01, E-02, E-09, E-05, E-04, E-13, then E-03 (C1, then C2 only if C1 is not clearly worse than the RL), then E-11.

## 9. Files

`reports/data/sft_control_prep_2026-11-04/`: `tp_stream_sample.py`, `tp_prepare.py`, `tp_prepare_report.json`, `tp_prepare_per_language.csv`, `tp_rows_needed.json`, `tp_shard_map.json`, `cut_replay.py`, `replay_plan_96000.json`, `replay_sample_scan.py`, `replay_sample_scan.json`, `make_control_schedule.py`, `control_language_plan_6000steps.csv`, `train_sft_eot.py`, `test_eot_loss.py`, `verify_tag_mode.py`, `control_c1.yaml`, `control_c2_eot.yaml`, `build_control_data.sh`, `run_control.sh`, `ledger_snippet.md`. Downloads in `research_archive_2026-11-03/sft_control_2026-11-04_dl/`: `tp_sample/` (222 MB, raw TranslatePsy en->xx sample, **NC data, do not publish**), `tp_clean/` and `tp_clean2/` (232 MB each; `tp_clean` is the first run of the scan, `tp_clean2` the final one with the per-set row counts, identical kept rows), `tp_sample_in/` (23 MB), `tp_packed_sample/` (29 MB), `s6_meta/` and `g0_meta/` (array files of the bucket packs, 72 + 161 MB, no text), `replay_sel_test/` (16 MB), `combined_test/` (44 MB), 1.0 GB in all; delete when the build is redone on the box.
