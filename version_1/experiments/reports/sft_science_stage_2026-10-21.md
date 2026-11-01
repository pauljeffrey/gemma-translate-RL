# Short science-SFT stage between the general RL and the science RL (D-58, agent R4, 2026-10-21)

Status: **data built and audited, script and evaluation harness written and smoke-tested, nothing trained, nothing pushed to the Hub or the bucket, the shared GPU was used only for three smoke tests (trainer smokes under a 10 GiB cap, a vLLM harness smoke at 0.08 of the card; each under 5 minutes). Waiting for the coordinator's GO.**

Start model: `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c@latest` = `/workspace/sft_build/runs/final_1433` (step 1433). Everything on the box lives in `/workspace/jobs/r4/` (433 MB now); the code is a copy of `gemma_mt_2c` plus the files below (`/workspace/jobs/r4/gemma_mt_sft/`), the same files are in the repo under `gemma_mt/`.

## 1. What was built

| what | repo | box (`/workspace/jobs/r4/`) |
|---|---|---|
| mixture builder (science + replay + hold-out + packing) | `gemma_mt/scripts/build_sft_science_mix.py` | `gemma_mt_sft/scripts/...` |
| the data (packed format of `prepare_sft`, 8,192-token packs) | tables in `reports/sft_science_stage_2026-10-21/` | `mix_v1/{train,val,val_sci,meta.json,mixture_tables.*}` (150 MB) |
| training config | `gemma_mt/configs/sft_science.yaml` | `gemma_mt_sft/configs/sft_science.yaml` |
| trainer changes (all off by default): L2-SP, drift tracking, extra validation splits, start evaluation | `gemma_mt/gemma_mt/train_sft.py` | same |
| weight interpolation (WiSE-FT / soup of two points) | `gemma_mt/scripts/interpolate_weights.py` | same |
| evaluation harness: three suites, one vLLM engine, sweep, before/after tables | `gemma_mt/scripts/sft_stage_eval.py` | same |
| phase driver (baseline / train / sweep / report / finalize) | `gemma_mt/scripts/run_sci_sft_stage.sh` | same |
| tests (10, CPU, all pass; plus the 57 existing SFT / packing / config tests) | `gemma_mt/tests/test_sft_science_stage.py` | same |

## 2. The mixture

**Replay source: the original packed SFT windows, not the Hub dataset.** `<HF_ORG_LEGACY>/interlaanguage-translation` (as staged, per its card) is the *inter-African X -> Y* dataset (`sft_pairs_train.parquet`, 6.6 M rows, pairs between African languages, English only as a pivot config): it has no English-source rows in the model's format, so it cannot replay what the model was trained on. `gemma_packed_fwd_sft_windows.tar.zst` (2.27 GB) is exactly the en -> X SFT data: 11.33 M samples, 77 languages, windows of 4 length bins (b64..b1024), already cleaned (exact duplicates and Pidgin train/val overlaps removed), audited, tokenised with the model's tokenizer (fingerprint equal to `final_1433`'s, checked) and with the validation split of the original run (hash of the English text) that the original SFT never trained on. The token ids are copied, not re-tokenised, so the replay is byte-identical to what the model saw. Cost: the tarball has no per-row `source_dataset` / licence columns (provenance is the original SFT build, D-21); the science part is Apache-2.0. Streaming: one pass over the 8.8 GB member through `zstd | tar -xO` (55-250 s), only the 87 k sampled windows are kept; the tarball was deleted afterwards (2.4 GB freed, re-downloadable in about a minute).

**Science part** (AfriScience-MT TRAIN, `dsfsi/afriscience_mt`, Apache-2.0; 177 papers, 5,792 sentence pairs per language, multi-parallel):
* **Two passes = "two epochs over the science part"**, each pass uses every sentence once, but cuts every paper differently. A paper gets one granularity per pass (never the same twice): `S` single sentences (the paper's sentence-level setting, 35 % of the papers), `A` 2-4 consecutive sentences (25 %), `B` 5-12 consecutive sentences (20 %), `W` the whole paper in one sample (20 %; sentences joined by single spaces, exactly how `eval_afriscience.py` builds its documents). The cut is the same for the six languages of a paper. This teaches both benchmark settings (970-sentence sentence level, 28-paper whole-paper level, where the final model finishes only 75 % of the dev papers) and the window lengths of the RL prompts (b64..b2048), and it means the model never sees the same context twice.
* Windows never cross a paper. Prompt = the exact SFT prompt (`Translate the text to Hausa:  ...`, two spaces), target = `hau <translation><end_of_turn>`, loss on the target only (`format.build_example`, the same code as the original SFT).
* **`seq_len` 8,192** (the original SFT: 4,096): a median Yoruba paper is 4,108 Gemma tokens (prompt + answer), the longest train paper 7,969; at 4,096 half of the Yoruba whole-paper samples (89 of 177) would have been dropped. 0 science samples are longer than 8,192.
* Dropped: 26 English sentences (same in all six languages) that share a word 8-gram with the hold-out, 1 Zulu pair with an outlier length ratio (target/source characters outside [0.3, 3] x the language's median). 0 exact sentence matches.
* Known data issue (from reading the train rows, not fixed): sentence alignments in the released sentence-level file are sometimes shifted by a clause (e.g. Amharic sentence 5 starts with the end of the translation of sentence 4). Windows of 2+ sentences and whole papers absorb such shifts; single-sentence samples (about a third of the science tokens) carry them. The length-ratio filter does not catch them.

**Replay part** (UniMax-style per-language token cap, all 77 languages):
* total = science tokens x 65/35 = 12.3 M tokens; water-filling gives every language the same budget c = 148 k tokens (the six science languages get 2x = 297 k, because they receive ~8x more science updates and their general ability is the most at risk); a language with less data than its budget would contribute all of it (none does: the smallest, Kabuverdianu, has 0.2 M tokens), at most 1 pass of its own data.
* so a big language is cut to 0.03-0.13 % of its data (Igbo 0.03 %, Hausa 0.12 %, Yoruba 0.13 %) and a small one is replayed with most of it (Kabuverdianu 73.5 %, Kongo 59.6 %, Esan 55.7 %, Ibibio 43.3 %, Herero 38.3 %, Ghomala 24.8 %): "low-resource languages repeat (are covered) more, big ones less". Within a language the windows are drawn at random without replacement (the natural mix of window lengths).
* **Per-language minimum N = 600 replay examples** (the floor binds for five languages with long windows: Kinyarwanda, Chichewa, Somali, Sesotho, Twi; everything else has 611-2,013) **and at least 148 k tokens**. All 77 languages are present: `min = 600 examples, 148,373 tokens`.
* Hold-out filter on the replay: 173 windows rejected by an exact sentence match and 51 by a word 8-gram (they were replaced by the next window of the same language).

**Hold-out (never in train, either part):** AfriScience-MT dev + test English (53 papers), AfriLongDoc `hard` (135 documents) + `core` (450), the general dev set `bench_dev_clean.jsonl` (8,111 distinct English sentences), FLORES+ dev English (997), the English sentences of the RL-vs-SFT evaluation pool (4,487): 36,817 held-out sentences and 684,273 word 8-grams. Exact + 8-gram only: paraphrases are not caught (same limit as A4's hold-out).

**Totals (train):** 96,939 samples in 2,344 packs of 8,192 tokens = **19,193,643 tokens** (11,771,538 target tokens), packing efficiency 0.9996, **science 34.5 % of the tokens** (6,631,094; 29,556 windows) and replay 65.5 % (12,562,549; 67,383 windows); `audit_packed` OK on train, val and val_sci (structure of every pack + token-level checks of 3,000 samples per split). **Validation:** `val/` = 6,934 replay samples (<= 100 per language, 48-100; from the original SFT validation split, hold-out-filtered), `val_sci/` = AfriScience-MT dev: 5,058 sentences + 150 whole papers.

### Table: examples and tokens per language, science vs replay (train), and validation sets

| tag | language | science ex | science tokens | replay ex | replay tokens | replay = % of the language's original SFT tokens | science share of the language's tokens | val ex (replay) | val ex (sci dev) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yor | Yoruba | 4,926 | 1,338,723 | 2,013 | 296,972 | 0.13% | 81.8% | 100 | 868 |
| hau | Hausa | 4,926 | 996,229 | 1,372 | 297,513 | 0.12% | 77.0% | 100 | 868 |
| ibo | Igbo | 0 | 0 | 1,076 | 149,101 | 0.03% | 0.0% | 100 | 0 |
| pcm | Pidgin | 0 | 0 | 781 | 148,443 | 0.65% | 0.0% | 100 | 0 |
| efi | Efik | 0 | 0 | 966 | 148,955 | 0.56% | 0.0% | 100 | 0 |
| twi | Twi | 0 | 0 | 600 | 288,978 | 0.11% | 0.0% | 100 | 0 |
| ewe | Ewe | 0 | 0 | 836 | 148,515 | 2.86% | 0.0% | 82 | 0 |
| fon | Fon | 0 | 0 | 983 | 148,416 | 0.28% | 0.0% | 100 | 0 |
| wol | Wolof | 0 | 0 | 816 | 148,443 | 1.64% | 0.0% | 100 | 0 |
| swh | Swahili | 0 | 0 | 665 | 148,402 | 0.24% | 0.0% | 100 | 0 |
| kin | Kinyarwanda | 0 | 0 | 600 | 156,502 | 0.32% | 0.0% | 100 | 0 |
| som | Somali | 0 | 0 | 600 | 188,725 | 0.31% | 0.0% | 100 | 0 |
| orm | Oromo | 0 | 0 | 648 | 149,082 | 0.64% | 0.0% | 100 | 0 |
| amh | Amharic | 4,926 | 1,046,659 | 1,289 | 296,841 | 2.12% | 77.9% | 100 | 868 |
| tir | Tigrinya | 0 | 0 | 760 | 148,373 | 1.66% | 0.0% | 100 | 0 |
| lug | Luganda | 4,926 | 1,085,971 | 1,619 | 296,792 | 1.37% | 78.5% | 100 | 868 |
| lin | Lingala | 0 | 0 | 912 | 148,466 | 2.28% | 0.0% | 90 | 0 |
| zul | Zulu | 4,926 | 1,030,061 | 1,203 | 296,858 | 0.54% | 77.6% | 100 | 868 |
| xho | Xhosa | 0 | 0 | 649 | 148,780 | 0.19% | 0.0% | 100 | 0 |
| afr | Afrikaans | 0 | 0 | 611 | 148,440 | 0.24% | 0.0% | 100 | 0 |
| sna | Shona | 0 | 0 | 631 | 148,810 | 0.55% | 0.0% | 100 | 0 |
| nya | Chichewa | 0 | 0 | 600 | 159,465 | 0.46% | 0.0% | 100 | 0 |
| sot | Sesotho | 0 | 0 | 600 | 177,057 | 0.44% | 0.0% | 100 | 0 |
| tsn | Setswana | 0 | 0 | 887 | 148,588 | 0.26% | 0.0% | 100 | 0 |
| mlg | Malagasy | 0 | 0 | 778 | 148,594 | 0.89% | 0.0% | 100 | 0 |
| ibb | Ibibio | 0 | 0 | 899 | 148,467 | 43.28% | 0.0% | 60 | 0 |
| urh | Urhobo | 0 | 0 | 946 | 148,393 | 7.33% | 0.0% | 100 | 0 |
| bem | Bemba | 0 | 0 | 854 | 148,599 | 1.71% | 0.0% | 100 | 0 |
| tum | Tumbuka | 0 | 0 | 826 | 149,017 | 2.29% | 0.0% | 100 | 0 |
| run | Kirundi | 0 | 0 | 778 | 148,451 | 2.14% | 0.0% | 95 | 0 |
| ven | Venda | 0 | 0 | 887 | 148,387 | 1.92% | 0.0% | 100 | 0 |
| tso | Tsonga | 0 | 0 | 917 | 148,678 | 1.01% | 0.0% | 100 | 0 |
| nso | Sepedi | 4,926 | 1,133,451 | 1,656 | 296,780 | 1.73% | 79.2% | 100 | 868 |
| ssw | Swati | 0 | 0 | 803 | 148,439 | 2.48% | 0.0% | 100 | 0 |
| nbl | South Ndebele | 0 | 0 | 827 | 148,618 | 2.21% | 0.0% | 76 | 0 |
| sag | Sango | 0 | 0 | 889 | 148,450 | 2.18% | 0.0% | 100 | 0 |
| tiv | Tiv | 0 | 0 | 874 | 148,570 | 2.11% | 0.0% | 100 | 0 |
| umb | Umbundu | 0 | 0 | 831 | 148,962 | 2.36% | 0.0% | 78 | 0 |
| kmb | Kimbundu | 0 | 0 | 851 | 148,434 | 3.50% | 0.0% | 83 | 0 |
| mos | Mossi | 0 | 0 | 900 | 148,581 | 1.95% | 0.0% | 100 | 0 |
| kik | Kikuyu | 0 | 0 | 884 | 148,439 | 2.70% | 0.0% | 100 | 0 |
| luo | Luo | 0 | 0 | 845 | 148,427 | 0.48% | 0.0% | 100 | 0 |
| kam | Kamba | 0 | 0 | 832 | 149,113 | 5.09% | 0.0% | 100 | 0 |
| bin | Edo | 0 | 0 | 1,014 | 148,700 | 5.34% | 0.0% | 100 | 0 |
| dag | Dagbani | 0 | 0 | 714 | 148,442 | 4.56% | 0.0% | 73 | 0 |
| oke | Okpe | 0 | 0 | 945 | 148,429 | 12.26% | 0.0% | 100 | 0 |
| shi | Tachelhit | 0 | 0 | 966 | 148,480 | 10.49% | 0.0% | 60 | 0 |
| dyu | Dyula | 0 | 0 | 870 | 148,383 | 18.90% | 0.0% | 70 | 0 |
| bam | Bambara | 0 | 0 | 1,001 | 148,402 | 23.61% | 0.0% | 48 | 0 |
| kbp | Kabiye | 0 | 0 | 1,005 | 148,399 | 22.40% | 0.0% | 62 | 0 |
| bbj | Ghomala | 0 | 0 | 869 | 148,565 | 24.79% | 0.0% | 60 | 0 |
| kon | Kongo | 0 | 0 | 797 | 148,435 | 59.59% | 0.0% | 60 | 0 |
| kea | Kabuverdianu | 0 | 0 | 763 | 149,530 | 73.52% | 0.0% | 60 | 0 |
| kab | Kabyle | 0 | 0 | 879 | 149,343 | 2.29% | 0.0% | 99 | 0 |
| nus | Nuer | 0 | 0 | 864 | 148,411 | 0.97% | 0.0% | 100 | 0 |
| swc | Congo Swahili | 0 | 0 | 661 | 149,420 | 1.26% | 0.0% | 100 | 0 |
| lua | Luba-Kasai | 0 | 0 | 747 | 148,786 | 0.99% | 0.0% | 100 | 0 |
| kqn | Kaonde | 0 | 0 | 779 | 148,448 | 1.36% | 0.0% | 100 | 0 |
| lun | Lunda | 0 | 0 | 750 | 148,666 | 2.31% | 0.0% | 100 | 0 |
| iso | Isoko | 0 | 0 | 841 | 148,423 | 1.23% | 0.0% | 100 | 0 |
| loz | Lozi | 0 | 0 | 830 | 149,321 | 0.70% | 0.0% | 100 | 0 |
| wal | Wolaytta | 0 | 0 | 836 | 149,065 | 2.35% | 0.0% | 100 | 0 |
| ach | Acholi | 0 | 0 | 780 | 148,385 | 4.53% | 0.0% | 77 | 0 |
| tsc | Tswa | 0 | 0 | 830 | 148,456 | 3.60% | 0.0% | 87 | 0 |
| vmw | Makhuwa | 0 | 0 | 816 | 148,406 | 4.15% | 0.0% | 72 | 0 |
| kua | Kuanyama | 0 | 0 | 793 | 148,412 | 2.89% | 0.0% | 100 | 0 |
| seh | Sena | 0 | 0 | 818 | 148,440 | 6.24% | 0.0% | 70 | 0 |
| nyn | Nyankore | 0 | 0 | 834 | 148,670 | 6.12% | 0.0% | 70 | 0 |
| her | Herero | 0 | 0 | 846 | 148,624 | 38.34% | 0.0% | 65 | 0 |
| yao | Yao | 0 | 0 | 812 | 148,816 | 6.75% | 0.0% | 100 | 0 |
| sid | Sidamo | 0 | 0 | 804 | 149,106 | 6.55% | 0.0% | 100 | 0 |
| ndc | Ndau | 0 | 0 | 780 | 148,432 | 10.63% | 0.0% | 61 | 0 |
| kri | Krio | 0 | 0 | 897 | 148,436 | 6.94% | 0.0% | 70 | 0 |
| ish | Esan | 0 | 0 | 916 | 148,394 | 55.70% | 0.0% | 63 | 0 |
| dua | Duala | 0 | 0 | 891 | 148,619 | 17.84% | 0.0% | 92 | 0 |
| bas | Basaa | 0 | 0 | 840 | 148,877 | 10.88% | 0.0% | 87 | 0 |
| wes | Cameroon Pidgin | 0 | 0 | 801 | 148,912 | 17.82% | 0.0% | 64 | 0 |
| **all 77** | | 29,556 | 6,631,094 | 67,383 | 12,562,549 | 0.57% | 34.5% | 6,934 | 5,208 |

(`science ex` counts both passes: 4,926 windows per science language = 3,393 single sentences + 1,166 windows of 2-4 + 288 of 5-12 + 79 whole papers; the last column of every non-science language is 0 by design. Full numbers, including target tokens: `reports/sft_science_stage_2026-10-21/mixture_tables.{md,json}`, `meta.json`.)

## 3. Training configuration and the learning-rate choice

`configs/sft_science.yaml`: AdamW (betas 0.9 / 0.95, eps 1e-8), fp32 master weights with bf16 autocast, varlen attention + fused loss (as in the SFT runs), gradient checkpointing, **65,536 tokens per step** (2 packs x 4 accumulation x 8,192; half of the SFT runs' 131,072, to get **293 steps** out of 19.2 M tokens instead of 146), `epochs: 1.0` over the packs (the two science passes are in the data), grad clip 1.0, warmup 6 % (18 steps), cosine to 10 %, evaluation at steps 0 / 100 / 200 / 293 (replay val, 77-language macro, and AfriScience dev loss), weights-only checkpoint every 50 steps (`last/trainer_state.pt`, 1.1 GB), no early stopping, **no Hub push**. Initial weights = `model_name: /workspace/sft_build/runs/final_1433` (the trainer loads an HF directory, fresh optimizer; the RL optimizer state is not used; the tokenizer of that directory is fingerprint-checked against the data).

**Peak learning rate 3e-6** (range 1e-6 .. 5e-6 considered). Reasons, with the measurements I have:
* The RL run that produced the start model used 2e-6 for 945 steps and moved the weights by 0.0036 % of their norm (measured: `final_1433` vs `rl_2b_stage_b64`, relative L2: 0.0027 % in the layers, 0.118 % in the embedding rows that were touched) while improving the weak languages by about 5 chrF++; the SFT runs ended their cosine at 5e-6. A supervised stage with consistent gradient direction at 3e-6 over 293 steps should move the weights by a similar small amount (smoke: 4 steps at <= 3e-6 gave a relative drift of 2.9e-6 in the layers, ~7e-7 per step; 293 steps with the cosine would be at most about 1e-4 = 0.01 %, less if gradients are noisy), i.e. "keep the weights close" is built in.
* The first four smoke steps already lowered the replay validation loss (macro 1.454 -> 1.387 on 6 packs) and the AfriScience dev loss (2.137 -> 2.120), so the learning signal is there at this lr; 1e-6 would make the stage mostly a no-op over 293 steps, 5e-6 doubles the drift for a gain I cannot yet quantify.
* Fresh Adam state: the second-moment estimate starts at zero, so the first steps are sign-like; hence the 18-step warmup.
* **Decision rule, to be applied after the first run**: if the AfriScience dev loss (logged as `val_val_sci_loss`) has fallen by less than ~0.15 nats at the end, rerun at 5e-6; if the forgetting check below fails, rerun at 3e-6 with L2-SP (or just use a smaller interpolation alpha, which costs no training).
* The stage is cheap enough (17-30 min per run, see section 6) to run two or three learning rates; I recommend the first run at 3e-6, the second (optional) at 5e-6 or 3e-6 + L2-SP 0.3 depending on what the first shows.

**Weight decay 0.0.** Honest note: AdamW's decoupled weight decay multiplies the weights by (1 - lr x wd) each step, i.e. it shrinks them towards **zero**, not towards the starting weights, so it is not a drift regulariser. At lr 3e-6, wd 0.01 and 293 steps the total shrink factor is 1 - 1e-5: no effect either way. It is left at 0.0 so that nothing pretends to regularise.

**Drift-control tools actually available (in order of strength):** (1) the small lr, (2) the short run (293 steps), (3) replay of all 77 languages (35 / 65 mixture, science languages' replay doubled), (4) **optional L2-SP**: `l2sp_coef: c` adds `c * (theta - theta_start)` to every gradient after clipping (so the pull is never scaled down by the clip; with Adam it is normalised like any gradient); `theta_start` = the weights the run starts from (a copy kept on the GPU, 1.1 GB). Magnitude guide: per-parameter data gradient is about 6e-5 (gradient norm ~1 after clipping over 2.7e8 parameters), the free drift per parameter about 1e-4..4e-4, so c = 0.3 makes the pull comparable to the data gradient at the typical drift, c = 1 clearly dominates it; suggested: 0.3 first, then 1.0 (tested on the CPU: a strong coefficient cuts the drift by more than 40 %; real-run effect not measured), (5) **end-of-run weight interpolation** `theta = (1 - a) theta_init + a theta_sft`, a in {0.3, 0.5, 0.7, 1.0}, every alpha evaluated on all suites, the winner saved as an HF model directory (`interpolate_weights.py`, 16 s on the CPU, exact: max error 0.0 against the arithmetic mean in the test); this is the cheapest and most reliable lever because it needs no retraining. (6) Monitoring: `drift_l2`, `drift_rel`, `drift_rel_layers`, `drift_rel_embed`, `l2sp_penalty` logged every 5 steps, validation loss of the untouched model at step 0.

## 4. Evaluation plan (after training; `scripts/sft_stage_eval.py`)

One vLLM engine per model (greedy, 512 new tokens per sentence, 7,168 per paper, fixed KV cache of 4 GiB because vLLM's own memory profile fails on a shared card: it did in my first smoke), three suites:

| suite | data | what it measures |
|---|---|---|
| `afs` | AfriScience-MT **dev**: 843 sentences + 25 papers x 6 languages (functions of A4's `eval_afriscience.py`, so numbers are comparable with its baseline) | sentence level chrF / chrF++ / BLEU / spBLEU; paper level (whole paper in one prompt) chrF, chrF++, BLEU, finished rate, length ratio. TEST is read only by `finalize` (once, for the chosen alpha). |
| `general` | `bench_dev_clean.jsonl`, 19 languages, ~10 k sentences | chrF++ per language = the forgetting check |
| `weak` | FLORES+ **dev**, 60 fixed sentences (same ids in every language; 44 languages have a FLORES+ dev file) + for the weak-signal languages without one, 60 reference sentences of the RL-vs-SFT pool (SmolSent, GlotStory, IbomMT, NTREX, FLORES-MOZ, SALT): 63 languages x 60 sentences | chrF++ per language; 32 of the 45 weak-signal languages (23 consensus-only + 22 blend) are covered, **13 have no reference text anywhere in our data** (bas bbj bin dua ish iso oke sid swc urh wal wes yao) and can only be judged by the validation loss of `val/` (every one of them has 60-100 replay validation windows) |

Reports: per language and per group (science6, weak-consensus, weak-blend, control, other, ALL) with the delta against the `final_1433` baseline and a 95 % bootstrap interval over languages, one column per alpha (`compare`). Reference numbers already measured for the baseline by A4 (`reports/afriscience_science_run_2026-10-21/baseline_final_1433/`, dev, greedy): sentence level chrF 48.59 / chrF++ 45.65 / BLEU 21.52 (amh 26.07, hau 61.45, lug 46.54, nso 58.10, yor 38.13, zul 61.23); whole papers chrF 36.00 / chrF++ 32.09 / BLEU 6.19, finished rate 0.75 (lug 0.48, yor 0.60). The baseline is re-run by `run_sci_sft_stage.sh baseline` with the same code for all three suites.

**Choice rule (proposed):** the alpha with the best AfriScience-dev sentence chrF++ subject to (a) general-dev macro chrF++ not lower than the baseline by more than 0.5, (b) every weak group (consensus, blend, control) not lower by more than 0.5 with the bootstrap interval not excluding 0 from below, (c) paper-level finished rate not below the baseline. The noise floor: one dev paper moves a paper-level chrF by about 1 point; the sentence-level 5,058 dev sentences have a chrF++ noise of about +-0.4 (A4); weak-language cells with 60 sentences are noisy per language (+-3), so only group means are decided on.

## 5. Smoke results (2026-10-21, shared RTX PRO 6000; the other jobs used 59-80 GB and 75-100 % of the card)

| test | result |
|---|---|
| unit tests, CPU | 10 new tests pass (window coverage / no cross-paper window / hold-out removal / UniMax water-filling / streaming extraction across chunk borders / interpolation endpoints exact / L2-SP cuts drift / start evaluation + extra validation split logged); 57 existing SFT, packing, loss, config tests pass; the 1 failure `test_stratified::test_pusher_is_disabled_without_repo_or_token` is the known pre-existing one (a Hub token exists in the environment) |
| data audit | `audit_packed` OK on train / val / val_sci |
| trainer on the real data and the real start weights, GPU capped at 10 GiB (`torch.cuda.set_per_process_memory_fraction`), 4 steps, micro-batch 1 x accum 2, checkpointing | runs: start evaluation (val macro 1.4537, sci 2.1369) -> 4 steps (train loss 1.25, 1.21, 1.30, 1.30; gradient norm 9-11 before clipping, so the clip of 1.0 is active at every step, harmless with Adam) -> evaluation (val macro 1.3868, sci 2.1199), checkpoint written, drift logged (layers 2.7e-6 relative after 4 steps). 4.0-4.8 k tokens/s |
| memory | micro-batch 2 x 8,192 with checkpointing: **peak 6.5 GiB allocated**; micro-batch 1 without checkpointing does **not** fit in 10 GiB (OOM), so `gradient_checkpointing: true` is the default; speed with micro-batch 2: about **19,000 tokens/s** while another job used the card (steps 2-3 of that smoke: 0.85-0.9 s per 16,384 tokens) |
| interpolation on the real weights | alpha 0.5 of `final_1433` and the 4-step checkpoint written as an HF directory in 16 s, tensors equal to the arithmetic mean (max abs error 0.0), loaded by vLLM |
| evaluation harness, vLLM at 0.08 of the card, 8 sentences per language | all three suites ran end to end (afs dev sentence + paper level, general 19 languages, weak 63 languages), `stage_eval.json` written, `compare` produced the tables; vLLM start-up 1.5 min (first try failed on vLLM's shared-GPU memory-profile assertion: fixed with a fixed KV cache + retry) |

Not done (waiting for the GPU): the baseline evaluation of `final_1433` on the three suites, the training run, the sweep. The step-0 numbers above are from 6 packs and are not a baseline.

## 6. Launch commands, time and cost

```bash
# on the box (ssh -o BatchMode=yes -i $SSH_KEY -p $BOX_PORT root@$BOX_HOST), after the coordinator's GO
cd /workspace/jobs/r4/gemma_mt_sft
B=scripts/run_sci_sft_stage.sh
nohup bash $B baseline > /workspace/jobs/r4/logs/phase_baseline.log 2>&1 &     # 1. ~10 min, vLLM 15 GB
nohup bash $B train    > /workspace/jobs/r4/logs/phase_train.log    2>&1 &     # 2. ~20-35 min, 6.5 GB (needs 30 GB free by default; FORCE=1 to skip the check)
#    extra runs:  bash $B train out_dir=/workspace/jobs/r4/runs/sci_sft_lr5 learning_rate=5e-6
#                 bash $B train out_dir=/workspace/jobs/r4/runs/sci_sft_l2sp l2sp_coef=0.3
nohup bash $B sweep    > /workspace/jobs/r4/logs/phase_sweep.log    2>&1 &     # 3. alphas 0.3 0.5 0.7 1.0, ~10 min each (RUN=/workspace/jobs/r4/runs/sci_sft_lr5 for another run)
bash $B report                                                                   # 4. -> /workspace/jobs/r4/report_tables.md
bash $B finalize 0.7                                                             # 5. HF dir /workspace/jobs/r4/models/sci_sft_alpha_0.7 + dev AND test (once)
# the equivalent single commands: python -m gemma_mt.train_sft --config configs/sft_science.yaml ;
#   python scripts/sft_stage_eval.py sweep --init /workspace/sft_build/runs/final_1433 --sft runs/sci_sft/last/trainer_state.pt --alphas 0.3 0.5 0.7 1.0 --out eval
#   python scripts/interpolate_weights.py --init ... --sft ... --alpha 0.7 --out DIR
```
All commands run under `nice -n 10`, `OMP_NUM_THREADS=6` (via `/workspace/jobs/r4/env.sh`); phases are restartable (training resumes from the weights-only checkpoint with a 10-step re-warmup; the sweep skips alphas that already have a `stage_eval.json`). If the card is busy, the trainer falls back by itself to micro-batch 1 x accum 8 after a crash.

**Time (estimates; the RTX PRO 6000 was not measured at a quiet moment).** Training: 19.19 M tokens + 4 evaluations (about 10 step-equivalents each, 2.2 M tokens) = about 21.5 M token-passes. At the measured 19 k tokens/s (card shared): **~19 min**; at the RTX 5070's measured 6.1 k tokens/s: 59 min; at the 4-5 k tokens/s of the first, unoptimised smoke: 75 min. Expect **20-35 min**. Evaluation: engine start 1.5 min + dev sentences / papers (up to 7,168 new tokens for the 150 papers, A4: dev + test both levels about 10 min) + general 10 k sentences + weak 3.8 k sentences: **~8-10 min per model**; baseline + 4 alphas = **~45 min**; the TEST run of the winner another ~5 min. Total GPU for one training run plus the full sweep: **about 1.3-1.7 h; at the 1.034 USD/h planning rate used by A4 and the cost report about 1.5-2 USD**, 3 training runs (lr / L2-SP variants) with their own sweeps about 4-5 USD. Disk: `mix_v1` 150 MB, one checkpoint 1.1 GB, one interpolated model 1.1 GB at a time (deleted after its evaluation): peak ~3 GB, inside the 6 GB budget.

## 7. Risks and open points

1. **Replay of human references is not replay of the RL policy.** The step-0 smoke shows the RL model has a high loss on the SFT references and that four tiny steps on the mixture lower it quickly (macro 1.454 -> 1.387). Part of the RL gain (fewer loops, 99 % finished against 78 % for the SFT model on the weak languages, +5 chrF++ on consensus-only languages that RL trained on its own consensus signal) could be undone by pulling the policy back towards noisy reference translations (several sources are JW / Bible material). This is what the `weak` suite and the `finished_rate` rows measure; the tools against it are the small lr, L2-SP and the interpolation. A different replay would use the **RL policy's own greedy outputs** for the same prompts (self-distillation: an exact anchor on the policy distribution, ~5-8 GPU-minutes of vLLM for 67 k prompts; not built, the builder can take it in about an hour if the first result shows regression).
2. **Greedy decoding of whole papers loops** (75 % finished at baseline). The science passes teach whole-paper samples (20 % of the paper-passes: 79 whole-paper samples per science language over the two passes) with the human end; whether that cures greedy loops is a result I cannot predict.
3. **Only dev is used for choosing; dev is small** (25 papers). A different seed or a second lr changes paper-level chrF by about a point; I would not choose between alphas that differ by less than that.
4. **Eval-set coverage:** 13 of the 77 languages have no reference in any of our evaluation sets, so forgetting there is visible only in the validation loss of the replay split; `weak` groups use 60 sentences per language.
5. **Hold-out is exact + 8-gram**, as in A4's science prompts; the benchmark papers' sources may be in PMC / arXiv under other wording.
6. **vLLM on a shared card:** the engine can fail its memory profile when other jobs allocate during start-up (observed once); the harness uses a fixed KV cache and retries three times; the card should be quiet for the baseline and the sweep to be timed reliably.
7. The AfriScience sentence alignment issue (section 2) caps what single-sentence samples can teach; whole-paper and window samples are the cleaner signal.
8. Next stage: the science RL (D-52) starts from `runs/sci/last` = a copy of the 2c trainer state. Starting it from the SFT-stage model needs a new trainer state: weights from the chosen alpha, a fresh optimizer (or the 2c Adam state with a re-warmup), and `model_name` = the new directory as KL reference. Not touched here.
