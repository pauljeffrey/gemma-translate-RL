# Gemma-3-270m-it English -> African: post-SFT benchmark evaluation (2026-10-18)

Pipeline: `gemma_mt/scripts/eval_sft_benchmarks.sh` (D-34 in `gemma_mt/DECISIONS.md`). Numbers below were produced on the OLD box (RTX 5070 12 GB, shared with foreign jobs) with the HF sampler, because that box has no vLLM; the vLLM path is CPU-tested only (see "Open problems"). Raw outputs of the two runs are next to this file in `gemma_sft_benchmarks_2026-10-18/` (`full_flores/`, `sub300/`: `results.json`, `results.csv`, `report.md`; `predictions.jsonl` stayed on the box, `/workspace/sft_build/eval/{full_flores,sub300}`).

## 1. Systems, data, method

* **SFT model**: Hub `<HF_ORG_LEGACY>/gemma3-270m-translate`, `main` at the moment of the run, **commit `ae0ea640e7f3ff8acb6d97eb7538246d6b786a52`, training step 20500, macro validation loss 1.3025** (the brief's "SFT best at step 17,500, macro val 1.3363" was already superseded: `main` is the best-validation checkpoint and kept moving while this was built; both runs are pinned to this one commit). The checkpoint is the best-validation one by construction of the push policy, not necessarily the last.
* **Base**: `google/gemma-3-270m-it` (`ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`), same prompt (`Translate the text to {language}:  {english}`), greedy, scored on its whole output (a leading language tag is stripped only if present; the base never writes one).
* **Benchmarks (English -> X, test splits only; dev is calibration data and is never fetched)**: FLORES+ devtest (1,012 sentences, 45 of our languages incl. 5 added codes; Twi = Akuapem `twi_Latn_akua1239`, Asante not evaluated; Oromo = `gaz_Latn`; Efik has no data), NTREX-128 newstest2019 (1,997 sentences, 24 languages), MAFAND-MT test (15 languages, 998-1,835 sentences). 113,477 rows in total. Language codes were checked against the real listings and fixed in `languages.py` (sna NTREX `sna-Latn`; luo MAFAND; NTREX bem nde nso shi ssw ven; FLORES+ kea sag dyu knc vmw).
* **Scale of the two runs**: FLORES+ was run on **all 45,540 rows** (SFT and base). NTREX and MAFAND were run on the **first 300 rows of every language** (7,200 and 4,500 rows) because the shared old-box GPU generated only 7-25 rows/s; the full NTREX / MAFAND rows are scored by the same script on the rental. Row order is the benchmark's own, so the NTREX subset is the same 300 sentences for all languages.
* **Decoding / metrics**: greedy, `max_new_tokens` 320 (the longest reference is 400 tokens, p99 about 140). chrF++ (sacrebleu `word_order=2`), spBLEU (`flores200` tokenizer), BLEU (13a) on NFC-normalised hypothesis and reference; AfriCOMET-STL-1.1 (with reference) and AfriCOMET-QE-STL-1.1 for the languages in `SUPPORTED_RL` (40 of the 45 FLORES+ languages, 21 of 24 NTREX, all 15 MAFAND), run in a child process with transformers 4.55 (D-33). Macro = unweighted mean over languages.
* **Contamination filter** (section 3): a row is *contaminated* if its English sentence is in the SFT training windows; **the headline numbers are the clean ones**; every table has both. The clean macro counts languages with >= 30 clean rows (all languages qualify here).

## 2. Results

### 2.1 Headline: clean rows (English sentence not in the SFT training text)

| benchmark | system | languages | rows | chrF++ | spBLEU | BLEU | AfriCOMET | AfriCOMET-QE |
|---|---|---|---|---|---|---|---|---|
| flores_plus | SFT | 45 | 2160 | 27.01 | 7.97 | 6.12 | 0.475 | 0.388 |
|  | base gemma-3-270m-it | 45 | 2160 | 12.96 | 1.56 | 0.77 | 0.208 | 0.353 |


FLORES+: full devtest; only 48 of its 1,012 sentences are clean (2,160 rows over 45 languages), so each per-language clean figure rests on 48 sentences and is noisy, but the macro over 45 languages is stable and all languages see the same 48 sentences.

| benchmark | system | languages | rows | chrF++ | spBLEU | BLEU | AfriCOMET | AfriCOMET-QE |
|---|---|---|---|---|---|---|---|---|
| ntrex | SFT | 24 | 4344 | 30.96 | 11.35 | 8.07 | 0.534 | 0.478 |
|  | base gemma-3-270m-it | 24 | 4344 | 12.77 | 1.88 | 0.92 | 0.238 | 0.379 |
| mafand | SFT | 15 | 3544 | 38.41 | 16.65 | 11.45 | 0.620 | 0.587 |
|  | base gemma-3-270m-it | 15 | 3544 | 15.93 | 3.06 | 1.73 | 0.283 | 0.428 |


NTREX / MAFAND: first 300 rows per language (clean rows: NTREX 181 of 300 per language, MAFAND 3,544 in total, from 32 for Igbo to 300 for Chichewa, Shona, Twi, Xhosa).

### 2.2 Same tables on all rows (contaminated rows included)

| benchmark | system | languages | rows | chrF++ | spBLEU | BLEU | AfriCOMET | AfriCOMET-QE |
|---|---|---|---|---|---|---|---|---|
| flores_plus | SFT | 45 | 45540 | 30.15 | 10.40 | 8.09 | 0.563 | 0.524 |
|  | base gemma-3-270m-it | 45 | 45540 | 13.57 | 1.98 | 1.21 | 0.246 | 0.429 |


| benchmark | system | languages | rows | chrF++ | spBLEU | BLEU | AfriCOMET | AfriCOMET-QE |
|---|---|---|---|---|---|---|---|---|
| ntrex | SFT | 24 | 7200 | 32.84 | 13.00 | 9.58 | 0.560 | 0.508 |
|  | base gemma-3-270m-it | 24 | 7200 | 13.49 | 2.21 | 1.23 | 0.250 | 0.394 |
| mafand | SFT | 15 | 4500 | 39.40 | 18.08 | 12.96 | 0.628 | 0.595 |
|  | base gemma-3-270m-it | 15 | 4500 | 15.78 | 3.11 | 1.79 | 0.276 | 0.425 |



Reading: the SFT model is +14 (FLORES+), +18 (NTREX), +22 (MAFAND) chrF++ above the base on clean rows and +0.27 / +0.30 / +0.34 AfriCOMET. The base model without the tag convention and with almost no African-language ability scores 13-16 chrF++. Contamination inflates the SFT model by about 3 chrF++ on FLORES+ (30.2 vs 27.0), 1.9 on NTREX and 1.0 on MAFAND, and the base only by 0.6 / 0.7 / -0.2.

## 3. Contamination

Training text = `gemma/windows_sft` (11,442,172 windows, the exact input of `prepare_sft` -> `packed_fwd`; the 0.2% validation windows are included, which is slightly conservative). Normalisation NFKC + case-fold + punctuation/whitespace runs -> one space. Rules: **sentence rule** = the filter that produced `bench_dev_clean.jsonl` (a window sentence equals the benchmark sentence): my scanner reproduces that file **exactly** (10,043 of 30,418 dev rows clean, zero differences); **contaminated (headline)** = sentence rule OR the benchmark sentence occurs as a contiguous token run inside a window (adds 1,709 dev rows the old filter missed); **near** = not contaminated but >= 50% of the sentence's 8-token n-grams occur in the training English; **pair seen** = contaminated and the reference translation is in the target side of a window of the same language. Flag files: `/workspace/sft_build/bench/contamination_<benchmark>.json` (key `<iso>:<row id>`, maps to booleans; also `near`, `pair`, `sentence`), made in 2 minutes with `python -m gemma_mt.eval.contamination --windows gemma/windows_sft --bench-root bench/test_sets --out-dir bench --dev-file bench/bench_dev.jsonl --dev-clean-file bench/bench_dev_clean.jsonl`.

| benchmark | rows (all languages) | contaminated (headline rule) | sentence rule (dev-filter rule) | near duplicate (8-gram >= 50%) | reference pair seen | clean rows |
|---|---|---|---|---|---|---|
| flores_plus | 45540 | 43380 (95.3%) | 38250 (84.0%) | 0 (0.0%) | 1848 (4.1%) | 2160 (4.7%) |
| ntrex | 47928 | 19104 (39.9%) | 16752 (35.0%) | 144 (0.3%) | 563 (1.2%) | 28824 (60.1%) |
| mafand | 20009 | 4375 (21.9%) | 3960 (19.8%) | 101 (0.5%) | 3096 (15.5%) | 15634 (78.1%) |
| dev | 30418 | 22084 (72.6%) | 20375 (67.0%) | 103 (0.3%) | 6416 (21.1%) | 8334 (27.4%) |


* FLORES+ and NTREX are multi-way parallel: the English side is identical for all languages, so the share is identical per language (FLORES+ 964 of 1,012 sentences; NTREX 796 of 1,997). The **pair** column shows that the model saw the real reference translation far more rarely (FLORES+: 4.1% of rows, mostly hau 842, ibo 559, yor 355; none for most languages): the English sentences come into the training windows through other language pairs.
* MAFAND per language (full test sets):

| language | rows | contaminated | sentence rule | near | pair seen | clean |
|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 1558 | 1311 (84.1%) | 1148 | 11 | 748 | 247 |
| Hausa (`hau`) | 1500 | 407 (27.1%) | 332 | 19 | 237 | 1093 |
| Igbo (`ibo`) | 1500 | 1369 (91.3%) | 1340 | 0 | 1352 | 131 |
| Pidgin (`pcm`) | 1564 | 206 (13.2%) | 199 | 6 | 190 | 1358 |
| Twi (`twi`) | 1500 | 14 (0.9%) | 14 | 7 | 0 | 1486 |
| Swahili (`swh`) | 1835 | 219 (11.9%) | 213 | 39 | 121 | 1616 |
| Kinyarwanda (`kin`) | 1006 | 5 (0.5%) | 5 | 0 | 0 | 1001 |
| Amharic (`amh`) | 1037 | 259 (25.0%) | 246 | 1 | 153 | 778 |
| Luganda (`lug`) | 1500 | 3 (0.2%) | 3 | 0 | 0 | 1497 |
| Zulu (`zul`) | 998 | 451 (45.2%) | 344 | 1 | 191 | 547 |
| Xhosa (`xho`) | 1002 | 3 (0.3%) | 3 | 0 | 0 | 999 |
| Shona (`sna`) | 1005 | 5 (0.5%) | 4 | 1 | 0 | 1000 |
| Chichewa (`nya`) | 1004 | 3 (0.3%) | 3 | 0 | 0 | 1001 |
| Setswana (`tsn`) | 1500 | 6 (0.4%) | 5 | 3 | 0 | 1494 |
| Luo (`luo`) | 1500 | 114 (7.6%) | 101 | 13 | 104 | 1386 |


Sensitivity of the macro chrF++ to the rule (the headline is `clean`):

| benchmark | system | all rows | clean (headline) | clean, sentence rule | clean, no near dup | reference pair not seen |
|---|---|---|---|---|---|---|
| flores_plus (full) | SFT | 30.15 (45L) | 27.01 (45L) | 29.78 (45L) | 27.01 (45L) | 29.95 (45L) |
| flores_plus (full) | base gemma-3-270m-it | 13.57 (45L) | 12.96 (45L) | 13.75 (45L) | 12.96 (45L) | 13.53 (45L) |
| ntrex (first 300 rows) | SFT | 32.84 (24L) | 30.96 (24L) | 31.61 (24L) | 30.96 (24L) | 32.78 (24L) |
| ntrex (first 300 rows) | base gemma-3-270m-it | 13.49 (24L) | 12.77 (24L) | 13.16 (24L) | 12.77 (24L) | 13.47 (24L) |
| mafand (first 300 rows) | SFT | 39.40 (15L) | 38.41 (15L) | 39.30 (15L) | 38.46 (15L) | 38.95 (15L) |
| mafand (first 300 rows) | base gemma-3-270m-it | 15.78 (15L) | 15.93 (15L) | 16.18 (15L) | 15.93 (15L) | 16.12 (15L) |


The sentence rule alone (the owner's dev-filter rule) would leave the SFT FLORES+ macro at 29.8, almost the all-rows number: the extra contiguous-run rule is what matters on FLORES+. Dropping near duplicates changes nothing measurable. Removing only rows whose reference pair was seen (29.95) is too weak a filter for FLORES+, because English-side exposure alone lifts the score.

## 4. Per-language results of the SFT model (chrF++, spBLEU, BLEU, AfriCOMET)

(n = rows, clean = English sentence not in the training text; the last two columns compare with the base on the clean rows.)

### flores_plus: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 1012 | 48 | 21.82 | 18.60 | 6.94 | 5.60 | 3.73 | 2.39 | 0.584 | 0.496 | 0.407 | 8.69 | +9.91 |
| Hausa (`hau`) | 1012 | 48 | 48.45 | 42.59 | 24.99 | 17.11 | 22.54 | 15.86 | 0.665 | 0.510 | 0.428 | 11.25 | +31.33 |
| Igbo (`ibo`) | 1012 | 48 | 39.84 | 34.98 | 19.50 | 14.95 | 16.23 | 12.91 | 0.651 | 0.547 | 0.450 | 12.28 | +22.70 |
| Twi (`twi`) | 1012 | 48 | 34.05 | 30.25 | 12.04 | 10.60 | 9.51 | 8.30 | 0.632 | 0.548 | 0.439 | 18.38 | +11.86 |
| Ewe (`ewe`) | 1012 | 48 | 28.98 | 24.44 | 8.62 | 5.71 | 7.02 | 4.37 | 0.569 | 0.497 | 0.385 | 13.40 | +11.05 |
| Fon (`fon`) | 1012 | 48 | 19.11 | 16.25 | 3.81 | 1.50 | 2.01 | 0.78 | 0.560 | 0.495 | 0.387 | 8.94 | +7.30 |
| Wolof (`wol`) | 1012 | 48 | 27.65 | 27.18 | 8.62 | 8.29 | 6.16 | 6.00 | 0.618 | 0.569 | 0.446 | 13.71 | +13.47 |
| Swahili (`swh`) | 1012 | 48 | 56.73 | 51.06 | 34.24 | 27.15 | 30.57 | 23.38 | 0.689 | 0.539 | 0.477 | 13.78 | +37.28 |
| Kinyarwanda (`kin`) | 1012 | 48 | 35.45 | 32.10 | 11.57 | 8.74 | 9.58 | 6.61 | 0.615 | 0.533 | 0.463 | 13.66 | +18.45 |
| Somali (`som`) | 1012 | 48 | 39.31 | 33.20 | 15.02 | 9.43 | 10.09 | 6.93 | 0.609 | 0.497 | 0.410 | 10.55 | +22.65 |
| Oromo (`orm`) | 1012 | 48 | 25.93 | 18.73 | 3.77 | 1.70 | 1.89 | 0.86 | 0.539 | 0.388 | 0.318 | 8.79 | +9.94 |
| Amharic (`amh`) | 1012 | 48 | 15.60 | 11.65 | 4.35 | 1.95 | 1.55 | 1.09 | 0.394 | 0.294 | 0.258 | 0.28 | +11.37 |
| Tigrinya (`tir`) | 1012 | 48 | 10.84 | 8.19 | 2.26 | 1.10 | 0.73 | 0.70 | 0.450 | 0.346 | 0.255 | 0.36 | +7.83 |
| Luganda (`lug`) | 1012 | 48 | 33.02 | 28.75 | 8.28 | 6.16 | 6.64 | 3.65 | 0.582 | 0.484 | 0.423 | 12.90 | +15.85 |
| Lingala (`lin`) | 1012 | 48 | 39.63 | 36.55 | 11.65 | 10.07 | 10.20 | 9.13 | 0.560 | 0.459 | 0.361 | 14.00 | +22.55 |
| Zulu (`zul`) | 1012 | 48 | 47.52 | 43.23 | 24.28 | 20.38 | 14.89 | 10.88 | 0.671 | 0.558 | 0.486 | 11.70 | +31.53 |
| Xhosa (`xho`) | 1012 | 48 | 44.06 | 44.36 | 19.77 | 21.21 | 11.82 | 12.62 | 0.633 | 0.560 | 0.468 | 14.23 | +30.13 |
| Afrikaans (`afr`) | 1012 | 48 | 62.66 | 59.76 | 41.93 | 37.98 | 37.25 | 35.23 | 0.729 | 0.636 | 0.561 | 24.35 | +35.41 |
| Shona (`sna`) | 1012 | 48 | 38.14 | 36.06 | 14.07 | 10.39 | 9.59 | 7.50 | 0.606 | 0.516 | 0.433 | 11.04 | +25.02 |
| Chichewa (`nya`) | 1012 | 48 | 40.85 | 34.23 | 15.24 | 9.00 | 12.05 | 7.69 | 0.639 | 0.528 | 0.454 | 15.98 | +18.26 |
| Sesotho (`sot`) | 1012 | 48 | 45.02 | 39.03 | 20.97 | 13.35 | 19.45 | 11.35 | 0.650 | 0.552 | 0.481 | 15.05 | +23.98 |
| Setswana (`tsn`) | 1012 | 48 | 44.32 | 39.23 | 20.01 | 14.14 | 17.85 | 11.93 | 0.671 | 0.570 | 0.511 | 16.03 | +23.20 |
| Malagasy (`mlg`) | 1012 | 48 | 41.87 | 38.14 | 15.38 | 11.67 | 11.44 | 10.11 | 0.571 | 0.436 | 0.376 | 12.70 | +25.43 |
| Bemba (`bem`) | 1012 | 48 | 27.98 | 26.67 | 6.44 | 6.21 | 4.63 | 4.18 | 0.501 | 0.420 | 0.338 | 13.59 | +13.08 |
| Tumbuka (`tum`) | 1012 | 48 | 28.40 | 25.75 | 7.94 | 6.25 | 6.77 | 5.24 | 0.538 | 0.436 | 0.395 | 13.80 | +11.95 |
| Kirundi (`run`) | 1012 | 48 | 31.40 | 25.59 | 8.00 | 4.37 | 5.82 | 2.35 | 0.604 | 0.494 | 0.438 | 11.81 | +13.78 |
| Tsonga (`tso`) | 1012 | 48 | 43.34 | 37.50 | 17.43 | 10.39 | 15.13 | 10.09 | 0.628 | 0.506 | 0.423 | 14.91 | +22.60 |
| Sepedi (`nso`) | 1012 | 48 | 47.94 | 43.62 | 22.86 | 18.26 | 20.87 | 16.93 | 0.670 | 0.560 | 0.478 | 15.12 | +28.51 |
| Swati (`ssw`) | 1012 | 48 | 34.77 | 32.73 | 10.83 | 8.94 | 5.70 | 4.46 | 0.571 | 0.475 | 0.412 | 15.47 | +17.26 |
| Sango (`sag`) | 1012 | 48 | 31.28 | 30.02 | 6.68 | 5.50 | 5.51 | 4.47 | - | - | - | 16.09 | +13.94 |
| Umbundu (`umb`) | 1012 | 48 | 19.93 | 19.42 | 1.80 | 1.38 | 1.09 | 0.65 | 0.498 | 0.440 | 0.383 | 12.52 | +6.90 |
| Kimbundu (`kmb`) | 1012 | 48 | 20.05 | 19.04 | 2.29 | 1.77 | 1.42 | 1.16 | 0.415 | 0.355 | 0.334 | 12.07 | +6.97 |
| Mossi (`mos`) | 1012 | 48 | 16.95 | 16.70 | 2.53 | 2.40 | 1.74 | 1.67 | 0.485 | 0.438 | 0.377 | 13.07 | +3.64 |
| Kikuyu (`kik`) | 1012 | 48 | 24.81 | 20.95 | 5.41 | 3.62 | 3.72 | 2.15 | 0.544 | 0.461 | 0.320 | 15.69 | +5.26 |
| Luo (`luo`) | 1012 | 48 | 22.84 | 19.95 | 3.71 | 3.06 | 2.54 | 1.85 | 0.546 | 0.462 | 0.382 | 13.91 | +6.04 |
| Kamba (`kam`) | 1012 | 48 | 16.36 | 13.64 | 2.16 | 1.20 | 1.70 | 0.72 | 0.387 | 0.346 | 0.269 | 15.23 | -1.59 |
| Dyula (`dyu`) | 1012 | 48 | 13.34 | 12.77 | 1.92 | 1.47 | 0.95 | 0.58 | - | - | - | 13.27 | -0.50 |
| Bambara (`bam`) | 1012 | 48 | 19.10 | 17.86 | 3.10 | 3.09 | 2.17 | 2.06 | 0.373 | 0.329 | 0.214 | 13.47 | +4.39 |
| Kabiye (`kbp`) | 1012 | 48 | 9.75 | 8.37 | 0.78 | 0.56 | 0.47 | 0.38 | 0.373 | 0.344 | 0.179 | 9.29 | -0.92 |
| Kabuverdianu (`kea`) | 1012 | 48 | 16.84 | 11.85 | 2.21 | 0.69 | 1.58 | 0.56 | - | - | - | 16.66 | -4.80 |
| Kanuri (`knc`) | 1012 | 48 | 5.47 | 5.09 | 0.42 | 0.37 | 0.21 | 0.09 | - | - | - | 14.43 | -9.35 |
| Kabyle (`kab`) | 1012 | 48 | 22.56 | 22.03 | 4.79 | 4.44 | 3.19 | 3.23 | 0.615 | 0.587 | 0.357 | 13.13 | +8.90 |
| Nuer (`nus`) | 1012 | 48 | 12.62 | 13.15 | 1.86 | 2.28 | 0.78 | 0.66 | 0.368 | 0.373 | 0.164 | 8.91 | +4.24 |
| Luba-Kasai (`lua`) | 1012 | 48 | 29.11 | 26.10 | 4.91 | 2.49 | 3.49 | 0.78 | 0.519 | 0.417 | 0.370 | 15.05 | +11.05 |
| Makhuwa (`vmw`) | 1012 | 48 | 20.93 | 18.13 | 2.80 | 1.81 | 1.79 | 0.72 | - | - | - | 13.48 | +4.65 |

### ntrex: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 300 | 181 | 23.63 | 21.60 | 8.68 | 7.52 | 4.33 | 3.33 | 0.604 | 0.570 | 0.456 | 8.37 | +13.23 |
| Hausa (`hau`) | 300 | 181 | 45.14 | 42.19 | 21.25 | 18.12 | 19.32 | 16.04 | 0.631 | 0.590 | 0.536 | 11.61 | +30.59 |
| Igbo (`ibo`) | 300 | 181 | 41.40 | 38.29 | 17.69 | 14.20 | 15.50 | 12.32 | 0.639 | 0.610 | 0.507 | 12.72 | +25.57 |
| Ewe (`ewe`) | 300 | 181 | 30.20 | 28.67 | 8.85 | 7.94 | 6.71 | 5.39 | 0.554 | 0.542 | 0.446 | 16.22 | +12.45 |
| Wolof (`wol`) | 300 | 181 | 27.04 | 25.26 | 7.59 | 6.17 | 5.16 | 3.97 | 0.588 | 0.576 | 0.477 | 13.86 | +11.40 |
| Swahili (`swh`) | 300 | 181 | 52.74 | 50.17 | 31.59 | 28.52 | 27.76 | 24.10 | 0.629 | 0.597 | 0.558 | 15.73 | +34.44 |
| Kinyarwanda (`kin`) | 300 | 181 | 32.76 | 31.00 | 10.34 | 9.37 | 8.58 | 7.25 | 0.573 | 0.561 | 0.500 | 15.11 | +15.89 |
| Somali (`som`) | 300 | 181 | 41.57 | 39.86 | 17.29 | 15.33 | 11.87 | 10.46 | 0.597 | 0.572 | 0.491 | 11.67 | +28.20 |
| Oromo (`orm`) | 300 | 181 | 23.14 | 22.36 | 2.46 | 2.05 | 1.45 | 1.11 | 0.515 | 0.502 | 0.459 | 9.48 | +12.88 |
| Amharic (`amh`) | 300 | 181 | 11.15 | 10.07 | 2.84 | 1.91 | 0.53 | 0.24 | 0.331 | 0.309 | 0.296 | 0.38 | +9.70 |
| Tigrinya (`tir`) | 300 | 181 | 8.90 | 7.84 | 1.50 | 1.01 | 0.39 | 0.30 | 0.404 | 0.392 | 0.315 | 1.03 | +6.80 |
| Zulu (`zul`) | 300 | 181 | 35.07 | 33.31 | 9.62 | 8.53 | 3.69 | 2.93 | 0.567 | 0.540 | 0.519 | 13.43 | +19.88 |
| Xhosa (`xho`) | 300 | 181 | 38.33 | 36.42 | 15.83 | 14.36 | 7.70 | 6.79 | 0.580 | 0.549 | 0.502 | 15.22 | +21.21 |
| Afrikaans (`afr`) | 300 | 181 | 59.62 | 58.01 | 37.99 | 36.39 | 33.19 | 30.59 | 0.648 | 0.624 | 0.571 | 25.29 | +32.71 |
| Shona (`sna`) | 300 | 181 | 36.32 | 34.93 | 14.00 | 13.03 | 8.95 | 7.61 | 0.566 | 0.545 | 0.493 | 11.69 | +23.25 |
| Chichewa (`nya`) | 300 | 181 | 44.13 | 40.87 | 21.07 | 17.37 | 17.05 | 13.61 | 0.601 | 0.572 | 0.513 | 15.77 | +25.10 |
| Setswana (`tsn`) | 300 | 181 | 46.14 | 43.67 | 26.11 | 23.24 | 24.19 | 21.12 | 0.646 | 0.621 | 0.560 | 16.53 | +27.14 |
| Malagasy (`mlg`) | 300 | 181 | 37.06 | 35.53 | 13.23 | 11.46 | 8.86 | 7.57 | 0.513 | 0.483 | 0.475 | 13.48 | +22.06 |
| Bemba (`bem`) | 300 | 181 | 27.01 | 24.33 | 6.01 | 4.22 | 4.36 | 2.78 | 0.473 | 0.445 | 0.399 | 16.24 | +8.09 |
| Venda (`ven`) | 300 | 181 | 34.12 | 31.50 | 10.65 | 8.82 | 7.33 | 5.97 | - | - | - | 14.20 | +17.30 |
| Sepedi (`nso`) | 300 | 181 | 30.40 | 28.56 | 9.46 | 7.94 | 6.66 | 5.11 | 0.559 | 0.522 | 0.518 | 17.25 | +11.32 |
| Swati (`ssw`) | 300 | 181 | 32.67 | 30.47 | 9.77 | 8.07 | 3.35 | 2.68 | 0.532 | 0.495 | 0.439 | 16.04 | +14.43 |
| North Ndebele (`nde`) | 300 | 181 | 29.20 | 27.95 | 8.05 | 6.78 | 3.01 | 2.43 | - | - | - | 14.63 | +13.31 |
| Tachelhit (`shi`) | 300 | 181 | 0.31 | 0.27 | 0.01 | 0.01 | 0.03 | 0.01 | - | - | - | 0.61 | -0.34 |


### mafand: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 300 | 21 | 26.30 | 18.84 | 12.21 | 7.22 | 7.80 | 4.98 | 0.623 | 0.567 | 0.471 | 6.98 | +11.86 |
| Hausa (`hau`) | 300 | 210 | 32.95 | 31.92 | 10.30 | 8.88 | 6.71 | 6.17 | 0.652 | 0.636 | 0.704 | 8.39 | +23.53 |
| Igbo (`ibo`) | 300 | 32 | 49.71 | 45.44 | 29.57 | 16.20 | 21.36 | 4.30 | 0.732 | 0.719 | 0.713 | 19.90 | +25.54 |
| Pidgin (`pcm`) | 300 | 246 | 47.22 | 46.83 | 24.36 | 24.32 | 18.83 | 18.67 | 0.612 | 0.603 | 0.390 | 36.27 | +10.56 |
| Twi (`twi`) | 300 | 300 | 34.64 | 34.64 | 12.95 | 12.95 | 7.73 | 7.73 | 0.628 | 0.628 | 0.558 | 18.71 | +15.93 |
| Swahili (`swh`) | 300 | 279 | 50.52 | 50.35 | 28.57 | 28.11 | 24.99 | 24.80 | 0.663 | 0.659 | 0.658 | 16.14 | +34.20 |
| Kinyarwanda (`kin`) | 300 | 298 | 43.70 | 43.73 | 18.15 | 18.16 | 14.62 | 14.64 | 0.658 | 0.658 | 0.619 | 16.11 | +27.62 |
| Amharic (`amh`) | 300 | 223 | 13.07 | 12.92 | 3.97 | 3.84 | 0.94 | 0.89 | 0.397 | 0.394 | 0.447 | 1.25 | +11.68 |
| Luganda (`lug`) | 300 | 298 | 44.39 | 44.31 | 20.39 | 20.36 | 15.67 | 15.60 | 0.641 | 0.641 | 0.630 | 17.80 | +26.52 |
| Zulu (`zul`) | 300 | 161 | 45.35 | 43.75 | 23.09 | 21.74 | 15.51 | 13.63 | 0.617 | 0.595 | 0.549 | 15.00 | +28.75 |
| Xhosa (`xho`) | 300 | 300 | 49.04 | 49.04 | 27.03 | 27.03 | 15.03 | 15.03 | 0.655 | 0.655 | 0.611 | 15.60 | +33.45 |
| Shona (`sna`) | 300 | 300 | 33.55 | 33.55 | 10.91 | 10.91 | 6.73 | 6.73 | 0.611 | 0.611 | 0.623 | 12.70 | +20.85 |
| Chichewa (`nya`) | 300 | 300 | 41.51 | 41.51 | 14.74 | 14.74 | 10.80 | 10.80 | 0.635 | 0.635 | 0.617 | 18.45 | +23.06 |
| Setswana (`tsn`) | 300 | 299 | 50.79 | 50.81 | 26.83 | 26.84 | 22.39 | 22.40 | 0.697 | 0.697 | 0.653 | 18.62 | +32.19 |
| Luo (`luo`) | 300 | 277 | 28.24 | 28.44 | 8.21 | 8.48 | 5.25 | 5.35 | 0.594 | 0.595 | 0.566 | 17.03 | +11.41 |


## 5. Output format failures (all rows, weighted by rows; per-language values are in the `report.md` files)

| benchmark | system | prefix_ok | finished (end token) | empty |
|---|---|---|---|---|
| flores_plus | SFT | 97.7% | 89.5% | 0.0% |
| flores_plus | base gemma-3-270m-it | 0.0% | 95.7% | 0.0% |


| benchmark | system | prefix_ok | finished (end token) | empty |
|---|---|---|---|---|
| ntrex | SFT | 95.8% | 92.3% | 0.0% |
| ntrex | base gemma-3-270m-it | 0.0% | 94.9% | 0.0% |
| mafand | SFT | 100.0% | 97.8% | 0.0% |
| mafand | base gemma-3-270m-it | 0.0% | 92.4% | 0.0% |


SFT, lowest `prefix_ok` (completion starts with the expected tag): FLORES+ `knc` (0%: no tag at all, the output loops), NTREX **`nde` 0%**. The NTREX `nde` reference is North Ndebele (Zimbabwe); the model answers `nbl <South Ndebele text>` for the North Ndebele prompt (the two languages are confused; chrF++ is still scored on the text). Lowest `finished` (no end token inside 320 new tokens; looping): FLORES+ `kbp` 24%, `knc` 28%, `nus` 51%, `kam` 72%; NTREX `shi` 37% (Tachelhit, garbage loops), `tir` 70%, `amh` 85%, `orm` 87%. Empty outputs: none for either system. The base model almost never writes the tag (prefix_ok 0%), finishes 92-96% of its outputs, and mostly does not translate.

SFT, lowest rates by benchmark (rows of the language):

prefix_ok:
- flores_plus: `knc` 0.0%, `hau` 99.8%, `kik` 99.8%, `sag` 99.8%, `wol` 99.8%, `afr` 99.9%
- ntrex: `nde` 0.0%, `mlg` 99.7%, `sna` 99.7%, `afr` 100.0%, `amh` 100.0%, `bem` 100.0%
- mafand: `pcm` 99.7%, `yor` 99.7%, `amh` 100.0%, `hau` 100.0%, `ibo` 100.0%, `kin` 100.0%
finished:
- flores_plus: `kbp` 24.4%, `knc` 27.6%, `nus` 50.6%, `kam` 72.2%, `kea` 75.4%, `tir` 79.7%
- ntrex: `shi` 37.0%, `tir` 69.7%, `amh` 85.0%, `orm` 87.3%, `ven` 88.0%, `bem` 90.7%
- mafand: `amh` 90.3%, `luo` 95.0%, `kin` 96.0%, `yor` 97.0%, `tsn` 98.0%, `lug` 98.3%

## 6. Runtime

| run | what | wall time |
|---|---|---|
| FLORES+ full, SFT, HF sampler, old RTX 5070 shared with foreign jobs at 100% | 45,540 rows generation (45 units) | 5,571 s generation (about 8 rows/s) |
| same, base | 45,540 rows | 4,617 s (about 10-25 rows/s as the foreign load changed) |
| AfriCOMET, both models, both systems, old box, batch 64 one model at a time | 40,480 pairs x 2 models x 2 systems | about 5 min per system (about 270 pairs/s for the base system's 80,960 scorings, 301 s) |
| NTREX + MAFAND first 300 rows, both systems | 11,700 rows x 2 | 935 + 493 s (SFT), 965 + 627 s (base) generation, 50-150 s scoring per benchmark |

All runs went through the resume loop (`BUDGET=840` with `flock`, each locked job <15 min), so wall-clock totals include waits for GPU memory held by foreign jobs. The per-system `scoring_seconds` in `results.json` cover only the last invocation. No part of this ran on the rental GPU.

## 7. How the pipeline is invoked

On the rental, after the SFT run has ended and before RL. Files: `/workspace/sft_build/gemma_mt_eval_candidate/` (code, NOT the live `gemma_mt`), `/workspace/sft_build/bench/contamination_{flores_plus,ntrex,mafand,dev}.json` and `/workspace/sft_build/bench/test_sets/` (test sets, 35 MB; the script re-downloads anything missing).

```bash
cd /workspace/sft_build && nohup bash gemma_mt_eval_candidate/scripts/eval_sft_benchmarks.sh > eval/sft_best.launch.log 2>&1 &
# wait for  /workspace/sft_build/eval/sft_best/EVAL_DONE  (or EVAL_FAILED with the reasons inside)
```

Defaults: SFT `<HF_ORG_LEGACY>/gemma3-270m-translate` main, base `google/gemma-3-270m-it`, vLLM worker at `gpu_memory_utilization 0.2` (one engine at a time; AfriCOMET workers about 3 GB each after it is closed), `python /workspace/venv_gemma/bin/python`, vLLM `/workspace/venv_vllm/bin/python`, AfriCOMET child `pylibs_tf4:pylibs`, caches under `/workspace/sft_build/vllm_cache`. Environment variables (see the header of the script): `SFT_REVISION` (pin a commit), `SAMPLER=hf`, `SAMPLER_FALLBACK=0`, `OUT`, `TOTAL_TIMEOUT` (14,400 s), `GEN_TIMEOUT`/`SCORE_TIMEOUT` (5,400 s per benchmark), `GPU_WAIT` (1,800 s), `GPU_MIN_FREE_GIB` (24), `GPU_LOCK_CMD`, `BUDGET`. Outputs in `OUT=/workspace/sft_build/eval/sft_best`: `results.json`, `results.csv`, `predictions.jsonl`, `report.md`, `eval.log`, `work/` (resume caches), and `EVAL_DONE` or `EVAL_FAILED`. A rerun resumes from `work/`; `rm -r OUT/work` starts from scratch. Safety: a total watchdog kills the process, the shell `timeout` kills the tree, leftovers of the run are found through the `GEMMA_EVAL_RUN_ID` environment marker (the trainer's processes are never matched), the vLLM worker exits on EOF.

## 8. Open problems and caveats

1. **The vLLM path has not been run on a GPU.** The old box has no vLLM and the rental GPU was off limits. Covered by CPU tests: the engine protocol through the real `WorkerEngine` worker (HF-backed stand-in), chunking, deadlines, caches, markers. Not covered: vLLM's start-up from a Hub snapshot directory (the RL smoke test used the same kind of directory) and its speed. If the worker cannot start, the script falls back to HF generate (`SAMPLER_FALLBACK=1`, recorded as a warning in `EVAL_DONE`), which is slower (about 1 h of generation on the idle 96 GB card).
2. NTREX and MAFAND numbers here use the first 300 rows per language; FLORES+ is complete. The rental run covers every row.
3. FLORES+ clean = 48 sentences per language: per-language clean values are noisy (ranking of close languages is not reliable); use macros.
4. Twi: only the Akuapem FLORES+ file is used. `nde` is scored although the model confuses it with `nbl`. `efi` and ~30 other target languages have no benchmark.
5. `max_new_tokens` 320 on the old box vs 768 in the script default: affects only the rows that loop (low `finished` rate) and slightly lowers their chrF++; the rental run uses 768.
6. The brief's reference point (step 17,500, 1.3363) was not re-evaluated; the evaluated commit is step 20500 (macro validation loss 1.3025).
