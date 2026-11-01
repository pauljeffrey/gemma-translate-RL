# SFT benchmark evaluation (English -> African)

* SFT model: `<HF_ORG_LEGACY>/gemma3-270m-translate` revision `ae0ea640e7f3ff8acb6d97eb7538246d6b786a52` commit `ae0ea640e7f3ff8acb6d97eb7538246d6b786a52` (training step 20500, macro validation loss 1.3025)
* base model: `google/gemma-3-270m-it`
* decoding: greedy, same prompt for both (`Translate the text to {language}:  {english}`), max_new_tokens 320, sampler hf; metrics on NFC-normalised text; spBLEU tokenizer flores200
* started 2026-10-19 03:02:41, finished 2026-10-19 03:09:30, 409 s; environment {"python": "3.12.14", "platform": "Linux-6.8.0-136-generic-x86_64-with-glibc2.39", "host": "043978645c01", "torch": "2.14.0+cu132", "transformers": "5.19.0", "huggingface_hub": "1.33.0", "gpu": "NVIDIA GeForce RTX 5070, 12227 MiB"}

## Results

### Macro averages, clean (headline: English sentence not in the training text)

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|
| **flores_plus** | SFT | 45 | 2160 | 27.01 (45) | 7.97 (45) | 6.12 (45) | 0.475 (40) | 0.388 (40) |
| flores_plus | base gemma-3-270m-it | 45 | 2160 | 12.96 (45) | 1.56 (45) | 0.77 (45) | 0.208 (40) | 0.353 (40) |

### Macro averages, all rows

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|
| **flores_plus** | SFT | 45 | 45540 | 30.15 (45) | 10.40 (45) | 8.09 (45) | 0.563 (40) | 0.524 (40) |
| flores_plus | base gemma-3-270m-it | 45 | 45540 | 13.57 (45) | 1.98 (45) | 1.21 (45) | 0.246 (40) | 0.429 (40) |

### Macro over every (benchmark, language) pair

| system | chrF++ clean (n) | chrF++ all rows (n) | AfriCOMET clean | AfriCOMET all |
|---|---|---|---|---|
| SFT | 27.01 (45) | 30.15 (45) | 0.475 | 0.563 |
| base gemma-3-270m-it | 12.96 (45) | 13.57 (45) | 0.208 | 0.246 |

## Contamination of the benchmarks by the SFT training text

A row is *contaminated* when the normalised English sentence (NFKC, case-folded, punctuation removed) of the benchmark row occurs in the SFT training windows (`gemma/windows_sft`, the input of the packed training set): as a whole sentence of a window or as a contiguous run inside one. *sentence rule* = the first half only (the filter that produced `bench_dev_clean.jsonl`; it reproduces that file exactly: 10,043 of 30,418 dev rows). *near* = not contaminated but >= 50% of its 8-token n-grams occur in the training English. *pair* = contaminated and the reference translation occurs in the target side of a window of the same language (the model saw the very pair).

| benchmark | rows | contaminated | of which sentence-rule hits | near | pair seen | clean rows |
|---|---|---|---|---|---|---|
| flores_plus | 45540 | 43380 (95.3%) | 38250 (84.0%) | 0 (0.0%) | 1848 (4.1%) | 2160 (4.7%) |

**flores_plus**: the English side is the same for every language (FLORES+/NTREX are multi-way parallel), so every language has the same 964 of 1012 rows contaminated (95.3%); rows whose reference pair was seen, per language: hau 842, ibo 559, lug 2, orm 24, tsn 30, yor 355, zul 36.

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

## Output format failures

`prefix_ok` = the completion starts with the expected language tag (the base model does not know the convention: its whole output is scored, the tag stripped only if present); `finished` = `<end_of_turn>` produced inside the token budget; `empty` = nothing left after stripping. All rows, unweighted mean over languages; the worst languages follow.

| benchmark | system | prefix_ok | finished | empty |
|---|---|---|---|---|
| flores_plus | SFT | 97.7% | 89.5% | 0.0% |
| flores_plus | base gemma-3-270m-it | 0.0% | 95.7% | 0.0% |

- flores_plus, SFT, lowest prefix_ok: knc 0.0%, hau 99.8%, kik 99.8%, sag 99.8%
- flores_plus, SFT, lowest finished: kbp 24.4%, knc 27.6%, nus 50.6%, kam 72.2%
- flores_plus, SFT, highest empty: afr 0.0%, amh 0.0%, bam 0.0%, bem 0.0%

## How the contamination rule changes the macro chrF++

| benchmark | system | all | clean | clean, sentence rule | clean, no near | pair not seen |
|---|---|---|---|---|---|---|
| flores_plus | SFT | 30.15 (45L) | 27.01 (45L) | 29.78 (45L) | 27.01 (45L) | 29.95 (45L) |
| flores_plus | base gemma-3-270m-it | 13.57 (45L) | 12.96 (45L) | 13.75 (45L) | 12.96 (45L) | 13.53 (45L) |

## Runtime

| system | benchmark | rows | sampler | generation s | scoring s (incl. AfriCOMET) |
|---|---|---|---|---|---|
| SFT | flores_plus | 45540 | hf | 5571 | 96 |
| base gemma-3-270m-it | flores_plus | 45540 | hf | 4617 | 301 |

Whole run: 409 s.
