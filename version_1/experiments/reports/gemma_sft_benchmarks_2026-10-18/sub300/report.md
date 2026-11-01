# SFT benchmark evaluation (English -> African)

* SFT model: `<HF_ORG_LEGACY>/gemma3-270m-translate` revision `ae0ea640e7f3ff8acb6d97eb7538246d6b786a52` commit `ae0ea640e7f3ff8acb6d97eb7538246d6b786a52` (training step 20500, macro validation loss 1.3025)
* base model: `google/gemma-3-270m-it`
* decoding: greedy, same prompt for both (`Translate the text to {language}:  {english}`), max_new_tokens 320, sampler hf; metrics on NFC-normalised text; spBLEU tokenizer flores200
* started 2026-10-19 05:10:21, finished 2026-10-19 05:19:52, 571 s; environment {"python": "3.12.14", "platform": "Linux-6.8.0-136-generic-x86_64-with-glibc2.39", "host": "043978645c01", "torch": "2.14.0+cu132", "transformers": "5.19.0", "huggingface_hub": "1.33.0", "gpu": "NVIDIA GeForce RTX 5070, 12227 MiB"}

## Results

### Macro averages, clean (headline: English sentence not in the training text)

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|
| **ntrex** | SFT | 24 | 4344 | 30.96 (24) | 11.35 (24) | 8.07 (24) | 0.534 (21) | 0.478 (21) |
| ntrex | base gemma-3-270m-it | 24 | 4344 | 12.77 (24) | 1.88 (24) | 0.92 (24) | 0.238 (21) | 0.379 (21) |
| **mafand** | SFT | 15 | 3544 | 38.41 (15) | 16.65 (15) | 11.45 (15) | 0.620 (15) | 0.587 (15) |
| mafand | base gemma-3-270m-it | 15 | 3544 | 15.93 (15) | 3.06 (15) | 1.73 (15) | 0.283 (15) | 0.428 (15) |

### Macro averages, all rows

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|
| **ntrex** | SFT | 24 | 7200 | 32.84 (24) | 13.00 (24) | 9.58 (24) | 0.560 (21) | 0.508 (21) |
| ntrex | base gemma-3-270m-it | 24 | 7200 | 13.49 (24) | 2.21 (24) | 1.23 (24) | 0.250 (21) | 0.394 (21) |
| **mafand** | SFT | 15 | 4500 | 39.40 (15) | 18.08 (15) | 12.96 (15) | 0.628 (15) | 0.595 (15) |
| mafand | base gemma-3-270m-it | 15 | 4500 | 15.78 (15) | 3.11 (15) | 1.79 (15) | 0.276 (15) | 0.425 (15) |

### Macro over every (benchmark, language) pair

| system | chrF++ clean (n) | chrF++ all rows (n) | AfriCOMET clean | AfriCOMET all |
|---|---|---|---|---|
| SFT | 33.83 (39) | 35.36 (39) | 0.570 | 0.588 |
| base gemma-3-270m-it | 13.99 (39) | 14.37 (39) | 0.257 | 0.261 |

## Contamination of the benchmarks by the SFT training text

A row is *contaminated* when the normalised English sentence (NFKC, case-folded, punctuation removed) of the benchmark row occurs in the SFT training windows (`gemma/windows_sft`, the input of the packed training set): as a whole sentence of a window or as a contiguous run inside one. *sentence rule* = the first half only (the filter that produced `bench_dev_clean.jsonl`; it reproduces that file exactly: 10,043 of 30,418 dev rows). *near* = not contaminated but >= 50% of its 8-token n-grams occur in the training English. *pair* = contaminated and the reference translation occurs in the target side of a window of the same language (the model saw the very pair).

| benchmark | rows | contaminated | of which sentence-rule hits | near | pair seen | clean rows |
|---|---|---|---|---|---|---|
| ntrex | 7200 | 2856 (39.7%) | 2424 (33.7%) | 0 (0.0%) | 68 (0.9%) | 4344 (60.3%) |
| mafand | 4500 | 956 (21.2%) | 860 (19.1%) | 16 (0.4%) | 640 (14.2%) | 3544 (78.8%) |

**ntrex**: the English side is the same for every language (FLORES+/NTREX are multi-way parallel), so every language has the same 119 of 300 rows contaminated (39.7%); rows whose reference pair was seen, per language: hau 57, yor 11.

**mafand** per language:

| language | rows | contaminated | sentence rule | near | pair | clean |
|---|---|---|---|---|---|---|
| Amharic (`amh`) | 300 | 77 (25.7%) | 73 | 0 | 50 | 223 |
| Hausa (`hau`) | 300 | 90 (30.0%) | 70 | 5 | 49 | 210 |
| Igbo (`ibo`) | 300 | 268 (89.3%) | 258 | 0 | 265 | 32 |
| Kinyarwanda (`kin`) | 300 | 2 (0.7%) | 2 | 0 | 0 | 298 |
| Luganda (`lug`) | 300 | 2 (0.7%) | 2 | 0 | 0 | 298 |
| Luo (`luo`) | 300 | 23 (7.7%) | 21 | 1 | 20 | 277 |
| Chichewa (`nya`) | 300 | 0 (0.0%) | 0 | 0 | 0 | 300 |
| Pidgin (`pcm`) | 300 | 54 (18.0%) | 52 | 1 | 46 | 246 |
| Shona (`sna`) | 300 | 0 (0.0%) | 0 | 1 | 0 | 300 |
| Swahili (`swh`) | 300 | 21 (7.0%) | 21 | 7 | 18 | 279 |
| Setswana (`tsn`) | 300 | 1 (0.3%) | 1 | 0 | 0 | 299 |
| Twi (`twi`) | 300 | 0 (0.0%) | 0 | 0 | 0 | 300 |
| Xhosa (`xho`) | 300 | 0 (0.0%) | 0 | 0 | 0 | 300 |
| Yoruba (`yor`) | 300 | 279 (93.0%) | 253 | 1 | 131 | 21 |
| Zulu (`zul`) | 300 | 139 (46.3%) | 107 | 0 | 61 | 161 |

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

## Output format failures

`prefix_ok` = the completion starts with the expected language tag (the base model does not know the convention: its whole output is scored, the tag stripped only if present); `finished` = `<end_of_turn>` produced inside the token budget; `empty` = nothing left after stripping. All rows, unweighted mean over languages; the worst languages follow.

| benchmark | system | prefix_ok | finished | empty |
|---|---|---|---|---|
| ntrex | SFT | 95.8% | 92.3% | 0.0% |
| ntrex | base gemma-3-270m-it | 0.0% | 94.9% | 0.0% |
| mafand | SFT | 100.0% | 97.8% | 0.0% |
| mafand | base gemma-3-270m-it | 0.0% | 92.4% | 0.0% |

- ntrex, SFT, lowest prefix_ok: nde 0.0%, mlg 99.7%, sna 99.7%, afr 100.0%
- ntrex, SFT, lowest finished: shi 37.0%, tir 69.7%, amh 85.0%, orm 87.3%
- ntrex, SFT, highest empty: afr 0.0%, amh 0.0%, bem 0.0%, ewe 0.0%
- mafand, SFT, lowest prefix_ok: pcm 99.7%, yor 99.7%, amh 100.0%, hau 100.0%
- mafand, SFT, lowest finished: amh 90.3%, luo 95.0%, kin 96.0%, yor 97.0%
- mafand, SFT, highest empty: amh 0.0%, hau 0.0%, ibo 0.0%, kin 0.0%

## How the contamination rule changes the macro chrF++

| benchmark | system | all | clean | clean, sentence rule | clean, no near | pair not seen |
|---|---|---|---|---|---|---|
| ntrex | SFT | 32.84 (24L) | 30.96 (24L) | 31.61 (24L) | 30.96 (24L) | 32.78 (24L) |
| ntrex | base gemma-3-270m-it | 13.49 (24L) | 12.77 (24L) | 13.16 (24L) | 12.77 (24L) | 13.47 (24L) |
| mafand | SFT | 39.40 (15L) | 38.41 (15L) | 39.30 (15L) | 38.46 (15L) | 38.95 (15L) |
| mafand | base gemma-3-270m-it | 15.78 (15L) | 15.93 (15L) | 16.18 (15L) | 15.93 (15L) | 16.12 (15L) |

## Runtime

| system | benchmark | rows | sampler | generation s | scoring s (incl. AfriCOMET) |
|---|---|---|---|---|---|
| SFT | ntrex | 7200 | hf | 935 | 50 |
| SFT | mafand | 4500 | hf | 493 | 44 |
| base gemma-3-270m-it | ntrex | 7200 | hf | 965 | 152 |
| base gemma-3-270m-it | mafand | 4500 | hf | 627 | 112 |

Whole run: 571 s.
