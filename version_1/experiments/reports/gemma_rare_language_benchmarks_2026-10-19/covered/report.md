# SFT benchmark evaluation (English -> African)

* SFT model: `<HF_ORG>/gemma3-270m-translate` revision `376a94f2314da6445878962fe22d59900ee2d2b9` commit `376a94f2314da6445878962fe22d59900ee2d2b9` (training step 27638, macro validation loss 1.2654)
* base model: `google/gemma-3-270m-it`
* decoding: greedy, same prompt for both (`Translate the text to {language}:  {english}`), max_new_tokens 320, sampler hf; metrics on NFC-normalised text; spBLEU tokenizer flores200
* started 2026-10-19 15:23:15, finished 2026-10-19 15:49:46, 1591 s; environment {"python": "3.12.14", "platform": "Linux-6.8.0-136-generic-x86_64-with-glibc2.39", "host": "043978645c01", "torch": "2.14.0+cu132", "transformers": "5.19.0", "huggingface_hub": "1.33.0", "gpu": "NVIDIA GeForce RTX 5070, 12227 MiB"}

## Results

### Macro averages, clean (headline: English sentence not in the training text)

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) | exact match (n lang) |
|---|---|---|---|---|---|---|---|---|---|
| **smol_sent** | SFT | 37 | 2812 | 24.11 (37) | 5.85 (37) | 4.19 (37) | 0.445 (35) | 0.340 (35) | - |
| smol_sent | base gemma-3-270m-it | 37 | 2812 | 11.01 (37) | 0.62 (37) | 0.42 (37) | 0.206 (35) | 0.364 (35) | - |
| **smol_doc** | SFT | 37 | 1282 | 29.55 (37) | 9.74 (37) | 8.02 (37) | - | - | - |
| smol_doc | base gemma-3-270m-it | 37 | 1282 | 14.98 (37) | 2.23 (37) | 1.97 (37) | - | - | - |
| **gatitos** | SFT | 26 | 29453 | 10.40 (26) | - | - | - | - | 0.022 (26) |
| gatitos | base gemma-3-270m-it | 26 | 29453 | 5.55 (26) | - | - | - | - | 0.002 (26) |

### Macro averages, all rows

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) | exact match (n lang) |
|---|---|---|---|---|---|---|---|---|---|
| **smol_sent** | SFT | 37 | 18500 | 29.77 (37) | 9.41 (37) | 7.17 (37) | 0.554 (35) | 0.493 (35) | - |
| smol_sent | base gemma-3-270m-it | 37 | 18500 | 10.62 (37) | 0.45 (37) | 0.32 (37) | 0.231 (35) | 0.428 (35) | - |
| **smol_doc** | SFT | 37 | 1480 | 30.98 (37) | 10.44 (37) | 8.59 (37) | - | - | - |
| smol_doc | base gemma-3-270m-it | 37 | 1480 | 15.13 (37) | 2.34 (37) | 2.01 (37) | - | - | - |
| **gatitos** | SFT | 26 | 103755 | 13.92 (26) | - | - | - | - | 0.042 (26) |
| gatitos | base gemma-3-270m-it | 26 | 103755 | 5.36 (26) | - | - | - | - | 0.001 (26) |

### Macro over every (benchmark, language) pair

| system | chrF++ clean (n) | chrF++ all rows (n) | AfriCOMET clean | AfriCOMET all |
|---|---|---|---|---|
| SFT | 22.56 (100) | 26.10 (100) | 0.445 | 0.554 |
| base gemma-3-270m-it | 11.06 (100) | 10.92 (100) | 0.206 | 0.231 |

## Contamination of the benchmarks by the SFT training text

A row is *contaminated* when the normalised English sentence (NFKC, case-folded, punctuation removed) of the benchmark row occurs in the SFT training windows (`gemma/windows_sft`, the input of the packed training set): as a whole sentence of a window or as a contiguous run inside one. *sentence rule* = the first half only (the filter that produced `bench_dev_clean.jsonl`; it reproduces that file exactly: 10,043 of 30,418 dev rows). *near* = not contaminated but >= 50% of its 8-token n-grams occur in the training English. *pair* = contaminated and the reference translation occurs in the target side of a window of the same language (the model saw the very pair).

| benchmark | rows | contaminated | of which sentence-rule hits | near | pair seen | clean rows |
|---|---|---|---|---|---|---|
| smol_sent | 18500 | 15688 (84.8%) | 15540 (84.0%) | 0 (0.0%) | 1146 (6.2%) | 2812 (15.2%) |
| smol_doc | 1480 | 198 (13.4%) | 0 (0.0%) | 22 (1.5%) | 0 (0.0%) | 1282 (86.6%) |
| gatitos | 103755 | 74302 (71.6%) | 74302 (71.6%) | 0 (0.0%) | 2181 (2.1%) | 29453 (28.4%) |

**smol_sent**: the English side is the same for every language (multi-way parallel set: FLORES+, NTREX, SmolSent, IbomMT, FLORES-MOZ), so every language has the same 424 of 500 rows contaminated (84.8%); rows whose reference pair was seen, per language: bam 302, ewe 287, fon 1, hau 36, ibo 13, orm 42, wol 333, yor 88, zul 44.

**smol_doc** per language:

| language | rows | contaminated | sentence rule | near | pair | clean |
|---|---|---|---|---|---|---|
| Afrikaans (`afr`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Amharic (`amh`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Bambara (`bam`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Bemba (`bem`) | 40 | 18 (45.0%) | 0 | 2 | 0 | 22 |
| Dyula (`dyu`) | 40 | 18 (45.0%) | 0 | 2 | 0 | 22 |
| Ewe (`ewe`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Fon (`fon`) | 40 | 18 (45.0%) | 0 | 2 | 0 | 22 |
| Hausa (`hau`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Igbo (`ibo`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Kabyle (`kab`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Kikuyu (`kik`) | 40 | 18 (45.0%) | 0 | 2 | 0 | 22 |
| Kinyarwanda (`kin`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Lingala (`lin`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Luganda (`lug`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Luo (`luo`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Malagasy (`mlg`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Mossi (`mos`) | 40 | 18 (45.0%) | 0 | 2 | 0 | 22 |
| Sepedi (`nso`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Chichewa (`nya`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Oromo (`orm`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Pidgin (`pcm`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Kirundi (`run`) | 40 | 18 (45.0%) | 0 | 2 | 0 | 22 |
| Shona (`sna`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Somali (`som`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Sesotho (`sot`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Swati (`ssw`) | 40 | 18 (45.0%) | 0 | 2 | 0 | 22 |
| Swahili (`swh`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Tigrinya (`tir`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Setswana (`tsn`) | 40 | 18 (45.0%) | 0 | 2 | 0 | 22 |
| Tsonga (`tso`) | 40 | 18 (45.0%) | 0 | 2 | 0 | 22 |
| Tumbuka (`tum`) | 40 | 18 (45.0%) | 0 | 2 | 0 | 22 |
| Twi (`twi`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Venda (`ven`) | 40 | 18 (45.0%) | 0 | 2 | 0 | 22 |
| Wolof (`wol`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Xhosa (`xho`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Yoruba (`yor`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |
| Zulu (`zul`) | 40 | 0 (0.0%) | 0 | 0 | 0 | 40 |

**gatitos** per language:

| language | rows | contaminated | sentence rule | near | pair | clean |
|---|---|---|---|---|---|---|
| Bambara (`bam`) | 4000 | 2865 (71.6%) | 2865 | 0 | 43 | 1135 |
| Bemba (`bem`) | 3977 | 2849 (71.6%) | 2849 | 0 | 32 | 1128 |
| Dyula (`dyu`) | 3997 | 2862 (71.6%) | 2862 | 0 | 4 | 1135 |
| Ewe (`ewe`) | 3990 | 2855 (71.6%) | 2855 | 0 | 39 | 1135 |
| Fon (`fon`) | 3998 | 2863 (71.6%) | 2863 | 0 | 190 | 1135 |
| Kabyle (`kab`) | 4001 | 2865 (71.6%) | 2865 | 0 | 127 | 1136 |
| Kikuyu (`kik`) | 3994 | 2861 (71.6%) | 2861 | 0 | 14 | 1133 |
| Lingala (`lin`) | 3994 | 2862 (71.7%) | 2862 | 0 | 30 | 1132 |
| Luba-Kasai (`lua`) | 3998 | 2862 (71.6%) | 2862 | 0 | 22 | 1136 |
| Luganda (`lug`) | 3998 | 2863 (71.6%) | 2863 | 0 | 38 | 1135 |
| Luo (`luo`) | 3941 | 2825 (71.7%) | 2825 | 0 | 74 | 1116 |
| Mossi (`mos`) | 4000 | 2864 (71.6%) | 2864 | 0 | 3 | 1136 |
| Sepedi (`nso`) | 3996 | 2861 (71.6%) | 2861 | 0 | 90 | 1135 |
| Nuer (`nus`) | 4001 | 2865 (71.6%) | 2865 | 0 | 24 | 1136 |
| Oromo (`orm`) | 3995 | 2860 (71.6%) | 2860 | 0 | 50 | 1135 |
| Pidgin (`pcm`) | 4001 | 2865 (71.6%) | 2865 | 0 | 92 | 1136 |
| Kirundi (`run`) | 3999 | 2863 (71.6%) | 2863 | 0 | 23 | 1136 |
| Sango (`sag`) | 3994 | 2859 (71.6%) | 2859 | 0 | 23 | 1135 |
| Swati (`ssw`) | 3924 | 2815 (71.7%) | 2815 | 0 | 18 | 1109 |
| Tigrinya (`tir`) | 4000 | 2865 (71.6%) | 2865 | 0 | 14 | 1135 |
| Setswana (`tsn`) | 3995 | 2859 (71.6%) | 2859 | 0 | 131 | 1136 |
| Tsonga (`tso`) | 3994 | 2859 (71.6%) | 2859 | 0 | 86 | 1135 |
| Tumbuka (`tum`) | 3983 | 2854 (71.7%) | 2854 | 0 | 20 | 1129 |
| Twi (`twi`) | 4000 | 2865 (71.6%) | 2865 | 0 | 258 | 1135 |
| Venda (`ven`) | 3984 | 2851 (71.6%) | 2851 | 0 | 25 | 1133 |
| Wolof (`wol`) | 4001 | 2865 (71.6%) | 2865 | 0 | 711 | 1136 |

### smol_sent: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 500 | 76 | 18.14 | 16.03 | 5.07 | 3.55 | 2.19 | 1.45 | 0.577 | 0.466 | 0.334 | 5.64 | +10.39 |
| Hausa (`hau`) | 500 | 76 | 36.20 | 30.25 | 11.03 | 6.74 | 10.49 | 7.35 | 0.570 | 0.449 | 0.401 | 8.21 | +22.04 |
| Igbo (`ibo`) | 500 | 76 | 43.29 | 34.01 | 23.69 | 15.11 | 20.21 | 10.63 | 0.644 | 0.535 | 0.402 | 7.66 | +26.35 |
| Pidgin (`pcm`) | 500 | 76 | 48.73 | 54.08 | 24.32 | 33.15 | 20.42 | 24.55 | 0.564 | 0.556 | 0.289 | 44.35 | +9.73 |
| Twi (`twi`) | 500 | 76 | 28.64 | 24.56 | 7.37 | 4.74 | 6.17 | 4.59 | 0.605 | 0.509 | 0.355 | 11.58 | +12.98 |
| Ewe (`ewe`) | 500 | 76 | 25.99 | 21.29 | 6.28 | 2.64 | 5.15 | 2.13 | 0.563 | 0.464 | 0.316 | 9.75 | +11.55 |
| Fon (`fon`) | 500 | 76 | 15.49 | 14.76 | 1.70 | 1.10 | 1.00 | 0.56 | 0.557 | 0.487 | 0.338 | 6.61 | +8.14 |
| Wolof (`wol`) | 500 | 76 | 22.53 | 19.49 | 5.80 | 3.13 | 4.05 | 2.26 | 0.601 | 0.513 | 0.352 | 10.76 | +8.73 |
| Swahili (`swh`) | 500 | 76 | 51.29 | 41.52 | 29.48 | 20.05 | 25.64 | 16.57 | 0.627 | 0.478 | 0.435 | 10.56 | +30.96 |
| Kinyarwanda (`kin`) | 500 | 76 | 29.87 | 22.24 | 8.30 | 3.75 | 5.39 | 2.30 | 0.573 | 0.449 | 0.361 | 11.42 | +10.82 |
| Somali (`som`) | 500 | 76 | 32.74 | 27.57 | 8.91 | 6.04 | 5.69 | 4.79 | 0.539 | 0.447 | 0.377 | 8.38 | +19.19 |
| Oromo (`orm`) | 500 | 76 | 22.38 | 15.10 | 2.82 | 0.84 | 1.01 | 0.36 | 0.510 | 0.386 | 0.279 | 8.80 | +6.30 |
| Amharic (`amh`) | 500 | 76 | 13.27 | 10.26 | 3.43 | 1.87 | 1.15 | 0.42 | 0.368 | 0.296 | 0.265 | 0.07 | +10.19 |
| Tigrinya (`tir`) | 500 | 76 | 8.92 | 6.16 | 1.33 | 0.40 | 0.41 | 0.24 | 0.461 | 0.332 | 0.223 | 0.09 | +6.07 |
| Luganda (`lug`) | 500 | 76 | 30.02 | 22.25 | 7.76 | 3.61 | 4.26 | 1.48 | 0.571 | 0.455 | 0.355 | 10.26 | +11.99 |
| Lingala (`lin`) | 500 | 76 | 33.74 | 24.24 | 6.20 | 2.30 | 5.68 | 2.12 | 0.519 | 0.359 | 0.248 | 12.05 | +12.19 |
| Zulu (`zul`) | 500 | 76 | 42.45 | 34.26 | 17.73 | 10.13 | 10.83 | 5.02 | 0.603 | 0.507 | 0.437 | 9.83 | +24.43 |
| Xhosa (`xho`) | 500 | 76 | 36.83 | 30.22 | 11.89 | 6.46 | 5.76 | 1.64 | 0.568 | 0.452 | 0.396 | 11.71 | +18.51 |
| Afrikaans (`afr`) | 500 | 76 | 55.79 | 46.49 | 34.88 | 25.88 | 30.33 | 20.20 | 0.675 | 0.545 | 0.517 | 19.55 | +26.94 |
| Shona (`sna`) | 500 | 76 | 32.92 | 26.49 | 8.20 | 4.22 | 4.87 | 3.05 | 0.558 | 0.434 | 0.361 | 7.59 | +18.90 |
| Chichewa (`nya`) | 500 | 76 | 31.81 | 25.46 | 7.31 | 5.36 | 4.84 | 3.43 | 0.562 | 0.429 | 0.386 | 10.88 | +14.58 |
| Sesotho (`sot`) | 500 | 76 | 37.79 | 29.59 | 14.44 | 7.08 | 13.22 | 4.85 | 0.588 | 0.466 | 0.393 | 12.34 | +17.25 |
| Setswana (`tsn`) | 500 | 76 | 42.80 | 36.18 | 19.51 | 11.72 | 17.57 | 10.25 | 0.646 | 0.527 | 0.426 | 13.68 | +22.50 |
| Malagasy (`mlg`) | 500 | 76 | 36.08 | 27.52 | 11.21 | 5.11 | 7.18 | 2.79 | 0.486 | 0.342 | 0.291 | 10.64 | +16.88 |
| Bemba (`bem`) | 500 | 76 | 22.17 | 17.23 | 2.05 | 0.87 | 1.06 | 0.60 | 0.469 | 0.359 | 0.250 | 10.70 | +6.52 |
| Tumbuka (`tum`) | 500 | 76 | 27.82 | 20.87 | 5.91 | 2.79 | 3.95 | 1.61 | 0.529 | 0.408 | 0.321 | 12.11 | +8.76 |
| Kirundi (`run`) | 500 | 76 | 27.12 | 21.22 | 4.24 | 1.45 | 2.30 | 0.40 | 0.577 | 0.466 | 0.363 | 8.98 | +12.24 |
| Venda (`ven`) | 500 | 76 | 29.83 | 21.79 | 8.44 | 3.54 | 5.52 | 2.65 | - | - | - | 10.09 | +11.69 |
| Tsonga (`tso`) | 500 | 76 | 41.36 | 29.67 | 15.71 | 5.61 | 14.67 | 5.09 | 0.600 | 0.475 | 0.352 | 12.10 | +17.56 |
| Sepedi (`nso`) | 500 | 76 | 36.18 | 32.92 | 13.16 | 8.00 | 12.02 | 6.82 | 0.582 | 0.473 | 0.398 | 12.31 | +20.61 |
| Swati (`ssw`) | 500 | 76 | 31.54 | 25.24 | 6.94 | 3.28 | 3.11 | 0.89 | 0.551 | 0.438 | 0.365 | 11.98 | +13.26 |
| Mossi (`mos`) | 500 | 76 | 13.18 | 11.45 | 0.48 | 0.62 | 0.38 | 0.63 | 0.459 | 0.379 | 0.298 | 10.14 | +1.31 |
| Kikuyu (`kik`) | 500 | 76 | 23.78 | 16.72 | 3.88 | 1.18 | 2.94 | 0.61 | 0.556 | 0.425 | 0.255 | 10.74 | +5.98 |
| Luo (`luo`) | 500 | 76 | 20.27 | 15.27 | 1.98 | 0.77 | 1.32 | 0.58 | 0.557 | 0.447 | 0.327 | 11.70 | +3.57 |
| Dyula (`dyu`) | 500 | 76 | 14.71 | 11.35 | 0.81 | 0.34 | 0.67 | 0.28 | - | - | - | 10.74 | +0.60 |
| Bambara (`bam`) | 500 | 76 | 14.80 | 10.73 | 1.71 | 0.64 | 1.60 | 0.75 | 0.348 | 0.276 | 0.142 | 11.67 | -0.94 |
| Kabyle (`kab`) | 500 | 76 | 20.85 | 17.62 | 4.31 | 2.51 | 2.32 | 1.25 | 0.630 | 0.559 | 0.275 | 11.73 | +5.88 |

### smol_doc: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 40 | 40 | 24.33 | 24.33 | 10.37 | 10.37 | 7.88 | 7.88 | - | - | - | 8.04 | +16.29 |
| Hausa (`hau`) | 40 | 40 | 39.05 | 39.05 | 13.13 | 13.13 | 12.69 | 12.69 | - | - | - | 9.26 | +29.79 |
| Igbo (`ibo`) | 40 | 40 | 39.55 | 39.55 | 21.77 | 21.77 | 18.96 | 18.96 | - | - | - | 13.67 | +25.88 |
| Pidgin (`pcm`) | 40 | 40 | 41.37 | 41.37 | 22.41 | 22.41 | 20.66 | 20.66 | - | - | - | 47.19 | -5.83 |
| Twi (`twi`) | 40 | 40 | 26.58 | 26.58 | 6.42 | 6.42 | 4.90 | 4.90 | - | - | - | 15.65 | +10.93 |
| Ewe (`ewe`) | 40 | 40 | 20.02 | 20.02 | 4.58 | 4.58 | 3.94 | 3.94 | - | - | - | 15.44 | +4.59 |
| Fon (`fon`) | 40 | 22 | 21.09 | 17.59 | 4.28 | 3.58 | 3.08 | 2.60 | - | - | - | 9.03 | +8.55 |
| Wolof (`wol`) | 40 | 40 | 21.86 | 21.86 | 6.08 | 6.08 | 5.18 | 5.18 | - | - | - | 15.41 | +6.44 |
| Swahili (`swh`) | 40 | 40 | 49.08 | 49.08 | 25.45 | 25.45 | 21.80 | 21.80 | - | - | - | 15.09 | +33.99 |
| Kinyarwanda (`kin`) | 40 | 40 | 33.22 | 33.22 | 9.69 | 9.69 | 7.60 | 7.60 | - | - | - | 15.35 | +17.87 |
| Somali (`som`) | 40 | 40 | 42.75 | 42.75 | 17.34 | 17.34 | 13.63 | 13.63 | - | - | - | 8.42 | +34.33 |
| Oromo (`orm`) | 40 | 40 | 22.99 | 22.99 | 4.09 | 4.09 | 2.40 | 2.40 | - | - | - | 10.66 | +12.33 |
| Amharic (`amh`) | 40 | 40 | 15.24 | 15.24 | 3.82 | 3.82 | 1.00 | 1.00 | - | - | - | 0.49 | +14.76 |
| Tigrinya (`tir`) | 40 | 40 | 7.26 | 7.26 | 1.25 | 1.25 | 0.37 | 0.37 | - | - | - | 0.57 | +6.69 |
| Luganda (`lug`) | 40 | 40 | 30.42 | 30.42 | 8.44 | 8.44 | 4.70 | 4.70 | - | - | - | 14.64 | +15.78 |
| Lingala (`lin`) | 40 | 40 | 34.04 | 34.04 | 8.52 | 8.52 | 8.11 | 8.11 | - | - | - | 15.74 | +18.30 |
| Zulu (`zul`) | 40 | 40 | 41.12 | 41.12 | 13.68 | 13.68 | 8.46 | 8.46 | - | - | - | 16.11 | +25.01 |
| Xhosa (`xho`) | 40 | 40 | 38.90 | 38.90 | 12.01 | 12.01 | 7.76 | 7.76 | - | - | - | 15.80 | +23.10 |
| Afrikaans (`afr`) | 40 | 40 | 47.26 | 47.26 | 29.04 | 29.04 | 26.93 | 26.93 | - | - | - | 28.71 | +18.55 |
| Shona (`sna`) | 40 | 40 | 36.78 | 36.78 | 11.27 | 11.27 | 7.49 | 7.49 | - | - | - | 15.40 | +21.38 |
| Chichewa (`nya`) | 40 | 40 | 37.87 | 37.87 | 9.45 | 9.45 | 7.90 | 7.90 | - | - | - | 16.66 | +21.21 |
| Sesotho (`sot`) | 40 | 40 | 36.87 | 36.87 | 13.66 | 13.66 | 13.88 | 13.88 | - | - | - | 16.71 | +20.16 |
| Setswana (`tsn`) | 40 | 22 | 46.68 | 39.98 | 24.18 | 18.53 | 22.75 | 17.28 | - | - | - | 18.21 | +21.77 |
| Malagasy (`mlg`) | 40 | 40 | 37.37 | 37.37 | 11.77 | 11.77 | 8.53 | 8.53 | - | - | - | 12.84 | +24.53 |
| Bemba (`bem`) | 40 | 22 | 29.06 | 24.57 | 5.30 | 2.96 | 3.46 | 1.63 | - | - | - | 16.40 | +8.17 |
| Tumbuka (`tum`) | 40 | 22 | 31.12 | 24.46 | 7.62 | 4.01 | 6.54 | 3.56 | - | - | - | 16.61 | +7.85 |
| Kirundi (`run`) | 40 | 22 | 30.54 | 27.01 | 5.83 | 4.93 | 4.64 | 3.59 | - | - | - | 13.97 | +13.04 |
| Venda (`ven`) | 40 | 22 | 32.59 | 25.08 | 12.37 | 8.61 | 9.43 | 6.61 | - | - | - | 14.90 | +10.18 |
| Tsonga (`tso`) | 40 | 22 | 42.10 | 36.25 | 14.62 | 10.60 | 14.43 | 10.98 | - | - | - | 15.80 | +20.46 |
| Sepedi (`nso`) | 40 | 40 | 31.44 | 31.44 | 13.50 | 13.50 | 13.19 | 13.19 | - | - | - | 16.86 | +14.58 |
| Swati (`ssw`) | 40 | 22 | 35.54 | 30.30 | 8.61 | 6.20 | 4.74 | 3.14 | - | - | - | 17.68 | +12.61 |
| Mossi (`mos`) | 40 | 22 | 15.18 | 14.45 | 2.04 | 2.30 | 1.88 | 2.39 | - | - | - | 14.70 | -0.25 |
| Kikuyu (`kik`) | 40 | 22 | 24.48 | 18.91 | 5.15 | 3.02 | 4.15 | 2.42 | - | - | - | 13.78 | +5.13 |
| Luo (`luo`) | 40 | 40 | 27.81 | 27.81 | 6.54 | 6.54 | 5.03 | 5.03 | - | - | - | 12.77 | +15.03 |
| Dyula (`dyu`) | 40 | 22 | 15.77 | 12.59 | 2.13 | 1.55 | 1.72 | 1.56 | - | - | - | 13.99 | -1.39 |
| Bambara (`bam`) | 40 | 40 | 20.32 | 20.32 | 5.75 | 5.75 | 5.46 | 5.46 | - | - | - | 14.41 | +5.90 |
| Kabyle (`kab`) | 40 | 40 | 18.59 | 18.59 | 4.10 | 4.10 | 2.48 | 2.48 | - | - | - | 17.23 | +1.36 |

### gatitos: SFT model per language (word level)

n = entries (English word or phrase); exact = the output equals ANY accepted translation (NFC, case-folded, surrounding punctuation removed); chrF++ = against the best accepted translation. clean = the entry is not a whole sentence of the training windows.

| language | n | n clean | chrF++ all | chrF++ clean | exact all | exact clean | base exact all | base exact clean | finished | prefix ok |
|---|---|---|---|---|---|---|---|---|---|---|
| Pidgin (`pcm`) | 4001 | 1136 | 26.27 | 22.84 | 10.0% | 10.5% | 1.7% | 1.0% | 92.3% | 100.0% |
| Twi (`twi`) | 4000 | 1135 | 17.23 | 11.31 | 10.3% | 5.3% | 0.3% | 0.6% | 99.7% | 100.0% |
| Ewe (`ewe`) | 3990 | 1135 | 10.48 | 7.87 | 2.6% | 1.4% | 0.2% | 0.5% | 96.4% | 100.0% |
| Fon (`fon`) | 3998 | 1135 | 10.26 | 7.55 | 1.2% | 0.6% | 0.0% | 0.0% | 97.0% | 100.0% |
| Wolof (`wol`) | 4001 | 1136 | 14.20 | 9.57 | 7.2% | 4.1% | 0.0% | 0.0% | 99.6% | 99.9% |
| Oromo (`orm`) | 3995 | 1135 | 17.10 | 12.00 | 5.4% | 1.6% | 0.0% | 0.0% | 94.3% | 100.0% |
| Tigrinya (`tir`) | 4000 | 1135 | 5.15 | 2.68 | 0.9% | 0.0% | 0.0% | 0.0% | 79.2% | 100.0% |
| Luganda (`lug`) | 3998 | 1135 | 18.29 | 13.11 | 9.4% | 4.6% | 0.0% | 0.0% | 99.8% | 100.0% |
| Lingala (`lin`) | 3994 | 1132 | 19.21 | 14.47 | 6.8% | 3.6% | 0.0% | 0.0% | 98.4% | 100.0% |
| Setswana (`tsn`) | 3995 | 1136 | 24.55 | 18.31 | 7.5% | 2.3% | 0.0% | 0.0% | 98.1% | 100.0% |
| Bemba (`bem`) | 3977 | 1128 | 6.54 | 3.59 | 1.9% | 0.8% | 0.1% | 0.3% | 99.6% | 100.0% |
| Tumbuka (`tum`) | 3983 | 1129 | 10.27 | 9.01 | 1.6% | 1.1% | 0.1% | 0.2% | 96.7% | 100.0% |
| Kirundi (`run`) | 3999 | 1136 | 14.17 | 11.07 | 5.1% | 2.9% | 0.1% | 0.3% | 99.1% | 100.0% |
| Venda (`ven`) | 3984 | 1133 | 15.88 | 11.17 | 3.3% | 2.2% | 0.1% | 0.3% | 94.8% | 100.0% |
| Tsonga (`tso`) | 3994 | 1135 | 22.65 | 14.89 | 7.1% | 3.2% | 0.2% | 0.4% | 96.7% | 100.0% |
| Sepedi (`nso`) | 3996 | 1135 | 22.43 | 16.02 | 9.2% | 4.1% | 0.1% | 0.3% | 95.5% | 100.0% |
| Swati (`ssw`) | 3924 | 1109 | 14.35 | 9.32 | 5.4% | 1.8% | 0.0% | 0.0% | 99.1% | 100.0% |
| Sango (`sag`) | 3994 | 1135 | 16.89 | 13.81 | 2.1% | 1.3% | 0.0% | 0.0% | 95.6% | 100.0% |
| Mossi (`mos`) | 4000 | 1136 | 5.08 | 4.48 | 0.0% | 0.0% | 0.0% | 0.0% | 79.8% | 100.0% |
| Kikuyu (`kik`) | 3994 | 1133 | 12.61 | 10.63 | 1.7% | 1.1% | 0.0% | 0.1% | 90.6% | 100.0% |
| Luo (`luo`) | 3941 | 1116 | 11.76 | 8.79 | 2.9% | 1.8% | 0.0% | 0.0% | 99.2% | 100.0% |
| Dyula (`dyu`) | 3997 | 1135 | 7.59 | 6.94 | 0.1% | 0.0% | 0.0% | 0.0% | 92.9% | 100.0% |
| Bambara (`bam`) | 4000 | 1135 | 7.63 | 6.32 | 1.4% | 1.1% | 0.0% | 0.1% | 97.4% | 100.0% |
| Kabyle (`kab`) | 4001 | 1136 | 10.82 | 6.69 | 3.7% | 0.9% | 0.0% | 0.0% | 99.9% | 100.0% |
| Nuer (`nus`) | 4001 | 1136 | 6.04 | 5.80 | 0.2% | 0.1% | 0.1% | 0.0% | 95.2% | 100.0% |
| Luba-Kasai (`lua`) | 3998 | 1136 | 14.58 | 12.05 | 2.2% | 0.5% | 0.0% | 0.0% | 98.2% | 100.0% |

## Output format failures

`prefix_ok` = the completion starts with the expected language tag (the base model does not know the convention: its whole output is scored, the tag stripped only if present); `finished` = `<end_of_turn>` produced inside the token budget; `empty` = nothing left after stripping. All rows, unweighted mean over languages; the worst languages follow.

| benchmark | system | prefix_ok | finished | empty |
|---|---|---|---|---|
| smol_sent | SFT | 100.0% | 95.2% | 0.0% |
| smol_sent | base gemma-3-270m-it | 0.0% | 92.3% | 0.0% |
| smol_doc | SFT | 99.8% | 62.8% | 0.0% |
| smol_doc | base gemma-3-270m-it | 0.0% | 95.7% | 0.0% |
| gatitos | SFT | 100.0% | 95.6% | 15.4% |
| gatitos | base gemma-3-270m-it | 0.0% | 98.9% | 0.0% |

- smol_sent, SFT, lowest prefix_ok: ewe 99.8%, fon 99.8%, nya 99.8%, som 99.8%
- smol_sent, SFT, lowest finished: bam 82.8%, tir 83.0%, kik 83.8%, orm 84.2%
- smol_sent, SFT, highest empty: twi 0.2%, afr 0.0%, amh 0.0%, bam 0.0%
- smol_doc, SFT, lowest prefix_ok: orm 97.5%, ssw 97.5%, ven 97.5%, afr 100.0%
- smol_doc, SFT, lowest finished: dyu 25.0%, tir 30.0%, kik 40.0%, bam 45.0%
- smol_doc, SFT, highest empty: afr 0.0%, amh 0.0%, bam 0.0%, bem 0.0%
- gatitos, SFT, lowest prefix_ok: wol 99.9%, pcm 100.0%, luo 100.0%, ven 100.0%
- gatitos, SFT, lowest finished: tir 79.2%, mos 79.8%, kik 90.6%, pcm 92.3%
- gatitos, SFT, highest empty: bem 75.2%, ssw 53.8%, kab 39.9%, fon 33.6%

## How the contamination rule changes the macro chrF++

| benchmark | system | all | clean | clean, sentence rule | clean, no near | pair not seen |
|---|---|---|---|---|---|---|
| smol_sent | SFT | 29.77 (37L) | 24.11 (37L) | 24.50 (37L) | 24.11 (37L) | 29.54 (37L) |
| smol_sent | base gemma-3-270m-it | 10.62 (37L) | 11.01 (37L) | 11.18 (37L) | 11.01 (37L) | 10.62 (37L) |
| smol_doc | SFT | 30.98 (37L) | 29.55 (37L) | 30.98 (37L) | 29.55 (37L) | 30.98 (37L) |
| smol_doc | base gemma-3-270m-it | 15.13 (37L) | 14.98 (37L) | 15.13 (37L) | 14.98 (37L) | 15.13 (37L) |
| gatitos | SFT | 13.92 (26L) | 10.40 (26L) | 10.40 (26L) | 10.40 (26L) | 13.58 (26L) |
| gatitos | base gemma-3-270m-it | 5.36 (26L) | 5.55 (26L) | 5.55 (26L) | 5.55 (26L) | 5.36 (26L) |

## Runtime

| system | benchmark | rows | sampler | generation s | scoring s (incl. AfriCOMET) |
|---|---|---|---|---|---|
| SFT | smol_sent | 18500 | hf | 614 | 50 |
| SFT | smol_doc | 1480 | hf | 512 | 26 |
| SFT | gatitos | 103755 | hf | 288 | 32 |
| base gemma-3-270m-it | smol_sent | 18500 | hf | 1083 | 257 |
| base gemma-3-270m-it | smol_doc | 1480 | hf | 733 | 26 |
| base gemma-3-270m-it | gatitos | 103755 | hf | 406 | 48 |

Whole run: 1591 s.
