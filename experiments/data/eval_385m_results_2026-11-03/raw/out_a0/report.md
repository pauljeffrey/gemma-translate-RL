# SFT benchmark evaluation (English -> African)

* SFT model: `/workspace/eval_385m/model_a0` revision `main` commit `None`
* base model: `-`
* decoding: greedy, same prompt for both (`Translate the text to {language}:  {english}`), max_new_tokens 768, sampler vllm; metrics on NFC-normalised text; spBLEU tokenizer flores200
* started 2026-11-03 04:21:04, finished 2026-11-03 04:42:24, 1280 s; environment {"python": "3.12.3", "platform": "Linux-5.15.0-187-generic-x86_64-with-glibc2.39", "host": "cca81a4f31dc", "torch": "2.15.0+cu132", "transformers": "4.57.6", "sacrebleu": "2.6.0", "huggingface_hub": "0.36.2", "gpu": "NVIDIA A100 80GB PCIe, 81920 MiB"}

## Results

### Macro averages, clean (headline: English sentence not in the training text)

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|

### Macro averages, all rows

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|
| **flores_plus** | SFT | 46 | 46552 | 20.20 (46) | 3.73 (46) | 2.95 (46) | 0.442 (40) | 0.397 (40) |
| **ntrex** | SFT | 24 | 47928 | 23.01 (24) | 5.15 (24) | 4.04 (24) | 0.435 (21) | 0.386 (21) |
| **mafand** | SFT | 15 | 20009 | 25.56 (15) | 6.07 (15) | 4.45 (15) | 0.472 (15) | 0.442 (15) |
| **smol_sent** | SFT | 45 | 38833 | 17.18 (45) | 1.98 (45) | 1.61 (45) | 0.435 (36) | 0.380 (36) |
| **ibom_mt** | SFT | 2 | 2024 | 9.87 (2) | 0.92 (2) | 0.56 (2) | 0.441 (1) | 0.366 (1) |
| **flores_moz** | SFT | 1 | 1012 | 21.08 (1) | 2.22 (1) | 1.74 (1) | - | - |
| **glot_story** | SFT | 8 | 3426 | 14.62 (8) | 1.51 (8) | 0.80 (8) | - | - |
| **salt** | SFT | 5 | 2500 | 25.13 (5) | 5.22 (5) | 3.29 (5) | 0.543 (3) | 0.550 (3) |

## Contamination of the benchmarks by the SFT training text

A row is *contaminated* when the normalised English sentence (NFKC, case-folded, punctuation removed) of the benchmark row occurs in the SFT training windows (`gemma/windows_sft`, the input of the packed training set): as a whole sentence of a window or as a contiguous run inside one. *sentence rule* = the first half only (the filter that produced `bench_dev_clean.jsonl`; it reproduces that file exactly: 10,043 of 30,418 dev rows). *near* = not contaminated but >= 50% of its 8-token n-grams occur in the training English. *pair* = contaminated and the reference translation occurs in the target side of a window of the same language (the model saw the very pair).

| benchmark | rows | contaminated | of which sentence-rule hits | near | pair seen | clean rows |
|---|---|---|---|---|---|---|
| flores_plus | 46552 | - | - | - | - | - |
| ntrex | 47928 | - | - | - | - | - |
| mafand | 20009 | - | - | - | - | - |
| smol_sent | 38833 | - | - | - | - | - |
| ibom_mt | 2024 | - | - | - | - | - |
| flores_moz | 1012 | - | - | - | - | - |
| glot_story | 3426 | - | - | - | - | - |
| salt | 2500 | - | - | - | - | - |

### flores_plus: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 1012 | - | 16.91 | - | 3.11 | - | 1.74 | - | 0.541 | - | - | - | - |
| Hausa (`hau`) | 1012 | - | 32.24 | - | 7.55 | - | 6.86 | - | 0.479 | - | - | - | - |
| Igbo (`ibo`) | 1012 | - | 26.83 | - | 6.48 | - | 5.65 | - | 0.544 | - | - | - | - |
| Twi (`twi`) | 1012 | - | 17.49 | - | 2.12 | - | 1.49 | - | 0.454 | - | - | - | - |
| Ewe (`ewe`) | 1012 | - | 20.49 | - | 3.07 | - | 2.32 | - | 0.462 | - | - | - | - |
| Fon (`fon`) | 1012 | - | 15.88 | - | 1.95 | - | 1.06 | - | 0.500 | - | - | - | - |
| Wolof (`wol`) | 1012 | - | 18.47 | - | 2.96 | - | 1.98 | - | 0.541 | - | - | - | - |
| Swahili (`swh`) | 1012 | - | 35.58 | - | 9.15 | - | 8.43 | - | 0.463 | - | - | - | - |
| Kinyarwanda (`kin`) | 1012 | - | 22.34 | - | 3.15 | - | 2.61 | - | 0.466 | - | - | - | - |
| Somali (`som`) | 1012 | - | 30.82 | - | 7.14 | - | 4.80 | - | 0.485 | - | - | - | - |
| Oromo (`orm`) | 1012 | - | 11.82 | - | 0.49 | - | 0.32 | - | 0.360 | - | - | - | - |
| Amharic (`amh`) | 1012 | - | 7.47 | - | 0.95 | - | 0.25 | - | 0.276 | - | - | - | - |
| Tigrinya (`tir`) | 1012 | - | 9.29 | - | 1.34 | - | 0.43 | - | 0.412 | - | - | - | - |
| Luganda (`lug`) | 1012 | - | 16.47 | - | 1.25 | - | 1.03 | - | 0.419 | - | - | - | - |
| Lingala (`lin`) | 1012 | - | 22.80 | - | 2.40 | - | 2.05 | - | 0.421 | - | - | - | - |
| Zulu (`zul`) | 1012 | - | 32.03 | - | 8.41 | - | 5.14 | - | 0.460 | - | - | - | - |
| Xhosa (`xho`) | 1012 | - | 28.04 | - | 5.41 | - | 2.88 | - | 0.434 | - | - | - | - |
| Afrikaans (`afr`) | 1012 | - | 51.89 | - | 28.37 | - | 24.86 | - | 0.572 | - | - | - | - |
| Shona (`sna`) | 1012 | - | 23.45 | - | 3.49 | - | 2.53 | - | 0.450 | - | - | - | - |
| Chichewa (`nya`) | 1012 | - | 24.05 | - | 2.96 | - | 2.78 | - | 0.473 | - | - | - | - |
| Sesotho (`sot`) | 1012 | - | 33.74 | - | 8.67 | - | 7.47 | - | 0.538 | - | - | - | - |
| Setswana (`tsn`) | 1012 | - | 28.80 | - | 5.38 | - | 4.91 | - | 0.494 | - | - | - | - |
| Malagasy (`mlg`) | 1012 | - | 21.39 | - | 2.23 | - | 1.82 | - | 0.330 | - | - | - | - |
| Bemba (`bem`) | 1012 | - | 16.82 | - | 1.60 | - | 1.24 | - | 0.395 | - | - | - | - |
| Tumbuka (`tum`) | 1012 | - | 21.28 | - | 3.38 | - | 2.96 | - | 0.442 | - | - | - | - |
| Kirundi (`run`) | 1012 | - | 19.81 | - | 2.01 | - | 1.55 | - | 0.458 | - | - | - | - |
| Tsonga (`tso`) | 1012 | - | 29.24 | - | 5.60 | - | 5.13 | - | 0.508 | - | - | - | - |
| Sepedi (`nso`) | 1012 | - | 32.79 | - | 7.67 | - | 7.49 | - | 0.537 | - | - | - | - |
| Swati (`ssw`) | 1012 | - | 18.87 | - | 1.80 | - | 1.03 | - | 0.383 | - | - | - | - |
| Sango (`sag`) | 1012 | - | 18.17 | - | 1.64 | - | 1.33 | - | - | - | - | - | - |
| Umbundu (`umb`) | 1012 | - | 10.07 | - | 0.36 | - | 0.20 | - | 0.397 | - | - | - | - |
| Kimbundu (`kmb`) | 1012 | - | 10.65 | - | 0.56 | - | 0.32 | - | 0.353 | - | - | - | - |
| Mossi (`mos`) | 1012 | - | 11.25 | - | 0.87 | - | 0.55 | - | 0.404 | - | - | - | - |
| Kikuyu (`kik`) | 1012 | - | 20.66 | - | 2.89 | - | 2.30 | - | 0.494 | - | - | - | - |
| Luo (`luo`) | 1012 | - | 11.98 | - | 0.81 | - | 0.55 | - | 0.410 | - | - | - | - |
| Kamba (`kam`) | 1012 | - | 10.89 | - | 0.88 | - | 0.69 | - | 0.342 | - | - | - | - |
| Dyula (`dyu`) | 1012 | - | 12.23 | - | 1.28 | - | 0.58 | - | - | - | - | - | - |
| Bambara (`bam`) | 1012 | - | 14.26 | - | 1.57 | - | 1.00 | - | 0.337 | - | - | - | - |
| Kabiye (`kbp`) | 1012 | - | 8.42 | - | 0.54 | - | 0.36 | - | 0.393 | - | - | - | - |
| Kabuverdianu (`kea`) | 1012 | - | 36.47 | - | 13.42 | - | 10.29 | - | - | - | - | - | - |
| Kanuri (`knc`) | 1012 | - | 2.58 | - | 0.09 | - | 0.07 | - | - | - | - | - | - |
| Kabyle (`kab`) | 1012 | - | 15.69 | - | 1.41 | - | 1.29 | - | 0.538 | - | - | - | - |
| Nuer (`nus`) | 1012 | - | 7.46 | - | 0.63 | - | 0.23 | - | 0.317 | - | - | - | - |
| Luba-Kasai (`lua`) | 1012 | - | 17.66 | - | 1.33 | - | 0.88 | - | 0.405 | - | - | - | - |
| Makhuwa (`vmw`) | 1012 | - | 12.79 | - | 0.82 | - | 0.56 | - | - | - | - | - | - |
| Sena (`seh`) | 1012 | - | 21.08 | - | 2.22 | - | 1.74 | - | - | - | - | - | - |

### ntrex: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 1997 | - | 19.47 | - | 4.19 | - | 2.37 | - | 0.561 | - | - | - | - |
| Hausa (`hau`) | 1997 | - | 34.72 | - | 9.35 | - | 8.40 | - | 0.464 | - | - | - | - |
| Igbo (`ibo`) | 1997 | - | 27.89 | - | 5.71 | - | 5.59 | - | 0.530 | - | - | - | - |
| Ewe (`ewe`) | 1997 | - | 21.55 | - | 3.68 | - | 2.68 | - | 0.461 | - | - | - | - |
| Wolof (`wol`) | 1997 | - | 19.78 | - | 3.17 | - | 2.23 | - | 0.531 | - | - | - | - |
| Swahili (`swh`) | 1997 | - | 35.28 | - | 9.36 | - | 8.52 | - | 0.425 | - | - | - | - |
| Kinyarwanda (`kin`) | 1997 | - | 22.93 | - | 3.46 | - | 3.12 | - | 0.446 | - | - | - | - |
| Somali (`som`) | 1997 | - | 33.62 | - | 8.95 | - | 6.20 | - | 0.467 | - | - | - | - |
| Oromo (`orm`) | 1997 | - | 11.40 | - | 0.47 | - | 0.31 | - | 0.369 | - | - | - | - |
| Amharic (`amh`) | 1997 | - | 5.50 | - | 0.63 | - | 0.10 | - | 0.255 | - | - | - | - |
| Tigrinya (`tir`) | 1997 | - | 8.94 | - | 1.44 | - | 0.37 | - | 0.409 | - | - | - | - |
| Zulu (`zul`) | 1997 | - | 30.50 | - | 7.35 | - | 3.94 | - | 0.430 | - | - | - | - |
| Xhosa (`xho`) | 1997 | - | 27.57 | - | 6.04 | - | 2.93 | - | 0.406 | - | - | - | - |
| Afrikaans (`afr`) | 1997 | - | 51.37 | - | 29.19 | - | 26.38 | - | 0.497 | - | - | - | - |
| Shona (`sna`) | 1997 | - | 23.34 | - | 3.75 | - | 2.96 | - | 0.431 | - | - | - | - |
| Chichewa (`nya`) | 1997 | - | 24.74 | - | 3.66 | - | 3.45 | - | 0.436 | - | - | - | - |
| Setswana (`tsn`) | 1997 | - | 30.15 | - | 7.24 | - | 6.65 | - | 0.467 | - | - | - | - |
| Malagasy (`mlg`) | 1997 | - | 22.11 | - | 2.71 | - | 2.18 | - | 0.315 | - | - | - | - |
| Bemba (`bem`) | 1997 | - | 17.89 | - | 1.80 | - | 1.32 | - | 0.392 | - | - | - | - |
| Venda (`ven`) | 1997 | - | 19.07 | - | 2.27 | - | 1.44 | - | - | - | - | - | - |
| Sepedi (`nso`) | 1997 | - | 23.93 | - | 4.25 | - | 3.41 | - | 0.457 | - | - | - | - |
| Swati (`ssw`) | 1997 | - | 20.09 | - | 2.24 | - | 1.13 | - | 0.384 | - | - | - | - |
| North Ndebele (`nde`) | 1997 | - | 20.11 | - | 2.79 | - | 1.26 | - | - | - | - | - | - |
| Tachelhit (`shi`) | 1997 | - | 0.39 | - | 0.01 | - | 0.02 | - | - | - | - | - | - |

### mafand: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 1558 | - | 17.66 | - | 4.28 | - | 2.35 | - | 0.554 | - | - | - | - |
| Hausa (`hau`) | 1500 | - | 27.47 | - | 4.85 | - | 3.72 | - | 0.508 | - | - | - | - |
| Igbo (`ibo`) | 1500 | - | 28.87 | - | 7.41 | - | 5.23 | - | 0.579 | - | - | - | - |
| Pidgin (`pcm`) | 1564 | - | 37.89 | - | 16.41 | - | 12.82 | - | 0.515 | - | - | - | - |
| Twi (`twi`) | 1500 | - | 20.37 | - | 2.99 | - | 1.66 | - | 0.466 | - | - | - | - |
| Swahili (`swh`) | 1835 | - | 36.66 | - | 11.96 | - | 10.30 | - | 0.474 | - | - | - | - |
| Kinyarwanda (`kin`) | 1006 | - | 29.71 | - | 6.07 | - | 4.91 | - | 0.503 | - | - | - | - |
| Amharic (`amh`) | 1037 | - | 6.62 | - | 0.98 | - | 0.36 | - | 0.302 | - | - | - | - |
| Luganda (`lug`) | 1500 | - | 22.95 | - | 3.65 | - | 2.73 | - | 0.479 | - | - | - | - |
| Zulu (`zul`) | 998 | - | 32.19 | - | 9.64 | - | 6.46 | - | 0.441 | - | - | - | - |
| Xhosa (`xho`) | 1002 | - | 29.59 | - | 7.06 | - | 3.79 | - | 0.438 | - | - | - | - |
| Shona (`sna`) | 1005 | - | 21.87 | - | 3.56 | - | 2.55 | - | 0.437 | - | - | - | - |
| Chichewa (`nya`) | 1004 | - | 28.01 | - | 4.25 | - | 3.48 | - | 0.462 | - | - | - | - |
| Setswana (`tsn`) | 1500 | - | 31.12 | - | 6.78 | - | 5.59 | - | 0.517 | - | - | - | - |
| Luo (`luo`) | 1500 | - | 12.47 | - | 1.09 | - | 0.74 | - | 0.398 | - | - | - | - |

### smol_sent: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 863 | - | 15.32 | - | 2.59 | - | 1.32 | - | 0.567 | - | - | - | - |
| Hausa (`hau`) | 863 | - | 23.51 | - | 2.61 | - | 2.63 | - | 0.424 | - | - | - | - |
| Igbo (`ibo`) | 863 | - | 24.28 | - | 5.33 | - | 4.46 | - | 0.541 | - | - | - | - |
| Pidgin (`pcm`) | 863 | - | 26.88 | - | 7.18 | - | 5.40 | - | 0.432 | - | - | - | - |
| Efik (`efi`) | 863 | - | 12.07 | - | 0.66 | - | 0.29 | - | 0.441 | - | - | - | - |
| Twi (`twi`) | 863 | - | 12.01 | - | 0.51 | - | 0.43 | - | 0.450 | - | - | - | - |
| Ewe (`ewe`) | 863 | - | 15.83 | - | 1.18 | - | 1.03 | - | 0.463 | - | - | - | - |
| Fon (`fon`) | 863 | - | 13.05 | - | 0.81 | - | 0.52 | - | 0.504 | - | - | - | - |
| Wolof (`wol`) | 863 | - | 14.11 | - | 1.26 | - | 0.72 | - | 0.523 | - | - | - | - |
| Swahili (`swh`) | 863 | - | 27.94 | - | 4.95 | - | 4.70 | - | 0.396 | - | - | - | - |
| Kinyarwanda (`kin`) | 863 | - | 15.41 | - | 1.13 | - | 0.63 | - | 0.435 | - | - | - | - |
| Somali (`som`) | 862 | - | 24.38 | - | 3.10 | - | 1.86 | - | 0.425 | - | - | - | - |
| Oromo (`orm`) | 863 | - | 8.59 | - | 0.23 | - | 0.10 | - | 0.355 | - | - | - | - |
| Amharic (`amh`) | 863 | - | 5.42 | - | 0.45 | - | 0.07 | - | 0.267 | - | - | - | - |
| Tigrinya (`tir`) | 863 | - | 6.85 | - | 0.68 | - | 0.13 | - | 0.403 | - | - | - | - |
| Luganda (`lug`) | 863 | - | 14.65 | - | 0.72 | - | 0.49 | - | 0.434 | - | - | - | - |
| Lingala (`lin`) | 863 | - | 18.29 | - | 1.15 | - | 1.16 | - | 0.409 | - | - | - | - |
| Zulu (`zul`) | 863 | - | 25.70 | - | 3.24 | - | 1.86 | - | 0.417 | - | - | - | - |
| Xhosa (`xho`) | 863 | - | 20.94 | - | 1.74 | - | 0.69 | - | 0.387 | - | - | - | - |
| Afrikaans (`afr`) | 863 | - | 43.05 | - | 18.44 | - | 17.76 | - | 0.491 | - | - | - | - |
| Shona (`sna`) | 863 | - | 18.28 | - | 1.12 | - | 0.78 | - | 0.426 | - | - | - | - |
| Chichewa (`nya`) | 863 | - | 18.32 | - | 1.00 | - | 0.94 | - | 0.419 | - | - | - | - |
| Sesotho (`sot`) | 863 | - | 26.84 | - | 4.62 | - | 4.63 | - | 0.492 | - | - | - | - |
| Setswana (`tsn`) | 863 | - | 23.77 | - | 3.42 | - | 3.42 | - | 0.462 | - | - | - | - |
| Malagasy (`mlg`) | 863 | - | 15.50 | - | 0.68 | - | 0.48 | - | 0.298 | - | - | - | - |
| Bemba (`bem`) | 863 | - | 13.33 | - | 0.35 | - | 0.25 | - | 0.390 | - | - | - | - |
| Tumbuka (`tum`) | 863 | - | 18.96 | - | 1.38 | - | 1.02 | - | 0.434 | - | - | - | - |
| Kirundi (`run`) | 863 | - | 15.55 | - | 0.60 | - | 0.40 | - | 0.449 | - | - | - | - |
| Venda (`ven`) | 863 | - | 14.78 | - | 1.39 | - | 0.77 | - | - | - | - | - | - |
| Tsonga (`tso`) | 863 | - | 24.26 | - | 2.71 | - | 2.78 | - | 0.499 | - | - | - | - |
| Sepedi (`nso`) | 863 | - | 23.99 | - | 3.18 | - | 3.42 | - | 0.473 | - | - | - | - |
| Swati (`ssw`) | 863 | - | 14.33 | - | 0.56 | - | 0.22 | - | 0.374 | - | - | - | - |
| South Ndebele (`nbl`) | 863 | - | 17.57 | - | 0.78 | - | 0.16 | - | - | - | - | - | - |
| Tiv (`tiv`) | 863 | - | 11.19 | - | 0.67 | - | 0.36 | - | - | - | - | - | - |
| Mossi (`mos`) | 863 | - | 8.93 | - | 0.20 | - | 0.16 | - | 0.420 | - | - | - | - |
| Kikuyu (`kik`) | 863 | - | 18.25 | - | 1.66 | - | 1.52 | - | 0.496 | - | - | - | - |
| Luo (`luo`) | 863 | - | 9.85 | - | 0.33 | - | 0.26 | - | 0.420 | - | - | - | - |
| Dyula (`dyu`) | 863 | - | 12.46 | - | 0.61 | - | 0.43 | - | - | - | - | - | - |
| Bambara (`bam`) | 863 | - | 9.22 | - | 0.43 | - | 0.33 | - | 0.304 | - | - | - | - |
| Kongo (`kon`) | 863 | - | 15.75 | - | 0.91 | - | 1.00 | - | - | - | - | - | - |
| Kabyle (`kab`) | 862 | - | 12.82 | - | 0.50 | - | 0.33 | - | 0.542 | - | - | - | - |
| Acholi (`ach`) | 863 | - | 16.45 | - | 1.27 | - | 0.92 | - | - | - | - | - | - |
| Nyankore (`nyn`) | 863 | - | 18.58 | - | 1.20 | - | 0.55 | - | - | - | - | - | - |
| Ndau (`ndc`) | 863 | - | 12.59 | - | 0.23 | - | 0.16 | - | - | - | - | - | - |
| Krio (`kri`) | 863 | - | 13.08 | - | 1.50 | - | 1.02 | - | - | - | - | - | - |

### ibom_mt: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Efik (`efi`) | 1012 | - | 15.89 | - | 1.71 | - | 1.04 | - | 0.441 | - | - | - | - |
| Ibibio (`ibb`) | 1012 | - | 3.86 | - | 0.14 | - | 0.07 | - | - | - | - | - | - |

### flores_moz: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sena (`seh`) | 1012 | - | 21.08 | - | 2.22 | - | 1.74 | - | - | - | - | - | - |

### glot_story: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| South Ndebele (`nbl`) | 321 | - | 16.90 | - | 2.07 | - | 0.49 | - | - | - | - | - | - |
| Dagbani (`dag`) | 135 | - | 6.26 | - | 0.40 | - | 0.15 | - | - | - | - | - | - |
| Kaonde (`kqn`) | 420 | - | 14.51 | - | 1.61 | - | 0.78 | - | - | - | - | - | - |
| Lunda (`lun`) | 420 | - | 15.03 | - | 1.42 | - | 0.82 | - | - | - | - | - | - |
| Lozi (`loz`) | 960 | - | 20.29 | - | 2.56 | - | 1.51 | - | - | - | - | - | - |
| Tswa (`tsc`) | 101 | - | 9.77 | - | 0.14 | - | 0.18 | - | - | - | - | - | - |
| Kuanyama (`kua`) | 520 | - | 20.09 | - | 2.51 | - | 1.45 | - | - | - | - | - | - |
| Herero (`her`) | 549 | - | 14.13 | - | 1.36 | - | 1.02 | - | - | - | - | - | - |

### salt: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Igbo (`ibo`) | 500 | - | 25.33 | - | 5.20 | - | 3.85 | - | 0.613 | - | - | - | - |
| Swahili (`swh`) | 500 | - | 32.63 | - | 6.88 | - | 5.98 | - | 0.500 | - | - | - | - |
| Luganda (`lug`) | 500 | - | 20.63 | - | 2.22 | - | 1.30 | - | 0.517 | - | - | - | - |
| Acholi (`ach`) | 500 | - | 21.89 | - | 5.16 | - | 3.10 | - | - | - | - | - | - |
| Nyankore (`nyn`) | 500 | - | 25.17 | - | 6.63 | - | 2.21 | - | - | - | - | - | - |

## Output format failures

`prefix_ok` = the completion starts with the expected language tag (the base model does not know the convention: its whole output is scored, the tag stripped only if present); `finished` = `<end_of_turn>` produced inside the token budget; `empty` = nothing left after stripping. All rows, unweighted mean over languages; the worst languages follow.

| benchmark | system | prefix_ok | finished | empty |
|---|---|---|---|---|
| flores_plus | SFT | 100.0% | 81.0% | 0.0% |
| ntrex | SFT | 99.9% | 86.8% | 0.0% |
| mafand | SFT | 99.8% | 89.8% | 0.1% |
| smol_sent | SFT | 97.8% | 87.4% | 0.0% |
| ibom_mt | SFT | 50.0% | 55.2% | 0.0% |
| flores_moz | SFT | 99.9% | 90.9% | 0.0% |
| glot_story | SFT | 90.4% | 88.9% | 0.0% |
| salt | SFT | 100.0% | 98.2% | 0.0% |

- flores_plus, SFT, lowest prefix_ok: seh 99.9%, som 99.9%, tsn 99.9%, wol 99.9%
- flores_plus, SFT, lowest finished: knc 5.1%, nus 26.5%, kbp 44.1%, orm 61.9%
- flores_plus, SFT, highest empty: afr 0.0%, amh 0.0%, bam 0.0%, bem 0.0%
- ntrex, SFT, lowest prefix_ok: afr 99.8%, hau 99.8%, wol 99.8%, kin 99.8%
- ntrex, SFT, lowest finished: amh 66.4%, orm 68.9%, ven 69.9%, shi 77.3%
- ntrex, SFT, highest empty: afr 0.0%, amh 0.0%, bem 0.0%, ewe 0.0%
- mafand, SFT, lowest prefix_ok: amh 97.0%, twi 99.9%, hau 100.0%, ibo 100.0%
- mafand, SFT, lowest finished: luo 72.3%, amh 76.2%, twi 84.1%, yor 85.8%
- mafand, SFT, highest empty: luo 0.9%, amh 0.1%, swh 0.1%, hau 0.0%
- smol_sent, SFT, lowest prefix_ok: nbl 0.0%, afr 99.9%, efi 99.9%, ibo 99.9%
- smol_sent, SFT, lowest finished: orm 62.3%, ven 73.3%, efi 76.8%, amh 77.1%
- smol_sent, SFT, highest empty: xho 0.1%, ach 0.0%, afr 0.0%, amh 0.0%
- ibom_mt, SFT, lowest prefix_ok: ibb 0.0%, efi 100.0%
- ibom_mt, SFT, lowest finished: ibb 33.3%, efi 77.1%
- ibom_mt, SFT, highest empty: efi 0.0%, ibb 0.0%
- flores_moz, SFT, lowest prefix_ok: seh 99.9%
- flores_moz, SFT, lowest finished: seh 90.9%
- flores_moz, SFT, highest empty: seh 0.0%
- glot_story, SFT, lowest prefix_ok: nbl 0.0%, dag 99.3%, loz 99.3%, her 100.0%
- glot_story, SFT, lowest finished: tsc 78.2%, her 79.4%, kqn 87.6%, lun 89.3%
- glot_story, SFT, highest empty: dag 0.0%, her 0.0%, kqn 0.0%, kua 0.0%
- salt, SFT, lowest prefix_ok: ach 100.0%, ibo 100.0%, lug 100.0%, nyn 100.0%
- salt, SFT, lowest finished: lug 96.6%, swh 97.4%, ibo 97.6%, ach 99.4%
- salt, SFT, highest empty: ach 0.0%, ibo 0.0%, lug 0.0%, nyn 0.0%

## How the contamination rule changes the macro chrF++

| benchmark | system | all | clean | clean, sentence rule | clean, no near | pair not seen |
|---|---|---|---|---|---|---|
| flores_plus | SFT | 20.20 (46L) | - | - | - | - |
| ntrex | SFT | 23.01 (24L) | - | - | - | - |
| mafand | SFT | 25.56 (15L) | - | - | - | - |
| smol_sent | SFT | 17.18 (45L) | - | - | - | - |
| ibom_mt | SFT | 9.87 (2L) | - | - | - | - |
| flores_moz | SFT | 21.08 (1L) | - | - | - | - |
| glot_story | SFT | 14.62 (8L) | - | - | - | - |
| salt | SFT | 25.13 (5L) | - | - | - | - |

## Runtime

| system | benchmark | rows | sampler | generation s | scoring s (incl. AfriCOMET) |
|---|---|---|---|---|---|
| SFT | flores_plus | 46552 | vllm | 964 | 412 |
| SFT | ntrex | 47928 | vllm | 874 | 376 |
| SFT | mafand | 20009 | vllm | 228 | 194 |
| SFT | smol_sent | 38833 | vllm | 474 | 252 |
| SFT | ibom_mt | 2024 | vllm | 78 | 12 |
| SFT | flores_moz | 1012 | vllm | 16 | 1 |
| SFT | glot_story | 3426 | vllm | 37 | 5 |
| SFT | salt | 2500 | vllm | 13 | 8 |

Whole run: 1280 s.
