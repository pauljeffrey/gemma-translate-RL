# SFT benchmark evaluation (English -> African)

* SFT model: `/workspace/eval_385m/model_m385` revision `main` commit `None`
* base model: `-`
* decoding: greedy, same prompt for both (`Translate the text to {language}:  {english}`), max_new_tokens 768, sampler vllm; metrics on NFC-normalised text; spBLEU tokenizer flores200
* started 2026-11-03 04:28:24, finished 2026-11-03 04:47:26, 1142 s; environment {"python": "3.12.3", "platform": "Linux-5.15.0-187-generic-x86_64-with-glibc2.39", "host": "cca81a4f31dc", "torch": "2.15.0+cu132", "transformers": "4.57.6", "sacrebleu": "2.6.0", "huggingface_hub": "0.36.2", "gpu": "NVIDIA A100 80GB PCIe, 81920 MiB"}

## Results

### Macro averages, clean (headline: English sentence not in the training text)

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|

### Macro averages, all rows

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|
| **flores_plus** | SFT | 54 | 54648 | 22.85 (54) | 6.14 (54) | 4.66 (54) | 0.554 (40) | 0.517 (40) |
| **ntrex** | SFT | 24 | 47928 | 29.25 (24) | 10.08 (24) | 7.61 (24) | 0.551 (21) | 0.512 (21) |
| **mafand** | SFT | 15 | 20009 | 35.77 (15) | 14.40 (15) | 10.24 (15) | 0.616 (15) | 0.586 (15) |
| **smol_sent** | SFT | 45 | 38833 | 22.53 (45) | 5.05 (45) | 3.76 (45) | 0.531 (36) | 0.482 (36) |
| **ibom_mt** | SFT | 2 | 2024 | 12.04 (2) | 1.80 (2) | 1.15 (2) | 0.519 (1) | 0.450 (1) |
| **flores_moz** | SFT | 1 | 1012 | 21.54 (1) | 2.30 (1) | 1.66 (1) | - | - |
| **glot_story** | SFT | 8 | 3426 | 20.22 (8) | 3.51 (8) | 1.94 (8) | - | - |
| **salt** | SFT | 5 | 2500 | 35.49 (5) | 14.14 (5) | 10.06 (5) | 0.683 (3) | 0.680 (3) |

## Contamination of the benchmarks by the SFT training text

A row is *contaminated* when the normalised English sentence (NFKC, case-folded, punctuation removed) of the benchmark row occurs in the SFT training windows (`gemma/windows_sft`, the input of the packed training set): as a whole sentence of a window or as a contiguous run inside one. *sentence rule* = the first half only (the filter that produced `bench_dev_clean.jsonl`; it reproduces that file exactly: 10,043 of 30,418 dev rows). *near* = not contaminated but >= 50% of its 8-token n-grams occur in the training English. *pair* = contaminated and the reference translation occurs in the target side of a window of the same language (the model saw the very pair).

| benchmark | rows | contaminated | of which sentence-rule hits | near | pair seen | clean rows |
|---|---|---|---|---|---|---|
| flores_plus | 54648 | - | - | - | - | - |
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
| Yoruba (`yor`) | 1012 | - | 21.32 | - | 6.18 | - | 3.15 | - | 0.589 | - | - | - | - |
| Hausa (`hau`) | 1012 | - | 39.88 | - | 14.30 | - | 12.69 | - | 0.616 | - | - | - | - |
| Igbo (`ibo`) | 1012 | - | 37.82 | - | 16.71 | - | 13.70 | - | 0.645 | - | - | - | - |
| Twi (`twi`) | 1012 | - | 27.54 | - | 6.16 | - | 4.66 | - | 0.594 | - | - | - | - |
| Ewe (`ewe`) | 1012 | - | 25.96 | - | 5.76 | - | 4.46 | - | 0.562 | - | - | - | - |
| Fon (`fon`) | 1012 | - | 20.17 | - | 4.07 | - | 2.06 | - | 0.573 | - | - | - | - |
| Wolof (`wol`) | 1012 | - | 26.87 | - | 7.55 | - | 5.61 | - | 0.628 | - | - | - | - |
| Swahili (`swh`) | 1012 | - | 56.24 | - | 33.08 | - | 28.27 | - | 0.693 | - | - | - | - |
| Kinyarwanda (`kin`) | 1012 | - | 25.81 | - | 4.28 | - | 3.94 | - | 0.564 | - | - | - | - |
| Somali (`som`) | 1012 | - | 35.47 | - | 10.14 | - | 6.62 | - | 0.590 | - | - | - | - |
| Oromo (`orm`) | 1012 | - | 12.06 | - | 0.62 | - | 0.29 | - | 0.419 | - | - | - | - |
| Amharic (`amh`) | 1012 | - | 16.62 | - | 4.74 | - | 1.73 | - | 0.504 | - | - | - | - |
| Tigrinya (`tir`) | 1012 | - | 10.48 | - | 2.32 | - | 0.58 | - | 0.507 | - | - | - | - |
| Luganda (`lug`) | 1012 | - | 16.17 | - | 1.30 | - | 1.15 | - | 0.465 | - | - | - | - |
| Lingala (`lin`) | 1012 | - | 28.56 | - | 4.63 | - | 3.66 | - | 0.520 | - | - | - | - |
| Zulu (`zul`) | 1012 | - | 45.91 | - | 22.23 | - | 13.67 | - | 0.660 | - | - | - | - |
| Xhosa (`xho`) | 1012 | - | 43.56 | - | 19.02 | - | 11.29 | - | 0.627 | - | - | - | - |
| Afrikaans (`afr`) | 1012 | - | 63.13 | - | 43.02 | - | 37.95 | - | 0.745 | - | - | - | - |
| Shona (`sna`) | 1012 | - | 33.92 | - | 9.94 | - | 6.55 | - | 0.596 | - | - | - | - |
| Chichewa (`nya`) | 1012 | - | 36.88 | - | 10.64 | - | 8.53 | - | 0.625 | - | - | - | - |
| Sesotho (`sot`) | 1012 | - | 37.87 | - | 12.61 | - | 10.76 | - | 0.615 | - | - | - | - |
| Setswana (`tsn`) | 1012 | - | 37.69 | - | 11.85 | - | 10.28 | - | 0.634 | - | - | - | - |
| Malagasy (`mlg`) | 1012 | - | 39.04 | - | 12.15 | - | 9.07 | - | 0.593 | - | - | - | - |
| Bemba (`bem`) | 1012 | - | 19.31 | - | 2.15 | - | 1.56 | - | 0.450 | - | - | - | - |
| Tumbuka (`tum`) | 1012 | - | 25.40 | - | 5.64 | - | 4.42 | - | 0.532 | - | - | - | - |
| Kirundi (`run`) | 1012 | - | 21.59 | - | 2.78 | - | 2.31 | - | 0.531 | - | - | - | - |
| Tsonga (`tso`) | 1012 | - | 32.92 | - | 7.84 | - | 6.73 | - | 0.597 | - | - | - | - |
| Sepedi (`nso`) | 1012 | - | 31.58 | - | 7.43 | - | 6.50 | - | 0.586 | - | - | - | - |
| Swati (`ssw`) | 1012 | - | 22.20 | - | 3.04 | - | 1.38 | - | 0.493 | - | - | - | - |
| Sango (`sag`) | 1012 | - | 19.68 | - | 1.77 | - | 1.56 | - | - | - | - | - | - |
| Umbundu (`umb`) | 1012 | - | 14.80 | - | 0.62 | - | 0.40 | - | 0.470 | - | - | - | - |
| Kimbundu (`kmb`) | 1012 | - | 16.02 | - | 1.21 | - | 0.73 | - | 0.419 | - | - | - | - |
| Mossi (`mos`) | 1012 | - | 14.09 | - | 1.26 | - | 0.92 | - | 0.459 | - | - | - | - |
| Kikuyu (`kik`) | 1012 | - | 25.98 | - | 5.64 | - | 4.08 | - | 0.597 | - | - | - | - |
| Luo (`luo`) | 1012 | - | 21.36 | - | 2.97 | - | 2.10 | - | 0.562 | - | - | - | - |
| Kamba (`kam`) | 1012 | - | 17.22 | - | 2.08 | - | 1.55 | - | 0.440 | - | - | - | - |
| Dyula (`dyu`) | 1012 | - | 9.90 | - | 0.72 | - | 0.54 | - | - | - | - | - | - |
| Bambara (`bam`) | 1012 | - | 19.39 | - | 2.86 | - | 2.12 | - | 0.452 | - | - | - | - |
| Kabiye (`kbp`) | 1012 | - | 14.51 | - | 1.58 | - | 1.15 | - | 0.538 | - | - | - | - |
| Kabuverdianu (`kea`) | 1012 | - | 18.91 | - | 2.90 | - | 1.96 | - | - | - | - | - | - |
| Kanuri (`knc`) | 1012 | - | 3.81 | - | 0.15 | - | 0.11 | - | - | - | - | - | - |
| Kabyle (`kab`) | 1012 | - | 16.35 | - | 1.81 | - | 1.33 | - | 0.558 | - | - | - | - |
| Nuer (`nus`) | 1012 | - | 13.49 | - | 2.12 | - | 0.77 | - | 0.474 | - | - | - | - |
| Luba-Kasai (`lua`) | 1012 | - | 17.18 | - | 1.21 | - | 0.82 | - | 0.430 | - | - | - | - |
| Makhuwa (`vmw`) | 1012 | - | 17.45 | - | 1.82 | - | 1.06 | - | - | - | - | - | - |
| Sena (`seh`) | 1012 | - | 21.67 | - | 2.36 | - | 1.69 | - | - | - | - | - | - |
| Mauritian Creole (`mfe`) | 1012 | - | 28.51 | - | 4.96 | - | 4.25 | - | - | - | - | - | - |
| Egyptian Arabic (`arz`) | 1012 | - | 4.59 | - | 0.11 | - | 0.04 | - | - | - | - | - | - |
| Chokwe (`cjk`) | 1012 | - | 10.72 | - | 0.58 | - | 0.35 | - | - | - | - | - | - |
| Dinka (`dik`) | 1012 | - | 3.76 | - | 0.23 | - | 0.11 | - | - | - | - | - | - |
| Tamasheq (`taq`) | 1012 | - | 2.81 | - | 0.09 | - | 0.08 | - | - | - | - | - | - |
| Standard Moroccan Tamazight (`zgh`) | 1012 | - | 2.29 | - | 0.13 | - | 0.05 | - | - | - | - | - | - |
| Moroccan Arabic (`ary`) | 1012 | - | 4.64 | - | 0.11 | - | 0.07 | - | - | - | - | - | - |
| Tunisian Arabic (`aeb`) | 1012 | - | 2.75 | - | 0.08 | - | 0.05 | - | - | - | - | - | - |

### ntrex: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 1997 | - | 23.74 | - | 7.85 | - | 4.31 | - | 0.597 | - | - | - | - |
| Hausa (`hau`) | 1997 | - | 42.15 | - | 16.14 | - | 14.11 | - | 0.597 | - | - | - | - |
| Igbo (`ibo`) | 1997 | - | 39.90 | - | 14.56 | - | 13.03 | - | 0.645 | - | - | - | - |
| Ewe (`ewe`) | 1997 | - | 24.28 | - | 4.71 | - | 3.39 | - | 0.519 | - | - | - | - |
| Wolof (`wol`) | 1997 | - | 25.29 | - | 5.70 | - | 4.22 | - | 0.593 | - | - | - | - |
| Swahili (`swh`) | 1997 | - | 55.18 | - | 31.42 | - | 27.52 | - | 0.646 | - | - | - | - |
| Kinyarwanda (`kin`) | 1997 | - | 26.40 | - | 4.96 | - | 4.72 | - | 0.551 | - | - | - | - |
| Somali (`som`) | 1997 | - | 35.84 | - | 9.94 | - | 6.82 | - | 0.549 | - | - | - | - |
| Oromo (`orm`) | 1997 | - | 9.63 | - | 0.31 | - | 0.16 | - | 0.395 | - | - | - | - |
| Amharic (`amh`) | 1997 | - | 10.58 | - | 2.30 | - | 0.54 | - | 0.444 | - | - | - | - |
| Tigrinya (`tir`) | 1997 | - | 9.10 | - | 1.73 | - | 0.46 | - | 0.488 | - | - | - | - |
| Zulu (`zul`) | 1997 | - | 40.92 | - | 15.80 | - | 8.63 | - | 0.600 | - | - | - | - |
| Xhosa (`xho`) | 1997 | - | 39.85 | - | 15.29 | - | 8.40 | - | 0.584 | - | - | - | - |
| Afrikaans (`afr`) | 1997 | - | 62.41 | - | 42.99 | - | 38.59 | - | 0.682 | - | - | - | - |
| Shona (`sna`) | 1997 | - | 34.24 | - | 11.02 | - | 7.38 | - | 0.573 | - | - | - | - |
| Chichewa (`nya`) | 1997 | - | 37.87 | - | 13.04 | - | 10.09 | - | 0.588 | - | - | - | - |
| Setswana (`tsn`) | 1997 | - | 36.96 | - | 12.31 | - | 10.97 | - | 0.577 | - | - | - | - |
| Malagasy (`mlg`) | 1997 | - | 36.91 | - | 12.75 | - | 8.80 | - | 0.547 | - | - | - | - |
| Bemba (`bem`) | 1997 | - | 18.68 | - | 2.00 | - | 1.51 | - | 0.433 | - | - | - | - |
| Venda (`ven`) | 1997 | - | 21.29 | - | 3.35 | - | 2.25 | - | - | - | - | - | - |
| Sepedi (`nso`) | 1997 | - | 20.14 | - | 2.93 | - | 2.18 | - | 0.487 | - | - | - | - |
| Swati (`ssw`) | 1997 | - | 21.58 | - | 3.11 | - | 1.50 | - | 0.471 | - | - | - | - |
| North Ndebele (`nde`) | 1997 | - | 28.83 | - | 7.76 | - | 3.11 | - | - | - | - | - | - |
| Tachelhit (`shi`) | 1997 | - | 0.24 | - | 0.01 | - | 0.01 | - | - | - | - | - | - |

### mafand: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 1558 | - | 23.46 | - | 9.21 | - | 5.07 | - | 0.614 | - | - | - | - |
| Hausa (`hau`) | 1500 | - | 32.03 | - | 8.98 | - | 6.59 | - | 0.630 | - | - | - | - |
| Igbo (`ibo`) | 1500 | - | 43.59 | - | 22.92 | - | 15.70 | - | 0.700 | - | - | - | - |
| Pidgin (`pcm`) | 1564 | - | 53.58 | - | 30.78 | - | 25.21 | - | 0.639 | - | - | - | - |
| Twi (`twi`) | 1500 | - | 28.58 | - | 6.40 | - | 3.82 | - | 0.601 | - | - | - | - |
| Swahili (`swh`) | 1835 | - | 50.58 | - | 28.02 | - | 23.06 | - | 0.680 | - | - | - | - |
| Kinyarwanda (`kin`) | 1006 | - | 36.50 | - | 10.89 | - | 8.71 | - | 0.623 | - | - | - | - |
| Amharic (`amh`) | 1037 | - | 11.64 | - | 2.86 | - | 0.99 | - | 0.456 | - | - | - | - |
| Luganda (`lug`) | 1500 | - | 26.97 | - | 5.60 | - | 4.23 | - | 0.558 | - | - | - | - |
| Zulu (`zul`) | 998 | - | 44.50 | - | 22.13 | - | 13.80 | - | 0.610 | - | - | - | - |
| Xhosa (`xho`) | 1002 | - | 49.77 | - | 26.60 | - | 15.99 | - | 0.672 | - | - | - | - |
| Shona (`sna`) | 1005 | - | 30.36 | - | 10.18 | - | 6.36 | - | 0.583 | - | - | - | - |
| Chichewa (`nya`) | 1004 | - | 38.14 | - | 10.77 | - | 8.07 | - | 0.627 | - | - | - | - |
| Setswana (`tsn`) | 1500 | - | 43.85 | - | 16.66 | - | 13.26 | - | 0.667 | - | - | - | - |
| Luo (`luo`) | 1500 | - | 22.95 | - | 3.97 | - | 2.80 | - | 0.579 | - | - | - | - |

### smol_sent: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 863 | - | 16.63 | - | 3.97 | - | 1.82 | - | 0.575 | - | - | - | - |
| Hausa (`hau`) | 863 | - | 28.18 | - | 4.88 | - | 4.41 | - | 0.543 | - | - | - | - |
| Igbo (`ibo`) | 863 | - | 35.67 | - | 12.97 | - | 11.60 | - | 0.632 | - | - | - | - |
| Pidgin (`pcm`) | 863 | - | 46.17 | - | 21.55 | - | 18.18 | - | 0.552 | - | - | - | - |
| Efik (`efi`) | 863 | - | 17.89 | - | 1.98 | - | 0.98 | - | 0.522 | - | - | - | - |
| Twi (`twi`) | 863 | - | 20.49 | - | 2.33 | - | 1.88 | - | 0.560 | - | - | - | - |
| Ewe (`ewe`) | 863 | - | 20.63 | - | 2.77 | - | 2.27 | - | 0.543 | - | - | - | - |
| Fon (`fon`) | 863 | - | 15.50 | - | 1.61 | - | 1.03 | - | 0.553 | - | - | - | - |
| Wolof (`wol`) | 863 | - | 21.30 | - | 4.44 | - | 2.75 | - | 0.593 | - | - | - | - |
| Swahili (`swh`) | 863 | - | 48.55 | - | 24.44 | - | 20.06 | - | 0.626 | - | - | - | - |
| Kinyarwanda (`kin`) | 863 | - | 18.76 | - | 1.87 | - | 1.09 | - | 0.530 | - | - | - | - |
| Somali (`som`) | 862 | - | 23.48 | - | 3.08 | - | 1.84 | - | 0.495 | - | - | - | - |
| Oromo (`orm`) | 863 | - | 8.47 | - | 0.32 | - | 0.12 | - | 0.392 | - | - | - | - |
| Amharic (`amh`) | 863 | - | 10.33 | - | 1.89 | - | 0.49 | - | 0.440 | - | - | - | - |
| Tigrinya (`tir`) | 863 | - | 6.16 | - | 0.85 | - | 0.21 | - | 0.453 | - | - | - | - |
| Luganda (`lug`) | 863 | - | 14.56 | - | 0.97 | - | 0.52 | - | 0.474 | - | - | - | - |
| Lingala (`lin`) | 863 | - | 22.53 | - | 2.03 | - | 1.83 | - | 0.485 | - | - | - | - |
| Zulu (`zul`) | 863 | - | 40.60 | - | 15.39 | - | 8.95 | - | 0.608 | - | - | - | - |
| Xhosa (`xho`) | 863 | - | 36.64 | - | 11.55 | - | 5.81 | - | 0.575 | - | - | - | - |
| Afrikaans (`afr`) | 863 | - | 58.36 | - | 37.43 | - | 32.27 | - | 0.704 | - | - | - | - |
| Shona (`sna`) | 863 | - | 27.53 | - | 4.76 | - | 3.21 | - | 0.552 | - | - | - | - |
| Chichewa (`nya`) | 863 | - | 25.59 | - | 4.05 | - | 2.72 | - | 0.550 | - | - | - | - |
| Sesotho (`sot`) | 863 | - | 25.92 | - | 4.26 | - | 4.04 | - | 0.542 | - | - | - | - |
| Setswana (`tsn`) | 863 | - | 34.30 | - | 9.24 | - | 8.52 | - | 0.605 | - | - | - | - |
| Malagasy (`mlg`) | 863 | - | 31.59 | - | 7.10 | - | 4.43 | - | 0.528 | - | - | - | - |
| Bemba (`bem`) | 863 | - | 13.79 | - | 0.44 | - | 0.23 | - | 0.417 | - | - | - | - |
| Tumbuka (`tum`) | 863 | - | 19.52 | - | 2.13 | - | 1.48 | - | 0.484 | - | - | - | - |
| Kirundi (`run`) | 863 | - | 15.10 | - | 0.92 | - | 0.45 | - | 0.499 | - | - | - | - |
| Venda (`ven`) | 863 | - | 18.44 | - | 2.52 | - | 1.55 | - | - | - | - | - | - |
| Tsonga (`tso`) | 863 | - | 29.86 | - | 6.27 | - | 5.35 | - | 0.579 | - | - | - | - |
| Sepedi (`nso`) | 863 | - | 22.45 | - | 3.53 | - | 3.15 | - | 0.530 | - | - | - | - |
| Swati (`ssw`) | 863 | - | 18.22 | - | 1.34 | - | 0.55 | - | 0.480 | - | - | - | - |
| South Ndebele (`nbl`) | 863 | - | 31.11 | - | 7.41 | - | 2.70 | - | - | - | - | - | - |
| Tiv (`tiv`) | 863 | - | 13.29 | - | 1.06 | - | 0.64 | - | - | - | - | - | - |
| Mossi (`mos`) | 863 | - | 11.47 | - | 0.43 | - | 0.30 | - | 0.466 | - | - | - | - |
| Kikuyu (`kik`) | 863 | - | 21.15 | - | 2.88 | - | 2.26 | - | 0.568 | - | - | - | - |
| Luo (`luo`) | 863 | - | 15.99 | - | 0.95 | - | 0.66 | - | 0.532 | - | - | - | - |
| Dyula (`dyu`) | 863 | - | 10.50 | - | 0.76 | - | 0.73 | - | - | - | - | - | - |
| Bambara (`bam`) | 863 | - | 13.31 | - | 1.35 | - | 1.09 | - | 0.394 | - | - | - | - |
| Kongo (`kon`) | 863 | - | 22.37 | - | 2.30 | - | 2.26 | - | - | - | - | - | - |
| Kabyle (`kab`) | 862 | - | 11.56 | - | 0.75 | - | 0.34 | - | 0.535 | - | - | - | - |
| Acholi (`ach`) | 863 | - | 21.51 | - | 3.02 | - | 2.18 | - | - | - | - | - | - |
| Nyankore (`nyn`) | 863 | - | 18.73 | - | 1.19 | - | 0.50 | - | - | - | - | - | - |
| Ndau (`ndc`) | 863 | - | 15.59 | - | 0.49 | - | 0.23 | - | - | - | - | - | - |
| Krio (`kri`) | 863 | - | 14.26 | - | 1.85 | - | 1.29 | - | - | - | - | - | - |

### ibom_mt: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Efik (`efi`) | 1012 | - | 21.80 | - | 3.54 | - | 2.27 | - | 0.519 | - | - | - | - |
| Ibibio (`ibb`) | 1012 | - | 2.27 | - | 0.07 | - | 0.03 | - | - | - | - | - | - |

### flores_moz: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sena (`seh`) | 1012 | - | 21.54 | - | 2.30 | - | 1.66 | - | - | - | - | - | - |

### glot_story: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| South Ndebele (`nbl`) | 321 | - | 28.06 | - | 5.81 | - | 2.89 | - | - | - | - | - | - |
| Dagbani (`dag`) | 135 | - | 2.43 | - | 0.06 | - | 0.02 | - | - | - | - | - | - |
| Kaonde (`kqn`) | 420 | - | 19.87 | - | 3.00 | - | 1.40 | - | - | - | - | - | - |
| Lunda (`lun`) | 420 | - | 21.99 | - | 3.68 | - | 1.69 | - | - | - | - | - | - |
| Lozi (`loz`) | 960 | - | 26.69 | - | 4.96 | - | 3.01 | - | - | - | - | - | - |
| Tswa (`tsc`) | 101 | - | 16.28 | - | 1.28 | - | 0.89 | - | - | - | - | - | - |
| Kuanyama (`kua`) | 520 | - | 28.50 | - | 6.70 | - | 3.90 | - | - | - | - | - | - |
| Herero (`her`) | 549 | - | 17.92 | - | 2.59 | - | 1.69 | - | - | - | - | - | - |

### salt: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Igbo (`ibo`) | 500 | - | 32.40 | - | 10.70 | - | 7.40 | - | 0.696 | - | - | - | - |
| Swahili (`swh`) | 500 | - | 56.76 | - | 34.35 | - | 29.27 | - | 0.753 | - | - | - | - |
| Luganda (`lug`) | 500 | - | 29.30 | - | 6.91 | - | 3.55 | - | 0.599 | - | - | - | - |
| Acholi (`ach`) | 500 | - | 28.42 | - | 9.36 | - | 5.76 | - | - | - | - | - | - |
| Nyankore (`nyn`) | 500 | - | 30.58 | - | 9.39 | - | 4.31 | - | - | - | - | - | - |

## Output format failures

`prefix_ok` = the completion starts with the expected language tag (the base model does not know the convention: its whole output is scored, the tag stripped only if present); `finished` = `<end_of_turn>` produced inside the token budget; `empty` = nothing left after stripping. All rows, unweighted mean over languages; the worst languages follow.

| benchmark | system | prefix_ok | finished | empty |
|---|---|---|---|---|
| flores_plus | SFT | 92.6% | 82.9% | 0.0% |
| ntrex | SFT | 100.0% | 89.0% | 0.0% |
| mafand | SFT | 100.0% | 96.3% | 0.0% |
| smol_sent | SFT | 100.0% | 90.4% | 0.0% |
| ibom_mt | SFT | 50.0% | 53.1% | 0.0% |
| flores_moz | SFT | 99.7% | 91.7% | 0.0% |
| glot_story | SFT | 95.9% | 92.7% | 0.0% |
| salt | SFT | 100.0% | 99.2% | 0.0% |

- flores_plus, SFT, lowest prefix_ok: aeb 0.0%, ary 0.0%, zgh 0.0%, taq 0.2%
- flores_plus, SFT, lowest finished: zgh 11.9%, taq 14.0%, dik 29.0%, knc 46.4%
- flores_plus, SFT, highest empty: aeb 0.0%, afr 0.0%, amh 0.0%, ary 0.0%
- ntrex, SFT, lowest prefix_ok: afr 100.0%, amh 100.0%, bem 100.0%, ewe 100.0%
- ntrex, SFT, lowest finished: shi 40.6%, orm 59.8%, tir 75.3%, ven 77.2%
- ntrex, SFT, highest empty: afr 0.0%, amh 0.0%, bem 0.0%, ewe 0.0%
- mafand, SFT, lowest prefix_ok: amh 100.0%, hau 100.0%, ibo 100.0%, kin 100.0%
- mafand, SFT, lowest finished: amh 89.1%, luo 92.1%, yor 93.8%, lug 93.9%
- mafand, SFT, highest empty: amh 0.0%, hau 0.0%, ibo 0.0%, kin 0.0%
- smol_sent, SFT, lowest prefix_ok: ndc 99.0%, kri 99.8%, ach 100.0%, afr 100.0%
- smol_sent, SFT, lowest finished: orm 57.6%, tir 66.9%, ven 80.5%, dyu 81.2%
- smol_sent, SFT, highest empty: ach 0.0%, afr 0.0%, amh 0.0%, bam 0.0%
- ibom_mt, SFT, lowest prefix_ok: ibb 0.0%, efi 100.0%
- ibom_mt, SFT, lowest finished: ibb 16.5%, efi 89.6%
- ibom_mt, SFT, highest empty: efi 0.0%, ibb 0.0%
- flores_moz, SFT, lowest prefix_ok: seh 99.7%
- flores_moz, SFT, lowest finished: seh 91.7%
- flores_moz, SFT, highest empty: seh 0.0%
- glot_story, SFT, lowest prefix_ok: dag 0.0%, tsc 96.0%, her 100.0%, kqn 100.0%
- glot_story, SFT, lowest finished: dag 69.6%, her 86.9%, kqn 90.7%, lun 93.3%
- glot_story, SFT, highest empty: dag 0.0%, her 0.0%, kqn 0.0%, kua 0.0%
- salt, SFT, lowest prefix_ok: ach 100.0%, ibo 100.0%, lug 100.0%, nyn 100.0%
- salt, SFT, lowest finished: lug 98.4%, ibo 99.0%, nyn 99.2%, ach 99.6%
- salt, SFT, highest empty: ach 0.0%, ibo 0.0%, lug 0.0%, nyn 0.0%

## How the contamination rule changes the macro chrF++

| benchmark | system | all | clean | clean, sentence rule | clean, no near | pair not seen |
|---|---|---|---|---|---|---|
| flores_plus | SFT | 22.85 (54L) | - | - | - | - |
| ntrex | SFT | 29.25 (24L) | - | - | - | - |
| mafand | SFT | 35.77 (15L) | - | - | - | - |
| smol_sent | SFT | 22.53 (45L) | - | - | - | - |
| ibom_mt | SFT | 12.04 (2L) | - | - | - | - |
| flores_moz | SFT | 21.54 (1L) | - | - | - | - |
| glot_story | SFT | 20.22 (8L) | - | - | - | - |
| salt | SFT | 35.49 (5L) | - | - | - | - |

## Runtime

| system | benchmark | rows | sampler | generation s | scoring s (incl. AfriCOMET) |
|---|---|---|---|---|---|
| SFT | flores_plus | 54648 | vllm | 1478 | 445 |
| SFT | ntrex | 47928 | vllm | 874 | 350 |
| SFT | mafand | 20009 | vllm | 200 | 121 |
| SFT | smol_sent | 38833 | vllm | 561 | 177 |
| SFT | ibom_mt | 2024 | vllm | 94 | 11 |
| SFT | flores_moz | 1012 | vllm | 23 | 1 |
| SFT | glot_story | 3426 | vllm | 59 | 4 |
| SFT | salt | 2500 | vllm | 19 | 6 |

Whole run: 1142 s.
