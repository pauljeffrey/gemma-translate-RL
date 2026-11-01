# SFT benchmark evaluation (English -> African)

* SFT model: `/dev/shm/lc1_models/final2` revision `main` commit `None` (training step 401, macro validation loss -)
* base model: `-`
* decoding: greedy, same prompt for both (`Translate the text to {language}:  {english}`), max_new_tokens 768, sampler vllm; metrics on NFC-normalised text; spBLEU tokenizer flores200
* started 2026-10-23 20:30:54, finished 2026-10-23 20:48:16, 1042 s; environment {"python": "3.12.3", "platform": "Linux-6.8.0-137-generic-x86_64-with-glibc2.39", "host": "92f224c16546", "torch": "2.14.1+cu130", "transformers": "5.19.0", "huggingface_hub": "2.2.0", "gpu": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB"}

## Results

### Macro averages, clean (headline: English sentence not in the training text)

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|
| **ntrex** | SFT | 24 | 28824 | 35.73 (24) | 14.95 (24) | 11.29 (24) | - | - |
| **mafand** | SFT | 15 | 15634 | 42.00 (15) | 20.33 (15) | 14.86 (15) | - | - |
| **smol_sent** | SFT | 45 | 3420 | 27.25 (45) | 7.13 (45) | 4.79 (45) | - | - |

### Macro averages, all rows

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|
| **ntrex** | SFT | 24 | 47928 | 37.31 (24) | 16.58 (24) | 12.75 (24) | - | - |
| **mafand** | SFT | 15 | 20009 | 42.62 (15) | 20.97 (15) | 15.62 (15) | - | - |
| **smol_sent** | SFT | 45 | 38833 | 31.84 (45) | 10.55 (45) | 7.85 (45) | - | - |

### Macro over every (benchmark, language) pair

| system | chrF++ clean (n) | chrF++ all rows (n) | AfriCOMET clean | AfriCOMET all |
|---|---|---|---|---|
| SFT | 32.31 (84) | 35.32 (84) | - | - |

## Contamination of the benchmarks by the SFT training text

A row is *contaminated* when the normalised English sentence (NFKC, case-folded, punctuation removed) of the benchmark row occurs in the SFT training windows (`gemma/windows_sft`, the input of the packed training set): as a whole sentence of a window or as a contiguous run inside one. *sentence rule* = the first half only (the filter that produced `bench_dev_clean.jsonl`; it reproduces that file exactly: 10,043 of 30,418 dev rows). *near* = not contaminated but >= 50% of its 8-token n-grams occur in the training English. *pair* = contaminated and the reference translation occurs in the target side of a window of the same language (the model saw the very pair).

| benchmark | rows | contaminated | of which sentence-rule hits | near | pair seen | clean rows |
|---|---|---|---|---|---|---|
| ntrex | 47928 | 19104 (39.9%) | 16752 (35.0%) | 144 (0.3%) | 563 (1.2%) | 28824 (60.1%) |
| mafand | 20009 | 4375 (21.9%) | 3960 (19.8%) | 101 (0.5%) | 3096 (15.5%) | 15634 (78.1%) |
| smol_sent | 38833 | 35413 (91.2%) | 35053 (90.3%) | 0 (0.0%) | 2128 (5.5%) | 3420 (8.8%) |

**ntrex**: the English side is the same for every language (multi-way parallel set: FLORES+, NTREX, SmolSent, IbomMT, FLORES-MOZ), so every language has the same 796 of 1997 rows contaminated (39.9%); rows whose reference pair was seen, per language: afr 2, hau 415, ibo 3, ssw 1, tsn 2, xho 2, yor 138.

**mafand** per language:

| language | rows | contaminated | sentence rule | near | pair | clean |
|---|---|---|---|---|---|---|
| Amharic (`amh`) | 1037 | 259 (25.0%) | 246 | 1 | 153 | 778 |
| Hausa (`hau`) | 1500 | 407 (27.1%) | 332 | 19 | 237 | 1093 |
| Igbo (`ibo`) | 1500 | 1369 (91.3%) | 1340 | 0 | 1352 | 131 |
| Kinyarwanda (`kin`) | 1006 | 5 (0.5%) | 5 | 0 | 0 | 1001 |
| Luganda (`lug`) | 1500 | 3 (0.2%) | 3 | 0 | 0 | 1497 |
| Luo (`luo`) | 1500 | 114 (7.6%) | 101 | 13 | 104 | 1386 |
| Chichewa (`nya`) | 1004 | 3 (0.3%) | 3 | 0 | 0 | 1001 |
| Pidgin (`pcm`) | 1564 | 206 (13.2%) | 199 | 6 | 190 | 1358 |
| Shona (`sna`) | 1005 | 5 (0.5%) | 4 | 1 | 0 | 1000 |
| Swahili (`swh`) | 1835 | 219 (11.9%) | 213 | 39 | 121 | 1616 |
| Setswana (`tsn`) | 1500 | 6 (0.4%) | 5 | 3 | 0 | 1494 |
| Twi (`twi`) | 1500 | 14 (0.9%) | 14 | 7 | 0 | 1486 |
| Xhosa (`xho`) | 1002 | 3 (0.3%) | 3 | 0 | 0 | 999 |
| Yoruba (`yor`) | 1558 | 1311 (84.1%) | 1148 | 11 | 748 | 247 |
| Zulu (`zul`) | 998 | 451 (45.2%) | 344 | 1 | 191 | 547 |

**smol_sent** per language:

| language | rows | contaminated | sentence rule | near | pair | clean |
|---|---|---|---|---|---|---|
| Acholi (`ach`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Afrikaans (`afr`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Amharic (`amh`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Bambara (`bam`) | 863 | 787 (91.2%) | 779 | 0 | 564 | 76 |
| Bemba (`bem`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Dyula (`dyu`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Efik (`efi`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Ewe (`ewe`) | 863 | 787 (91.2%) | 779 | 0 | 524 | 76 |
| Fon (`fon`) | 863 | 787 (91.2%) | 779 | 0 | 1 | 76 |
| Hausa (`hau`) | 863 | 787 (91.2%) | 779 | 0 | 72 | 76 |
| Igbo (`ibo`) | 863 | 787 (91.2%) | 779 | 0 | 31 | 76 |
| Kabyle (`kab`) | 862 | 786 (91.2%) | 778 | 0 | 0 | 76 |
| Kikuyu (`kik`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Kinyarwanda (`kin`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Kongo (`kon`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Krio (`kri`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Lingala (`lin`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Luganda (`lug`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Luo (`luo`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Malagasy (`mlg`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Mossi (`mos`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| South Ndebele (`nbl`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Ndau (`ndc`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Sepedi (`nso`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Chichewa (`nya`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Nyankore (`nyn`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Oromo (`orm`) | 863 | 787 (91.2%) | 779 | 0 | 58 | 76 |
| Pidgin (`pcm`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Kirundi (`run`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Shona (`sna`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Somali (`som`) | 862 | 786 (91.2%) | 778 | 0 | 0 | 76 |
| Sesotho (`sot`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Swati (`ssw`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Swahili (`swh`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Tigrinya (`tir`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Tiv (`tiv`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Setswana (`tsn`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Tsonga (`tso`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Tumbuka (`tum`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Twi (`twi`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Venda (`ven`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Wolof (`wol`) | 863 | 787 (91.2%) | 779 | 0 | 630 | 76 |
| Xhosa (`xho`) | 863 | 787 (91.2%) | 779 | 0 | 0 | 76 |
| Yoruba (`yor`) | 863 | 787 (91.2%) | 779 | 0 | 171 | 76 |
| Zulu (`zul`) | 863 | 787 (91.2%) | 779 | 0 | 77 | 76 |

### ntrex: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 1997 | 1201 | 31.53 | 30.10 | 15.38 | 14.30 | 9.19 | 7.84 | - | - | - | - | - |
| Hausa (`hau`) | 1997 | 1201 | 51.07 | 48.75 | 26.44 | 23.75 | 24.17 | 21.60 | - | - | - | - | - |
| Igbo (`ibo`) | 1997 | 1201 | 45.75 | 43.71 | 22.96 | 21.01 | 19.55 | 17.70 | - | - | - | - | - |
| Ewe (`ewe`) | 1997 | 1201 | 31.97 | 30.37 | 10.26 | 8.90 | 7.70 | 6.30 | - | - | - | - | - |
| Wolof (`wol`) | 1997 | 1201 | 29.72 | 28.57 | 9.52 | 8.46 | 6.78 | 5.96 | - | - | - | - | - |
| Swahili (`swh`) | 1997 | 1201 | 59.44 | 56.66 | 38.57 | 34.92 | 35.21 | 31.65 | - | - | - | - | - |
| Kinyarwanda (`kin`) | 1997 | 1201 | 39.15 | 37.36 | 15.73 | 13.99 | 12.94 | 11.24 | - | - | - | - | - |
| Somali (`som`) | 1997 | 1201 | 44.85 | 43.26 | 20.15 | 18.52 | 14.51 | 13.10 | - | - | - | - | - |
| Oromo (`orm`) | 1997 | 1201 | 28.52 | 28.04 | 4.72 | 4.47 | 2.53 | 2.28 | - | - | - | - | - |
| Amharic (`amh`) | 1997 | 1201 | 14.06 | 13.43 | 4.36 | 3.76 | 1.20 | 0.93 | - | - | - | - | - |
| Tigrinya (`tir`) | 1997 | 1201 | 12.42 | 11.91 | 3.27 | 2.92 | 1.03 | 0.93 | - | - | - | - | - |
| Zulu (`zul`) | 1997 | 1201 | 44.95 | 42.75 | 20.80 | 18.17 | 11.64 | 9.80 | - | - | - | - | - |
| Xhosa (`xho`) | 1997 | 1201 | 42.73 | 41.15 | 18.68 | 16.77 | 10.29 | 9.08 | - | - | - | - | - |
| Afrikaans (`afr`) | 1997 | 1201 | 63.21 | 61.31 | 43.11 | 40.84 | 39.53 | 37.35 | - | - | - | - | - |
| Shona (`sna`) | 1997 | 1201 | 40.73 | 38.67 | 16.25 | 14.16 | 12.09 | 10.04 | - | - | - | - | - |
| Chichewa (`nya`) | 1997 | 1201 | 45.67 | 42.86 | 20.47 | 17.42 | 17.77 | 14.75 | - | - | - | - | - |
| Setswana (`tsn`) | 1997 | 1201 | 50.28 | 48.16 | 29.35 | 26.90 | 27.52 | 25.05 | - | - | - | - | - |
| Malagasy (`mlg`) | 1997 | 1201 | 40.44 | 38.84 | 15.51 | 13.98 | 10.85 | 9.69 | - | - | - | - | - |
| Bemba (`bem`) | 1997 | 1201 | 34.89 | 33.21 | 10.16 | 8.72 | 9.00 | 7.32 | - | - | - | - | - |
| Venda (`ven`) | 1997 | 1201 | 40.04 | 38.81 | 17.21 | 15.93 | 11.98 | 10.95 | - | - | - | - | - |
| Sepedi (`nso`) | 1997 | 1201 | 33.14 | 31.83 | 11.59 | 10.26 | 8.97 | 7.76 | - | - | - | - | - |
| Swati (`ssw`) | 1997 | 1201 | 39.37 | 37.77 | 14.50 | 12.87 | 7.96 | 6.52 | - | - | - | - | - |
| North Ndebele (`nde`) | 1997 | 1201 | 30.37 | 29.03 | 8.95 | 7.77 | 3.57 | 2.98 | - | - | - | - | - |
| Tachelhit (`shi`) | 1997 | 1201 | 1.05 | 1.01 | 0.10 | 0.11 | 0.12 | 0.13 | - | - | - | - | - |

### mafand: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 1558 | 247 | 34.41 | 31.23 | 19.30 | 16.52 | 12.63 | 10.32 | - | - | - | - | - |
| Hausa (`hau`) | 1500 | 1093 | 34.32 | 33.57 | 10.61 | 9.92 | 7.67 | 7.15 | - | - | - | - | - |
| Igbo (`ibo`) | 1500 | 131 | 46.64 | 42.87 | 27.32 | 22.37 | 20.00 | 12.89 | - | - | - | - | - |
| Pidgin (`pcm`) | 1564 | 1358 | 60.28 | 60.47 | 38.69 | 39.08 | 32.68 | 32.93 | - | - | - | - | - |
| Twi (`twi`) | 1500 | 1486 | 36.95 | 36.98 | 13.94 | 13.95 | 8.94 | 8.93 | - | - | - | - | - |
| Swahili (`swh`) | 1835 | 1616 | 52.86 | 52.46 | 30.68 | 30.14 | 26.80 | 26.22 | - | - | - | - | - |
| Kinyarwanda (`kin`) | 1006 | 1001 | 49.32 | 49.34 | 24.30 | 24.31 | 19.34 | 19.35 | - | - | - | - | - |
| Amharic (`amh`) | 1037 | 778 | 14.94 | 14.93 | 6.03 | 5.97 | 2.32 | 2.32 | - | - | - | - | - |
| Luganda (`lug`) | 1500 | 1497 | 45.42 | 45.41 | 21.37 | 21.37 | 16.22 | 16.21 | - | - | - | - | - |
| Zulu (`zul`) | 998 | 547 | 47.75 | 46.50 | 24.84 | 23.80 | 15.87 | 14.82 | - | - | - | - | - |
| Xhosa (`xho`) | 1002 | 999 | 52.64 | 52.62 | 29.55 | 29.54 | 18.34 | 18.35 | - | - | - | - | - |
| Shona (`sna`) | 1005 | 1000 | 34.48 | 34.47 | 11.13 | 11.11 | 7.56 | 7.56 | - | - | - | - | - |
| Chichewa (`nya`) | 1004 | 1001 | 42.22 | 42.23 | 13.90 | 13.90 | 10.53 | 10.53 | - | - | - | - | - |
| Setswana (`tsn`) | 1500 | 1494 | 54.34 | 54.34 | 31.73 | 31.73 | 26.72 | 26.71 | - | - | - | - | - |
| Luo (`luo`) | 1500 | 1386 | 32.67 | 32.64 | 11.21 | 11.25 | 8.66 | 8.64 | - | - | - | - | - |

### smol_sent: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 863 | 76 | 29.23 | 25.18 | 13.81 | 10.77 | 7.79 | 4.85 | - | - | - | - | - |
| Hausa (`hau`) | 863 | 76 | 39.54 | 33.92 | 14.24 | 10.32 | 12.86 | 9.23 | - | - | - | - | - |
| Igbo (`ibo`) | 863 | 76 | 45.35 | 37.60 | 26.77 | 20.66 | 22.48 | 14.46 | - | - | - | - | - |
| Pidgin (`pcm`) | 863 | 76 | 51.88 | 54.39 | 25.88 | 33.47 | 22.55 | 26.21 | - | - | - | - | - |
| Efik (`efi`) | 863 | 76 | 26.39 | 23.60 | 7.18 | 4.88 | 3.39 | 1.92 | - | - | - | - | - |
| Twi (`twi`) | 863 | 76 | 29.68 | 26.16 | 7.68 | 5.94 | 6.67 | 4.66 | - | - | - | - | - |
| Ewe (`ewe`) | 863 | 76 | 27.86 | 24.03 | 6.98 | 3.71 | 6.17 | 3.33 | - | - | - | - | - |
| Fon (`fon`) | 863 | 76 | 17.75 | 15.65 | 2.45 | 0.74 | 1.49 | 0.60 | - | - | - | - | - |
| Wolof (`wol`) | 863 | 76 | 25.19 | 22.40 | 7.43 | 5.01 | 4.73 | 2.97 | - | - | - | - | - |
| Swahili (`swh`) | 863 | 76 | 51.46 | 42.71 | 29.48 | 21.07 | 25.76 | 17.14 | - | - | - | - | - |
| Kinyarwanda (`kin`) | 863 | 76 | 35.43 | 28.47 | 13.07 | 8.14 | 8.52 | 3.79 | - | - | - | - | - |
| Somali (`som`) | 862 | 76 | 34.16 | 28.51 | 10.54 | 6.99 | 6.71 | 4.56 | - | - | - | - | - |
| Oromo (`orm`) | 863 | 76 | 29.75 | 24.29 | 6.77 | 2.84 | 3.03 | 1.47 | - | - | - | - | - |
| Amharic (`amh`) | 863 | 76 | 15.73 | 12.99 | 5.24 | 3.35 | 1.49 | 0.72 | - | - | - | - | - |
| Tigrinya (`tir`) | 863 | 76 | 11.83 | 9.32 | 2.81 | 0.74 | 0.97 | 0.69 | - | - | - | - | - |
| Luganda (`lug`) | 863 | 76 | 32.97 | 24.88 | 10.06 | 4.78 | 6.03 | 1.25 | - | - | - | - | - |
| Lingala (`lin`) | 863 | 76 | 36.89 | 31.21 | 8.66 | 4.81 | 8.18 | 4.55 | - | - | - | - | - |
| Zulu (`zul`) | 863 | 76 | 45.10 | 37.78 | 20.58 | 13.63 | 12.56 | 6.22 | - | - | - | - | - |
| Xhosa (`xho`) | 863 | 76 | 38.40 | 32.10 | 13.58 | 7.97 | 7.13 | 3.53 | - | - | - | - | - |
| Afrikaans (`afr`) | 863 | 76 | 57.55 | 46.60 | 36.41 | 25.90 | 31.70 | 19.58 | - | - | - | - | - |
| Shona (`sna`) | 863 | 76 | 35.96 | 29.43 | 10.80 | 5.55 | 7.08 | 3.57 | - | - | - | - | - |
| Chichewa (`nya`) | 863 | 76 | 34.80 | 30.02 | 9.29 | 8.05 | 6.82 | 5.25 | - | - | - | - | - |
| Sesotho (`sot`) | 863 | 76 | 37.91 | 32.67 | 13.37 | 8.31 | 13.12 | 6.46 | - | - | - | - | - |
| Setswana (`tsn`) | 863 | 76 | 45.87 | 39.88 | 24.05 | 16.07 | 22.74 | 14.19 | - | - | - | - | - |
| Malagasy (`mlg`) | 863 | 76 | 38.56 | 32.54 | 13.30 | 8.07 | 8.62 | 4.68 | - | - | - | - | - |
| Bemba (`bem`) | 863 | 76 | 27.70 | 24.20 | 4.05 | 2.94 | 2.69 | 1.64 | - | - | - | - | - |
| Tumbuka (`tum`) | 863 | 76 | 31.78 | 28.12 | 7.97 | 5.86 | 6.04 | 3.67 | - | - | - | - | - |
| Kirundi (`run`) | 863 | 76 | 29.36 | 25.26 | 5.71 | 3.41 | 3.13 | 1.08 | - | - | - | - | - |
| Venda (`ven`) | 863 | 76 | 34.20 | 27.63 | 12.73 | 7.08 | 8.19 | 3.84 | - | - | - | - | - |
| Tsonga (`tso`) | 863 | 76 | 43.18 | 35.27 | 18.32 | 9.66 | 16.06 | 8.24 | - | - | - | - | - |
| Sepedi (`nso`) | 863 | 76 | 41.67 | 35.84 | 18.68 | 11.19 | 17.28 | 9.66 | - | - | - | - | - |
| Swati (`ssw`) | 863 | 76 | 32.78 | 28.61 | 7.66 | 5.08 | 3.53 | 1.30 | - | - | - | - | - |
| South Ndebele (`nbl`) | 863 | 76 | 37.16 | 31.60 | 13.05 | 9.18 | 5.26 | 3.50 | - | - | - | - | - |
| Tiv (`tiv`) | 863 | 76 | 27.15 | 23.55 | 6.39 | 3.80 | 4.08 | 2.36 | - | - | - | - | - |
| Mossi (`mos`) | 863 | 76 | 17.79 | 15.61 | 1.03 | 0.92 | 0.90 | 0.95 | - | - | - | - | - |
| Kikuyu (`kik`) | 863 | 76 | 30.69 | 25.30 | 8.67 | 4.68 | 7.12 | 4.47 | - | - | - | - | - |
| Luo (`luo`) | 863 | 76 | 25.58 | 22.29 | 4.30 | 2.10 | 3.27 | 1.41 | - | - | - | - | - |
| Dyula (`dyu`) | 863 | 76 | 20.08 | 18.07 | 2.10 | 0.95 | 1.81 | 0.98 | - | - | - | - | - |
| Bambara (`bam`) | 863 | 76 | 20.27 | 16.65 | 3.21 | 1.53 | 3.08 | 1.38 | - | - | - | - | - |
| Kongo (`kon`) | 863 | 76 | 16.63 | 15.80 | 1.00 | 0.76 | 0.65 | 0.62 | - | - | - | - | - |
| Kabyle (`kab`) | 862 | 76 | 23.57 | 20.67 | 5.75 | 4.62 | 3.03 | 1.44 | - | - | - | - | - |
| Acholi (`ach`) | 863 | 76 | 26.17 | 23.02 | 3.79 | 2.10 | 2.69 | 0.91 | - | - | - | - | - |
| Nyankore (`nyn`) | 863 | 76 | 25.13 | 21.53 | 2.44 | 0.55 | 1.06 | 0.43 | - | - | - | - | - |
| Ndau (`ndc`) | 863 | 76 | 21.97 | 19.64 | 1.41 | 0.55 | 0.72 | 0.22 | - | - | - | - | - |
| Krio (`kri`) | 863 | 76 | 24.61 | 21.06 | 6.26 | 2.32 | 4.30 | 1.46 | - | - | - | - | - |

## Output format failures

`prefix_ok` = the completion starts with the expected language tag (the base model does not know the convention: its whole output is scored, the tag stripped only if present); `finished` = `<end_of_turn>` produced inside the token budget; `empty` = nothing left after stripping. All rows, unweighted mean over languages; the worst languages follow.

| benchmark | system | prefix_ok | finished | empty |
|---|---|---|---|---|
| ntrex | SFT | 95.8% | 100.0% | 0.0% |
| mafand | SFT | 100.0% | 100.0% | 0.0% |
| smol_sent | SFT | 100.0% | 100.0% | 0.0% |

- ntrex, SFT, lowest prefix_ok: nde 0.0%, ven 99.8%, ewe 99.8%, hau 99.9%
- ntrex, SFT, lowest finished: shi 99.9%, amh 99.9%, afr 100.0%, bem 100.0%
- ntrex, SFT, highest empty: afr 0.0%, amh 0.0%, bem 0.0%, ewe 0.0%
- mafand, SFT, lowest prefix_ok: lug 99.9%, tsn 99.9%, pcm 99.9%, amh 99.9%
- mafand, SFT, lowest finished: amh 100.0%, hau 100.0%, ibo 100.0%, kin 100.0%
- mafand, SFT, highest empty: amh 0.9%, hau 0.0%, ibo 0.0%, kin 0.0%
- smol_sent, SFT, lowest prefix_ok: fon 99.9%, twi 99.9%, ach 100.0%, afr 100.0%
- smol_sent, SFT, lowest finished: ach 100.0%, afr 100.0%, amh 100.0%, bam 100.0%
- smol_sent, SFT, highest empty: twi 0.1%, ach 0.0%, afr 0.0%, amh 0.0%

## How the contamination rule changes the macro chrF++

| benchmark | system | all | clean | clean, sentence rule | clean, no near | pair not seen |
|---|---|---|---|---|---|---|
| ntrex | SFT | 37.31 (24L) | 35.73 (24L) | 36.04 (24L) | 35.71 (24L) | 37.25 (24L) |
| mafand | SFT | 42.62 (15L) | 42.00 (15L) | 42.41 (15L) | 42.02 (15L) | 42.20 (15L) |
| smol_sent | SFT | 31.84 (45L) | 27.25 (45L) | 27.70 (45L) | 27.25 (45L) | 31.64 (45L) |

## Runtime

| system | benchmark | rows | sampler | generation s | scoring s (incl. AfriCOMET) |
|---|---|---|---|---|---|
| SFT | ntrex | 47928 | vllm | 318 | 175 |
| SFT | mafand | 20009 | vllm | 129 | 83 |
| SFT | smol_sent | 38833 | vllm | 212 | 72 |

Whole run: 1042 s.
