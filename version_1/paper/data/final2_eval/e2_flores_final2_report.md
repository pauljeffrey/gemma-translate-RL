# SFT benchmark evaluation (English -> African)

* SFT model: `/dev/shm/lc1_models/final2` revision `main` commit `None` (training step 401, macro validation loss -)
* base model: `-`
* decoding: greedy, same prompt for both (`Translate the text to {language}:  {english}`), max_new_tokens 768, sampler vllm; metrics on NFC-normalised text; spBLEU tokenizer flores200
* started 2026-10-23 20:18:31, finished 2026-10-23 20:24:38, 367 s; environment {"python": "3.12.3", "platform": "Linux-6.8.0-137-generic-x86_64-with-glibc2.39", "host": "92f224c16546", "torch": "2.14.1+cu130", "transformers": "5.19.0", "huggingface_hub": "2.2.0", "gpu": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97887 MiB"}

## Results

### Macro averages, clean (headline: English sentence not in the training text)

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|
| **flores_plus** | SFT | 45 | 2160 | 31.83 (45) | 10.48 (45) | 8.07 (45) | - | - |

### Macro averages, all rows

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|
| **flores_plus** | SFT | 45 | 45540 | 34.52 (45) | 12.89 (45) | 10.04 (45) | - | - |

### Macro over every (benchmark, language) pair

| system | chrF++ clean (n) | chrF++ all rows (n) | AfriCOMET clean | AfriCOMET all |
|---|---|---|---|---|
| SFT | 31.83 (45) | 34.52 (45) | - | - |

## Contamination of the benchmarks by the SFT training text

A row is *contaminated* when the normalised English sentence (NFKC, case-folded, punctuation removed) of the benchmark row occurs in the SFT training windows (`gemma/windows_sft`, the input of the packed training set): as a whole sentence of a window or as a contiguous run inside one. *sentence rule* = the first half only (the filter that produced `bench_dev_clean.jsonl`; it reproduces that file exactly: 10,043 of 30,418 dev rows). *near* = not contaminated but >= 50% of its 8-token n-grams occur in the training English. *pair* = contaminated and the reference translation occurs in the target side of a window of the same language (the model saw the very pair).

| benchmark | rows | contaminated | of which sentence-rule hits | near | pair seen | clean rows |
|---|---|---|---|---|---|---|
| flores_plus | 45540 | 43380 (95.3%) | 38250 (84.0%) | 0 (0.0%) | 1848 (4.1%) | 2160 (4.7%) |

**flores_plus**: the English side is the same for every language (multi-way parallel set: FLORES+, NTREX, SmolSent, IbomMT, FLORES-MOZ), so every language has the same 964 of 1012 rows contaminated (95.3%); rows whose reference pair was seen, per language: hau 842, ibo 559, lug 2, orm 24, tsn 30, yor 355, zul 36.

### flores_plus: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Yoruba (`yor`) | 1012 | 48 | 30.43 | 28.73 | 14.32 | 14.44 | 8.19 | 6.69 | - | - | - | - | - |
| Hausa (`hau`) | 1012 | 48 | 50.12 | 44.95 | 26.49 | 19.55 | 24.42 | 18.07 | - | - | - | - | - |
| Igbo (`ibo`) | 1012 | 48 | 41.23 | 38.34 | 20.36 | 19.67 | 17.04 | 16.81 | - | - | - | - | - |
| Twi (`twi`) | 1012 | 48 | 37.43 | 34.22 | 14.53 | 12.74 | 11.93 | 10.47 | - | - | - | - | - |
| Ewe (`ewe`) | 1012 | 48 | 31.50 | 27.48 | 10.00 | 7.37 | 8.21 | 5.44 | - | - | - | - | - |
| Fon (`fon`) | 1012 | 48 | 21.65 | 20.09 | 4.85 | 2.93 | 2.54 | 1.65 | - | - | - | - | - |
| Wolof (`wol`) | 1012 | 48 | 29.87 | 28.67 | 10.44 | 9.72 | 7.68 | 6.64 | - | - | - | - | - |
| Swahili (`swh`) | 1012 | 48 | 58.63 | 51.97 | 36.07 | 27.68 | 32.76 | 25.33 | - | - | - | - | - |
| Kinyarwanda (`kin`) | 1012 | 48 | 42.15 | 39.37 | 17.88 | 15.44 | 14.49 | 12.37 | - | - | - | - | - |
| Somali (`som`) | 1012 | 48 | 40.66 | 36.00 | 15.91 | 11.09 | 10.85 | 8.23 | - | - | - | - | - |
| Oromo (`orm`) | 1012 | 48 | 31.75 | 28.65 | 7.06 | 5.70 | 3.78 | 3.78 | - | - | - | - | - |
| Amharic (`amh`) | 1012 | 48 | 20.00 | 16.95 | 7.98 | 4.51 | 2.97 | 2.10 | - | - | - | - | - |
| Tigrinya (`tir`) | 1012 | 48 | 13.88 | 11.65 | 4.03 | 2.59 | 1.42 | 0.99 | - | - | - | - | - |
| Luganda (`lug`) | 1012 | 48 | 35.80 | 30.93 | 10.61 | 6.84 | 7.83 | 3.88 | - | - | - | - | - |
| Lingala (`lin`) | 1012 | 48 | 42.57 | 39.60 | 14.38 | 11.53 | 12.74 | 11.06 | - | - | - | - | - |
| Zulu (`zul`) | 1012 | 48 | 49.42 | 43.40 | 25.60 | 20.46 | 15.71 | 11.50 | - | - | - | - | - |
| Xhosa (`xho`) | 1012 | 48 | 45.17 | 43.75 | 20.39 | 20.64 | 11.96 | 11.74 | - | - | - | - | - |
| Afrikaans (`afr`) | 1012 | 48 | 63.60 | 61.86 | 43.00 | 41.22 | 38.36 | 38.41 | - | - | - | - | - |
| Shona (`sna`) | 1012 | 48 | 39.63 | 35.50 | 14.81 | 11.02 | 10.18 | 7.48 | - | - | - | - | - |
| Chichewa (`nya`) | 1012 | 48 | 43.44 | 40.83 | 16.42 | 13.73 | 13.49 | 10.87 | - | - | - | - | - |
| Sesotho (`sot`) | 1012 | 48 | 45.55 | 41.23 | 20.34 | 16.25 | 18.95 | 15.31 | - | - | - | - | - |
| Setswana (`tsn`) | 1012 | 48 | 46.19 | 41.59 | 22.26 | 17.07 | 20.15 | 15.00 | - | - | - | - | - |
| Malagasy (`mlg`) | 1012 | 48 | 43.51 | 39.67 | 16.06 | 12.41 | 12.51 | 11.01 | - | - | - | - | - |
| Bemba (`bem`) | 1012 | 48 | 34.34 | 32.51 | 9.73 | 8.06 | 8.53 | 7.06 | - | - | - | - | - |
| Tumbuka (`tum`) | 1012 | 48 | 33.76 | 29.54 | 10.22 | 7.64 | 8.86 | 6.23 | - | - | - | - | - |
| Kirundi (`run`) | 1012 | 48 | 34.15 | 30.26 | 10.67 | 6.97 | 7.87 | 4.57 | - | - | - | - | - |
| Tsonga (`tso`) | 1012 | 48 | 46.01 | 42.03 | 20.87 | 15.09 | 18.31 | 13.74 | - | - | - | - | - |
| Sepedi (`nso`) | 1012 | 48 | 49.89 | 46.62 | 25.39 | 22.49 | 23.15 | 20.62 | - | - | - | - | - |
| Swati (`ssw`) | 1012 | 48 | 37.99 | 33.76 | 13.17 | 8.85 | 7.34 | 4.80 | - | - | - | - | - |
| Sango (`sag`) | 1012 | 48 | 34.86 | 33.52 | 9.35 | 6.94 | 7.86 | 5.54 | - | - | - | - | - |
| Umbundu (`umb`) | 1012 | 48 | 25.36 | 24.63 | 3.11 | 1.89 | 2.01 | 0.94 | - | - | - | - | - |
| Kimbundu (`kmb`) | 1012 | 48 | 25.83 | 24.58 | 4.23 | 3.35 | 2.74 | 2.24 | - | - | - | - | - |
| Mossi (`mos`) | 1012 | 48 | 22.87 | 22.70 | 4.28 | 3.83 | 3.00 | 2.50 | - | - | - | - | - |
| Kikuyu (`kik`) | 1012 | 48 | 32.10 | 28.86 | 9.74 | 7.64 | 8.57 | 6.19 | - | - | - | - | - |
| Luo (`luo`) | 1012 | 48 | 32.61 | 29.61 | 10.00 | 7.96 | 8.00 | 6.19 | - | - | - | - | - |
| Kamba (`kam`) | 1012 | 48 | 25.00 | 23.42 | 5.94 | 4.35 | 4.72 | 2.93 | - | - | - | - | - |
| Dyula (`dyu`) | 1012 | 48 | 18.13 | 17.00 | 2.99 | 1.80 | 1.64 | 0.92 | - | - | - | - | - |
| Bambara (`bam`) | 1012 | 48 | 24.71 | 23.38 | 5.50 | 5.19 | 4.05 | 3.35 | - | - | - | - | - |
| Kabiye (`kbp`) | 1012 | 48 | 22.23 | 20.16 | 3.86 | 2.25 | 3.29 | 2.50 | - | - | - | - | - |
| Kabuverdianu (`kea`) | 1012 | 48 | 26.34 | 24.62 | 6.46 | 5.12 | 4.68 | 4.01 | - | - | - | - | - |
| Kanuri (`knc`) | 1012 | 48 | 15.20 | 14.80 | 2.69 | 2.77 | 1.75 | 0.82 | - | - | - | - | - |
| Kabyle (`kab`) | 1012 | 48 | 26.70 | 25.74 | 7.34 | 6.89 | 4.99 | 5.36 | - | - | - | - | - |
| Nuer (`nus`) | 1012 | 48 | 20.89 | 21.60 | 6.82 | 7.60 | 2.63 | 2.37 | - | - | - | - | - |
| Luba-Kasai (`lua`) | 1012 | 48 | 33.44 | 31.20 | 8.26 | 6.03 | 6.02 | 3.31 | - | - | - | - | - |
| Makhuwa (`vmw`) | 1012 | 48 | 26.92 | 26.36 | 5.41 | 4.43 | 3.75 | 2.01 | - | - | - | - | - |

## Output format failures

`prefix_ok` = the completion starts with the expected language tag (the base model does not know the convention: its whole output is scored, the tag stripped only if present); `finished` = `<end_of_turn>` produced inside the token budget; `empty` = nothing left after stripping. All rows, unweighted mean over languages; the worst languages follow.

| benchmark | system | prefix_ok | finished | empty |
|---|---|---|---|---|
| flores_plus | SFT | 97.7% | 100.0% | 0.0% |

- flores_plus, SFT, lowest prefix_ok: knc 0.0%, afr 99.7%, luo 99.8%, bam 99.9%
- flores_plus, SFT, lowest finished: afr 100.0%, amh 100.0%, bam 100.0%, bem 100.0%
- flores_plus, SFT, highest empty: afr 0.0%, amh 0.0%, bam 0.0%, bem 0.0%

## How the contamination rule changes the macro chrF++

| benchmark | system | all | clean | clean, sentence rule | clean, no near | pair not seen |
|---|---|---|---|---|---|---|
| flores_plus | SFT | 34.52 (45L) | 31.83 (45L) | 34.31 (45L) | 31.83 (45L) | 34.31 (45L) |

## Runtime

| system | benchmark | rows | sampler | generation s | scoring s (incl. AfriCOMET) |
|---|---|---|---|---|---|
| SFT | flores_plus | 45540 | vllm | 216 | 84 |

Whole run: 367 s.
