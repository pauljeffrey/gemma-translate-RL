# SFT benchmark evaluation (English -> African)

* SFT model: `<HF_ORG>/gemma3-270m-translate` revision `376a94f2314da6445878962fe22d59900ee2d2b9` commit `376a94f2314da6445878962fe22d59900ee2d2b9` (training step 27638, macro validation loss 1.2654)
* base model: `google/gemma-3-270m-it`
* decoding: greedy, same prompt for both (`Translate the text to {language}:  {english}`), max_new_tokens 320, sampler hf; metrics on NFC-normalised text; spBLEU tokenizer flores200
* started 2026-10-19 13:36:34, finished 2026-10-19 13:49:08, 754 s; environment {"python": "3.12.14", "platform": "Linux-6.8.0-136-generic-x86_64-with-glibc2.39", "host": "043978645c01", "torch": "2.14.0+cu132", "transformers": "5.19.0", "huggingface_hub": "1.33.0", "gpu": "NVIDIA GeForce RTX 5070, 12227 MiB"}

## Results

### Macro averages, clean (headline: English sentence not in the training text)

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) | exact match (n lang) |
|---|---|---|---|---|---|---|---|---|---|
| **smol_sent** | SFT | 8 | 608 | 15.55 (8) | 1.61 (8) | 0.74 (8) | - | - | - |
| smol_sent | base gemma-3-270m-it | 8 | 608 | 11.68 (8) | 0.28 (8) | 0.16 (8) | - | - | - |
| **smol_doc** | SFT | 8 | 253 | 18.41 (8) | 3.33 (8) | 2.16 (8) | - | - | - |
| smol_doc | base gemma-3-270m-it | 8 | 253 | 16.03 (8) | 1.61 (8) | 1.63 (8) | - | - | - |
| **gatitos** | SFT | 9 | 10247 | 7.61 (9) | - | - | - | - | 0.008 (9) |
| gatitos | base gemma-3-270m-it | 9 | 10247 | 5.56 (9) | - | - | - | - | 0.001 (9) |

### Macro averages, all rows

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) | exact match (n lang) |
|---|---|---|---|---|---|---|---|---|---|
| **smol_sent** | SFT | 8 | 6904 | 20.23 (8) | 3.34 (8) | 1.66 (8) | - | - | - |
| smol_sent | base gemma-3-270m-it | 8 | 6904 | 11.53 (8) | 0.28 (8) | 0.20 (8) | - | - | - |
| **smol_doc** | SFT | 8 | 1117 | 24.79 (8) | 5.54 (8) | 3.57 (8) | - | - | - |
| smol_doc | base gemma-3-270m-it | 8 | 1117 | 16.51 (8) | 1.91 (8) | 1.67 (8) | - | - | - |
| **gatitos** | SFT | 9 | 36015 | 9.39 (9) | - | - | - | - | 0.017 (9) |
| gatitos | base gemma-3-270m-it | 9 | 36015 | 5.30 (9) | - | - | - | - | 0.001 (9) |

### Macro over every (benchmark, language) pair

| system | chrF++ clean (n) | chrF++ all rows (n) | AfriCOMET clean | AfriCOMET all |
|---|---|---|---|---|
| SFT | 13.60 (25) | 17.79 (25) | - | - |
| base gemma-3-270m-it | 10.87 (25) | 10.88 (25) | - | - |

## Contamination of the benchmarks by the SFT training text

A row is *contaminated* when the normalised English sentence (NFKC, case-folded, punctuation removed) of the benchmark row occurs in the SFT training windows (`gemma/windows_sft`, the input of the packed training set): as a whole sentence of a window or as a contiguous run inside one. *sentence rule* = the first half only (the filter that produced `bench_dev_clean.jsonl`; it reproduces that file exactly: 10,043 of 30,418 dev rows). *near* = not contaminated but >= 50% of its 8-token n-grams occur in the training English. *pair* = contaminated and the reference translation occurs in the target side of a window of the same language (the model saw the very pair).

| benchmark | rows | contaminated | of which sentence-rule hits | near | pair seen | clean rows |
|---|---|---|---|---|---|---|
| smol_sent | 6904 | 6296 (91.2%) | 6232 (90.3%) | 0 (0.0%) | 0 (0.0%) | 608 (8.8%) |
| smol_doc | 1117 | 864 (77.4%) | 0 (0.0%) | 96 (8.6%) | 0 (0.0%) | 253 (22.6%) |
| gatitos | 36015 | 25768 (71.5%) | 25768 (71.5%) | 0 (0.0%) | 150 (0.4%) | 10247 (28.5%) |

**smol_sent**: the English side is the same for every language (FLORES+/NTREX are multi-way parallel), so every language has the same 787 of 863 rows contaminated (91.2%); rows whose reference pair was seen, per language: none.

**smol_doc** per language:

| language | rows | contaminated | sentence rule | near | pair | clean |
|---|---|---|---|---|---|---|
| Acholi (`ach`) | 130 | 108 (83.1%) | 0 | 12 | 0 | 22 |
| Efik (`efi`) | 130 | 108 (83.1%) | 0 | 12 | 0 | 22 |
| Kongo (`kon`) | 130 | 108 (83.1%) | 0 | 12 | 0 | 22 |
| Krio (`kri`) | 207 | 108 (52.2%) | 0 | 12 | 0 | 99 |
| South Ndebele (`nbl`) | 130 | 108 (83.1%) | 0 | 12 | 0 | 22 |
| Ndau (`ndc`) | 130 | 108 (83.1%) | 0 | 12 | 0 | 22 |
| Nyankore (`nyn`) | 130 | 108 (83.1%) | 0 | 12 | 0 | 22 |
| Tiv (`tiv`) | 130 | 108 (83.1%) | 0 | 12 | 0 | 22 |

**gatitos** per language:

| language | rows | contaminated | sentence rule | near | pair | clean |
|---|---|---|---|---|---|---|
| Acholi (`ach`) | 4001 | 2865 (71.6%) | 2865 | 0 | 11 | 1136 |
| Efik (`efi`) | 4001 | 2865 (71.6%) | 2865 | 0 | 28 | 1136 |
| Isoko (`iso`) | 4001 | 2865 (71.6%) | 2865 | 0 | 19 | 1136 |
| Kongo (`kon`) | 3998 | 2862 (71.6%) | 2862 | 0 | 4 | 1136 |
| Krio (`kri`) | 4020 | 2860 (71.1%) | 2860 | 0 | 9 | 1160 |
| South Ndebele (`nbl`) | 4001 | 2865 (71.6%) | 2865 | 0 | 54 | 1136 |
| Ndau (`ndc`) | 3991 | 2856 (71.6%) | 2856 | 0 | 3 | 1135 |
| Nyankore (`nyn`) | 4001 | 2865 (71.6%) | 2865 | 0 | 4 | 1136 |
| Tiv (`tiv`) | 4001 | 2865 (71.6%) | 2865 | 0 | 18 | 1136 |

### smol_sent: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Efik (`efi`) | 863 | 76 | 19.60 | 15.99 | 3.05 | 1.39 | 1.28 | 0.41 | - | - | - | 10.71 | +5.27 |
| South Ndebele (`nbl`) | 863 | 76 | 35.83 | 29.60 | 12.32 | 8.09 | 4.79 | 3.36 | - | - | - | 11.64 | +17.97 |
| Tiv (`tiv`) | 863 | 76 | 21.39 | 15.99 | 3.46 | 1.37 | 2.23 | 0.87 | - | - | - | 12.10 | +3.89 |
| Kongo (`kon`) | 863 | 76 | 6.11 | 5.00 | 0.09 | 0.04 | 0.03 | 0.02 | - | - | - | 10.46 | -5.46 |
| Acholi (`ach`) | 863 | 76 | 21.39 | 15.24 | 2.29 | 0.64 | 1.60 | 0.45 | - | - | - | 12.43 | +2.80 |
| Nyankore (`nyn`) | 863 | 76 | 20.91 | 16.11 | 1.52 | 0.25 | 0.60 | 0.24 | - | - | - | 11.04 | +5.07 |
| Ndau (`ndc`) | 863 | 76 | 18.04 | 13.11 | 0.77 | 0.29 | 0.41 | 0.09 | - | - | - | 11.99 | +1.12 |
| Krio (`kri`) | 863 | 76 | 18.59 | 13.38 | 3.23 | 0.82 | 2.32 | 0.50 | - | - | - | 13.04 | +0.34 |

### smol_doc: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Efik (`efi`) | 130 | 22 | 26.21 | 17.13 | 7.74 | 3.66 | 4.40 | 1.90 | - | - | - | 13.74 | +3.40 |
| South Ndebele (`nbl`) | 130 | 22 | 40.56 | 33.20 | 13.29 | 7.98 | 6.09 | 3.39 | - | - | - | 16.42 | +16.78 |
| Tiv (`tiv`) | 130 | 22 | 24.10 | 15.58 | 5.10 | 2.98 | 3.82 | 2.47 | - | - | - | 16.38 | -0.80 |
| Kongo (`kon`) | 130 | 22 | 7.76 | 7.72 | 0.25 | 0.28 | 0.19 | 0.18 | - | - | - | 15.31 | -7.59 |
| Acholi (`ach`) | 130 | 22 | 27.76 | 19.03 | 5.08 | 2.59 | 4.00 | 2.17 | - | - | - | 18.16 | +0.86 |
| Nyankore (`nyn`) | 130 | 22 | 28.18 | 20.40 | 3.61 | 2.25 | 2.73 | 1.65 | - | - | - | 14.55 | +5.85 |
| Ndau (`ndc`) | 130 | 22 | 22.53 | 17.99 | 2.04 | 1.21 | 1.26 | 0.84 | - | - | - | 16.28 | +1.71 |
| Krio (`kri`) | 207 | 99 | 21.18 | 16.19 | 7.20 | 5.68 | 6.05 | 4.66 | - | - | - | 17.37 | -1.18 |

### gatitos: SFT model per language (word level)

n = entries (English word or phrase); exact = the output equals ANY accepted translation (NFC, case-folded, surrounding punctuation removed); chrF++ = against the best accepted translation. clean = the entry is not a whole sentence of the training windows.

| language | n | n clean | chrF++ all | chrF++ clean | exact all | exact clean | base exact all | base exact clean | finished | prefix ok |
|---|---|---|---|---|---|---|---|---|---|---|
| Efik (`efi`) | 4001 | 1136 | 7.18 | 6.19 | 0.1% | 0.1% | 0.0% | 0.0% | 95.4% | 100.0% |
| South Ndebele (`nbl`) | 4001 | 1136 | 11.50 | 7.44 | 3.5% | 0.9% | 0.0% | 0.0% | 99.7% | 100.0% |
| Tiv (`tiv`) | 4001 | 1136 | 12.06 | 9.89 | 1.4% | 1.1% | 0.1% | 0.4% | 98.0% | 100.0% |
| Kongo (`kon`) | 3998 | 1136 | 3.79 | 3.34 | 0.9% | 0.9% | 0.1% | 0.0% | 92.0% | 100.0% |
| Isoko (`iso`) | 4001 | 1136 | 10.51 | 7.52 | 1.8% | 0.2% | 0.0% | 0.0% | 97.2% | 100.0% |
| Acholi (`ach`) | 4001 | 1136 | 8.99 | 6.94 | 1.7% | 1.5% | 0.0% | 0.2% | 99.3% | 100.0% |
| Nyankore (`nyn`) | 4001 | 1136 | 10.65 | 8.89 | 1.9% | 1.2% | 0.0% | 0.1% | 98.4% | 100.0% |
| Ndau (`ndc`) | 3991 | 1135 | 5.32 | 3.80 | 0.7% | 0.1% | 0.0% | 0.0% | 97.2% | 100.0% |
| Krio (`kri`) | 4020 | 1160 | 14.54 | 14.45 | 2.8% | 0.9% | 0.2% | 0.3% | 93.9% | 100.0% |

## Output format failures

`prefix_ok` = the completion starts with the expected language tag (the base model does not know the convention: its whole output is scored, the tag stripped only if present); `finished` = `<end_of_turn>` produced inside the token budget; `empty` = nothing left after stripping. All rows, unweighted mean over languages; the worst languages follow.

| benchmark | system | prefix_ok | finished | empty |
|---|---|---|---|---|
| smol_sent | SFT | 99.7% | 84.2% | 0.0% |
| smol_sent | base gemma-3-270m-it | 0.0% | 95.3% | 0.0% |
| smol_doc | SFT | 99.9% | 52.2% | 0.0% |
| smol_doc | base gemma-3-270m-it | 0.0% | 99.0% | 0.0% |
| gatitos | SFT | 100.0% | 96.8% | 31.2% |
| gatitos | base gemma-3-270m-it | 0.0% | 97.3% | 0.0% |

- smol_sent, SFT, lowest prefix_ok: ndc 97.3%, tiv 99.9%, ach 100.0%, efi 100.0%
- smol_sent, SFT, lowest finished: kon 35.3%, kri 83.5%, tiv 88.3%, ndc 90.7%
- smol_sent, SFT, highest empty: nbl 0.1%, ach 0.0%, efi 0.0%, kon 0.0%
- smol_doc, SFT, lowest prefix_ok: efi 99.2%, ach 100.0%, kon 100.0%, kri 100.0%
- smol_doc, SFT, lowest finished: kon 0.8%, kri 44.4%, tiv 52.3%, efi 55.4%
- smol_doc, SFT, highest empty: ach 0.0%, efi 0.0%, kon 0.0%, kri 0.0%
- gatitos, SFT, lowest prefix_ok: efi 100.0%, kri 100.0%, ndc 100.0%, kon 100.0%
- gatitos, SFT, lowest finished: kon 92.0%, kri 93.9%, efi 95.4%, ndc 97.2%
- gatitos, SFT, highest empty: nbl 63.5%, ndc 61.6%, kon 54.9%, ach 43.4%

## How the contamination rule changes the macro chrF++

| benchmark | system | all | clean | clean, sentence rule | clean, no near | pair not seen |
|---|---|---|---|---|---|---|
| smol_sent | SFT | 20.23 (8L) | 15.55 (8L) | 15.74 (8L) | 15.55 (8L) | 20.23 (8L) |
| smol_sent | base gemma-3-270m-it | 11.53 (8L) | 11.68 (8L) | 11.93 (8L) | 11.68 (8L) | 11.53 (8L) |
| smol_doc | SFT | 24.79 (8L) | 18.41 (8L) | 24.79 (8L) | 18.41 (8L) | 24.79 (8L) |
| smol_doc | base gemma-3-270m-it | 16.51 (8L) | 16.03 (8L) | 16.51 (8L) | 16.03 (8L) | 16.51 (8L) |
| gatitos | SFT | 9.39 (9L) | 7.61 (9L) | 7.61 (9L) | 7.61 (9L) | 9.37 (9L) |
| gatitos | base gemma-3-270m-it | 5.30 (9L) | 5.56 (9L) | 5.56 (9L) | 5.56 (9L) | 5.30 (9L) |

## Runtime

| system | benchmark | rows | sampler | generation s | scoring s (incl. AfriCOMET) |
|---|---|---|---|---|---|
| SFT | smol_sent | 6904 | hf | 283 | 8 |
| SFT | smol_doc | 1117 | hf | 386 | 13 |
| SFT | gatitos | 36015 | hf | 98 | 13 |
| base gemma-3-270m-it | smol_sent | 6904 | hf | 276 | 8 |
| base gemma-3-270m-it | smol_doc | 1117 | hf | 299 | 12 |
| base gemma-3-270m-it | gatitos | 36015 | hf | 97 | 19 |

Whole run: 754 s.
