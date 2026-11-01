# SFT benchmark evaluation (English -> African)

* SFT model: `<HF_ORG>/gemma3-270m-translate` revision `main` commit `376a94f2314da6445878962fe22d59900ee2d2b9` (training step 27638, macro validation loss 1.2654)
* base model: `google/gemma-3-270m-it`
* decoding: greedy, same prompt for both (`Translate the text to {language}:  {english}`), max_new_tokens 320, sampler hf; metrics on NFC-normalised text; spBLEU tokenizer flores200
* started 2026-10-19 15:49:55, finished 2026-10-19 16:10:15, 1220 s; environment {"python": "3.12.14", "platform": "Linux-6.8.0-136-generic-x86_64-with-glibc2.39", "host": "043978645c01", "torch": "2.14.0+cu132", "transformers": "5.19.0", "huggingface_hub": "1.33.0", "gpu": "NVIDIA GeForce RTX 5070, 12227 MiB"}

## Results

### Macro averages, clean (headline: English sentence not in the training text)

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|
| **ibom_mt** | SFT | 2 | 96 | 16.64 (2) | 2.16 (2) | 0.88 (2) | 0.406 (1) | 0.327 (1) |
| ibom_mt | base gemma-3-270m-it | 2 | 96 | 13.73 (2) | 1.87 (2) | 0.55 (2) | 0.217 (1) | 0.405 (1) |
| **flores_moz** | SFT | 1 | 48 | 25.02 (1) | 2.85 (1) | 2.06 (1) | - | - |
| flores_moz | base gemma-3-270m-it | 1 | 48 | 12.82 (1) | 2.02 (1) | 1.33 (1) | - | - |
| **glot_story** | SFT | 8 | 3179 | 22.87 (8) | 4.21 (8) | 2.49 (8) | - | - |
| glot_story | base gemma-3-270m-it | 8 | 3179 | 12.24 (8) | 1.65 (8) | 0.99 (8) | - | - |
| **salt** | SFT | 5 | 390 | 33.83 (5) | 13.40 (5) | 11.02 (5) | 0.661 (3) | 0.641 (3) |
| salt | base gemma-3-270m-it | 5 | 390 | 8.36 (5) | 0.23 (5) | 0.35 (5) | 0.240 (3) | 0.416 (3) |

### Macro averages, all rows

| benchmark | system | languages | rows | chrF++ (n lang) | spBLEU (n lang) | BLEU (n lang) | AfriCOMET (n lang) | AfriCOMET-QE (n lang) |
|---|---|---|---|---|---|---|---|---|
| **ibom_mt** | SFT | 2 | 2024 | 18.49 (2) | 3.11 (2) | 1.84 (2) | 0.476 (1) | 0.454 (1) |
| ibom_mt | base gemma-3-270m-it | 2 | 2024 | 13.89 (2) | 1.82 (2) | 1.09 (2) | 0.242 (1) | 0.459 (1) |
| **flores_moz** | SFT | 1 | 1012 | 25.59 (1) | 4.02 (1) | 2.86 (1) | - | - |
| flores_moz | base gemma-3-270m-it | 1 | 1012 | 13.16 (1) | 2.00 (1) | 1.16 (1) | - | - |
| **glot_story** | SFT | 8 | 3426 | 22.85 (8) | 4.20 (8) | 2.47 (8) | - | - |
| glot_story | base gemma-3-270m-it | 8 | 3426 | 12.12 (8) | 1.60 (8) | 0.97 (8) | - | - |
| **salt** | SFT | 5 | 2500 | 37.91 (5) | 16.54 (5) | 11.94 (5) | 0.726 (3) | 0.709 (3) |
| salt | base gemma-3-270m-it | 5 | 2500 | 8.67 (5) | 0.15 (5) | 0.14 (5) | 0.256 (3) | 0.444 (3) |

### Macro over every (benchmark, language) pair

| system | chrF++ clean (n) | chrF++ all rows (n) | AfriCOMET clean | AfriCOMET all |
|---|---|---|---|---|
| SFT | 25.65 (16) | 27.18 (16) | 0.597 | 0.664 |
| base gemma-3-270m-it | 11.25 (16) | 11.33 (16) | 0.234 | 0.252 |

## Contamination of the benchmarks by the SFT training text

A row is *contaminated* when the normalised English sentence (NFKC, case-folded, punctuation removed) of the benchmark row occurs in the SFT training windows (`gemma/windows_sft`, the input of the packed training set): as a whole sentence of a window or as a contiguous run inside one. *sentence rule* = the first half only (the filter that produced `bench_dev_clean.jsonl`; it reproduces that file exactly: 10,043 of 30,418 dev rows). *near* = not contaminated but >= 50% of its 8-token n-grams occur in the training English. *pair* = contaminated and the reference translation occurs in the target side of a window of the same language (the model saw the very pair).

| benchmark | rows | contaminated | of which sentence-rule hits | near | pair seen | clean rows |
|---|---|---|---|---|---|---|
| ibom_mt | 2024 | 1928 (95.3%) | 1700 (84.0%) | 0 (0.0%) | 0 (0.0%) | 96 (4.7%) |
| flores_moz | 1012 | 964 (95.3%) | 850 (84.0%) | 0 (0.0%) | 0 (0.0%) | 48 (4.7%) |
| glot_story | 3426 | 247 (7.2%) | 228 (6.7%) | 53 (1.5%) | 0 (0.0%) | 3179 (92.8%) |
| salt | 2500 | 2110 (84.4%) | 2110 (84.4%) | 0 (0.0%) | 419 (16.8%) | 390 (15.6%) |

**ibom_mt**: the English side is the same for every language (multi-way parallel set: FLORES+, NTREX, SmolSent, IbomMT, FLORES-MOZ), so every language has the same 964 of 1012 rows contaminated (95.3%); rows whose reference pair was seen, per language: none.

**flores_moz**: the English side is the same for every language (multi-way parallel set: FLORES+, NTREX, SmolSent, IbomMT, FLORES-MOZ), so every language has the same 964 of 1012 rows contaminated (95.3%); rows whose reference pair was seen, per language: none.

**glot_story** per language:

| language | rows | contaminated | sentence rule | near | pair | clean |
|---|---|---|---|---|---|---|
| Dagbani (`dag`) | 135 | 29 (21.5%) | 29 | 3 | 0 | 106 |
| Herero (`her`) | 549 | 39 (7.1%) | 36 | 9 | 0 | 510 |
| Kaonde (`kqn`) | 420 | 33 (7.9%) | 30 | 8 | 0 | 387 |
| Kuanyama (`kua`) | 520 | 33 (6.3%) | 30 | 8 | 0 | 487 |
| Lozi (`loz`) | 960 | 63 (6.6%) | 57 | 14 | 0 | 897 |
| Lunda (`lun`) | 420 | 33 (7.9%) | 30 | 8 | 0 | 387 |
| South Ndebele (`nbl`) | 321 | 16 (5.0%) | 15 | 3 | 0 | 305 |
| Tswa (`tsc`) | 101 | 1 (1.0%) | 1 | 0 | 0 | 100 |

**salt**: the English side is the same for every language (multi-way parallel set: FLORES+, NTREX, SmolSent, IbomMT, FLORES-MOZ), so every language has the same 422 of 500 rows contaminated (84.4%); rows whose reference pair was seen, per language: lug 419.

### ibom_mt: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Efik (`efi`) | 1012 | 48 | 21.33 | 18.56 | 4.39 | 2.83 | 2.61 | 1.15 | 0.476 | 0.406 | 0.327 | 14.84 | +3.72 |
| Ibibio (`ibb`) | 1012 | 48 | 15.66 | 14.71 | 1.82 | 1.49 | 1.07 | 0.61 | - | - | - | 12.62 | +2.09 |

### flores_moz: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sena (`seh`) | 1012 | 48 | 25.59 | 25.02 | 4.02 | 2.85 | 2.86 | 2.06 | - | - | - | 12.82 | +12.20 |

### glot_story: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| South Ndebele (`nbl`) | 321 | 305 | 16.10 | 16.20 | 3.19 | 3.18 | 1.67 | 1.69 | - | - | - | 11.76 | +4.44 |
| Dagbani (`dag`) | 135 | 106 | 18.97 | 19.33 | 3.13 | 3.29 | 2.59 | 2.71 | - | - | - | 10.52 | +8.81 |
| Kaonde (`kqn`) | 420 | 387 | 27.60 | 27.54 | 5.95 | 5.99 | 2.99 | 3.04 | - | - | - | 12.44 | +15.09 |
| Lunda (`lun`) | 420 | 387 | 24.27 | 24.24 | 4.68 | 4.69 | 2.27 | 2.31 | - | - | - | 12.44 | +11.80 |
| Lozi (`loz`) | 960 | 897 | 32.32 | 32.18 | 6.35 | 6.27 | 3.88 | 3.82 | - | - | - | 12.87 | +19.31 |
| Tswa (`tsc`) | 101 | 100 | 14.42 | 14.43 | 0.80 | 0.81 | 0.65 | 0.65 | - | - | - | 9.20 | +5.23 |
| Kuanyama (`kua`) | 520 | 487 | 28.28 | 28.26 | 5.96 | 5.90 | 3.35 | 3.35 | - | - | - | 14.72 | +13.54 |
| Herero (`her`) | 549 | 510 | 20.82 | 20.81 | 3.54 | 3.53 | 2.36 | 2.36 | - | - | - | 13.94 | +6.87 |

### salt: SFT model per language

n = rows; clean = rows whose English sentence is not in the training text. Per-language clean values on few rows are noisy (the clean macro leaves out languages below the minimum number of clean rows).

| language | n | n clean | chrF++ all | chrF++ clean | spBLEU all | spBLEU clean | BLEU all | BLEU clean | AfriCOMET all | AfriCOMET clean | QE clean | base chrF++ clean | Δ chrF++ clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Igbo (`ibo`) | 500 | 78 | 35.20 | 35.47 | 13.65 | 15.81 | 9.79 | 13.00 | 0.721 | 0.681 | 0.634 | 5.65 | +29.83 |
| Swahili (`swh`) | 500 | 78 | 57.47 | 52.05 | 34.55 | 31.35 | 29.47 | 27.72 | 0.758 | 0.680 | 0.695 | 8.21 | +43.84 |
| Luganda (`lug`) | 500 | 78 | 45.64 | 36.94 | 24.36 | 13.28 | 15.10 | 9.13 | 0.700 | 0.621 | 0.595 | 7.38 | +29.56 |
| Acholi (`ach`) | 500 | 78 | 24.50 | 22.33 | 4.82 | 3.68 | 2.94 | 2.93 | - | - | - | 10.74 | +11.59 |
| Nyankore (`nyn`) | 500 | 78 | 26.72 | 22.35 | 5.32 | 2.87 | 2.39 | 2.33 | - | - | - | 9.81 | +12.54 |

## Output format failures

`prefix_ok` = the completion starts with the expected language tag (the base model does not know the convention: its whole output is scored, the tag stripped only if present); `finished` = `<end_of_turn>` produced inside the token budget; `empty` = nothing left after stripping. All rows, unweighted mean over languages; the worst languages follow.

| benchmark | system | prefix_ok | finished | empty |
|---|---|---|---|---|
| ibom_mt | SFT | 99.9% | 86.2% | 0.0% |
| ibom_mt | base gemma-3-270m-it | 0.0% | 98.4% | 0.0% |
| flores_moz | SFT | 99.9% | 93.6% | 0.0% |
| flores_moz | base gemma-3-270m-it | 0.0% | 100.0% | 0.0% |
| glot_story | SFT | 100.0% | 90.4% | 0.1% |
| glot_story | base gemma-3-270m-it | 0.0% | 99.3% | 0.0% |
| salt | SFT | 100.0% | 98.8% | 0.0% |
| salt | base gemma-3-270m-it | 0.0% | 83.2% | 0.0% |

- ibom_mt, SFT, lowest prefix_ok: ibb 99.8%, efi 99.9%
- ibom_mt, SFT, lowest finished: ibb 84.4%, efi 88.0%
- ibom_mt, SFT, highest empty: efi 0.0%, ibb 0.0%
- flores_moz, SFT, lowest prefix_ok: seh 99.9%
- flores_moz, SFT, lowest finished: seh 93.6%
- flores_moz, SFT, highest empty: seh 0.0%
- glot_story, SFT, lowest prefix_ok: dag 100.0%, her 100.0%, kqn 100.0%, kua 100.0%
- glot_story, SFT, lowest finished: tsc 77.2%, her 81.8%, dag 87.4%, kqn 88.6%
- glot_story, SFT, highest empty: dag 1.5%, her 0.0%, kqn 0.0%, kua 0.0%
- salt, SFT, lowest prefix_ok: ach 100.0%, ibo 100.0%, lug 100.0%, nyn 100.0%
- salt, SFT, lowest finished: ach 97.4%, nyn 98.2%, ibo 98.8%, lug 99.8%
- salt, SFT, highest empty: ach 0.0%, ibo 0.0%, lug 0.0%, nyn 0.0%

## How the contamination rule changes the macro chrF++

| benchmark | system | all | clean | clean, sentence rule | clean, no near | pair not seen |
|---|---|---|---|---|---|---|
| ibom_mt | SFT | 18.49 (2L) | 16.64 (2L) | 18.23 (2L) | 16.64 (2L) | 18.49 (2L) |
| ibom_mt | base gemma-3-270m-it | 13.89 (2L) | 13.73 (2L) | 14.24 (2L) | 13.73 (2L) | 13.89 (2L) |
| flores_moz | SFT | 25.59 (1L) | 25.02 (1L) | 27.08 (1L) | 25.02 (1L) | 25.59 (1L) |
| flores_moz | base gemma-3-270m-it | 13.16 (1L) | 12.82 (1L) | 13.05 (1L) | 12.82 (1L) | 13.16 (1L) |
| glot_story | SFT | 22.85 (8L) | 22.87 (8L) | 22.89 (8L) | 22.81 (8L) | 22.85 (8L) |
| glot_story | base gemma-3-270m-it | 12.12 (8L) | 12.24 (8L) | 12.23 (8L) | 12.21 (8L) | 12.12 (8L) |
| salt | SFT | 37.91 (5L) | 33.83 (5L) | 33.83 (5L) | 33.83 (5L) | 36.28 (5L) |
| salt | base gemma-3-270m-it | 8.67 (5L) | 8.36 (5L) | 8.36 (5L) | 8.36 (5L) | 8.62 (5L) |

## Runtime

| system | benchmark | rows | sampler | generation s | scoring s (incl. AfriCOMET) |
|---|---|---|---|---|---|
| SFT | ibom_mt | 2024 | hf | 135 | 46 |
| SFT | flores_moz | 1012 | hf | 67 | 1 |
| SFT | glot_story | 3426 | hf | 227 | 9 |
| SFT | salt | 2500 | hf | 118 | 43 |
| base gemma-3-270m-it | ibom_mt | 2024 | hf | 127 | 44 |
| base gemma-3-270m-it | flores_moz | 1012 | hf | 33 | 1 |
| base gemma-3-270m-it | glot_story | 3426 | hf | 137 | 8 |
| base gemma-3-270m-it | salt | 2500 | hf | 159 | 53 |

Whole run: 1220 s.
