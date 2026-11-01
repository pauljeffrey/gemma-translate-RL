# NLLB-200 as teacher and as reward signal for the English -> African Gemma-3-270m model (R14, 2026-10-22)

Task D-66. Owner decisions relayed 2026-10-22: use NLLB-200 (first the distilled-600M, then **the dense 3.3B as the main model**, CC-BY-NC accepted) (1) to answer whether it works on our languages, (2) as a **teacher for SFT data** (reinstated after a brief "reward only" ruling) and (3) as a **reward signal in RL** (reward use only: pseudo-references and a forced-decoding score). Nothing was pushed anywhere; no dataset was copied to the laptop (reports, CSV and code only). Box work: `/workspace/jobs/r14` (GPU capped at ~20 GB, `nice -n 5`; other agents' jobs and `/workspace/datagen` untouched). Short stage-1 report: `reports/nllb600m_stage1_2026-10-22.md`.

## 0. Answers (read this first)

1. **Does NLLB work on our languages?** Where it covers a language, yes; the **3.3B model clearly, the 600M model only partly.** Over the 44 FLORES+ languages that both cover, English -> X devtest chrF++ (rows whose reference pair our SFT never saw): ours (final_1433, greedy) **34.3**, NLLB-600M beam 4 **35.9** (+1.7 [+1.6, +1.8]; greedy 600M +0.4), **NLLB-3.3B beam 4 38.1 (+3.8 [+3.8, +3.9]; spBLEU +4.4)**. The advantage holds on every other human benchmark: NTREX +4.2 (21 languages), MAFAND +3.5 (14), SmolSent +5.5 (36), SALT +3.5 (3), and its sign agrees with FLORES+ in 64 of 73 language x benchmark pairs. The 3.3B model is better than ours by >= +5 chrF++ in 18 languages, by +2..+5 in 8, within +-2 in 17 and worse by more than 2 in two (Dyula -2.8, Umbundu -2.5; Yoruba, Fon, Wolof, Kamba, Kimbundu are -1 to -2). The gain is largest where we are weakest: **Ge'ez (Amharic +18.9, Tigrinya +11.0), Kabuverdianu +21.1, Kanuri +9.7 (we have no Kanuri data), Nuer +7.9, Kirundi +7.7, Ewe +7.2, Kabiye +7.0, Malagasy +7.0, Kinyarwanda +6.2, Bambara +6.0**, Nilo-Saharan +7.7 and Afroasiatic +7.2 on average; in Atlantic-Congo (non-Bantu) it is only +1.4 (the 600M model is *worse* than us there, -1.5).
2. **Where it is useless: coverage.** NLLB-200 covers **44 of our 77 RL languages and 2 of 8 SFT-only ones** (46 of 85 table rows exactly; Congo Swahili, Fante and Fulah only through a related variety). It does **not** cover **22 of the 23 consensus-only languages** (only Kabuverdianu is covered), **8 of the 12 listed/blend languages** (Dyula, Sango, Kongo are covered), nor Pidgin, Efik, Ibibio, Tiv, Venda, Edo, Dagbani, Okpe, Tachelhit, Ghomala, Kaonde, Lunda, Isoko, Lozi, Wolaytta, Acholi, Tswa, Makhuwa, Kuanyama, Sena, Nyankore, Herero, Yao, Sidamo, Ndau, Krio, Esan, Duala, Basaa, Cameroon Pidgin, Ndebele, Ga, Mende, Luyia. Of the 8 weakest (`ish bin bbj tsc kon nus dag oke`) only `kon` and `nus` are covered. So NLLB **cannot replace curation** for the languages that have the weakest reward signal; it is a strong teacher/pseudo-reference for 26 covered languages, 2 of which are in the weakest group.
3. **Teacher SFT data (ready):** `/workspace/jobs/r14/teacher_v1` + `READY_TEACHER` + `manifest.json`: **5.52 M tokens, 39,749 samples, 27 languages**, R4 packed format (8192-token packs, tokenizer fingerprint of final_1433, `audit_packed` 0 problems), every row `licence_class = nc_teacher`, `teacher = nllb-200-3.3B`, with the English source's `source_dataset` and licence in `rows.parquet`. Quality filters (GlotLID, human length-ratio window, copy and loop detectors, AfriCOMET-QE where validated) removed 0.3-2 % of the sentences of the languages without QE filter and 13-29 % of the others (Tigrinya 29 %, Oromo 29 %, Tumbuka 27 %, Amharic 21 %; mostly QE and LID). Hold-out filter (exact + 8-gram against FLORES+ dev/devtest, NTREX, MAFAND, SmolSent, SALT, AfriScience dev+test, AfriLongDoc, bench_dev, rl-vs-sft pool): 0 hits left in an independent re-check of all 39,749 English sides.
4. **Reward signal:** the NLLB **pseudo-reference chrF++ (A)** is a good, cheap signal where the teacher is better: AUC(human vs mismatched) 0.97 (QE 0.87), hard-class failure 0.08 (QE 0.10), exact-copy failure 0.03, and it **tracks human chrF++ on policy samples better than AfriCOMET-QE (Spearman 0.57 vs 0.28; within a GRPO group 0.40 vs 0.15) and adds information beyond the group consensus (partial Spearman 0.31; AfriCOMET-QE adds 0.07 beyond consensus)**. Its weaknesses: padding (appending an unrelated sentence to a human translation still scores >= the human one in 27 % of sentences, a property of chrF++'s recall weight) and style pull. The **forced-decoding score (B)** is excellent against padding, omission and shuffles (hard-class failure 0.03) but **prefers copies of the English source (92 % of copies score above the human translation)**, so it must only be used inside `min(A, B)`; it adds little beyond consensus (partial 0.10) and costs one NLLB forward pass per reward call (0.57 s per 64 chunk pairs for 3.3B int8, 0.33 s for 600M). **Recommendation:** enable A (flag-gated) for the covered weak-signal languages `kea kbp nus bam kab` (and `kon` after the owner decides the variety question) with weight 0.23-0.35; for the 19 covered QE-mode languages with a clearly better teacher use it only as a check/low weight; never for `dyu sag mos kam kmb umb fon yor wol` and the other languages where the teacher is not clearly better; B off by default.
5. **Precomputed NLLB-3.3B translations for reward use** (`reward_only_nc`): `/workspace/jobs/r14/pseudorefs_reward_only_nc_v1/<iso>.parquet` for kea kon nus kbp bam kab (30,000 sentences each = ~2.6 k RL prompt windows, 0.4-1.2 % of the sentences rejected as empty/copy/loop) and tir amh orm luo kik (8,000 sentences each = ~0.7 k windows), 82 MB in total, manifest `pseudorefs_reward_only_nc_v1/manifest.json`, keyed `sha1(iso + "\n" + NFC(source))[:16]`, sentences split with `rl/segments.split_sentences`. The full pools hold ~247 k sentences per language, i.e. ~2.5 h of GPU each at the measured 25-45 sentences/s: not precomputed in full (a run consumes ~1.3 k prompts per language, one parquet of 15 k sentences covers ~1.3 k windows).

## 1. Stage 1: coverage and benchmark (details in `reports/nllb600m_stage1_2026-10-22.md`)

Models and pipeline: NLLB-200-distilled-600M converted by us to CTranslate2 float16; **NLLB-200-3.3B = the CTranslate2 authors' Hub build `OpenNMT/nllb-200-3.3B-ct2-int8`** (3.4 GB; no fp32 download, disk was the constraint), run with `int8_float16`. Beam 4, sentence level, `max_decoding_length` 256. **Pipeline check against the published NLLB-200-3.3B FLORES-200 devtest chrF++** (`published_baselines.csv`, 43 languages, all 1,012 rows, NFC scoring): mean difference -0.17, **mean absolute 0.40 chrF++**, maximum +3.0 (Yoruba, because the repo scores NFC text and 28 % of the Yoruba references are decomposed). The repeat guard `no_repeat_ngram_size 4` changes devtest chrF++ by +0.13 on average (10 languages; kbp +0.83; amh -0.15) and was used for all generated data. Scoring: the repo's `gemma_mt.eval.metrics` (chrF++ = sacrebleu word_order 2; spBLEU = flores200 tokenizer; NFC); ours = a fresh vLLM run of `final_1433` on the same rows (per-sentence outputs are needed for the paired bootstrap; 34.25 here vs 34.1 in the per-language CSV for the 45-language set). Bootstrap: 1,000 resamples of the sentences, segment statistics summed.

Macro results per benchmark (English -> X, `no_pair` = reference pair not in our SFT text; `clean` = English sentence not in our SFT text):

| benchmark (English -> X) | langs | ours chrF++ | NLLB-3.3B | delta [95% CI] | ours spBLEU | NLLB-3.3B | delta [95% CI] |
|---|---|---|---|---|---|---|---|
| FLORES+ devtest `no_pair` | 44 | 34.25 | 38.09 | +3.84 [+3.75, +3.93] | 12.58 | 17.01 | +4.43 [+4.31, +4.53] |
| FLORES+ devtest `clean` | 44 | 31.86 | 34.99 | +3.13 [+2.77, +3.49] | 10.49 | 14.34 | +3.85 [+3.42, +4.26] |
| NTREX-128 `no_pair` | 21 | 38.98 | 43.2 | +4.22 [+4.14, +4.31] | 17.28 | 21.83 | +4.55 [+4.44, +4.66] |
| NTREX-128 `clean` | 21 | 37.37 | 41.47 | +4.10 [+4.00, +4.20] | 15.64 | 20.03 | +4.39 [+4.26, +4.52] |
| MAFAND-MT test `no_pair` | 14 | 40.72 | 44.18 | +3.47 [+3.28, +3.67] | 18.89 | 23.56 | +4.66 [+4.44, +4.91] |
| MAFAND-MT test `clean` | 14 | 40.5 | 43.99 | +3.49 [+3.25, +3.72] | 18.67 | 23.34 | +4.67 [+4.40, +4.94] |
| SmolSent `no_pair` | 36 | 31.68 | 37.22 | +5.54 [+5.41, +5.65] | 10.53 | 16.04 | +5.51 [+5.37, +5.67] |
| SmolSent `clean` | 36 | 27.01 | 32.45 | +5.45 [+5.13, +5.78] | 6.92 | 12.31 | +5.39 [+5.04, +5.82] |
| SALT `no_pair` | 3 | 45.2 | 48.68 | +3.48 [+2.23, +4.70] | 22.69 | 26.34 | +3.65 [+2.17, +5.11] |
| SALT `clean` | 3 | 43.06 | 47.2 | +4.14 [+2.19, +6.20] | 22.25 | 26.24 | +3.99 [+1.48, +6.55] |

By training group and family on FLORES+ devtest `no_pair` (delta = NLLB minus ours):

| set | langs | ours chrF++ | NLLB-3.3B | delta [95% CI] | NLLB-600M | delta [95% CI] | NLLB-600M greedy | delta |
|---|---|---|---|---|---|---|---|---|
| all covered FLORES+ languages | 44 | 34.25 | 38.09 | +3.84 [+3.75, +3.93] | 35.94 | +1.68 [+1.60, +1.77] | 34.67 | +0.42 |
| QE-validated | 40 | 35.32 | 38.8 | +3.49 [+3.39, +3.58] | 36.8 | +1.48 [+1.39, +1.57] | 35.56 | +0.25 |
| QE-blend | 2 | 26.51 | 25.9 | -0.61 [-0.91, -0.33] | 25.27 | -1.24 [-1.54, -0.94] | 24.52 | -1.99 |
| consensus-only | 1 | 26.74 | 47.82 | +21.08 [+20.26, +21.94] | 44.07 | +17.33 [+16.47, +18.17] | 40.86 | +14.12 |
| SFT-only (no RL) | 1 | 14.7 | 24.37 | +9.67 [+9.21, +10.13] | 14.66 | -0.04 [-0.64, +0.56] | 13.11 | -1.59 |
| Bantu | 21 | 39.08 | 41.86 | +2.78 [+2.66, +2.91] | 40.45 | +1.37 [+1.24, +1.49] | 39.1 | +0.03 |
| Atlantic-Congo (non-Bantu) | 9 | 29.41 | 30.79 | +1.37 [+1.18, +1.55] | 27.95 | -1.46 [-1.66, -1.26] | 26.75 | -2.66 |
| Afroasiatic (Ge'ez, Cushitic, Berber) | 6 | 29.47 | 36.66 | +7.19 [+6.84, +7.53] | 34.15 | +4.68 [+4.35, +4.95] | 32.96 | +3.48 |
| Nilo-Saharan | 3 | 22.81 | 30.5 | +7.69 [+7.43, +7.97] | 25.57 | +2.76 [+2.44, +3.06] | 24.37 | +1.56 |
| Mande | 2 | 21.62 | 23.21 | +1.59 [+1.26, +1.90] | 22.5 | +0.87 [+0.54, +1.19] | 21.98 | +0.36 |
| Afrikaans, Malagasy | 2 | 53.3 | 57.15 | +3.85 [+3.43, +4.32] | 54.82 | +1.52 [+1.10, +1.96] | 53.99 | +0.69 |
| Creole (Kabuverdianu) | 1 | 26.74 | 47.82 | +21.08 [+20.26, +21.94] | 44.07 | +17.33 [+16.47, +18.17] | 40.86 | +14.12 |

Greedy vs beam: the 600M model loses 1.3 chrF++ when decoded greedily; 3.3B greedy was not run (GPU contention; see 7).

### Decision table (languages NLLB covers; sorted by the 3.3B delta)

Categories: much better >= +5, better +2..+5, similar +-2, worse < -2. "Useful as reward signal?" = yes when the 3.3B delta is >= +2 with the interval above 0 (teacher clearly better than student), no when <= -2 or the interval is entirely below 0, else weak/uncertain. Uncovered languages (36 rows) and the 5 languages without a benchmark row are in `reports/data/nllb600m_decision_table.csv`.

| iso | language | group | bench (n) | ours | NLLB-3.3B | d3.3B [CI] | NLLB-600M | d600M [CI] | category (3.3B) | useful as reward signal? (3.3B) |
|---|---|---|---|---|---|---|---|---|---|---|
| `kon` | Kongo | blend | smol_sent (863) | 16.7 | 47.81 | +31.1 [+30.3, +31.9] | - | - | NLLB much better (>=+5) | yes |
| `kea` | Kabuverdianu | consensus_only | flores_plus (1012) | 26.74 | 47.82 | +21.1 [+20.3, +21.9] | 44.07 | +17.3 [+16.5, +18.2] | NLLB much better (>=+5) | yes |
| `amh` | Amharic | qe_validated | flores_plus (1012) | 19.06 | 37.96 | +18.9 [+17.9, +19.8] | 37.26 | +18.2 [+17.4, +19.0] | NLLB much better (>=+5) | yes |
| `tir` | Tigrinya | qe_validated | flores_plus (1012) | 13.73 | 24.73 | +11.0 [+10.4, +11.5] | 22.37 | +8.6 [+8.1, +9.1] | NLLB much better (>=+5) | yes |
| `knc` | Kanuri | sft_only | flores_plus (1012) | 14.7 | 24.37 | +9.7 [+9.2, +10.1] | 14.66 | -0.0 [-0.6, +0.6] | NLLB much better (>=+5) | yes |
| `nus` | Nuer | qe_validated | flores_plus (1012) | 20.78 | 28.7 | +7.9 [+7.5, +8.4] | 25.27 | +4.5 [+4.0, +5.0] | NLLB much better (>=+5) | yes |
| `run` | Kirundi | qe_validated | flores_plus (1012) | 33.88 | 41.56 | +7.7 [+7.0, +8.3] | 38.94 | +5.1 [+4.5, +5.6] | NLLB much better (>=+5) | yes |
| `ewe` | Ewe | qe_validated | flores_plus (1012) | 31.12 | 38.32 | +7.2 [+6.7, +7.7] | 35.23 | +4.1 [+3.5, +4.8] | NLLB much better (>=+5) | yes |
| `kbp` | Kabiye | qe_validated | flores_plus (1012) | 21.77 | 28.82 | +7.0 [+6.4, +7.7] | 23.5 | +1.7 [+1.0, +2.4] | NLLB much better (>=+5) | yes |
| `mlg` | Malagasy | qe_validated | flores_plus (1012) | 43.05 | 50.01 | +7.0 [+6.4, +7.6] | 47.75 | +4.7 [+4.2, +5.3] | NLLB much better (>=+5) | yes |
| `kin` | Kinyarwanda | qe_validated | flores_plus (1012) | 42.05 | 48.21 | +6.2 [+5.5, +6.8] | 43.49 | +1.4 [+0.8, +2.1] | NLLB much better (>=+5) | yes |
| `bam` | Bambara | qe_validated | flores_plus (1012) | 24.96 | 30.97 | +6.0 [+5.5, +6.5] | 28.35 | +3.4 [+2.8, +3.9] | NLLB much better (>=+5) | yes |
| `zul` | Zulu | qe_validated | flores_plus (976) | 48.59 | 54.39 | +5.8 [+5.2, +6.5] | 51.61 | +3.0 [+2.4, +3.6] | NLLB much better (>=+5) | yes |
| `lin` | Lingala | qe_validated | flores_plus (1012) | 42.04 | 47.77 | +5.7 [+5.1, +6.3] | 46.32 | +4.3 [+3.6, +4.8] | NLLB much better (>=+5) | yes |
| `ssw` | Swati | qe_validated | flores_plus (1012) | 37.57 | 43.13 | +5.6 [+5.0, +6.2] | 41.14 | +3.6 [+2.9, +4.2] | NLLB much better (>=+5) | yes |
| `luo` | Luo | qe_validated | flores_plus (1012) | 32.95 | 38.44 | +5.5 [+5.0, +6.0] | 36.78 | +3.8 [+3.3, +4.4] | NLLB much better (>=+5) | yes |
| `kab` | Kabyle | qe_validated | flores_plus (1012) | 26.67 | 31.83 | +5.2 [+4.7, +5.6] | 27.36 | +0.7 [+0.3, +1.1] | NLLB much better (>=+5) | yes |
| `kik` | Kikuyu | qe_validated | flores_plus (1012) | 31.88 | 37.02 | +5.2 [+4.6, +5.7] | 34.66 | +2.8 [+2.2, +3.3] | NLLB much better (>=+5) | yes |
| `orm` | Oromo | qe_validated | flores_plus (988) | 31.34 | 35.58 | +4.2 [+3.7, +4.7] | 31.14 | -0.2 [-0.8, +0.4] | NLLB better (+2..+5) | yes |
| `tso` | Tsonga | qe_validated | flores_plus (1012) | 46.14 | 50.11 | +4.0 [+3.3, +4.6] | 48.95 | +2.8 [+2.2, +3.4] | NLLB better (+2..+5) | yes |
| `xho` | Xhosa | qe_validated | flores_plus (1012) | 44.91 | 48.48 | +3.6 [+2.9, +4.2] | 46.43 | +1.5 [+0.9, +2.1] | NLLB better (+2..+5) | yes |
| `lug` | Luganda | qe_validated | flores_plus (1010) | 35.16 | 38.52 | +3.4 [+2.8, +3.8] | 34.56 | -0.6 [-1.1, -0.1] | NLLB better (+2..+5) | yes |
| `bem` | Bemba | qe_validated | flores_plus (1012) | 34.06 | 37.28 | +3.2 [+2.7, +3.7] | 34.74 | +0.7 [+0.2, +1.2] | NLLB better (+2..+5) | yes |
| `sna` | Shona | qe_validated | flores_plus (1012) | 39.49 | 42.62 | +3.1 [+2.6, +3.6] | 42.68 | +3.2 [+2.7, +3.7] | NLLB better (+2..+5) | yes |
| `tsn` | Setswana | qe_validated | flores_plus (982) | 45.9 | 48.36 | +2.5 [+1.9, +3.0] | 46.32 | +0.4 [-0.1, +1.0] | NLLB better (+2..+5) | yes |
| `tum` | Tumbuka | qe_validated | flores_plus (1012) | 33.72 | 35.77 | +2.0 [+1.5, +2.6] | 33.49 | -0.2 [-0.9, +0.4] | NLLB better (+2..+5) | yes |
| `som` | Somali | qe_validated | flores_plus (1012) | 40.63 | 42.61 | +2.0 [+1.5, +2.4] | 41.37 | +0.7 [+0.3, +1.1] | similar (-2..+2) | weak/uncertain |
| `hau` | Hausa | qe_validated | flores_plus (170) | 45.41 | 47.26 | +1.9 [+0.3, +3.4] | 45.4 | -0.0 [-1.4, +1.4] | similar (-2..+2) | weak/uncertain |
| `lua` | Luba-Kasai | qe_validated | flores_plus (1012) | 33.36 | 35.07 | +1.7 [+1.2, +2.2] | 33.44 | +0.1 [-0.5, +0.6] | similar (-2..+2) | weak/uncertain |
| `sag` | Sango | blend | flores_plus (1012) | 34.73 | 36.35 | +1.6 [+1.2, +2.0] | 33.9 | -0.8 [-1.3, -0.3] | similar (-2..+2) | weak/uncertain |
| `nya` | Chichewa | qe_validated | flores_plus (1012) | 43.29 | 44.84 | +1.6 [+1.0, +2.1] | 44.03 | +0.7 [+0.2, +1.3] | similar (-2..+2) | weak/uncertain |
| `nso` | Sepedi | qe_validated | flores_plus (1012) | 48.84 | 50.35 | +1.5 [+0.9, +2.0] | 48.87 | +0.0 [-0.6, +0.5] | similar (-2..+2) | weak/uncertain |
| `swh` | Swahili | qe_validated | flores_plus (1012) | 58.41 | 59.83 | +1.4 [+0.8, +2.1] | 57.96 | -0.5 [-1.1, +0.2] | similar (-2..+2) | weak/uncertain |
| `mos` | Mossi | qe_validated | flores_plus (1012) | 22.77 | 23.68 | +0.9 [+0.5, +1.3] | 20.69 | -2.1 [-2.5, -1.6] | similar (-2..+2) | weak/uncertain |
| `sot` | Sesotho | qe_validated | flores_plus (1012) | 45.45 | 46.25 | +0.8 [+0.3, +1.3] | 45.36 | -0.1 [-0.6, +0.5] | similar (-2..+2) | weak/uncertain |
| `twi` | Twi | qe_validated | flores_plus (1012) | 37.35 | 38.08 | +0.7 [+0.2, +1.3] | 34.93 | -2.4 [-3.0, -1.8] | similar (-2..+2) | weak/uncertain |
| `afr` | Afrikaans | qe_validated | flores_plus (1012) | 63.55 | 64.28 | +0.7 [+0.1, +1.4] | 61.89 | -1.7 [-2.3, -1.0] | similar (-2..+2) | weak/uncertain |
| `ibo` | Igbo | qe_validated | flores_plus (453) | 38.49 | 37.74 | -0.8 [-1.4, -0.0] | 36.9 | -1.6 [-2.4, -0.9] | similar (-2..+2) | no |
| `yor` | Yoruba | qe_validated | flores_plus (657) | 27.25 | 26.22 | -1.0 [-1.8, -0.3] | 25.18 | -2.1 [-2.7, -1.5] | similar (-2..+2) | no |
| `fon` | Fon | qe_validated | flores_plus (1012) | 21.67 | 20.27 | -1.4 [-1.9, -0.9] | 17.92 | -3.8 [-4.3, -3.2] | similar (-2..+2) | no |
| `kmb` | Kimbundu | qe_validated | flores_plus (1012) | 25.62 | 23.7 | -1.9 [-2.3, -1.5] | 24.84 | -0.8 [-1.2, -0.3] | similar (-2..+2) | no |
| `wol` | Wolof | qe_validated | flores_plus (1012) | 29.56 | 27.59 | -2.0 [-2.5, -1.5] | 23.31 | -6.3 [-6.9, -5.7] | similar (-2..+2) | no |
| `kam` | Kamba | qe_validated | flores_plus (1012) | 24.9 | 22.9 | -2.0 [-2.5, -1.6] | 26.34 | +1.4 [+1.0, +1.9] | similar (-2..+2) | no |
| `umb` | Umbundu | qe_validated | flores_plus (1012) | 25.34 | 22.88 | -2.5 [-2.9, -2.0] | 25.25 | -0.1 [-0.5, +0.3] | NLLB worse (<-2) | no |
| `dyu` | Dyula | blend | flores_plus (1012) | 18.29 | 15.45 | -2.8 [-3.2, -2.5] | 16.64 | -1.6 [-2.0, -1.3] | NLLB worse (<-2) | no |

**Teacher worse than the student: do not use as teacher or signal** (3.3B): Umbundu, Dyula (significantly worse than -2); Yoruba, Fon, Wolof, Kamba, Kimbundu (-1 to -2, intervals below zero); the 600M model is also worse for Twi, Igbo, Mossi, Afrikaans. **Kongo:** the +31 on SmolSent comes mainly from the *variety*: NLLB's `kon_Latn` produces the Kituba-like Kikongo of the SmolSent references, our model produces another variety (example in section 2.3, caution iv); treat as an owner decision, not as a quality gap of 31 points.

## 2. Stage 2: teacher SFT data (`/workspace/jobs/r14/teacher_v1`)

### 2.1 Design
* **Languages** (27): the covered languages where the 3.3B teacher is clearly better than final_1433 on human references (delta >= +2, interval above 0: 24 languages), plus `knc` (Kanuri: no SFT data at all, +9.7), `kon` (weakest language, variety question), `sag` (blend language without a good signal, +1.6: the only one below the +2 rule, flagged). **Not included** (teacher not better): Yoruba, Fon, Wolof, Kamba, Kimbundu, Umbundu, Dyula, Mossi, Ibo, Twi, Afrikaans, Sesotho, Sepedi, Chichewa, Swahili, Hausa (+1.9), Somali (+1.98, just below the rule), Luba-Kasai (+1.7).
* **Budget** (equal-token idea, weak languages most): base 140 k tokens x 2.0 for student chrF++ < 25 or Ge'ez/consensus (kea amh tir knc nus kbp bam kon), x 1.5 for 25-35, x 1.0 above; total 5.53 M of the ~6 M allowed. Languages that could not fill their budget after filtering keep what passed (run, ewe, kin, ...: see the table); a top-up round (+19 k sentences) already raised 16 of them.
* **English sources:** the language's own RL prompt pool (`rl_prompts_v3` 20,000 windows + `rl_prompts_sci_v1`), hold-out-filtered; `knc` has no pool and uses the Sango pool. Sources are web text (FineWeb-Edu 72 %, Wikipedia 24 %) plus science (PubMed, LibreTexts, arXiv, AfriScience train English); each window is cut into 1 sentence (35 %), 2-4 (25 %), 5-12 (20 %) or the whole window (20 %), exactly the S/A/B/W granularities of the science SFT stage; **translation is sentence by sentence (NLLB fails on long inputs) and chunks are the sentences joined by single spaces**, the SFT convention.
* **Decoding:** 3.3B int8_float16, beam 4, no_repeat_ngram_size 4, max 256 tokens, sentence level. About 118 k sentences were translated for the teacher data (11-28 sentences/s on the shared GPU).
* **Filters per sentence, then per chunk:** empty; verbatim copy of the English (word Jaccard > 0.6); 3-gram loop (share > 0.25 and not in the source); character-length ratio outside `p50 -/+ 2.2 x (p50-p05 | p95-p50)` of the human FLORES-dev ratio of the language (`shaping_flores_dev_v8.json`; word ratio where the char ratio is missing); **GlotLID** top-1 not in the language's accepted labels (sentences with at least the language's `min_chars`, plus the whole chunk); **AfriCOMET-QE** below the language's `t_fail` anchor (P90 of QE on mismatched translations) for the languages whose content mode is `qe` (the weakly validated "blend" languages kbp nus bam kab and the consensus/uncovered ones are not QE-filtered because their QE is not reliable). The RL anchors (`rl/anchors.py` v1) were used for the loop test (`repeat_shares`) and the tokenisation of words; `anchors_v2` (R9) was not needed.
* **Hold-out:** R4's `HoldOut` (exact sentence > 20 characters or any word 8-gram) built from AfriScience dev+test, AfriLongDoc hard+core, bench_dev_clean, bench_dev (all rows), FLORES+ dev and devtest English, NTREX-128, MAFAND, SmolSent and SALT English sides and the rl-vs-sft pool (57,581 sentences, 970,453 8-grams); whole windows are dropped on any hit (0-1 per language); an independent re-check of the English side of all 39,749 samples found no hit.
* **Format and columns:** R4's packed format (`gemma_mt.data.prepare_sft`, 8192-token packs, `meta.json` with the tokenizer fingerprint of final_1433, `iso_table`, `directions: [en2x]`), audited with `gemma_mt.data.audit_packed` (0 problems; pack fill 0.9985). Per-sample provenance is in `teacher_v1/rows.parquet` (`iso, english, translation, mode, source_dataset, english_licence, licence_class = nc_teacher, teacher = nllb-200-3.3B, teacher_decode, n_tokens, prompt_len, target_len, ids_sha1`); the packed bins cannot carry columns, so `ids_sha1` joins a row to its sample. `manifest.json` (also `/workspace/jobs/r14/manifest.json`, `reports/data/nllb600m_teacher_manifest.csv`) holds language, rows, tokens, teacher chrF++ vs human on FLORES+ devtest **and dev**, the student's chrF++, filter counts, window-selection stats and the yield figures.

### 2.2 What went in
| iso | language | rows | tokens (k) | budget (k) | teacher chrF++ devtest / dev | student chrF++ | delta | sentences translated | dropped (LID / ratio / copy / loop / QE) | tokens a teacher pass yields per 100k English sentences (M) |
|---|---|---|---|---|---|---|---|---|---|---|
| `kea` | Kabuverdianu | 2157 | 280 | 280 | 47.82 / 47.43 | 26.74 | +21.1 | 5086 | 158 / 78 / 12 / 0 / 0 |  6.1 |
| `amh` | Amharic | 2292 | 277 | 280 | 37.96 / 34.53 | 19.06 | +18.9 | 6195 | 1 / 356 / 0 / 0 / 951 |  4.5 |
| `tir` | Tigrinya | 2148 | 280 | 280 | 24.73 / 25.7 | 13.73 | +11.0 | 7280 | 675 / 141 / 6 / 0 / 1315 |  4.1 |
| `knc` | Kanuri | 1598 | 280 | 280 | 24.37 / - | 14.7 | +9.7 | 5088 | 13 / 0 / 35 / 0 / 0 |  8.2 |
| `nus` | Nuer | 1391 | 282 | 280 | 28.7 / 27.75 | 20.78 | +7.9 | 5102 | 5 / 24 / 58 / 0 / 0 |  9.6 |
| `run` | Kirundi | 1556 | 207 | 210 | 41.56 / 42.33 | 33.88 | +7.7 | 3817 | 10 / 17 / 8 / 0 / 520 |  5.4 |
| `ewe` | Ewe | 1449 | 210 | 210 | 38.32 / 38.62 | 31.12 | +7.2 | 4934 | 6 / 3 / 28 / 0 / 919 |  4.7 |
| `kbp` | Kabiye | 1303 | 282 | 280 | 28.82 / 28.79 | 21.77 | +7.0 | 5101 | 1 / 13 / 21 / 0 / 0 |  10.2 |
| `mlg` | Malagasy | 1087 | 140 | 140 | 50.01 / 49.61 | 43.05 | +7.0 | 3045 | 7 / 14 / 2 / 0 / 495 |  4.7 |
| `kin` | Kinyarwanda | 1190 | 140 | 140 | 48.21 / 47.85 | 42.05 | +6.2 | 3413 | 2 / 39 / 5 / 0 / 667 |  4.5 |
| `bam` | Bambara | 1838 | 280 | 280 | 30.97 / 31.51 | 24.96 | +6.0 | 5092 | 22 / 26 / 34 / 0 / 0 |  7.8 |
| `zul` | Zulu | 1153 | 140 | 140 | 54.39 / 53.31 | 48.59 | +5.8 | 3310 | 99 / 7 / 5 / 0 / 448 |  4.6 |
| `lin` | Lingala | 1108 | 138 | 140 | 47.77 / 48.08 | 42.04 | +5.7 | 2870 | 3 / 67 / 11 / 0 / 379 |  4.8 |
| `ssw` | Swati | 1162 | 141 | 140 | 43.13 / 43.64 | 37.57 | +5.6 | 3647 | 40 / 25 / 7 / 0 / 742 |  4.0 |
| `luo` | Luo | 1619 | 210 | 210 | 38.44 / 38.92 | 32.95 | +5.5 | 4751 | 12 / 21 / 19 / 0 / 829 |  4.6 |
| `kab` | Kabyle | 1322 | 210 | 210 | 31.83 / 32.38 | 26.67 | +5.2 | 3823 | 13 / 32 / 15 / 0 / 0 |  8.3 |
| `kik` | Kikuyu | 1336 | 204 | 210 | 37.02 / 37.54 | 31.88 | +5.2 | 3818 | 11 / 15 / 13 / 0 / 633 |  5.3 |
| `orm` | Oromo | 1715 | 210 | 210 | 35.58 / 35.9 | 31.34 | +4.2 | 6509 | 5 / 83 / 41 / 0 / 1763 |  3.5 |
| `tso` | Tsonga | 1061 | 140 | 140 | 50.11 / 49.87 | 46.14 | +4.0 | 2909 | 7 / 24 / 10 / 0 / 539 |  4.9 |
| `xho` | Xhosa | 1099 | 140 | 140 | 48.48 / 47.76 | 44.91 | +3.6 | 2819 | 13 / 3 / 8 / 0 / 374 |  5.0 |
| `lug` | Luganda | 1102 | 140 | 140 | 38.52 / 38.37 | 35.16 | +3.4 | 2830 | 9 / 32 / 9 / 0 / 418 |  5.0 |
| `bem` | Bemba | 1835 | 210 | 210 | 37.28 / 36.83 | 34.06 | +3.2 | 5940 | 19 / 56 / 85 / 0 / 1281 |  4.0 |
| `sna` | Shona | 1279 | 140 | 140 | 42.62 / 42.68 | 39.49 | +3.1 | 3730 | 2 / 7 / 31 / 0 / 944 |  3.9 |
| `tsn` | Setswana | 1018 | 140 | 140 | 48.36 / 48.69 | 45.9 | +2.5 | 2560 | 7 / 10 / 11 / 0 / 316 |  5.5 |
| `tum` | Tumbuka | 1839 | 204 | 210 | 35.77 / 36.43 | 33.72 | +2.0 | 5353 | 151 / 50 / 39 / 0 / 1186 |  3.8 |
| `sag` | Sango | 1263 | 210 | 210 | 36.35 / 36.6 | 34.73 | +1.6 | 3822 | 4 / 38 / 10 / 1 / 0 |  8.0 |
| `kon` | Kongo | 1829 | 280 | 280 | 47.81 / - | 16.7 | +31.1 | 5087 | 0 / 0 / 15 / 0 / 0 |  8.1 |
| | **total** | **39749** | **5516** | 5530 | | | | | | |

Teacher quality vs human references on FLORES+ dev (not used in training) and devtest agree (macro 38.7 dev vs 38.1 devtest over the 43/44 languages): `reports/data/nllb600m_dev_per_language.csv`. "Yield per 100 k English sentences" is the number of SFT tokens (English prompt + translation) a teacher pass produces per 100 k English sentences of that language after the filters; the pools hold ~250 k sentences per language (and the SFT corpus' English sides, FineWeb-Edu and Wikipedia are far larger), so **the data volume is limited by GPU time (25-45 sentences/s = 0.1-0.25 M tokens per minute), not by sources**: 1 M tokens of a language needs ~16-25 k translated sentences = 10-15 minutes. Languages with a low pass rate (Tigrinya 60 %, Amharic 75 %, Oromo 70 %: LID and QE; Tumbuka, Bemba) need correspondingly more.

### 2.3 Which languages benefit, estimate of the gain, and the decision rule (no SFT was run here; R7 runs SFT)
* **Teacher gap = headroom.** The student can at most approach the teacher: 21 points for Kabuverdianu, 19 Amharic, 11 Tigrinya, 10 Kanuri, 8 Nuer, 8 Kirundi, 7 Ewe/Kabiye/Malagasy ... down to +2 for Tumbuka/Tswana. Evidence on how much of a gap a small SFT stage closes: R4's science stage (5,792 sentence pairs per language, 2 passes) raised in-domain chrF++ by +0.7..+3.4 on six languages; the teacher data here is ~1.4-2.3 k chunks per language (~5 k sentences) of general web text. A cautious prior is that a 140-280 k token stage closes 10-30 % of the teacher gap on in-domain text (so +2..+6 for Kabuverdianu, +2..+5 for Amharic, +1..+3 for Tigrinya, +0.5..+2 for languages with gap 5-7 and ~0 for gap 2-3), and less out of domain. This is a prior, not a measurement.
* **Mini probe (for R7):** pick kea (gap 21, consensus-only, no QE), tir (Ge'ez, gap 11, QE-validated) and nus (gap 8, blend). Arm A = replay-only control with the same number of tokens; arm B = teacher data of the three languages (about 0.84 M tokens) + R4's replay mixture (the science-stage recipe, `scripts/build_sft_science_mix.py`, replay share 65 %); same learning rate and steps. Evaluate FLORES+ dev (selection) and devtest (report) and SmolSent for the three languages, and the replay validation set of the other 74 languages as forgetting monitor. **Decision rule:** adopt the teacher data of a language if the devtest chrF++ gain of arm B over A is >= +1.0 with the paired-bootstrap interval above 0 and no language of the replay validation loses more than 0.5 % relative loss; add tokens only while the dev gain per added 100 k tokens is >= 0.3 chrF++; drop a language's teacher data if arm B's output is closer to NLLB than to the human references on dev in the sense that chrF++(B, NLLB) rises while chrF++(B, human) does not.
* **Cautions.** (i) Teacher errors are learned: e.g. the Amharic sample "mounts" -> "mountains"; the QE/LID/ratio filters cannot catch fluent mistranslations. (ii) The teacher is better on FLORES-like Wikimedia text, which NLLB's mined training data resembles; the other benchmarks show the same sign, but the SFT data is web text. (iii) A student trained on NLLB output and then rewarded by NLLB pseudo-references (section 3) is pulled twice toward NLLB: for languages with teacher data use a lower reward weight or QE-based reward. (iv) `kon`: the teacher's variety (Kituba-like) differs from the model's current Kikongo; example (SmolSent `kg`): reference "Ngolo ya mingi ya moyo ya bantu yina ta lomba beto bakonti fwete katula yo.", NLLB "Ngolo ya ntima ya bantu yina kesosaka kupesa beto ndola kekatulaka mambu yina yonso." (same variety markers `ya`, `yina`, `ke...`), final_1433 "O mpangi ya muntu yina yina yikulindila o muntu yina yina yikutulaka mambu yina yina yikulindila ..." (looping, another variety); `knc`: no tag/data exists in the SFT run, the model has never been trained to write the `knc` tag.

## 3. Stage 3: NLLB as a reward signal

### 3.1 What was built (all flag-gated, off by default)
* `gemma_mt/gemma_mt/rl/nllb_signal.py` (new): archive-compatible key, `PseudoRefStore` (per-language parquet, window / chunk / sentence lookup via the reward's own segmenter, rejected sentences make a chunk unknown), calibration (`calibrate_scores`: anchors `t_fail = P90(mismatched)`, `t_mid = P50(human)`, `t_great = P90(human)`, AUC-based reliability), `shape01` (the reward curve without the AfriCOMET floor), `NllbSignal` (per-language weight with hard gates, `blend_chunk`, `prepare` for batching the forced passes), `NllbCT2` / `ForcedDecodingScorer` (CTranslate2, `translate` and `score_batch`).
* `gemma_mt/scripts/precompute_nllb_pseudorefs.py` (new): the precompute (resumable; reads an extra `--reuse-cache`; marks every row `reward_only_nc`; writes `LICENCE_REWARD_ONLY_NC.json`).
* `gemma_mt/patches/r14_nllb_reward_hook.patch` (60 lines against the current `rl/reward.py`; `patch -p1 < ...` from `gemma_mt/gemma_mt` after the `a/gemma_mt/rl/` prefix, or apply by hand): the `RewardConfig` fields, construction in `TranslationReward.__init__`, one batching call after the chunk plan is built, **one clearly marked line in the chunk loop** `r = self.nllb.blend_chunk(lang, seg_src[seg], seg_mt[seg], r, mode_i)`. Nothing else in `reward.py` changes; with the flag off the reward is bit-identical (`tests/test_nllb_signal.py::test_reward_hook_end_to_end`). `reward.py` itself was NOT edited (R8/R9 are editing it).
* `gemma_mt/tests/test_nllb_signal.py` (14 CPU tests; with `tests/test_consensus.py` 33 pass on the box with the patched tree): keys equal the archive key and are NFC-invariant; store lookups (window, chunk, sentence, rejected, other language); calibration anchors/AUC; curve knots; weight gating (flag off, mode, teacher delta, AUC, `use`, overrides, cap); the hook end to end; forced-decoding batching/cache/min-combination with a fake model; the precompute script with a fake translator and real parquet files.
* Scripts for the study (`gemma_mt/scripts/r14/`): `s3_signals.py` (signals for the a5 corruption rows, the policy samples, R3's stuffing attack and the dev calibration sets), `s3_analyze.py` (all tables and `nllb_calibration_v1.json`), `s3_qe_partial.py`, plus the stage-1/2 scripts.

### 3.2 Evidence
Data: R1/R2's corruption study (`/workspace/jobs/a5/hack_rows.jsonl`: 45 languages x 80 FLORES-dev sentences x 12 classes; the NLLB-covered 43 are scored here), the policy samples (`policy_samples.jsonl`: 80 prompts x 8 samples per language of the RL model), R3's lexicon stuffing attack. Calibration (anchors, AUC) uses **other FLORES-dev sentences** (300 per language, ids not in the study sets), i.e. human-validated and disjoint from the evaluation rows. Signals: A = chrF++ of the chunk against the NLLB translation of its source; B = mean per-token log-probability of the chunk under NLLB given the source (forced decoding, language code excluded). `failure` = share of sentences whose corrupted version scores >= the human reference of the same source; hard classes = mismatch, neighbour, drop_half, append_unrelated, replace_words.

**all covered** (43 languages for the 3.3B rows)

| signal | AUC human vs mismatched | hard-class failure | copy_source | copy_noisy | partial copy | append_unrelated | drop_half | shuffle |
|---|---|---|---|---|---|---|---|---|
| AfriCOMET-QE (current) | 0.871 | 0.102 | 0.011 | 0.039 | 0.243 | 0.251 | 0.016 | 0.003 |
| SSA-COMET-MTL (current ens. member) | 0.983 | 0.026 | 0.408 | 0.383 | 0.104 | 0.053 | 0.013 | 0.003 |
| NLLB-3.3B pseudo-reference chrF++ (A) | 0.974 | 0.079 | 0.027 | 0.019 | 0.094 | 0.268 | 0.003 | 0.057 |
| NLLB-3.3B forced-decoding (B) | 0.959 | 0.027 | 0.929 | 0.468 | 0.533 | 0.037 | 0.008 | 0.001 |
| min(A, B) shaped | 0.964 | 0.045 | 0.084 | 0.069 | 0.22 | 0.076 | 0.026 | 0.023 |
| mean(A, B) shaped | 0.977 | 0.024 | 0.334 | 0.157 | 0.393 | 0.055 | 0.002 | 0.008 |
| NLLB-600M pseudo-reference (A) | 0.958 | 0.085 | 0.042 | 0.031 | 0.098 | 0.261 | 0.011 | 0.078 |
| NLLB-600M forced-decoding (B) | 0.947 | 0.036 | 0.954 | 0.545 | 0.537 | 0.04 | 0.013 | 0.001 |

**teacher better (3.3B delta>=2)** (24 languages for the 3.3B rows)

| signal | AUC human vs mismatched | hard-class failure | copy_source | copy_noisy | partial copy | append_unrelated | drop_half | shuffle |
|---|---|---|---|---|---|---|---|---|
| AfriCOMET-QE (current) | 0.883 | 0.091 | 0.007 | 0.034 | 0.225 | 0.236 | 0.013 | 0.003 |
| SSA-COMET-MTL (current ens. member) | 0.987 | 0.019 | 0.385 | 0.356 | 0.086 | 0.033 | 0.009 | 0.002 |
| NLLB-3.3B pseudo-reference chrF++ (A) | 0.988 | 0.061 | 0.017 | 0.012 | 0.076 | 0.223 | 0.002 | 0.034 |
| NLLB-3.3B forced-decoding (B) | 0.978 | 0.013 | 0.92 | 0.379 | 0.508 | 0.014 | 0.003 | 0.0 |
| min(A, B) shaped | 0.984 | 0.017 | 0.032 | 0.021 | 0.178 | 0.039 | 0.002 | 0.001 |
| mean(A, B) shaped | 0.989 | 0.011 | 0.274 | 0.082 | 0.362 | 0.024 | 0.0 | 0.001 |
| NLLB-600M pseudo-reference (A) | 0.975 | 0.069 | 0.03 | 0.023 | 0.078 | 0.244 | 0.008 | 0.049 |
| NLLB-600M forced-decoding (B) | 0.959 | 0.022 | 0.946 | 0.481 | 0.512 | 0.022 | 0.009 | 0.0 |

**mode=blend** (11 languages for the 3.3B rows)

| signal | AUC human vs mismatched | hard-class failure | copy_source | copy_noisy | partial copy | append_unrelated | drop_half | shuffle |
|---|---|---|---|---|---|---|---|---|
| AfriCOMET-QE (current) | 0.777 | 0.164 | 0.027 | 0.099 | 0.433 | 0.36 | 0.036 | 0.008 |
| SSA-COMET-MTL (current ens. member) | 0.963 | 0.057 | 0.556 | 0.555 | 0.157 | 0.113 | 0.035 | 0.008 |
| NLLB-3.3B pseudo-reference chrF++ (A) | 0.926 | 0.147 | 0.065 | 0.049 | 0.18 | 0.448 | 0.014 | 0.141 |
| NLLB-3.3B forced-decoding (B) | 0.882 | 0.076 | 0.997 | 0.809 | 0.622 | 0.125 | 0.024 | 0.006 |
| min(A, B) shaped | 0.887 | 0.135 | 0.245 | 0.214 | 0.389 | 0.2 | 0.098 | 0.086 |
| mean(A, B) shaped | 0.927 | 0.069 | 0.538 | 0.41 | 0.506 | 0.168 | 0.007 | 0.031 |
| NLLB-600M pseudo-reference (A) | 0.93 | 0.126 | 0.075 | 0.057 | 0.153 | 0.383 | 0.018 | 0.12 |
| NLLB-600M forced-decoding (B) | 0.899 | 0.07 | 0.995 | 0.75 | 0.598 | 0.095 | 0.035 | 0.003 |

**Padding / stuffing (R3's attack):** a constant string of the 30 most frequent lexicon translations (+ 30 non-candidate filler words) beats the human reference of the same source in 0-15 % of the sentences with A and 0-11 % with B (fon 0.015/0.03 for A, kab 0.025/0.03, kbp 0.015/0.02, kea 0/0, sag 0.04/0.05, mos 0.025/0.045; **umb 0.14 and kmb 0.15 where the teacher is also weaker**; `reports/data/nllb600m_reward_stuffing_attack.csv`).

**Policy samples** (what GRPO sees; human chrF++ of each sample against the FLORES reference; 640 samples per language, valid samples only; * = 14-language subset for the 600M model):

| metric | Spearman with human chrF++ (pooled) | within-group Spearman (8 samples of a prompt) | partial Spearman given consensus | partial given AfriCOMET-QE | langs |
|---|---|---|---|---|---|
| group consensus (current) | 0.59 | - | - | - | 43 |
| AfriCOMET-QE (current) | 0.28 | 0.15 | - | - | 43 |
| SSA-COMET-MTL | 0.29 | 0.16 | - | - | 43 |
| NLLB-3.3B pseudo-reference (A) | 0.57 | 0.40 | 0.31 | 0.53 | 43 |
| NLLB-3.3B forced (B) | 0.43 | 0.24 | 0.10 | 0.36 | 43 |
| min(A, B) | 0.49 | 0.29 | 0.18 | 0.43 | 43 |
| mean(A, B) | 0.55 | 0.36 | 0.23 | 0.50 | 43 |
| NLLB-600M pseudo-reference (A)* | 0.57 | 0.39 | 0.32 | 0.54 | 14 |
| NLLB-600M forced (B)* | 0.39 | 0.23 | 0.07 | 0.32 | 14 |

Reading: the **pseudo-reference tracks human chrF++ as well as the group consensus does (0.57 vs 0.59) and far better than AfriCOMET-QE (0.28); inside a group it is the best of all signals (0.40 vs 0.15 for QE)**; partial correlation: given the consensus it still carries 0.31, i.e. it adds information, while AfriCOMET-QE adds 0.07 beyond the consensus (`s3_qe_partial.py`); the consensus given the pseudo-reference keeps 0.34, so the two are complementary. The forced-decoding score alone is weaker (0.43/0.24) and adds almost nothing beyond consensus (0.10). **Caveat:** the target (chrF++ against ONE human reference) is an n-gram overlap metric and favours overlap-based signals (consensus, pseudo-reference) over embedding-based QE; human judgements (SSA-MTE) would be the fairer target and were not available for these samples. The 600M model's pseudo-reference is as discriminating as the 3.3B one on these samples (0.57 on its 14-language subset vs 0.57 for the 3.3B model over all 43), but the teacher it represents is weaker (section 1), which matters because the signal pulls the student toward the teacher.

### 3.3 Per-language reliability and weights
`use = yes` needs: teacher - student >= +2 on devtest with the interval above 0; dev AUC >= 0.80; hack-study hard failure <= 0.25; within-group Spearman on the policy samples >= 0.20. Weight `w = 0.35 x AUC reliability x factor` with `factor = clip(delta/6, 0.4, 1) x clip((0.25 - failure)/0.15, 0, 1)`, capped at 0.5 (`nllb_weights` overrides). The per-language noise is real: each language has 80 sentences in the hack/policy sets, so single-language values are uncertain (bootstrap intervals are in `nllb600m_reward_hack_per_language.csv`); the aggregates are the reliable part.

| iso | group | teacher - student (chrF++) | AUC pseudo-ref (dev) | hack: AUC / hard-failure | policy: within-group Spearman / partial given consensus | forced AUC | use | weight (default 0.35 x factor) |
|---|---|---|---|---|---|---|---|---|
| `kea` | consensus_only | +21.1 | 0.998 | 1.000 / 0.013 | 0.62 / 0.65 | 1.000 | yes | 0.35 |
| `amh` | qe_validated | +18.9 | 0.994 | 0.985 / 0.017 | 0.51 / 0.40 | 0.997 | yes | 0.35 |
| `tir` | qe_validated | +11.0 | 0.996 | 0.991 / 0.045 | 0.57 / 0.44 | 0.994 | yes | 0.35 |
| `nus` | qe_validated | +7.9 | 0.971 | 0.951 / 0.152 | 0.46 / 0.28 | 0.968 | yes | 0.227 |
| `run` | qe_validated | +7.7 | 0.995 | 0.994 / 0.040 | 0.50 / 0.37 | 0.988 | yes | 0.35 |
| `ewe` | qe_validated | +7.2 | 0.983 | 0.985 / 0.085 | 0.51 / 0.51 | 0.974 | yes | 0.35 |
| `kbp` | qe_validated | +7.0 | 0.988 | 0.941 / 0.092 | 0.45 / 0.29 | 0.986 | yes | 0.35 |
| `mlg` | qe_validated | +7.0 | 0.999 | 0.999 / 0.033 | 0.41 / 0.31 | 0.997 | yes | 0.35 |
| `kin` | qe_validated | +6.2 | 0.999 | 1.000 / 0.022 | 0.44 / 0.47 | 0.996 | yes | 0.35 |
| `bam` | qe_validated | +6.0 | 0.957 | 0.975 / 0.072 | 0.40 / 0.38 | 0.928 | yes | 0.35 |
| `zul` | qe_validated | +5.8 | 0.996 | 0.999 / 0.022 | 0.50 / 0.24 | 1.000 | yes | 0.339 |
| `lin` | qe_validated | +5.7 | 0.998 | 0.994 / 0.055 | 0.50 / 0.35 | 0.998 | yes | 0.334 |
| `ssw` | qe_validated | +5.6 | 0.999 | 0.991 / 0.070 | 0.45 / 0.46 | 0.996 | yes | 0.324 |
| `luo` | qe_validated | +5.5 | 0.995 | 0.994 / 0.062 | 0.53 / 0.38 | 0.981 | yes | 0.32 |
| `kab` | qe_validated | +5.2 | 0.986 | 0.985 / 0.095 | 0.49 / 0.37 | 0.992 | yes | 0.301 |
| `kik` | qe_validated | +5.2 | 0.986 | 0.974 / 0.080 | 0.40 / 0.24 | 0.979 | yes | 0.3 |
| `orm` | qe_validated | +4.2 | 0.992 | 0.998 / 0.087 | 0.45 / 0.24 | 0.986 | yes | 0.247 |
| `tso` | qe_validated | +4.0 | 0.996 | 1.000 / 0.038 | 0.46 / 0.32 | 1.000 | yes | 0.232 |
| `xho` | qe_validated | +3.6 | 1.000 | 0.998 / 0.040 | 0.42 / 0.22 | 0.995 | yes | 0.208 |
| `lug` | qe_validated | +3.4 | 0.995 | 0.990 / 0.088 | 0.42 / 0.20 | 0.979 | yes | 0.196 |
| `bem` | qe_validated | +3.2 | 0.991 | 0.990 / 0.068 | 0.44 / 0.42 | 0.994 | yes | 0.187 |
| `sna` | qe_validated | +3.1 | 0.992 | 0.996 / 0.048 | 0.40 / 0.28 | 0.998 | yes | 0.183 |
| `tsn` | qe_validated | +2.5 | 0.998 | 0.989 / 0.065 | 0.36 / 0.39 | 0.994 | yes | 0.143 |
| `tum` | qe_validated | +2.0 | 0.999 | 0.991 / 0.062 | 0.39 / 0.37 | 0.985 | yes | 0.14 |
| `som` | qe_validated | +2.0 | 0.997 | 0.998 / 0.045 | 0.35 / 0.24 | 0.994 | no | 0 |
| `hau` | qe_validated | +1.9 | 0.996 | 0.998 / 0.022 | 0.46 / 0.19 | 1.000 | no | 0 |
| `lua` | qe_validated | +1.7 | 0.961 | 0.970 / 0.150 | 0.36 / 0.32 | 0.977 | no | 0 |
| `sag` | blend | +1.6 | 0.969 | 0.983 / 0.128 | 0.30 / 0.09 | 0.883 | no | 0 |
| `nya` | qe_validated | +1.6 | 0.998 | 1.000 / 0.048 | 0.31 / 0.23 | 0.997 | no | 0 |
| `nso` | qe_validated | +1.5 | 0.998 | 0.999 / 0.048 | 0.34 / 0.24 | 0.996 | no | 0 |
| `swh` | qe_validated | +1.4 | 0.999 | 1.000 / 0.008 | 0.38 / 0.45 | 1.000 | no | 0 |
| `mos` | qe_validated | +0.9 | 0.954 | 0.938 / 0.125 | 0.37 / 0.29 | 0.898 | no | 0 |
| `sot` | qe_validated | +0.8 | 0.996 | 1.000 / 0.022 | 0.37 / 0.43 | 0.999 | no | 0 |
| `afr` | qe_validated | +0.7 | 1.000 | 1.000 / 0.003 | 0.46 / 0.26 | 1.000 | no | 0 |
| `twi` | qe_validated | +0.7 | 0.995 | 1.000 / 0.050 | 0.45 / 0.38 | 0.986 | no | 0 |
| `ibo` | qe_validated | -0.8 | 0.994 | 0.994 / 0.055 | 0.35 / 0.18 | 0.989 | no | 0 |
| `yor` | qe_validated | -1.0 | 0.924 | 0.909 / 0.175 | 0.15 / 0.09 | 0.943 | no | 0 |
| `fon` | qe_validated | -1.4 | 0.908 | 0.909 / 0.172 | 0.28 / 0.17 | 0.827 | no | 0 |
| `kmb` | qe_validated | -1.9 | 0.937 | 0.919 / 0.133 | 0.24 / 0.21 | 0.902 | no | 0 |
| `wol` | qe_validated | -2.0 | 0.986 | 0.961 / 0.123 | 0.28 / 0.16 | 0.950 | no | 0 |
| `kam` | qe_validated | -2.0 | 0.888 | 0.859 / 0.247 | 0.31 / 0.30 | 0.862 | no | 0 |
| `umb` | qe_validated | -2.5 | 0.907 | 0.914 / 0.165 | 0.27 / 0.21 | 0.798 | no | 0 |
| `dyu` | blend | -2.8 | 0.833 | 0.813 / 0.237 | 0.19 / 0.18 | 0.794 | no | 0 |

Not calibrated (no study data): `kon` (FLORES has no Kongo; SmolSent has no policy samples), `knc` and `tzm` (not RL languages). The `use` rows for qe-mode languages (amh tir run ewe mlg kin ...) apply only if listed in `nllb_languages`; by default the signal covers the `consensus` and `blend` content modes, i.e. **`kea` (consensus), `kbp nus bam kab` (blend)** among the languages calibrated here.

### 3.4 Risks and where it would hurt
* **Style collapse / teacher errors:** the signal rewards chrF++ overlap with NLLB; where the teacher is worse than the student (`dyu sag mos kam kmb umb fon yor wol` and the other `use = no`) weight is 0 by construction. Even where the teacher is better, a student that copies NLLB's mistakes is rewarded; the weight cap (0.5) and the typical 0.23-0.35 leave the QE/consensus signal the larger share; watch dev chrF++ against human references of the covered weak languages (kea kbp nus bam kab) and stop/lower `nllb_weight_default` if it falls while chrF++ against NLLB rises.
* **Double dipping with teacher SFT:** see 2.3 (iii).
* **Padding:** chrF++'s recall weight makes the pseudo-reference lenient to appended text (27 % of "human + unrelated sentence" score >= the human one); the existing length/extra-output penalties and the padded-output detectors of anchors v2 remain necessary. `min(A, B)` fixes this (0.04) but costs a forward pass and inherits B's copy weakness only partly (copy_source 0.03-0.08 with the min, partial copies 0.18-0.22).
* **Long inputs:** NLLB works at sentence level; the store is keyed by sentence and the reward's chunk plan already works sentence by sentence, but a chunk that merges sentences whose NLLB translations are missing gets no value (identity). Documents were translated sentence by sentence and are re-joined by lookup.
* **Languages NLLB does not cover:** no signal, by design.
* **Benchmark leakage in NLLB:** NLLB was trained with FLORES-200 dev and a mined corpus rich in Wikimedia text; FLORES+ numbers are the most favourable to it, which is why the other benchmarks are reported (same sign, +3.5..+5.5).

### 3.5 Exact configuration, patch and cost per step
Apply `gemma_mt/patches/r14_nllb_reward_hook.patch`, copy `reports/data/nllb_reward_calibration_v1.json` next to the other calibration files, point `nllb_pseudoref_dir` at `/workspace/jobs/r14/pseudorefs_reward_only_nc_v1` and add under `reward:` of the RL yaml:
```yaml
  # R14 (D-66): NLLB pseudo-reference signal, reward-only (rl/nllb_signal.py). Off unless nllb_signal: true.
  nllb_signal: true
  nllb_pseudoref_dir: /workspace/jobs/r14/pseudorefs_reward_only_nc_v1
  nllb_calibration_path: /workspace/jobs/r7/assets/calibration/nllb_reward_calibration_v1.json
  nllb_modes: [consensus, blend]       # kea kbp nus bam kab among the calibrated languages
  nllb_languages: []                   # add qe-mode languages (amh tir run ewe mlg kin ...) to use it as a low-weight check there
  nllb_weight_default: 0.35            # w = default x reliability x factor (calibration), capped by nllb_weight_max
  nllb_weight_max: 0.5
  nllb_weights: {}                     # per-language override; 0 disables (e.g. {kbp: 0.2})
  nllb_min_teacher_delta: 2.0
  nllb_min_auc: 0.80
  nllb_forced: false                   # (B) forced-decoding score, used only as min(A, B); needs a GPU copy of NLLB-3.3B int8 (~4-6 GB)
  nllb_forced_combine: min
  nllb_forced_min_auc: 0.80
  nllb_ct2_dir: /workspace/jobs/r14/models/nllb3p3b-ct2-int8
  nllb_hf_dir: /workspace/jobs/r14/models/nllb600m-hf     # tokenizer files only
  nllb_ct2_compute_type: int8_float16
  nllb_forced_batch_tokens: 4096
```
**Cost per step** (final yaml: 84 prompts x group 16 = 1,344 completions; measured on the shared RTX PRO 6000): (A) is a dict lookup plus a sentence chrF++ (0.26 ms per chunk, store load 0.4 s per language, ~2 MB of RAM per 15 k sentences): for the 5 default languages (~6.5 % of the draws, ~3 chunks per short completion) about 260 chunks = **0.07 s per step**; with all 24 `use` languages (31 % of draws) ~1,250 chunks = **0.3 s**; long bins (b1024/b2048: 20-40 chunks per completion) up to ~3,000 chunks = 0.8 s. (B): **8.9 ms per chunk pair with 3.3B int8 (0.57 s per 64 pairs), 5.2 ms with 600M (0.33 s per 64)** while sharing the GPU with other jobs: 2.3 s per step for 260 pairs, 11 s for 1,250, ~27 s for 3,000 pairs in the long bins; plus 4-6 GB of GPU memory. Relative to a step of tens of seconds (vLLM sampling, scoring forward, AfriCOMET), (A) is free and (B) is affordable but not needed; the 600M model is 1.7x cheaper per pair but a weaker teacher.

## 4. Files
* Laptop: `gemma_mt/gemma_mt/rl/nllb_signal.py`, `gemma_mt/scripts/precompute_nllb_pseudorefs.py`, `gemma_mt/tests/test_nllb_signal.py`, `gemma_mt/patches/r14_nllb_reward_hook.patch`, `gemma_mt/scripts/r14/` (`nllb_langmap.py`, `nllb_ct2.py`, `nllb_run.py`, `dump_rows.py`, `score_stage1.py`, `make_decision_table.py`, `s2_holdout.py`, `s2_translate.py`, `s2_qe.py`, `s2_build.py`, `s2_finalize.py`, `s3_signals.py`, `s3_analyze.py`, `s3_qe_partial.py`, `make_report_tables.py`), `DECISIONS.md` D-66, this report and the stage-1 report.
* Data (`reports/data/`): `nllb600m_decision_table.csv`, `nllb600m_per_language_benchmark.csv` (every language x benchmark x subset score and interval), `nllb600m_macro.csv`, `nllb600m_dev_per_language.csv`, `nllb600m_teacher_manifest.csv`, `nllb600m_reward_{hack_per_language,hack_aggregate,policy_per_language,stuffing_attack}.csv`, `nllb_reward_calibration_v1.json`.
* Box (`/workspace/jobs/r14`): `teacher_v1/` (packed data, `rows.parquet`, `manifest.json`), `READY_TEACHER`, `manifest.json`, `STAGE1_DONE`, `s2/` (sentence translations `trans_<iso>.jsonl`, QE scores, window plans), `pseudorefs_reward_only_nc_v1/` (`<iso>.parquet`, caches, `LICENCE_REWARD_ONLY_NC.json`), `s3/` (signals and analysis), `out/` (benchmark translations, tables). Scratch **was deleted at the end** (the CTranslate2 builds `models/` 3.4 GB, the `pylibs` with ctranslate2 + CUDA-12 cuBLAS/cuDNN 2.4 GB, the per-sentence outputs of our own benchmark run `ours/` 150 MB, the code tree copy); what remains in `/workspace/jobs/r14` is 0.38 GB of deliverables and logs; to rebuild: `uv pip install --target pylibs ctranslate2 nvidia-cublas-cu12 nvidia-cudnn-cu12` and `snapshot_download("OpenNMT/nllb-200-3.3B-ct2-int8")` (the tokenizer files are in `models/nllb600m-hf`).
* Licences: every NLLB-derived file is CC-BY-NC-4.0: `teacher_v1` rows carry `licence_class = nc_teacher`, the pseudo-reference parquets and their directory carry `reward_only_nc`; the English sources keep their own licences (`english_licence` column). The owner accepted the NC licence for the model that is trained on `nc_teacher` data; the `reward_only_nc` files must never enter any training set.

## 5. Uncertainty and negative results
* **Coverage is the main negative result** (section 0.2). NLLB does not help the 22 consensus-only and 8 blend languages that most need an external signal, except Kabuverdianu, Dyula (teacher worse), Sango (not clearly better) and Kongo (variety question).
* The 600M model is not a good teacher for us: +1.7 on average, worse than the student in Atlantic-Congo non-Bantu (-1.5) and for 5 languages individually; its reward signal is as discriminating as the 3.3B one but represents a worse teacher. 3.3B greedy and a full NTREX/MAFAND run of the 600M model were not run (GPU contention; the macro comparison on FLORES+ is the basis).
* The study sets are small per language (80 sentences x 12 corruption classes; 80 prompts x 8 samples); per-language weights are therefore soft, aggregates are firm. Policy samples come from the RL model, whose errors are not those of a model under GRPO pressure (reward hacking would change the distribution); the signal has never been run inside a training loop.
* Teacher data were not trained on (no SFT here); the gain estimates are priors. The yield per 100 k sentences is measured on the 2.5-7 k sentences actually translated per language; filters and pass rates vary by source type.
* Within the QE filter, the "blend"/consensus-mode languages (kbp nus bam kab kea kon) have no reference-free validity check; their teacher quality rests on devtest chrF++ against humans (kbp +7.0, nus +7.9, bam +6.0, kab +5.2, kea +21.1).
