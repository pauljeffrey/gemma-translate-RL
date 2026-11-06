# Post-editor proof experiment, run log and results (2026-11-03)

Owner question: does a small post-editor SFT (English source + machine-translation draft -> HUMAN reference) improve the draft translations of our RL model? Everything below was run on a dedicated RTX 3090 Ti (24 GB) Vast instance; nothing on the A100 / RL runs was touched. Design: `reports/post_editor_plan_2026-11-03.md`; arms after RL stage A: `reports/experiments_after_stage_A_2026-11-03.md`. Data files of every table: `reports/data/post_editor_2026-11-03/` (`e1/` = this report, `selection/` = data coverage).

## 0. Verdict (E1, the only editor trained)

* **Applied to every sentence the editor does NOT help.** On the RL stage-A greedy drafts (clean subsets, macro over languages, chrF++): FLORES+ -0.01 [-0.11, +0.10], NTREX -0.58 [-0.75, -0.41], MAFAND -0.93 [-1.14, -0.74], SmolSent -0.37 [-0.52, -0.24]; AfriCOMET (reference) falls by about 0.01 on every benchmark (all CIs below 0). It is far below the equal-compute baselines beam-4 and MBR-16 (section 2).
* **It repairs hard failures and hurts good drafts.** Where the draft is a loop / unfinished / chrF++ < 10 the editor gains +3.5 to +6.7 chrF++ per sentence; where the draft is decent (chrF++ >= 30) it loses 0.9 to 1.8 per sentence because it rewrites about half of all drafts whatever their quality (it cannot tell a good draft from a bad one: no quality signal in the prompt).
* **A repair-only use works.** Applying the SAME editor only to drafts that are empty, unfinished or loops (a rule, 3.9% of the sentences; no extra model) gives FLORES+ +0.70 [+0.63, +0.78], NTREX +0.29, MAFAND +0.15, SmolSent +0.47 chrF++ (all CIs above 0, 0 languages worse on FLORES+), but AfriCOMET and SSA-COMET do not move (between -0.0007 and +0.0009): much of that chrF++ gain is length (removing the tail of a loop), which AfriCOMET does not reward. Beam-4 does the same job for the whole sentence set and is stronger (+1.6 chrF++, +0.023 AfriCOMET on FLORES+).
* **The editor behaves like a re-translation at about the level of our own model.** It lowers drafts that are better than that level (NLLB-3.3B -4.0 chrF++ / -0.05 AfriCOMET, RL beam-4 -0.9 / -0.02 on FLORES+), leaves our greedy drafts flat, and raises drafts that are worse: MADLAD-400-3B drafts (28.3 chrF++ on FLORES+) gain +3.0 chrF++ and +0.05 AfriCOMET (NTREX +4.5 / +0.05), ending at about 31.4, i.e. at the level of the RL model's own greedy output. The gain of an editor of this type is the gap between the draft and the editor's own translation quality, which is also what simply re-translating with the RL model would give.
* **Recommendation:** do NOT train more editor variants of this kind (E2 external drafts, E2q QE-filtered targets, E3 held-out system): the E1 failure is structural (the target is a reference of unknown relation to the draft; the model learns "produce a reference-like sentence" and drops information of a good draft), not a lack of draft diversity. Worth considering instead: (a) the repair-only gate as a free post-processor (a loop / unfinished detector + this editor) if beam is too expensive; (b) an editor with an explicit draft-quality signal (QE bucket in the prompt) trained so that the identity map is optimal for good drafts; (c) editor-RL with the improvement-over-draft reward, which optimises exactly the quantity that SFT cannot (the plan, section 9). Details and caveats below; E2 / E2q / E3 were HELD by the owner and were not trained.

## 1. What was run

| step | what | where / settings |
|---|---|---|
| data | selection v2 for E1 = 47,423 human pairs, 115 languages (all 81 RL languages covered; religious share 26%, 28 of the 81 RL languages religious-only), drafts from RL stage-A greedy / T0.7 / T1.0 / beam-4 / MBR-16 (orm ewe lug tir), SFT-final greedy / T0.7, A0 270M greedy, 8 kinds of reference corruption, 5.2% draft == reference, 2.2% near-reference; edit-rate matching to the RL benchmark drafts | `reports/data/post_editor_2026-11-03/selection_v2/`, plan section 1-2 |
| E1 set | 61,274 train + 2,636 validation examples, 8,974,899 tokens, mean sentence chrF++ of the matched training drafts 36.02 vs 30.57 for the RL drafts on the benchmarks (the matching is imperfect: the training drafts are a little better) | `e1/packed_e1/` |
| E1 training | base weights = RL stage A (`<HF_ORG>/gemma-0.4B-translate-rl@stage-a-b64`, step 200; the editor is initialised from the RL weights, NOT from SFT-final), full fine-tune, Muon + AdamW, lr 2e-5 -> 1e-5, 2 epochs = 278 steps of 32 packs x 2,048 tokens, micro batch 2 with gradient checkpointing, 24 GB | train log `/workspace/logs/train_e1.log`, `e1/run_e1/out/` |
| evaluation | RL stage-A drafts: greedy (34,500 sentences: FLORES+ devtest 54 languages x 250, NTREX 24 x 250, MAFAND 15 x 250, SmolSent 45 x 250; deterministic subsample, the SAME sentences before / after) and beam-4 / MBR-16 (FLORES+ + NTREX, 100 per language); the editor runs greedy | `e1/eval_e1s/` (greedy), `e1/eval_e1b/` (beam-4, NLLB, MADLAD drafts) |
| scores | chrF++ and spBLEU (sacrebleu, NFC) with paired sentence bootstrap (1,000 resamples, equal weight per language, same resample for both); AfriCOMET reference-based (STL-1.1) and AfriCOMET-QE on the 40 / 15 / 21 / 36 covered languages of the four benchmarks (never mixed with the uncovered ones), SSA-COMET-MTL reference-free on all languages as its own table | `e1/comet_e1/`, `e1/comet_ssa/` |

Failures that were fixed on the way (all environment, none changed a number): venv_comet lacked `pkg_resources` (setuptools < 81), venv_vllm lacked `fasttext` (the first beam-draft evaluation died in the GlotLID step: it was re-run), the first E1 training attempt ran out of memory at micro batch 4 without checkpointing (re-run at micro batch 2 + gradient checkpointing; +25 min), vLLM beam search is unusably slow (the HF batched beam is used), MADLAD-3B needs a small batch on 24 GB.

## 2. Main result: editor vs draft, with AfriCOMET (clean subsets, macro over languages)

chrF++ / spBLEU, paired bootstrap, delta = after - draft. Draft = RL stage-A greedy. `editor (every sentence)` = E1 on all drafts; `editor (repair only)` = E1 on drafts that are empty / unfinished / loops, drafts left alone otherwise.

| draft source / system | benchmark | languages | chrF++ draft | chrF++ after | delta [95% CI] | spBLEU delta [CI] | langs better / ns / worse |
|---|---|---:|---:|---:|---|---|---|
| editor (every sentence) | flores_plus | 54 | 30.48 | 30.47 | **-0.01** [-0.11, +0.10] | -0.14 [-0.23, -0.02] | 12 / 25 / 17 |
| editor (every sentence) | ntrex | 24 | 36.54 | 35.95 | **-0.58** [-0.75, -0.41] | -0.34 [-0.49, -0.10] | 1 / 11 / 12 |
| editor (every sentence) | mafand | 15 | 40.45 | 39.52 | **-0.93** [-1.14, -0.74] | -0.86 [-1.08, -0.60] | 1 / 7 / 7 |
| editor (every sentence) | smol_sent | 45 | 31.13 | 30.76 | **-0.37** [-0.52, -0.24] | +0.02 [-0.18, +0.13] | 5 / 29 / 11 |
| editor (repair only) | flores_plus | 54 | 30.48 | 31.18 | **+0.70** [+0.63, +0.78] | +0.31 [+0.26, +0.42] | 18 / 35 / 1 |
| editor (repair only) | ntrex | 24 | 36.54 | 36.83 | **+0.29** [+0.19, +0.40] | +0.23 [+0.13, +0.44] | 5 / 19 / 0 |
| editor (repair only) | mafand | 15 | 40.45 | 40.60 | **+0.15** [+0.06, +0.27] | +0.11 [+0.01, +0.32] | 1 / 14 / 0 |
| editor (repair only) | smol_sent | 45 | 31.13 | 31.59 | **+0.47** [+0.37, +0.57] | +0.53 [+0.37, +0.61] | 11 / 34 / 0 |

Equal-compute baselines on the same sentences (scored against the SAME greedy draft; beam-4 and MBR-16 were generated on a 100-sentence-per-language FLORES+ / NTREX subset, so their rows cover those two benchmarks and the comparison with the editor row must use the same benchmark; MBR-16 costs about 17 greedy passes, beam-4 about 2-4):

| system | benchmark | subset | chrF++ delta [CI] | spBLEU delta [CI] | langs better / ns / worse |
|---|---|---|---|---|---|
| beam4 | flores_plus | clean | +1.56 [+1.37, +1.74] | +1.13 [+0.99, +1.29] | 34 / 20 / 0 |
| beam4 | ntrex | clean | +1.69 [+1.42, +1.96] | +1.45 [+1.18, +1.77] | 20 / 4 / 0 |
| mbr16 | flores_plus | clean | +2.12 [+1.96, +2.28] | +0.60 [+0.46, +0.75] | 44 / 10 / 0 |
| mbr16 | ntrex | clean | +1.64 [+1.36, +1.91] | +0.89 [+0.64, +1.17] | 20 / 4 / 0 |

AfriCOMET (same sentences and the same paired bootstrap; metric rows: reference-based AfriCOMET-STL-1.1, AfriCOMET-QE-STL-1.1; only languages the model covers are scored and counted in `languages (covered)`):

| comparison | benchmark | metric | languages (covered) | macro draft | macro after | delta [95% CI] | langs better / ns / worse |
|---|---|---|---:|---:|---:|---|---|
| editor (every sentence) | flores_plus | AfriCOMET (reference) | 40 | 0.6184 | 0.6055 | **-0.0129** [-0.0142, -0.0115] | 1 / 23 / 16 |
| editor (every sentence) | flores_plus | AfriCOMET-QE | 40 | 0.5798 | 0.5652 | **-0.0147** [-0.0161, -0.0133] | 0 / 21 / 19 |
| editor (every sentence) | ntrex | AfriCOMET (reference) | 21 | 0.6120 | 0.6023 | **-0.0097** [-0.0116, -0.0079] | 0 / 17 / 4 |
| editor (every sentence) | ntrex | AfriCOMET-QE | 21 | 0.5688 | 0.5575 | **-0.0114** [-0.0133, -0.0097] | 0 / 13 / 8 |
| editor (every sentence) | mafand | AfriCOMET (reference) | 15 | 0.6514 | 0.6387 | **-0.0127** [-0.0152, -0.0106] | 1 / 11 / 3 |
| editor (every sentence) | mafand | AfriCOMET-QE | 15 | 0.6191 | 0.6034 | **-0.0157** [-0.0181, -0.0135] | 0 / 8 / 7 |
| editor (every sentence) | smol_sent | AfriCOMET (reference) | 36 | 0.5903 | 0.5809 | **-0.0093** [-0.0107, -0.0080] | 2 / 23 / 11 |
| editor (every sentence) | smol_sent | AfriCOMET-QE | 36 | 0.5378 | 0.5266 | **-0.0112** [-0.0125, -0.0099] | 1 / 19 / 16 |
| editor (repair only) | flores_plus | AfriCOMET (reference) | 40 | 0.6184 | 0.6190 | **+0.0007** [+0.0004, +0.0010] | 5 / 35 / 0 |
| editor (repair only) | flores_plus | AfriCOMET-QE | 40 | 0.5798 | 0.5805 | **+0.0007** [+0.0004, +0.0010] | 5 / 35 / 0 |
| editor (repair only) | ntrex | AfriCOMET (reference) | 21 | 0.6120 | 0.6126 | **+0.0007** [+0.0003, +0.0011] | 3 / 18 / 0 |
| editor (repair only) | ntrex | AfriCOMET-QE | 21 | 0.5688 | 0.5693 | **+0.0005** [+0.0002, +0.0009] | 1 / 20 / 0 |
| editor (repair only) | mafand | AfriCOMET (reference) | 15 | 0.6514 | 0.6517 | **+0.0003** [+0.0000, +0.0006] | 0 / 15 / 0 |
| editor (repair only) | mafand | AfriCOMET-QE | 15 | 0.6191 | 0.6193 | **+0.0003** [-0.0000, +0.0007] | 0 / 15 / 0 |
| editor (repair only) | smol_sent | AfriCOMET (reference) | 36 | 0.5903 | 0.5912 | **+0.0009** [+0.0006, +0.0013] | 5 / 30 / 1 |
| editor (repair only) | smol_sent | AfriCOMET-QE | 36 | 0.5378 | 0.5386 | **+0.0009** [+0.0005, +0.0012] | 5 / 30 / 1 |
| beam-4 | flores_plus | AfriCOMET (reference) | 40 | 0.6182 | 0.6414 | **+0.0233** [+0.0209, +0.0255] | 33 / 7 / 0 |
| beam-4 | flores_plus | AfriCOMET-QE | 40 | 0.5808 | 0.6047 | **+0.0240** [+0.0213, +0.0264] | 32 / 8 / 0 |
| beam-4 | ntrex | AfriCOMET (reference) | 21 | 0.6104 | 0.6347 | **+0.0243** [+0.0206, +0.0281] | 16 / 5 / 0 |
| beam-4 | ntrex | AfriCOMET-QE | 21 | 0.5676 | 0.5895 | **+0.0219** [+0.0183, +0.0258] | 16 / 5 / 0 |
| MBR-16 | flores_plus | AfriCOMET (reference) | 40 | 0.6182 | 0.6277 | **+0.0096** [+0.0075, +0.0117] | 13 / 27 / 0 |
| MBR-16 | flores_plus | AfriCOMET-QE | 40 | 0.5808 | 0.5887 | **+0.0079** [+0.0055, +0.0102] | 9 / 31 / 0 |
| MBR-16 | ntrex | AfriCOMET (reference) | 21 | 0.6104 | 0.6208 | **+0.0104** [+0.0073, +0.0136] | 8 / 13 / 0 |
| MBR-16 | ntrex | AfriCOMET-QE | 21 | 0.5676 | 0.5753 | **+0.0077** [+0.0044, +0.0110] | 6 / 15 / 0 |

SSA-COMET-MTL (reference-free, all languages of the benchmark rows; its own table, not comparable with the AfriCOMET macros):

| comparison | benchmark | metric | languages (covered) | macro draft | macro after | delta [95% CI] | langs better / ns / worse |
|---|---|---|---:|---:|---:|---|---|
| editor (every sentence) | flores_plus | SSA-COMET-MTL (QE) | 54 | 0.4479 | 0.4320 | **-0.0158** [-0.0172, -0.0145] | 2 / 23 / 29 |
| editor (every sentence) | ntrex | SSA-COMET-MTL (QE) | 24 | 0.4791 | 0.4665 | **-0.0125** [-0.0145, -0.0107] | 1 / 12 / 11 |
| editor (every sentence) | mafand | SSA-COMET-MTL (QE) | 15 | 0.5479 | 0.5315 | **-0.0165** [-0.0195, -0.0138] | 1 / 8 / 6 |
| editor (every sentence) | smol_sent | SSA-COMET-MTL (QE) | 45 | 0.4570 | 0.4496 | **-0.0075** [-0.0088, -0.0061] | 5 / 26 / 14 |
| editor (repair only) | flores_plus | SSA-COMET-MTL (QE) | 54 | 0.4479 | 0.4472 | **-0.0007** [-0.0011, -0.0003] | 3 / 49 / 2 |
| editor (repair only) | ntrex | SSA-COMET-MTL (QE) | 24 | 0.4791 | 0.4798 | **+0.0007** [+0.0003, +0.0011] | 2 / 22 / 0 |
| editor (repair only) | mafand | SSA-COMET-MTL (QE) | 15 | 0.5479 | 0.5482 | **+0.0003** [-0.0001, +0.0007] | 1 / 14 / 0 |
| editor (repair only) | smol_sent | SSA-COMET-MTL (QE) | 45 | 0.4570 | 0.4579 | **+0.0008** [+0.0005, +0.0012] | 4 / 41 / 0 |

Reading: chrF++ and AfriCOMET agree on the editor applied to everything (negative everywhere), and agree that beam-4 is clearly better than the draft. They disagree on the repair-only gain (chrF++ +0.15 to +0.70, AfriCOMET +0.0003 to +0.0009, SSA-COMET -0.0007 to +0.0008): chrF++ is sensitive to the long garbage tails of loops, AfriCOMET (length / entity bias, `reports/data/africomet_bias_2026-11-02`) is not. Neither is circular here: the editor was trained on human references and judged against human references; AfriCOMET only sits next to chrF++.

## 3. Other draft sources: beam-4, NLLB-3.3B, MADLAD-3B, AfriScience-MT test

| draft source / system | benchmark | languages | chrF++ draft | chrF++ after | delta [95% CI] | spBLEU delta [CI] | langs better / ns / worse |
|---|---|---:|---:|---:|---|---|---|
| RL beam-4 drafts | flores_plus | 54 | 32.19 | 31.28 | **-0.91** [-1.05, -0.76] | -0.62 [-0.76, -0.51] | 3 / 18 / 33 |
| RL beam-4 drafts | ntrex | 24 | 37.98 | 36.73 | **-1.25** [-1.49, -1.04] | -0.93 [-1.17, -0.74] | 0 / 8 / 16 |
| NLLB-200-3.3B greedy drafts | flores_plus | 50 | 36.32 | 32.30 | **-4.02** [-4.29, -3.77] | -2.98 [-3.25, -2.72] | 2 / 12 / 36 |
| NLLB-200-3.3B greedy drafts | ntrex | 22 | 40.42 | 37.50 | **-2.93** [-3.26, -2.58] | -2.80 [-3.19, -2.35] | 0 / 3 / 19 |
| MADLAD-400-3B greedy drafts | flores_plus | 37 | 28.35 | 31.39 | **+3.04** [+2.66, +3.44] | +1.63 [+1.29, +1.99] | 23 / 10 / 4 |
| MADLAD-400-3B greedy drafts | ntrex | 21 | 31.30 | 35.84 | **+4.54** [+4.01, +5.05] | +2.81 [+2.33, +3.32] | 13 / 7 / 1 |

Reading the three sources together: edited chrF++ on FLORES+ is 31.3 (beam-4 drafts, from 32.2), 32.3 (NLLB drafts, from 36.3) and 31.4 (MADLAD-3B drafts, from 28.3): whatever the draft, the output sits around 31-32, the level of our own model. The editor does not add information from a better draft, it replaces it.
NLLB / MADLAD drafts are generated with the FLORES / NTREX source of the same 100-sentence subsets (languages the system supports: NLLB 52 of the 116 languages of the pool, MADLAD-3B (the 7B model was too slow on 24 GB and was abandoned after 28 minutes) 38 of 57 languages tried: `e1/bench_mt/mt_coverage.csv`). Their drafts are BETTER than ours on the languages they cover (NLLB-3.3B 36.3 vs 30.5 chrF++ on FLORES+), and the editor rewrites them towards our model's output, which costs 3-4 chrF++: an editor trained on references whose relation to the draft is unknown does not preserve a better draft.

AfriScience-MT TEST (sentence level, 970 x 6 languages; draft = RL greedy): chrF++ 41.37 -> 41.11, delta -0.26 [-0.56, +0.00], languages better / ns / worse 3 / 2 / 1. (AfriScience TRAIN is not in E1's data: selection v2 predates it; it is in the v4 selection that was prepared for E2.)

| comparison | benchmark | metric | languages (covered) | macro draft | macro after | delta [95% CI] | langs better / ns / worse |
|---|---|---|---:|---:|---:|---|---|
| editor (every sentence) | afriscience_test | AfriCOMET (reference) | 6 | 0.6758 | 0.6523 | **-0.0235** [-0.0259, -0.0212] | 1 / 3 / 2 |
| editor (every sentence) | afriscience_test | AfriCOMET-QE | 6 | 0.6395 | 0.6118 | **-0.0277** [-0.0301, -0.0255] | 0 / 3 / 3 |

AfriCOMET on the beam-4 / NLLB / MADLAD drafts before and after the editor:

| comparison | benchmark | metric | languages (covered) | macro draft | macro after | delta [95% CI] | langs better / ns / worse |
|---|---|---|---:|---:|---:|---|---|
| editor on RL beam-4 drafts | flores_plus | AfriCOMET (reference) | 40 | 0.6414 | 0.6219 | **-0.0196** [-0.0218, -0.0174] | 0 / 11 / 29 |
| editor on RL beam-4 drafts | flores_plus | AfriCOMET-QE | 40 | 0.6047 | 0.5826 | **-0.0222** [-0.0245, -0.0199] | 0 / 7 / 33 |
| editor on RL beam-4 drafts | ntrex | AfriCOMET (reference) | 21 | 0.6347 | 0.6192 | **-0.0156** [-0.0186, -0.0127] | 0 / 14 / 7 |
| editor on RL beam-4 drafts | ntrex | AfriCOMET-QE | 21 | 0.5895 | 0.5722 | **-0.0173** [-0.0203, -0.0144] | 0 / 13 / 8 |
| editor on NLLB-3.3B drafts | flores_plus | AfriCOMET (reference) | 40 | 0.6709 | 0.6240 | **-0.0469** [-0.0500, -0.0437] | 0 / 2 / 38 |
| editor on NLLB-3.3B drafts | flores_plus | AfriCOMET-QE | 40 | 0.6365 | 0.5818 | **-0.0547** [-0.0579, -0.0516] | 0 / 1 / 39 |
| editor on NLLB-3.3B drafts | ntrex | AfriCOMET (reference) | 21 | 0.6827 | 0.6339 | **-0.0488** [-0.0532, -0.0442] | 0 / 0 / 21 |
| editor on NLLB-3.3B drafts | ntrex | AfriCOMET-QE | 21 | 0.6363 | 0.5845 | **-0.0518** [-0.0561, -0.0470] | 0 / 0 / 21 |
| editor on MADLAD-3B drafts | flores_plus | AfriCOMET (reference) | 30 | 0.5283 | 0.5804 | **+0.0520** [+0.0470, +0.0575] | 15 / 11 / 4 |
| editor on MADLAD-3B drafts | flores_plus | AfriCOMET-QE | 30 | 0.5173 | 0.5420 | **+0.0247** [+0.0193, +0.0303] | 12 / 10 / 8 |
| editor on MADLAD-3B drafts | ntrex | AfriCOMET (reference) | 20 | 0.5353 | 0.5857 | **+0.0504** [+0.0442, +0.0569] | 10 / 7 / 3 |
| editor on MADLAD-3B drafts | ntrex | AfriCOMET-QE | 20 | 0.5090 | 0.5430 | **+0.0340** [+0.0274, +0.0403] | 8 / 7 / 5 |

## 4. Strata

**Religious-only / religious-mostly vs mixed-register languages** (by the editor's training data; FLORES+ clean, macro of the languages in the stratum; `religious-only` = no non-religious pair in the training set, `religious-mostly` = > 50% religious):

| stratum | benchmark | languages | chrF++ draft | chrF++ after | delta [95% CI] | langs better / ns / worse |
|---|---|---:|---:|---:|---|---|
| mixed-register | flores_plus | 43 | 33.21 | 32.97 | -0.24 [-0.35, -0.12] | 7 / 19 / 17 |
| religious-mostly | flores_plus | 4 | 33.18 | 33.35 | +0.17 [-0.39, +0.74] | 0 / 4 / 0 |
| religious-only | flores_plus | 5 | 15.01 | 16.04 | +1.02 [+0.63, +1.41] | 3 / 2 / 0 |
| mixed-register | ntrex | 22 | 36.98 | 36.32 | -0.66 [-0.83, -0.49] | 1 / 9 / 12 |
| religious-mostly | ntrex | 2 | 31.64 | 31.94 | +0.30 [-0.37, +0.93] | 0 / 2 / 0 |
| mixed-register | smol_sent | 37 | 32.32 | 31.93 | -0.39 [-0.55, -0.26] | 4 / 24 / 9 |
| religious-mostly | smol_sent | 3 | 25.54 | 25.83 | +0.29 [-0.19, +0.79] | 0 / 3 / 0 |
| religious-only | smol_sent | 5 | 25.68 | 25.08 | -0.60 [-1.15, -0.02] | 1 / 2 / 2 |
| mixed-register | mafand | 15 | 40.45 | 39.52 | -0.93 [-1.14, -0.74] | 1 / 7 / 7 |

The editor gains in the religious-only languages of FLORES+ (+1.0, CI above 0: their drafts are very poor, so there is room, and the training text IS their only human text) and loses in the mixed-register languages (-0.24 on FLORES+, -0.66 on NTREX): the gain follows the draft quality, not the register of the training data.

**Resource tier and language group** (FLORES+ clean; tier = tertile of NLLB-3.3B chrF++, `eval_385m_results`; mean of the per-language deltas):

| stratum | languages | mean delta chrF++ (per-language deltas, equal weight) | better | worse |
|---|---:|---:|---:|---:|
| NLLB-3.3B resource tier: high | 14 | -0.90 | 2 | 7 |
| NLLB-3.3B resource tier: mid | 14 | -0.06 | 2 | 4 |
| NLLB-3.3B resource tier: tail | 15 | -0.94 | 1 | 6 |
| no NLLB reference (tier unknown) | 11 | +2.46 | 7 | 0 |
| group: Berber / Arabic | 1 | -0.12 | 0 | 0 |
| group: Creoles | 2 | +0.50 | 1 | 0 |
| group: Cushitic / Ge'ez / Ethiopic | 4 | +0.02 | 1 | 1 |
| group: Nguni | 3 | -0.71 | 0 | 2 |
| group: Nilotic | 2 | -0.34 | 1 | 1 |
| group: Other (Indo-European, Austronesian) | 2 | -0.98 | 0 | 1 |
| group: Other Bantu | 15 | -0.66 | 1 | 5 |
| group: Sotho-Tswana | 3 | -0.17 | 1 | 1 |
| group: West African | 11 | -1.21 | 0 | 6 |
| group: other | 11 | +2.46 | 7 | 0 |

Largest losses (FLORES+, chrF++): yor -8.2, lin -8.0, fon -1.9, nus -1.8, umb -1.2, mlg -1.1, tsn -1.0, hau -1.0. Largest gains: zgh +15.6, arz +2.7, aeb +2.0, ary +2.0, kea +1.4, knc +1.3, lug +1.1, luo +1.1. yor and lin lose 8 points: in yor the edited sentences are 43% longer than the reference (169 vs 118 characters) and the diacritics density rises (0.36 vs 0.25 per character in the reference): a mix of orthographies of the training sources (tone-marked FLORES / Vukuzenzele style vs JW) and run-on output; in lin the editor sometimes ignores the input and emits a short unrelated phrase. The big gains (zgh +15.6, arz, aeb, ary) are languages whose drafts are English / MSA or loops (chrF++ 2-5) where almost any in-script output beats the draft.

## 5. What the editor does

Mean over benchmark x language cells (FLORES+, NTREX, MAFAND, SmolSent greedy drafts; editor applied to every sentence):

| statistic | RL greedy drafts | after editor |
|---|---:|---:|
| share of drafts left unchanged (exact string) | - | 0.497 |
| mean word edit rate (edit distance / length) | - | 0.184 |
| edit-rate bins: 0-10% / 10-30% / 30-60% / > 60% | - | 0.145 / 0.139 / 0.082 / 0.136 |
| loops (rl.guards repetition detectors) | 0.039 | 0.048 |
| empty outputs | 0.000 | 0.000 |
| copy of the English source | 0.002 | 0.002 |
| wrong language (GlotLID top-1 not the language or a sibling) | 0.062 | 0.064 |
| finished (stopped on its own end token) | 0.987 | 0.979 |

The editor rewrites about half of all drafts, and it also CREATES loops (3.9% -> 4.8% of the sentences; 1.7% -> 5.3% on AfriScience) while repairing others. Wrong-language output does not change. With the repair-only gate: loops 0.039 -> 0.029, unchanged 0.961.

Where it helps and hurts (sentence-level chrF++ change by draft quality; E1, all four benchmarks, 34,500 sentences; `e1/eval_e1s/where.csv`):

| group | n | share changed | mean draft chrF++ | mean edited chrF++ | mean delta | better (> +0.5) | worse (< -0.5) |
|---|---:|---:|---:|---:|---:|---:|---:|
| draft chrF 0-10 | 2032 | 0.809 | 4.22 | 7.71 | 3.49 | 0.461 | 0.098 |
| draft chrF 10-20 | 4878 | 0.529 | 15.94 | 15.99 | 0.05 | 0.166 | 0.18 |
| draft chrF 20-30 | 9041 | 0.498 | 25.06 | 24.73 | -0.33 | 0.153 | 0.189 |
| draft chrF 30-40 | 8301 | 0.483 | 34.77 | 33.88 | -0.89 | 0.139 | 0.213 |
| draft chrF 40-50 | 5360 | 0.468 | 44.63 | 43.43 | -1.2 | 0.126 | 0.228 |
| draft chrF 50-60 | 2941 | 0.447 | 54.28 | 52.5 | -1.78 | 0.11 | 0.25 |
| draft chrF 60-80 | 1747 | 0.409 | 67.05 | 65.67 | -1.38 | 0.113 | 0.232 |
| draft chrF 80-100 | 200 | 0.37 | 87.51 | 86.55 | -0.96 | 0.15 | 0.195 |
| draft is a loop | 1338 | 0.993 | 6.16 | 10.19 | 4.04 | 0.622 | 0.16 |
| draft unfinished | 449 | 1.0 | 2.68 | 9.33 | 6.65 | 0.849 | 0.031 |
| source <= 10 words | 4323 | 0.507 | 31.77 | 31.63 | -0.14 | 0.213 | 0.184 |
| source 11-25 words | 23579 | 0.491 | 32.43 | 31.9 | -0.53 | 0.155 | 0.2 |
| source > 25 words | 6598 | 0.544 | 35.33 | 34.68 | -0.64 | 0.141 | 0.218 |
| all | 34500 | 0.503 | 32.9 | 32.4 | -0.5 | 0.16 | 0.202 |

Wins / losses: sentence level (chrF++) edited better 7,803, worse 9,165, tie 17,532. Language level, FLORES+ clean: 12 better / 25 not different / 17 worse; MAFAND 1 / 7 / 7; NTREX 1 / 11 / 12; SmolSent 5 / 29 / 11.

Copy control: the planned false-edit measurement on draft == reference rows (`--copy-control`) was part of the run that crashed in the LID step (the scoring pass was re-done on the saved outputs without the control); its substitute is the validation loss on identity rows below (0.17 nats per token: the editor does copy exact drafts, the failure is on non-identical drafts of decent quality). The copy-control run is the first thing to add if the editor is revisited.

## 6. Why the editor's validation loss is lower than the base model's (owner question)

The two numbers are not comparable. The editor's `val_loss` (1.54 nats per token at the end of training, macro 1.56) is the loss of the HUMAN reference **given a draft** in the prompt; the base translator's loss is the loss of the reference given only the English. A draft that shares most words with the answer makes every reference token far more predictable (5-7% of the rows have draft == reference, 2-3% near-copies, and the corruption drafts are damaged references). Like-for-like numbers on the SAME 2,636 held-out editor validation rows (token-weighted mean negative log-likelihood of `tag + reference + <end_of_turn>`, nats per token; `e1/val_loss_e1.json`):

| model / prompt | NLL | what it measures |
|---|---:|---|
| base (RL stage A), translation prompt, English only | **2.0321** (macro 2.0144) | the task the base was trained on, on these references |
| base, editor prompt with draft (never trained for it) | 2.0312 | prompt + draft alone give nothing to an untrained model |
| editor, translation prompt, English only | 1.834 (macro 1.8413) | the editor as a plain translator: **-0.20 vs the base**: fine-tuning on these human sources (their register, orthography, JW / gazette style) alone already lowers the loss |
| editor, editor prompt with an EMPTY draft | 1.8186 | same |
| editor, editor prompt with the matched draft = the training log's `val_loss` | **1.5367** (macro 1.5626) | a further -0.28 from the information in the draft |

So of the 0.50 nats gap (2.03 -> 1.54): about 0.20 is domain / orthography adaptation to the validation sources (the base model was last trained on RL data, not on these reference styles) and about 0.28 is the draft. By draft kind the editor's loss is 1.6285 on matched drafts, 0.3554 on near-references and 0.1709 on identity rows (the base: 2.0384 / 2.1386 / 1.8505); by draft chrF++ bin (0-20 / 20-40 / 40-60 / 60+) the editor reaches 2.322 / 1.8291 / 1.141 / 0.4512. A lower validation loss therefore says nothing about translating better: the benchmark scores above are the test.

## 7. Examples (FLORES+ devtest, E1 on RL greedy drafts)

* **loop repaired** (ibo; chrF++ draft 10.7 -> edited 29.7)  
  EN: Lion prides act much like packs of wolves or dogs, animals surprisingly similar to lions (but not other big cats) in behavior, and also very deadly to their prey.  
  REF: Ezinụlọ Ọdụm na-akpa agwa ka ezinụlọ oke nkịta ọhịa maọbụ nkịta ụlọ, anụ ndị nwere oyiyi dị ịtụ n'anya nye Ọdụm (mana odịghị otu a n'ebe nnukwu nwamba ndị ọzọ nọ) n'omume, ma bụrụ ihe oke iyi egwu nye ihe ọbụla ha na-achụ na nta.  
  DRAFT: Lion na-eme ihe dị ka akpa wolf ma ọ bụ nkịta, anụmanụ yiri ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị ọdịdị  
  EDITED: Ọdụ nnụnụ na-eme ihe dịka akpa ọdụ ma ọ bụ nkịta, anụmanụ yiri ọdụ (ma ọ bụghị nnukwu nwamba ndị ọzọ) n'omume ha, ma na-egbu ihe oriri ha nke ukwuu.
* **good draft damaged** (yor; chrF++ draft 53.9 -> edited 8.4)  
  EN: Lakkha Singh presented the chhappan bhog bhajan as well. Singer, Raju Khandelwal was accompanying him.  
  REF: Lakkha Singh ti se ìfàkalẹ̀ chhappan bhog bhajan pẹ̀lú. Olórin Raju Khandelwal ló ń kọ́ọ̀wọ́rìn pẹ̀lú rẹ̀.  
  DRAFT: Lakkha Singh tún ṣafihan chhappan bhog bhajan gẹ́gẹ́ bíi tirẹ̀. Olórin, Raju Khandelwal ló ń bá a lọ.  
  EDITED: Lakkha Singh ṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe àwọn àwòṣe a
* **good draft damaged** (lin; chrF++ draft 47.7 -> edited 2.8)  
  EN: Posted speed limits are noticeably lower than in previous and subsequent sections — commonly 35-40 mph (56-64 km/h) — and strict obedience to them is even more important than otherwise.  
  REF: Bandelo ya vitese oyo batiaka eza na se mingi koleka na biteni ya liboso pe oyo elandi — mbala mingi ezalaka 35-40 mph (56-64 km/h) — mpe kotosa yango na makambo nyonso eza ntina mingi koleka eloko mosusu.  
  DRAFT: Bipimo ya mbangu ebimaka mingi koleka oyo ezalaki na biteni ya kala mpe oyo ekolanda yango — mingimingi 35-40 mph (56-64 km/h) — mpe botosi makasi na yango ezali mpenza na ntina koleka.  
  EDITED: Mabóngoli lokótá makɛ́ngɛlámí
* **edit that helped** (sna; chrF++ draft 33.1 -> edited 49.1)  
  EN: The work done was mostly theoretical, but the program was written to simulate observations made of the Sagittarius galaxy.  
  REF: Basa rakaitwa raiva rekungoverenga, asi chirongwa chakanyorwa kuti chitevedzere zvakaonekwa munyeredzi dzekuSagittarius.  
  DRAFT: Basa rakaumbwa rakanyanya kureva, asi purogiramu yacho yakanyorerwa kuedza kuona zvakaitwa ne Sagittarius galaxy.  
  EDITED: Basa rakanyanya kuitwa rakanyanya kuve rinozivikanwa, asi chirongwa chakanyorwa kuti chive chitarisiko chekuona kwakaitwa kune Sagittarius galaxy.
* **edit that helped** (kin; chrF++ draft 21.5 -> edited 57.6)  
  EN: Everyone can achieve enlightenment. The only thing standing in the way of this goal is our own tension and negativity.  
  REF: Umuntu wese arashobora kumurikirwa. Ikintu cyonyine kitambika mu nzira y’iyi ntego ni imbaraga zacu n’ibitekerezo bibi.  
  DRAFT: Buri wese ashobora kugera ku mibereho ye. Icyo gihugu cyacu gishobora gukomeza gukora ni uburyo bwacu bwihariye bwo gukomeza gukora neza.  
  EDITED: Umuntu wese ashobora kugera ku mpererekano. Ikintu cyonyine cyariho mu nzira y'iyi ntego ni imbaraga zacu n'uburakari bwawe.
* **draft left alone** (swh; chrF++ draft 61.0 -> edited 61.0)  
  EN: If you want some small though rich pastries, try what depending on region are called Berliner, Pfannkuchen or Krapfen.  
  REF: Ikiwa unataka vitumbua vidogo ingawa bora, jaribu kile ambacho kutegemea eneo kinaitwa Berliner, Pfannkuchen au Krapfen.  
  DRAFT: Ikiwa unataka baadhi ya vitunguu vidogo ingawa tajiri, jaribu kulingana na eneo huitwa Berliner, Pfannkuchen au Krapfen.  
  EDITED: Ikiwa unataka baadhi ya vitunguu vidogo ingawa tajiri, jaribu kulingana na eneo huitwa Berliner, Pfannkuchen au Krapfen.
* **draft left alone** (zul; chrF++ draft 67.5 -> edited 67.5)  
  EN: I don't know if you realize it or not, but most of the goods from Central America came into this country duty-free.  
  REF: Angazi ukuthi uyakuphawula noma cha, kodwa iningi lezimpahla ezivela eMelika ePhakathi zeza kulelizwe ngaphandle kokukhokhiswa intela.  
  DRAFT: Angazi ukuthi uyayiqaphela noma cha, kodwa iningi lezimpahla ezivela eMelika Ephakathi zangena kuleli zwe ngaphandle kwemisebenzi.  
  EDITED: Angazi ukuthi uyayiqaphela noma cha, kodwa iningi lezimpahla ezivela eMelika Ephakathi zangena kuleli zwe ngaphandle kwemisebenzi.

## 8. Timeline, GPU hours, cost

Times UTC 2026-11-03. Instance: RTX 3090 Ti 24 GB, 36 CPU cores, 512 GB disk (Vast; the hourly price is not visible from the box, so no dollar figure: GPU hours below x the owner's rate).

| time | what | GPU |
|---|---|---|
| 14:34 | instance up; secrets installed; three environments (vLLM 0.32, torch/transformers 4.57, AfriCOMET with transformers 4.49) built in parallel | idle |
| 14:41-15:07 | model downloads (RL stage A, SFT-final, A0, NLLB, MADLAD, TranslatePsy, AfriCOMET, SSA-COMET, GlotLID), benchmark text + protected index from the bucket, pair selection in 8 CPU slices (12 states, 6 + 4 min) and merge | idle / drafts |
| 15:07-15:15 | RL stage-A greedy drafts of the benchmarks (34,500 sentences) | vLLM, 5 min |
| 15:15-15:46 | vLLM beam search attempted (too slow, killed) and replaced by the HF batched beam | lost 30 min |
| 15:51-16:28 | beam-4 and MBR-16 benchmark drafts (7,800 sentences each) | 18 + 19 min |
| 16:28-17:40 | training-pair drafts: RL x3 systems, SFT x2, A0, MBR (4 languages), beam (60 per language) | 72 min |
| 17:40-18:24 | E1 build + training (first attempt OOM 17:44, restart at micro batch 2; 40 min of training) | 44 min |
| 18:24-18:31 | E1 editor on 34,500 RL greedy drafts (vLLM, 5 min) | 7 min |
| 18:31-19:27 | selection v3 / v4 work, QE scoring of the v4 targets (1.5 min), AfriScience drafts, COMET of E1 (20 min), partial MT drafts (aborted by OOM when COMET shared the GPU) | mixed |
| 19:13-19:30 | beam-4 and NLLB drafts through E1, MADLAD drafts, like-for-like validation loss, COMET of those | ~20 min |
| 19:30-20:20 | MADLAD-3B drafts (13 min after MADLAD-7B was abandoned), E1 on them, AfriCOMET of beam / NLLB / MADLAD / AfriScience, SSA-COMET; training held by the owner | ~1 h |

About 6 hours of wall time since the instance came up; GPU busy roughly 4.5 hours (drafts 2.1 h, training 0.7 h, editor evaluation 0.3 h, AfriCOMET / SSA 0.8 h, MADLAD / NLLB drafts 0.5 h); the rest was selection (CPU), my debugging and waiting. The planned E2 / E2q / E3 trainings (about 35 min of training each at 1 epoch) were not run (owner hold).

## 9. Caveats

* One training seed, one editor (E1). E1 differs from the final plan: selection v2 (no AfriScience / SALT / Infopankki / Chichewa / Efik NT, 26% religious), 2 epochs (the validation loss stopped improving around epoch 1.3: macro 1.5707 @ step 120, 1.5664 @ 270), drafts from our models only (no external MT drafts), no QE filter. The plan's later selections (v3: + SALT train, Infopankki, Chichewa, Efik NT sheet; v4: AfriScience cap 3,000 with 40% science) are built and CPU-tested but were not trained on.
* Benchmarks are subsampled (250 sentences per language and benchmark; beam-4 / MBR-16 / NLLB / MADLAD 100): the CIs cover test-sentence variance only.
* AfriCOMET macros cover the 42 validated languages only (counts in the tables); the QE and reference-based models disagree little here. SSA-COMET is reported separately.
* The editor was evaluated greedy; beam-4 decoding of the editor itself was not run.

## 10. Files

Code: `gemma_mt/gemma_mt/post_edit.py`, `gemma_mt/scripts/post_editor_{select_data,extra_sources,make_drafts,mt_drafts,bench_drafts,build_sft,qe_filter,eval,comet,analyze,val_loss}.py`, `gemma_mt/configs/post_editor/pe_sft.yaml`, tests `gemma_mt/tests/test_post_edit.py` (22 pass; run on the box in venv_train). Data (`reports/data/post_editor_2026-11-03/`): `selection/` (v4: coverage_by_language.csv, availability tables), `selection_v2/`, `selection_v3/`, `drafts/` (draft quality per system and language, matching report), `e1/eval_e1s/` (greedy drafts: metrics_by_language.csv = per-language chrF++ / spBLEU of draft and edited, bootstrap_by_language.csv = per-language deltas with CIs, bootstrap_macro.csv, diagnostics.csv, wins_losses.csv, where.csv, report.md), `e1/eval_e1_gated/` (repair-only), `e1/eval_e1b/{beam4,nllb3.3b_greedy,madlad3b_greedy_x}/`, `e1/eval_e1_afsci/`, `e1/comet_e1/` + `comet_e1c/` + `comet_afsci/` + `comet_ssa/` (comet_by_language.csv = per-language AfriCOMET / QE / SSA, comet_macro.csv, comet_pairs.csv = paired bootstrap, comet_pairs_by_language.csv), `e1/val_loss_e1.json`, `e1/examples_e1.json`. On the box (`/workspace/pe`, instance NOT destroyed): editor weights `run_e1/out/final`, all drafts, edited predictions, logs `/workspace/logs/`.

Per-language AfriCOMET and chrF++ for every benchmark and draft source are in the CSVs above (one row per language); the tables in this report are macros and strata.
