# Full results comparison for the 385M translation paper: our models against other systems (2026-11-06)

CPU only, on the laptop, from data that already existed. No GPU job, no box access, nothing pushed, no paper file edited. Everything below is reproduced by `reports/data/full_results_comparison_2026-11-06/` (scripts `s01`..`s03`, section 9). Every number in this file is a row of a CSV in that folder; `tables_generated.md` there holds all 14 tables with intervals. LaTeX snippets and the figure: `paper_385m/latex_snippets_full_results/` (snippets were syntax-checked by brace/environment/column counts only; no TeX engine was run, `pdflatex` is not installed).

## 0. What is new here, in five lines

1. **Sentence-level NLLB-600M / NLLB-3.3B / TranslatePsy-0.8B hypotheses and per-sentence AfriCOMET are archived** (`research_archive_2026-10-23/box/jobs/e2/scores/*.pkl*`, `.../r14/out/*.jsonl.gz`). `reports/nllb_comparison_rl_models_2026-11-04.md` section 8 and ledger N2/N2b say they are not stored and a sentence-paired bootstrap is impossible; that is wrong. The paired bootstrap of the 385M models against NLLB on FLORES+ is done here (section 3).
2. **On identical FLORES+ sentences the final 385M RL model is 1.05 chrF++ below NLLB-600M and 3.21 below NLLB-3.3B** (paired 95% intervals exclude 0), and 0.038 / 0.054 below on held-out AfriCOMET. It beats NLLB-3.3B in 7 of 44 languages, NLLB-600M in 15.
3. **NTREX and MAFAND: 4.3 and 4.7 chrF++ below NLLB-3.3B** (language-level intervals only: NLLB output there exists only as corpus-level aggregates). No AfriCOMET exists for NLLB on those benchmarks.
4. **TranslatePsy-0.8B** (distilled from NLLB-3.3B) is the best system on its 19 languages: 45.3 chrF++ / 0.717 AfriCOMET against 41.0 / 0.642 for our RL (paired difference -4.23 [-4.39, -4.07]). The SFT-only control (E-03) equals RL on chrF++ there (+0.08 [-0.13, +0.30]) and beats it on AfriCOMET (+0.023).
5. **Stage A -> final adds little** (+0.72 chrF++ on the headline rows); the final RL model is +6.32 above the SFT-only control and +9.30 above the SFT on the same rows.

## 1. Inventory: which system x benchmark x metric exists

Full machine-readable list: `inventory.csv` (every row: system, benchmark, metric, status, level, row set, languages, source) and `missing.csv`. Summary (S = per-sentence output archived, so paired bootstrap and any rescoring possible; A = corpus-level per-language aggregate only; P = published number only; "--" = does not exist):

| system | FLORES+ | NTREX-128 | MAFAND-MT | SmolSent | SALT | GlotStory, IbomMT, FLORES-MOZ | AfriScience-MT |
|---|---|---|---|---|---|---|---|
| 385M SFT | S, AfriCOMET (40 lang.) | S, AfriCOMET | S, AfriCOMET | S, AfriCOMET (33) | S, AfriCOMET (3) | S (AfriCOMET: Ibom only) | S, chrF/chrF++/AfriCOMET/SSA-QE |
| 385M SFT-only control (E-03) | S | S | S | -- | -- | -- | S |
| 385M RL stage A, B1 best, final (mixed best) | S | S | S | -- | -- | -- | S |
| 270M SFT, RL-final (E-13 rerun) | S | S | S | -- | -- | -- | stored values (earlier code state) |
| NLLB-200-distilled-600M | S (44 lang.), AfriCOMET (42 lang., no_pair rows), SSA | -- | -- | -- | -- | -- | P (fine-tuned) |
| NLLB-200-3.3B | S, AfriCOMET (42, no_pair rows), SSA | A (21 lang.) | A (14) | A (36) | A (3) | -- | -- |
| NLLB-1.3B, distilled-1.3B, MoE-54.5B | P (43 lang., FLORES-200 devtest) | -- | -- | -- | -- | -- | P (1.3B) |
| TranslatePsy-AfriSLM-0.8B | S (19 lang.), AfriCOMET, SSA | -- | -- | -- | -- | -- | -- |
| TranslatePsy-AfriSLM-2B | S (9 of 19 lang., no_pair rows), AfriCOMET, SSA | -- | -- | -- | -- | -- | -- |
| TranslatePsy-AfriSLM-4B | -- (published: bidirectional aggregates, not comparable) | -- | -- | -- | -- | -- | -- |
| AfriqueLLM family, base LLMs | P (21 lang., 5-shot, chrF++; SSA-COMET in the paper, checkpoint unnamed) | -- | -- | -- | -- | -- | -- |
| AfriNLLB-600M | P (AfriCOMET-mtl v1.0 on 9 languages: **not comparable**, not used) | -- | -- | -- | -- | -- | -- |

Metrics per cell: chrF++ (sacreBLEU word order 2), plain chrF, spBLEU (flores200) wherever S or A; AfriCOMET only where S has a stored score; SSA-COMET-MTL (reference-based) only on FLORES+ for the NLLB/TranslatePsy systems and the 270M-era runs, never for the 385M models.

## 2. Row sets and scoring stack (the comparability rule)

* **Scoring stack.** chrF++ = sacreBLEU 2.6.0 `CHRF(word_order=2)` on NFC text, recomputed by me from stored hypotheses for all 385M and 270M systems (`s01_rescore.py`; the n-gram statistics equal the stored evidence statistics on 24,000+ sampled rows, 0 mismatches, `checks.json`), and from stored per-sentence statistics for NLLB/TranslatePsy. Corpus statistics per language, then macro over languages. spBLEU: sacreBLEU `BLEU(tokenize="flores200")`. AfriCOMET: `masakhane/africomet-stl-1.1`, reference-based, **no other version used** (AfriNLLB's mtl-v1.0 numbers are excluded). The two AfriCOMET scoring runs (270M-era run that scored NLLB/TranslatePsy; 385M-harness run that scored our models) agree to a mean absolute difference of 2.3e-4 on 15,400 identical (source, hypothesis, reference) triples (Pearson 0.99999, `checks_comet_scorer.json`), so they are on one scale. The per-language chrF++ and spBLEU of the archived NLLB aggregates equal my per-sentence recomputation to 5e-5 (132 cells, same stack).
* **Sentence identity.** FLORES+ sources and references of the 385M predictions and of the NLLB/TranslatePsy files are identical for all 44,528 rows (`alignment_ours_vs_nllb_flores`, 0 mismatches). NTREX/MAFAND/SmolSent/SALT row sets reconstructed from the contamination flag files reproduce the row counts of the NLLB aggregates in every cell except one (SmolSent afr: 862 vs 863 because of a duplicated key; afr dropped from the SmolSent group).
* **Row sets.** `all` = every row (for the 385M models FLORES+ has 0 flagged rows; NTREX 120 and MAFAND 94 flagged rows, which change the macro by at most 0.006 chrF++, `checks_clean385.json`). `np270` = rows whose reference pair the earlier 270M SFT never saw (FLORES+: 42,680 of 44,528 rows; this is the set on which NLLB/TranslatePsy AfriCOMET exists). `clean270` = rows whose English side is not in the 270M SFT text (FLORES+ 48 per language; NTREX 1,201; MAFAND about 1,000): the only rows on which the earlier 270M models are not exposure-inflated. For the NLLB aggregates the 270M flags are used; our systems are scored on exactly those rows (adding the 385M flags would change the clean270 macro by up to 0.09 chrF++, MAFAND).
* **Language sets.** Macro over the languages common to all systems of a table; counts printed as (chrF++ languages / AfriCOMET languages). FLORES+ NLLB set: 44 languages (the published NLLB table has 43; the 44th is knc), AfriCOMET covers 40 of them (our harness has no AfriCOMET score for dyu, sag, kea, knc; the 270M-era NLLB run scored dyu and sag as well, they are dropped for identity). TranslatePsy-0.8B: 19 languages, TranslatePsy-2B: 9.
* **Bootstrap.** Paired, Poisson(1) weights per source sentence shared by all systems and, for FLORES+ and NTREX, across languages; 1,000 resamples; seed 20261104; percentile 95% intervals of the language macro. Intervals cover test-sentence sampling only (one seed per model; NLLB/TranslatePsy single runs). Where NLLB exists only as a per-language aggregate (NTREX, MAFAND) the interval is a bootstrap over languages (10,000 resamples) and is labelled so.
* **Decoding.** Ours greedy (768 tokens, vLLM bf16); NLLB beam 4 (CT2); TranslatePsy greedy 256 tokens with its model-card prompt. E-02 measured RL beam-4 on 20 languages x 200 sentences: -0.77 chrF++ against greedy (MBR-16 +0.92, metric-matched), so our greedy is not handicapped much against NLLB's beam, but no beam row for the full set exists.

## 3. Headline table (FLORES+ devtest, `np270` rows, 44 languages, 42,680 sentences; AfriCOMET on 40 languages)

All cells on the same sentences. 95% intervals are paired sentence bootstraps. "RL - system" = final RL model minus the row system.

| system | chrF++ [95% CI] | AfriCOMET-STL-1.1 [95% CI] | RL - system, chrF++ [CI] | RL - system, AfriCOMET [CI] | languages RL above (of 44) |
|---|---:|---:|---|---|---:|
| 385M SFT | 25.59 [25.21, 25.98] | 0.5502 [0.5459, 0.5548] | +9.30 [+9.00, +9.55] | +0.0732 [+0.0711, +0.0752] | 44 |
| 385M SFT-only control (E-03; trained on NLLB-3.3B translations for 19 languages, never released) | 28.56 [28.23, 28.89] | 0.5858 [0.5815, 0.5901] | +6.32 [+6.11, +6.52] | +0.0377 [+0.0360, +0.0394] | 34 |
| 385M RL stage A | 34.16 [33.87, 34.46] | 0.6196 [0.6156, 0.6236] | +0.72 [+0.65, +0.81] | +0.0039 [+0.0031, +0.0046] | 38 |
| 385M RL B1 best | 34.54 [34.25, 34.84] | 0.6211 [0.6173, 0.6251] | +0.34 [+0.29, +0.41] | +0.0024 [+0.0017, +0.0030] | 34 |
| **385M RL (final = mixed best = Hub main)** | **34.88 [34.60, 35.18]** | **0.6234 [0.6195, 0.6273]** | | | |
| NLLB-200-distilled-600M | 35.94 [35.59, 36.27] | 0.6615 [0.6575, 0.6653] | -1.05 [-1.24, -0.86] | -0.0381 [-0.0399, -0.0362] | 15 |
| NLLB-200-3.3B | 38.09 [37.76, 38.40] | 0.6770 [0.6734, 0.6806] | -3.21 [-3.38, -3.02] | -0.0535 [-0.0553, -0.0518] | 7 |

Plain chrF and spBLEU for the same rows: RL 37.95 / 13.20, NLLB-600M 39.04 / 15.06, NLLB-3.3B 41.35 / 17.01 (the gap in spBLEU, 1.9 and 3.8, is larger than in chrF++). The gap to NLLB-3.3B is the same as the published-table comparison (section 6: -3.26 against the published 38.80).

Same system set on the 19 TranslatePsy-0.8B languages (`np270` rows, 17,380 sentences; all metrics):

| system | chrF++ | chrF | spBLEU | AfriCOMET | RL - system, chrF++ [CI] | RL - system, AfriCOMET [CI] |
|---|---:|---:|---:|---:|---|---|
| 385M SFT | 34.00 | 36.83 | 12.53 | 0.5886 | +7.04 [+6.64, +7.41] | +0.0536 [+0.0513, +0.0559] |
| 385M SFT-only control (E-03) | 40.96 | 44.44 | 17.66 | 0.6648 | +0.08 [-0.13, +0.30] | -0.0226 [-0.0242, -0.0208] |
| 385M RL (final) | 41.04 | 44.50 | 18.06 | 0.6422 | | |
| NLLB-200-distilled-600M | 42.58 | 46.14 | 20.35 | 0.6878 | -1.54 [-1.78, -1.28] | -0.0456 [-0.0480, -0.0434] |
| NLLB-200-3.3B | 44.66 | 48.34 | 22.43 | 0.7007 | -3.62 [-3.84, -3.39] | -0.0586 [-0.0609, -0.0563] |
| TranslatePsy-AfriSLM-0.8B | 45.27 | 48.94 | 22.39 | 0.7167 | -4.23 [-4.39, -4.07] | -0.0745 [-0.0765, -0.0727] |

On the 9 languages of TranslatePsy-2B (afr amh hau ibo kin lin lug mlg nya; 7,705 sentences): RL 41.56 / 0.631, NLLB-3.3B 46.29 / 0.701, TranslatePsy-0.8B 46.61 / 0.715, TranslatePsy-2B 46.44 / 0.716 (T2b in `tables_generated.md`).

### NTREX-128 and MAFAND-MT (all rows; NLLB-3.3B = per-language corpus-level aggregate on the same rows, no AfriCOMET exists)

| benchmark (languages) | system | chrF++ [95% CI] | AfriCOMET [95% CI] | RL - system, chrF++ |
|---|---|---:|---:|---|
| NTREX (21) | 385M SFT | 31.03 [30.69, 31.42] | 0.5508 | +7.97 [+7.73, +8.21] |
| | SFT-only control (E-03) | 35.86 [35.54, 36.18] | 0.6088 | +3.14 [+2.97, +3.32] |
| | 385M RL stage A / B1 best | 38.59 / 38.84 | 0.6125 / 0.6134 | +0.41 / +0.16 |
| | **385M RL (final)** | **39.00 [38.73, 39.25]** | **0.6138 [0.6099, 0.6177]** | |
| | NLLB-3.3B (aggregate) | 43.26 | -- | -4.26 [-5.22, -3.24] (language bootstrap; RL above in 1 of 21) |
| MAFAND (14) | 385M SFT | 34.50 [34.23, 34.76] | 0.6143 | +5.39 [+5.15, +5.63] |
| | SFT-only control (E-03) | 39.55 [39.31, 39.75] | 0.6696 | +0.34 [+0.16, +0.52] (AfriCOMET -0.0158: control ahead) |
| | 385M RL stage A / B1 best | 39.93 / 39.93 | 0.6563 / 0.6550 | -0.05 / -0.04 (not different) |
| | **385M RL (final)** | **39.88 [39.71, 40.03]** | **0.6537 [0.6521, 0.6555]** | |
| | NLLB-3.3B (aggregate) | 44.61 | -- | -4.73 [-6.56, -3.12] (language bootstrap; RL above in 1 of 14) |

spBLEU: NTREX RL 17.42 vs NLLB-3.3B 21.90; MAFAND 18.09 vs 23.89 (`macro_table.csv`).

### With the earlier 270M models (rows clean for them too: FLORES+ 48 sentences per language, NTREX 25,221, MAFAND 14,276 sentences)

| benchmark (languages) | 270M SFT | 270M RL-final | 385M SFT | 385M RL | NLLB-600M | NLLB-3.3B |
|---|---|---|---|---|---|---|
| FLORES+ chrF++ (44) | 24.94 | 31.80 | 22.54 | 32.14 | 32.53 | 34.99 |
| FLORES+ AfriCOMET (40) | 0.4798 | 0.5488 | 0.4685 | 0.5437 | 0.5790 | 0.6013 |
| NTREX chrF++ (21) / AfriCOMET (21) | 33.18 / 0.5458 | 37.35 / 0.5977 | 29.03 / 0.5209 | 37.31 / 0.5848 | -- | 41.47 / -- |
| MAFAND chrF++ (14) / AfriCOMET (14) | 36.64 / 0.6179 | 40.52 / 0.6603 | 33.44 / 0.6034 | 39.37 / 0.6417 | -- | 43.99 / -- |

385M RL minus 270M RL-final: FLORES+ +0.35 [+0.08, +0.63] chrF++, -0.0052 [-0.0102, -0.0003] AfriCOMET; NTREX -0.04 [-0.13, +0.04], -0.0129; MAFAND -1.14 [-1.31, -0.98], -0.0186. The growth to 385M gave no clear gain over the earlier model after RL (consistent with E-13; its old-RL values 31.68 / 35.52 / 41.80 are on 45 / 24 / 15 languages and the union of both flag sets, mine on the 44 / 21 / 14 languages that NLLB covers, so the numbers are not interchangeable).

### Other benchmarks

| benchmark (languages) | system | chrF++ | AfriCOMET | note |
|---|---|---:|---:|---|
| SmolSent, all rows (35; AfriCOMET 33) | 385M SFT / NLLB-3.3B (aggregate) | 21.68 [21.47, 21.91] / 36.78 | 0.5254 / -- | **no RL model was evaluated on SmolSent**; SFT vs NLLB is not an RL comparison |
| SALT, all rows (3) | 385M SFT / NLLB-3.3B | 39.49 / 51.81 | 0.6828 / -- | no RL model |
| GlotStoryBook (8), IbomMT (2), FLORES-MOZ (1) | 385M SFT only | 20.22 / 12.04 / 21.54 | -- / 0.5193 (1 lang.) / -- | no comparator exists; no RL model evaluated |
| AfriScience-MT test, 6 languages, plain chrF | 385M SFT 36.00; stage A 45.06; B1 best 45.30; **RL final 45.40**; control 44.71; earlier 270M RL-final 47.9 (stored); published fine-tuned NLLB-1.3B 52.3, NLLB-600M 51.7, NLLB-1.3B zero-shot 44.1, GPT-5.4 10-shot 53.3 | AfriCOMET-STL: 0.6064 / 0.6822 / 0.6813 / 0.6828 / 0.6947 | | published numbers use the benchmark paper's scoring; ours is sacreBLEU chrF word order 0; published Yoruba values are anomalously low (ledger N6). SSA-COMET-QE (reward family): 0.569 / 0.612 / 0.611 / 0.613 / 0.621 (`afriscience_block.csv`) |

### Block (a): published whole-devtest NLLB numbers (FLORES-200 devtest chrF++ from the NLLB release; our systems rescored on FLORES+ all 1,012 rows; 43 shared languages; NOT sentence-paired)

| system | chrF++ macro | RL - system [language-bootstrap CI] | languages RL above |
|---|---:|---|---:|
| 385M RL (final) | 35.54 | | |
| 385M SFT / SFT-only control / stage A / B1 best | 26.30 / 29.35 / 34.91 / 35.25 | +9.24 / +6.19 / +0.63 / +0.29 | |
| NLLB-600M, our rerun rescored | 36.62 | -1.08 [-1.98, -0.20] | 15 |
| NLLB-3.3B, our rerun rescored | 38.63 | -3.09 [-4.13, -2.05] | 7 |
| NLLB-3.3B published | 38.80 | -3.26 [-4.30, -2.21] | 7 |
| NLLB-1.3B published | 38.09 | -2.55 [-3.46, -1.62] | 7 |
| NLLB-distilled-1.3B published | 38.34 | -2.80 [-3.72, -1.87] | 7 |
| NLLB-MoE-54.5B published | 39.08 | -3.54 [-4.59, -2.52] | 6 |

This reproduces NLLBRL (35.54, -3.26) exactly. Our NLLB-3.3B rerun scores 0.17 below the published value (the E2 report's -0.17). The AfriqueLLM 5-shot comparison (21 languages, mixed best 39.40 vs 41.71 / 38.58 / 36.71 / 36.67 / 31.12) stays as in ledger N3b: published, indicative only, not recomputed here.

### SSA-COMET-MTL, reference-based, x100 (REWARD FAMILY, NOT INDEPENDENT; FLORES+ `np270` rows; no value exists for any 385M model)

| system | languages | macro, own languages | macro, 19 TPsy-0.8B languages | macro, 9 TPsy-2B languages |
|---|---:|---:|---:|---:|
| NLLB-600M | 22 | 56.71 | 56.72 | 56.91 |
| NLLB-3.3B | 22 | 59.28 | 59.14 | 58.90 |
| TranslatePsy-0.8B | 19 | 60.75 | 60.75 | 60.31 |
| TranslatePsy-2B | 9 | 60.43 | | 60.43 |
| earlier 270M RL-final (first research's own run, not the E-13 rerun) | 22 | 51.92 | 52.81 | 50.29 |
| earlier 270M SFT-final (same) | 22 | 47.54 | 48.78 | 46.19 |

## 4. Caveats that belong next to every use of these tables

1. **Greedy vs beam, one seed, one run.** Intervals are sentence-sampling only. NLLB beam search, TranslatePsy and NLLB single runs.
2. **NTREX/MAFAND/SmolSent/SALT NLLB numbers are aggregates** from the 270M-era R14 run (same rows, same stack, verified on FLORES+); no paired interval, no AfriCOMET, no plain chrF; NLLB-600M not run there.
3. **Metric independence.** chrF++, chrF, spBLEU vs human references: independent of every reward. AfriCOMET-STL-1.1 (reference-based) is held out of every reward but is the same family as the QE reward member and was used to filter TranslatePsy; it therefore **favours the SFT-only control and TranslatePsy** (control beats RL on AfriCOMET on the TranslatePsy languages and on MAFAND while equal or lower on chrF++). The 5 NLLB-pseudo-reference languages (kea kbp nus bam kab) and the 7 human-reference-term languages (IND*) are partly optimised towards chrF++.
4. **Training data and licence.** NLLB output and TranslatePsy data are non-commercial; the NLLB-derived share of our SFT/stage-B data, the NLLB-3.3B pseudo-reference reward term on 5 languages, and the E-03 control (trained on TranslatePsy NLLB-3.3B translations for 19 languages, checkpoints never released) must be disclosed next to any table that shows these systems (`paper_385m/DISCLOSURES.md`). TranslatePsy-0.8B itself was distilled from NLLB-3.3B outputs, so its lead over NLLB-3.3B on the 19 languages is a student-teacher effect, not independent evidence of a better architecture.
5. **FLORES+ vs FLORES-200 references** differ in places (block (a) only); NLLB may have seen benchmark text that we cannot audit.
6. **Exposure.** The earlier 270M models saw about 95% of FLORES+ English; their numbers are only used on the clean270 rows (48 sentences per language on FLORES+, so that column has wider intervals, +-1.3 on the macro).
7. **The headline chrF++ macros are higher than the 54-language numbers elsewhere** (e.g. 31.66 for 54 languages): the 44-language NLLB set excludes the Arabic dialects, Tamazight, Dinka and Tamasheq.
8. **The figure's chrF++ and AfriCOMET axes do not start at zero** (20 and 0.5); state this in the caption or redraw if the editor prefers.
9. **FLORES+ clean270 / SmolSent afr**: SmolSent afr dropped (862 vs 863 rows); knc (not in the published table) is in the 44-language set.

## 5. Missing list (what would be needed to fill each gap; A100 hours are ESTIMATES, none was run)

Details in `missing.csv`.

| # | gap | why missing | run needed | A100 h |
|---|---|---|---|---:|
| 1 | NLLB-3.3B per-sentence outputs on NTREX (21 lang.), MAFAND (14) | only corpus aggregates archived (outputs lived on the wiped box) | `r14/code/nllb_run.py`, NLLB_MODEL=3b3, beam4, benches ntrex,mafand (CT2 int8). R14 logged 25,961 NTREX sentences in 1,532 s on its box | 1.2 |
| 2 | NLLB-600M on NTREX, MAFAND, SmolSent, SALT | R14 ran 600M on FLORES+ only | same script, NLLB_MODEL=600m | 0.5 |
| 3 | NLLB-3.3B per-sentence outputs on SmolSent (36), SALT (3) | aggregates only | same, benches smol_sent,salt | 0.6 |
| 4 | AfriCOMET-STL-1.1 for the outputs of 1-3 and for the 1.8k FLORES+ pair-seen rows of NLLB/TranslatePsy | NLLB/TranslatePsy scored on `np270` rows only | `e2_comet.py`-style scoring | 0.3 |
| 5 | TranslatePsy-0.8B on NTREX and MAFAND (shared languages) | only FLORES+ run | `e2_gen_chat.py tpsy08` (vLLM greedy, 256 tokens), then AfriCOMET | 0.4 |
| 6 | TranslatePsy-2B: the other 10 languages on FLORES+, NTREX, MAFAND | stopped by the 6 GB GPU cap | `e2_gen_chat.py tpsy2b` | 0.8 |
| 7 | TranslatePsy-4B, any benchmark | never run (8.6 GB) | `e2_gen_chat.py tpsy4b` + AfriCOMET | 1.5 |
| 8 | 385M RL stages, control, and the 270M models on SmolSent, SALT, GlotStoryBook, IbomMT, FLORES-MOZ | RL models evaluated on 3 benchmarks only | `eval_sft_benchmarks` with the five benchmarks (the 122,585-row 3-benchmark run took 456 s per model) + AfriCOMET | 0.8 |
| 9 | SSA-COMET-MTL (ref.) for the 385M models on FLORES+ | E-05 scored AfriScience only | `e2_ssa.py` on 22 languages for SFT, control, RL | 0.3 |
| 10 | Beam-4 row for the RL model on full FLORES+ (decoder-matched to NLLB) | E-02 covers 20 languages x 200 sentences | E-19, vLLM beam search, 44 x 1,012 (4,000 prompts took 827 s) | 2.5 |
| 11 | AfriqueLLM / NLLB-1.3B / MoE on identical sentences with AfriCOMET | no per-sentence outputs; 4-54 GB weights | not recommended; keep published-only | n/a |

Items 1+2+3+4+5 together (about 3 h) would give a complete NLLB / TranslatePsy comparison on NTREX and MAFAND with paired intervals and AfriCOMET, and item 8 (about 0.8 h) removes the "no RL model" gap on five benchmarks.

## 6. Inconsistencies found between sources

1. **"NLLB hypotheses are not stored" (NLLBRL section 8, ledger N2 caveat and N2b caveat, EXPERIMENTS_LEFT E-06) is wrong.** Per-sentence NLLB-600M/3.3B outputs on FLORES+ (1,012 rows x 44 languages), their per-sentence chrF++/spBLEU statistics, per-sentence AfriCOMET (no_pair rows) and SSA-COMET, and TranslatePsy-0.8B/2B outputs are in `research_archive_2026-10-23/box/jobs/e2/scores/` and `r14/out/`. The sentence-paired bootstrap that NLLBRL calls impossible is in section 3.
2. **"NTREX and MAFAND have no NLLB comparison" (NLLBRL section 8, ledger N2) is outdated**: `reports/data/nllb600m_per_language_benchmark.csv` holds NLLB-3.3B corpus-level chrF++/spBLEU per language on NTREX (21), MAFAND (14), SmolSent (36), SALT (3), on the same rows.
3. **Language count.** NLLBRL and ledger say 43 languages; the per-sentence NLLB data has 44 (knc); AfriCOMET languages: E2 report 42 (includes dyu, sag), the 385M harness 40. Tables here state the counts used.
4. **E-13 old-RL FLORES+ = 31.68 (45 languages, clean for both) vs 31.80 here (44 languages, clean270 rows).** Difference is the language set (vmw is not in NLLB) and the rows removed by the 385M flags (FLORES+ none); not a discrepancy in method.
5. **N2b "our NLLB run on 100 sentences is 1.23 below the published value"** versus -0.17 on the full 1,012 rows here (and in the E2 report): the 100-sentence gap is sample noise, not a systematic effect of FLORES+ vs FLORES-200.
6. **E2 report (10-23) positions the 270M RL-final at 34.7 chrF++ on `no_pair` (43 languages) against NLLB-600M 36.4;** the 385M RL gives 34.88 on the 44-language `no_pair` set against 35.94: the 385M RL is closer to NLLB-600M (1.05) than the 270M was (1.7), but this is a different language set and a different run; the E-13 same-harness comparison above shows 385M RL and 270M RL are equal in chrF++ on clean rows.
7. **Coverage of the RL models.** E-13 caveat 3 already notes that only FLORES+, NTREX and MAFAND were rerun for the old models; the same holds for every 385M RL model, so SmolSent, SALT, GlotStoryBook, IbomMT and FLORES-MOZ have no RL row anywhere (missing item 8).

## 7. Suggested paragraph for the core results section (about 105 words, 3 numbers)

On FLORES+ sentences whose reference translations were not in our training data, the final 385M model reaches 34.9 chrF++ (language macro), about one point below NLLB-200-distilled-600M (35.9) and three below NLLB-200-3.3B (38.1); paired bootstrap intervals exclude zero in both cases, and held-out AfriCOMET gives the same ordering with a wider gap. NTREX and MAFAND show the same ordering against NLLB-3.3B. RL improves on the SFT in every benchmark and language set; continued SFT on teacher translations matches RL on chrF++ only for the languages that have such data. TranslatePsy-0.8B, distilled from NLLB-3.3B, is the strongest system on its own languages.

## 8. Files

* This report: `reports/full_results_comparison_2026-11-06.md`.
* Data and scripts: `reports/data/full_results_comparison_2026-11-06/` -- `macro_table.csv` (every group x system, intervals, differences), `per_language_long.csv`, `per_language_wide_*.csv` (per benchmark, per language, all systems, chrF++/chrF/spBLEU/AfriCOMET), `published_block*.csv`, `ssa_reward_family.csv`, `afriscience_block.csv`, `inventory.csv`, `missing.csv`, `checks*.json`, `tables_generated.md`.
* LaTeX (booktabs) and figure: `paper_385m/latex_snippets_full_results/`: `tabA_core_main_comparison.tex` (Table A, column width, `\resizebox`), `tabA2_translatepsy19.tex`, `tabA3_with_earlier_270M_clean_rows.tex`, `tabC_all_systems_macro.tex`, `tabD_paired_differences.tex`, appendix `tabB1`..`tabB10` (per language: chrF++, AfriCOMET, plain chrF, spBLEU on FLORES+; NTREX and MAFAND chrF++ and AfriCOMET; SmolSent; SALT), `fig_full_results.pdf/.png`. Requires `booktabs`, `graphicx`; captions define SFT, RL, SFT-ctl, F+, NT, MF; abbreviation footnote `a` marks the TranslatePsy-trained control.

## 9. Reproduction (laptop, CPU, ~15 minutes)

Python with numpy, matplotlib, sacrebleu 2.6.0 and sentencepiece (the session venv; the flores200 SentencePiece model was already in `~/.sacrebleu`). From `reports/data/full_results_comparison_2026-11-06/`:

```
python -I s01_rescore.py CACHE_DIR            # ~6 min: per-sentence statistics from the stored predictions of 7 systems
python -I s02_compute.py CACHE_DIR .          # all groups, bootstrap, published block, SSA
python -I s02b_comet_scorer_check.py .        # AfriCOMET scorer agreement
python -I s02c_clean385_check.py CACHE_DIR .  # 385M-flag robustness
python -I s03_build.py . ../../../paper_385m/latex_snippets_full_results
```
