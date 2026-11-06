# E-06: RL models against NLLB-200 and the other published systems (2026-11-04)

CPU only, from stored tables; nothing was run on a GPU or on the A100. Script and data: `reports/data/nllb_comparison_rl_models_2026-11-04/` (`b1_compare.py` makes all CSVs, `b2_make_md_tables.py` renders `tables.md`, `b1_compare.log` is the console output). Every number below is a row of one of those CSVs; the language count is printed next to each number. Closes ledger item N2 / pending P5 / experiment E-06 (the sentence-identical part is only partly possible, see section 4).

## 0. Result in five lines

* On the **43 FLORES+ devtest languages shared with the published NLLB-200 table**, the chrF++ macro is: SFT 26.30, RL stage A 34.91, B1 best 35.25, mixed best 35.54, weak final 35.50, against NLLB-3.3B 38.80, NLLB-1.3B 38.09, NLLB distilled-1.3B 38.34, NLLB-MoE-54.5B 39.08 (`table1_macro_43_languages.csv`).
* Mixed best is **3.26 chrF++ below NLLB-3.3B** (language-bootstrap 95% CI [-4.29, -2.20], n = 43), 2.55 below NLLB-1.3B [-3.45, -1.62], 3.54 below the MoE [-4.62, -2.52]: it reaches **92% of NLLB-3.3B** (macro ratio 0.92; median per-language ratio 0.92; SFT 68%). The RL closes **74% of the SFT-to-NLLB-3.3B gap** (12.50 -> 3.26, arithmetic of the two differences).
* The RL model **matches or beats NLLB-3.3B on 7 of 43 languages** (mixed best minus NLLB-3.3B: yor +5.43, fon +2.95, dyu +2.77, wol +2.48, kam +2.32, kmb +2.22, umb +1.86; all seven are in the lowest NLLB tertile); within 1 chrF++ on 9 (adds afr -0.57, ibo -0.96). The SFT matched or beat it on 1 (fon, 20.17 vs 19.30).
* The 3.26 gap is **not a loop/finish artefact**: on the 43 languages the mixed model has finished >= 0.99 and repeat <= 0.02 in every language (`table4_clean_language_subsets.csv`). It is a gap in translation quality as chrF++ sees it.
* Against NLLB-3.3B on **identical sentences** (post-editor run, 100 FLORES+ sentences per language, NLLB run by us on FLORES+ references): RL stage A **beam-4** is 1.44 below on 43 languages ([-2.34, -0.56], language-level bootstrap), 3.67 below on 50 languages; held-out AfriCOMET (reference-based) 0.6414 vs 0.6709, -0.0294 [-0.0366, -0.0222], 40 languages, the RL ahead in 5 of 40. Greedy stage A is 2.91 below on 43. Per-sentence hypotheses are not stored, so no sentence-paired bootstrap was possible.

## 1. What was compared

| item | value | source |
|---|---|---|
| our systems | SFT (385M, A0 up6 + 15 layers), RL stage A (step 400 final, `stage-a-final`), B1 best (step 50), mixed best (final general RL model), weak final (step 85) | `reports/data/eval_weak_group_2026-11-03/scores/metrics_by_language.csv` (FLORES+ devtest, 1,012 sentences, greedy 768 tokens, vLLM bf16, chrF++ = sacreBLEU word order 2); identical to EVID/POSTB1 to 1e-14 for SFT and mixed best (script check) |
| published | NLLB-200 3.3B, 1.3B, distilled-1.3B, MoE 54.5B; en -> xx; FLORES-200 devtest; chrF++ from the NLLB release metrics files; beam search | `reports/data/eval_385m_results_2026-11-03/published_nllb_flores_per_language.csv` (43 rows, built from `reports/data/published_baselines.csv`); the SFT column of that file equals the weak-group SFT column exactly (max difference 0.0) |
| languages | 43 = languages of our FLORES+ 54 that the published NLLB table has (the 11 others, aeb ary arz cjk dik knc mfe seh taq vmw zgh, have no published NLLB row) | same file |
| clean rows | FLORES+ has **0 flagged rows** in the 385M contamination audit (CONT; EVID: "FLORES+ 0"), so `all` = `clean` for every number here; NTREX/MAFAND have no NLLB per-language published numbers, so no clean-row variant exists for them | EVID header; CONT |

The 43 languages: afr amh bam bem dyu ewe fon hau ibo kab kam kbp kea kik kin kmb lin lua lug luo mlg mos nso nus nya orm run sag sna som sot ssw swh tir tsn tso tum twi umb wol xho yor zul (`table2_per_language_43.csv`).

## 2. Macro over the 43 shared languages (chrF++)

Differences are model minus system; the CI is a bootstrap over the 43 languages (10,000 resamples), i.e. it describes which languages were chosen, not sentence noise (per-sentence hypotheses of NLLB on all 1,012 sentences are not stored; the published numbers have no interval).

| model | ours | NLLB-3.3B | diff to NLLB-3.3B [CI] | macro ratio | median per-language ratio | languages ours >= NLLB-3.3B | within 1 | ratio >= 0.8 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| SFT | 26.30 | 38.80 | -12.50 [-14.56, -10.40] | 0.68 | 0.66 | 1 | 1 | 11 |
| RL stage A (step 400) | 34.91 | 38.80 | -3.89 [-4.95, -2.80] | 0.90 | 0.90 | 7 | 9 | 36 |
| B1 best (step 50) | 35.25 | 38.80 | -3.55 [-4.58, -2.48] | 0.91 | 0.91 | 7 | 9 | 38 |
| mixed best | 35.54 | 38.80 | -3.26 [-4.29, -2.20] | 0.92 | 0.92 | 7 | 9 | 40 |
| weak final (step 85) | 35.50 | 38.80 | -3.30 [-4.35, -2.23] | 0.91 | 0.92 | 7 | 8 | 41 |

Difference to the other NLLB variants (same 43 languages; `table1_macro_43_languages.csv`):

| model | vs NLLB-1.3B (38.09) | vs NLLB-distilled-1.3B (38.34) | vs NLLB-MoE-54.5B (39.08) |
|---|---|---|---|
| SFT | -11.79 | -12.04 | -12.78 |
| RL stage A | -3.18 | -3.43 | -4.17 |
| B1 best | -2.84 | -3.09 | -3.83 |
| mixed best | -2.55 | -2.80 | -3.54 |
| weak final | -2.59 | -2.85 | -3.58 |

The SFT row reproduces the existing ledger N1 (26.30 vs 38.80, 1 language matched). The stage chain follows the known one: stage A gives 8.61 of the 9.24 chrF++ that mixed best adds over SFT on these 43 languages (93%, arithmetic of the table).

## 3. By NLLB-3.3B tertile and per language

Tertiles of NLLB-3.3B chrF++ (14 / 14 / 15 languages; this reproduces the EVAL resource-tier numbers 38.4 / 24.9 / 16.3 for the SFT and 51.1 / 39.4 / 26.8 for NLLB) (`table3_by_nllb_tertile.csv`):

| tertile | n | NLLB-3.3B | NLLB-MoE | SFT | stage A | B1 best | mixed best | weak final | mixed best as % of NLLB-3.3B (SFT %) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| high | 14 | 51.05 | 50.59 | 38.43 | 46.49 | 46.65 | 46.82 | 46.70 | 91.7 (75.3) |
| mid | 14 | 39.42 | 39.58 | 24.93 | 33.95 | 34.33 | 34.57 | 34.51 | 87.7 (63.2) |
| tail | 15 | 26.78 | 27.87 | 16.26 | 25.00 | 25.47 | 25.92 | 25.96 | 96.8 (60.7) |
| all | 43 | 38.80 | 39.08 | 26.30 | 34.91 | 35.25 | 35.54 | 35.50 | 91.6 (67.8) |

* The RL model is relatively **closest to NLLB in the tail tertile (96.8% of NLLB-3.3B, 0.9 chrF++ below)** and furthest in the mid tertile (87.7%, 4.9 below). In absolute terms the largest gaps stay in the high tertile (4.2 below).
* The 7 languages where mixed best >= NLLB-3.3B are all tail: yor +5.43 (30.33 vs 24.90), fon +2.95, dyu +2.77, wol +2.48, kam +2.32, kmb +2.22, umb +1.86. For comparison the SFT beat NLLB-3.3B only on fon. Caution for yor: the published NLLB Yoruba number is anomalously low next to the other systems (ledger N6), so the largest "win" is the least trustworthy one.
* Largest remaining gaps of mixed best to NLLB-3.3B: amh -11.28, kin -9.74, kea -8.72 (NLLB 47.9; the SFT collapse language), run -8.02, ewe -7.03, lug -6.77, tir -6.06, hau -6.04 (`table2_per_language_43.csv`, sorted).
* All numbers per language, all five models and the four NLLB variants: `table2_per_language_43.csv` (and `tables.md`, Table 2).

## 4. Same sentences: NLLB-3.3B drafts of the post-editor run

The post-editor run (`reports/post_editor_run_2026-11-03.md` section 3) produced NLLB-3.3B and MADLAD-400-3B greedy drafts on the **same 100-sentence FLORES+ subsets** that the RL stage-A **beam-4** drafts were made on, and scored them with the same code and FLORES+ references. Stored per language: chrF++ (`reports/data/post_editor_2026-11-03/e1/eval_e1b/{nllb3.3b_greedy,beam4,madlad3b_greedy_x}/metrics_by_language.csv`, 100 sentences) and held-out AfriCOMET (`.../e1/comet_e1c/comet_by_language.csv`, 40 languages). Per-sentence hypotheses are **not** in the stored data (they are on the A100 `/workspace/pe`), so a sentence-paired bootstrap was not possible; the intervals below are over languages. `pairs.jsonl.gz` in that folder is the editor training pairs, not hypotheses.

| comparison (FLORES+, same sentences within a row) | languages | A | B | A - B [language-bootstrap CI] | A >= B in |
|---|---:|---:|---:|---|---:|
| RL stage A beam-4 vs NLLB-3.3B (our run), 100 sentences | 43 | 36.12 | 37.56 | -1.44 [-2.34, -0.56] | 13 |
| same, all languages both have | 50 | 32.65 | 36.32 | -3.67 [-5.89, -1.82] | 13 |
| RL stage A **greedy** (250-sentence subsample) vs NLLB-3.3B (100) | 43 | 34.65 | 37.56 | -2.91 [-3.96, -1.86] | 8 |
| AfriCOMET (reference, held-out checkpoint): beam-4 vs NLLB-3.3B | 40 | 0.6414 | 0.6709 | -0.0294 [-0.0366, -0.0222] | 5 |
| NLLB-3.3B our run (100 sentences) vs NLLB-3.3B published (1,012, beam) | 43 | 37.56 | 38.80 | -1.23 [-1.60, -0.86] | 7 |
| RL stage A greedy 250-subsample vs stage A greedy full 1,012 | 50 | 31.07 | 31.36 | -0.29 [-0.46, -0.12] | 14 |
| MADLAD-400-3B drafts vs RL stage A beam-4 (100) | 35 | 28.27 | 35.29 | -7.02 (arithmetic of the two macros) | |

(`table6b_same_sentence_summary.csv`, `table6c_same_sentence_43_languages.csv`, `table7_africomet_same_100_sentences.csv`, `table6_same_sentence_postedit_run.csv` for the per-language values.)

Readings:
* With **beam search on both sides** (stage A beam-4 vs NLLB greedy of our run) the gap on identical sentences shrinks to 1.44 chrF++ on the 43 shared languages; it is 1.47 [+1.17, +1.79] larger for the beam than for the 250-sentence greedy of the same model (39 of 43 languages higher), which matches the existing "beam-4 +1.6 chrF++" statement of the post-editor report. The 1.23 [-1.60, -0.86] difference between our NLLB run on 100 sentences and the published full-devtest value shows how much the 100-sentence subsample and FLORES+ versus FLORES-200 matter (7 of 43 languages higher in our run); the published-table comparison of section 2 is therefore not biased in our disfavour by more than about a point from this source.
* The 250-sentence greedy subsample is a good proxy of the full set (-0.29 [-0.46, -0.12] on 50 languages); whether the 100 beam sentences are a subset of the 250 greedy sentences is not stated in the report (both are "deterministic subsamples"), so the row "greedy 250 vs NLLB 100" is not strictly sentence-identical. Use the beam-4 rows as the identical-sentence evidence.
* On the held-out AfriCOMET the RL beam-4 drafts are behind NLLB-3.3B by 0.0294 on 40 languages; AfriCOMET has a length/entity bias (ledger S5) and the RL model optimises the same family (QE member), so this row is not independent evidence for the RL.
* MADLAD-400-3B (35 shared languages): 28.27 vs RL beam-4 35.29; the RL model, greedy or beam, is well above MADLAD-3B on the languages MADLAD supports (the post-editor report gives 28.35 on its 37 languages).

## 5. Other published systems

* **AfriqueLLM family and base LLMs** (5-shot, AfroBench-Lite harness, FLORES-200 en -> xx per language, 21 languages): the per-language values are in `reports/data/published_baselines.csv` and the 21 languages are afr amh ewe hau ibo kin lin lug nya orm sna som sot swh tir tsn twi wol xho yor zul. On exactly these 21 languages (`table5_afrique_21_languages.csv`): SFT 32.37, stage A 39.13, B1 best 39.32, **mixed best 39.40**, weak final 39.26. AfriqueQwen3.5-4B-50Langs 41.71 (mixed best -2.31; ahead in 4 of 21 languages), AfriqueGemma-12B 38.58 (+0.83; 6 of 21), AfriqueLlama-8B 36.71 (+2.69), AfriqueQwen-14B 36.67 (+2.74), AfriqueQwen-8B 35.06 (+4.35), AfriqueGemma-4B 31.12 (+8.28), Gemma3-12B 18.82, Llama3.1-8B 16.45. This extends ledger N3 (SFT 32.4) to the RL model. Caveats: those systems are base models in 5-shot prompting, scored with the AfroBench chrF++ implementation (sacreBLEU word order "not verified"), the FLORES split is not stated in the Afrique paper, and the harness decodes with 256 new tokens; indicative only (ledger N3 status `weak` stays).
* **TranslatePsy-AfriSLM (0.8B/2B/4B)**: published numbers are the bidirectional (average of en -> xx and xx -> en) 19-language aggregates (FLORES-200 4B = 51.13 etc., `published_baselines.csv` rows `MACRO19`); the per-language rows exist but hold no number (`flag` = no_number: only figures, not read). Not comparable with our en -> xx numbers (ledger N4); not used.
* **AfriNLLB-600M, Hunyuan-MT-7B, Apertus-8B, TranslateGemma, Gemini-2.5-Flash**: bidirectional aggregates, MetricX-only reports, WMT24++ pairs, or 10-sentence samples (`published_baselines.csv` `flag` column): not comparable, not used.
* **AfriScience-MT** published systems (NLLB-1.3B fine-tuned etc.) compare only with the science stage (ledger N5), not in this report.

## 6. Comparability caveats (state all of these next to any NLLB number in the paper)

1. **References.** The published NLLB numbers are FLORES-200 devtest; ours are FLORES+ devtest (same 1,012 sentences, some references corrected by the community). The effect cannot be separated from the data here; our own NLLB run on FLORES+ is 1.23 below the published value on 100 sentences (section 4), which bounds it together with the sample effect.
2. **Decoding.** NLLB numbers are beam search with the NLLB default; ours is greedy (768 new tokens). Beam-4 is worth about +1.5 chrF++ for our stage A on the same sentences (section 4) and +2.5 for the SFT (ledger D1), so a beam row for the RL model would narrow the gap; it was not run on the full set (E-02).
3. **Detokenisation and normalisation.** The published pipeline's text normalisation is not specified beyond "chrF++ computed by the NLLB evaluation pipeline"; ours is sacreBLEU chrF++ (word order 2) on the raw detokenised output.
4. **Language set.** Only the 43 languages in both tables; the other 11 FLORES+ languages of our evaluation (aeb ary arz cjk dik knc mfe seh taq vmw zgh) have no published NLLB row, and the hardest languages (Arabic dialects, Tamazight, Dinka) are not in the 43; the 43-language macro is therefore easier than our 54-language macro (mixed best 35.54 on 43 vs 31.66 on 54).
5. **Failures.** Our model finishes and does not loop on these 43 languages (finished >= 0.99, repeat <= 0.02 for every language), so unfinished or looping outputs do not explain the gap; NLLB's own failures (hallucination, under-generation) are not measured by us.
6. **Data overlap.** FLORES+ devtest is clean in our SFT (0 benchmark sentences of 3 or more words, ledger CT1); the FLORES-200 dev/devtest are in NLLB's evaluation conventions but NLLB-200 was trained on data that may overlap public benchmarks that we cannot audit; the FLORES+ dev train half is in OUR SFT on purpose (ledger CT3).
7. **Metric independence.** chrF++ against human references is independent (IND); for the 5 NLLB-pseudo-reference languages (kea kbp nus bam kab) the RL reward contained an NLLB-3.3B chrF++ term, so on those five languages the RL model was partly optimised towards the system it is being compared with: kea, kbp, nus, bam, kab are in the table above (mixed best minus NLLB-3.3B: kea -8.72, kbp -4.62, nus -4.08, bam -4.23, kab -5.31). Dropping those five languages leaves 38 languages; see `table2_per_language_43.csv` to recompute. (Reward-side data are disclosed in `paper_385m/DISCLOSURES.md` section 1.2.)
8. **Seeds and intervals.** One training seed per model; the language-bootstrap intervals above do not include training variance or sentence variance.
9. **Licence.** NLLB output and TranslatePsy data are non-commercial; the NLLB-derived share of the SFT and the pseudo-reference reward (DISCLOSURES.md) must be stated next to this comparison.

## 7. What the paper may and may not say

* May say: "On the 43 FLORES+ devtest languages shared with the published NLLB-200 table, the final RL model reaches 35.5 chrF++ against 38.8 for NLLB-3.3B (92%; 3.3 below, language-bootstrap CI [-4.3, -2.2]), 38.1 for NLLB-1.3B and 39.1 for the 54.5B MoE; the SFT reached 26.3 (68%). The RL model matches or exceeds NLLB-3.3B on 7 of 43 languages, all in the lowest NLLB tertile." State the references, decoding and one-seed caveats of section 6.
* May say (with the same-sentence evidence): "with beam-4 the RL stage-A model is 1.4 chrF++ below NLLB-3.3B on identical 100-sentence subsets of 43 languages".
* Must not say: that the model is "competitive with" or "matches" NLLB overall; that it beats NLLB on low-resource languages in general (7 languages, one of them the suspect Yoruba reference value; five languages have NLLB in the reward); anything about sentence-level significance against NLLB (not computable from stored data).

## 8. What could not be done

* Sentence-paired bootstrap of RL against NLLB on the full 1,012-sentence devtest: NLLB hypotheses on all 1,012 sentences are not stored (only published per-language numbers and the 100-sentence drafts on the A100). Needs ~0.2 GPU-h (E-06 option in EXPERIMENTS_LEFT) or the stored `/workspace/pe` drafts (then at most 100 sentences per language).
* RL-model chrF++ on the 100 beam sentences with greedy decoding (the stored greedy run is on 250 sentences; the overlap is unverified).
* NTREX and MAFAND comparisons with NLLB: no published per-language NLLB numbers for them; the post-editor run has NLLB drafts for NTREX (22 languages, 100 sentences) in `eval_e1b/nllb3.3b_greedy/metrics_by_language.csv` and the RL beam-4 drafts for the same sentences (`beam4`), which could be added in a follow-up with the same script.
