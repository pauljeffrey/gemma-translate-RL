# E-12: language family, script and resource level of the RL gain and of the SFT level (2026-11-04)

CPU only, from stored files. Scripts and data: `reports/data/family_script_resource_analysis_2026-11-04/` (`c1_build_registry.py` -> `registry_125.csv`; `c2_analysis.py` -> regressions; `c3_tables_and_figure.py` -> 125-language table, coverage table, figures). All analyses are **observational**; n is printed next to every number. The gain is "mixed best minus SFT", FLORES+ devtest chrF++ (identical to EVID Table 6 to 1e-14, script check); FLORES+ has no flagged rows, so all rows = clean rows.

## 0. Findings

1. **External family table.** Glottolog 5.3 (CLDF release v5.3, tag commit `072ca0d0410039fb8b779be8fc165bac575d2cda`, released 2026-03-02, CC BY 4.0, retrieved 2026-11-04 from `https://github.com/glottolog/glottolog-cldf`, files `cldf/languages.csv` (stored, 2.4 MB: `glottolog_v5.3_languages.csv`) and `cldf/values.csv` (`classification` paths, 21 MB, not stored; the 125 extracted paths are in `registry_125.csv`)). All 125 registry languages are mapped (120 by ISO 639-3 code, 5 macrolanguage/collective codes by a stated manual choice: orm, mlg, kon, luy, ber, see `mapping_note`). Caveat: the CLDF metadata file labels its own citation string "Glottolog 5.2.1" while the release tag is v5.3 (its provenance block names glottolog v5.3 and CLDF repository v5.2.1); cite the tag.
2. **The 8 languages without a group** (EVID Table 4 "family unlabelled", FLORES+ n = 8: aeb ary arz cjk dik mfe taq zgh) now have an external label (table in section 2). Across the whole 125-language registry 59 languages are "family unlabelled" in `groups_family.json`; all 125 have a Glottolog family in `registry_125.csv` / `languages_125_family_script_gain.csv`.
3. **The RL gain is not explained by the token budget.** Spearman rho(gain, log10 SFT tokens) = -0.16 [bootstrap -0.44, +0.13] (n = 54); a one-predictor OLS slope is +0.34 chrF++ per log10 unit [-2.03, +1.79], R2 0.003 (n = 54). The **SFT level is** (rho = +0.68 [+0.47, +0.82] with log10 tokens, n = 54; OLS slope +11.8 chrF++ per log10 unit [+8.5, +17.0], R2 0.45). I.e. the RL gain is roughly flat in the data volume while the SFT quality is not: the RL model is a much flatter function of the token budget than the SFT.
4. **What the gain does track** is the SFT's failure rate: rho(gain, SFT finished rate) = -0.47 [-0.74, -0.15], rho(gain, SFT repeat rate) = +0.46 [+0.13, +0.72], rho(gain, SFT chrF++) = -0.28 [-0.54, +0.04] (n = 54 each). This agrees with ledger R4/R5 (about half the gain is finishing/looping) and means family, script and resource effects on the raw gain are largely effects on how badly the SFT failed.
5. **Family and script.** Mean gain by Glottolog top family (FLORES+, language mean [bootstrap CI over languages]): Nilotic +11.58 [10.30, 13.13] (n = 3), Atlantic-Congo +9.14 [7.58, 10.79] (n = 33; Narrow Bantu +9.91 [7.96, 11.71], n = 24; other Atlantic-Congo +7.06 [4.76, 9.61], n = 9), Mande +8.42 (n = 2), Afro-Asiatic +6.89 [4.29, 9.89] (n = 11), Indo-European (creoles, Afrikaans) +10.23 [1.01, 20.27] (n = 3). By script: Latin +9.32 [8.03, 10.68] (n = 48), Ethiopic +9.13 (n = 2), Arabic +2.73 [1.36, 4.59] (n = 3), Tifinagh +1.78 (n = 1). **The lowest gains are Arabic-script dialects (aeb ary arz) and Tifinagh (zgh)**, the languages where the SFT's output was English/MSA or loops and the RL did not repair them (ledger W2, W3, R7).
6. **Regression results (n = 53, zgh has no NLLB-share value).** Model A (family + log tokens + non-Latin script + AfriCOMET availability + NLLB-mined share) explains little of the gain (R2 0.27, adjusted R2 0.09); only the contrast "non-Bantu Atlantic-Congo vs Narrow Bantu" is clearly non-zero (-4.45 chrF++, HC3 p = 0.020, bootstrap [-8.36, -1.24]) and the NLLB-mined share is positive (+7.57 per unit share, bootstrap [+0.91, +14.67], HC3 p = 0.066). It explains most of the SFT level (R2 0.67, adjusted 0.59; log tokens +6.45 [-0.25, +16.69], NLLB share -15.61 [-30.74, -6.79]). Section 4 has all coefficients.
7. **Not separable from each other** (collinearity): family, script and AfriCOMET availability are strongly tied to the number of tokens (VIF of log tokens 2.1 without and 3.4 with family dummies); the single-family leave-out check shows the token slope on the gain is driven by one family (dropping Afro-Asiatic turns +0.34 into -3.00, SE 1.06, `leave_one_family_out.csv`), so "no relation" is not robust to which languages are in the sample; n = 54 languages, 7 families.

## 1. Data and definitions

| item | definition | source |
|---|---|---|
| registry | 125 languages of the single-token tags | `reports/data/tag_token_map_2026-11-01.csv` |
| family | Glottolog 5.3 top-level family of the language (Family_ID), plus a regression family `family_reg`: Atlantic-Congo split into "Narrow Bantu" (path contains Glottolog node Narrow Bantu) and "non-Bantu", Afro-Asiatic, Nilotic, Mande, Indo-European (Afrikaans and the English/French/Portuguese-lexifier creoles; Glottolog classifies creoles by lexifier), Other (Austronesian, Saharan, Songhay, Ta-Ne-Omotic). `family_branch` is the second Glottolog level (Berber, Semitic, Cushitic, Chadic, Egyptian, Benue-Congo, Kwa Volta-Congo, Gur, ...) | `registry_125.csv` |
| script | dominant Unicode script of the human reference text of the language in any of the benchmark files (empirical; 74 languages, share of dominant script 0.976 to 1.0) else assigned from prior knowledge (51 languages: all Latin except cop Coptic and tzm Tifinagh; tzm, ber, taq (Latin in the benchmark), shi, knc, zgh are the uncertain assignments; flagged `UNCERTAIN` in `script_source`). Result: Latin 116, Arabic 4 (aeb ary arz shi), Ethiopic 2 (amh tir), Tifinagh 2 (zgh, tzm), Coptic 1 (cop). Note the benchmark text of shi (Tachelhit; NTREX) is Arabic script, not the Latin script of FLORES+ | `registry_125.csv` (`script`, `script_empirical_share`) from `research_archive_2026-11-03/eval_385m/out_m385/predictions.jsonl` (reference field) |
| SFT tokens | `all_streams_Mtok` of the provisional token table (A0 + new translation + instruction streams) for 114 languages; the stage-B pack tokens of `nllb_share_by_language.csv` for 5 (ary aeb bci teo adh); none for 6 (ibb dag bbj fat gaa men: the languages have no entry in either table). Correlation of the two token measures on the 111 languages with both: 0.97 (log10) | `reports/data/token_table_provisional_2026-11-01.csv`, `reports/data/nllb_regression_link_2026-11-03/nllb_share_by_language.csv` |
| NLLB share | `share_nllb_mined` = NLLB-mined tokens / stage-B pack tokens (0.00 to 0.95 on the 53 FLORES+ languages that have it; mean 0.26, median 0.15); and the published NLLB-3.3B FLORES-200 chrF++ as a **resource proxy** (43 languages) | `nllb_share_by_language.csv`; `published_nllb_flores_per_language.csv` |
| AfriCOMET availability | 1 if the language has a held-out AfriCOMET-STL score on FLORES+ (40 of 54), else 0 | `results_mixedbest.csv` comet column |
| script_nonlatin | 1 for Ethiopic, Arabic, Tifinagh (6 of the 54 FLORES+ languages: amh tir aeb ary arz zgh; shi is Arabic-script but not on FLORES+) | `flores54_regression_table.csv` |
| gain, SFT level | per-language chrF++ SFT and mixed best, FLORES+ devtest (54 languages) | `cells_gain_sft_rl_chrfpp.csv`, from `rl_vs_sft_evidence_2026-11-04/evidence.json` cells |

Coverage of the 125 languages (`coverage_125_by_group.csv`, `languages_125_family_script_gain.csv`): Atlantic-Congo 87 (Narrow Bantu 59), Afro-Asiatic 17, Indo-European 7, Nilotic 7, Mande 3, Austronesian 1, Saharan 1, Songhay 1, Ta-Ne-Omotic 1. 54 languages have a FLORES+ gain, 58 have a gain on at least one of FLORES+/NTREX/MAFAND (the three benchmarks the RL models were evaluated on; 92 to 93 language-benchmark cells), 74 have any reference test, 51 have none (23 of them Narrow Bantu, 15 other Atlantic-Congo). Languages without any reference test cannot enter the regressions; the family result therefore describes the 54 (or 58) evaluated languages, which are the better-resourced ones (median 25.4 Mtok on FLORES+ against 8.6 for the Latin-script registry overall).

## 2. The 8 languages that lacked a group (FLORES+, n = 8)

`unlabelled8_labels.csv`. Our group in EVID Table 4 was "family unlabelled" (the "new language (40)" set).

| iso | language | Glottolog language (glottocode) | Glottolog top family | branch / lexifier | script | SFT | mixed best | gain |
|---|---|---|---|---|---|---:|---:|---:|
| aeb | Tunisian Arabic | Tunisian Arabic (tuni1259) | Afro-Asiatic | Semitic (Maghrebi Arabic) | Arabic | 2.75 | 4.12 | +1.36 |
| ary | Moroccan Arabic | Moroccan Arabic (moro1292) | Afro-Asiatic | Semitic (Maghrebi Arabic) | Arabic | 4.64 | 6.88 | +2.24 |
| arz | Egyptian Arabic | Egyptian Arabic (egyp1253) | Afro-Asiatic | Semitic (Egyptic Arabic) | Arabic | 4.59 | 9.18 | +4.59 |
| cjk | Chokwe | Chokwe (chok1245) | Atlantic-Congo | Narrow Bantu | Latin | 10.72 | 23.02 | +12.31 |
| dik | Dinka | Southwestern Dinka (sout2832) | Nilotic | Western Nilotic (Dinka-Nuer) | Latin | 3.76 | 14.06 | +10.30 |
| mfe | Mauritian Creole | Morisyen (mori1278) | Indo-European | Romance (French-lexifier creole) | Latin | 28.51 | 37.93 | +9.42 |
| taq | Tamasheq | Tamasheq (tama1365) | Afro-Asiatic | Berber (Tuareg) | Latin (benchmark text) | 2.81 | 11.06 | +8.25 |
| zgh | Standard Moroccan Tamazight | Standard Moroccan Tamazight (stan1324) | Afro-Asiatic | Berber | Tifinagh | 2.29 | 4.07 | +1.78 |

Reading: five of the eight are Afro-Asiatic (three Arabic dialects, two Berber) and they are the five lowest-scoring languages after RL (mixed best 4.07 to 11.06) and have the lowest gains except taq; cjk is a Bantu language (Chokwe) and dik a Nilotic one. The earlier statement that the 8 added languages show "-0.13 on clean rows" (R7) is therefore mostly a statement about Arabic dialects and Tamazight (script and language-identity problems), not about the other added languages (mfe +9.42, cjk +12.31, dik +10.30 over all rows).

## 3. Gain and SFT level by family and script (FLORES+, language means, bootstrap CI over languages, 2,000 resamples)

`gain_by_group_flores54.csv`, `sft_level_by_group_flores54.csv`. CIs are given only where n >= 2 (n = 1 has no interval).

| grouping | group | n | mean SFT | mean gain [CI] |
|---|---|---:|---:|---|
| Glottolog top family | Nilotic | 3 | 12.87 | +11.58 [10.30, 13.13] |
| | Saharan (knc) | 1 | 3.81 | +10.41 |
| | Indo-European (kea mfe afr) | 3 | 36.85 | +10.23 [1.01, 20.27] |
| | Atlantic-Congo | 33 | 26.20 | +9.14 [7.58, 10.79] |
| | Mande | 2 | 14.65 | +8.42 [7.78, 9.06] |
| | Afro-Asiatic | 11 | 13.45 | +6.89 [4.29, 9.89] |
| | Austronesian (mlg) | 1 | 39.04 | +5.68 |
| regression family | Atlantic-Congo: Narrow Bantu | 24 | 27.36 | +9.91 [7.96, 11.71] |
| | Atlantic-Congo: non-Bantu | 9 | 23.11 | +7.06 [4.76, 9.61] |
| Glottolog branch | Romance-lexifier creoles (kea mfe) | 2 | 23.71 | +14.84 [9.42, 20.27] |
| | Afro-Asiatic: Cushitic (orm som) | 2 | 23.76 | +11.18 [4.24, 18.12] |
| | Atlantic-Congo: Gur | 2 | 14.30 | +9.54 [8.70, 10.37] |
| | Afro-Asiatic: Berber (kab taq zgh) | 3 | 7.15 | +6.79 [1.78, 10.34] |
| | Afro-Asiatic: Semitic (amh tir aeb ary arz) | 5 | 7.82 | +5.29 [2.36, 8.22] |
| | Atlantic-Congo: Kwa Volta-Congo | 3 | 24.56 | +4.94 [2.07, 7.53] |
| | Atlantic-Congo: Benue-Congo | 2 | 29.57 | +5.67 [2.33, 9.01] |
| | Germanic (afr) | 1 | 63.13 | +1.01 |
| script | Latin | 48 | 24.84 | +9.32 [8.03, 10.68] |
| | Ethiopic (amh tir) | 2 | 13.55 | +9.13 [8.26, 10.00] |
| | Arabic (aeb ary arz) | 3 | 4.00 | +2.73 [1.36, 4.59] |
| | Tifinagh (zgh) | 1 | 2.29 | +1.78 |
| AfriCOMET available | yes (n = 40) / no (n = 14) | | 27.06 / 10.82 | +8.83 [7.39, 10.28] / +8.73 [6.18, 11.38] |

(All branches and the 8-group labels of EVID Table 4 are in the CSV; our own groups reproduce EVID Table 4: Bantu +9.81 (23), Atlantic-Congo non-Bantu +7.06 (9), Afroasiatic +9.59 (6), Nilo-Saharan +11.62 (3), unlabelled +6.28 (8).)

* The **higher the SFT level, the lower the gain** within every family (Spearman gain vs SFT level -0.28, n = 54), the clearest example being Afrikaans (SFT 63.13, gain +1.01) and swh (+2.12) and zul (+2.36). The mean gain of Narrow Bantu (+9.91) beats the non-Bantu Atlantic-Congo (+7.06) because Narrow Bantu has many mid-resource languages whose SFT failed by loops (nso, lua, lin, ssw, lug, bem, tso).
* Glottolog's Nilotic family on FLORES+ is luo, nus and dik (+11.58); the old "Nilo-Saharan" group of EVID is luo, nus and knc (+11.62): they differ by dik (unlabelled in EVID) versus knc (Saharan in Glottolog, alone in its family at +10.41), which is why the two rows agree.

## 4. Regressions (FLORES+, n = 54 languages; 53 where a covariate is missing)

`regression_coefficients.csv` (coefficients, HC3 robust SE and p, bootstrap 95% percentile CI over languages with 4,000 resamples), `spearman.csv`, `mixed_models.csv`, `vif.json`, `leave_one_family_out.csv`, `pooled_cells_regression.csv`. Reference family: Atlantic-Congo Narrow Bantu.

**Model A, full** `y ~ log10 tokens + nonLatin + AfriCOMET_available + NLLB_share + family_reg` (n = 53):

| term | gain: coef [bootstrap CI] (HC3 p) | SFT level: coef [bootstrap CI] (HC3 p) |
|---|---|---|
| intercept | 9.03 [6.22, 13.21] | 13.05 [4.54, 19.77] |
| log10 SFT tokens (M) | +1.93 [-3.31, +4.22] (0.345) | +6.45 [-0.25, +16.69] (0.183) |
| non-Latin script | -3.95 [-11.32, +1.91] (0.229) | -8.64 [-22.39, +5.45] (0.214) |
| AfriCOMET available | -4.20 [-8.30, +4.16] (0.260) | +9.75 [-3.32, +22.05] (0.238) |
| NLLB-mined share (0 to 1) | +7.57 [+0.91, +14.67] (0.066) | -15.61 [-30.74, -6.79] (0.055) |
| Atlantic-Congo non-Bantu | -4.45 [-8.36, -1.24] (0.020) | -1.54 [-7.64, +5.48] (0.664) |
| Afro-Asiatic | +0.01 [-4.64, +5.76] (0.997) | -4.96 [-14.72, +5.63] (0.364) |
| Nilotic | +2.02 [-4.89, +6.44] (0.499) | -3.24 [-13.09, +15.46] (0.651) |
| Mande | -4.95 [-14.26, +0.82] (0.502) | -4.77 [-17.81, +17.20] (0.788) |
| Indo-European (creoles, Afrikaans) | -2.22 [-8.59, +8.32] (0.670) | +15.63 [+0.13, +28.82] (0.134) |
| Other | -2.14 [-5.30, +2.94] (0.426) | -5.06 [-23.19, +8.40] (0.765) |
| R2 / adjusted R2 | 0.267 / 0.093 | 0.667 / 0.588 |

Other specifications (same CSV): B (no family terms, n = 53): gain R2 0.14 (log tokens +0.20 [-3.82, +2.17], NLLB share +4.74 [-1.06, +10.07]), SFT level R2 0.57 (log tokens +9.05 [+3.96, +16.75], non-Latin -13.12 [-23.22, 0.00], NLLB share -14.03 [-25.04, -6.77]); D (log tokens only, n = 54): gain slope +0.34, SFT slope +11.82; C (adds the published NLLB-3.3B chrF++, n = 43): the NLLB chrF++ is the strongest correlate of the SFT level (+0.77 chrF++ per NLLB chrF++ point [+0.47, +1.04], rho = 0.82, R2 0.79) and unrelated to the gain (-0.045 [-0.22, +0.24], rho = -0.16 [-0.48, +0.18]); E (gain given SFT level, n = 54, R2 0.51): SFT chrF++ -0.354 per point [-0.465, -0.239], log tokens +3.80 [+0.97, +6.44] (HC3 p = 0.007), Afro-Asiatic -5.26 [-8.68, -2.16], non-Bantu Atlantic-Congo -4.58 [-7.70, -1.96]: at equal SFT level, languages with more data gained more, and Afro-Asiatic and non-Bantu Atlantic-Congo gained less than Bantu. E is the only model where the token budget matters for the gain and it conditions on the outcome-related SFT level (a regression-to-the-mean pattern is possible), so it is a descriptive statement.

**Spearman** (n, rho, bootstrap CI): gain vs log tokens 54, -0.16 [-0.44, +0.13]; gain vs NLLB share 53, +0.40 [+0.15, +0.60] (p = 0.003); gain vs NLLB-3.3B chrF++ 43, -0.16 [-0.48, +0.18]; gain vs AfriCOMET availability 54, +0.02 [-0.27, +0.29]; gain vs non-Latin script 54, -0.31 [-0.51, -0.06]; SFT level vs log tokens 54, +0.68 [+0.47, +0.82]; vs NLLB-3.3B chrF++ 43, +0.82 [+0.66, +0.90]; vs AfriCOMET availability 54, +0.54 [+0.32, +0.71] (AfriCOMET covers the better-resourced languages); vs non-Latin script 54, -0.46 [-0.62, -0.24]; vs NLLB share 53, -0.12 [-0.45, +0.20].

**Mixed model** (random intercept by group, fixed log tokens, script, AfriCOMET, NLLB share; REML) `mixed_models.csv`: share of the variance at the family level (ICC, null model, FLORES+ n = 54): gain 0.03 (family_reg, 7 groups), 0.09 (Glottolog branch, 16 groups), 0.01 (top family, did not converge); SFT level 0.21 (family_reg), 0.52 (branch), 0.35 (top family). I.e. the family explains little of the gain and a moderate-to-large part of the SFT level (the branch ICC 0.52 is inflated by many singleton branches and should not be quoted). Three of the six gain models did not converge in statsmodels; the `converged` column marks them.

**Pooled check** (gain and SFT level of all FLORES+/NTREX/MAFAND language cells, benchmark fixed effects, SE clustered by language; 92 cells and 57 languages after dropping cells with a missing covariate, `pooled_cells_regression.csv`): gain: log tokens -0.04 (SE 0.91), NLLB-mined share +7.20 (SE 2.83, p = 0.011), non-Bantu Atlantic-Congo -3.36 (SE 1.82, p = 0.064); SFT level: log tokens +9.48 (SE 2.31), non-Latin script -16.47 (SE 6.23), NLLB share -14.04 (SE 5.43). The signs agree with FLORES+ alone.

**Collinearity** (VIF, `vif.json`): numeric covariates log tokens 2.13, non-Latin 1.15, AfriCOMET 2.15, NLLB share 1.18; with family dummies log tokens 3.39, AfriCOMET 3.33, script 1.92, NLLB share 1.47; with the NLLB chrF++ proxy (n = 43) NLLB share 2.86 and log tokens 2.38. No VIF exceeds 5, but token volume, resource tier (AfriCOMET availability, NLLB chrF++) and family overlap in substance.

**Why the NLLB-mined share is "positive" for the gain and "negative" for the SFT level**: languages that depend on NLLB-mined stage-B data are the tail languages (median NLLB-3.3B chrF++ lowest tertile), they started low and failed by loops; the RL repaired them most. This is a confounded observational association; it says nothing on whether NLLB data hurt the SFT (ledger K1/K2 stay unresolved) and must not be worded causally.

## 5. Limitations and what was not done

* Only 54 languages (58 with NTREX/MAFAND) have an RL gain; 67 of 125 registry languages have no gain, 51 have no reference test at all, so no family or script statement is possible for them (the registry table gives their Glottolog label and token budget only).
* Seven families, several with 1 to 3 languages; family effects other than Bantu vs non-Bantu Atlantic-Congo and Afro-Asiatic are not estimable. Script: 7 non-Latin languages (Arabic dialects, Ethiopic, Tifinagh) so "script" is largely "Arabic dialects + Tamazight".
* Token budget is a proxy: the token table is "provisional" (its own file name), stage-B pack tokens are used for five languages, and six languages have none; tokens of different data classes (religious text, instruction, NLLB-mined) are not distinguished here.
* Script of 51 languages without a benchmark is assigned from prior knowledge, not from data (six marked uncertain).
* Glottolog classifies creoles by lexifier, so "Indo-European" contains kea, mfe, pcm, kri, wes, crs and Afrikaans; do not read it as a genetic statement about the African languages. The macrolanguage codes were mapped by a stated choice that does not change the family.
* The gain is "mixed best minus SFT" on one seed; language-level bootstrap intervals describe language sampling only, and all the regressions use n = 54 with up to 11 parameters (model A), i.e. they are exploratory. Multiple comparisons are not corrected.
* AfriCOMET availability, tokens and NLLB resource are mutually confounded; the paper should show this analysis as descriptive (appendix), not as a causal claim.
* Not done: per-language gain with sentence-level intervals by family (EVID Table 4 already gives family-level paired bootstrap intervals for our own groups); a phylogenetic random effect (Glottolog trees exist in `classification.nex` but were not used); regression on the AfriScience-MT six languages (n too small).

## 6. Files

`registry_125.csv`, `languages_125_family_script_gain.csv`, `coverage_125_by_group.csv`, `flores54_regression_table.csv`, `cells_gain_sft_rl_chrfpp.csv`, `gain_by_group_flores54.csv`, `sft_level_by_group_flores54.csv`, `unlabelled8_labels.csv`, `regression_coefficients.csv`, `spearman.csv`, `mixed_models.csv`, `leave_one_family_out.csv`, `pooled_cells_regression.csv`, `vif.json`, `fig_gain_sft_tokens_family.png`, `fig_gain_by_glottolog_branch.png`, `glottolog_v5.3_languages.csv` (all in `reports/data/family_script_resource_analysis_2026-11-04/`).
