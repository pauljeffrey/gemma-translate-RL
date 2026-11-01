# Triage difficulty score vs translation quality (T1 report, saved 2026-10-23)

Classifier: `<HF_ORG_LEGACY>/modernbert-large-triage-v7-pangolin` (weights vocab 0.25, grammar 0.20, discourse 0.20, cultural 0.20, format 0.15; bands <=3.0 / 3.0-4.0 / 4.0-5.5 / 5.5-7.0 / >7.0). Files: `reports/data/triage_difficulty_2026-10-23/` (CSVs 01-13, `figures/fig1-9`, `code/`). Per-chunk tables for the other benchmarks are on the box in `/workspace/jobs/t1/out/`.

## Answer
1. On the benchmarks the score is a weak predictor of quality. There is no cliff and no sharp turning point.
2. AfriCOMET (against human references) is flat up to about 3.0-3.5, then declines roughly linearly by about 0.02 per score point. Relative to chunks scoring <=3.0: -0.017 at 3.5-4.0, -0.023 at 4.0-4.5, -0.041 at 4.5-5.0. NLLB-3.3B declines the same way (-0.013 / -0.021 / -0.039), so this is a property of the text (or metric), not of our model.
3. The gap RL-final minus NLLB-3.3B barely changes with score (-0.003 to -0.008 AfriCOMET per band, on top of a gap of about -0.05 in every band and group).
4. chrF++ shows no drop: the score correlates with source length (Spearman 0.35-0.68) and longer chunks get higher chrF++; after controlling for length the slope is about 0 (+0.06 [-0.18, +0.31] per point).
5. The benchmarks reach only 5.7: 3 chunks above 5.5, none above 5.7. "Hard" and "very hard" are untested; thresholds above about 4.5 cannot be calibrated here.

## Method
- English sources classified once (35,596 distinct chunk texts; official integer-rounded score reproduced exactly, max diff 2e-16).
- Chunks of k = 1, 3, 5 consecutive same-document sentences (FLORES+ by `url`, GlotStoryBook by story id; NTREX/MAFAND have no document ids so some k=3/5 chunks straddle articles, est. 10-25%). Identical chunking for all models.
- chrF++ on joined hypothesis vs joined reference; AfriCOMET / SSA-COMET = mean of existing sentence scores (spot check vs joined-text scoring: r 0.93-0.94 for RL, 0.69-0.84 for NLLB; RL-minus-NLLB gap about 0.01 more negative when joined).
- Primary subset: `no_pair` rows. Cluster bootstrap over distinct English chunks. Turning-point outcome = quality relative to the language's own mean on that benchmark and k.
- Language groups = terciles of RL-final FLORES+ chrF++ (15 strong, 15 mid, 15 weak); they are quality strata, not independent resource tiers.
- New GPU work: AfriCOMET-ref for 158,523 non-FLORES rows (SFT, RL, NLLB-3.3B). Not run: SSA-COMET beyond FLORES+, AfriCOMET for GlotStoryBook NLLB.

## Score distribution (distinct English chunks)
| Benchmark | k | n | mean | very easy | easy | moderate | hard | english-form |
|---|---|---|---|---|---|---|---|---|
| FLORES+ | 1 | 1012 | 3.15 | 419 | 542 | 42 | 0 | 9 |
| FLORES+ | 3 / 5 | 234 / 71 | 3.57 / 3.76 | 27 / 2 | 172 / 55 | 34 / 14 | 0 | 1 / 0 |
| NTREX | 1 | 1997 | 3.31 | 600 | 1125 | 250 | 0 | 22 |
| NTREX | 3 / 5 | 665 / 399 | 3.91 / 4.11 | 44 / 8 | 326 / 159 | 295 / 232 | 0 | 0 |
| MAFAND | 1 | 17986 | 3.30 | 5491 | 10705 | 1699 | 1 | 90 |
| SmolSent | 1 | 863 | 2.78 | 614 | 231 | 8 | 0 | 10 |
| SALT | 1 | 500 | 2.56 | 390 | 103 | 1 | 0 | 6 |
| GlotStoryBook | 1 | 938 | 2.10 | 897 | 25 | 0 | 0 | 16 |

The score grows with chunk size (FLORES+ 3.15 -> 3.57 -> 3.76 for k = 1 -> 3 -> 5): classify the unit that will actually be translated. Pooled k=1 quintile cut points 2.65 | 3.10 | 3.50 | 3.75; terciles 2.90 | 3.55. The score is compressed, so ranks within the owner's own data are more useful than the absolute bands.

## Per band: RL-final vs NLLB-3.3B (FLORES+ k=1, no_pair, 45 languages)
| Band | RL chrF++ | RL AfriCOMET | NLLB chrF++ | gap chrF++ | gap AfriCOMET |
|---|---|---|---|---|---|
| very easy | 32.9 | 0.621 | 37.2 | -4.15 [-4.43, -3.83] | -0.051 |
| easy | 34.0 | 0.616 | 38.1 | -3.96 [-4.19, -3.75] | -0.053 |
| moderate | 35.2 | 0.622 | 39.5 | -4.19 [-4.62, -3.70] | -0.055 |

RL wins about 30% of chunks on chrF++ and 20-24% on AfriCOMET in every band. By group the AfriCOMET gap is about -0.04 (strong), -0.04/-0.05 (mid), -0.07/-0.08 (weak); on NTREX weak languages -0.12 to -0.14. RL >= NLLB-3.3B on FLORES+ AfriCOMET for 4 of 45 languages (fon, yor, wol, ibo) in every band (effectively a tie).

Overall FLORES+ k=1 (chrF++, gap to NLLB-3.3B): RL 33.6 (-4.06), SFT-final 30.4 (-7.28), science-stage start 33.7 (-3.99), science step 250 33.9 (-3.80), NLLB-3.3B 37.8, NLLB-600M 35.8 (-2.02), TranslatePsy-0.8B 45.5 (+0.59, 19 languages), TranslatePsy-2B 46.7 (9 languages).

## Turning points (language-relative AfriCOMET, all benchmarks pooled, k=1)
| Bin | n | RL | NLLB-3.3B |
|---|---|---|---|
| (3.0, 3.5] | 5248 | -0.002 [-0.007, +0.003] | 0.000 |
| (3.5, 4.0] | 5854 | -0.017 [-0.022, -0.012] | -0.013 |
| (4.0, 4.5] | 1506 | -0.023 [-0.031, -0.015] | -0.021 |
| (4.5, 5.0] | 272 | -0.041 [-0.055, -0.029] | -0.039 |
| >5.0 | 22 | -0.009 (ns) | |

First significant drop is at >3.5 for strong and mid languages, >3.0 for weak. Hinge model breakpoint 3.10 [2.50, 3.50]; slope above -0.025 per point. A gradual knee, not a cliff. NTREX is steeper (slope -0.031); FLORES+ alone is not significant (little range). Long documents: quality tracks length, not score.

## Confounders
- Length: the AfriCOMET decline remains after controlling for length (pooled -0.013 alone, -0.013 given length) and in every length tertile.
- Dimensions (partial r with RL AfriCOMET): vocabulary -0.066, discourse -0.063, cultural -0.061 (NTREX -0.212), grammar +0.068 (collinearity), format -0.008. Best single dimension: cultural_load, then vocabulary_load; all weak. AUC for "worst decile": cultural 0.510 (NTREX 0.565), weighted score 0.488 (chance).

## Agreement with NLLB-3.3B output
chrF++ against NLLB-3.3B output vs against human reference (FLORES+ k=1): RL 40.7-43.1 vs 33.1-35.3 (+7.6 to +7.8); SFT about +7; NLLB-600M +16.7; TranslatePsy-0.8B +19.6 to +20.9. Our models are not inflated like the imitators. As a quality signal (AUC for RL's worst language-relative AfriCOMET quartile): low agreement with NLLB-3.3B 0.675 [0.670, 0.680] pooled; difficulty score 0.504; source length 0.475. Caveat for the owner: this signal needs NLLB output for the same text, which the owner cannot use for production (licence).

## english_form_dependent
0.7% of chunk-language rows (language-lesson text: capitalisation, pronunciation, spelling). The score does not find them (mean 2.5 vs 3.2). Quality RL: AfriCOMET 0.584 vs 0.622, chrF++ 29.7 vs 35.0 (NTREX 29.1 vs 36.4).

## Risk by band (RL)
Any failure (not finished, loop, wrong language) 5.8 / 4.5 / 4.4% (very easy / easy / moderate). Not finished <=0.2%, loops <=0.5%. Wrong language dominates and is highest for the shortest chunks (MAFAND very easy 8.2% vs 2.5% moderate; GlotLID is less reliable on very short text); failures do not rise with difficulty up to 5.5. SFT-final fails in 12-15% of chunks in every band. "Poor" (AfriCOMET <0.5): 19.4 / 16.8 / 16.4%.

## Expected quality if translating only chunks with score <= x (RL, pooled, k=1)
| x | coverage | AfriCOMET | chrF++ |
|---|---|---|---|
| 3.0 | 36% | 0.621 | 33.1 |
| 3.5 | 62% | 0.624 | 34.2 |
| 4.0 | 91% | 0.623 | 34.9 |
| 4.5 | 98.5% | 0.622 | 35.0 |

Mean quality of the translated set hardly changes with the threshold; each extra band is a small share of the chunks. Recommendation from this evidence: score <=3.5 is the "no measurable loss" tier; 3.5-4.5 translate with slight loss (about -0.02 AfriCOMET); above 4.5 loss about -0.04 on few chunks. Weak languages: start the slight-loss tier at 3.0. Benchmarks do not support a quality-based cut-off at 5.5 or above; do not read them as evidence that hard text is safe.

## Caveats
Domain shift (benchmarks are news/wiki/short sentences; the owner's data are instruction/general datasets with more mass in the moderate band and more format/cultural variety); effects are small (0.01-0.04 AfriCOMET against a between-system gap of 0.05); n is small above 4.5; classifier labels are LLM-produced (card: 74% band accuracy); AfriCOMET per chunk is a sentence mean. Use "human references and human-judgement-correlated metrics (AfriCOMET, SSA-COMET)" in the paper.

## Proposed 200-sample check on the owner's real data (not run)
Classify the unit as it will be translated; draw 200 items stratified on the score (40 per bin of <=3.0, 3.0-3.5, 3.5-4.0, 4.0-4.5, >4.5, oversampling english-form); 3-4 target languages (strong, mid, weak); translate with RL-final (greedy, 768 tokens); score reference-free (AfriCOMET-QE, SSA-COMET-QE) plus failure flags; analyse bin vs "very easy" and the AUC of the score against QE. A 0.02 drop is visible with about 40 per bin only if the effect is as large as in NTREX.
