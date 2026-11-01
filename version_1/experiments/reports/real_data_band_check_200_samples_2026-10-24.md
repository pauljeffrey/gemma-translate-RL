# 200-sample check of the final checkpoint on the owner's `<HF_ORG>/*-triage` datasets (B200 report, saved 2026-10-24)

IMPORTANT: no human references and no native-speaker review. Everything is reference-free quality estimation (AfriCOMET-QE, SSA-COMET-QE) plus automatic failure flags: screening signals, not measured translation quality. AfriCOMET-QE was our RL reward model (not independent); SSA-COMET-QE, GlotLID, loop, copy, length and self-consistency are the independent signals. No NLLB or TranslatePsy output used. Files (ids, scores, bands, flags; NO text): `reports/data/real_data_band_check_2026-10-24/` (01 sample list, 02 unit-language metrics, 03-17 aggregates, figures 1-5). Texts and outputs were on the box (`/workspace/jobs/b200`, now wiped; the whole folder is in the private Hub backup `<HF_ORG_LEGACY>/sabiyarn-box-final-backup`).

## Setup
- Model: final2-step401 (greedy plus 4 samples at T=0.7), languages swh, zul, hau, yor, kin, amh; 200 units x 6 languages = 1,200 unit-language rows.
- 34 datasets in the sample (chat/instruction, medical, finance, maths, RAG/QA, Wikipedia, books, web documents, customer support, agriculture); candidate pool 6,029 units from two random parquet parts per dataset (seed 20261024); chat sets: the assistant reply of a random turn; units 6+ words, at most 30% non-ASCII.
- DEVIATION: the published `*-triage` datasets carry only the categorical `translation_difficulty` label and the `english_form_dependent` flag, not the numeric score; the unit was re-scored with `modernbert-large-triage-v7-pangolin` (published band agrees exactly for 71.5% of units, because the published one was computed on the assistant turn plus earlier turns).
- Strata: about 30 units per bin (<=3.0, 3.0-3.5, 3.5-4.0, 4.0-4.5, 4.5-5.5, >5.5) plus 20 english_form_dependent units. The >5.5 bin is thin and narrow: 24 of its 30 units are billsum (US legislative text).
- 163 units single-pass (<=512 Gemma tokens), 37 sentence-by-sentence with repair (sentfix). Token budget: 768 new tokens left 23.7% of rows with 257-512 source tokens unfinished (yor 50.8%); re-run with 2,048 new tokens for all single-pass units over 256 tokens (1.7% unfinished after); units up to 256 tokens: 0.2% unfinished at 768. Output tokens are 1.5-2.3x the source tokens (yor 2.26).
- Scoring caveat: COMET truncates each side at 510 tokens, so QE of long single-pass units is lower than sentence-level QE for structural reasons (3.9-28% of rows per band affected).

## Headline findings
1. The triage score does not predict failure on this real data: AUC for any failure 0.42 [0.34, 0.51], for poor SSA-COMET-QE 0.48 [0.41, 0.55], for poor AfriCOMET-QE 0.54 [0.47, 0.63]; wrong-language output is inversely related (AUC 0.31: the shortest, easiest texts are the ones GlotLID flags most). Agrees with the benchmark finding (weak predictor).
2. Raw QE drops with the score, but mostly through length and segment type: AfriCOMET-QE for score above 4.0 is 0.09-0.105 lower than for <=3.0 (significant); SSA-COMET-QE drop 0.04 (not significant). After adjusting for log source tokens and sentfix the slopes are -0.05 [-0.12, +0.02] and -0.02 [-0.10, +0.07] SD per point (benchmarks: knee about 3.5, -0.02 AfriCOMET per point). Unit-level COMET on 129-512-token single-pass units is much lower (AfriCOMET-QE 0.36, SSA 0.33) than on sentence-aligned long documents (0.55 / 0.47) or short units (0.54 / 0.43).
3. Failure rate by band (<=3.0 / 3.0-3.5 / 3.5-4.0 / 4.0-4.5 / 4.5-5.5 / >5.5): 7.2% / 6.7% / 8.9% / 6.1% / 6.1% / 3.3%; per-band CIs wide; no evidence of a rise with score.
4. Failure is predicted by QE (AUC 0.70-0.73), less by self-consistency (0.66), not by the score (0.42).
5. A cheap filter (language-ID + finished + loop + copy) rejects 6.4% of unit-language rows; adding consistency or QE at the language 10th percentile gives about 12%.

## Quality by band (all languages, 30 units per band)
| band | AfriCOMET-QE | SSA-COMET-QE | consistency chrF++ | any failure |
|---|---|---|---|---|
| <=3.0 | 0.507 [0.456, 0.557] | 0.419 | 58.0 | 7.2% |
| 3.0-3.5 | 0.485 | 0.403 | 56.3 | 6.7% |
| 3.5-4.0 | 0.454 | 0.420 | 58.5 | 8.9% |
| 4.0-4.5 | 0.403 [0.359, 0.448] | 0.377 | 56.3 | 6.1% |
| 4.5-5.5 | 0.419 | 0.377 | 57.1 | 6.1% |
| >5.5 | 0.418 [0.381, 0.454] | 0.374 | 56.7 | 3.3% |
| english_form_dependent (n=20) | 0.389 | 0.320 | 45.7 | 13.3% |
By language (means over bins; compare only within a language): swh SSA 0.488 / consistency 73.8 / failure 4.4%; zul 0.395 / 63.0 / 4.4%; hau 0.357 / 68.8 / 5.6%; yor 0.449 / 53.0 / 6.1%; kin 0.466 / 57.4 / 8.3%; amh 0.215 / 26.9 / 9.4%. Amharic is clearly worst (self-consistency 27 chrF++: the 4 samples disagree with the greedy output). Where QE drops (bin vs <=3.0): amh from 3.0-3.5 (AfriCOMET-QE -0.08, -0.13 to -0.165 above 3.5); mid (hau, kin) first significant at 4.0-4.5; strong (swh, zul) from 4.0 on AfriCOMET-QE, none on SSA; yor only above 5.5.
Failure components (non-english_form_dependent): 1.1% not finished, 3.2% wrong language, 2.7% loop, 0% copy; wrong language highest for kin (8.3%) and for units of 32 tokens or fewer (9.6%); loops highest for amh (6.1%). Domain (small n): rag_qa best (3.2% failure), maths worst (14.2% failure, wrong language 6.7%, loops 6.7%; maths/code-like text 14.1% vs prose 6.0%), medical 3.3%, web documents 4.9%, Wikipedia 12.2% (mostly loops). english_form_dependent units are scored as easy by the classifier (mean 2.59) but translate worse (QE 0.389 / 0.320 vs 0.496 / 0.411 for other easy units, consistency 45.7 vs 57.2, wrong language 13.3% vs 5.0%): keep them in English.

## Recommendations (from the report)
- Do not use the triage score as a quality gate below 5.5; use it as an ordering. By language group: strong (swh, zul) no score cut-off; mid (hau, kin) tier 1 <= 4.0, 4.0-5.5 only through the automatic filter; weak (yor) fine up to 5.5 with the token budget below; Ge'ez (amh) translate only <= 3.5 and treat the rest as needing native review.
- Single-pass units up to 256 source tokens: 768 new tokens is enough; 257-512 tokens: budget at least 2.5x the source tokens (3.5x for yor, about 2,048) or send everything above 256 tokens through sentence-by-sentence; drop or re-run sentfix documents with an unfinished piece (13.5% of yor documents). Maths/code-like units: protect formulas with placeholders and apply the stricter filter.
- Cheap automatic filter rejection rates: language-ID + finished + loop + copy 6.4% (strong 4.4%, mid 6.9%, weak 6.1%, Ge'ez 9.4%); plus consistency <= language 10th percentile 11.9% (share of bottom-20% SSA-COMET-QE rows in the kept set 18.1% -> 11.9%); consistency <= 20th percentile 20% (9.1%). Language 10th-percentile thresholds from this sample (not validated): SSA-COMET-QE swh 0.327, zul 0.220, hau 0.198, yor 0.287, kin 0.287, amh 0.085; AfriCOMET-QE swh 0.253, zul 0.263, hau 0.240, yor 0.328, kin 0.302, amh 0.185; consistency swh 51, zul 38, hau 42, yor 26, kin 35, amh 12.5. A human check of a few rejected and kept rows per language is needed before using them.

## Caveats
30 units per band, 12-18 failures per language, 20 english_form_dependent units, one source (billsum) in the top bin: differences below about 0.05 are within noise. The pool is a random sample of two parts per dataset, not the whole datasets (pool coverage by score: 34.0% <=3.0, 53.9% <=3.5, 72.2% <=4.0, 88.2% <=4.5, 97.6% <=5.5). GlotLID is unreliable on very short and symbol-heavy text and has no `kon` label. These are quality-estimation numbers; no human references or native-speaker review.
