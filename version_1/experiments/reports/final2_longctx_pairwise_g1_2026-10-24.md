# G1 report: LongCtx-MT and pairwise learned-metric comparison for the final checkpoint (final2, step 401), saved 2026-10-24

Protocol, documents, languages and scoring scripts identical to `reports/long_context_benchmark_2026-10-23.md`; one vLLM process (gpu util 0.25, nice 19, 12 min wall-clock: whole document 249 s, sentence units about 6 min, repair 6 s). Step-250 rows re-aggregated with the same code reproduce exactly (max difference 0). Section 6.7 evaluation script gives 48.12 / 47.30 sentence-level chrF++ on AfriScience dev/test; this run 48.06 / 47.26.

## (A) LongCtx-MT, chrF++ against human references, macro over common languages, whole document (95% CI over documents)
| bin | final2 | step 250 | A2W | RL-final |
|---|---|---|---|---|
| stitched ~1,024 | 39.6 [38.3, 40.7] | 39.6 | 39.4 | 25.6 |
| stitched ~2,048 | 27.7 [25.8, 29.7] | 28.2 | 28.8 | 17.9 |
| AfriScience dev | 43.2 [40.7, 45.3] | 43.8 | 43.7 | 33.8 |
| AfriScience test | 41.3 [39.2, 43.3] | 41.5 | 41.6 | 31.5 |
- Sentence + repair (sentfix): 52.6 / 55.5 / 55.6 / 55.4 (same as step 250); the wrapper adds +13.1 [+11.9, +14.4] at ~1,024 and +27.8 [+26.4, +29.1] at ~2,048. 222 of 262 unfinished sentence units were repaired, 40 remain unfinished.
- Paired final2 minus step 250 (whole): the chrF++ interval includes zero in all 8 columns (e.g. -0.46 [-2.31, +1.40] at ~2,048). Against RL-final final2 is +9.8 [+8.0, +11.5] at ~2,048.
- Failure metrics (whole, stitched ~1,024 / ~2,048): finished 0.98 / 0.92; loop 0.08 / 0.25; coverage 0.88 / 0.73; length ratio 0.81 / 0.59; GlotLID 0.95 for both. Step 250 had loop 0.04 / 0.10; paired loop difference +0.04 [+0.01, +0.08] and +0.15 [+0.10, +0.20] (loops clearly higher; one run per checkpoint, may be a property of this checkpoint). Against A2W final2 finishes more documents at ~2,048 (+0.12 [0.00, +0.24] stitched, +0.19 [+0.12, +0.30] core).
- Reading: final2 matches step 250 in chrF++ and omits as much text (about 12% and 27%); the further 151 RL steps did not help long input.

## (B) Pairwise learned-metric comparison, final2 vs RL-final (NTREX, MAFAND, SmolSent pooled, no_pair rows, ties below 1 chrF++ / 0.01, paired sentence bootstrap; AfriCOMET-QE not used)
| metric | languages / sentences | RL-final -> final2 | difference [95% CI] | final2 preferred, % of non-tied [95% CI] |
|---|---|---|---|---|
| sentence chrF++ | 47 / 100,982 | 32.37 -> 32.47 | +0.10 [+0.07, +0.13] | 52.4 [52.0, 52.8] |
| AfriCOMET-ref | 40 / 92,673 | 0.6039 -> 0.6012 | -0.0027 [-0.0032, -0.0023] | 47.5 [47.1, 47.8] |
| SSA-COMET-ref | 22 / 42,700 | 0.4918 -> 0.4885 | -0.0033 [-0.0041, -0.0025] | 47.6 [47.1, 48.2] |
- Pipeline check: on 14,339 rows where both systems wrote identical text the two scoring pipelines differ by 0.0002 on average (max 0.005). final2 scores came from the private Hub dataset `<HF_ORG_LEGACY>/SabiYarn-MT-benchmark-translations` (read only); RL-final scores from E1 neural files and T1 comet_extra. SSA-COMET-ref for RL-final exists only on E1's hash sample (22 languages).
- AfriCOMET-ref clearly lower in 17 of 40 languages, clearly higher in 1 (Luganda); largest drops Amharic -0.0154 [-0.0188, -0.0119], Yoruba -0.0099, Oromo -0.0083, Northern Sotho -0.0064, Igbo -0.0068 (flat in chrF++). The only preference interval including 50% is SSA-COMET on MAFAND-MT, 49.2 [48.1, 50.3]. RL over SFT was preferred in about 70% of non-tied sentences by the learned metrics.
- Reading: level. chrF++ and the learned metrics disagree on the sign of a very small difference, and the learned metrics are not held-out for final2 (its reward contained a QE ensemble).

## Not run
Learned metrics on AfriScience documents (too long for the encoders); pairwise comparison on SALT, GlotStoryBook, IbomMT, FLORES-MOZ; human evaluation; seed-to-seed variance.

## Paper and files
Sections 6.7, 6.9, 6.10, Appendix H, 10, 11, 13, 14, abstract and TODO.md updated; ledger Part K (K1-K14); tables `t_final2_pw`, `t_final2_pw_perlang`; `t_longctx_*` and figs 14-16 include the final2 row. New files: `paper/scripts/box_g1_pairwise_final2.py`, `box_g1_lc_paired.py`, `paper/data/final2_eval/g1_pairwise_final2_{summary,per_language}.csv`, `g1_identical_hypothesis_check.txt`, `reports/data/longctx/final2_paired_vs_other.csv`, addendum in `reports/long_context_benchmark_2026-10-23.md`. Builds: paper.html/pdf, `paper/overleaf_paper_2026-10-24.zip`, archive re-synced.
