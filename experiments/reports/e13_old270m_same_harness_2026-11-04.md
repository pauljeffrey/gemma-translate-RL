# E-13: the first research's 270M models re-run in the 385M harness (2026-11-04)

> Parallel write-up of the same analysis: `reports/e13_old_models_same_harness_2026-11-04.md` (launcher); this file is the source cited by `paper_385m/paper.md`. All shared numbers agree; differences are listed in the final consistency report.

Question (paper_385m/EXPERIMENTS_LEFT.md E-13; ledger S7, R16): the earlier comparison of the 385M models with the first research's 270M models used old numbers from an earlier code state, without intervals. Here both old models were generated again with the same harness, flags and metrics as every 385M model, and compared on identical rows with a paired bootstrap. CPU analysis on the box, copied to the laptop (`research_archive_2026-11-03/box_final_2026-11-04/shm_results/evid_e13/out/evidence_e13.json`); data, tables and scripts: `reports/data/e13_old270m_same_harness_2026-11-04/` (`evidence_e13.json`, `tables.md` made by `make_tables.py`, `e13_analyse.sh`, `e13_analyse.out`, `old_sft.log`, `old_rl2c.log`; the `.npz` per-sentence statistics stay in the archive). Every number below is read from `evidence_e13.json` (or the logs named).

## 1. What was run

* **Models.** Old 270M SFT: `/dev/shm/model_old_sft` (the report header of the evaluation says training step 27,638, macro validation loss 1.2654; the Hub repository is `<HF_ORG>/gemma3-270m-translate` (main) according to the parallel write-up `reports/e13_old_models_same_harness_2026-11-04.md`, not named in the logs I read). Old 270M RL-final ("rl2c"): `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c`, revision `latest`, training step 1,433. 385M SFT = `<HF_ORG>/gemma-0.4B-translate-sft`; 385M RL = the mixed best (Hub branch `main` of `gemma-0.4B-translate-rl`); their predictions are those of the earlier 385M evaluation.
* **Generation.** `eval_sft_benchmarks` of the 385M harness (greedy, 768 new tokens, vLLM bf16, prompt `Translate the text to {language}:  {english}`), FLORES+ devtest, NTREX and MAFAND, 114,489 rows per old model (46 / 24 / 15 languages); SFT run 09:22-09:47 UTC (1,498 s), rl2c 09:47 UTC onward, on the A100 (earlier attempts of both failed on the noexec Triton cache and were rerun, see the overnight log). Old SFT finished 0.9263 / loop 0.082 and rl2c finished 0.9997 / loop 0.0023 over all rows (`old_sft.log`, `old_rl2c.log`).
* **Rows.** The analysis (`e13_analyse.sh` -> `ev_analyse2.py ... --tag _e13`) keeps only rows that are clean for BOTH generations: the union of the 385M contamination audit's flags and the first research's own flags (exact-sentence keys). 113,477 rows of the 49 selected languages (`afr amh bam bem dyu ewe fon hau ibo kab kam kbp kea kik kin kmb knc lin lua lug luo mlg mos nde nso nus nya orm pcm run sag shi sna som sot ssw swh tir tsn tso tum twi umb ven vmw wol xho yor zul`) were considered and **67,139 were removed** (FLORES+ 43,380, NTREX 19,248, MAFAND 4,511). What remains: FLORES+ 2,160 rows (45 languages x 48 sentences; 4.7% of the 45,540 rows of those languages, because the old models saw the English side of about 95% of FLORES+ devtest), NTREX 28,680 (24 languages), MAFAND 15,498 (15 languages); 84 benchmark-language cells.
* **Statistics.** Language-macro chrF++ and chrF, held-out AfriCOMET-STL (cached scores, 89.5% coverage for the old models), finished and loop rates (the reward's flag) per system; paired bootstrap, 1,000 resamples, Poisson weights per source sentence shared by the two systems, seed 20261104, 95% percentile intervals (`meta` of the json); same method as `reports/rl_vs_sft_evidence_2026-11-04.md`. Pairs: old SFT vs 385M SFT, old RL vs 385M RL, 385M SFT vs 385M RL, old SFT vs old RL.

## 2. Results (all from `evidence_e13.json`; language macro; 95% CI)

chrF++ on rows clean for both generations:

| benchmark (languages) | old 270M SFT | 385M SFT | 385M SFT - old SFT | old 270M RL (rl2c) | 385M RL (mixed best) | 385M RL - old RL |
|---|---:|---:|---|---:|---:|---|
| FLORES+ devtest (45) | 24.69 | 22.34 | -2.35 [-3.25, -1.38] | 31.68 | 32.03 | +0.35 [+0.08, +0.62] |
| NTREX (24) | 31.46 | 27.33 | -4.13 [-4.40, -3.85] | 35.52 | 35.55 | +0.02 [-0.07, +0.11] |
| MAFAND (15) | 38.26 | 34.71 | -3.54 [-4.22, -2.75] | 41.80 | 40.25 | -1.55 [-1.72, -1.39] |

Held-out reference-based AfriCOMET-STL (languages with a score: 40 / 21 / 15): 385M SFT minus old SFT -0.0113 [-0.0189, -0.0037] (FLORES+), -0.0250 [-0.0273, -0.0229] (NTREX), -0.0158 [-0.0188, -0.0130] (MAFAND); 385M RL minus old RL -0.0051 [-0.0104, -0.0001], -0.0129 [-0.0144, -0.0115], -0.0203 [-0.0222, -0.0182]. Plain chrF gives the same signs as chrF++ in every cell (`tables.md`).

Same rows, 385M RL minus 385M SFT: FLORES+ +9.69 [+8.51, +10.69] (22.34 -> 32.03), NTREX +8.22 [+7.95, +8.50], MAFAND +5.54 [+4.86, +6.10]. The old models' own RL gain on these rows: old SFT -> old RL +6.99 [+6.08, +7.69] (FLORES+), +4.06 [+3.83, +4.30] (NTREX), +3.54 [+2.91, +4.09] (MAFAND), +5.54 [+5.03, +5.91] over all 84 cells (`macro.all`).

Finished / loop (reward's flag) / hypothesis-to-reference length, same rows: old SFT FLORES+ 0.850 / 0.171 / 2.53, NTREX 0.922 / 0.083 / 1.78, MAFAND 0.974 / 0.029 / 1.38; 385M SFT 0.852 / 0.196 / 2.61, 0.876 / 0.137 / 2.50, 0.961 / 0.044 / 1.59; old RL 1.000 / 0.002 / 1.06, 0.999 / 0.002 / 1.04, 1.000 / 0.000 / 1.00; 385M RL 1.000 / 0.017 / 1.04, 0.999 / 0.007 / 1.02, 1.000 / 0.001 / 0.99.

## 3. Reading

* In the same harness on rows clean for both, the 385M SFT is **behind the old 270M SFT on all three benchmarks** in chrF++ (-2.35, -4.13, -3.54) and in AfriCOMET; after RL the 385M model is **at the old RL's level in chrF++ on FLORES+ (+0.35 [+0.08, +0.62]) and NTREX (+0.02 [-0.07, +0.11]), 1.55 chrF++ below it on MAFAND, and lower on held-out AfriCOMET on all three benchmarks** (-0.0051, -0.0129, -0.0203). Growth to 385M therefore did not give a clear win over the first research's model on sentence-level benchmarks. The earlier stored numbers (ledger S7, R16: old SFT 25.06 / 31.46 / 38.18, old RL 31.68 / 35.55 / 41.84 on the old-flag clean rows) agree with the rerun to within 0.4 chrF++ for the old models (old SFT FLORES+ 24.69 here against 25.06), arithmetic of the two sets of reported numbers.
* The first research's own RL gain on these rows is smaller than the 385M's (+5.54 over 84 cells against +9.69 / +8.22 / +5.54 on the three benchmarks for the 385M), but the old SFT was a better starting point (see the rows above); the two gains are on the same rows but are not a comparison of "RL methods", since data, steps, tokenizer and reward differ.

## 4. Caveats

1. **Row filter.** Clean for both generations removes 59% of the rows overall and **95.3% of FLORES+** (2,160 of 45,540 rows remain, 48 sentences per language), so the FLORES+ column rests on 48 sentences per language and its per-language values are noisy (per-language cells are in `evidence_e13.json`, `cells`); NTREX and MAFAND keep 59.8% and 77.5% of their rows (arithmetic of the counts above). The 385M numbers here (e.g. mixed best NTREX 35.55) are therefore on a different row set from the 385M numbers of `rl_vs_sft_evidence` (37.17 on all clean rows) and must not be mixed with them.
2. **Composite comparison.** The two generations differ in data, number of steps, tokenizer layout (text tags against single-token tags), reward and code state; this is a re-measurement under one harness, not a controlled ablation. One seed per model; the intervals are over test sentences (Poisson weights), not training runs.
3. **Languages.** 45 / 24 / 15 languages (49 distinct in the list); benchmarks other than FLORES+, NTREX and MAFAND (SmolSent, SALT, GlotStoryBook, IbomMT, FLORES-MOZ, AfriScience-MT) were not re-run for the old models, so the earlier "7 of 8 benchmarks" statement (S7) is not re-measured for those five.
4. **Source of the old SFT weights** is named only in the parallel report `reports/e13_old_models_same_harness_2026-11-04.md` (see section 1); that report and its data `reports/data/e13_old_models_2026-11-04/` cover the same analysis.
5. AfriCOMET is the QE reward member's family (held-out checkpoint); chrF++ and chrF are the lexical evidence.
