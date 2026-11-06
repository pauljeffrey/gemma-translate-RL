# Does RL beat SFT? Evidence from the stored evaluations (2026-11-04)

Question (owner): how much of the RL gain over the SFT is real translation improvement, and how much is the RL model finishing its sentences and not looping? Everything here was computed on the CPUs of the A100 box (`nice 19`, no GPU) from predictions and AfriCOMET scores that were already stored; no model was run, the science RL was not touched. Scripts and raw output: `reports/data/rl_vs_sft_evidence_2026-11-04/` (`ev_stats.py` per-sentence statistics, `ev_analyse.py` paired bootstrap, `ev_extra.py` retention and detector floor, `build_evidence_tables.py`, `evidence.json`, `extra.json`, `tables.md`). Section 5 (science) was filled in on 2026-11-04 after the science RL ended.

## 0. Short answer

* **The RL model is much better than the SFT on every benchmark, and the difference is far outside sampling noise.** FLORES+ devtest chrF++ (54 languages, language macro) 22.85 -> 31.66, **+8.81 [+8.59, +9.03]**; NTREX (24) +7.91 [+7.69, +8.12]; MAFAND (15) +5.01 [+4.76, +5.23]; all 93 benchmark-language cells **+7.96 [+7.82, +8.12]**. Plain chrF: +8.59 [+8.43, +8.76]. The independent neural check (held-out `masakhane/africomet-stl-1.1`, reference-based, 76 cells): 0.5653 -> 0.6284, **+0.0631 [+0.0620, +0.0643]**. All 54 FLORES+ languages and all 24 NTREX languages improve on all three metrics with confidence intervals above zero; MAFAND: 14 of 15 (pcm: -0.25 [-0.69, +0.30], not different from zero).
* **A large part of this is finishing and looping, not translation quality.** On FLORES+ the SFT finishes 82.9 % of the sentences and the loop detector flags 20.9 % of its outputs (RL: 99.5 % and 4.8 %; the same detector flags 0.01 % of the human references, so the RL's 4.8 % are real repetition; 95 % of the RL's flagged FLORES+ rows come from 7 near-failed languages: taq, knc, zgh, dik, aeb, arz, ary). SFT outputs are 3.0 times as long as the reference (RL 1.14).
* **Real translation improvement exists and is smaller: +2.4 to +4.1 chrF++ and +0.037 AfriCOMET.** Cutting the SFT's looping outputs at the loop onset (no translation change at all) already gives the SFT +3.9 chrF++ (26.59 -> 30.49), i.e. **49 % of the chrF++ gain is recovered by cutting loops**; the RL then still wins by **+4.06 [+3.97, +4.15]**. On the rows where the SFT output is itself finished and loop-free the RL still wins by **+2.41 [+2.33, +2.49] chrF++** and **+0.0366 [+0.0357, +0.0374] AfriCOMET** (59 % of the neural gain), and by +2.61 [+2.54, +2.70] / +0.0370 when the RL's own failures are removed too. In 50 of 54 FLORES+ languages the RL is significantly better on those clean rows.
* **Where it does not help beyond finishing:** the Arabic dialects aeb, ary (and arz when the SFT is cut at its loop) and Tamazight zgh (FLORES+, chrF++ 2 to 9 for both models, the RL still loops in 11 to 46 % of them), and on MAFAND hau, ibo, pcm (-0.3 to -0.5 chrF++ on clean rows). The 8 languages added after the A0 set show **no demonstrated translation gain beyond finishing** (-0.13 [-0.54, +0.27]); their whole gain is finishing.
* **Nearly all of the gain comes from stage A** (SFT -> stage A +7.96 of the +8.50 to the B1 best on FLORES+; B1 adds +0.54, mixed +0.31, both ~0 on MAFAND).
* **What this cannot establish** (section 6): whether an equally long continued SFT, or a decoding fix, would give most of the same gain (no control was run), seed-to-seed variation (one run), and independence from the reward for everything that is a QE or chrF++-agreement metric. Estimates for the controls are in section 5.

## 1. What was compared and how

Systems (same prompts, greedy, 768 new tokens, vLLM, the 385M evaluation harness of `reports/eval_post_B1_385m_2026-11-03.md`): `sft` = the 385M SFT final (`g15_385m_sft_final`); `stage A` = final of stage A (400 steps); `B1 best` (stage B1, 135 steps run, best checkpoint by dev); `mixed best` = the final RL model (stage mixed, 60 steps, best checkpoint by dev). "RL" below is the mixed best unless stated; table 3 shows the three stages. Benchmarks: FLORES+ devtest (54 languages x 1,012 rows), NTREX (24 x ~2,000), MAFAND (15 x ~1,500), 122,585 rows in all, all English -> African.

Rows removed everywhere: 214 rows whose English side is flagged by the 385M SFT contamination audit (`reports/contamination_audit_385m_2026-11-03.md`; exact / sentence / near hit): FLORES+ 0, NTREX 120, MAFAND 94 (one- and two-word items). Nothing else was filtered. (The earlier statement in this session that the 385M audit was missing was wrong; it exists and its flags are used here.)

Scoring: sacrebleu chrF++ (word order 2) and plain chrF (word order 0) recomputed from per-sentence n-gram statistics (NFC); the recomputed SFT / stage A / B1 FLORES+ macros (22.85, 30.81, 31.35) equal the numbers already reported, which checks the pipeline. AfriCOMET = `masakhane/africomet-stl-1.1` per-sentence scores from the stored caches (joined by the cache key; coverage 83.6 % of rows, i.e. the 76 benchmark-language cells of AfriCOMET-supported languages). Macro = mean over benchmark-language cells. Intervals: 95 % percentile intervals of a paired bootstrap, 1,000 resamples, Poisson(1) weights per source sentence shared by the two systems and (for FLORES+ and NTREX, which are parallel) by all languages, so the macro intervals include the cross-language correlation of the same English sentence. They quantify test-sentence sampling only, not training-run variation (section 6).

### Which metrics are in the reward family

| metric | in the RL reward? | use here |
|---|---|---|
| `masakhane/africomet-qe-stl-1.1` (reference-free) | yes (QE ensemble member) | **excluded** (not computed) |
| `McGill-NLP/ssa-comet-mtl` (and SSA-COMET-QE, same model family and training data) | yes (QE ensemble member; in the science RL also reference-based) | **excluded** from the RL-vs-SFT evidence; SSA-COMET-QE appears only as a hacking monitor in the science reports |
| consensus utility (chrF++ agreement among the 16 samples of a prompt, no reference) | yes, all stages | not a reference score; but it optimises chrF++-style agreement |
| NLLB-3.3B pseudo-reference chrF++ (reward only, CC-BY-NC) | yes, 5 languages (kea kbp nus bam kab), calibrated weight | chrF++ of those 5 languages is not independent of the reward |
| human-reference chrF++ term (weight 0.35) | yes, B1 and mixed only, 15 % of prompts: 7 languages (amh efi hau lug nso yor zul), source AfriScience-MT **train** pairs | the benchmarks here are different data, so chrF / chrF++ on FLORES+, NTREX, MAFAND is not a reward input, but the language-level chrF++ gain on those 7 languages can be partly trained |
| guards and penalties (language ID, copy, repetition, length, unfinished = hard failure -2) | yes | the finished / loop rates are optimised by construction; the loop flag here is the same detector as the reward guard (`is_repetitive`, `is_self_repetitive`), see section 6 |
| `masakhane/africomet-stl-1.1` (reference-based) | **no**: held out since patch 3 and used only as a monitor and in the hacking alarm; it never entered the reward | **the independent neural check**. Caveat: it shares architecture and training data (AfriMTE) with the QE reward member, so it is independent in weights and in use of the reference, not in family |
| chrF / chrF++ against the human references of FLORES+, NTREX, MAFAND | no | the main lexical evidence; checkpoint choice used a separate dev set (720 sentences in 16 languages in stage A; the dev macro of B1 and mixed), never these test sets |

## 2. Gain of RL over SFT per language and macro (task 1)

#### Table 1. Macro gain of the final RL model over SFT (mean over languages of corpus scores; 95 % paired bootstrap CI, 1000 resamples)

### chrF++

| rows | benchmark | languages | SFT | RL | RL - SFT [95 % CI] |
|---|---|---:|---:|---:|---|
| all clean rows | FLORES+ devtest | 54 | 22.85 | 31.66 | +8.81 [+8.59, +9.03] |
| all clean rows | NTREX | 24 | 29.25 | 37.17 | +7.91 [+7.69, +8.12] |
| all clean rows | MAFAND | 15 | 35.77 | 40.78 | +5.01 [+4.76, +5.23] |
| all clean rows | all 93 benchmark-language cells | 93 | 26.59 | 34.55 | +7.96 [+7.82, +8.12] |
| only rows where BOTH models finished and neither loops | FLORES+ devtest | 54 | 29.74 | 32.52 | +2.78 [+2.66, +2.90] |
| only rows where BOTH models finished and neither loops | NTREX | 24 | 34.73 | 37.43 | +2.70 [+2.62, +2.79] |
| only rows where BOTH models finished and neither loops | MAFAND | 15 | 39.03 | 40.93 | +1.89 [+1.78, +2.00] |
| only rows where BOTH models finished and neither loops | all 93 benchmark-language cells | 93 | 32.53 | 35.14 | +2.61 [+2.54, +2.70] |
| SFT output cut at its first loop (RL as is) | FLORES+ devtest | 54 | 27.14 | 31.66 | +4.51 [+4.37, +4.66] |
| SFT output cut at its first loop (RL as is) | NTREX | 24 | 33.12 | 37.17 | +4.04 [+3.93, +4.16] |
| SFT output cut at its first loop (RL as is) | MAFAND | 15 | 38.33 | 40.78 | +2.45 [+2.33, +2.57] |
| SFT output cut at its first loop (RL as is) | all 93 benchmark-language cells | 93 | 30.49 | 34.55 | +4.06 [+3.97, +4.15] |
| both outputs cut at their first loop | FLORES+ devtest | 54 | 27.14 | 31.85 | +4.70 [+4.56, +4.85] |
| both outputs cut at their first loop | NTREX | 24 | 33.12 | 37.21 | +4.09 [+3.97, +4.20] |
| both outputs cut at their first loop | MAFAND | 15 | 38.33 | 40.79 | +2.47 [+2.34, +2.58] |
| both outputs cut at their first loop | all 93 benchmark-language cells | 93 | 30.49 | 34.67 | +4.18 [+4.10, +4.28] |
| only rows where SFT did not finish or looped | FLORES+ devtest | 49 | 5.71 | 28.39 | +22.69 [+22.28, +23.08] |
| only rows where SFT did not finish or looped | NTREX | 23 | 6.02 | 33.49 | +27.48 [+26.77, +28.11] |
| only rows where SFT did not finish or looped | MAFAND | 9 | 7.12 | 34.55 | +27.43 [+26.65, +28.19] |
| only rows where SFT did not finish or looped | all 93 benchmark-language cells | 81 | 5.95 | 30.53 | +24.57 [+24.26, +24.90] |
| only rows where SFT finished and did not loop | FLORES+ devtest | 54 | 29.69 | 32.14 | +2.45 [+2.33, +2.59] |
| only rows where SFT finished and did not loop | NTREX | 24 | 34.73 | 37.38 | +2.65 [+2.57, +2.74] |
| only rows where SFT finished and did not loop | MAFAND | 15 | 39.03 | 40.90 | +1.87 [+1.76, +1.98] |
| only rows where SFT finished and did not loop | all 93 benchmark-language cells | 93 | 32.50 | 34.91 | +2.41 [+2.33, +2.49] |

### chrF (word order 0)

| rows | benchmark | languages | SFT | RL | RL - SFT [95 % CI] |
|---|---|---:|---:|---:|---|
| all clean rows | FLORES+ devtest | 54 | 25.00 | 34.50 | +9.51 [+9.27, +9.74] |
| all clean rows | NTREX | 24 | 31.97 | 40.51 | +8.54 [+8.30, +8.76] |
| all clean rows | MAFAND | 15 | 39.00 | 44.36 | +5.36 [+5.10, +5.61] |
| all clean rows | all 93 benchmark-language cells | 93 | 29.06 | 37.65 | +8.59 [+8.43, +8.76] |
| only rows where BOTH models finished and neither loops | FLORES+ devtest | 54 | 32.43 | 35.39 | +2.96 [+2.84, +3.09] |
| only rows where BOTH models finished and neither loops | NTREX | 24 | 37.92 | 40.76 | +2.84 [+2.76, +2.93] |
| only rows where BOTH models finished and neither loops | MAFAND | 15 | 42.52 | 44.51 | +1.99 [+1.88, +2.10] |
| only rows where BOTH models finished and neither loops | all 93 benchmark-language cells | 93 | 35.48 | 38.25 | +2.77 [+2.70, +2.86] |
| SFT output cut at its first loop (RL as is) | FLORES+ devtest | 54 | 29.63 | 34.50 | +4.87 [+4.73, +5.03] |
| SFT output cut at its first loop (RL as is) | NTREX | 24 | 36.19 | 40.51 | +4.33 [+4.21, +4.45] |
| SFT output cut at its first loop (RL as is) | MAFAND | 15 | 41.76 | 44.36 | +2.60 [+2.47, +2.72] |
| SFT output cut at its first loop (RL as is) | all 93 benchmark-language cells | 93 | 33.28 | 37.65 | +4.37 [+4.28, +4.47] |
| both outputs cut at their first loop | FLORES+ devtest | 54 | 29.63 | 34.70 | +5.07 [+4.92, +5.22] |
| both outputs cut at their first loop | NTREX | 24 | 36.19 | 40.56 | +4.37 [+4.25, +4.50] |
| both outputs cut at their first loop | MAFAND | 15 | 41.76 | 44.38 | +2.62 [+2.49, +2.74] |
| both outputs cut at their first loop | all 93 benchmark-language cells | 93 | 33.28 | 37.77 | +4.49 [+4.40, +4.59] |
| only rows where SFT did not finish or looped | FLORES+ devtest | 49 | 6.60 | 31.13 | +24.52 [+24.13, +24.92] |
| only rows where SFT did not finish or looped | NTREX | 23 | 7.01 | 37.05 | +30.04 [+29.30, +30.72] |
| only rows where SFT did not finish or looped | MAFAND | 9 | 8.23 | 38.03 | +29.80 [+28.98, +30.62] |
| only rows where SFT did not finish or looped | all 93 benchmark-language cells | 81 | 6.90 | 33.57 | +26.67 [+26.34, +27.00] |
| only rows where SFT finished and did not loop | FLORES+ devtest | 54 | 32.38 | 35.00 | +2.62 [+2.48, +2.76] |
| only rows where SFT finished and did not loop | NTREX | 24 | 37.92 | 40.71 | +2.79 [+2.71, +2.88] |
| only rows where SFT finished and did not loop | MAFAND | 15 | 42.52 | 44.48 | +1.96 [+1.85, +2.07] |
| only rows where SFT finished and did not loop | all 93 benchmark-language cells | 93 | 35.45 | 38.00 | +2.56 [+2.47, +2.65] |

### AfriCOMET-STL-1.1 (reference-based, held out)

| rows | benchmark | languages | SFT | RL | RL - SFT [95 % CI] |
|---|---|---:|---:|---:|---|
| all clean rows | FLORES+ devtest | 40 | 0.5538 | 0.6267 | +0.0728 [+0.0710, +0.0750] |
| all clean rows | NTREX | 21 | 0.5510 | 0.6139 | +0.0629 [+0.0613, +0.0645] |
| all clean rows | MAFAND | 15 | 0.6159 | 0.6531 | +0.0372 [+0.0356, +0.0388] |
| all clean rows | all 93 benchmark-language cells | 76 | 0.5653 | 0.6284 | +0.0631 [+0.0620, +0.0643] |
| only rows where BOTH models finished and neither loops | FLORES+ devtest | 40 | 0.5902 | 0.6316 | +0.0414 [+0.0400, +0.0427] |
| only rows where BOTH models finished and neither loops | NTREX | 21 | 0.5822 | 0.6202 | +0.0380 [+0.0366, +0.0393] |
| only rows where BOTH models finished and neither loops | MAFAND | 15 | 0.6311 | 0.6553 | +0.0242 [+0.0228, +0.0256] |
| only rows where BOTH models finished and neither loops | all 93 benchmark-language cells | 76 | 0.5961 | 0.6331 | +0.0370 [+0.0362, +0.0379] |
| only rows where SFT did not finish or looped | FLORES+ devtest | 35 | 0.2566 | 0.5843 | +0.3277 [+0.3213, +0.3338] |
| only rows where SFT did not finish or looped | NTREX | 20 | 0.2118 | 0.5485 | +0.3367 [+0.3274, +0.3465] |
| only rows where SFT did not finish or looped | MAFAND | 9 | 0.2491 | 0.6031 | +0.3540 [+0.3431, +0.3655] |
| only rows where SFT did not finish or looped | all 93 benchmark-language cells | 64 | 0.2416 | 0.5757 | +0.3342 [+0.3295, +0.3391] |
| only rows where SFT finished and did not loop | FLORES+ devtest | 40 | 0.5902 | 0.6310 | +0.0408 [+0.0395, +0.0422] |
| only rows where SFT finished and did not loop | NTREX | 21 | 0.5822 | 0.6197 | +0.0376 [+0.0362, +0.0389] |
| only rows where SFT finished and did not loop | MAFAND | 15 | 0.6311 | 0.6550 | +0.0239 [+0.0226, +0.0253] |
| only rows where SFT finished and did not loop | all 93 benchmark-language cells | 76 | 0.5960 | 0.6326 | +0.0366 [+0.0357, +0.0374] |

Per language (FLORES+ devtest, sorted by gain; "paired rows" = both models finished and neither loops; the last column cuts the SFT at its loop and leaves the RL as is):

#### Table 6. Per language, FLORES+ devtest (54 languages; chrF++ unless stated)

| iso | family | weak list | SFT | RL | Δ all rows [CI] | Δ chrF [CI] | Δ AfriCOMET [CI] | SFT finished / loop | RL finished / loop | rows kept (paired) | Δ paired rows [CI] | Δ SFT cut at loop [CI] |
|---|---|---|---:|---:|---|---|---|---|---|---:|---|---|
| kea | Creole / Pidgin | yes | 18.91 | 39.18 | +20.27 [+18.91, +21.53] | +20.53 [+19.13, +21.81] | - | 0.83 / 0.20 | 1.00 / 0.00 | 0.80 | +7.66 [+6.93, +8.42] | +11.80 [+10.83, +12.78] |
| orm | Afroasiatic |  | 12.06 | 30.18 | +18.12 [+17.49, +18.76] | +20.90 [+20.13, +21.64] | +0.191 [+0.181, +0.202] | 0.59 / 0.45 | 1.00 / 0.00 | 0.55 | +2.99 [+2.51, +3.50] | +8.05 [+7.45, +8.64] |
| nso | Bantu |  | 31.58 | 49.01 | +17.43 [+15.75, +19.09] | +17.98 [+16.25, +19.66] | +0.103 [+0.092, +0.113] | 0.86 / 0.15 | 1.00 / 0.00 | 0.85 | +4.22 [+3.62, +4.91] | +7.47 [+6.58, +8.35] |
| lua | Bantu |  | 17.18 | 33.37 | +16.19 [+15.21, +17.13] | +17.57 [+16.50, +18.58] | +0.126 [+0.117, +0.134] | 0.85 / 0.25 | 1.00 / 0.00 | 0.75 | +5.45 [+4.94, +6.03] | +8.59 [+7.98, +9.28] |
| lin | Bantu |  | 28.56 | 44.35 | +15.78 [+14.19, +17.22] | +16.34 [+14.66, +17.84] | +0.097 [+0.088, +0.107] | 0.90 / 0.13 | 1.00 / 0.00 | 0.86 | +5.27 [+4.69, +5.84] | +7.73 [+7.04, +8.44] |
| ssw | Bantu |  | 22.20 | 37.86 | +15.66 [+14.46, +16.85] | +17.72 [+16.34, +19.05] | +0.113 [+0.104, +0.123] | 0.87 / 0.17 | 1.00 / 0.00 | 0.83 | +5.38 [+4.85, +5.93] | +8.71 [+8.03, +9.44] |
| lug | Bantu |  | 16.17 | 31.73 | +15.57 [+14.66, +16.52] | +17.29 [+16.27, +18.36] | +0.112 [+0.104, +0.121] | 0.79 / 0.25 | 1.00 / 0.01 | 0.74 | +5.32 [+4.78, +5.87] | +7.93 [+7.23, +8.67] |
| sag | Atlantic-Congo (non-Bantu) |  | 19.68 | 34.31 | +14.62 [+13.64, +15.55] | +15.55 [+14.51, +16.54] | - | 0.87 / 0.20 | 1.00 / 0.00 | 0.80 | +5.99 [+5.51, +6.47] | +8.09 [+7.50, +8.64] |
| bem | Bantu |  | 19.31 | 33.88 | +14.56 [+13.63, +15.61] | +15.73 [+14.71, +16.86] | +0.111 [+0.103, +0.121] | 0.88 / 0.18 | 1.00 / 0.01 | 0.81 | +7.16 [+6.60, +7.75] | +9.14 [+8.49, +9.87] |
| tso | Bantu |  | 32.92 | 46.86 | +13.93 [+12.43, +15.51] | +14.54 [+12.97, +16.19] | +0.070 [+0.062, +0.079] | 0.91 / 0.11 | 1.00 / 0.00 | 0.89 | +4.45 [+3.82, +5.07] | +7.02 [+6.27, +7.85] |
| luo | Nilo-Saharan | yes | 21.36 | 34.49 | +13.13 [+11.91, +14.15] | +13.82 [+12.51, +14.92] | +0.101 [+0.092, +0.109] | 0.92 / 0.13 | 1.00 / 0.00 | 0.87 | +6.44 [+5.88, +6.95] | +8.16 [+7.56, +8.76] |
| kin | Bantu |  | 25.81 | 38.66 | +12.86 [+11.66, +14.03] | +13.94 [+12.61, +15.25] | +0.092 [+0.083, +0.101] | 0.89 / 0.12 | 1.00 / 0.00 | 0.88 | +5.66 [+5.09, +6.21] | +6.85 [+6.22, +7.46] |
| cjk | family unlabelled |  | 10.72 | 23.02 | +12.31 [+11.70, +12.89] | +14.18 [+13.50, +14.85] | - | 0.71 / 0.31 | 1.00 / 0.01 | 0.68 | +2.13 [+1.86, +2.42] | +5.02 [+4.60, +5.50] |
| run | Bantu |  | 21.59 | 33.88 | +12.29 [+11.21, +13.15] | +13.66 [+12.50, +14.65] | +0.107 [+0.098, +0.116] | 0.88 / 0.15 | 1.00 / 0.01 | 0.85 | +5.03 [+4.52, +5.57] | +6.79 [+6.19, +7.38] |
| nus | Nilo-Saharan | yes | 13.49 | 24.82 | +11.32 [+10.73, +11.91] | +12.16 [+11.51, +12.77] | +0.150 [+0.140, +0.159] | 0.67 / 0.41 | 1.00 / 0.00 | 0.59 | +1.97 [+1.56, +2.36] | +5.47 [+4.99, +5.95] |
| umb | Bantu | yes | 14.80 | 25.46 | +10.66 [+9.77, +11.44] | +12.21 [+11.17, +13.15] | +0.080 [+0.072, +0.087] | 0.89 / 0.15 | 1.00 / 0.01 | 0.84 | +4.84 [+4.43, +5.26] | +6.17 [+5.73, +6.62] |
| kmb | Bantu | yes | 16.02 | 26.62 | +10.61 [+9.82, +11.37] | +11.70 [+10.84, +12.56] | +0.081 [+0.074, +0.087] | 0.88 / 0.23 | 1.00 / 0.00 | 0.77 | +3.93 [+3.56, +4.29] | +6.21 [+5.73, +6.67] |
| knc | Nilo-Saharan |  | 3.81 | 14.22 | +10.41 [+9.95, +10.88] | +11.96 [+11.45, +12.48] | - | 0.46 / 0.94 | 1.00 / 0.48 | 0.04 | +2.49 [+0.82, +3.99] | +7.68 [+7.17, +8.18] |
| kbp | Atlantic-Congo (non-Bantu) | yes | 14.51 | 24.88 | +10.37 [+9.70, +11.00] | +11.03 [+10.32, +11.69] | +0.123 [+0.115, +0.131] | 0.85 / 0.26 | 1.00 / 0.01 | 0.74 | +5.48 [+4.98, +5.94] | +7.77 [+7.20, +8.35] |
| kab | Afroasiatic | yes | 16.35 | 26.69 | +10.34 [+9.50, +11.14] | +10.80 [+9.94, +11.62] | +0.116 [+0.107, +0.126] | 0.84 / 0.20 | 1.00 / 0.00 | 0.79 | +2.26 [+1.92, +2.61] | +4.74 [+4.24, +5.24] |
| dik | family unlabelled |  | 3.76 | 14.06 | +10.30 [+9.75, +10.81] | +11.37 [+10.80, +11.94] | - | 0.29 / 0.85 | 1.00 / 0.40 | 0.13 | +1.80 [+1.10, +2.60] | +5.02 [+4.49, +5.58] |
| amh | Afroasiatic | yes | 16.62 | 26.62 | +10.00 [+9.04, +10.96] | +10.77 [+9.72, +11.82] | +0.088 [+0.078, +0.099] | 0.87 / 0.13 | 1.00 / 0.00 | 0.87 | +2.80 [+2.29, +3.29] | +4.29 [+3.70, +4.86] |
| vmw | Bantu | yes | 17.45 | 27.14 | +9.69 [+8.92, +10.48] | +10.87 [+10.02, +11.74] | - | 0.84 / 0.18 | 1.00 / 0.00 | 0.82 | +2.11 [+1.81, +2.44] | +4.37 [+3.92, +4.87] |
| mfe | family unlabelled |  | 28.51 | 37.93 | +9.42 [+7.98, +10.94] | +9.57 [+8.10, +11.10] | - | 0.94 / 0.07 | 1.00 / 0.01 | 0.93 | +3.91 [+3.43, +4.37] | +3.94 [+3.18, +4.61] |
| dyu | Mande | yes | 9.90 | 18.97 | +9.06 [+8.47, +9.62] | +10.27 [+9.62, +10.88] | - | 0.83 / 0.25 | 1.00 / 0.00 | 0.75 | +4.11 [+3.80, +4.44] | +7.78 [+7.21, +8.31] |
| yor | Atlantic-Congo (non-Bantu) | yes | 21.32 | 30.33 | +9.01 [+7.99, +10.00] | +9.02 [+7.99, +10.04] | +0.118 [+0.108, +0.128] | 0.95 / 0.05 | 1.00 / 0.00 | 0.95 | +5.83 [+5.09, +6.56] | +6.41 [+5.66, +7.14] |
| tum | Bantu | yes | 25.40 | 34.27 | +8.87 [+7.89, +9.79] | +9.56 [+8.49, +10.58] | +0.073 [+0.064, +0.081] | 0.94 / 0.08 | 1.00 / 0.00 | 0.92 | +4.57 [+4.08, +5.06] | +5.56 [+5.04, +6.10] |
| kam | Bantu | yes | 17.22 | 25.92 | +8.70 [+7.93, +9.50] | +9.19 [+8.35, +10.06] | +0.063 [+0.057, +0.069] | 0.92 / 0.14 | 1.00 / 0.00 | 0.85 | +4.42 [+4.01, +4.83] | +6.01 [+5.49, +6.53] |
| mos | Atlantic-Congo (non-Bantu) | yes | 14.09 | 22.79 | +8.70 [+8.13, +9.33] | +9.41 [+8.80, +10.05] | +0.088 [+0.081, +0.095] | 0.90 / 0.19 | 1.00 / 0.01 | 0.80 | +4.54 [+4.17, +4.89] | +6.16 [+5.72, +6.63] |
| tir | Afroasiatic | yes | 10.48 | 18.74 | +8.26 [+7.69, +8.86] | +9.01 [+8.39, +9.66] | +0.144 [+0.134, +0.155] | 0.77 / 0.23 | 1.00 / 0.00 | 0.77 | +2.23 [+1.83, +2.66] | +3.81 [+3.38, +4.25] |
| taq | family unlabelled |  | 2.81 | 11.06 | +8.25 [+7.82, +8.67] | +9.14 [+8.68, +9.60] | - | 0.14 / 0.91 | 0.99 / 0.56 | 0.07 | +3.27 [+1.81, +4.58] | +5.52 [+5.05, +5.99] |
| seh | Bantu | yes | 21.67 | 29.65 | +7.98 [+7.00, +8.86] | +9.04 [+7.93, +10.04] | - | 0.92 / 0.12 | 1.00 / 0.00 | 0.88 | +2.01 [+1.70, +2.33] | +3.38 [+2.93, +3.84] |
| bam | Mande | yes | 19.39 | 27.17 | +7.78 [+6.96, +8.59] | +8.34 [+7.50, +9.21] | +0.069 [+0.062, +0.076] | 0.89 / 0.13 | 1.00 / 0.00 | 0.86 | +2.47 [+2.14, +2.80] | +4.11 [+3.66, +4.58] |
| twi | Atlantic-Congo (non-Bantu) |  | 27.54 | 35.07 | +7.53 [+6.47, +8.61] | +7.81 [+6.72, +8.90] | +0.054 [+0.047, +0.061] | 0.94 / 0.08 | 1.00 / 0.00 | 0.92 | +2.47 [+2.02, +2.92] | +3.67 [+3.12, +4.23] |
| tsn | Bantu |  | 37.69 | 45.05 | +7.36 [+6.18, +8.65] | +7.59 [+6.35, +8.93] | +0.040 [+0.034, +0.048] | 0.96 / 0.06 | 1.00 / 0.00 | 0.94 | +2.09 [+1.67, +2.59] | +3.05 [+2.56, +3.61] |
| kik | Bantu |  | 25.98 | 33.28 | +7.30 [+6.39, +8.22] | +7.67 [+6.71, +8.66] | +0.065 [+0.058, +0.073] | 0.93 / 0.08 | 1.00 / 0.00 | 0.92 | +3.32 [+2.89, +3.81] | +4.28 [+3.81, +4.80] |
| hau | Afroasiatic |  | 39.88 | 46.46 | +6.58 [+5.09, +8.34] | +7.24 [+5.62, +9.18] | +0.027 [+0.020, +0.035] | 0.96 / 0.04 | 1.00 / 0.01 | 0.95 | +0.90 [+0.50, +1.30] | +1.48 [+0.97, +2.01] |
| mlg | Other (Indo-European, Austronesian) |  | 39.04 | 44.73 | +5.68 [+4.54, +6.77] | +6.32 [+5.10, +7.50] | +0.041 [+0.034, +0.048] | 0.97 / 0.03 | 1.00 / 0.00 | 0.97 | +2.03 [+1.59, +2.45] | +2.58 [+2.09, +3.06] |
| nya | Bantu |  | 36.88 | 42.21 | +5.33 [+4.23, +6.54] | +5.90 [+4.71, +7.21] | +0.026 [+0.019, +0.034] | 0.97 / 0.04 | 1.00 / 0.00 | 0.96 | +1.82 [+1.41, +2.24] | +2.49 [+2.01, +2.99] |
| ewe | Atlantic-Congo (non-Bantu) | yes | 25.96 | 31.17 | +5.21 [+4.46, +6.04] | +5.55 [+4.75, +6.43] | +0.046 [+0.039, +0.052] | 0.95 / 0.06 | 1.00 / 0.00 | 0.94 | +2.12 [+1.75, +2.51] | +2.75 [+2.37, +3.15] |
| sot | Bantu |  | 37.87 | 42.72 | +4.85 [+3.70, +6.08] | +5.13 [+3.93, +6.45] | +0.027 [+0.021, +0.033] | 0.97 / 0.04 | 1.00 / 0.00 | 0.96 | +1.15 [+0.82, +1.50] | +1.66 [+1.22, +2.08] |
| arz | family unlabelled |  | 4.59 | 9.18 | +4.59 [+4.01, +5.15] | +4.94 [+4.27, +5.58] | - | 0.49 / 0.52 | 0.87 / 0.19 | 0.43 | +0.67 [+0.05, +1.25] | -1.08 [-1.70, -0.48] |
| som | Afroasiatic |  | 35.47 | 39.71 | +4.24 [+3.14, +5.34] | +4.56 [+3.41, +5.71] | +0.030 [+0.023, +0.037] | 0.97 / 0.03 | 1.00 / 0.00 | 0.97 | +1.38 [+1.02, +1.75] | +1.78 [+1.37, +2.18] |
| wol | Atlantic-Congo (non-Bantu) | yes | 26.87 | 30.58 | +3.72 [+2.74, +4.63] | +3.99 [+2.91, +5.00] | +0.024 [+0.018, +0.030] | 0.97 / 0.03 | 1.00 / 0.00 | 0.97 | +1.14 [+0.82, +1.44] | +1.56 [+1.22, +1.94] |
| sna | Bantu |  | 33.92 | 37.56 | +3.64 [+2.47, +4.85] | +4.29 [+3.00, +5.65] | +0.023 [+0.017, +0.030] | 0.97 / 0.04 | 1.00 / 0.00 | 0.96 | +0.26 [-0.11, +0.66] | +0.66 [+0.05, +1.25] |
| zul | Bantu |  | 45.91 | 48.27 | +2.36 [+1.56, +3.14] | +2.66 [+1.79, +3.52] | +0.014 [+0.009, +0.020] | 0.99 / 0.01 | 1.00 / 0.00 | 0.99 | +1.26 [+0.84, +1.67] | +1.43 [+1.00, +1.84] |
| ibo | Atlantic-Congo (non-Bantu) |  | 37.82 | 40.14 | +2.33 [+1.57, +3.08] | +2.45 [+1.65, +3.23] | +0.014 [+0.009, +0.019] | 0.98 / 0.02 | 1.00 / 0.00 | 0.98 | +0.88 [+0.53, +1.24] | +1.08 [+0.71, +1.48] |
| ary | family unlabelled |  | 4.64 | 6.88 | +2.24 [+1.76, +2.74] | +1.93 [+1.35, +2.51] | - | 0.54 / 0.48 | 0.97 / 0.11 | 0.48 | -3.70 [-4.39, -3.03] | -2.52 [-3.07, -1.96] |
| swh | Bantu |  | 56.24 | 58.36 | +2.12 [+1.10, +3.38] | +2.13 [+1.14, +3.33] | +0.016 [+0.010, +0.022] | 1.00 / 0.01 | 1.00 / 0.00 | 0.99 | +0.88 [+0.45, +1.30] | +1.09 [+0.60, +1.61] |
| fon | Atlantic-Congo (non-Bantu) | yes | 20.17 | 22.25 | +2.07 [+1.60, +2.55] | +2.28 [+1.78, +2.77] | +0.029 [+0.024, +0.035] | 0.98 / 0.04 | 1.00 / 0.00 | 0.96 | +0.77 [+0.56, +0.97] | +1.36 [+1.03, +1.67] |
| xho | Bantu |  | 43.56 | 45.43 | +1.87 [+1.05, +2.80] | +2.22 [+1.31, +3.24] | +0.011 [+0.006, +0.017] | 0.99 / 0.01 | 1.00 / 0.00 | 0.99 | +0.67 [+0.28, +1.09] | +0.89 [+0.49, +1.33] |
| zgh | family unlabelled |  | 2.29 | 4.07 | +1.78 [+1.30, +2.30] | +1.40 [+0.88, +1.96] | - | 0.12 / 0.89 | 0.97 / 0.46 | 0.07 | -6.69 [-9.11, -4.34] | +0.54 [-0.02, +1.11] |
| aeb | family unlabelled |  | 2.75 | 4.12 | +1.36 [+0.96, +1.78] | +1.08 [+0.62, +1.55] | - | 0.53 / 0.49 | 0.96 / 0.25 | 0.40 | -2.46 [-3.19, -1.74] | -1.58 [-2.06, -1.06] |
| afr | Other (Indo-European, Austronesian) |  | 63.13 | 64.13 | +1.01 [+0.53, +1.61] | +1.05 [+0.56, +1.68] | +0.010 [+0.006, +0.015] | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 | +0.78 [+0.46, +1.11] | +0.79 [+0.46, +1.11] |

NTREX and MAFAND per language:

#### Table 7a. Per language, NTREX (clean rows)

| iso | n | SFT chrF++ | RL chrF++ | Δ all rows [CI] | Δ chrF [CI] | Δ AfriCOMET [CI] | Δ paired rows [CI] | rows kept |
|---|---:|---:|---:|---|---|---|---|---:|
| orm | 1992 | 9.63 | 27.33 | +17.70 [+17.16, +18.25] | +20.84 [+20.20, +21.47] | +0.168 [+0.161, +0.175] | +3.15 [+2.81, +3.49] | 0.56 |
| ven | 1992 | 21.30 | 38.81 | +17.52 [+16.72, +18.31] | +18.28 [+17.47, +19.11] | - | +3.89 [+3.50, +4.27] | 0.74 |
| ssw | 1992 | 21.58 | 38.80 | +17.22 [+16.40, +18.02] | +19.52 [+18.60, +20.40] | +0.122 [+0.115, +0.130] | +6.88 [+6.46, +7.31] | 0.82 |
| bem | 1992 | 18.68 | 34.79 | +16.11 [+15.43, +16.84] | +17.40 [+16.64, +18.21] | +0.111 [+0.105, +0.117] | +7.46 [+7.04, +7.91] | 0.81 |
| nso | 1992 | 20.14 | 32.73 | +12.60 [+11.83, +13.35] | +13.33 [+12.50, +14.13] | +0.101 [+0.095, +0.108] | +2.50 [+2.24, +2.78] | 0.82 |
| tsn | 1992 | 36.97 | 48.71 | +11.74 [+10.68, +12.81] | +11.83 [+10.77, +12.95] | +0.073 [+0.067, +0.080] | +4.13 [+3.72, +4.54] | 0.92 |
| kin | 1992 | 26.40 | 37.13 | +10.73 [+9.95, +11.52] | +11.51 [+10.67, +12.35] | +0.072 [+0.066, +0.078] | +4.88 [+4.51, +5.30] | 0.91 |
| som | 1992 | 35.84 | 44.23 | +8.39 [+7.44, +9.25] | +8.85 [+7.82, +9.79] | +0.061 [+0.055, +0.066] | +3.29 [+2.95, +3.63] | 0.95 |
| yor | 1992 | 23.77 | 32.10 | +8.33 [+7.58, +9.10] | +8.53 [+7.79, +9.28] | +0.104 [+0.097, +0.110] | +5.04 [+4.49, +5.67] | 0.94 |
| tir | 1992 | 9.10 | 17.00 | +7.90 [+7.52, +8.25] | +8.56 [+8.17, +8.94] | +0.144 [+0.136, +0.151] | +1.77 [+1.49, +2.06] | 0.75 |
| amh | 1992 | 10.58 | 18.05 | +7.46 [+7.00, +7.93] | +8.16 [+7.66, +8.67] | +0.078 [+0.071, +0.084] | +2.39 [+2.11, +2.67] | 0.84 |
| ewe | 1992 | 24.28 | 31.58 | +7.30 [+6.64, +8.00] | +7.79 [+7.07, +8.55] | +0.064 [+0.059, +0.069] | +3.44 [+3.14, +3.74] | 0.93 |
| hau | 1992 | 42.15 | 48.57 | +6.42 [+5.24, +7.58] | +6.82 [+5.58, +8.06] | +0.028 [+0.023, +0.034] | +1.38 [+1.08, +1.66] | 0.96 |
| nya | 1992 | 37.87 | 44.23 | +6.36 [+5.46, +7.34] | +6.84 [+5.86, +7.89] | +0.025 [+0.020, +0.030] | +2.57 [+2.22, +2.93] | 0.96 |
| wol | 1992 | 25.29 | 30.37 | +5.08 [+4.34, +5.86] | +5.46 [+4.68, +6.26] | +0.032 [+0.027, +0.037] | +1.72 [+1.46, +1.96] | 0.95 |
| sna | 1992 | 34.25 | 39.23 | +4.99 [+4.17, +5.83] | +5.64 [+4.72, +6.61] | +0.024 [+0.019, +0.029] | +0.97 [+0.68, +1.25] | 0.96 |
| ibo | 1992 | 39.90 | 44.87 | +4.97 [+4.17, +5.82] | +5.14 [+4.29, +6.04] | +0.028 [+0.024, +0.032] | +1.83 [+1.58, +2.12] | 0.97 |
| nde | 1992 | 28.83 | 32.95 | +4.12 [+3.51, +4.74] | +4.67 [+3.95, +5.41] | - | +1.85 [+1.59, +2.11] | 0.97 |
| mlg | 1992 | 36.91 | 40.85 | +3.94 [+3.19, +4.74] | +4.43 [+3.63, +5.33] | +0.026 [+0.021, +0.031] | +1.15 [+0.90, +1.41] | 0.97 |
| swh | 1992 | 55.18 | 58.73 | +3.55 [+2.55, +4.54] | +3.48 [+2.47, +4.48] | +0.017 [+0.012, +0.022] | +1.20 [+0.88, +1.54] | 0.99 |
| zul | 1992 | 40.92 | 43.50 | +2.58 [+1.89, +3.29] | +2.92 [+2.13, +3.73] | +0.015 [+0.010, +0.019] | +1.15 [+0.84, +1.44] | 0.99 |
| xho | 1992 | 39.85 | 42.36 | +2.51 [+1.92, +3.16] | +2.95 [+2.25, +3.67] | +0.013 [+0.009, +0.017] | +1.04 [+0.79, +1.31] | 0.99 |
| afr | 1992 | 62.42 | 63.92 | +1.50 [+1.00, +2.15] | +1.50 [+0.97, +2.24] | +0.016 [+0.012, +0.020] | +1.14 [+0.86, +1.43] | 1.00 |
| shi | 1992 | 0.24 | 1.16 | +0.93 [+0.85, +1.02] | +0.50 [+0.43, +0.60] | - | +0.03 [-0.01, +0.09] | 0.35 |

#### Table 7b. Per language, MAFAND (clean rows)

| iso | n | SFT chrF++ | RL chrF++ | Δ all rows [CI] | Δ chrF [CI] | Δ AfriCOMET [CI] | Δ paired rows [CI] | rows kept |
|---|---:|---:|---:|---|---|---|---|---:|
| lug | 1500 | 26.97 | 39.57 | +12.60 [+11.60, +13.70] | +13.37 [+12.26, +14.53] | +0.065 [+0.058, +0.071] | +6.78 [+6.22, +7.34] | 0.93 |
| yor | 1550 | 23.47 | 33.45 | +9.98 [+9.15, +10.70] | +10.12 [+9.28, +10.87] | +0.127 [+0.119, +0.133] | +6.42 [+5.85, +6.96] | 0.94 |
| kin | 1005 | 36.51 | 46.41 | +9.90 [+8.70, +11.20] | +10.47 [+9.18, +11.87] | +0.064 [+0.056, +0.073] | +5.31 [+4.75, +5.90] | 0.94 |
| luo | 1500 | 22.95 | 32.62 | +9.67 [+8.76, +10.50] | +10.50 [+9.54, +11.38] | +0.072 [+0.065, +0.079] | +3.60 [+3.22, +3.99] | 0.89 |
| tsn | 1500 | 43.85 | 53.10 | +9.24 [+7.87, +10.70] | +9.32 [+7.93, +10.81] | +0.045 [+0.039, +0.052] | +2.51 [+2.06, +2.95] | 0.95 |
| amh | 1014 | 11.62 | 18.33 | +6.71 [+5.91, +7.46] | +7.46 [+6.62, +8.31] | +0.061 [+0.052, +0.070] | +1.64 [+1.21, +2.06] | 0.89 |
| twi | 1500 | 28.58 | 34.14 | +5.56 [+4.75, +6.29] | +5.99 [+5.14, +6.76] | +0.042 [+0.037, +0.047] | +1.46 [+1.18, +1.75] | 0.93 |
| nya | 1003 | 38.14 | 40.95 | +2.81 [+1.96, +3.71] | +3.18 [+2.23, +4.12] | +0.009 [+0.002, +0.015] | +0.60 [+0.27, +0.91] | 0.97 |
| sna | 1004 | 30.35 | 32.91 | +2.56 [+1.64, +3.39] | +3.11 [+2.05, +4.07] | +0.018 [+0.011, +0.025] | -0.14 [-0.44, +0.15] | 0.96 |
| xho | 1001 | 49.76 | 51.78 | +2.02 [+1.06, +2.98] | +2.35 [+1.33, +3.40] | +0.008 [+0.002, +0.014] | +0.79 [+0.29, +1.31] | 0.99 |
| ibo | 1458 | 43.55 | 44.89 | +1.34 [+0.43, +2.32] | +1.46 [+0.50, +2.50] | +0.005 [+0.000, +0.010] | -0.41 [-0.79, -0.04] | 0.99 |
| swh | 1822 | 50.68 | 51.79 | +1.12 [+0.38, +1.89] | +1.15 [+0.41, +1.94] | +0.012 [+0.007, +0.017] | +0.01 [-0.31, +0.32] | 0.99 |
| zul | 998 | 44.50 | 45.58 | +1.08 [+0.49, +1.73] | +1.32 [+0.70, +2.04] | +0.011 [+0.005, +0.017] | +0.58 [+0.20, +0.97] | 0.99 |
| hau | 1498 | 32.03 | 32.80 | +0.77 [+0.21, +1.41] | +1.05 [+0.45, +1.75] | +0.014 [+0.009, +0.019] | -0.32 [-0.54, -0.11] | 0.99 |
| pcm | 1562 | 53.58 | 53.34 | -0.25 [-0.69, +0.30] | -0.46 [-0.93, +0.14] | +0.007 [+0.004, +0.009] | -0.45 [-0.77, -0.16] | 1.00 |

## 3. Finishing and looping against translation improvement (task 2)

#### Table 2. Finishing and looping (clean rows; share of rows, language-macro)

| system | FLORES+ devtest finished / loop / hyp-to-ref length | NTREX finished / loop / hyp-to-ref length | MAFAND finished / loop / hyp-to-ref length |
|---|---|---|---|
| SFT | 0.829 / 0.209 / 3.03 | 0.890 / 0.122 / 2.35 | 0.963 / 0.043 / 1.48 |
| RL stage A | 0.983 / 0.068 / 1.33 | 0.996 / 0.016 / 1.07 | 0.999 / 0.003 / 1.00 |
| RL B1 best | 0.993 / 0.054 / 1.17 | 0.999 / 0.009 / 1.02 | 1.000 / 0.001 / 0.99 |
| RL mixed best | 0.995 / 0.048 / 1.14 | 0.999 / 0.006 / 1.02 | 1.000 / 0.001 / 0.98 |

Share of rows kept by the paired subset (both finished, no loop): FLORES+ devtest 0.783, NTREX 0.876, MAFAND 0.957. Loop detector (`is_repetitive` or `is_self_repetitive` of the RL reward guards) on the human references: FLORES+ devtest 0.0001, NTREX 0.0000, MAFAND 0.0000 of rows.

Four ways to remove the finishing effect, all with the same bootstrap (table 1):

| row set | chrF++ macro, 93 cells | share of the +7.96 | AfriCOMET macro, 76 cells | share of +0.0631 |
|---|---|---:|---|---:|
| all clean rows | +7.96 [+7.82, +8.12] | 100 % | +0.0631 [+0.0620, +0.0643] | 100 % |
| SFT cut at its first loop onset, RL as is | +4.06 [+3.97, +4.15] | 51 % | not scorable (cut text has no cached score) | |
| both cut at their first loop | +4.18 [+4.10, +4.28] | 53 % | | |
| rows where the SFT is finished and loop-free (RL failures kept) | +2.41 [+2.33, +2.49] | 30 % | +0.0366 [+0.0357, +0.0374] | 58 % |
| rows where both are finished and loop-free | +2.61 [+2.54, +2.70] | 33 % | +0.0370 [+0.0362, +0.0379] | 59 % |
| rows where the SFT failed (unfinished or loop; 21 % of FLORES+ rows) | +24.57 [+24.26, +24.90] | | +0.3342 [+0.3295, +0.3391] | |

Reading: on the SFT's failed rows its chrF++ is 5.95 and its AfriCOMET 0.24 (the SFT keeps generating after the translation), the RL gets 30.5 and 0.58, so the failed rows carry a large part of the macro gain. The finishing effect is therefore **about half of the chrF++ gain (49 % recovered by loop-cutting alone; 67 to 70 % attributable to rows where either model failed) and about 41 % of the AfriCOMET gain**. What is left, **+2.4 to +4.1 chrF++ and +0.037 AfriCOMET**, is translation quality on outputs that are well-formed in both systems; it is significant in 50 of 54 languages on clean rows (table 5 and the per-language table).

How far each estimate can be trusted: the "both finished" subset removes the RL's own failures (favours RL) and the SFT's (favours SFT) and keeps only the sentences the SFT can handle, which are easier (rows kept: FLORES+ 78 %, NTREX 88 %, MAFAND 96 %; for knc 4 %, taq 7 %, zgh 7 %, dik 13 %, so the subset of those languages is a handful of easy sentences); the "SFT finished" subset keeps RL failures and is the fairer lower bound; cutting the SFT at the loop is the most generous treatment of the SFT but leaves unfinished-without-loop rows and any other trailing text untouched. The three estimates agree on the sign and rough size, and none of them makes the gain vanish.

## 4. Where the gain is: language groups and the number of languages with RL worse (task 3)

#### Table 4. By language group (language-macro Δ, final RL minus SFT; FLORES+ devtest, then NTREX where the group is present)

### all clean rows

| grouping | group | FLORES+ languages | Δ chrF++ [CI] | Δ chrF [CI] | Δ AfriCOMET [CI] | NTREX Δ chrF++ [CI] |
|---|---|---:|---|---|---|---|
| language family | Afroasiatic | 6 | +9.59 [+9.14, +10.06] | +10.55 [+10.04, +11.05] | +0.0995 [+0.0951, +0.1036] | +8.13 [+7.81, +8.43] |
| language family | Atlantic-Congo (non-Bantu) | 9 | +7.06 [+6.75, +7.37] | +7.45 [+7.12, +7.77] | +0.0619 [+0.0592, +0.0645] | +6.42 [+6.02, +6.83] |
| language family | Bantu | 23 | +9.81 [+9.48, +10.13] | +10.65 [+10.30, +10.99] | +0.0691 [+0.0668, +0.0717] | +9.17 [+8.89, +9.44] |
| language family | Creole / Pidgin | 1 | +20.27 [+18.91, +21.53] | +20.53 [+19.13, +21.81] | - | - |
| language family | Mande | 2 | +8.42 [+7.92, +8.94] | +9.31 [+8.76, +9.88] | +0.0688 [+0.0624, +0.0758] | - |
| language family | Nilo-Saharan | 3 | +11.62 [+11.16, +12.04] | +12.64 [+12.15, +13.10] | +0.1254 [+0.1186, +0.1319] | - |
| language family | Other (Indo-European, Austronesian) | 2 | +3.34 [+2.67, +4.01] | +3.68 [+2.96, +4.42] | +0.0257 [+0.0215, +0.0296] | +2.72 [+2.26, +3.22] |
| language family | family unlabelled | 8 | +6.28 [+6.00, +6.56] | +6.70 [+6.40, +7.01] | - | - |
| weak list | other | 34 | +8.53 [+8.29, +8.78] | +9.23 [+8.97, +9.50] | +0.0634 [+0.0614, +0.0657] | +8.50 [+8.24, +8.74] |
| weak list | weak (x2.5 RL list) | 20 | +9.29 [+9.01, +9.53] | +9.98 [+9.68, +10.23] | +0.0870 [+0.0846, +0.0894] | +6.17 [+5.93, +6.41] |
| A0 language | A0 language (first 85) | 46 | +9.25 [+9.00, +9.48] | +9.99 [+9.73, +10.25] | +0.0728 [+0.0710, +0.0750] | +7.91 [+7.69, +8.12] |
| A0 language | new language (40) | 8 | +6.28 [+6.00, +6.56] | +6.70 [+6.40, +7.01] | - | - |
| terciles of the SFT's own FLORES+ chrF++ | T1 weakest third of SFT chrF++ | 18 | +9.13 [+8.91, +9.32] | +10.04 [+9.81, +10.25] | +0.1204 [+0.1168, +0.1242] | +12.80 [+12.44, +13.16] |
| terciles of the SFT's own FLORES+ chrF++ | T2 middle third | 18 | +10.90 [+10.57, +11.22] | +11.71 [+11.35, +12.05] | +0.0858 [+0.0830, +0.0886] | +11.19 [+10.88, +11.52] |
| terciles of the SFT's own FLORES+ chrF++ | T3 strongest third | 18 | +6.40 [+6.06, +6.74] | +6.77 [+6.41, +7.13] | +0.0370 [+0.0348, +0.0391] | +5.74 [+5.43, +6.03] |
| terciles of the SFT's own FLORES+ chrF++ | not on FLORES+ | 0 | - | - | - | +7.52 [+7.16, +7.86] |

### only rows where BOTH models finished and neither loops

| grouping | group | FLORES+ languages | Δ chrF++ [CI] | Δ chrF [CI] | Δ AfriCOMET [CI] | NTREX Δ chrF++ [CI] |
|---|---|---:|---|---|---|---|
| language family | Afroasiatic | 6 | +2.09 [+1.91, +2.28] | +2.32 [+2.12, +2.52] | +0.0459 [+0.0427, +0.0490] | +2.00 [+1.88, +2.13] |
| language family | Atlantic-Congo (non-Bantu) | 9 | +3.25 [+3.09, +3.42] | +3.46 [+3.29, +3.63] | +0.0410 [+0.0389, +0.0431] | +3.01 [+2.82, +3.21] |
| language family | Bantu | 23 | +3.53 [+3.39, +3.69] | +3.78 [+3.63, +3.95] | +0.0407 [+0.0390, +0.0425] | +3.21 [+3.09, +3.33] |
| language family | Creole / Pidgin | 1 | +7.66 [+6.93, +8.42] | +7.93 [+7.17, +8.68] | - | - |
| language family | Mande | 2 | +3.29 [+3.06, +3.54] | +3.82 [+3.57, +4.09] | +0.0412 [+0.0360, +0.0470] | - |
| language family | Nilo-Saharan | 3 | +3.63 [+3.07, +4.18] | +4.01 [+3.37, +4.60] | +0.0572 [+0.0524, +0.0620] | - |
| language family | Other (Indo-European, Austronesian) | 2 | +1.41 [+1.12, +1.67] | +1.57 [+1.29, +1.84] | +0.0205 [+0.0168, +0.0243] | +1.15 [+0.95, +1.33] |
| language family | family unlabelled | 8 | -0.13 [-0.54, +0.27] | -0.38 [-0.86, +0.11] | - | - |
| weak list | other | 34 | +2.30 [+2.16, +2.45] | +2.42 [+2.26, +2.58] | +0.0356 [+0.0341, +0.0373] | +2.80 [+2.71, +2.91] |
| weak list | weak (x2.5 RL list) | 20 | +3.58 [+3.46, +3.72] | +3.88 [+3.75, +4.02] | +0.0500 [+0.0482, +0.0516] | +2.40 [+2.27, +2.55] |
| A0 language | A0 language (first 85) | 46 | +3.28 [+3.17, +3.41] | +3.54 [+3.42, +3.67] | +0.0414 [+0.0400, +0.0427] | +2.70 [+2.62, +2.79] |
| A0 language | new language (40) | 8 | -0.13 [-0.54, +0.27] | -0.38 [-0.86, +0.11] | - | - |
| terciles of the SFT's own FLORES+ chrF++ | T1 weakest third of SFT chrF++ | 18 | +1.95 [+1.73, +2.17] | +2.12 [+1.88, +2.37] | +0.0586 [+0.0560, +0.0609] | +2.46 [+2.23, +2.70] |
| terciles of the SFT's own FLORES+ chrF++ | T2 middle third | 18 | +4.40 [+4.24, +4.58] | +4.65 [+4.49, +4.84] | +0.0550 [+0.0529, +0.0573] | +5.02 [+4.84, +5.20] |
| terciles of the SFT's own FLORES+ chrF++ | T3 strongest third | 18 | +1.98 [+1.84, +2.12] | +2.09 [+1.96, +2.24] | +0.0211 [+0.0195, +0.0225] | +1.85 [+1.74, +1.96] |
| terciles of the SFT's own FLORES+ chrF++ | not on FLORES+ | 0 | - | - | - | +1.92 [+1.77, +2.08] |

### SFT output cut at its first loop (RL as is)

| grouping | group | FLORES+ languages | Δ chrF++ [CI] | Δ chrF [CI] | Δ AfriCOMET [CI] | NTREX Δ chrF++ [CI] |
|---|---|---:|---|---|---|---|
| language family | Afroasiatic | 6 | +4.02 [+3.80, +4.25] | +4.46 [+4.21, +4.71] | - | +3.46 [+3.31, +3.62] |
| language family | Atlantic-Congo (non-Bantu) | 9 | +4.32 [+4.13, +4.51] | +4.60 [+4.39, +4.82] | - | +3.85 [+3.62, +4.08] |
| language family | Bantu | 23 | +5.11 [+4.91, +5.32] | +5.48 [+5.28, +5.71] | - | +4.83 [+4.67, +5.01] |
| language family | Creole / Pidgin | 1 | +11.80 [+10.83, +12.78] | +12.41 [+11.38, +13.49] | - | - |
| language family | Mande | 2 | +5.95 [+5.56, +6.33] | +6.85 [+6.43, +7.28] | - | - |
| language family | Nilo-Saharan | 3 | +7.10 [+6.79, +7.43] | +7.84 [+7.52, +8.21] | - | - |
| language family | Other (Indo-European, Austronesian) | 2 | +1.68 [+1.38, +1.98] | +1.85 [+1.55, +2.15] | - | +1.45 [+1.25, +1.66] |
| language family | family unlabelled | 8 | +1.86 [+1.64, +2.08] | +1.95 [+1.72, +2.19] | - | - |
| weak list | other | 34 | +3.99 [+3.84, +4.15] | +4.30 [+4.13, +4.47] | - | +4.26 [+4.13, +4.40] |
| weak list | weak (x2.5 RL list) | 20 | +5.39 [+5.22, +5.57] | +5.85 [+5.68, +6.04] | - | +3.40 [+3.23, +3.56] |
| A0 language | A0 language (first 85) | 46 | +4.97 [+4.83, +5.13] | +5.38 [+5.22, +5.55] | - | +4.04 [+3.93, +4.16] |
| A0 language | new language (40) | 8 | +1.86 [+1.64, +2.08] | +1.95 [+1.72, +2.19] | - | - |
| terciles of the SFT's own FLORES+ chrF++ | T1 weakest third of SFT chrF++ | 18 | +4.59 [+4.42, +4.77] | +5.11 [+4.92, +5.31] | - | +5.44 [+5.10, +5.76] |
| terciles of the SFT's own FLORES+ chrF++ | T2 middle third | 18 | +6.15 [+5.93, +6.36] | +6.55 [+6.33, +6.78] | - | +6.81 [+6.61, +7.04] |
| terciles of the SFT's own FLORES+ chrF++ | T3 strongest third | 18 | +2.80 [+2.63, +2.99] | +2.96 [+2.78, +3.15] | - | +2.63 [+2.49, +2.77] |
| terciles of the SFT's own FLORES+ chrF++ | not on FLORES+ | 0 | - | - | - | +3.72 [+3.49, +3.95] |

* **Weak list vs other** (the x2.5 RL list, 20 FLORES+ languages): all rows +9.29 vs +8.53 chrF++, AfriCOMET +0.087 vs +0.063, clean rows +3.58 vs +2.30. The weak list gains more on FLORES+, but on NTREX the six weak-list languages present gain less (+6.17 vs +8.50), so the advantage is not uniform across benchmarks.
* **Resource level** (no external labels exist for the 54 languages; terciles of the SFT's own FLORES+ chrF++ are the proxy): weakest third +9.13, middle third +10.90, strongest third +6.40 chrF++ (all rows). The strongest languages gain least (afr +1.0, the largest gain in a language near the SFT ceiling is small) and the weakest third's clean-row gain is only +1.95 because its clean subset is tiny (survivor bias) while its AfriCOMET gain is the largest (+0.12 all rows).
* **Family:** Nilo-Saharan +11.6, Bantu +9.8, Afroasiatic +9.6 (dragged down by the Arabic dialects, see below), Mande +8.4, Atlantic-Congo non-Bantu +7.1, Indo-European / Austronesian (afr, mlg) +3.3; pcm +20.3 on FLORES+ (but -0.25 on MAFAND).
* **Languages added after A0** (8 on FLORES+: aeb ary arz cjk dik mfe taq zgh): +6.28 all rows, clean rows -0.13 [-0.54, +0.27]. Inside the group, cjk (+2.13), mfe (+3.91), arz (+0.67), dik (+1.8), taq (+3.3) are positive on clean rows, aeb (-2.46), ary (-3.70), zgh (-6.69) negative; the Arabic dialects and Tamazight sit at chrF++ 2 to 9 in both models.
* **Number of languages where the RL is worse than the SFT** (table 5): on all rows, FLORES+ 0 / 54, NTREX 0 / 24, MAFAND 1 / 15 (pcm, not significant) for chrF++ and chrF, and 0 on AfriCOMET everywhere. On clean rows FLORES+ chrF++ 3 / 54 (aeb, ary, zgh, all significant), MAFAND 4 / 15 (hau, ibo, pcm, sna; significant for hau, ibo, pcm, about -0.3 to -0.5 chrF++; AfriCOMET is not significantly lower for any language).

#### Table 5. Languages where the RL model is WORSE than the SFT (point estimate; ‘sig’ = 95 % CI entirely below zero)

| rows | benchmark | metric | cells | worse | sig. worse | sig. better |
|---|---|---|---:|---:|---:|---:|
| all clean rows | FLORES+ devtest | chrF++ | 54 | 0 | 0 | 54 |
| all clean rows | FLORES+ devtest | chrF | 54 | 0 | 0 | 54 |
| all clean rows | FLORES+ devtest | AfriCOMET | 40 | 0 | 0 | 40 |
| all clean rows | NTREX | chrF++ | 24 | 0 | 0 | 24 |
| all clean rows | NTREX | chrF | 24 | 0 | 0 | 24 |
| all clean rows | NTREX | AfriCOMET | 21 | 0 | 0 | 21 |
| all clean rows | MAFAND | chrF++ | 15 | 1 (pcm) | 0 | 14 |
| all clean rows | MAFAND | chrF | 15 | 1 (pcm) | 0 | 14 |
| all clean rows | MAFAND | AfriCOMET | 15 | 0 | 0 | 15 |
| only rows where BOTH models finished and neither loops | FLORES+ devtest | chrF++ | 54 | 3 (aeb, ary, zgh) | 3 (aeb, ary, zgh) | 50 |
| only rows where BOTH models finished and neither loops | FLORES+ devtest | chrF | 54 | 3 (aeb, ary, zgh) | 3 (aeb, ary, zgh) | 50 |
| only rows where BOTH models finished and neither loops | FLORES+ devtest | AfriCOMET | 40 | 0 | 0 | 40 |
| only rows where BOTH models finished and neither loops | NTREX | chrF++ | 24 | 0 | 0 | 23 |
| only rows where BOTH models finished and neither loops | NTREX | chrF | 24 | 0 | 0 | 23 |
| only rows where BOTH models finished and neither loops | NTREX | AfriCOMET | 21 | 0 | 0 | 21 |
| only rows where BOTH models finished and neither loops | MAFAND | chrF++ | 15 | 4 (hau, ibo, pcm, sna) | 3 (hau, ibo, pcm) | 10 |
| only rows where BOTH models finished and neither loops | MAFAND | chrF | 15 | 4 (hau, ibo, pcm, sna) | 1 (pcm) | 10 |
| only rows where BOTH models finished and neither loops | MAFAND | AfriCOMET | 15 | 1 (nya) | 0 | 11 |
| SFT output cut at its first loop (RL as is) | FLORES+ devtest | chrF++ | 54 | 3 (aeb, ary, arz) | 3 (aeb, ary, arz) | 50 |
| SFT output cut at its first loop (RL as is) | FLORES+ devtest | chrF | 54 | 3 (aeb, ary, arz) | 3 (aeb, ary, arz) | 51 |
| SFT output cut at its first loop (RL as is) | NTREX | chrF++ | 24 | 0 | 0 | 24 |
| SFT output cut at its first loop (RL as is) | NTREX | chrF | 24 | 0 | 0 | 24 |
| SFT output cut at its first loop (RL as is) | MAFAND | chrF++ | 15 | 3 (hau, ibo, pcm) | 1 (pcm) | 10 |
| SFT output cut at its first loop (RL as is) | MAFAND | chrF | 15 | 2 (ibo, pcm) | 1 (pcm) | 11 |

### Along the RL chain (which stage produced the gain)

#### Table 3. Where the gain accrued along the RL chain (all clean rows, macro; paired bootstrap)

| step | benchmark | chrF++ before | after | Δ chrF++ [CI] | Δ chrF [CI] | Δ AfriCOMET [CI] |
|---|---|---:|---:|---|---|---|
| SFT -> stage A (400 steps) | FLORES+ devtest | 22.85 | 30.81 | +7.96 [+7.76, +8.16] | +8.57 [+8.35, +8.78] | +0.0691 [+0.0674, +0.0709] |
| SFT -> stage A (400 steps) | NTREX | 29.25 | 36.68 | +7.43 [+7.22, +7.62] | +7.99 [+7.77, +8.19] | +0.0617 [+0.0602, +0.0630] |
| SFT -> stage A (400 steps) | MAFAND | 35.77 | 40.81 | +5.04 [+4.80, +5.28] | +5.34 [+5.08, +5.59] | +0.0394 [+0.0378, +0.0409] |
| stage A -> B1 best (135 steps run, best checkpoint) | FLORES+ devtest | 30.81 | 31.35 | +0.54 [+0.49, +0.59] | +0.59 [+0.54, +0.65] | +0.0016 [+0.0010, +0.0022] |
| stage A -> B1 best (135 steps run, best checkpoint) | NTREX | 36.68 | 36.98 | +0.30 [+0.24, +0.36] | +0.33 [+0.26, +0.40] | +0.0009 [+0.0002, +0.0016] |
| stage A -> B1 best (135 steps run, best checkpoint) | MAFAND | 40.81 | 40.81 | -0.00 [-0.08, +0.08] | +0.02 [-0.07, +0.11] | -0.0011 [-0.0021, -0.0003] |
| B1 best -> mixed best (60 steps) | FLORES+ devtest | 31.35 | 31.66 | +0.31 [+0.26, +0.35] | +0.35 [+0.30, +0.39] | +0.0021 [+0.0015, +0.0027] |
| B1 best -> mixed best (60 steps) | NTREX | 36.98 | 37.17 | +0.18 [+0.12, +0.25] | +0.22 [+0.15, +0.29] | +0.0004 [-0.0003, +0.0011] |
| B1 best -> mixed best (60 steps) | MAFAND | 40.81 | 40.78 | -0.03 [-0.10, +0.04] | +0.01 [-0.06, +0.08] | -0.0010 [-0.0019, -0.0002] |
| SFT -> B1 best | FLORES+ devtest | 22.85 | 31.35 | +8.50 [+8.29, +8.72] | +9.16 [+8.93, +9.39] | +0.0707 [+0.0690, +0.0727] |
| SFT -> B1 best | NTREX | 29.25 | 36.98 | +7.73 [+7.51, +7.92] | +8.32 [+8.09, +8.53] | +0.0625 [+0.0610, +0.0640] |
| SFT -> B1 best | MAFAND | 35.77 | 40.81 | +5.04 [+4.79, +5.27] | +5.36 [+5.10, +5.60] | +0.0383 [+0.0367, +0.0398] |

SFT -> stage A is 94 % of the SFT -> B1 gain on FLORES+ (7.96 of 8.50) and 96 % on NTREX; B1 and the mixed stage add +0.54 and +0.31 on FLORES+, +0.30 and +0.18 on NTREX, and nothing measurable on MAFAND (AfriCOMET on MAFAND -0.001, a significant but negligible decrease at each of the two later stages). The dev series of stage A (`reports/rl_stage_A_385m_2026-11-03.md`) agrees: +3.3 of +3.7 dev chrF++ in the first 100 of 400 steps.

## 5. Science: SFT alpha 0.5 against the science RL (task 4)

Full evaluation: `reports/eval_science_rl_385m_2026-11-04.md` (paired bootstrap over sentences, 1000 resamples, macro over the 6 AfriScience-MT languages, TEST split; the rules were fixed before the run: independent evidence = held-out reference-based AfriCOMET-STL-1.1 and plain chrF against the human references; **excluded as reward family**: `africomet-qe-stl-1.1`, `ssa-comet-mtl`, and with them SSA-COMET-QE, shown only as the paper's headline).

| science RL checkpoint minus science SFT alpha 0.5 (the RL's start) | chrF [95 % CI] | AfriCOMET-STL, held out [95 % CI] | SSA-COMET-QE (reward family, not evidence) |
|---|---|---|---|
| best/ (step 75) | +0.16 [-0.04, +0.37] | +0.0013 [-0.0005, +0.0031] | +0.0018 [+0.0006, +0.0030] |
| final/ (step 160) | **+0.46 [+0.27, +0.65]** | **+0.0053 [+0.0036, +0.0071]** | +0.0041 [+0.0029, +0.0053] |

Per language (final/, chrF): amh +0.79, hau +0.64, lug +0.60, yor +0.51 (intervals above 0), zul +0.26 [-0.09, +0.60], nso -0.03 [-0.34, +0.34]; held-out AfriCOMET is higher in all six (five with intervals above 0; hau +0.0015 [-0.0026, +0.0055]). NTREX / MAFAND chrF++ (final/ minus alpha 0.5): +0.42 [+0.37, +0.48] / +0.10 [+0.02, +0.17]. Finishing: AfriScience sentence finished rate 0.9957 (alpha 0.5) -> 0.9990 (final/); length ratio 1.039 -> 1.004. So the science RL is a small sentence-level gain beyond finishing; the science SFT remains the larger step (alpha 0.5 minus alpha 0: +1.23 chrF, +0.0052 AfriCOMET). At document level nothing is gained (chunked window SSA-COMET-QE 0.574 -> 0.574 / 0.577; same report). One seed.

## 6. What the stored data cannot prove, and what the controls would cost (task 5)

**Cannot be established from what exists**

1. **No matched-compute control.** The SFT was not continued for the 8.7 GPU-hours (553,600 rollouts, 34,600 prompts) the RL used. The comparison is "SFT" against "SFT + RL", so it does not show that RL is better use of the compute than more SFT. The finishing part in particular (SFT 83 % finished on FLORES+, 17 % unfinished or looping) looks like something more SFT with end-of-turn emphasis, or any decoding guard, could fix.
2. **No decoding control.** The SFT was decoded greedily with 768 new tokens and no repetition guard (no repetition penalty, no n-gram blocking, no stop on a repeated block). A decode-time loop guard on the SFT would remove part of the +3.9 chrF++ that loop-cutting recovers; it was not run.
3. **One seed.** The intervals are over test sentences. Variation between training runs (seed, prompt draw, sampling) is not measured; the dev macro's own noise was estimated at sd 0.09 to 0.10 chrF++ per evaluation, but that is evaluation noise, not training-run noise. The gain of +8 is large compared with plausible seed noise, the +2.4 beyond finishing less obviously so, and the per-language differences of under 1 chrF++ (pcm, hau, ibo) cannot be separated from it.
4. **Reward-family overlap.** AfriCOMET-STL is held out of the reward but is the same family as one reward member; the chrF++ gain is partly in the optimised family through consensus and, for 5 + 7 languages, the NLLB and human-reference terms. Both agree here (chrF++ +7.96, chrF +8.59, AfriCOMET +0.063), which is evidence against a pure metric artefact but not independent evidence of quality. No human evaluation is part of this analysis.
5. **The loop flag is the reward's own guard.** `is_repetitive` / `is_self_repetitive` feed the penalties during RL, so the RL is optimised against exactly this detector. The detector is quiet on the human references (0.01 % of FLORES+ rows) and the chrF++ and AfriCOMET results do not depend on it except through the subsets, but a loop type the detector misses would be undercounted for the RL.
6. **Survivor and selection effects** of the clean-row subsets (section 3); and the cut-at-loop rule is a heuristic (earliest block of up to 20 words repeated three times, else the second occurrence of the first 4-gram that occurs three times).
7. **Low-score languages.** For aeb, ary, zgh and several others both models score chrF++ 2 to 9; differences there say little about translation quality.
8. **Benchmarks were seen several times** (stage A, B1, mixed and weak-run evaluations; the "keep RL" decision used NTREX and MAFAND). Checkpoint choice itself used only the dev set, so test-set selection bias is small, but the reported test numbers are not from a single pre-registered look.
9. **Contamination:** the audit covers the 385M SFT text only; the base model's pretraining text is unknown, and it is the same for SFT and RL, so it does not change the paired comparison.
10. **Same-data caveat on the RL prompts:** the RL used English prompts from the clean pools with the protected-index filter; the 7 new-language prompt files that needed filtering are disclosed in `reports/rl_stage_A_385m_2026-11-03.md`. No benchmark source sentence is a prompt by that filter, which is a property of the filter, not independently re-checked here.

**Options for closing the gaps** (estimates only, nothing was run; the GPU is on the science RL, then the long-document RL, so each option displaces that queue)

Measured rates used: stage A 15,920 s for 400 steps (39.8 s per step, 64 prompts x 16 samples), B1 10,479 s for 135 steps (77.6 s per step, 48 prompts), mixed 4,786 s for 60 steps (79.8 s per step, 42 prompts); the chain A + B1 + mixed is 8.66 GPU-h on one A100 (B1 started from stage A's final, mixed from B1's best checkpoint). SFT throughput measured in the science SFT: ~16.5 k tokens/s at micro-batch 4 (65,536 tokens per step), other jobs sharing the GPU part of the time.

| option | what it tests | estimated GPU-h | notes |
|---|---|---:|---|
| A. continued SFT, matched wall-clock / GPU time | is RL a better use of 8.7 GPU-h than more SFT | **~8.7 h training (~510 M tokens at 16.5 k tok/s) + ~0.8 h evaluation** | needs a data and learning-rate choice (continue on the shard mix at the SFT's final LR, with and without end-of-turn emphasis); the faster the SFT code runs, the more tokens fit |
| A'. continued SFT, matched in rollouts / tokens | same data volume as the RL saw | ~1 h training (~55 M target tokens) + 0.8 h evaluation | there are no references for RL prompts, so the targets must be distilled (best-of-16 or consensus outputs): add ~3 to 4 GPU-h of generation and scoring (rough, 70 to 80 % of the 4.4 h of stage A), i.e. rejection-sampling fine-tuning ~5 h in all |
| B. decode-time loop guard on the SFT (repetition penalty / n-gram block / stop on repeated block) | how much of the finishing gain is a decoding problem | **~0.5 h** (inference of 122,585 rows + AfriCOMET), no training | cheapest and most informative first control |
| C. second RL seed on a subset | seed variance of the gain | stage A recipe, 100 steps (64 x 16): 1.1 h; with half the prompts per step on a 20-language subset ~0.6 to 0.8 h; + 0.5 h evaluation: **~1.2 to 1.7 h per seed** | most of the gain arrives by step 100 (dev +3.3 of +3.7), so 100 steps is enough to measure the gain; 2 extra seeds ~3 h |
| D. full replication of the chain | seed variance of the whole pipeline | 8.7 h (+ 0.8 h evaluation) | only if C shows large variation |

Recommended order if GPU time is made available: B (0.5 h, decides how much of the finishing gain needs RL at all), then C with two seeds (about 3 h), then A if the paper wants the "RL against continued SFT" claim.
