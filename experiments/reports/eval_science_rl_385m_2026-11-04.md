# Science RL (385M, stage `final` of the science config): evaluation of best and final (2026-11-04)

Run: `runs_385m_sci/final`, started from the science-SFT alpha-0.5 interpolation (`models/sci_sft_start_385m`), Muon lr 1e-6, KL coef 0.1, guard 0.25, group 16, 42 prompts per step, science 0.4 / reference 0.3 / general 0.3 prompts of 64 and 256 tokens, reward = QE ensemble (`africomet-qe-stl-1.1` + `ssa-comet-mtl`) + reference-based SSA-COMET-MTL term + chrF++ against the human reference (weight 0.35, AfriScience-MT TRAIN pairs) + recall / precision / length / preservation terms (`science_terms.py`). It ran 160 steps (150 + an extension) and stopped on the combined dev rule (`dev_plateau`, patience 50 steps); the hacking alarm never fired. Hub: branch `science-rl` (best), `science-rl-latest`. Evaluation: as armed (`post_sci_rl_eval.sh`: sentence dev/test, NTREX/MAFAND, 168 documents whole + chunked, neural passes) plus the paired comparison `sci_pairs.py` (sentence-level paired bootstrap). Data and scripts: `reports/data/eval_science_rl_2026-11-04/`.

## 0. Short answer

* **The science RL gives a small but real sentence-level improvement over its start, and none at document level.** AfriScience-MT TEST, macro over 6 languages, final/ minus science SFT alpha 0.5: chrF **+0.46 [+0.27, +0.65]**, held-out AfriCOMET-STL **+0.0053 [+0.0036, +0.0071]** (independent of the reward), SSA-COMET-QE +0.0041 [+0.0029, +0.0053] (reward family, shown because it is the AfriScience-MT paper's headline). `best/` (step 75) is smaller and its test chrF gain is not distinguishable from zero: +0.16 [-0.04, +0.37] (AfriCOMET +0.0013 [-0.0005, +0.0031]); `final/` is better than `best/` on both dev and test (+0.30 chrF [+0.11, +0.49] and +0.0039 AfriCOMET on test).
* **Against the grandparent (mixed best = alpha 0) the gain is +1.4 chrF**, but most of it comes from the science SFT (+1.23 chrF, +0.0052 AfriCOMET), not from the RL.
* **SSA-COMET-QE, the paper's headline, moves by +0.004** (0.6118 -> 0.6158 on test) after the science RL, and the science SFT had shown no gain (0.6130 -> 0.6118). On this metric the whole science stage adds about +0.003 over the mixed best, and that metric is in the reward family of the science RL (its encoder family is the reward's `ssa-comet-mtl`): read it next to chrF and the held-out AfriCOMET.
* **Non-regression holds:** NTREX chrF++ +0.42 [+0.37, +0.48] and MAFAND +0.10 [+0.02, +0.17] for final/ against alpha 0.5; for best/ no language is significantly worse on chrF++ (per-language cells of the primary pair; final/ per language not tabulated).
* **Document level (168 test documents): no gain.** Chunked (sentence by sentence + best-of-4 repair): window SSA-COMET-QE 0.574 (alpha 0.5) -> 0.574 (best) / 0.577 (final); chrF 58.8 -> 58.7 / 58.7. Whole-document generation still fails: finished 0.51 -> 0.62 (best) / 0.70 (final), chrF 24.9 -> 26.7 / 27.4, loop rate 0.86 -> 0.85. The science RL trained on prompts of at most 256 source tokens, so this is expected; it is the reason for the long-document stage.
* **Comparison with the old 270M (now fair: both have a science stage):** the 385M science RL final is below the old 270M models on AfriScience-MT sentence-level TEST chrF: 47.10 against 47.9 (old RL-final), 50.0 (old science-stage start) and 50.2 (old final2) (stored numbers of the first research, other code state, not re-run); at document level (chunked) the old rl2c is also ahead: window SSA-COMET-QE 0.593 against 0.577.

## 1. Training record (trainer's own dev, 200 sentences per language of AfriScience-MT dev, greedy)

| step | dev chrF++ macro | held-out AfriCOMET | finished |
|---:|---:|---:|---:|
| 0 | 44.618 | - | 0.9967 |
| 25 | 44.787 | 0.7044 | 0.9983 |
| 50 | 44.914 | 0.7069 | 0.9967 |
| 75 | 45.190 | 0.7096 | 1.0000 |
| 100 | 45.057 | 0.7070 | 1.0000 |
| 125 | 45.204 | 0.7092 | 1.0000 |
| 150 | 45.049 | 0.7087 | 0.9992 |
| 160 (end) | 45.131 | 0.7087 | 0.9992 |

`best/` = step 75 (the trainer saves a new best only for a gain of 0.2 chrF++; step 125's 45.204 is not 0.2 above 45.190). The trainer's dev subset moves by +0.5 chrF++ in 160 steps, with a noise of about 0.1 per evaluation; the whole dev split (5,058 sentences) separates `best/` and `final/` (Table S2). Known issue left unchanged mid-run: the reward's recall / precision floors (0.5) penalised 76-83 % of the reference rows (mean adjustment -0.40), which compresses the within-group differences on those rows.

## 2. Sentence level, AfriScience-MT

## Table S1a. AfriScience-MT TEST, sentence level, greedy, macro over the 6 languages (amh hau lug nso yor zul)

| model | chrF (word order 0) | chrF++ | BLEU | finished | length ratio | AfriCOMET-STL (ref, held out) | SSA-COMET-QE (reward family) | SSA-COMET-MTL ref (reward family) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 385M mixed best = science-SFT alpha 0 (science RL's grandparent) | 45.40 | 42.44 | 17.72 | 0.9978 | 1.010 | 0.6828 | 0.6130 | 0.5123 |
| 385M science SFT, alpha 0.5 (**the science RL's start**) | 46.63 | 43.80 | 19.27 | 0.9957 | 1.039 | 0.6879 | 0.6118 | 0.5150 |
| 385M science RL, best/ (step 75) | 46.79 | 43.93 | 19.20 | 0.9978 | 1.021 | 0.6893 | 0.6135 | 0.5185 |
| 385M science RL, final/ (step 160) | 47.10 | 44.22 | 19.53 | 0.9990 | 1.004 | 0.6932 | 0.6158 | 0.5224 |

## Table S1b. AfriScience-MT DEV, sentence level, greedy, macro over the 6 languages (amh hau lug nso yor zul)

| model | chrF (word order 0) | chrF++ | BLEU | finished | length ratio | AfriCOMET-STL (ref, held out) | SSA-COMET-QE (reward family) | SSA-COMET-MTL ref (reward family) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 385M mixed best = science-SFT alpha 0 (science RL's grandparent) | 45.74 | 42.81 | 18.68 | 0.9984 | 0.996 | 0.6988 | 0.6242 | 0.5249 |
| 385M science SFT, alpha 0.5 (**the science RL's start**) | 47.17 | 44.35 | 20.34 | 0.9970 | 1.023 | 0.7044 | 0.6235 | 0.5293 |
| 385M science RL, best/ (step 75) | 47.38 | 44.51 | 20.53 | 0.9986 | 1.002 | 0.7067 | 0.6255 | 0.5326 |
| 385M science RL, final/ (step 160) | 47.62 | 44.74 | 20.64 | 0.9994 | 0.994 | 0.7098 | 0.6277 | 0.5360 |

## Table S2. Paired differences, AfriScience-MT sentence level (paired Poisson bootstrap over sentences, 1000 resamples, macro over the 6 languages; 95 % CI)

| comparison | split | chrF | AfriCOMET-STL (ref, held out) | SSA-COMET-QE (reward family) |
|---|---|---|---|---|
| science RL best minus science SFT alpha 0.5 | test | +0.16 [-0.04, +0.37] | +0.0013 [-0.0005, +0.0031] | +0.0018 [+0.0006, +0.0030] |
| science RL best minus science SFT alpha 0.5 | dev | +0.21 [+0.01, +0.43] | +0.0023 [+0.0005, +0.0041] | +0.0021 [+0.0009, +0.0032] |
| science RL final minus science SFT alpha 0.5 | test | +0.46 [+0.27, +0.65] | +0.0053 [+0.0036, +0.0071] | +0.0041 [+0.0029, +0.0053] |
| science RL final minus science SFT alpha 0.5 | dev | +0.45 [+0.23, +0.66] | +0.0054 [+0.0034, +0.0072] | +0.0043 [+0.0030, +0.0055] |
| science RL best minus alpha 0 (mixed best) | test | +1.39 [+1.17, +1.63] | +0.0065 [+0.0046, +0.0084] | +0.0005 [-0.0008, +0.0017] |
| science RL best minus alpha 0 (mixed best) | dev | +1.64 [+1.43, +1.85] | +0.0079 [+0.0061, +0.0099] | +0.0013 [+0.0001, +0.0026] |
| science SFT alpha 0.5 minus alpha 0 (for scale) | test | +1.23 [+1.00, +1.47] | +0.0052 [+0.0031, +0.0070] | -0.0012 [-0.0025, -0.0000] |
| science SFT alpha 0.5 minus alpha 0 (for scale) | dev | +1.43 [+1.19, +1.67] | +0.0056 [+0.0036, +0.0078] | -0.0007 [-0.0020, +0.0007] |
| science RL final minus best | test | +0.30 [+0.11, +0.49] | +0.0039 [+0.0025, +0.0055] | +0.0023 [+0.0012, +0.0034] |
| science RL final minus best | dev | +0.24 [+0.07, +0.41] | +0.0031 [+0.0016, +0.0047] | +0.0022 [+0.0011, +0.0033] |

## Table S3. Per language, TEST, science RL final minus science SFT alpha 0.5

| iso | chrF alpha 0.5 | chrF RL final | diff [CI] | AfriCOMET alpha 0.5 | AfriCOMET RL final | diff [CI] |
|---|---:|---:|---|---:|---:|---|
| amh | 29.90 | 30.69 | +0.79 [+0.34, +1.22] | 0.6061 | 0.6154 | +0.0093 [+0.0038, +0.0147] |
| hau | 56.10 | 56.74 | +0.64 [+0.01, +1.28] | 0.6927 | 0.6941 | +0.0015 [-0.0026, +0.0055] |
| lug | 39.27 | 39.87 | +0.60 [+0.01, +1.20] | 0.6312 | 0.6378 | +0.0066 [+0.0015, +0.0120] |
| nso | 54.72 | 54.69 | -0.03 [-0.34, +0.34] | 0.7156 | 0.7205 | +0.0049 [+0.0016, +0.0085] |
| yor | 38.11 | 38.62 | +0.51 [+0.21, +0.83] | 0.7594 | 0.7647 | +0.0053 [+0.0025, +0.0082] |
| zul | 61.71 | 61.97 | +0.26 [-0.09, +0.60] | 0.7227 | 0.7268 | +0.0042 [+0.0010, +0.0073] |


Reward-family labels: the science RL reward contains `africomet-qe-stl-1.1`, `ssa-comet-mtl` (reference-free and reference-based) and a chrF++ term against human references of AfriScience-MT TRAIN pairs in the same six languages. **Independent of the reward: the held-out reference-based `masakhane/africomet-stl-1.1`** (same family and AfriMTE training data as the QE reward member, so independent in weights and use of the reference, not in family) **and, with the caveat that the reward trains chrF++ on the train split of the same languages and domain, chrF on the test split.** SSA-COMET-QE and SSA-COMET-MTL(ref) are reward family and are not evidence of quality. All three move in the same direction (final over alpha 0.5: chrF +0.46, AfriCOMET +0.0053, SSA-QE +0.0041, SSA-MTL(ref) +0.0074 on test), which excludes a pure artefact of one metric but is not human evaluation.

## 3. NTREX and MAFAND (non-regression; chrF++, macro over languages, 214 contamination-flagged rows removed, paired bootstrap)

| comparison | NTREX (24 languages) | MAFAND (15 languages) |
|---|---|---|
| science RL best minus alpha 0.5 | +0.31 [+0.26, +0.37] | +0.10 [+0.03, +0.18] |
| science RL final minus alpha 0.5 | +0.42 [+0.37, +0.48] | +0.10 [+0.02, +0.17] |
| science RL final minus alpha 0 (mixed best) | +0.16 [+0.11, +0.22] | +0.37 [+0.31, +0.44] |
| alpha 0.5 minus alpha 0 (for scale) | -0.26 [-0.33, -0.19] | +0.27 [+0.19, +0.36] |
| absolute, final/ | 37.33 | 41.15 |

The science RL repairs the small NTREX loss of the science SFT (-0.26) and ends above the mixed best on both benchmarks. Finished rate 0.9997 / 0.9999, loop rate 0.0078 / 0.0014 (NTREX / MAFAND) for final/. No AfriCOMET for these runs (generation only).

## 4. Document level, 168 test documents (28 papers x 6 languages)

## Table 1. Document level, 168 test documents (28 papers x 6 languages), macro over the 6 languages

**Headline = SSA-COMET-QE on 4-sentence WINDOWS (the honest document score), then plain chrF.** The COMET encoders only read the first 512 tokens, so the *concatenated* document score (`concat`) sees only the first part of a ~1,000-token document and is NOT a full-document score; it is shown only because it is the literal 'metric on the concatenated document'. Windows need a sentence-aligned hypothesis, so they exist for the `chunked` mode only; for `whole` mode (unaligned output) only the truncated `concat` value exists and chrF / failure statistics are the real evidence.

| model | mode | **SSA-COMET-QE window** | chrF | AfriCOMET window | SSA-MTL(ref) window | SSA-QE concat (first 512 tokens) | AfriCOMET concat | SSA-MTL(ref) concat | chrF++ | BLEU | spBLEU | finished | loop | length ratio | tail coverage | concat COMET x100 - chrF |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 385M start (alpha 0, no science SFT) | whole | - | 22.8 | - | - | 0.378 | 0.470 | 0.214 | 20.0 | 4.0 | 5.4 | 0.71 | 0.86 | 1.03 | 0.27 | 15.0 |
| 385M start (alpha 0, no science SFT) | chunked | **0.574** | 57.3 | 0.687 | 0.448 | 0.436 | 0.661 | 0.219 | 53.0 | 20.0 | 25.8 | 1.00 | 0.14 | 0.99 | 0.61 | -13.7 |
| 385M science SFT (alpha 0.5) | whole | - | 24.9 | - | - | 0.381 | 0.495 | 0.208 | 22.1 | 5.2 | 6.5 | 0.51 | 0.86 | 1.30 | 0.30 | 13.2 |
| 385M science SFT (alpha 0.5) | chunked | **0.574** | 58.8 | 0.692 | 0.448 | 0.434 | 0.661 | 0.213 | 54.7 | 22.0 | 28.5 | 1.00 | 0.23 | 1.01 | 0.63 | -15.4 |
| old 270M RL-final (rl2c, first research) | whole | - | 34.9 | - | - | 0.420 | 0.599 | 0.268 | 30.9 | 6.9 | 10.2 | 0.70 | 0.61 | 1.10 | 0.41 | 7.2 |
| old 270M RL-final (rl2c, first research) | chunked | **0.593** | 59.3 | 0.703 | 0.468 | 0.457 | 0.682 | 0.244 | 55.0 | 22.5 | 29.3 | 1.00 | 0.11 | 1.00 | 0.64 | -13.6 |
| 385M science RL (best) | whole | - | 26.7 | - | - | 0.388 | 0.519 | 0.216 | 23.8 | 5.7 | 7.9 | 0.62 | 0.85 | 1.19 | 0.31 | 12.1 |
| 385M science RL (best) | chunked | **0.574** | 58.7 | 0.692 | 0.450 | 0.437 | 0.661 | 0.216 | 54.5 | 21.9 | 28.3 | 1.00 | 0.18 | 1.00 | 0.63 | -14.9 |
| 385M science RL (final) | whole | - | 27.4 | - | - | 0.398 | 0.531 | 0.228 | 24.4 | 5.8 | 8.1 | 0.70 | 0.85 | 1.12 | 0.32 | 12.4 |
| 385M science RL (final) | chunked | **0.577** | 58.7 | 0.696 | 0.455 | 0.440 | 0.669 | 0.217 | 54.6 | 21.9 | 28.3 | 1.00 | 0.20 | 1.00 | 0.63 | -14.8 |

`whole` = the document in one generation (greedy, 4,096 new tokens); `chunked` = sentence by sentence (greedy) + best-of-4 repair of unfinished sentences, sentences joined by single spaces. finished = share of documents (whole) or of documents with every sentence finished (chunked); loop = repetition detector (rl/guards); tail coverage = chrF recall of the last third of the reference against the hypothesis (does the output reach the end of the paper); the last column is the paper's degeneration flag computed with the truncated concat value (a COMET far above chrF means fluent but off-target output).


The honest document score is the 4-sentence-window SSA-COMET-QE of the chunked mode (concatenated COMET scores see only the first 512 tokens); the ranking by window SSA-COMET-QE is old rl2c 0.593 > science RL final 0.577 > start = alpha 0.5 = science RL best 0.574. SSA-COMET-QE is reward family for the science RL (see section 2).

## 5. What this does and does not show

* A small gain at sentence level on held-out data with metrics that are not the reward (chrF +0.46, AfriCOMET +0.0053), larger for `final/` than for `best/`; no regression on NTREX / MAFAND; no document-level effect; no change in the headline SSA-COMET-QE beyond +0.004 (reward family).
* One run, one seed: the intervals are over test sentences only. The science RL's +0.5 dev chrF++ over 160 steps is of the size of the B1 and mixed steps whose reality needs a seed estimate (E-04).
* The `best/` choice by the trainer (min gain 0.2 on a 200-sentence subset) was worse than the last checkpoint on the whole dev split; the long-document stage therefore starts from `final/` (rule refined 2026-11-04 on DEV data only: final/ wins when its dev chrF is higher by >= 0.2 and the held-out AfriCOMET is not lower by more than 0.004; full dev set: +0.24 chrF [+0.07, +0.41], +0.0031 AfriCOMET [+0.0016, +0.0047]; `science_sft/long_start_choice.json`). The change was made after seeing dev and test results of this evaluation; the test split was not used for the rule.
