# Science SFT of the 385M model: interpolation sweep and evaluation (2026-11-04)

Arm 2 (science SFT, then RL). Start checkpoint: the mixed-stage best (dev chrF++ 40.90). SFT: AdamW 3e-6, 283 steps of 65,536 tokens (6.34 M AfriScience tokens in 6 languages, two passes, plus 12.19 M replay tokens over 109 languages), 17 minutes. Validation: AfriScience dev loss (macro of 6 languages) 1.699 -> 1.322, monotone, no early stop; the replay loss (76 languages, the forgetting monitor) 1.515 -> 1.355 (it *fell*: the RL-tuned start was further from the SFT distribution than the SFT end). The SFT end point (alpha 1.0) is not the best model: it loses NTREX 2.45 and MAFAND 1.20 chrF++ against the start, so the weights were interpolated against the start (WiSE-FT style), alpha in {0, 0.3, 0.5, 0.7, 1.0}. All numbers: greedy, AfriScience-MT sentence level (dev 843 and test 970 sentences per language, 6 languages), NTREX and MAFAND all rows. Data and builders: `reports/data/eval_science_sft_2026-11-04/`.

## Result
* **Headline (as in the AfriScience-MT paper): SSA-COMET, here SSA-COMET-QE; chrF (plain) second.** By the headline the science SFT gives **no gain**: test 0.6130 (alpha 0) vs 0.6118 (alpha 0.5), dev 0.6242 vs 0.6235 (the best alpha is 0.3 on dev, 0.6251, and 0 on test, all within 0.002); alpha 1.0 is clearly worse (0.5986 test).
* **Chosen alpha = 0.5, by the pre-set rule** (made before the headline metric was fixed): best AfriScience dev chrF subject to NTREX and MAFAND macro chrF++ not lower than the start (alpha 0) by more than 0.5. alpha 0.7 fails it (NTREX -0.73), alpha 1.0 fails it (NTREX -2.45, MAFAND -1.20); alpha 0.3 passes but is lower on dev chrF. The other metrics: AfriCOMET-STL, BLEU, spBLEU and chrF pick alpha 0.5 on dev and test; chrF++ picks 0.5 on dev, 0.7 on test (43.81 vs 43.80); SSA-COMET-MTL (reference-based) picks 0.3 by 0.0003 / 0.0009 (noise). The science RL was started from alpha 0.5.
* **Size of the effect at alpha 0.5**: +1.2 chrF on AfriScience test (45.40 -> 46.63), +1.1 chrF++, +1.5 BLEU, +2.0 spBLEU, +0.005 AfriCOMET-STL, +0.003 SSA-COMET-MTL, -0.006 SSA-COMET-QE (test); NTREX -0.26, MAFAND +0.27. The SFT alone leaves the 385M below the old 270M science-stage models on AfriScience test chrF (46.6 against 47.9 for the old RL-final, 50.0 for the old science-stage start, 50.2 for final2; Table 2): behind on hau, lug, nso, zul, ahead on amh and yor. The science RL is the stage that has to close this gap.

### Goodhart caution (read before comparing RL models)
The RL trains against a QE reward that contains a model of the same family as the headline metric, so SSA-COMET-QE gains obtained by the RL **overstate quality**. Exactly which checkpoint is where:
* **In the RL reward** (stages A, B1, mixed, weak group, science RL): `masakhane/africomet-qe-stl-1.1` (primary, reference-free QE) and `McGill-NLP/ssa-comet-mtl` (second ensemble member, run reference-free as QE); in the science RL additionally `McGill-NLP/ssa-comet-mtl` **reference-based** (src + mt + ref) on the 6 reference languages, a chrF++ term against the human reference, and recall / precision / length / preservation terms computed from the reference.
* **In the evaluation (not in any reward):** headline `McGill-NLP/ssa-comet-qe` (reference-free; the paper's variant, same family as the reward's ssa-comet-mtl but a different checkpoint), the held-out `masakhane/africomet-stl-1.1` (reference-based; a different checkpoint from the reward's africomet-qe-stl-1.1; it is also the RL's hacking-alarm monitor, never a reward), plain chrF / chrF++ / BLEU / spBLEU.
* **Not independent for RL models:** SSA-COMET-MTL (reference-based) is in the science RL reward; chrF++ against the reference is a reward term (weight 0.35) on the reference rows of the science RL, although the AfriScience TEST references are never used (the RL's reference pool is the train split). The independent checks of an RL gain are therefore chrF (plain) on held-out test sentences and the reference-based AfriCOMET-STL; a QE gain without them is not evidence of better translations. (The SFT alpha sweep above involves no RL, so its SSA-COMET-QE numbers are not affected.)

## Alpha sweep (all metrics)

Headline = **SSA-COMET-QE** (`McGill-NLP/ssa-comet-qe`, reference-free: the variant that reproduces the paper's released SSA-COMET values, calibration in `eval_afriscience_document_level_2026-11-04.md`), second = plain chrF (word order 0), as in the AfriScience-MT paper; chrF++, BLEU, spBLEU, AfriCOMET-STL and SSA-COMET-MTL (both reference-based) are extra columns. AfriScience-MT sentence level, macro over the 6 languages, greedy.

| alpha | split | **SSA-COMET-QE** (headline) | chrF (plain, secondary) | chrF++ | BLEU | spBLEU | AfriCOMET-STL (ref) | SSA-COMET-MTL (ref) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | dev | **0.6242** | 45.74 | 42.81 | 18.68 | 23.13 | 0.6988 | 0.5249 |
| 0.3 | dev | **0.6251** | 46.53 | 43.64 | 19.65 | 24.45 | 0.7031 | 0.5296 |
| 0.5 (chosen) | dev | **0.6235** | 47.17 | 44.35 | 20.34 | 25.48 | 0.7044 | 0.5293 |
| 0.7 | dev | **0.6200** | 46.96 | 44.22 | 20.00 | 25.38 | 0.7011 | 0.5264 |
| 1.0 | dev | **0.6117** | 46.02 | 43.36 | 18.15 | 22.87 | 0.6902 | 0.5144 |
| 0 | test | **0.6130** | 45.40 | 42.44 | 17.72 | 21.89 | 0.6828 | 0.5123 |
| 0.3 | test | **0.6128** | 46.28 | 43.40 | 18.73 | 23.25 | 0.6865 | 0.5159 |
| 0.5 (chosen) | test | **0.6118** | 46.63 | 43.80 | 19.27 | 23.87 | 0.6879 | 0.5150 |
| 0.7 | test | **0.6092** | 46.56 | 43.81 | 18.98 | 23.83 | 0.6859 | 0.5114 |
| 1.0 | test | **0.5986** | 44.92 | 42.30 | 16.78 | 20.87 | 0.6708 | 0.4949 |

**The science SFT shows NO SSA-COMET-QE gain.** Test: 0.6130 at alpha 0 (the start, no science SFT) against 0.6118 at alpha 0.5; dev 0.6242 against 0.6235. The best alpha by the headline metric is 0.3 on dev and 0 on test, but every alpha from 0 to 0.5 lies within 0.002 of the others (noise), and alpha 1.0 (the SFT end point) is clearly worse (0.5986 on test). The gains of alpha 0.5 in chrF (+1.2 on test), BLEU, spBLEU and AfriCOMET-STL (+0.005) are gains in surface agreement with the references that the paper's primary metric does not register.

#### Best alpha per metric

* dev: chrF: 0.5; chrF++: 0.5; BLEU: 0.5; spBLEU: 0.5; AfriCOMET-STL: 0.5; SSA-COMET-MTL: 0.3; SSA-COMET-QE: 0.3
* test: chrF: 0.5; chrF++: 0.7; BLEU: 0.5; spBLEU: 0.5; AfriCOMET-STL: 0.5; SSA-COMET-MTL: 0.3; SSA-COMET-QE: 0

#### Per language, AfriScience test (SSA-COMET-QE / chrF / AfriCOMET-STL)

| alpha | amh | hau | lug | nso | yor | zul |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.581 / 29.1 / 0.600 | 0.581 / 55.2 / 0.680 | 0.602 / 38.1 / 0.624 | 0.656 / 52.1 / 0.713 | 0.649 / 37.7 / 0.761 | 0.610 / 60.2 / 0.718 |
| 0.5 | 0.583 / 29.9 / 0.606 | 0.584 / 56.1 / 0.693 | 0.600 / 39.3 / 0.631 | 0.652 / 54.7 / 0.716 | 0.641 / 38.1 / 0.759 | 0.611 / 61.7 / 0.723 |
| 1.0 | 0.570 / 27.8 / 0.587 | 0.576 / 52.4 / 0.677 | 0.590 / 38.9 / 0.613 | 0.633 / 52.5 / 0.690 | 0.613 / 36.1 / 0.734 | 0.611 / 61.7 / 0.724 |

## Table 1. Interpolation sweep: theta(alpha) = (1 - alpha) theta_start + alpha theta_science-SFT (alpha 0 = the start checkpoint)

| alpha | AfriScience dev chrF (macro 6) | AfriScience test chrF | NTREX macro chrF++ | d vs alpha 0 | MAFAND macro chrF++ | d vs alpha 0 | passes non-regression (>= start - 0.5) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 45.74 | 45.40 | 37.16 | +0.00 | 40.78 | +0.00 | yes |
| 0.3 | 46.53 | 46.28 | 37.13 | -0.04 | 41.06 | +0.28 | yes |
| 0.5 (chosen) | 47.17 | 46.63 | 36.90 | -0.26 | 41.05 | +0.27 | yes |
| 0.7 | 46.96 | 46.56 | 36.43 | -0.73 | 40.83 | +0.05 | no |
| 1.0 | 46.02 | 44.92 | 34.71 | -2.45 | 39.58 | -1.20 | no |

Chosen alpha: **0.5** (rule: best AfriScience dev chrF subject to NTREX and MAFAND not below the start by more than 0.5).

## Table 2. AfriScience-MT test, sentence level, plain chrF (word order 0), greedy: now a FAIR comparison with the old 270M (both have had a science stage)

| system | amh | hau | lug | nso | yor | zul | macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| 385M alpha 0 (start, no science SFT) | 29.1 | 55.2 | 38.1 | 52.1 | 37.7 | 60.2 | 45.4 |
| 385M alpha 0.5 (chosen) | 29.9 | 56.1 | 39.3 | 54.7 | 38.1 | 61.7 | 46.6 |
| old 270M RL-final (1,433) | 25.3 | 59.4 | 46.2 | 55.9 | 37.4 | 63.4 | 47.9 |
| old 270M science-stage start (A2W) | 26.2 | 62.0 | 48.9 | 58.3 | 39.8 | 64.7 | 50.0 |
| old 270M final2 (step 401) | 26.6 | 61.9 | 49.3 | 58.5 | 40.1 | 64.7 | 50.2 |
| NLLB-200-1.3B fine-tuned (published) | 45.5 | 67.7 | 51.5 | 64.3 | 17.7 | 67.2 | 52.3 |
| NLLB-200-distilled-600M fine-tuned (published) | 44.4 | 66.9 | 50.6 | 64.2 | 17.6 | 66.3 | 51.7 |
| M2M100-1.2B fine-tuned (published) | 39.0 | 63.1 | 49.5 | 62.4 | 16.6 | 58.4 | 48.2 |

## Table 3. NTREX / MAFAND non-regression on the clean rows of the old models (chrF++, languages both cover)

| benchmark | languages | old SFT (clean) | old RL (clean) | 385M start (alpha 0) | 385M chosen alpha |
|---|---:|---:|---:|---:|---:|
| NTREX | 24 | 31.5 | 35.6 | 37.2 | 36.9 |
| MAFAND | 15 | 38.2 | 41.8 | 40.8 | 41.1 |


## Caveats
* The metric set of the AfriScience-MT paper (from the dataset card and the released `metrics` config, as read in `reports/afriscience_mt_2026-10-21.md`): SSA-COMET as the primary metric ("COMET"), chrF as a surface check, BLEU computed but not reported; the paper's sacrebleu chrF settings are not stated in the text we could read. Plain chrF (word order 0) is shown next to chrF++; which SSA-COMET checkpoint the paper used is not verified: `McGill-NLP/ssa-comet-mtl` (the one in our reward ensemble) is scored with src + mt + ref, mean of the three views, and AfriCOMET with `masakhane/africomet-stl-1.1` (the held-out monitor of the RL stages, never a reward model). Both neural scores are language means of sentence scores (macro over 6 languages), no sentence-level intervals.
* The replay loss exists for alpha 0 (the start, 1.5146) and alpha 1.0 (the SFT end, 1.3546) only; the interpolated models were not scored on the replay split. NTREX / MAFAND AfriCOMET of the chosen model was not run (the GPU is used by the science RL and the CPU path starved the RL; the RL's own dev evaluations log the held-out AfriCOMET).
* Process notes: a parallel pre-run of two alphas failed (vLLM could not start next to the sequential run, fell back to slow HF generation, killed) and the alpha 0.5 AfriScience evaluation of the pipeline died in that overlap, so it was rerun; the first pipeline pass then wrote an empty alpha, which launched the science RL with an invalid model path (it stopped itself at start, nothing trained) before the choice was recomputed. The first neural-metric pass on the GPU made the science RL run out of memory (restart at ladder 3); the CPU variant starved the RL and was killed; the final pass runs under a 5.5 GB cap in about 6 minutes.
* One seed, one SFT run, greedy decoding.
