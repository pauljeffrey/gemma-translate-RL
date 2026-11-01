# SSA-COMET-QE vs AfriCOMET-QE as the Gemma-MT RL reward (2026-10-15)

**Status: the main experiment is COMPLETE (49 languages x 2 models, all scored); the run was stopped on the owner's request only for two optional extras.** Missing / incomplete: the reverse-orientation side check lacks AfriCOMET `nya`; no chrF++ correlation (see section 9). Everything below is CPU-only fp32; no GPU, nothing pushed, no data copied off the box. Raw per-pair scores: `/workspace/sft_build/ssa_cmp/out/{africomet,ssacomet}/<iso>.jsonl`, resume notes: `/workspace/sft_build/ssa_cmp/RESUME.md`.

Models: `masakhane/africomet-qe-stl-1.1` (AfriCOMET-QE, in the RL now) and `McGill-NLP/ssa-comet-qe` (SSA-COMET-QE). Both are the same COMET architecture (`referenceless_regression_metric`, afro-xlmr-large-76L encoder, avg pooling, layer mix/sparsemax, hidden 3072-1024); only the weights differ.

## 1. Bottom line

* **Do not swap.** SSA-COMET-QE is *not* a better reward than AfriCOMET-QE-1.1 in any of the 49 languages tested. Human-vs-mismatched AUC is lower for SSA-COMET in 48 of 49 languages (the exception is nqo, +0.007, where both are at chance; mean -0.039, range -0.082 .. +0.007; 34 of 49 differences have a 90% bootstrap CI below zero, 15 are within noise, none above). Mean failure rate on the "hard" classes (mismatched, neighbour, half-omitted, appended, swapped names/numbers, replaced words) is higher for SSA-COMET in 49/49 languages (mean +0.048).
* This holds on the 12 languages SSA-COMET was explicitly fine-tuned on (amh hau ibo kik kin luo twi yor zul ewe lin wol): mean AUC change -0.040, the same as on the other 37 (-0.038). So the "fine-tuned on 12" claim does not translate into a better en->xx QE here.
* **Pidgin: SSA-COMET is worse** (AUC 0.639 vs 0.705; mismatched failure rate 0.40 vs 0.27). The only Pidgin class where it is nominally better is copying the English source (0.01 vs 0.02 failures, 0.08 vs 0.11 for a lightly edited copy) and that is noise-level and already covered by the copy guard. Twi (the other weak language): 0.629 vs 0.678, also worse. Yoruba: equal on AUC (0.789 vs 0.802, CI includes 0) but more failures.
* **The extra languages do not need the new model.** AfriCOMET-QE-1.1 itself scores the ~37 "unsupported" languages with AUC >= 0.80 (point estimate, n=60) in 23 of 37 and 0.70-0.80 in 11 more (dyu 0.68, sag 0.69 and nqo 0.54 are below 0.70); its model card lists the same 76 languages as SSA-COMET's. COMET QE takes no language code (text only), so "AfriCOMET cannot score language X" is a validity policy in `languages.py`, not a model limit. What the new languages need is **per-language calibration** (and removal of the 0.46 floor, section 7).
* SSA-COMET is more compressed (human P10..P90 spread averages 0.15 vs 0.25) and shifted up (only 6% of human references of the new languages fall below 0.46, vs 41% for AfriCOMET). That is a scale difference, not better discrimination; the AUC/failure columns are scale free.
* Where SSA-COMET is better: copy-the-source robustness in the most under-resourced languages (copy failure rate taq 0.20 -> 0.00, cjk 0.17 -> 0.00, fuv 0.10 -> 0.02, nqo 0.80 -> 0.45). Not enough to justify a swap (the repo has a hard copy guard).

## 2. What was run

* **Data (human references, English source -> African language).** 12 MAFAND languages from `gemma_calibration/mafand_dev.jsonl` (the file the earlier AfriCOMET calibration used; a contiguous block from the middle of the dev file: pcm 150, twi 150, yor 100, nya 100, the other 8 x 80 sentences). 37 further languages from FLORES+ **dev** (`openlanguagedata/flores_plus`, downloaded to the box with the read token; 60 consecutive sentences from position 100 of 997 per language). `bench/bench_dev_clean.jsonl` only has 70 FLORES rows for 7 of these languages and nothing for the other 30, so I used the full FLORES+ dev; the contamination filter concerns SFT-data overlap of the English side, which does not affect a QE-model validity test (it would matter for the chrF++ test, not done).
* **Pairs.** Positive: (en source, human reference). Corruptions of the *same* source, definitions copied from `pcm_calib_exp3.py`: `mismatch` (reference of another sentence of the block, offset 7+), `neighbour` (reference of a sentence 1-3 positions away), `copy_source` (English source as output), `drop_half`, `shuffle`, `append_unrelated` (correct translation + another sentence), `swap_numbers_names`, `replace_words` (25% random words), plus one new class `copy_noisy` (source with 20% of the words dropped; the realistic Pidgin failure). MAFAND languages: all 9 classes; FLORES languages: mismatch, neighbour, copy_source, drop_half, append_unrelated, replace_words (shorter set to fit the CPU budget). 27,891 pairs per model, identical for both models.
* **Metrics.** AUC = P(score(human) > score(mismatched reference)) over all human x mismatched combinations, ties 0.5 (same as `calibrate_shaping.auc`; 90% bootstrap CI over sentences, 600 resamples; the CI of the difference is paired). Failure rate per class = share of sentences whose corrupted version scores >= the human reference of the same sentence (the GRPO-relevant "same-source" criterion of D-24). Scores are exact: sentence embeddings were computed once per unique text through the same COMET path (`encoder.prepare_sample` -> `get_sentence_embedding` -> `estimator`), validated against `model.predict()` (max difference 1e-7 / 4e-7).
* **Verdict rule.** "SSA-COMET better/worse" if the 90% paired CI excludes 0 for AUC or for the mean hard-class failure rate (and none of the other signals goes the opposite way); otherwise "equal (within noise)". Note with n = 60-150 the AUC CI is about +-0.04 to +-0.09, so small differences are not resolvable per language; the consistency across 49 languages is the evidence.

## 3. Results: the 12 MAFAND languages (fail rates = mismatched / neighbour / copy / half-omitted / appended)

| lang | n sent. | AUC human-vs-mismatched AfriCOMET [CI90] | AUC SSA-COMET [CI90] | fail rate mism/neigh/copy/half/append, AfriCOMET | same, SSA-COMET | human P10/P50/P90 AfriC | human P10/P50/P90 SSA | verdict (SSA vs AfriC) |
|---|---|---|---|---|---|---|---|---|
| pcm | 150 | 0.705 [0.66,0.75] | 0.639 [0.59,0.69] | 0.27/0.31/0.02/0.05/0.55 | 0.40/0.35/0.01/0.08/0.61 | 0.23/0.37/0.52 | 0.35/0.44/0.54 | SSA-COMET worse |
| twi | 150 | 0.678 [0.63,0.72] | 0.629 [0.58,0.68] | 0.31/0.29/0.01/0.03/0.44 | 0.39/0.34/0.00/0.03/0.55 | 0.38/0.51/0.62 | 0.54/0.62/0.69 | SSA-COMET worse |
| yor | 100 | 0.802 [0.77,0.84] | 0.789 [0.75,0.84] | 0.10/0.17/0.00/0.00/0.22 | 0.13/0.18/0.00/0.00/0.37 | 0.50/0.62/0.73 | 0.61/0.69/0.73 | SSA-COMET worse |
| nya | 100 | 0.835 [0.80,0.87] | 0.797 [0.76,0.84] | 0.16/0.23/0.01/0.02/0.33 | 0.22/0.32/0.00/0.01/0.46 | 0.48/0.58/0.70 | 0.55/0.64/0.70 | SSA-COMET worse |
| hau | 80 | 0.877 [0.84,0.92] | 0.844 [0.80,0.89] | 0.06/0.09/0.00/0.00/0.29 | 0.09/0.15/0.00/0.00/0.46 | 0.60/0.72/0.82 | 0.61/0.69/0.74 | SSA-COMET worse |
| ibo | 80 | 0.874 [0.83,0.92] | 0.829 [0.77,0.88] | 0.09/0.16/0.01/0.01/0.30 | 0.19/0.21/0.00/0.03/0.51 | 0.55/0.66/0.79 | 0.58/0.67/0.73 | SSA-COMET worse |
| kin | 80 | 0.862 [0.82,0.90] | 0.818 [0.77,0.86] | 0.09/0.11/0.00/0.00/0.24 | 0.10/0.10/0.00/0.00/0.33 | 0.55/0.66/0.77 | 0.61/0.67/0.73 | SSA-COMET worse |
| sna | 80 | 0.889 [0.85,0.93] | 0.832 [0.78,0.88] | 0.07/0.10/0.00/0.01/0.30 | 0.12/0.17/0.00/0.01/0.40 | 0.58/0.66/0.75 | 0.61/0.68/0.73 | SSA-COMET worse |
| swh | 80 | 0.922 [0.89,0.95] | 0.918 [0.88,0.95] | 0.07/0.07/0.00/0.01/0.30 | 0.06/0.09/0.00/0.01/0.46 | 0.59/0.75/0.85 | 0.62/0.70/0.75 | SSA-COMET worse |
| tsn | 80 | 0.873 [0.83,0.91] | 0.807 [0.76,0.86] | 0.10/0.16/0.00/0.00/0.25 | 0.15/0.20/0.00/0.00/0.35 | 0.56/0.68/0.80 | 0.57/0.67/0.73 | SSA-COMET worse |
| xho | 80 | 0.909 [0.87,0.94] | 0.859 [0.81,0.90] | 0.06/0.12/0.00/0.01/0.31 | 0.10/0.19/0.00/0.01/0.40 | 0.55/0.69/0.80 | 0.56/0.65/0.70 | SSA-COMET worse |
| zul | 80 | 0.798 [0.75,0.85] | 0.742 [0.69,0.81] | 0.25/0.25/0.00/0.01/0.45 | 0.28/0.23/0.00/0.01/0.55 | 0.42/0.59/0.73 | 0.46/0.58/0.67 | SSA-COMET worse |


## 4. Results: 37 new languages (FLORES+ dev, n = 60 sentences)

| lang | n sent. | AUC human-vs-mismatched AfriCOMET [CI90] | AUC SSA-COMET [CI90] | fail rate mism/neigh/copy/half/append, AfriCOMET | same, SSA-COMET | human P10/P50/P90 AfriC | human P10/P50/P90 SSA | verdict (SSA vs AfriC) | SSA usable as new-language QE? |
|---|---|---|---|---|---|---|---|---|---|
| amh | 60 | 0.971 [0.95,0.99] | 0.939 [0.91,0.97] | 0.00/0.05/0.00/0.00/0.27 | 0.07/0.08/0.00/0.00/0.38 | 0.58/0.72/0.80 | 0.55/0.65/0.70 | SSA-COMET worse | AUC ok, fails worse than AfriCOMET |
| tir | 60 | 0.887 [0.84,0.93] | 0.849 [0.79,0.90] | 0.13/0.12/0.00/0.00/0.33 | 0.17/0.15/0.00/0.00/0.45 | 0.49/0.66/0.75 | 0.54/0.62/0.68 | SSA-COMET worse | PASS |
| lin | 60 | 0.899 [0.85,0.95] | 0.841 [0.78,0.90] | 0.07/0.13/0.00/0.02/0.23 | 0.13/0.15/0.00/0.05/0.40 | 0.38/0.53/0.66 | 0.51/0.62/0.68 | SSA-COMET worse | AUC ok, fails worse than AfriCOMET |
| lug | 60 | 0.807 [0.75,0.86] | 0.764 [0.70,0.83] | 0.15/0.18/0.00/0.00/0.32 | 0.22/0.23/0.00/0.00/0.40 | 0.39/0.54/0.66 | 0.54/0.63/0.70 | SSA-COMET worse | marginal (0.70-0.80) |
| mlg | 60 | 0.947 [0.92,0.97] | 0.931 [0.90,0.96] | 0.02/0.10/0.00/0.00/0.27 | 0.08/0.12/0.00/0.00/0.45 | 0.52/0.67/0.80 | 0.56/0.63/0.70 | SSA-COMET worse | AUC ok, fails worse than AfriCOMET |
| run | 60 | 0.949 [0.92,0.98] | 0.906 [0.87,0.94] | 0.03/0.08/0.00/0.00/0.25 | 0.07/0.08/0.00/0.02/0.40 | 0.53/0.64/0.73 | 0.59/0.67/0.73 | SSA-COMET worse | PASS |
| bem | 60 | 0.839 [0.77,0.90] | 0.811 [0.74,0.88] | 0.12/0.17/0.00/0.00/0.27 | 0.18/0.23/0.00/0.02/0.38 | 0.40/0.54/0.69 | 0.56/0.63/0.69 | SSA-COMET worse | AUC ok, fails worse than AfriCOMET |
| tso | 60 | 0.915 [0.87,0.95] | 0.849 [0.79,0.90] | 0.02/0.08/0.00/0.00/0.23 | 0.18/0.12/0.00/0.00/0.37 | 0.54/0.67/0.77 | 0.59/0.68/0.73 | SSA-COMET worse | AUC ok, fails worse than AfriCOMET |
| nso | 60 | 0.857 [0.81,0.90] | 0.813 [0.76,0.87] | 0.13/0.12/0.00/0.00/0.18 | 0.18/0.20/0.00/0.00/0.35 | 0.36/0.56/0.73 | 0.52/0.61/0.71 | SSA-COMET worse | AUC ok, fails worse than AfriCOMET |
| tum | 60 | 0.883 [0.83,0.92] | 0.853 [0.80,0.90] | 0.07/0.17/0.00/0.00/0.22 | 0.07/0.18/0.00/0.00/0.48 | 0.52/0.68/0.77 | 0.60/0.68/0.74 | SSA-COMET worse | PASS |
| umb | 60 | 0.768 [0.70,0.84] | 0.728 [0.66,0.80] | 0.20/0.22/0.08/0.07/0.42 | 0.25/0.23/0.00/0.05/0.55 | 0.27/0.44/0.58 | 0.49/0.59/0.65 | SSA-COMET worse | marginal (0.70-0.80) |
| mos | 60 | 0.839 [0.79,0.88] | 0.775 [0.70,0.84] | 0.10/0.17/0.00/0.00/0.43 | 0.28/0.20/0.00/0.02/0.58 | 0.42/0.54/0.64 | 0.55/0.63/0.69 | SSA-COMET worse | marginal (0.70-0.80) |
| kik | 60 | 0.872 [0.83,0.91] | 0.823 [0.77,0.87] | 0.07/0.08/0.03/0.00/0.30 | 0.15/0.10/0.00/0.02/0.42 | 0.39/0.50/0.61 | 0.59/0.67/0.73 | SSA-COMET worse | AUC ok, fails worse than AfriCOMET |
| luo | 60 | 0.860 [0.81,0.91] | 0.842 [0.78,0.90] | 0.12/0.17/0.00/0.00/0.28 | 0.13/0.18/0.00/0.00/0.48 | 0.45/0.59/0.72 | 0.58/0.67/0.73 | SSA-COMET worse | PASS |
| bam | 60 | 0.819 [0.77,0.87] | 0.753 [0.69,0.82] | 0.13/0.25/0.02/0.00/0.38 | 0.22/0.30/0.00/0.05/0.67 | 0.30/0.46/0.58 | 0.46/0.55/0.64 | SSA-COMET worse | marginal (0.70-0.80) |
| dyu | 60 | 0.679 [0.60,0.75] | 0.656 [0.57,0.74] | 0.35/0.28/0.05/0.07/0.60 | 0.37/0.33/0.00/0.08/0.72 | 0.27/0.37/0.55 | 0.48/0.55/0.63 | SSA-COMET worse | not usable (<0.70) |
| kab | 60 | 0.741 [0.68,0.80] | 0.719 [0.65,0.78] | 0.25/0.25/0.02/0.03/0.53 | 0.28/0.33/0.00/0.00/0.80 | 0.33/0.45/0.58 | 0.52/0.61/0.67 | SSA-COMET worse | marginal (0.70-0.80) |
| kam | 60 | 0.762 [0.71,0.82] | 0.745 [0.68,0.81] | 0.22/0.30/0.03/0.07/0.47 | 0.18/0.27/0.00/0.02/0.63 | 0.27/0.44/0.59 | 0.53/0.61/0.70 | equal (within noise) | marginal (0.70-0.80) |
| kbp | 60 | 0.786 [0.73,0.85] | 0.716 [0.65,0.79] | 0.22/0.22/0.00/0.00/0.22 | 0.25/0.30/0.00/0.00/0.33 | 0.41/0.50/0.60 | 0.52/0.59/0.65 | SSA-COMET worse | marginal (0.70-0.80) |
| kmb | 60 | 0.809 [0.75,0.86] | 0.803 [0.74,0.86] | 0.20/0.23/0.02/0.00/0.37 | 0.18/0.20/0.00/0.00/0.53 | 0.33/0.43/0.55 | 0.54/0.61/0.68 | SSA-COMET worse | PASS |
| nus | 60 | 0.748 [0.68,0.81] | 0.719 [0.64,0.79] | 0.20/0.23/0.05/0.02/0.25 | 0.27/0.30/0.00/0.02/0.35 | 0.29/0.40/0.57 | 0.46/0.54/0.63 | SSA-COMET worse | marginal (0.70-0.80) |
| sag | 60 | 0.689 [0.64,0.75] | 0.630 [0.57,0.70] | 0.22/0.23/0.07/0.00/0.40 | 0.37/0.30/0.02/0.00/0.45 | 0.25/0.36/0.49 | 0.44/0.53/0.61 | SSA-COMET worse | not usable (<0.70) |
| ssw | 60 | 0.901 [0.86,0.94] | 0.858 [0.80,0.91] | 0.07/0.12/0.00/0.02/0.32 | 0.13/0.18/0.00/0.02/0.32 | 0.46/0.62/0.74 | 0.54/0.64/0.70 | SSA-COMET worse | AUC ok, fails worse than AfriCOMET |
| lua | 60 | 0.867 [0.81,0.92] | 0.829 [0.77,0.88] | 0.15/0.25/0.05/0.02/0.28 | 0.13/0.25/0.00/0.00/0.35 | 0.31/0.40/0.53 | 0.49/0.56/0.64 | SSA-COMET worse | PASS |
| fuv | 60 | 0.737 [0.66,0.81] | 0.695 [0.60,0.78] | 0.25/0.25/0.10/0.08/0.45 | 0.33/0.37/0.02/0.08/0.60 | 0.26/0.39/0.54 | 0.50/0.58/0.68 | SSA-COMET worse | not usable (<0.70) |
| dik | 60 | 0.703 [0.63,0.77] | 0.677 [0.60,0.75] | 0.25/0.33/0.07/0.05/0.40 | 0.33/0.38/0.00/0.05/0.53 | 0.28/0.35/0.46 | 0.48/0.53/0.62 | SSA-COMET worse | not usable (<0.70) |
| cjk | 60 | 0.713 [0.65,0.78] | 0.690 [0.62,0.77] | 0.28/0.30/0.17/0.12/0.38 | 0.32/0.28/0.00/0.03/0.62 | 0.19/0.34/0.49 | 0.46/0.55/0.62 | equal (within noise) | not usable (<0.70) |
| knc | 60 | 0.751 [0.69,0.81] | 0.733 [0.66,0.80] | 0.15/0.25/0.08/0.03/0.55 | 0.22/0.27/0.00/0.08/0.65 | 0.29/0.35/0.44 | 0.49/0.52/0.57 | SSA-COMET worse | marginal (0.70-0.80) |
| taq | 60 | 0.743 [0.69,0.80] | 0.695 [0.63,0.76] | 0.23/0.30/0.20/0.10/0.45 | 0.25/0.30/0.00/0.05/0.62 | 0.22/0.30/0.40 | 0.44/0.50/0.56 | SSA-COMET worse | not usable (<0.70) |
| nqo | 60 | 0.541 [0.50,0.58] | 0.548 [0.49,0.61] | 0.42/0.37/0.80/0.05/0.80 | 0.42/0.40/0.45/0.03/0.85 | 0.12/0.15/0.20 | 0.28/0.31/0.35 | equal (within noise) | not usable (<0.70) |
| afr | 60 | 0.976 [0.96,0.99] | 0.965 [0.95,0.98] | 0.00/0.02/0.00/0.00/0.10 | 0.02/0.02/0.00/0.00/0.22 | 0.55/0.74/0.84 | 0.58/0.67/0.72 | SSA-COMET worse | PASS |
| ewe | 60 | 0.841 [0.78,0.90] | 0.800 [0.73,0.86] | 0.15/0.20/0.00/0.00/0.22 | 0.15/0.33/0.00/0.00/0.42 | 0.41/0.54/0.66 | 0.54/0.63/0.68 | SSA-COMET worse | AUC ok, fails worse than AfriCOMET |
| fon | 60 | 0.790 [0.73,0.85] | 0.735 [0.67,0.80] | 0.15/0.18/0.00/0.03/0.30 | 0.22/0.32/0.00/0.03/0.45 | 0.33/0.44/0.50 | 0.49/0.57/0.62 | SSA-COMET worse | marginal (0.70-0.80) |
| orm | 60 | 0.847 [0.79,0.89] | 0.765 [0.70,0.83] | 0.17/0.17/0.00/0.03/0.45 | 0.27/0.23/0.00/0.03/0.55 | 0.47/0.57/0.72 | 0.54/0.62/0.70 | SSA-COMET worse | marginal (0.70-0.80) |
| som | 60 | 0.917 [0.87,0.96] | 0.889 [0.84,0.94] | 0.07/0.03/0.00/0.02/0.32 | 0.10/0.05/0.00/0.00/0.42 | 0.41/0.61/0.75 | 0.52/0.61/0.69 | SSA-COMET worse | PASS |
| sot | 60 | 0.888 [0.84,0.93] | 0.852 [0.80,0.90] | 0.08/0.12/0.00/0.00/0.23 | 0.15/0.20/0.00/0.00/0.40 | 0.42/0.59/0.72 | 0.55/0.64/0.72 | SSA-COMET worse | AUC ok, fails worse than AfriCOMET |
| wol | 60 | 0.856 [0.81,0.90] | 0.817 [0.75,0.88] | 0.07/0.18/0.00/0.02/0.43 | 0.13/0.23/0.00/0.02/0.57 | 0.42/0.51/0.64 | 0.62/0.68/0.72 | SSA-COMET worse | AUC ok, fails worse than AfriCOMET |



## 5. Pidgin (pcm), Twi, Yoruba, Nyanja

MAFAND en-pcm dev, 150 contiguous sentences, 1,493 pairs per model (all 9 classes):

| class (failure = corrupted >= human of same sentence) | AfriCOMET | SSA-COMET |
|---|---|---|
| mismatched reference of another sentence | 0.27 | 0.40 |
| neighbouring sentence | 0.31 | 0.35 |
| English source copied | 0.02 | 0.01 |
| English copy with 20% words dropped | 0.11 | 0.08 |
| half of the words dropped | 0.05 | 0.08 |
| words shuffled | 0.02 | 0.01 |
| unrelated sentence appended | 0.55 | 0.61 |
| names / numbers swapped | 0.08 | 0.10 |
| 25% words replaced | 0.05 | 0.07 |
| mean of the 9 classes | 0.163 | 0.189 |
| AUC human vs mismatched [CI90] | 0.705 [0.66,0.75] | 0.639 [0.59,0.69] |
| same-source accuracy mean(mismatch, half-omitted) -> reliability_same_source | 0.84 -> 0.84 | 0.76 -> 0.65 |
| human P10/P50/P90 | 0.23/0.37/0.52 | 0.35/0.44/0.54 |
| share of human refs under the 0.46 floor | 0.77 | 0.68 |
| anchors (t_fail, t_mid, t_great) as `pair_anchors` would set them | 0.46, 0.51, 0.56 | 0.49, 0.54, 0.59 |

* SSA-COMET does not fix Pidgin: wrong-but-fluent sentences are ranked above the reference more often (0.40 vs 0.27), and appended text is not detected by either model (0.55 / 0.61, near chance; this is why D-24's length window is needed with either model). The mean over classes for AfriCOMET (0.163) reproduces the earlier 13k-row result (0.153), which supports the small sample being representative.
* The corrections already in the RL reward (D-24 length window / extra-output penalty, D-25 lexical-fidelity penalty, copy guard) remain necessary and are model-independent; nothing here replaces them.
* Twi: AUC 0.678 -> 0.629 (SSA worse; mismatched failure 0.31 -> 0.39). Yoruba 0.802 -> 0.789 (CI of the difference [-0.04,+0.01], equal; failures 0.10 -> 0.13 mismatched, 0.22 -> 0.37 appended). Nyanja 0.835 -> 0.797 (worse). Zulu 0.798 -> 0.742 (worse; both weak: AfriCOMET mismatched failure 0.25).
* The earlier "reliability" figures (0.43 Pidgin, 0.50 Twi, 0.58 Yoruba, 0.66 Nyanja in `shaping_mafand_v2.json`) are `reliability_from_auc` values; on this block the AUCs are 0.705/0.678/0.802/0.835. Different samples (500 shuffled vs a contiguous block), so compare only the model ordering.

**Reverse orientation (African text as `src`, English original as `mt`; the X->en direction of the SSA-COMET card example).** I checked whether SSA-COMET is simply trained for X->en. AUC human vs mismatched English, SSA-COMET / AfriCOMET: pcm 0.664 / 0.758, twi 0.769 / 0.743, yor 0.814 / 0.796, hau 0.918 / 0.920, ibo 0.867 / 0.855, zul 0.671 / 0.675 (nya SSA 0.883, AfriCOMET not run). So in that orientation SSA-COMET is level with or marginally above AfriCOMET (+0.01..+0.03, within noise) on twi/yor/ibo, but still worse on Pidgin. This is not the RL use (en->xx), so it does not change the verdict; it only shows SSA-COMET is not broken and that its training signal is probably dominated by X->en data.

## 6. New languages: which ones can SSA-COMET score, and does it beat AfriCOMET?

All new-language rows are FLORES+ dev, 60 sentences, 420-480 pairs per model. Reference sets were available (FLORES+ dev) for 37 of the owner's list; **not available** (no FLORES+/MAFAND/bench reference, so untested): loz, ven, iso, tiv, kon, wal, kua, lun, kqn (and guw, efi, gaa, lue, toi, tll, zne, lub, kwy, bci, rnd, nyk, tzm of the card list). No references were invented.

* **SSA-COMET passes the criterion** (AUC >= 0.80 vs mismatched, and failure rate on mismatched/neighbour/copy/half-omitted not worse than AfriCOMET by more than 0.05): **tir, run, tum, luo, kmb, lua, afr, som** (8; lower 90% CI bound of the AUC 0.79, 0.87, 0.80, 0.78, 0.74, 0.77, 0.95, 0.84; only run, afr, som, tum clear 0.80 with their lower bound). AUC >= 0.80 but failures worse than AfriCOMET by > 0.05 on at least one of those classes: amh lin mlg bem tso nso kik ssw ewe sot wol (11). Marginal (0.70-0.80): lug umb mos bam kab kam kbp nus knc fon orm (11). Not usable (< 0.70): dyu sag fuv dik cjk taq nqo (7).
* **But none of these is a case where SSA-COMET beats AfriCOMET.** On the same data AfriCOMET-QE-1.1 has AUC >= 0.80 in 23 of the 37 (amh tir lin lug mlg run bem tso nso tum mos kik luo bam kmb ssw lua afr ewe orm som sot wol) and is above SSA-COMET in all of them. Languages below 0.80 for AfriCOMET: umb 0.77, dyu 0.68, kab 0.74, kam 0.76, kbp 0.79, nus 0.75, sag 0.69, fuv 0.74, dik 0.70, cjk 0.71, knc 0.75, taq 0.74, nqo 0.54, fon 0.79. For these, **neither model is a good reward** (dyu, sag, fuv, dik, cjk, taq, nqo are below the repo's `min_auc` 0.7 for SSA-COMET and mostly for AfriCOMET; they would be `usable: false`).
* N'Ko (nqo): both models fail (AUC 0.54/0.55, copy failure 0.80/0.45); the afro-xlmr tokenizer maps 48% of the N'Ko reference tokens to `<unk>`. Not scoreable by either; exclude. Small unk rates elsewhere: kab 0.7%, nus 0.45%, bam 0.17%, kbp 0.16% (none problematic).

## 7. Calibration and exact changes needed

Swapping the model (not recommended) needs exactly:

* `gemma_mt/rl/comet.py`: `QE_MODEL_ID = "McGill-NLP/ssa-comet-qe"`. `CometScorer` needs no code change: `download_model("McGill-NLP/ssa-comet-qe")` + `load_from_checkpoint(path)` works with unbabel-comet 2.2.7 (verified, CPU, offline from the HF cache); same class `ReferencelessRegression`, same `prepare_sample`, same `encoder.max_positions = 512`, so `COMET_MAX_SEGMENT_TOKENS = 510` and `token_lengths` stay valid. `ENCODER_ID` is the same (`Davlan/afro-xlmr-large-76L`). `REF_MODEL_ID` (reference-based AfriCOMET-STL for evaluation) has no SSA counterpart (SSA-COMET is only published as QE). fp16/`.half()` on GPU was **not** tested (CPU run).
* Where the id is hard-coded: `configs/rl.yaml` `scorer.model_id`; `train_rl.py` lines 95 and 173 (defaults); `comet_worker.py --model` default and docstring; `calibrate_shaping.py --qe-model` default; `tests/test_integration_gated.py` (uses `QE_MODEL_ID`). Environment: the encoder forward in comet 2.2.7 needs `transformers < 5` (it unpacks 3 outputs; with transformers 5 it fails with "not enough values to unpack"), exactly as for AfriCOMET, so on the box it must run from `pylibs_comet`/the `WorkerScorer` venv.
* **Recalibrate everything.** `reward_shaping.FAIL_FLOOR = 0.46` and `DEFAULT_ANCHORS = (0.46, 0.60, 0.75)` are AfriCOMET-scale. SSA-COMET human medians are 0.50-0.70 (0.44 pcm, 0.31 nqo) with a 0.15 P10-P90 spread (see the P10/P50/P90 columns), so t_fail/t_mid/t_great move per language (anchors table in section 8 gives the values that `pair_anchors` would produce). Rerun `calibrate_shaping.py --qe-model McGill-NLP/ssa-comet-qe` on human references; `reliability_from: same_source` will drop for most languages (SSA failure rates are higher), i.e. a smaller advantage weight and a weaker training signal.

Needed whether or not the model changes, **for AfriCOMET on the new languages**:

* The 0.46 floor is wrong for them: AfriCOMET human medians are 0.15-0.50 in 17 languages (more than half the human references are below 0.46 in pcm umb bam dyu kab kam kmb nus sag lua fuv dik cjk knc taq nqo fon). With the floor, `t_mid = max(P50, t_fail + 0.05)` pins t_fail/t_mid/t_great to 0.46/0.51/0.56 and a human-quality translation earns <= 0. Use per-language `t_fail = P90(mismatched)` and `t_mid/t_great = P50/P90(human)` with the floor lowered (e.g. `floor = 0.0`) or defined per language from the mismatched distribution. The "anchors AfriC" column in section 8 shows what is produced today (0.46,0.51,0.56 means floor-limited).
* Calibration data needed per new language: human en->xx reference pairs not present in SFT/RL data, ideally >= 300-500 (calibration default `--max-per-language 500`; FLORES+ dev has 997 per language, MAFAND dev 460-1,800) plus the same set's mismatched/neighbour/half/appended/copy corruptions (already built in `calibrate_shaping`); the human length-ratio window (`len_ratio`) for D-24 and the lexical thresholds for D-25 come from the same file. Languages with only 60 sentences here must be recalibrated with the full 997 before use: AUC CI is +-0.05-0.09 at n = 60. Also needed: RL prompts (English sources) for the new languages, GlotLID labels, and the `languages.py` `africomet_status`/`rl_eligible` flags (a bookkeeping change; the model takes no language code).
* Languages to keep out of RL even after calibration (AfriCOMET AUC < 0.70 or tokeniser coverage): nqo, dyu, sag; treat fuv dik cjk taq umb kab kam kbp nus knc fon (0.70-0.80, with half or more of the human references under 0.46 and copy failures of 0.05-0.20 for some) as low-weight (`reliability` from AUC / same-source accuracy), and re-test them with 997 sentences first.

## 8. More detail (per language)

### Failure rate on the remaining classes, NFC effect, anchors and floor share

(fail rates are AfriCOMET/SSA-COMET; `nan` = class not run for that language; "anchors" are what `pair_anchors` gives with the 0.46 floor, using mismatched P90 and human P50/P90)


| lang | shuffle | swapN | replace | copy_noisy | human_nfd mean|delta| AfriC/SSA | anchors SSA (t_fail,t_mid,t_great) | anchors AfriC | share human<0.46 AfriC/SSA |
|---|---|---|---|---|---|---|---|---|
| pcm | 0.02/0.01 | 0.08/0.10 | 0.05/0.07 | 0.11/0.08 | - | 0.49,0.54,0.59 | 0.46,0.51,0.56 | 0.77/0.68 |
| twi | 0.00/0.01 | 0.12/0.19 | 0.03/0.05 | 0.03/0.00 | 0.000/0.000 (n=4) | 0.66,0.71,0.76 | 0.56,0.61,0.66 | 0.30/0.02 |
| yor | 0.00/0.00 | 0.08/0.08 | 0.01/0.04 | 0.01/0.00 | 0.018/0.012 (n=100) | 0.69,0.74,0.79 | 0.62,0.67,0.73 | 0.08/0.00 |
| nya | 0.00/0.00 | 0.04/0.03 | 0.00/0.01 | 0.02/0.01 | - | 0.64,0.69,0.74 | 0.58,0.63,0.70 | 0.07/0.01 |
| hau | 0.00/0.01 | 0.00/0.03 | 0.00/0.00 | 0.03/0.01 | - | 0.67,0.72,0.77 | 0.69,0.74,0.82 | 0.00/0.00 |
| ibo | 0.00/0.01 | 0.06/0.10 | 0.05/0.07 | 0.01/0.00 | 0.000/0.000 (n=76) | 0.67,0.72,0.77 | 0.61,0.66,0.79 | 0.05/0.00 |
| kin | 0.00/0.00 | 0.00/0.03 | 0.00/0.00 | 0.00/0.00 | 0.000/0.000 (n=4) | 0.68,0.73,0.78 | 0.64,0.69,0.77 | 0.01/0.00 |
| sna | 0.00/0.00 | 0.03/0.03 | 0.00/0.00 | 0.00/0.00 | - | 0.68,0.73,0.78 | 0.62,0.67,0.75 | 0.03/0.00 |
| swh | 0.01/0.01 | 0.03/0.03 | 0.00/0.00 | 0.01/0.01 | - | 0.64,0.70,0.75 | 0.65,0.75,0.85 | 0.00/0.00 |
| tsn | 0.00/0.00 | 0.03/0.03 | 0.01/0.01 | 0.00/0.00 | 0.000/0.000 (n=11) | 0.66,0.71,0.76 | 0.64,0.69,0.80 | 0.06/0.00 |
| xho | 0.00/0.00 | 0.01/0.03 | 0.00/0.00 | 0.00/0.00 | - | 0.62,0.67,0.72 | 0.61,0.69,0.80 | 0.01/0.00 |
| zul | 0.00/0.00 | 0.04/0.03 | 0.01/0.03 | 0.03/0.01 | - | 0.60,0.65,0.70 | 0.58,0.63,0.73 | 0.20/0.10 |
| amh | nan/nan | nan/nan | 0.00/0.02 | nan/nan | - | 0.58,0.65,0.70 | 0.57,0.72,0.80 | 0.02/0.00 |
| tir | nan/nan | nan/nan | 0.03/0.02 | nan/nan | - | 0.59,0.64,0.69 | 0.59,0.66,0.75 | 0.05/0.03 |
| lin | nan/nan | nan/nan | 0.00/0.02 | nan/nan | 0.000/0.000 (n=3) | 0.58,0.63,0.68 | 0.46,0.53,0.66 | 0.23/0.07 |
| lug | nan/nan | nan/nan | 0.02/0.03 | nan/nan | - | 0.63,0.68,0.73 | 0.54,0.59,0.66 | 0.32/0.03 |
| mlg | nan/nan | nan/nan | 0.00/0.00 | nan/nan | 0.000/0.000 (n=8) | 0.59,0.64,0.70 | 0.55,0.67,0.80 | 0.03/0.00 |
| run | nan/nan | nan/nan | 0.00/0.00 | nan/nan | - | 0.62,0.67,0.73 | 0.54,0.64,0.73 | 0.00/0.00 |
| bem | nan/nan | nan/nan | 0.02/0.02 | nan/nan | - | 0.62,0.67,0.72 | 0.51,0.56,0.69 | 0.23/0.00 |
| tso | nan/nan | nan/nan | 0.00/0.00 | nan/nan | - | 0.66,0.71,0.76 | 0.58,0.67,0.77 | 0.05/0.00 |
| nso | nan/nan | nan/nan | 0.00/0.00 | nan/nan | 0.000/0.000 (n=52) | 0.60,0.65,0.71 | 0.50,0.56,0.73 | 0.22/0.03 |
| tum | nan/nan | nan/nan | 0.00/0.00 | nan/nan | 0.000/0.000 (n=41) | 0.66,0.71,0.76 | 0.62,0.68,0.77 | 0.05/0.00 |
| umb | nan/nan | nan/nan | 0.05/0.08 | nan/nan | 0.000/0.000 (n=31) | 0.59,0.64,0.69 | 0.46,0.51,0.58 | 0.60/0.08 |
| mos | nan/nan | nan/nan | 0.03/0.08 | nan/nan | 0.000/0.000 (n=57) | 0.62,0.67,0.72 | 0.53,0.58,0.64 | 0.17/0.00 |
| kik | nan/nan | nan/nan | 0.02/0.00 | nan/nan | 0.000/0.000 (n=60) | 0.65,0.70,0.75 | 0.46,0.51,0.61 | 0.32/0.00 |
| luo | nan/nan | nan/nan | 0.02/0.02 | nan/nan | - | 0.65,0.70,0.75 | 0.58,0.63,0.72 | 0.13/0.00 |
| bam | nan/nan | nan/nan | 0.05/0.10 | nan/nan | - | 0.56,0.61,0.66 | 0.46,0.51,0.58 | 0.52/0.12 |
| dyu | nan/nan | nan/nan | 0.07/0.18 | nan/nan | 0.000/0.000 (n=46) | 0.59,0.64,0.69 | 0.47,0.52,0.57 | 0.75/0.08 |
| kab | nan/nan | nan/nan | 0.03/0.05 | nan/nan | 0.000/0.000 (n=54) | 0.62,0.67,0.72 | 0.46,0.51,0.58 | 0.55/0.00 |
| kam | nan/nan | nan/nan | 0.05/0.10 | nan/nan | 0.000/0.000 (n=26) | 0.63,0.68,0.73 | 0.46,0.51,0.59 | 0.58/0.00 |
| kbp | nan/nan | nan/nan | 0.00/0.02 | nan/nan | 0.000/0.000 (n=18) | 0.61,0.66,0.71 | 0.52,0.57,0.62 | 0.28/0.00 |
| kmb | nan/nan | nan/nan | 0.02/0.08 | nan/nan | 0.000/0.000 (n=10) | 0.61,0.66,0.71 | 0.46,0.51,0.56 | 0.68/0.02 |
| nus | nan/nan | nan/nan | 0.02/0.03 | nan/nan | 0.000/0.000 (n=60) | 0.56,0.61,0.66 | 0.46,0.51,0.57 | 0.68/0.08 |
| sag | nan/nan | nan/nan | 0.02/0.07 | nan/nan | 0.000/0.000 (n=60) | 0.58,0.63,0.68 | 0.46,0.51,0.56 | 0.80/0.20 |
| ssw | nan/nan | nan/nan | 0.00/0.03 | nan/nan | - | 0.61,0.66,0.71 | 0.56,0.62,0.74 | 0.10/0.02 |
| lua | nan/nan | nan/nan | 0.00/0.00 | nan/nan | - | 0.57,0.62,0.67 | 0.46,0.51,0.56 | 0.72/0.02 |
| fuv | nan/nan | nan/nan | 0.12/0.10 | nan/nan | - | 0.60,0.65,0.70 | 0.46,0.51,0.56 | 0.72/0.08 |
| dik | nan/nan | nan/nan | 0.05/0.05 | nan/nan | 0.000/0.000 (n=59) | 0.58,0.63,0.68 | 0.46,0.51,0.56 | 0.88/0.07 |
| cjk | nan/nan | nan/nan | 0.07/0.07 | nan/nan | 0.000/0.000 (n=10) | 0.57,0.62,0.67 | 0.46,0.51,0.56 | 0.87/0.12 |
| knc | nan/nan | nan/nan | 0.03/0.12 | nan/nan | - | 0.54,0.59,0.64 | 0.46,0.51,0.56 | 0.93/0.03 |
| taq | nan/nan | nan/nan | 0.10/0.15 | nan/nan | 0.000/0.000 (n=52) | 0.54,0.59,0.64 | 0.46,0.51,0.56 | 0.97/0.20 |
| nqo | nan/nan | nan/nan | 0.53/0.53 | nan/nan | - | 0.46,0.51,0.56 | 0.46,0.51,0.56 | 1.00/1.00 |
| afr | nan/nan | nan/nan | 0.00/0.00 | nan/nan | 0.000/0.000 (n=21) | 0.58,0.67,0.72 | 0.58,0.74,0.84 | 0.00/0.00 |
| ewe | nan/nan | nan/nan | 0.00/0.00 | nan/nan | 0.000/0.000 (n=31) | 0.62,0.67,0.72 | 0.52,0.57,0.66 | 0.28/0.02 |
| fon | nan/nan | nan/nan | 0.03/0.08 | nan/nan | 0.000/0.000 (n=18) | 0.57,0.62,0.67 | 0.46,0.51,0.56 | 0.63/0.05 |
| orm | nan/nan | nan/nan | 0.03/0.08 | nan/nan | - | 0.63,0.68,0.73 | 0.55,0.60,0.72 | 0.10/0.00 |
| som | nan/nan | nan/nan | 0.00/0.00 | nan/nan | - | 0.58,0.63,0.69 | 0.46,0.61,0.75 | 0.15/0.02 |
| sot | nan/nan | nan/nan | 0.00/0.00 | nan/nan | - | 0.62,0.67,0.72 | 0.55,0.60,0.72 | 0.13/0.03 |
| wol | nan/nan | nan/nan | 0.05/0.07 | nan/nan | 0.000/0.000 (n=57) | 0.68,0.73,0.78 | 0.49,0.54,0.64 | 0.25/0.00 |


### Speed and tokens per language (texts/s are unique texts embedded per second, 2 threads, both processes running at the same time)

| lang | AfriC texts/s | SSA texts/s | pairs/s AfriC | pairs/s SSA | mean src/ref tokens | max ref tokens | texts >510 | ref unk rate |
|---|---|---|---|---|---|---|---|---|
| pcm | 2.45 | 2.42 | 3.08 | 3.05 | 37.7/39.5 | 157 | 0 | 0.0 |
| twi | 1.58 | 1.56 | 1.99 | 1.98 | 42.6/67.8 | 174 | 0 | 0.0003 |
| yor | 1.12 | 1.11 | 1.38 | 1.36 | 35.0/99.3 | 265 | 0 | 0.00031 |
| nya | 1.37 | 1.67 | 1.72 | 2.09 | 51.0/83.0 | 231 | 0 | 0.0 |
| hau | 2.51 | 2.38 | 3.17 | 3.0 | 39.7/56.7 | 183 | 0 | 0.0 |
| ibo | 3.35 | 5.02 | 4.15 | 6.23 | 21.6/41.7 | 135 | 0 | 0.0 |
| kin | 2.16 | 1.31 | 2.7 | 1.64 | 62.4/100.1 | 515 | 11 | 0.0 |
| sna | 3.69 | 1.95 | 4.67 | 2.47 | 31.3/63.0 | 166 | 0 | 0.0 |
| swh | 5.15 | 2.99 | 6.47 | 3.75 | 34.8/40.4 | 92 | 0 | 0.0 |
| tsn | 4.36 | 3.28 | 5.44 | 4.09 | 30.4/50.0 | 150 | 0 | 0.0 |
| xho | 3.15 | 2.36 | 3.96 | 2.97 | 36.0/60.8 | 120 | 0 | 0.0 |
| zul | 3.19 | 2.61 | 4.36 | 3.56 | 25.1/54.1 | 142 | 0 | 0.0 |
| amh | 4.14 | 3.51 | 5.79 | 4.91 | 33.2/44.2 | 91 | 0 | 0.0 |
| tir | 2.47 | 1.67 | 3.45 | 2.33 | 33.2/60.0 | 112 | 0 | 0.0 |
| lin | 2.81 | 1.89 | 3.92 | 2.63 | 33.2/48.4 | 113 | 0 | 0.0 |
| lug | 2.85 | 2.07 | 3.98 | 2.9 | 33.2/50.6 | 108 | 0 | 0.0 |
| mlg | 2.18 | 2.19 | 3.02 | 3.04 | 33.2/50.9 | 113 | 0 | 0.0 |
| run | 1.68 | 3.17 | 2.34 | 4.42 | 33.2/54.1 | 135 | 0 | 0.0 |
| bem | 1.73 | 2.04 | 2.41 | 2.85 | 33.2/54.8 | 136 | 0 | 0.0 |
| tso | 1.98 | 1.89 | 2.77 | 2.65 | 33.2/59.9 | 133 | 0 | 0.0 |
| nso | 1.9 | 1.76 | 2.55 | 2.35 | 33.2/55.5 | 124 | 0 | 0.0 |
| tum | 1.89 | 1.52 | 2.56 | 2.05 | 33.2/61.0 | 147 | 0 | 0.0 |
| umb | 3.45 | 1.93 | 4.69 | 2.63 | 33.2/49.4 | 92 | 0 | 0.0 |
| mos | 2.31 | 2.09 | 3.08 | 2.79 | 33.2/57.1 | 128 | 0 | 0.0 |
| kik | 1.32 | 1.41 | 1.76 | 1.87 | 33.2/74.9 | 136 | 0 | 0.0 |
| luo | 1.86 | 1.83 | 2.59 | 2.55 | 33.2/49.8 | 88 | 0 | 0.0 |
| bam | 1.69 | 1.6 | 2.36 | 2.24 | 33.2/60.9 | 126 | 0 | 0.0017 |
| dyu | 2.15 | 2.08 | 2.89 | 2.79 | 33.2/50.8 | 134 | 0 | 0.0 |
| kab | 1.64 | 1.62 | 2.19 | 2.16 | 33.2/59.1 | 140 | 0 | 0.00671 |
| kam | 1.83 | 1.86 | 2.5 | 2.54 | 33.2/51.8 | 107 | 0 | 0.0 |
| kbp | 1.04 | 1.0 | 1.43 | 1.38 | 33.2/98.4 | 208 | 0 | 0.00156 |
| kmb | 1.82 | 1.78 | 2.52 | 2.46 | 33.2/53.2 | 109 | 0 | 0.0 |
| nus | 1.09 | 1.1 | 1.45 | 1.47 | 33.2/90.7 | 181 | 0 | 0.00451 |
| sag | 1.54 | 1.51 | 2.05 | 2.01 | 33.2/62.1 | 142 | 0 | 0.0 |
| ssw | 1.71 | 1.72 | 2.4 | 2.41 | 33.2/53.6 | 146 | 0 | 0.0 |
| lua | 1.96 | 1.94 | 2.74 | 2.71 | 33.2/47.4 | 106 | 0 | 0.0 |
| fuv | 1.98 | 1.95 | 2.77 | 2.72 | 33.2/46.5 | 96 | 0 | 0.00037 |
| dik | 1.72 | 1.8 | 2.29 | 2.4 | 33.2/54.1 | 100 | 0 | 0.0 |
| cjk | 1.84 | 1.95 | 2.55 | 2.7 | 33.2/49.2 | 137 | 0 | 0.0 |
| knc | 1.43 | 1.53 | 2.0 | 2.14 | 33.2/66.3 | 170 | 0 | 0.0 |
| taq | 1.73 | 1.78 | 2.32 | 2.39 | 33.2/54.1 | 113 | 0 | 0.0 |
| nqo | 1.73 | 1.73 | 2.41 | 2.43 | 33.2/58.6 | 137 | 0 | 0.48322 |
| afr | 2.66 | 2.32 | 3.65 | 3.18 | 33.2/39.6 | 94 | 0 | 0.0 |
| ewe | 1.82 | 1.76 | 2.47 | 2.39 | 33.2/55.5 | 136 | 0 | 0.00093 |
| fon | 1.11 | 1.16 | 1.53 | 1.6 | 33.2/92.5 | 251 | 0 | 0.00203 |
| orm | 1.7 | 2.86 | 2.37 | 4.0 | 33.2/58.0 | 130 | 0 | 0.0 |
| som | 2.1 | 4.21 | 2.93 | 5.89 | 33.2/44.0 | 91 | 0 | 0.0 |
| sot | 1.64 | 2.99 | 2.29 | 4.18 | 33.2/56.2 | 129 | 0 | 0.0 |
| wol | 1.98 | 2.1 | 2.65 | 2.81 | 33.2/50.6 | 108 | 0 | 0.0 |

## 9. Encoding, licence, speed, limits

* **Licence.** SSA-COMET-QE: Apache-2.0 (Hub card metadata and README; checkpoint 2.3 GB `checkpoints/model.ckpt`, no gating, revision `e999f0cc`, last modified 2025-05-23). AfriCOMET-QE-1.1: Apache-2.0; encoder Davlan/afro-xlmr-large-76L: MIT; unbabel-comet: Apache-2.0. No licence obstacle to either. The card gives no paper (said "coming soon") and no numbers, so there is no external evaluation to check these results against, and its training data are not documented (possible overlap with MAFAND/FLORES-like data cannot be excluded for either model).
* **Loading call that works (comet 2.2.7, offline from cache):** `from comet import download_model, load_from_checkpoint; m = load_from_checkpoint(download_model("McGill-NLP/ssa-comet-qe")); m.predict([{"src":..,"mt":..}], batch_size=16, gpus=0)`; or via the repo's `CometScorer("McGill-NLP/ssa-comet-qe")`. Run from the `pylibs_comet` path (transformers 4.55.4) as for AfriCOMET.
* **NFC.** The encoder tokenizer does not compose Yoruba (dot-below + tone mark) sequences: NFC and NFD text give different token ids for 46 of 48 Yoruba references (for every other language that has NFC text whose NFD differs - ibo, nso, tum, kik, mos, dyu, kab, kam, kbp, kmb, nus, ewe, fon, wol, ... - the NFD form scores identically, differences ~1e-8, so those are normalised by the tokenizer). The score effect for Yoruba: mean |NFC - NFD| = 0.012 (SSA) / 0.018 (AfriCOMET), max 0.08 / 0.10 on 100 references (P10..P90 spread of human scores is 0.1-0.2), same for both models. Both models are therefore fine on NFC text; keep the RL outputs NFC-normalised before scoring (the reward already sanitises text). The raw reference text was already NFC except 41 of 60 fon FLORES references and 2 of 60 mos references (so the fon scores above are on non-NFC text; scores are identical for NFC/NFD forms). Unk rate on references is 0 except nqo (48%), kab (0.7%), nus (0.45%), bam (0.17%), kbp (0.16%), fon (0.2%), ewe (0.09%), others ~0. No encoding errors.
* **Length.** Same as AfriCOMET: `max_positions = 512`, source and translation tokenised separately and truncated at 510 tokens each (comet `prepare_sample`; `texts > 510` column above). In this data only kin has any (11 unique texts: 1 human reference of 515 tokens plus appended-text corruptions). Mean reference length 40-100 tokens (Yoruba ~100: the tokeniser is token-hungry for tonal Yoruba).
* **Speed (CPU, fp32, 2 threads per model process, 4 logical CPUs = 3 physical cores, shared with other jobs).** Identical architecture, hence identical cost. `model.predict()` with batch 16: 0.85 pairs/s for SSA-COMET on 48 Yoruba pairs (~100-token references; 0.48 pairs/s for AfriCOMET measured while three of my processes ran at once). With unique-text embedding caching (source, references and the synthetic variants encoded once; exactly the same scores): median 1.9 unique texts/s per process, 2.6 pairs/s (range 1.0-5.2 texts/s; slower for Yoruba/Fon/Kabiyè/Nuer where references are 90-100 tokens, faster for short sentences). 27,891 pairs per model took ~3.0 h of processing each; 4.2 h wall clock end to end. GPU speed was not measured (GPU forbidden).

## 10. What I could not do / caveats

* Sample sizes: 60 sentences per FLORES+ language (CI90 of an AUC about +-0.05-0.09, of a failure rate +-0.06-0.10), 80-150 per MAFAND language; one contiguous block per language, news (MAFAND) or Wikimedia (FLORES+) text, not the RL prompt distribution. A per-language verdict of "worse" with CI touching 0 (yor, swh, zul hard-class) is a statement about direction; the 49-language consistency (48 negative AUC differences, 49/49 higher hard-class failure rates) is the stronger evidence.
* Corruptions are synthetic (the same ones as D-24). Real policy outputs (paraphrases, partial errors, fluent hallucinations) may rank differently.
* **No Spearman(QE, chrF++) on SFT-model outputs**: no model outputs on the box (no translation hypotheses for the bench sets exist; generating them on CPU was out of budget and the GPU is off limits). This is the test that matters most for a reward and remains open (use `benchmark_eval` samples from the RL run).
* The score scale is the models' own; thresholds were only recomputed as `pair_anchors` on these small blocks, not as final calibration values.
* fp16 on GPU (the RL setting) was not tested; scores here are fp32 on CPU.
* The reverse-orientation check is incomplete (AfriCOMET `nya` missing); a model ensemble (mean of both scores) was not tested.
* No reference sets (not invented) for loz ven iso tiv kon wal kua lun kqn and the other card languages not in FLORES+/MAFAND.
