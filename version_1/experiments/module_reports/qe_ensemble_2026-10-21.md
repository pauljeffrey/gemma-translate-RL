# QE ensemble of the reward: AfriCOMET-QE-STL-1.1 + SSA-COMET-MTL reference-free (D-55, agent B1, 2026-10-21)

Owner decision: the QE part of the reward becomes an ensemble of `masakhane/africomet-qe-stl-1.1` (current) and `McGill-NLP/ssa-comet-mtl` run reference-free (src + mt only). Per scored chunk
(a sentence pair of the chunk plan): `r1 = shape(AfriCOMET, v7 anchors)`, `r2 = shape(SSA-COMET-MTL, own anchors)`, same curve and reward scale ([-2, +2.25]);
`r = (r1 + r2) / 2` if `|r1 - r2| <= delta`, else `min(r1, r2)`. **delta = 1.0 reward unit.** OFF by default (`reward.qe_ensemble: false`): with the flag off the reward is bit-for-bit the old one (tested).
Evidence for the choice of metric: `reports/qe_metric_choice_2026-10-21.md`.

## 1. What was built

New files: `gemma_mt/rl/qe_ensemble.py` (the rule, the shaper pair, per-language plan / weights, `QEPairScorer`, `reward/qe_ens/*` statistics), `gemma_mt/rl/joint_budget.py` (src + mt <= 508 content tokens for the
joint-encoder model), `scripts/calibrate_qe_ensemble.py` (score / compose / delta), `scripts/qe_ensemble_gpu_check.py` (old vs new reward on the same samples, memory and time),
`calibration/shaping_flores_dev_v8.json`, `tests/test_qe_ensemble.py` (31 tests), `patches/qe_ensemble_D55.patch`.
Edits of existing files (the whole patch is `patches/qe_ensemble_D55.patch`, clean against the tree WITHOUT the D-52 / D-56 hunks of other agents):
* `rl/comet.py`: `CometScorer` loads `UnifiedMetric` checkpoints (`self.unified`), scoring reference-free through `prepare_sample` + `predict_step` (the call of `qe_choice_score.py::unified_call`); a pair over the
  508-token joint budget is cut recursively (sentence boundary nearest the middle of BOTH texts, else word boundary) into pieces that fit and the piece scores are averaged by source characters, so a long
  translation never silently truncates the end of the source (SSA-COMET-MTL encodes ONE sequence `<s> mt </s></s> src </s>` of 512 positions; AfriCOMET encodes src and mt separately, 510 each). A pair that fits is
  untouched; the AfriCOMET path (single model) is unchanged. `WorkerScorer(extra_model_ids=[...])` + `view(k)`: several checkpoints in ONE child process (transformers 4.x), picked per request by `"model": k`
  (absent = 0: the protocol of the single-model worker is unchanged). New optional `max_batch_chars` (batch size x longest src + mt characters) bounds the memory of batches of long chunks.
* `rl/comet_worker.py`: repeated `--model`, `--max-batch-chars`; with several models it sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and returns the cache after every request.
* `rl/reward.py` (QE scoring / shaping step only): 6 config fields, `RewardOutput.qe2`, one construction block, the second scoring call next to the first, `r = self._shape_qe(c1, c2, lang)` instead of `shaper(c1, lang)`,
  `self._weight(lang)` instead of `shaper.reliability(lang)`, the `reward/qe_ens/*` statistics. Everything after the QE value (consensus blend, length / lexical / diacritic / science / constraint penalties,
  failed chunks, aggregation) is untouched.
* `rl/train_rl.py`: `build_scorer` returns a `QEPairScorer` when the `scorer` block has `ensemble_model_id`; the smoke test uses it.

How the pieces combine
* **Content modes.** qe and blend languages: the ensemble value replaces the QE value `r`; the blend `(1 - beta) * r + beta * consensus` is unchanged. Consensus-mode languages (23, incl. kea and vmw): no QE call at all, as before.
* **Languages.** both models usable -> the rule; only SSA usable -> r2; only AfriCOMET usable (or no `qe2` block) -> r1 as before. No language is unusable for SSA (see section 3), so the fallback is never taken today.
  Languages without a human dev reference (iso kon kqn kua loz lun swc tiv ven wal) take a pooled-prior `qe2` (median of the 44 calibrated languages: t_fail/t_mid/t_great = 0.326 / 0.567 / 0.686; swc borrows swh) and keep their
  owner-set weights 0.3. `qe_ensemble_prior_langs: false` leaves them on AfriCOMET alone.
* **Advantage weights** (`reward.qe_ensemble_reliability`): `ensemble` (default) = baked `ens.reliability*` = max(v7 weight, the ensemble's own measured reliability, same formulas as v7: AUC linear 0.65 -> 0.90,
  same-source accuracy of mismatch and half-omitted); owner-set weights of v7 (dyu / sag 0.4, priors 0.3, consensus 0.5) are never changed. The yaml of the run uses `reliability_from: same_source` (the group-ranking accuracy), and under that rule only 11 languages move,
  by at most +0.12, none down: nus 0.86 -> 0.98, umb 0.79 -> 0.87, kam 0.82 -> 0.90, kbp 0.86 -> 0.91, fon 0.91 -> 0.96, mos 0.90 -> 0.95, lua 0.96 -> 1.00, kab, efi, kmb, bam by < 0.03 (mean over the 44 languages 0.942 -> 0.954).
  Under the AUC rule (`reliability_from: auc`) 29 languages would go up (mean 0.74 -> 0.90; blend languages 0.36 -> 0.69; kbp 0.28 -> 0.82, yor 0.61 -> 0.98). `keep` = the v7 weights unchanged. See risks.
* **Flags** (reward block): `qe_ensemble`, `qe_ensemble_delta` (null = `_meta_v8.delta` of the file = 1.0), `qe_ensemble_model_id` (checked against the model the anchors were measured for),
  `qe_ensemble_calibration_path` (null = `calibration_path`; v8 is a superset of v7), `qe_ensemble_prior_langs`, `qe_ensemble_reliability`. Scorer block: `ensemble_model_id`, optional `ensemble_batch_size`, `max_batch_chars`.
* **Logs** `reward/qe_ens/*`: chunks, seconds_primary / seconds_secondary, joint_split_rate, both_share / only1_share / only2_share, raw_mean/{africomet,ssa_mtl}, abs_diff_mean, signed_diff_mean (SSA minus AfriCOMET, shaped),
  min_share, min_share_ssa_high (only SSA likes it: the copy-hack signature), min_share_afri_high, r1_mean, r2_mean, combined_mean, and per language `min_share/<iso>` and `abs_diff/<iso>`.
  `RewardOutput.qe2` holds the mean raw SSA score per sample.

## 2. Choice of delta = 1.0

Data: the A5 policy samples (`/workspace/jobs/a5`, final RL checkpoint at T=1, 8 samples x 80 FLORES+ dev prompts x 45 languages; 27,502 gate-open samples judged by both models) and the A5 hack classes
(43 languages x 80 sentences x 12 classes), scored with the v8 anchors (`scripts/calibrate_qe_ensemble.py delta`; full grid and per-language tables in `reports/qe_ensemble_2026-10-21/delta.md`).
Rule: the smallest delta whose min-branch share on the policy samples is <= 15% (target 10-15%).
* The two shaped rewards differ a lot even on good outputs: |r1 - r2| on policy samples p50 0.48, p75 0.79, p90 1.12, p99 2.02 (the curves are steep: 1 reward unit = t_fail -> t_mid).
* delta 1.0: **14.0%** of policy chunks in the min branch (qe languages 12.3%, blend languages 19.3%; in the 4 real-sample reward steps 10.7 / 12.5 / 13.3 / 14.6%); 1.1 -> 10.6%; 0.9 -> 18.6%; 0.7 -> 31%.
  Languages with a systematic scale offset go to the min branch more often (pcm 69%: SSA rates Pidgin far above AfriCOMET, signed shaped diff +1.25; fon / mos / umb / kmb / wol 28-30%; afr, kbp, tir, swh, run <= 4%), i.e. there the
  reward is mostly the AfriCOMET value, as before.
* The min branch fires on the hack: of exact copies 96% (copy_source), noisy copies 91%, half-copies (mix_half_src) 33%, swapped quarter copies 38%, append_source 17%; of human references 22.5% (they get min(r1, r2) of two high scores,
  a small loss). Failure rate (corrupted reward >= human reward of the same sentence): copy_source 0.8% (AfriCOMET 0.9%, SSA 40.6%, plain mean 8.7%), copy_noisy 3.3% (AfriCOMET 3.7%, SSA 37.5%, mean 12.6%),
  mismatch 6.1% (AfriCOMET 10.1%, SSA 1.7%, mean 2.9%), neighbour 7.2% (11.3 / 1.5 / 3.0), append_unrelated 21.8% (25.6 / 5.1 / 10.2), append_source 41.6% (53.0 / 16.5 / 40.2), mix_half_src 17.6% (17.2 / 13.4 / 12.4).
  Failure rates are flat for delta 1.0-1.5 (copy 0.8-1.0%), so inside the 10-15% band the smaller delta is the cautious choice. The ensemble keeps AfriCOMET's copy robustness and takes about half of SSA's gain on mismatched
  and neighbouring references; append_source and mix_half_src (partial copies) remain the weak spots of every metric (best single: SSA 16.5% / 13.4%).
* Calibration sets (997 sentences / language, same-source accuracy human > copy of the English source): SSA alone 0.56 on average (0.25-0.83), AfriCOMET 0.99, **ensemble 0.99** (min over languages 0.94).

## 3. Recalibration (v8)

`calibration/shaping_flores_dev_v8.json` = v7 byte-for-byte (every v7 key unchanged, tested) + per language a `qe2` block (anchors, usable, reliability, reliability_same_source, AUC / same-source accuracy against mismatched /
copy / half / append, human quantiles, means) + an `ens` block (the ensemble's own AUC and same-source accuracy at delta 1.0, min-branch shares per set, measured and baked reliability) + `_meta_v8`.
Semantics of `scripts/calibrate_shaping.py`: FLORES+ dev (devtest never used), 997 sentences per language, fail_floor 0.0 (t_fail = the language's own P90 on mismatched references, t_mid = human P50, t_great = human P90),
corruptions mismatch (reference of another sentence) / copy (English source) / half (first half of the words) / append (reference + another reference), same sentences for both models. 46 languages have human references: the 44
FLORES+ dev languages (twi: akua / asan by id parity), pcm (997 of the 1,484 MAFAND dev rows used by v7), efi (IbomMT linguist translation of the FLORES+ devtest English, calibration only, as v7), dyu and sag (FLORES+ dev raw files);
kea and vmw were measured too (consensus mode: stored as `qe2_diagnostic`, never used). The AfriCOMET re-measure agrees with the v7 anchors (max |difference| 0.014). The 10 languages without a reference are listed above.
* Mean AUC human vs mismatched (44 qe + blend languages): AfriCOMET **0.845**, SSA-COMET-MTL **0.982**, ensemble 0.915; qe-mode languages 0.885 / 0.993 / 0.945; blend-mode languages 0.738 / 0.953 / 0.833.
* **Unusable (AUC < 0.7): none for SSA-COMET-MTL (lowest: efi 0.893, dyu 0.904, umb 0.944, fon 0.951, mos 0.952, kbp 0.957), none for both.** AfriCOMET was already below 0.75 for efi, fon, kab, kbp, nus, umb, mos, kam, kmb,
  dyu (blend mode in v7); the ensemble lifts all of them (efi 0.73 -> 0.81, fon 0.74 -> 0.84, nus 0.71 -> 0.86, kbp 0.72 -> 0.86, kab 0.73 -> 0.85, dyu 0.68 -> 0.77, pcm 0.77 -> 0.78). The documented fallback (one model alone when the
  other is unusable, `plan`: only1 / only2) is implemented and tested but not triggered by any language.
* SSA-COMET-MTL's weakness is visible per language: same-source accuracy against the copy of the source is only 0.25-0.37 for efi, sag, nus, dyu, kbp, fon (best 0.82-0.83 for afr, kik; 0.56 on average) - exactly what the min branch fixes.
* pcm remains the weakest language of the ensemble (AUC 0.78, append accuracy 0.47): SSA gives AUC 0.999 but a very different scale than AfriCOMET (pcm goes to the min branch 63% of the time on the calibration sets), so its
  reward is nearly AfriCOMET's; weight 0.46 -> 0.52 only.

Per language (AUC = human vs mismatched; acc = human > corrupted version of the SAME source, columns mismatch / half / copy / append; the last column is the AUC-based reliability; the run's `reliability_from: same_source` weights are in the file, see section 1):

Per-language calibration (FLORES+ dev, n=997 unless stated; delta 1.0). A = AfriCOMET-QE-STL-1.1, S = SSA-COMET-MTL reference-free; t_fail/t_mid/t_great = reward 0/1/2 anchors; AUC = human vs mismatched reference; acc = human > corrupted version of the SAME source.

| lang | mode | n | A anchors | A AUC | A acc mism/half/copy/append | S anchors | S AUC | S acc mism/half/copy/append | S usable | ens AUC | ens acc mism/half/copy/append | ens min-branch share human/copy | weight v7 -> v8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| afr | qe | 997 | 0.536/0.743/0.845 | 0.982 | 0.99/0.99/1.00/0.90 | 0.302/0.659/0.726 | 1.000 | 1.00/1.00/0.82/0.99 | yes | 0.996 | 1.00/0.99/1.00/0.95 | 0.03/0.96 | 1.00 -> 1.00 |
| amh | qe | 997 | 0.578/0.727/0.839 | 0.960 | 0.98/1.00/1.00/0.76 | 0.134/0.553/0.678 | 0.999 | 1.00/0.99/0.55/0.99 | yes | 0.993 | 1.00/1.00/1.00/0.88 | 0.04/1.00 | 1.00 -> 1.00 |
| bam | blend | 997 | 0.469/0.519/0.629 | 0.807 | 0.85/0.95/0.98/0.64 | 0.326/0.512/0.665 | 0.981 | 0.98/0.97/0.47/0.94 | yes | 0.884 | 0.90/0.90/0.98/0.70 | 0.34/0.94 | 0.64 -> 0.94 |
| bem | qe | 997 | 0.519/0.569/0.696 | 0.861 | 0.90/0.95/0.99/0.73 | 0.380/0.624/0.720 | 0.991 | 0.99/0.98/0.69/0.95 | yes | 0.935 | 0.94/0.95/0.99/0.79 | 0.22/0.89 | 0.84 -> 1.00 |
| dyu | blend | 997 | 0.461/0.511/0.561 | 0.686 | 0.72/0.89/0.96/0.45 | 0.391/0.449/0.564 | 0.904 | 0.91/0.92/0.31/0.79 | yes | 0.773 | 0.80/0.86/0.96/0.52 | 0.42/0.88 | 0.40 -> 0.40 |
| efi | blend | 997 | 0.476/0.526/0.582 | 0.733 | 0.77/0.93/0.97/0.65 | 0.284/0.399/0.575 | 0.893 | 0.90/0.94/0.25/0.84 | yes | 0.810 | 0.82/0.90/0.96/0.70 | 0.42/0.96 | 0.32 -> 0.64 |
| ewe | qe | 997 | 0.552/0.602/0.698 | 0.831 | 0.89/0.97/1.00/0.76 | 0.368/0.558/0.693 | 0.980 | 0.99/0.98/0.56/0.95 | yes | 0.911 | 0.94/0.94/0.99/0.78 | 0.24/0.92 | 0.74 -> 1.00 |
| fon | blend | 997 | 0.473/0.523/0.573 | 0.733 | 0.80/0.93/0.97/0.70 | 0.347/0.474/0.607 | 0.951 | 0.96/0.96/0.37/0.89 | yes | 0.839 | 0.88/0.89/0.98/0.69 | 0.39/0.92 | 0.34 -> 0.76 |
| hau | qe | 997 | 0.558/0.683/0.792 | 0.938 | 0.96/0.98/1.00/0.83 | 0.140/0.513/0.674 | 0.998 | 1.00/0.99/0.45/0.99 | yes | 0.988 | 1.00/0.98/1.00/0.90 | 0.07/1.00 | 1.00 -> 1.00 |
| ibo | qe | 997 | 0.536/0.586/0.708 | 0.843 | 0.87/0.95/1.00/0.73 | 0.274/0.520/0.632 | 0.994 | 0.99/0.99/0.46/0.97 | yes | 0.929 | 0.93/0.93/0.99/0.76 | 0.28/0.97 | 0.78 -> 1.00 |
| kab | blend | 997 | 0.487/0.537/0.597 | 0.733 | 0.81/0.96/0.98/0.50 | 0.390/0.542/0.669 | 0.968 | 0.98/0.98/0.52/0.92 | yes | 0.846 | 0.88/0.91/0.98/0.57 | 0.42/0.88 | 0.33 -> 0.79 |
| kam | blend | 997 | 0.451/0.501/0.609 | 0.739 | 0.76/0.89/0.96/0.56 | 0.432/0.572/0.722 | 0.959 | 0.97/0.96/0.60/0.82 | yes | 0.817 | 0.85/0.87/0.94/0.57 | 0.40/0.79 | 0.36 -> 0.67 |
| kbp | blend | 997 | 0.547/0.597/0.647 | 0.722 | 0.78/0.93/0.99/0.71 | 0.326/0.455/0.595 | 0.957 | 0.96/0.95/0.32/0.95 | yes | 0.856 | 0.87/0.86/1.00/0.72 | 0.40/0.95 | 0.28 -> 0.82 |
| kea | consensus | 997 | 0.369/0.419/0.551 | 0.796 | 0.82/0.85/0.92/0.67 | 0.336/0.497/0.623 | 0.985 | 0.99/0.93/0.42/0.93 | yes | - | - | - | 0.50 (unchanged) |
| kik | qe | 997 | 0.527/0.577/0.693 | 0.853 | 0.90/0.96/0.99/0.78 | 0.437/0.696/0.841 | 0.984 | 0.99/0.97/0.83/0.96 | yes | 0.928 | 0.95/0.95/0.99/0.80 | 0.22/0.82 | 0.82 -> 1.00 |
| kin | qe | 997 | 0.591/0.682/0.788 | 0.933 | 0.97/0.98/1.00/0.81 | 0.396/0.639/0.697 | 0.997 | 1.00/0.99/0.74/0.99 | yes | 0.978 | 0.99/0.98/1.00/0.89 | 0.09/0.90 | 1.00 -> 1.00 |
| kmb | blend | 997 | 0.468/0.518/0.596 | 0.739 | 0.77/0.90/0.97/0.58 | 0.422/0.566/0.684 | 0.970 | 0.97/0.98/0.58/0.90 | yes | 0.800 | 0.82/0.88/0.96/0.59 | 0.45/0.82 | 0.35 -> 0.60 |
| lin | qe | 997 | 0.501/0.569/0.723 | 0.882 | 0.92/0.95/1.00/0.80 | 0.246/0.576/0.705 | 0.996 | 1.00/0.98/0.61/0.98 | yes | 0.951 | 0.95/0.94/1.00/0.81 | 0.21/0.97 | 0.92 -> 1.00 |
| lua | qe | 997 | 0.446/0.496/0.634 | 0.809 | 0.84/0.92/0.97/0.75 | 0.327/0.528/0.688 | 0.981 | 0.98/0.97/0.49/0.95 | yes | 0.879 | 0.90/0.90/0.97/0.75 | 0.31/0.92 | 0.65 -> 0.92 |
| lug | qe | 997 | 0.520/0.570/0.694 | 0.826 | 0.86/0.93/1.00/0.69 | 0.365/0.567/0.677 | 0.982 | 0.99/0.97/0.56/0.95 | yes | 0.899 | 0.92/0.91/1.00/0.73 | 0.31/0.92 | 0.67 -> 1.00 |
| luo | qe | 997 | 0.572/0.622/0.746 | 0.849 | 0.90/0.98/1.00/0.71 | 0.352/0.651/0.766 | 0.994 | 0.99/0.99/0.76/0.92 | yes | 0.942 | 0.95/0.95/1.00/0.75 | 0.24/0.94 | 0.81 -> 1.00 |
| mlg | qe | 997 | 0.573/0.690/0.804 | 0.948 | 0.97/0.99/1.00/0.80 | 0.254/0.604/0.694 | 0.999 | 1.00/0.99/0.67/0.99 | yes | 0.985 | 0.99/0.99/1.00/0.90 | 0.08/0.98 | 1.00 -> 1.00 |
| mos | blend | 997 | 0.536/0.586/0.654 | 0.751 | 0.80/0.92/0.99/0.53 | 0.393/0.514/0.654 | 0.952 | 0.96/0.93/0.45/0.89 | yes | 0.850 | 0.87/0.89/0.99/0.63 | 0.35/0.90 | 0.39 -> 0.80 |
| nso | qe | 997 | 0.523/0.608/0.755 | 0.894 | 0.94/0.98/1.00/0.81 | 0.276/0.568/0.697 | 0.996 | 1.00/0.99/0.59/0.98 | yes | 0.954 | 0.97/0.97/1.00/0.85 | 0.16/0.97 | 0.99 -> 1.00 |
| nus | blend | 997 | 0.459/0.509/0.565 | 0.711 | 0.75/0.94/0.96/0.68 | 0.261/0.423/0.573 | 0.973 | 0.98/0.96/0.26/0.91 | yes | 0.858 | 0.88/0.90/0.97/0.67 | 0.44/0.97 | 0.26 -> 0.83 |
| nya | qe | 997 | 0.558/0.640/0.763 | 0.906 | 0.94/0.98/1.00/0.78 | 0.343/0.624/0.731 | 0.997 | 1.00/1.00/0.74/0.99 | yes | 0.959 | 0.98/0.97/1.00/0.83 | 0.13/0.94 | 1.00 -> 1.00 |
| orm | qe | 997 | 0.594/0.644/0.761 | 0.852 | 0.91/0.97/1.00/0.62 | 0.370/0.573/0.718 | 0.990 | 0.99/0.96/0.63/0.97 | yes | 0.937 | 0.95/0.96/1.00/0.76 | 0.20/0.93 | 0.79 -> 1.00 |
| pcm | qe | 997 | 0.456/0.506/0.566 | 0.766 | 0.82/0.89/0.99/0.44 | 0.326/0.584/0.653 | 0.999 | 1.00/0.99/0.55/0.98 | yes | 0.780 | 0.81/0.82/0.99/0.47 | 0.63/0.94 | 0.46 -> 0.52 |
| run | qe | 997 | 0.579/0.650/0.762 | 0.907 | 0.94/0.98/1.00/0.76 | 0.414/0.625/0.691 | 0.993 | 1.00/0.99/0.71/0.98 | yes | 0.966 | 0.98/0.98/1.00/0.85 | 0.13/0.88 | 1.00 -> 1.00 |
| sag | blend | 997 | 0.430/0.480/0.581 | 0.782 | 0.82/0.91/0.96/0.75 | 0.246/0.419/0.563 | 0.986 | 0.99/0.97/0.25/0.93 | yes | 0.859 | 0.90/0.87/0.96/0.73 | 0.39/0.97 | 0.40 -> 0.40 |
| sna | qe | 997 | 0.567/0.647/0.752 | 0.910 | 0.95/0.98/1.00/0.78 | 0.339/0.608/0.701 | 0.997 | 1.00/0.99/0.68/0.97 | yes | 0.961 | 0.97/0.97/1.00/0.82 | 0.16/0.94 | 1.00 -> 1.00 |
| som | qe | 997 | 0.504/0.620/0.767 | 0.922 | 0.94/0.98/1.00/0.80 | 0.220/0.495/0.641 | 0.996 | 1.00/0.99/0.41/0.98 | yes | 0.969 | 0.98/0.97/1.00/0.86 | 0.11/0.99 | 1.00 -> 1.00 |
| sot | qe | 997 | 0.543/0.636/0.758 | 0.922 | 0.96/0.99/1.00/0.83 | 0.302/0.612/0.717 | 0.997 | 1.00/0.99/0.70/0.98 | yes | 0.967 | 0.98/0.98/1.00/0.88 | 0.12/0.96 | 1.00 -> 1.00 |
| ssw | qe | 997 | 0.571/0.643/0.772 | 0.892 | 0.93/0.98/1.00/0.73 | 0.259/0.558/0.653 | 0.995 | 1.00/0.98/0.52/0.98 | yes | 0.961 | 0.97/0.96/1.00/0.81 | 0.17/0.98 | 0.97 -> 1.00 |
| swh | qe | 997 | 0.559/0.721/0.827 | 0.961 | 0.97/0.98/1.00/0.86 | 0.290/0.643/0.736 | 0.998 | 1.00/1.00/0.77/0.99 | yes | 0.988 | 0.99/0.99/1.00/0.93 | 0.04/0.96 | 1.00 -> 1.00 |
| tir | qe | 997 | 0.592/0.661/0.779 | 0.896 | 0.93/0.98/1.00/0.67 | 0.260/0.580/0.691 | 0.996 | 0.99/0.99/0.58/0.96 | yes | 0.967 | 0.98/0.98/1.00/0.79 | 0.14/0.98 | 0.97 -> 1.00 |
| tsn | qe | 997 | 0.539/0.625/0.754 | 0.902 | 0.94/0.98/1.00/0.79 | 0.295/0.596/0.706 | 0.996 | 1.00/0.98/0.62/0.98 | yes | 0.956 | 0.97/0.97/1.00/0.83 | 0.16/0.96 | 1.00 -> 1.00 |
| tso | qe | 997 | 0.566/0.639/0.762 | 0.896 | 0.94/0.98/1.00/0.78 | 0.351/0.640/0.733 | 0.996 | 0.99/0.98/0.73/0.98 | yes | 0.963 | 0.98/0.96/1.00/0.86 | 0.15/0.94 | 0.98 -> 1.00 |
| tum | qe | 997 | 0.592/0.663/0.768 | 0.900 | 0.93/0.97/1.00/0.75 | 0.413/0.654/0.746 | 0.991 | 0.99/0.99/0.78/0.97 | yes | 0.962 | 0.97/0.97/1.00/0.81 | 0.15/0.89 | 0.99 -> 1.00 |
| twi | qe | 997 | 0.519/0.569/0.678 | 0.836 | 0.88/0.94/1.00/0.74 | 0.294/0.534/0.658 | 0.994 | 1.00/0.98/0.49/0.97 | yes | 0.921 | 0.94/0.92/0.99/0.74 | 0.30/0.96 | 0.72 -> 1.00 |
| umb | blend | 997 | 0.477/0.527/0.595 | 0.719 | 0.75/0.88/0.95/0.57 | 0.397/0.528/0.661 | 0.944 | 0.94/0.94/0.47/0.86 | yes | 0.802 | 0.82/0.87/0.96/0.60 | 0.41/0.86 | 0.28 -> 0.61 |
| vmw | consensus | 997 | 0.434/0.484/0.534 | 0.650 | 0.70/0.82/0.94/0.43 | 0.476/0.526/0.616 | 0.884 | 0.89/0.94/0.47/0.74 | yes | - | - | - | 0.50 (unchanged) |
| wol | qe | 997 | 0.530/0.580/0.673 | 0.799 | 0.85/0.96/1.00/0.61 | 0.465/0.634/0.760 | 0.967 | 0.97/0.98/0.72/0.82 | yes | 0.873 | 0.89/0.91/0.99/0.62 | 0.36/0.79 | 0.57 -> 0.89 |
| xho | qe | 997 | 0.524/0.614/0.759 | 0.905 | 0.93/0.97/1.00/0.78 | 0.179/0.506/0.616 | 0.998 | 1.00/0.98/0.40/0.98 | yes | 0.969 | 0.98/0.97/1.00/0.83 | 0.14/0.99 | 1.00 -> 1.00 |
| yor | qe | 997 | 0.525/0.575/0.667 | 0.801 | 0.85/0.97/0.99/0.66 | 0.314/0.545/0.666 | 0.992 | 1.00/0.99/0.52/0.95 | yes | 0.896 | 0.92/0.93/0.99/0.70 | 0.34/0.95 | 0.61 -> 0.98 |
| zul | qe | 997 | 0.546/0.670/0.793 | 0.943 | 0.97/0.98/1.00/0.84 | 0.184/0.541/0.632 | 0.999 | 1.00/0.99/0.49/0.98 | yes | 0.981 | 0.99/0.98/1.00/0.88 | 0.13/0.99 | 1.00 -> 1.00 |

Pooled prior for languages without a dev reference (qe2 anchors): [0.326, 0.567, 0.686] (median of 44 calibrated non-consensus languages).
Languages on the pooled / borrowed prior (weight unchanged): iso, kon, kqn, kua, loz, lun, swc, tiv, ven, wal.
Unusable for SSA-COMET-MTL (AUC < 0.7; fall back to AfriCOMET alone): none.
Unusable for both: none.


## 4. Real-GPU check: old vs new reward on the same samples

`scripts/qe_ensemble_gpu_check.py` (raw: `reports/qe_ensemble_2026-10-21/gpu_check_final.json`): 4 steps of 672 policy samples (84 prompts x 8, mixed languages, the real `reward:` block of `configs/rl_science.yaml` with GlotLID off),
RTX PRO 6000 shared with other jobs, one child worker per scorer.
* **Correlation** of old and new reward over the samples scored by both: Pearson 0.924 / 0.948 / 0.929 / 0.951, Spearman 0.912 / 0.943 / 0.928 / 0.951; mean within-group Spearman (the quantity GRPO uses) 0.77-0.83.
* **Scale:** mean reward old 0.593 / 0.145 / 0.490 / 0.473 vs new 0.553 / 0.180 / 0.473 / 0.463 (|difference| <= 0.04); std 0.81-0.90 -> 0.74-0.81 (about 10% smaller: the min branch trims the top); range -2.0 .. 2.09.
  Advantage weights (yaml rule `same_source`) differ from v7 for 120-208 of 672 samples per step (the 11 raised languages, by <= 0.12).
* **Min branch:** 10.7-14.6% of chunks (target 10-15%).
* **Hack probe** (12 sentences x 45 languages per class from A5, scored alone, mean reward old -> new): human 0.448 -> 0.496, mismatch -0.908 -> -1.083, drop_half -1.254 -> -1.092, append_unrelated -0.781 -> -0.747,
  mix_half_src -0.357 -> -0.266, append_source -0.660 -> -0.686, swap25_src -1.134 -> -0.971, copy_noisy -1.292 -> -1.268; exact copies of the source (copy_source, 540 of 540) are caught by the existing copy guard before QE in both,
  100 of 540 copy_noisy too. The min branch took 92% of the remaining copy_noisy chunks (all with SSA the higher judge). The gap human - mismatched grows 1.36 -> 1.58, human - drop_half shrinks 1.70 -> 1.59 and human - swap25 1.58 -> 1.47.
* **Seconds of QE per 672-sample step** (policy samples = single sentences, 712-824 chunks): AfriCOMET 1.2-1.9 s, SSA-COMET-MTL 1.5-2.1 s, ensemble 2.6-4.0 s in the QE calls (reward call total 3.4-4.4 s vs 2.0-3.4 s old). This is SHORT FLORES
  text; the owner's figure for a mixed step with science documents is 25-40 s for AfriCOMET alone. On a stress test of 384 very long chunks (240 words each, 1,016 joint pieces for SSA) SSA took 13.6 s against 6.1 s for AfriCOMET (2.2x).
  Estimate for the real mixed step: SSA costs 1.2x (sentence chunks) to 2.2x (long un-chunked blocks) of the current QE time, i.e. the QE share of a step goes from ~25-40 s to ~55-90 s (one more model pass, no extra generation). Measure with
  `reward/qe_ens/seconds_primary` and `seconds_secondary` in the first steps.
* **GPU memory of the worker** (nvidia-smi, process level, peak): one worker with both models, scorer.batch_size 128: **6.7 GB** on the typical step, 7.5 GB on the long-chunk stress test (idle 2.9 GB); batch 64: 5.0 GB; batch 256: 11.4 GB
  (without the cost cap and expandable segments: 16 GB at batch 256 and, on the stress test, 13 GB at batch 128 and 23 GB at batch 256). The single-model worker of the current yaml (batch 256) peaks at 8.1 GB, at batch 128 6.0 GB. Recommended: `batch_size: 128`
  (second model on the same batch size), `max_batch_chars: 64000` (default of the pair worker). Scores with the cap are identical to those without it to <= 1e-4.
* Startup: 24-47 s (single) vs 42-75 s (pair; both checkpoints are cached by the hub after the first download).

## 5. Tests

`tests/test_qe_ensemble.py`, 31 tests, CPU, stub scorers: the rule (mean inside delta, min outside, symmetry, delta 0), joint budget (pieces fit, no word lost, sentence-boundary cut), plan / fallbacks / prior languages /
weights / validation, statistics (`qe_ens/*`), config parsing and the delta from the calibration file, `QEPairScorer`, the two-model worker (fake mode, protocol unchanged for model 0), the reward integration (rule applied,
disagreement takes the lower judge, language without second model never calls it, chunked reward scores every chunk with both, **flag off = old reward bit-for-bit**, blend keeps the consensus term), the CPU smoke
`run_smoke` with the ensemble on (reward/qe_ens/* in the metrics), compose / delta of the calibration script on synthetic scores (the copy weakness of a joint encoder is fixed by the min branch, owner weights kept, unusable
secondary falls back), the shipped v8 file is a superset of v7, cost-capped batching keeps order. Full suite on the box: 852 passed, 8 skipped, 6 failed, none in this change (test_lexicon x4 and test_b2048_bin
`kl_estimator` belong to the D-56 / run-control work, test_stratified pusher depends on an HF token in the environment).

## 6. Enable in the yaml

```yaml
reward:
  calibration_path: /workspace/sft_build/gemma_mt/calibration/shaping_flores_dev_v8.json   # v7 + qe2 / ens blocks (copy the file there first)
  qe_ensemble: true
  # qe_ensemble_delta: 1.0            # null = _meta_v8.delta (1.0); watch reward/qe_ens/min_share (expected 0.10-0.15) and retune here
  # qe_ensemble_reliability: keep     # fallback: leave the v7 advantage weights unchanged
scorer:
  type: worker
  python: /workspace/venv_gemma/bin/python
  pythonpath: /workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs
  model_id: masakhane/africomet-qe-stl-1.1
  ensemble_model_id: McGill-NLP/ssa-comet-mtl      # present = ONE worker with both checkpoints
  batch_size: 128                                  # 256 would peak at 11.4 GB for the pair
```
Flag off = the old reward (v8 can be used with `qe_ensemble: false`, it is a superset of v7). The patch applies to the tree without D-52 / D-56: `patch -p1 < patches/qe_ensemble_D55.patch`.

## 7. Risks and open points

* **Advantage weights** (default `ensemble`): with the run's `same_source` rule 11 languages gain at most +0.12 (nus, umb, kam, kbp, fon, mos mostly), a minor second change next to the metric change; with `reliability_from: auc` it would be
  much larger (29 languages, blend 0.36 -> 0.69), so do not switch that rule without re-reading section 1. `qe_ensemble_reliability: keep` removes the change.
* **Mean branch still inherits SSA's lenience** for partial copies (append_source 41.6% failures, mix_half_src 17.6%): the existing copy guard and GlotLID (not active in this check) are the main defence there; the min-branch share per
  language (`reward/qe_ens/min_share/<iso>`) and `min_share_ssa_high` are the monitors. A policy that learns to hit the region where the two judges agree but are both wrong cannot be excluded.
* delta was fitted on single FLORES-style sentences of the final checkpoint; science text / long chunks may shift |r1 - r2| (watch `abs_diff_mean`, `min_share`).
* pcm (69% min branch) and the languages with a large scale offset are effectively scored by AfriCOMET alone; their v7 weakness is unchanged.
* Time: +~1.2-2.2x of the current QE seconds per step; the pair worker needs ~7 GB (idle 3 GB) next to the policy at batch 128.
* SSA-COMET-MTL needs `transformers` 4.x like AfriCOMET (same child process). Joint-budget splits (`joint_split_rate`) were 0 on the sentence-level steps; they only occur for chunks over 508 tokens.
* The v8 file lives in `/workspace/jobs/b1/gemma_mt/calibration/` on the box; it must be copied to the live calibration directory by whoever launches the run (nothing outside `/workspace/jobs/b1` was modified).
