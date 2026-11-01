# Which QE metric should be the GRPO reward? AfriCOMET-QE-STL-1.1 vs SSA-COMET (D-53, 2026-10-21)

Agent A5. Everything below was measured on the vast box (`/workspace/jobs/a5/`), nothing was pushed anywhere. Raw tables (all languages, all pairs, all classes): `reports/qe_metric_choice_2026-10-21/{ssamte,afrimte,policy,hack,anchors}_report.md`. Scripts: `gemma_mt/scripts/qe_choice_{score,sample,study}.py` (section 8).

## 1. Bottom line

* **The 2026-10-15 conclusion ("do not swap", based on AUC against synthetic corruptions) stands for SSA-COMET-QE (the STL model) but not for the question we actually care about.** Against real human adequacy scores our model of record, AfriCOMET-QE-STL-1.1 used reference-free, is behind SSA-COMET-MTL on both human datasets and behind SSA-COMET-QE on SSA-MTE (level with it on AfriMTE-WMT2024). On SSA-MTE (7 en->X test pairs, 8,345 segments) Spearman is **0.470 for ours vs 0.550 SSA-COMET-QE vs 0.604 SSA-COMET-MTL (reference-free)**; on the independent AfriMTE-WMT2024 challenge set (9 en->X pairs, 1,916 segments) **0.413 vs 0.421 vs 0.454**. I reproduce the SSA-COMET paper's numbers (their 0.547 / 0.595; mine 0.550 / 0.604).
* **SSA-COMET-QE (STL) is not worth switching to.** Its human-agreement gain is large on SSA-MTE (+0.081, partly in-domain: it was trained on the SSA-MTE train split) and zero out of domain (+0.007, CI [-0.010, +0.023] on AfriMTE-WMT2024). It is worse than ours at separating human from mismatched translations (AUC -0.044, 45/45 languages, hard-class failure rate +0.051, 45/45) and worse at ranking real policy samples (within-prompt Spearman with chrF++ -0.025). It is dominated.
* **SSA-COMET-MTL run reference-free (src + mt only) is the strongest single judge**: best on humans (+0.135 on SSA-MTE, +0.040 [+0.011, +0.068] on AfriMTE-WMT2024), best at human-vs-mismatch (AUC 0.983 vs 0.866), best at within-prompt ranking of policy samples against chrF++ (+0.019 [+0.009, +0.029]) **but it is hackable by copying/mixing the English source** (exact copy of the source scores >= the human reference in 41% of sentences, a source with 20% of the words dropped in 38%; ours 1.3% / 4.5%). The existing guards (Jaccard 0.9 copy guard + GlotLID `wrong_language`) remove most of it (residual 1.7% vs 0.4% for ours on the 20%-dropped copy) but it is a new reward-hacking surface that a policy under RL will probe.
* **Recommendation: an ensemble, not a switch.** Average the per-language *shaped* rewards of AfriCOMET-QE-STL-1.1 and SSA-COMET-MTL (reference-free) (`ens_shaped` below; the z-score average `ens_afri_ssamtl` behaves the same on the policy and corruption sets, and is what the human-label rows use). It keeps most of the human-agreement gain (z-mean variant: AfriMTE 0.465, i.e. at least as good as MTL alone; SSA-MTE 0.584 vs 0.604), cuts the hard-class failure rate from 0.109 to 0.044, and after the existing guards its residual copy-hack rate is 0.6% (ours 0.4%, MTL alone 1.7%). Expected effect on translation quality is **small** (see section 6): the best metric changes the chrF++ of the group's top pick by ~0.1 point (CI overlaps); the benefit is a cleaner, less noisy ranking signal, and better scoring of the weak/blend languages. Cost is negligible (about +0.4 s of GPU scoring per 128-sample step) but it needs code (UnifiedMetric support, joint-length budget) and a full recalibration. **Do not restart the current run for it; apply it at the next stage boundary** after the checklist in section 7.

## 2. What was measured and how

Models (all fp16 on GPU through the same `CometScorer` path as the RL reward; unbabel-comet 2.2.7, transformers 4.x): `masakhane/africomet-qe-stl-1.1` (ours), `McGill-NLP/ssa-comet-qe`, `McGill-NLP/ssa-comet-mtl` (UnifiedMetric; "qe" = only src + mt are passed, so a single forward over the concatenated [mt, src]; "ref" = src + mt + ref, mean of three views), `masakhane/africomet-stl-1.1` (with reference), `masakhane/africomet-mtl` (the only MTL checkpoint on the Hub is v1.0; the paper's baseline "AfriCOMET v1.1-MTL" is not available, so my AfriCOMET-MTL rows are v1.0).

Ensembles (`ens_*`): `ens_afri_ssaqe` = z(AfriCOMET-QE) + z(SSA-COMET-QE); `ens_afri_ssamtl` = z(AfriCOMET-QE) + z(SSA-COMET-MTL-qe), averaged; `ens3` = all three; `ens_min_afri_ssamtl` = min of the two z-scores; `ens_shaped` = mean of the two per-language shaped rewards (reward_shaping curve, floor 0, anchors measured per language from human references, each metric with its own anchors). z uses ONE (mean, sd) per metric pooled over the dataset (a fixed affine map).

Datasets:

| dataset | what | n | note |
|---|---|---|---|
| SSA-MTE test (`McGill-NLP/SSA-MTE`, cc-by-sa-4.0, not gated) | human DA, z-scored; 6 MT systems per source | 12,769 over 11 pairs (7 en->X: amh hau kik kin luo twi yor) | the SSA-COMET training split is the same domain/systems (document-level split) -> in-domain for SSA-COMET |
| AfriMTE-WMT2024 (`masakhane/AfriMTE-WMT2024`, the WMT24 African challenge set) | human z-scores, one hypothesis per source | 2,815 over 13 pairs (9 en->X: hau ibo kik luo som swh twi xho yor) | independent of SSA-COMET training as far as I can tell (the paper says AfriMTE dev was used as validation); overlap with AfriCOMET-1.1 training not verifiable |
| policy samples | final model `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c@latest` (sha 80baefff), 8 samples at T=1, top_p=1, top_k=0 (the RL setting), HF generate bf16, production prompt format | 45 languages x 80 prompts x 8 = 28,800 (99.9% valid) | prompts = the same seeded 80 FLORES+ **dev** sentence ids for every language (pcm: MAFAND dev); no devtest |
| corruption set | the 49-language protocol re-implemented on those 80 sentences/language | 45 languages x 80 x 12 classes = 43,200 per metric | classes: human, mismatch, neighbour, copy_source, copy_noisy, drop_half, shuffle, append_unrelated, replace_words + 3 new partial-copy classes |

The scripts of the 2026-10-15 study are no longer on the box (`/workspace/sft_build/ssa_cmp` is gone), so the protocol was re-implemented from the report; numbers are consistent (mean AUC difference SSA-QE vs ours -0.044 here, -0.039 there; hard-class failure +0.051 vs +0.048).

Intervals are 90% bootstrap intervals (1000 draws for human datasets, resampling segments within a pair; 400 draws for the policy/hack studies, resampling prompts/sentences within a language). "Better/worse" counts are languages where the point estimate is higher/lower.

## 3. Task 1: agreement with human adequacy scores

### 3.1 SSA-MTE test, Spearman with the human z-score (reference-free metrics; ref-based for context)

| pair (our mode) | n | **ours: AfriCOMET-QE-STL-1.1** | SSA-COMET-QE | SSA-COMET-MTL (qe) | ens_afri_ssamtl | ref: AfriCOMET-STL-1.1 | ref: SSA-COMET-MTL |
|---|---|---|---|---|---|---|---|
| eng-amh (qe) | 1166 | 0.547 | 0.552 | 0.594 | 0.597 | 0.588 | 0.630 |
| eng-hau (qe) | 1192 | 0.351 | 0.390 | 0.430 | 0.423 | 0.465 | 0.501 |
| eng-kik (qe) | 1172 | 0.592 | 0.685 | 0.730 | 0.708 | 0.492 | 0.759 |
| eng-kin (qe) | 1210 | 0.397 | 0.511 | 0.563 | 0.542 | 0.507 | 0.601 |
| eng-luo (qe) | 1199 | 0.571 | 0.653 | 0.728 | 0.704 | 0.616 | 0.772 |
| eng-twi (qe) | 1200 | 0.458 | 0.614 | 0.658 | 0.623 | 0.527 | 0.702 |
| eng-yor (qe) | 1206 | 0.374 | 0.448 | 0.530 | 0.493 | 0.442 | 0.558 |
| **avg of the 7 en->X pairs** | 8345 | **0.470** [0.455, 0.484] | 0.550 [0.537, 0.562] | 0.604 [0.592, 0.616] | 0.584 | 0.520 | 0.646 |
| paper (Table 4, QE) | | 0.490 (AfriCOMET v1.1-**MTL**) | 0.547 | 0.595 | | | |
| fra-ewe (qe) | 1077 | 0.388 | 0.526 | 0.591 | 0.557 | 0.484 | 0.639 |
| fra-wol (qe) | 1175 | 0.433 | 0.638 | 0.690 | 0.649 | 0.493 | 0.730 |
| por-vmw (consensus mode) | 930 | 0.120 | 0.199 | 0.249 | 0.204 | 0.239 | 0.328 |
| por-nya (qe) | 1242 | 0.665 | 0.670 | 0.670 | 0.682 | 0.685 | 0.695 |

(The fra-/por- pairs have a non-English source, which is not our en->X direction; shown for completeness. Our `mode` is from `calibration/shaping_flores_dev_v7.json`; all 7 en->X languages and ewe/wol/nya are in the 77, vmw is consensus-mode.)

Paired bootstrap of the 7-pair average, difference vs ours: SSA-COMET-QE **+0.081 [+0.073, +0.088]**, SSA-COMET-MTL (qe) **+0.135 [+0.123, +0.146]**, ens_afri_ssamtl +0.114 [+0.107, +0.121], ens3 +0.113, ens_min +0.104 [+0.095, +0.113], ens_afri_ssaqe +0.067; reference-based AfriCOMET-STL-1.1 +0.050, reference-based SSA-COMET-MTL +0.176. Per pair, SSA-COMET-MTL (qe) is significantly above ours in **all 7** en->X pairs (+0.047 amh, +0.079 hau, +0.138 kik, +0.166 kin, +0.156 luo, +0.199 twi, +0.155 yor); SSA-COMET-QE in 6 of 7 (amh +0.004, not significant). Pearson (7-pair average) ours 0.519, SSA-COMET-QE 0.611, SSA-COMET-MTL 0.669; against the raw 0-100 DA score the Spearman values are within 0.01 of the z-score ones.

GRPO ranks translations of the *same* source, so I also computed the **within-source** agreement (sources with >= 3 translations by different systems, mean Spearman inside the source group, 7-pair average): ours **0.439** [0.423, 0.456], SSA-COMET-QE 0.503 (+0.064 [+0.052, +0.075]), SSA-COMET-MTL (qe) **0.556** (+0.116 [+0.100, +0.131]), ens_afri_ssamtl 0.539 (+0.099).

Other AfriCOMET baselines: AfriCOMET-MTL v1.0 reference-free 0.406 (the weakest; it is much worse than ours on kik/luo/twi/ewe/wol), with a reference 0.526.

### 3.2 AfriMTE-WMT2024 (independent set), Spearman with the human z-score

| pair | n | **ours** | SSA-COMET-QE | SSA-COMET-MTL (qe) | ens_afri_ssamtl | ref: AfriCOMET-STL-1.1 | ref: SSA-COMET-MTL |
|---|---|---|---|---|---|---|---|
| eng-hau | 240 | 0.516 | 0.507 | 0.579 | 0.589 | 0.561 | 0.620 |
| eng-ibo | 120 | 0.550 | 0.559 | 0.521 | 0.575 | 0.522 | 0.556 |
| eng-kik | 202 | 0.478 | 0.476 | 0.460 | 0.510 | 0.431 | 0.531 |
| eng-luo | 242 | 0.392 | 0.401 | 0.484 | 0.475 | 0.326 | 0.505 |
| eng-som | 226 | 0.458 | 0.460 | 0.456 | 0.479 | 0.502 | 0.518 |
| eng-swh | 157 | 0.603 | 0.636 | 0.755 | 0.710 | 0.703 | 0.781 |
| eng-twi | 247 | 0.193 | 0.183 | 0.187 | 0.207 | 0.223 | 0.188 |
| eng-xho | 243 | 0.117 | 0.146 | 0.151 | 0.144 | 0.204 | 0.154 |
| eng-yor | 239 | 0.414 | 0.421 | 0.493 | 0.501 | 0.338 | 0.502 |
| **avg of the 9 en->X pairs** | 1916 | **0.413** [0.377, 0.443] | 0.421 | 0.454 [0.420, 0.483] | 0.465 | 0.423 | 0.484 |

Paired difference vs ours (9-pair average): SSA-COMET-QE **+0.007 [-0.010, +0.023]** (no gain), SSA-COMET-MTL (qe) **+0.040 [+0.011, +0.068]**, ens_afri_ssamtl **+0.052 [+0.036, +0.066]**, ens3 +0.043, ens_min +0.038 [+0.020, +0.057], ens_afri_ssaqe +0.012 [+0.003, +0.020]; reference-based AfriCOMET-STL-1.1 +0.010 [-0.020, +0.037]. Per pair only a few differences are resolvable (n = 120-250): MTL (qe) is significantly better on luo and swh; the ensemble on hau, luo, swh, yor. Twi and xho are near chance for every metric (0.12-0.22). On ibo, kik, som and twi the point estimate of SSA-COMET-MTL (qe) is not above ours (-0.03..0.00).

### 3.3 Reading the human results

* Our model of record is **behind the SSA-COMET models on humans, in every en->X pair of SSA-MTE and on average in AfriMTE-WMT2024**. The size of the gap depends on domain: in domain for SSA-COMET (SSA-MTE) it is 0.08-0.135 Spearman, out of domain about 0.01-0.04 (MTL-qe) and nothing for SSA-COMET-QE. The honest effect size for our RL distribution is therefore somewhere between the two, closer to the out-of-domain figure (+0.04 to +0.05 for the metrics/ensemble).
* Reference-free AfriCOMET-STL-1.1 is, on SSA-MTE, even below its own reference-based sibling (0.470 vs 0.520), as expected; SSA-COMET-MTL loses only 0.04 without reference.
* Caveats: (1) MT systems in both sets are Gemini/GPT-4o/Claude/Google/NLLB/M2M, not a 270M model with its own error profile; (2) humans are z-scored per annotator in SSA-MTE; (3) I could not verify overlap of either test set with the AfriCOMET-1.1 training data.

## 4. Task 2: ranking of realistic policy samples (GRPO-relevant)

45 languages with human references (FLORES+ dev; pcm from MAFAND dev), 80 prompts x 8 samples each. A sample is "valid" if it finished with an end-of-turn, has the right tag, is non-empty and is not a verbatim copy (99.9% were; so there is no valid/invalid selection effect here). Statistic: Spearman *within the 8 samples of a prompt* (what a group-standardised GRPO advantage depends on) averaged over prompts, then over languages. Proxies for "what a human would prefer": (a) sentence chrF++ against the reference (independent of all metrics); (b) reference-based AfriCOMET-STL-1.1 and (c) reference-based SSA-COMET-MTL; each reference-based proxy is biased toward its own model family (the MTL-qe view and MTL-ref are the same weights: 0.87 correlated), so (d) is the z-mean of both. `consensus` = the existing reward component (mean chrF++ against the other valid siblings, `rl/consensus.py`, min_valid 3).

Within-prompt Spearman, mean over the 45 languages [90% CI]; "diff" is vs ours; "sel. gain" = target value of the sample the metric ranks first minus the group mean, in chrF++ points (oracle = best sample in the group):

| metric | vs chrF++ | diff vs ours | langs better/worse | sel. gain chrF++ pts | vs ref-AfriCOMET | vs ref-SSA-COMET-MTL | vs z-mean of both ref-COMETs |
|---|---|---|---|---|---|---|---|
| **ours (AfriCOMET-QE-STL-1.1)** | 0.145 [0.134, 0.156] | - | - | 0.87 [0.78, 0.98] | 0.696 | 0.537 | 0.686 |
| SSA-COMET-QE | 0.120 [0.110, 0.132] | -0.025 [-0.031, -0.019] | 10 / 35 | 0.70 [0.60, 0.81] | 0.650 | 0.566 | 0.676 |
| SSA-COMET-MTL (qe) | 0.164 [0.152, 0.174] | +0.019 [+0.009, +0.029] | 27 / 18 | 0.86 [0.75, 0.96] | 0.492 | 0.873 | 0.753 |
| ens_afri_ssaqe | 0.137 | -0.008 [-0.012, -0.005] | 14 / 31 | 0.86 | 0.693 | 0.568 | 0.701 |
| ens_afri_ssamtl (z-mean) | 0.170 [0.159, 0.181] | +0.025 [+0.018, +0.031] | 36 / 9 | 0.99 [0.88, 1.09] | 0.654 | 0.794 | 0.819 |
| ens_min_afri_ssamtl | 0.164 | +0.019 [+0.012, +0.026] | 36 / 9 | 0.93 | 0.605 | 0.739 | 0.750 |
| ens3 | 0.158 | +0.013 [+0.007, +0.018] | 30 / 15 | 0.93 | 0.682 | 0.734 | 0.800 |
| **ens_shaped** | 0.172 [0.160, 0.182] | +0.026 [+0.020, +0.033] | 36 / 9 | 0.99 | 0.658 | 0.774 | 0.807 |
| consensus (current signal) | 0.370 [0.360, 0.379] | +0.224 | 45 / 0 | 1.99 [1.91, 2.06] | 0.212 | 0.246 | 0.254 |
| oracle (best of 8) | | | | 5.12 | | | |

By reward mode (chrF++ target, diff vs ours): **qe languages (32)**: SSA-COMET-MTL (qe) +0.024 [+0.012, +0.037], ens_afri_ssamtl +0.027 [+0.020, +0.035], ens_shaped +0.027, SSA-COMET-QE -0.028. **blend languages (11: bam dyu fon kab kam kbp kmb mos nus sag umb)**: SSA-COMET-MTL (qe) +0.003 [-0.020, +0.022] (no gain alone), ens_afri_ssamtl +0.016 [+0.003, +0.029], ens_shaped +0.021 [+0.006, +0.036]. **consensus languages**: only kea and vmw have FLORES+ dev references (the other 21 consensus-only languages have none, so nothing can be said about them from references): ens3 +0.041 [+0.017, +0.064], ens_shaped +0.035 [+0.004, +0.062]; the other options +0.02..0.03 with intervals including 0 (n = 2 languages). Per-language numbers are in `policy_report.md`.

What this says:

* **All QE metrics are weak rankers inside a group of 8 samples of the same 270M model**: within-prompt Spearman with chrF++ is 0.12-0.17 for the best of them, i.e. the QE-picked sample is 0.9 chrF++ points above the group mean out of a possible 5.1. The existing `consensus` signal is far stronger against chrF++ (0.37; partly by construction, since both are chrF++ based) and also weakly aligned with reference-COMET (0.21-0.25), which is why consensus is only a blend component, and why the choice among QE metrics changes the training signal by little.
* Against the **independent** proxy (chrF++), SSA-COMET-MTL (qe) is better than ours (+0.019, significant in 27 of 45 languages vs 18 worse) and SSA-COMET-QE is worse (-0.025). Against the reference-COMET proxies each model wins on its own family (ours 0.696 vs SSA-MTL 0.492 with the AfriCOMET proxy; 0.537 vs 0.873 with the SSA proxy), which is the circularity I expected; neither is neutral. The **z-mean of both** proxies, which is half built from AfriCOMET, still prefers SSA-COMET-MTL (qe) 0.753 vs 0.686 and the ensembles most (0.80-0.82, because they contain both families); treat this as weak evidence only.
* Redundancy: within a group, ours and SSA-COMET-QE are 0.82 correlated, but SSA-COMET-MTL (qe) is only 0.53 correlated with ours, so the two carry different information (an ensemble is a real change of signal; adding SSA-COMET-QE is nearly a no-op: ens_afri_ssaqe 0.92 correlated with ours).
* Pooled Spearman over all samples (not within prompt): ours 0.429 / SSA-QE 0.305 / SSA-MTL(qe) 0.479 / ens_afri_ssamtl 0.495 vs chrF++ (confounded by language and prompt difficulty; shown in `policy_report.md`).

## 5. Task 3: hackability (human vs corrupted, 45 languages x 80 sentences)

Mean over the 45 languages; failure = corrupted version scores >= the human reference of the same sentence. Hard classes = mismatch, neighbour, drop_half, append_unrelated, replace_words.

| metric | AUC human vs mismatch [CI] | hard-class failure [CI] | copy_source | copy_noisy (20% words dropped) | partial copy mix_half_src / append_source / swap25_src | langs better (AUC) / worse |
|---|---|---|---|---|---|---|
| **ours** | 0.866 [0.860, 0.872] | 0.109 [0.104, 0.113] | 0.013 | 0.045 | 0.18 / 0.55 / 0.02 | - |
| SSA-COMET-QE | 0.822 [0.814, 0.829] (-0.044) | 0.160 [0.155, 0.165] (+0.051) | 0.001 | 0.004 | 0.07 / 0.35 / 0.02 | 0 / 45 |
| SSA-COMET-MTL (qe) | 0.983 [0.981, 0.985] (+0.117) | 0.027 [0.025, 0.029] (-0.081) | **0.409** | **0.384** | 0.14 / 0.17 / 0.02 | 45 / 0 |
| ens_afri_ssamtl (z-mean) | 0.970 (+0.104) | 0.036 (-0.073) | 0.088 | 0.133 | 0.13 / 0.37 / 0.01 | 45 / 0 |
| ens_min_afri_ssamtl | 0.972 (+0.106) | 0.045 (-0.064) | 0.021 | 0.071 | 0.16 / 0.34 / 0.02 | 45 / 0 |
| ens3 | 0.945 (+0.079) | 0.058 (-0.051) | 0.016 | 0.032 | 0.09 / 0.36 / 0.01 | 45 / 0 |
| **ens_shaped** | 0.960 [0.957, 0.964] (+0.094) | 0.044 [0.040, 0.046] (-0.065) | 0.088 | 0.125 | 0.13 / 0.41 / 0.01 | 45 / 0 |

(All intervals for the differences exclude 0; see `hack_report.md` for per-class, per-mode and per-language tables. The same holds for the 32 qe-mode and the 11 blend-mode languages separately: SSA-COMET-MTL (qe) AUC 0.991 / 0.963 vs ours 0.903 / 0.777; the weak "blend" languages bam dyu fon kab kam kbp kmb mos nus sag umb gain the most, +0.185 AUC.)

**SSA-COMET-MTL reference-free is a genuinely better discriminator** of wrong-content translations (mismatched failure 0.019 vs 0.104; appended unrelated text 0.055 vs 0.267), because its single joint encoding of [mt, src] sees both texts at once, **and for the same reason it rewards copying the source**: when mt == src (or nearly) the joint sequence looks like a perfectly adequate pair. Copy classes that a policy could drift toward (English leakage) are therefore its Achilles heel; ours and SSA-COMET-QE (separate encoders) are not affected (<= 4.5%). Mixed translation+English (first half translated, second half the English source) is NOT a weakness of the MTL model (0.14 vs ours 0.18); appending the English source to a correct translation is rewarded by every metric (0.17-0.55; ours worst) and is handled only by the length-window / extra-output guard, not by QE.

**What the existing hard guards remove** (Jaccard >= 0.9 copy guard, `guards.is_copy_of_source`, plus GlotLID `wrong_language` exactly as enabled in the v7 calibration: 44 of 45 languages here, min 30 chars, accepted label set of the language), failure rate of rows *not* caught, over all rows:

| metric | copy_source | copy_noisy | mix_half_src | swap25_src |
|---|---|---|---|---|
| ours | 0.000 | 0.004 | 0.160 | 0.016 |
| SSA-COMET-MTL (qe) | 0.000 | **0.017** | 0.115 | 0.021 |
| ens_afri_ssamtl | 0.000 | 0.007 | 0.108 | 0.014 |
| ens_min_afri_ssamtl | 0.000 | 0.004 | 0.134 | 0.018 |
| ens_shaped | 0.000 | 0.006 | 0.116 | 0.014 |

(The Jaccard guard alone catches all `copy_source` rows but only 19% of `copy_noisy` rows; GlotLID raises that to 93%, so the guard that matters is LID, which is only as good as its per-language recall and is off for short sentences and for 1 of 45 languages.) The static test cannot show whether a policy under RL would walk toward the residual; this is the main risk of using SSA-COMET-MTL (qe) alone and the reason for preferring the ensemble. `ens_min` is the most conservative ensemble but it is the lowest of the ensembles on the reference-based proxies and on humans (0.573 SSA-MTE, 0.452 AfriMTE), so I would not pick it.

## 6. Task 4: recommendation, with the logic

Column D uses the z-mean ensemble on the human rows (the shaped variant needs per-language anchors that exist only for the FLORES-based sets) and the shaped ensemble on the policy and corruption rows. Options: A keep AfriCOMET-QE-STL-1.1; B SSA-COMET-QE; C SSA-COMET-MTL (reference-free); D ensemble of A and C.

| criterion | A ours | B SSA-QE | C SSA-MTL (qe) | D ens (shaped) |
|---|---|---|---|---|
| human Spearman SSA-MTE (7 pairs, in-domain for SSA) | 0.470 | 0.550 | 0.604 | 0.584 (z-mean) |
| human Spearman AfriMTE-WMT24 (9 pairs, out of domain) | 0.413 | 0.421 | 0.454 | 0.465 (z-mean) |
| within-source human ranking (SSA-MTE) | 0.439 | 0.503 | 0.556 | 0.539 (z-mean) |
| human-vs-mismatch AUC / hard-class failure | 0.866 / 0.109 | 0.822 / 0.160 | 0.983 / 0.027 | 0.960 / 0.044 |
| within-prompt Spearman with chrF++ on policy samples | 0.145 | 0.120 | 0.164 | 0.172 |
| copy hack after guards (copy_noisy residual) | 0.004 | 0.002 | 0.017 | 0.006 |
| engineering | none | model id + recalibration | code change (UnifiedMetric), joint length budget, recalibration | C's changes x2 models |

* **B is dominated** (worse than A on 3 of 4 behavioural measures and no out-of-domain human gain). This is also why the 2026-10-15 study was right to reject it.
* **A is the weakest on humans in every comparison** and the weakest discriminator (hard-class failure 0.109, 22 of 45 languages have t_fail within 0.05 of t_mid in the 80-sentence anchors, i.e. little reward resolution), so "keep" is not supported by the human-label evidence; its only advantage is that it is the least hackable and already integrated.
* **C is best on every quality measure but has a known exploit.** I would not run it alone.
* **D (average of the two shaped rewards) takes most of C's gain and nearly all of A's robustness.** It matches or beats C on AfriMTE-WMT24 and on policy-sample ranking, gives up ~0.02 on SSA-MTE (in-domain for C), has a hard-class failure rate of 0.044 (A: 0.109) and, with the existing guards, a copy residual of 0.6% (A: 0.4%). The two judges disagree usefully (within-group correlation 0.53) so a disagreement is also a natural alarm: a sample that C likes and A dislikes by a large margin is the signature of the hack (log `|z_C - z_A|`).
* **Expected effect size (be sober):** the QE signal's chrF++ alignment inside a group is ~0.15-0.17 for every option, the metric-selected sample differs from ours by ~+0.1 chrF++ (0.87 -> 0.99, intervals overlap) per group, and the differences between options are 0.02-0.03 in within-prompt Spearman. On human agreement the gain is +0.04..+0.05 Spearman out of domain (+0.11 in domain). I would not expect a visible change in final FLORES+/benchmark scores from this alone; expect a cleaner signal in the 11 blend languages (+0.021 [+0.006, +0.036] within-prompt, AUC +0.14) and lower reliance on guards for wrong-content outputs. Whether it is worth doing depends on whether the RL stages remaining are many; it is cheap enough to do at the next stage boundary.
* **Do not drop the consensus component**: it is by far the strongest independent signal against chrF++ and is the only protection for the 21 languages that have no references and for which nothing here can be measured.

### Recalibration and cost of D (or C)

* **Every anchor changes.** From the 80-sentence human/mismatch sets (sanity: my AfriCOMET anchors differ from v7's n=997 ones by a median 0.019 / 0.016 / 0.012 for t_fail / t_mid / t_great): SSA-COMET-MTL (qe) anchors (median over languages) are t_fail 0.337, t_mid 0.569, t_great 0.675 vs v7 AfriCOMET 0.535 / 0.600 / 0.723; shift of t_fail -0.198, of t_mid -0.032, of t_great -0.036; t_mid moves by more than 0.05 in 49% of languages. The headroom between failure line and great (t_great - t_fail) grows from 0.17 to 0.34, and only 1 of 45 languages has t_fail close to t_mid (22 of 45 for AfriCOMET), i.e. more reward resolution. SSA-COMET-QE would be much more compressed (human P10..P90 0.52-0.71, 39 of 45 languages with t_fail within 0.05 of t_mid).
* **`FAIL_FLOOR` 0.46 must not be applied to SSA-COMET-MTL.** The v7 file already uses `fail_floor: 0.0` (meta), the 0.46 is only the code default; with it, 43 of 45 SSA-MTL t_fail values (min 0.08, median 0.34) would be pinned up to 0.46 and human-quality outputs would earn <= 0. Calibrate with `--fail-floor 0`.
* Procedure: `calibrate_shaping.py --qe-model McGill-NLP/ssa-comet-mtl --fail-floor 0` on the full FLORES+ dev (997 sentences; 80 here give CIs of +-0.03 on a quantile) -> a second calibration file; the ensemble reward is the mean of the two shaped values, so no new combining constants (or z constants from human refs: AfriCOMET (0.5604, 0.141), SSA-MTL-qe (0.5572, 0.1151)). `reliability`/`content.mode` per language must be recomputed with the new metric before touching blend/consensus assignments: the AUC for SSA-MTL (qe) in the 11 blend languages is 0.92-0.99, so some could move to `qe`, but their ranking evidence is weak (no within-prompt gain for the MTL alone) so keep them in blend until measured.
* **Code**: `CometScorer` / `comet_worker.py` do not support `UnifiedMetric` (its `prepare_sample` returns a tuple of encoded inputs and the score comes from `predict_step`; `qe_choice_score.py::unified_call` is a working implementation, verified against the paper's numbers). The input is one joint sequence [mt, src] truncated at **512 tokens in total** (mt first, so a long translation silently truncates the *source*), unlike the 510+510 separate encodings of STL-QE: `COMET_MAX_SEGMENT_TOKENS`, the chunk plan and `token_lengths` must be revised so that src + mt <= ~508 tokens.
* **Cost**: scoring throughput on the shared (busy) GPU was ~300 rows/s per model (28,800 rows in 83-92 s for each of the three QE models), so a 128-sample step costs ~0.4 s per model; two models +~0.4 s/step and ~1.1 GB fp16 each. Negligible against generation. Dev/calibration effort: one calibration run (~minutes of GPU) + code + tests.

### What I did not measure

* No human labels exist for the outputs of our 270M model; all policy-sample evidence rests on chrF++ and reference-COMET proxies (family-biased). Human evidence is from other MT systems.
* The reward dynamics under RL (whether the policy finds the copy/English-mixing direction of SSA-COMET-MTL) are not simulated.
* Languages without references (the 21 consensus-only ones other than kea/vmw) are not covered; SSA-MTE covers 11 and AfriMTE-WMT24 13 pairs (twi, xho near chance for all metrics).
* 80 sentences/language for anchors and corruptions (CIs of per-language numbers are wide; the cross-language averages are what I interpret). Pcm uses MAFAND dev.
* fp16 production path was used for all models; the paper's fp32 numbers are reproduced within 0.01.
* AfriCOMET v1.1-MTL (the paper's baseline) is not on the Hub; I used v1.0 MTL.

## 7. Proposed D-53 text and checklist

D-53 (proposal): keep AfriCOMET-QE-STL-1.1 for the running stage; do not adopt SSA-COMET-QE; adopt the ensemble of per-language-shaped AfriCOMET-QE-STL-1.1 and SSA-COMET-MTL (reference-free) at the next stage boundary, after (1) UnifiedMetric support in `comet.py`/`comet_worker.py` with a joint-length budget and tests, (2) a second calibration with `--fail-floor 0` on full FLORES+ dev, (3) logging of `fail/wrong_language`, copy rates and `|z_C - z_A|` per language during training, (4) a 200-step A/B on a few qe and blend languages comparing chrF++/AfriCOMET-ref on FLORES+ dev against the AfriCOMET-only arm.

## 8. Reproduce

All on the box (env: `PYTHONPATH=pylibs_tf4:pylibs:gemma_mt_2c` for scoring, `pylibs:gemma_mt_2c` for sampling):

```
python gemma_mt/scripts/qe_choice_study.py ssamte-build --data-dir <SSA-MTE clone> --out ssamte_rows.jsonl      # likewise afrimte-build
python gemma_mt/scripts/qe_choice_score.py --rows ssamte_rows.jsonl --out-dir ssamte_scores --spec NAME=MODEL:qe|ref ...
python gemma_mt/scripts/qe_choice_study.py ssamte-report --rows ssamte_rows.jsonl --scores ssamte_scores --out report.md
python gemma_mt/scripts/qe_choice_sample.py --out policy_samples.jsonl --n-prompts 80        # then policy-build, qe_choice_score, policy-report
python gemma_mt/scripts/qe_choice_study.py hack-build --samples policy_samples.jsonl --out hack_rows.jsonl   # then score, anchors --dump, hack-report --lid --anchors
```

Data: box `/workspace/jobs/a5/{ssamte,afrimte,policy_samples.jsonl,*_rows.jsonl,*_scores/,anchors.json}`; models and FLORES cache in `/workspace/jobs/a5/hf` (re-downloadable; deleted at the end of the study).
