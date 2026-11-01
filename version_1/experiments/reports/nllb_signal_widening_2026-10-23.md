# Widening the NLLB-3.3B pseudo-reference reward signal (signal A) to 24 languages (R17, continuing R14, 2026-10-23)

Scope: widen signal A (chrF++ of a chunk against the NLLB-200-3.3B translation of its source; `gemma_mt/rl/nllb_signal.py`, flag-gated, reward-only, CC-BY-NC) from `kea kbp nus bam kab` to every language where the 3.3B teacher is clearly better than our model, for a NEXT RL round. The live run (`/workspace/jobs/r7/runs/final2`, `/workspace/jobs/r7/gemma_mt_run2`) was not touched. Nothing pushed, no data copied to the laptop (tables, JSON calibration and code only). Box work: `/workspace/jobs/r14`; scripts `gemma_mt/scripts/r14/w_*.py`.

## 0. Answers

1. **Precompute finished for 25 languages** (24 calibrated + `kon`): `/workspace/jobs/r14/pseudorefs_v3/<iso>.parquet` + `DONE_<iso>` + `PRECOMPUTE_V3_DONE`, 154 MB total (`reward_only_nc`, beam 4, no-repeat-4-gram, `rl/segments.split_sentences`, same rejection rules, key `sha1(iso+NFC(source))[:16]`). Windows of `/workspace/jobs/r7/assets2/rl_prompts_sci_v2`: all b64/b256/b512 windows, 25 % of b1024 and 15 % of b2048 (deterministic hash order), merged with the old store (`pseudorefs_reward_only_nc_v1`) for the 11 languages it covered. **Coverage:** b64-b512 windows ~96-100 % covered in every language; b1024 / b2048 14-30 % (kea, nus, kbp, kab, kon, bam 37-69 %). Expected share of chunks that find a pseudo-reference when the bins are drawn uniformly: **0.67-0.84 (mean 0.72)**; chunks without one silently keep the current reward (`nllb/coverage_<iso>` is logged). Covering the long bins fully would cost ~2.5 GPU h per language, not done.
2. **Per-language evidence (section 2): signal A is better aligned with human chrF++ than the current reward in all 24 languages** (Spearman on policy samples: A minus current reward +0.11 to +0.42; the 95 % CI is above 0 in 21 of 24; kbp, tso, sna intervals include 0). Within a GRPO group A is better in every language (0.36-0.62 vs 0.06-0.41). A adds a lot beyond the current reward (partial Spearman +0.35 to +0.69, CI above 0 everywhere) and beyond QE-ensemble plus consensus (0.21-0.65).
3. **Big caveat, new in this round: chrF++ against one reference favours an overlap signal.** Against **human direct assessment (SSA-MTE, systems other than NLLB; amh kik kin luo + twi hau yor)** A alone is **worse** than SSA-COMET-MTL in all 7 languages (difference CI entirely below 0) and not better than the QE ensemble in amh/kin/yor/hau (kik, luo, twi roughly equal). But **mixing** A into the QE ensemble does not hurt and mostly helps DA alignment: with w = 0.25 the Spearman with human DA rises by +0.03 to +0.05 in kik luo twi yor (CIs above 0), +0.014 amh, +0.004 kin (CI includes 0); at w = 0.35 amh and kin are flat (CI includes 0) and at w = 0.5 amh/kin/hau/yor go down. Hence **w <= 0.35 everywhere and 0.25 for amh and kin**. No human DA data exist for the other 17 languages, so for them the evidence is chrF++-based only.
4. **Known weak points:** (i) padding: human + appended unrelated sentence still scores near the human translation (AUC human vs padded 0.46-0.65 for A, vs 0.55-0.82 for the QE ensemble (kea has none); A is worse in all 23 languages with QE), (ii) exact copies are also separated slightly less well than by QE (A better in only 2 of 23 languages, AUC 0.90-1.00), drop-half in 17 of 23 (A 0.83-0.97; below 0.90 in amh bam kab kbp kik ssw tir tum), partial copy in 18 of 23. Keep the current length/ratio guards and consensus on; do not set w above 0.35. (iii) the signal pulls the student toward NLLB's style and into "double pull" where the SFT mixture C3 already contains NLLB output (section 3).
5. **Cost:** CPU only, ~0.53 ms per chunk, ~2.3 s per step for 4,300 chunks (42 prompts x 16 completions, one core); store load 2 s one time for 24 languages (parquet 154 MB). No GPU, forced decoding (B) stays off.

## 1. Calibration and weights

File: `reports/data/nllb_reward_calibration_v2.json` (also `/workspace/jobs/r14/s3/widening/nllb_reward_calibration_v2_final.json`); per language `nllb.pseudoref` (anchors, AUC, evidence fields, gates), `weight` (no NLLB data in the SFT mixture), `weight_C3` (C3 mixture), `teacher_share_C3`, `double_pull_factor_C3`, `da_cap`. Rules: `use` = teacher - student >= +2 with CI above 0, dev AUC >= 0.80, hard-class hack failure <= 0.25, within-group Spearman >= 0.20 (all 24 pass); `w = 0.35 x reliability x clip(delta/6, 0.4, 1) x clip((0.25 - failure)/0.15, 0, 1) x alignment x partial`, alignment = 1.0 if the CI of (A - current) is above 0, 0.7 if only the point estimate is, 0.4 if only the upper bound is (kbp, tso, sna: 0.7); capped at 0.35 (0.25 for amh and kin by the human-DA result). **Double pull:** factor = clip(1 - 0.7 x share of NLLB teacher tokens of the language in the C3 SFT mixture, 0.4, 1) (reports/sft_mixture_v2_2026-10-22/C3/c3_vs_a2w.md: kea/nus/kbp/bam/kon 69-76 %, tir 60 %, kab 53 %, ewe/bem/tum/run/orm/sag/kik/luo 41-47 %, kin/mlg/lin/xho/sna/tsn/tso/ssw 21-27 %, lug/zul 4-5 %, amh 18 %). This is a heuristic, not an estimated quantity. The next RL round starts from which SFT checkpoint decides the column: A2W-type (no NLLB data) use `weight`, C3 use `weight_C3`.

Not in the set (teacher not clearly better or gates fail): som hau lua sag nya nso swh mos sot twi afr ibo yor fon kmb wol kam umb dyu. `kon` has a pseudo-reference store (variety question open, no FLORES/policy study data): not calibrated, weight 0 until the owner decides.

### Weights (reward-only NLLB-3.3B, `0` = never)

| iso | weight (SFT mixture without NLLB teacher data) | weight with the double-pull rule (teacher data in the mixture) | teacher data in teacher_v1 | alignment factor | factor |
|---|---|---|---|---|---|
| `bam` | 0.35 | 0.18 | yes | 1.0 | 1.00 |
| `ewe` | 0.35 | 0.235 | yes | 1.0 | 1.00 |
| `kea` | 0.35 | 0.169 | yes | 1.0 | 1.00 |
| `mlg` | 0.35 | 0.285 | yes | 1.0 | 1.00 |
| `run` | 0.35 | 0.245 | yes | 1.0 | 1.00 |
| `tir` | 0.35 | 0.203 | yes | 1.0 | 1.00 |
| `zul` | 0.339 | 0.328 | yes | 1.0 | 0.97 |
| `lin` | 0.334 | 0.28 | yes | 1.0 | 0.95 |
| `ssw` | 0.324 | 0.264 | yes | 1.0 | 0.93 |
| `luo` | 0.32 | 0.229 | yes | 1.0 | 0.91 |
| `kab` | 0.301 | 0.19 | yes | 1.0 | 0.86 |
| `kik` | 0.3 | 0.212 | yes | 1.0 | 0.86 |
| `amh` | 0.25 | 0.218 | yes | 1.0 | 0.71 |
| `kin` | 0.25 | 0.206 | yes | 1.0 | 0.71 |
| `orm` | 0.247 | 0.175 | yes | 1.0 | 0.71 |
| `kbp` | 0.245 | 0.12 | yes | 0.7 | 0.70 |
| `nus` | 0.227 | 0.106 | yes | 1.0 | 0.65 |
| `xho` | 0.208 | 0.176 | yes | 1.0 | 0.59 |
| `lug` | 0.196 | 0.19 | yes | 1.0 | 0.56 |
| `bem` | 0.187 | 0.127 | yes | 1.0 | 0.53 |
| `tso` | 0.162 | 0.135 | yes | 0.7 | 0.46 |
| `tsn` | 0.143 | 0.122 | yes | 1.0 | 0.41 |
| `tum` | 0.14 | 0.094 | yes | 1.0 | 0.40 |
| `sna` | 0.128 | 0.105 | yes | 0.7 | 0.37 |

## 2. Validation per language

### 2.1 Policy samples (80 prompts x 8 samples, RL model; target = human chrF++ against the FLORES reference; 95 % CIs from a cluster bootstrap over prompts, 300 resamples)
"Current reward" = the content reward the policy gets today: QE ensemble in `qe` languages, 0.5 QE + 0.5 consensus value in `blend`, consensus value in `consensus` (kea). Spearman pooled over all valid samples; within-group = mean over prompts. Partial Spearman = rank-residual correlation of A with human chrF++ controlling for the current reward / for QE ensemble + consensus.

| iso | group / reward mode | teacher - student | Spearman A [95% CI] | Spearman current reward [95% CI] | A - current [95% CI] | within-group A / current | partial A given current [95% CI] | partial A given QE-ens + consensus | use | weight |
|---|---|---|---|---|---|---|---|---|---|---|
| `kea` | consensus_only / consensus | +21.1 | 0.80 [0.73, 0.85] | 0.65 [0.53, 0.75] | +0.15 [+0.07, +0.25] | 0.62 / 0.41 | +0.65 [+0.55, +0.73] | 0.65 | yes | 0.35 |
| `amh` | qe_validated / qe | +18.9 | 0.67 [0.58, 0.75] | 0.31 [0.17, 0.44] | +0.36 [+0.23, +0.50] | 0.51 / 0.17 | +0.64 [+0.54, +0.72] | 0.40 | yes | 0.25 |
| `tir` | qe_validated / qe | +11.0 | 0.69 [0.60, 0.79] | 0.39 [0.30, 0.50] | +0.31 [+0.19, +0.41] | 0.57 / 0.27 | +0.63 [+0.53, +0.72] | 0.44 | yes | 0.35 |
| `nus` | qe_validated / blend | +7.9 | 0.42 [0.26, 0.55] | 0.25 [0.11, 0.39] | +0.17 [+0.03, +0.30] | 0.46 / 0.15 | +0.35 [+0.18, +0.48] | 0.29 | yes | 0.227 |
| `run` | qe_validated / qe | +7.7 | 0.62 [0.51, 0.71] | 0.29 [0.14, 0.42] | +0.33 [+0.15, +0.50] | 0.50 / 0.22 | +0.59 [+0.49, +0.68] | 0.37 | yes | 0.35 |
| `ewe` | qe_validated / qe | +7.2 | 0.72 [0.62, 0.82] | 0.30 [0.18, 0.43] | +0.42 [+0.28, +0.55] | 0.51 / 0.06 | +0.69 [+0.58, +0.80] | 0.51 | yes | 0.35 |
| `kbp` | qe_validated / blend | +7.0 | 0.62 [0.50, 0.71] | 0.50 [0.36, 0.62] | +0.11 [-0.03, +0.25] | 0.45 / 0.31 | +0.49 [+0.37, +0.59] | 0.29 | yes | 0.245 |
| `mlg` | qe_validated / qe | +7.0 | 0.61 [0.45, 0.72] | 0.46 [0.32, 0.58] | +0.14 [+0.02, +0.28] | 0.41 / 0.29 | +0.50 [+0.34, +0.65] | 0.29 | yes | 0.35 |
| `kin` | qe_validated / qe | +6.2 | 0.68 [0.57, 0.75] | 0.33 [0.17, 0.48] | +0.35 [+0.19, +0.51] | 0.44 / 0.24 | +0.65 [+0.54, +0.73] | 0.47 | yes | 0.25 |
| `bam` | qe_validated / blend | +6.0 | 0.64 [0.49, 0.75] | 0.47 [0.33, 0.61] | +0.16 [+0.06, +0.29] | 0.40 / 0.20 | +0.51 [+0.37, +0.65] | 0.38 | yes | 0.35 |
| `zul` | qe_validated / qe | +5.8 | 0.62 [0.51, 0.71] | 0.34 [0.22, 0.48] | +0.27 [+0.10, +0.43] | 0.50 / 0.17 | +0.59 [+0.49, +0.69] | 0.24 | yes | 0.339 |
| `lin` | qe_validated / qe | +5.7 | 0.65 [0.54, 0.74] | 0.38 [0.21, 0.52] | +0.26 [+0.11, +0.44] | 0.50 / 0.21 | +0.58 [+0.46, +0.71] | 0.35 | yes | 0.334 |
| `ssw` | qe_validated / qe | +5.6 | 0.70 [0.59, 0.79] | 0.37 [0.22, 0.51] | +0.34 [+0.19, +0.49] | 0.45 / 0.27 | +0.65 [+0.52, +0.75] | 0.45 | yes | 0.324 |
| `luo` | qe_validated / qe | +5.5 | 0.77 [0.69, 0.84] | 0.50 [0.38, 0.62] | +0.27 [+0.17, +0.38] | 0.53 / 0.23 | +0.68 [+0.60, +0.76] | 0.37 | yes | 0.32 |
| `kab` | qe_validated / blend | +5.2 | 0.58 [0.47, 0.69] | 0.40 [0.26, 0.51] | +0.19 [+0.08, +0.31] | 0.49 / 0.21 | +0.47 [+0.34, +0.60] | 0.37 | yes | 0.301 |
| `kik` | qe_validated / qe | +5.2 | 0.62 [0.50, 0.74] | 0.36 [0.22, 0.49] | +0.27 [+0.14, +0.40] | 0.40 / 0.20 | +0.56 [+0.42, +0.67] | 0.24 | yes | 0.3 |
| `orm` | qe_validated / qe | +4.2 | 0.55 [0.41, 0.64] | 0.34 [0.22, 0.46] | +0.21 [+0.09, +0.33] | 0.45 / 0.16 | +0.49 [+0.36, +0.57] | 0.23 | yes | 0.247 |
| `tso` | qe_validated / qe | +4.0 | 0.58 [0.44, 0.72] | 0.45 [0.31, 0.57] | +0.13 [-0.06, +0.29] | 0.46 / 0.15 | +0.50 [+0.32, +0.66] | 0.31 | yes | 0.162 |
| `xho` | qe_validated / qe | +3.6 | 0.53 [0.40, 0.66] | 0.24 [0.09, 0.36] | +0.30 [+0.15, +0.46] | 0.42 / 0.15 | +0.51 [+0.39, +0.64] | 0.22 | yes | 0.208 |
| `lug` | qe_validated / qe | +3.4 | 0.54 [0.39, 0.66] | 0.24 [0.08, 0.37] | +0.31 [+0.14, +0.44] | 0.42 / 0.16 | +0.51 [+0.35, +0.63] | 0.21 | yes | 0.196 |
| `bem` | qe_validated / qe | +3.2 | 0.74 [0.65, 0.80] | 0.43 [0.31, 0.55] | +0.31 [+0.19, +0.43] | 0.44 / 0.16 | +0.67 [+0.58, +0.76] | 0.41 | yes | 0.187 |
| `sna` | qe_validated / qe | +3.1 | 0.51 [0.34, 0.64] | 0.30 [0.13, 0.45] | +0.20 [-0.02, +0.41] | 0.40 / 0.11 | +0.47 [+0.28, +0.61] | 0.28 | yes | 0.128 |
| `tsn` | qe_validated / qe | +2.5 | 0.54 [0.38, 0.68] | 0.33 [0.16, 0.45] | +0.21 [+0.02, +0.39] | 0.36 / 0.09 | +0.50 [+0.34, +0.64] | 0.38 | yes | 0.143 |
| `tum` | qe_validated / qe | +2.0 | 0.61 [0.50, 0.73] | 0.36 [0.24, 0.47] | +0.24 [+0.09, +0.40] | 0.39 / 0.22 | +0.54 [+0.41, +0.68] | 0.37 | yes | 0.14 |

The other 19 covered languages not in the table (som hau lua sag nya nso swh mos sot twi afr ibo yor fon kmb wol kam umb dyu) are in `reports/data/nllb_widening_policy_per_language.csv`; there A also beats the current reward in 12 of 19 with the CI above 0; not significant (7): dyu -0.09, sag -0.02, fon +0.03, yor +0.03, kam +0.10, kmb +0.09, umb +0.06, which is why the gate "teacher clearly better" is the binding one, not alignment.

### 2.2 Corruption study (45 languages x 80 FLORES-dev sentences x 12 classes; AUC human > corrupted, A / current QE ensemble; kea has no QE ensemble, its QE columns are 0 = not available)
| iso | AUC mismatch A / QE-ens | AUC copy A / QE-ens | AUC padded (human + unrelated sentence) A / QE-ens | AUC drop-half A / QE-ens | AUC partial copy A / QE-ens |
|---|---|---|---|---|---|
| `kea` | 1.000 [1.00, 1.00] / 0.000 | 0.984 [0.97, 1.00] / 0.000 | 0.647 [0.62, 0.68] / 0.000 | 0.955 [0.93, 0.98] / 0.000 | 0.738 [0.71, 0.77] / 0.000 |
| `amh` | 0.985 [0.96, 1.00] / 0.997 | 0.997 [0.99, 1.00] / 1.000 | 0.596 [0.57, 0.63] / 0.757 | 0.867 [0.83, 0.91] / 0.995 | 0.753 [0.72, 0.80] / 0.910 |
| `tir` | 0.991 [0.98, 1.00] / 0.984 | 0.979 [0.94, 1.00] / 0.999 | 0.580 [0.56, 0.61] / 0.685 | 0.848 [0.81, 0.89] / 0.980 | 0.737 [0.70, 0.78] / 0.843 |
| `nus` | 0.951 [0.92, 0.98] / 0.863 | 0.900 [0.85, 0.95] / 0.908 | 0.462 [0.42, 0.50] / 0.581 | 0.900 [0.86, 0.94] / 0.866 | 0.607 [0.56, 0.65] / 0.555 |
| `run` | 0.994 [0.98, 1.00] / 0.966 | 0.985 [0.97, 1.00] / 0.994 | 0.618 [0.59, 0.66] / 0.765 | 0.931 [0.90, 0.97] / 0.976 | 0.738 [0.71, 0.78] / 0.824 |
| `ewe` | 0.985 [0.97, 1.00] / 0.915 | 0.960 [0.93, 0.98] / 0.983 | 0.544 [0.52, 0.58] / 0.688 | 0.909 [0.88, 0.94] / 0.948 | 0.681 [0.65, 0.72] / 0.701 |
| `kbp` | 0.941 [0.89, 0.97] / 0.858 | 0.961 [0.94, 0.98] / 0.975 | 0.556 [0.53, 0.59] / 0.598 | 0.830 [0.79, 0.87] / 0.856 | 0.655 [0.63, 0.69] / 0.576 |
| `mlg` | 0.999 [1.00, 1.00] / 0.993 | 0.994 [0.99, 1.00] / 0.999 | 0.637 [0.60, 0.68] / 0.824 | 0.962 [0.93, 0.98] / 0.982 | 0.768 [0.74, 0.80] / 0.920 |
| `kin` | 1.000 [1.00, 1.00] / 0.982 | 0.996 [0.99, 1.00] / 0.998 | 0.641 [0.61, 0.69] / 0.806 | 0.967 [0.94, 0.99] / 0.982 | 0.750 [0.72, 0.79] / 0.858 |
| `bam` | 0.975 [0.95, 0.99] / 0.921 | 0.925 [0.87, 0.96] / 0.965 | 0.563 [0.52, 0.60] / 0.615 | 0.898 [0.86, 0.93] / 0.894 | 0.655 [0.63, 0.69] / 0.639 |
| `zul` | 0.999 [1.00, 1.00] / 0.983 | 0.992 [0.98, 1.00] / 0.989 | 0.652 [0.63, 0.69] / 0.755 | 0.939 [0.91, 0.97] / 0.953 | 0.769 [0.74, 0.81] / 0.819 |
| `lin` | 0.994 [0.99, 1.00] / 0.961 | 0.988 [0.98, 1.00] / 0.976 | 0.618 [0.58, 0.67] / 0.740 | 0.953 [0.92, 0.98] / 0.947 | 0.765 [0.73, 0.80] / 0.769 |
| `ssw` | 0.991 [0.98, 1.00] / 0.966 | 0.972 [0.95, 0.99] / 0.989 | 0.584 [0.55, 0.61] / 0.685 | 0.886 [0.85, 0.93] / 0.947 | 0.699 [0.67, 0.73] / 0.788 |
| `luo` | 0.994 [0.98, 1.00] / 0.940 | 0.967 [0.94, 0.98] / 0.985 | 0.579 [0.55, 0.61] / 0.629 | 0.943 [0.92, 0.97] / 0.923 | 0.694 [0.68, 0.73] / 0.754 |
| `kab` | 0.985 [0.96, 1.00] / 0.880 | 0.929 [0.89, 0.97] / 0.953 | 0.537 [0.50, 0.57] / 0.547 | 0.891 [0.85, 0.93] / 0.896 | 0.636 [0.60, 0.67] / 0.572 |
| `kik` | 0.974 [0.95, 0.99] / 0.940 | 0.963 [0.94, 0.99] / 0.978 | 0.582 [0.55, 0.63] / 0.643 | 0.883 [0.84, 0.93] / 0.962 | 0.698 [0.67, 0.74] / 0.666 |
| `orm` | 0.998 [0.99, 1.00] / 0.958 | 0.978 [0.95, 1.00] / 0.993 | 0.561 [0.53, 0.60] / 0.577 | 0.947 [0.92, 0.98] / 0.883 | 0.703 [0.67, 0.74] / 0.734 |
| `tso` | 1.000 [1.00, 1.00] / 0.963 | 0.989 [0.98, 1.00] / 0.995 | 0.607 [0.58, 0.65] / 0.711 | 0.938 [0.91, 0.97] / 0.968 | 0.717 [0.69, 0.76] / 0.795 |
| `xho` | 0.998 [0.99, 1.00] / 0.984 | 0.986 [0.97, 1.00] / 0.994 | 0.611 [0.59, 0.65] / 0.765 | 0.948 [0.92, 0.98] / 0.955 | 0.740 [0.71, 0.77] / 0.799 |
| `lug` | 0.990 [0.98, 1.00] / 0.912 | 0.951 [0.91, 0.98] / 0.963 | 0.568 [0.54, 0.60] / 0.656 | 0.920 [0.88, 0.95] / 0.913 | 0.696 [0.65, 0.74] / 0.703 |
| `bem` | 0.991 [0.98, 1.00] / 0.923 | 0.959 [0.92, 0.99] / 0.975 | 0.577 [0.55, 0.61] / 0.683 | 0.904 [0.87, 0.94] / 0.960 | 0.669 [0.64, 0.71] / 0.760 |
| `sna` | 0.996 [0.99, 1.00] / 0.967 | 0.974 [0.95, 0.99] / 0.991 | 0.612 [0.58, 0.66] / 0.705 | 0.925 [0.90, 0.96] / 0.949 | 0.721 [0.70, 0.76] / 0.793 |
| `tsn` | 0.989 [0.96, 1.00] / 0.969 | 0.986 [0.97, 1.00] / 0.990 | 0.595 [0.56, 0.64] / 0.749 | 0.945 [0.91, 0.97] / 0.970 | 0.719 [0.70, 0.75] / 0.806 |
| `tum` | 0.991 [0.98, 1.00] / 0.945 | 0.972 [0.95, 0.99] / 0.995 | 0.610 [0.59, 0.64] / 0.701 | 0.897 [0.86, 0.94] / 0.979 | 0.690 [0.67, 0.72] / 0.794 |
Reading: mismatched translations are separated better by A than by QE in 22 of 23 languages with QE (A 0.94-1.00); exact copies are separated slightly worse (A better in 2 of 23, A 0.90-1.00); **padding is the weak class for A** (0.46-0.65) and also for QE (0.55-0.82).

### 2.3 Human direct assessment (SSA-MTE, other MT systems, NLLB's own output excluded; Spearman with z-score, bootstrap over source sentences)
| language | judgements / sources | Spearman A [95% CI] | AfriCOMET-QE | SSA-COMET-MTL | QE ensemble | A - SSA-COMET [95% CI] | within-source A / AfriCOMET / SSA |
|---|---|---|---|---|---|---|---|
| `amh` | 970 / 211 | 0.38 [0.32, 0.44] | 0.55 | 0.60 | 0.59 | [-0.28, -0.16] | 0.37 / 0.49 / 0.54 |
| `hau` | 996 / 208 | 0.20 [0.12, 0.27] | 0.36 | 0.44 | 0.41 | [-0.32, -0.16] | 0.09 / 0.27 / 0.36 |
| `kik` | 935 / 245 | 0.52 [0.47, 0.57] | 0.47 | 0.65 | 0.53 | [-0.19, -0.09] | 0.43 / 0.45 / 0.65 |
| `kin` | 1013 / 207 | 0.37 [0.31, 0.42] | 0.41 | 0.58 | 0.57 | [-0.28, -0.15] | 0.41 / 0.49 / 0.55 |
| `luo` | 997 / 208 | 0.59 [0.56, 0.63] | 0.59 | 0.75 | 0.68 | [-0.19, -0.11] | 0.61 / 0.63 / 0.74 |
| `twi` | 1001 / 204 | 0.57 [0.52, 0.62] | 0.48 | 0.69 | 0.60 | [-0.17, -0.07] | 0.54 / 0.50 / 0.64 |
| `yor` | 1008 / 208 | 0.36 [0.29, 0.43] | 0.39 | 0.53 | 0.45 | [-0.24, -0.10] | 0.37 / 0.34 / 0.52 |
Blend with the current QE ensemble, change in pooled Spearman (95 % CI) versus the QE ensemble alone (`nllb_widening_blend_da.csv`):

| language | QE ensemble alone | w=0.15 | w=0.25 | w=0.35 | w=0.5 |
|---|---|---|---|---|---|
| amh | 0.587 | +0.012 [+0.004, +0.020] | +0.014 [+0.002, +0.030] | +0.011 [-0.009, +0.032] | -0.011 [-0.042, +0.021] |
| kin | 0.566 | +0.004 [-0.004, +0.012] | +0.004 [-0.009, +0.017] | -0.000 [-0.019, +0.017] | -0.015 [-0.043, +0.010] |
| kik | 0.529 | +0.035 [+0.025, +0.046] | +0.053 [+0.036, +0.070] | +0.066 [+0.043, +0.090] | +0.075 [+0.043, +0.107] |
| luo | 0.684 | +0.022 [+0.017, +0.028] | +0.034 [+0.024, +0.044] | +0.041 [+0.027, +0.056] | +0.043 [+0.019, +0.066] |
| twi | 0.599 | +0.022 [+0.014, +0.029] | +0.033 [+0.021, +0.046] | +0.042 [+0.026, +0.060] | +0.051 [+0.026, +0.077] |
| yor | 0.453 | +0.026 [+0.011, +0.039] | +0.034 [+0.012, +0.055] | +0.037 [+0.005, +0.065] | +0.029 [-0.016, +0.070] |
| hau | 0.415 | +0.016 [+0.005, +0.029] | +0.020 [+0.000, +0.042] | +0.015 [-0.013, +0.049] | -0.013 [-0.057, +0.037] |

## 3. Config block (next RL round; all keys exist in `NllbSignalConfig`, apply the R14 hook patch first)
```yaml
reward:
  nllb_signal: true
  nllb_pseudoref_dir: /workspace/jobs/r14/pseudorefs_v3            # copy beside the run; parquet only, 154 MB, reward_only_nc
  nllb_calibration_path: reports/data/nllb_reward_calibration_v2.json
  nllb_modes: [consensus, blend]
  nllb_languages: [amh, bam, bem, ewe, kab, kbp, kea, kik, kin, lin, lug, luo, mlg, nus, orm, run, sna, ssw, tir, tsn, tso, tum, xho, zul]
  nllb_forced: false
  nllb_weight_default: 0.35
  nllb_weight_max: 0.35
  nllb_min_teacher_delta: 2.0
  nllb_min_auc: 0.80
  # (a) start checkpoint WITHOUT NLLB data in its SFT mixture: no overrides, weights come from the calibration (mean 0.27).
  # (b) start from the C3 mixture (NLLB teacher data): explicit weights = calibration `weight_C3` (mean 0.19; leave nllb_double_pull_languages empty, it would halve a second time)
  nllb_weights: {amh: 0.218, bam: 0.18, bem: 0.127, ewe: 0.235, kab: 0.19, kbp: 0.12, kea: 0.169, kik: 0.212, kin: 0.206, lin: 0.28, lug: 0.19, luo: 0.229, mlg: 0.285, nus: 0.106, orm: 0.175, run: 0.245, sna: 0.105, ssw: 0.264, tir: 0.203, tsn: 0.122, tso: 0.135, tum: 0.094, xho: 0.176, zul: 0.328}
```
Expected cost per step: ~2.3 s CPU (0.53 ms x 4,300 chunks), +2 s one-off store load, 0 GPU, ~160 MB RAM. Log `nllb/applied_rate` and `nllb/coverage_<iso>` (expect 0.55-0.85 per language; lower in long-window bins). Cross-check each language's A-weighted reward against human chrF++ on the held-out FLORES-dev evaluation every N steps: A is trained toward, so a rise of A-reward with flat human chrF++ means the student is converging to NLLB's style, not improving.

## 4. Uncertainty and limits
* Per language the policy/hack sets have 640 / 80 samples; single-language numbers are noisy (CIs given), aggregates are the reliable part. The human-chrF++ target favours overlap signals (A, consensus) by construction; only 7 languages have human DA (4 of them in the set).
* The recomputed point "A is better than the current reward" for languages in `qe` mode compares with the QE ensemble only, but the live run can use consensus-blend; the partial correlation given QE ensemble + consensus (0.21-0.65, section 2.1) is the conservative comparison.
* Pseudo-references exist for ~72 % of the chunks drawn uniformly over bins, ~100 % for windows up to 512 tokens; the rest silently use the old reward.
* `kon`: store built (4,505 windows) but not calibrated; `kea` has no QE ensemble, so there A competes with the consensus only.
* The CT2 model dir and scratch on the box were deleted at the end; the parquet stores, teacher_v1 and all earlier outputs are kept.
