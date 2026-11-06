# E-14: science RL without the science SFT (2026-11-05)

Executor: the E-02/E-04/E-11/E-14 agent. Scripts: `data_pipeline/sft_clean/fresh/overnight/e2405/` (`e04_cfg.py sci`, `e14_run.sh`, `e14_eval.sh`, `e14_neural.py`, `e14_pairs.py`). Section 0 is the pre-registration, written before the run exists and not edited afterwards.

## 0. Pre-registration (written 2026-11-05 05:40 UTC)

**Question (SC3).** The science SFT (alpha 0.5 interpolation, 28,750 AfriScience-MT train windows + replay) gave +1.23 chrF on the AfriScience-MT test sentences and no gain in SSA-COMET-QE; the science RL that followed added +0.46 chrF. Is the science SFT needed, or does the same science RL started directly from the general RL model reach the same place?

**Run.** The executed science RL (`runs_385m_sci/final/config.json`, reproduced by `e04_cfg.py sci`; diff against the executed config printed by `e_cfgcheck.py`: only the start model, 160 steps without extension, no early stops, planned winddown of the last 10 steps, hub off) with **the start model = mixed best** (Hub `<HF_ORG>/gemma-0.4B-translate-rl` branch `main`, sha256 8cb0dd5af976...) = science-SFT alpha 0, which is the model the science SFT was built from. Same seed 1234 (so the prompt schedule is the same as in the executed run), same pool `prompts/sci_rl` (clean_sci_v2 without LibreTexts + AfriScience train reference rows), same reward (QE ensemble + reference-based SSA-COMET-MTL term + chrF++ against human references weight 0.35 + recall/precision/length/preservation terms + science anchor + NLLB-3.3B pseudo-reference signal for 5 languages), Muon lr 1e-6, warm-up 20, KL coefficient 0.1 against the START model (here mixed best), KL guard 0.25 with recovery, beam/MBR guide, group 16, 42 prompts per step, b64 + b256 bins. **160 steps** = the executed run's step count (150 steps, then a 10-step winddown triggered by the dev plateau; reproduced as 150 steps + planned winddown of 10). The executed run's `best/` was step 75 and the final step 160. Data disclosure: as the executed science RL (`paper_385m/DISCLOSURES.md` 1.2): AfriScience-MT train (Apache-2.0), PMC/arXiv science text, FineWeb-Edu/Wikipedia, human reference pairs of AfriScience-MT/IbomMT, NLLB-3.3B pseudo-references as a reward signal; no TranslatePsy, no NLLB-mined training text.

**Pre-declared checkpoint: final (step 160)**, as for the executed science RL (final was better than best there, and the long-document RL started from final). `best/` (selected by the trainer's dev chrF++) is evaluated at sentence level only if it differs from final and is reported separately.

**Evaluation (the E-16 evaluation, fixed now).** (1) AfriScience-MT dev and test, sentence level, macro over 6 languages (amh hau lug nso yor zul): chrF (word order 0, the evaluation's definition), chrF++, BLEU, finished rate, length ratio, held-out AfriCOMET-STL-1.1, SSA-COMET-QE and SSA-COMET-MTL(ref) (reward family: always shown next to chrF and AfriCOMET and labelled). (2) NTREX (24) and MAFAND (15) chrF++, and the full benchmark harness FLORES+ (54) / NTREX / MAFAND with held-out AfriCOMET (needed for the comparison with B1 final and mixed best on identical sentences). (3) 168 test documents, whole-document and chunked modes, as E-16 (window SSA-COMET-QE, chrF, finished, loop, tail coverage). vLLM 0.32.0 bf16 greedy, same prompts, same max tokens as E-16 (sentences 768, documents 4,096).

**Comparisons (all on identical sentences; paired bootstrap, Poisson weights, 1,000 resamples, seed 20261104; AfriScience: independent weights per language as in `sci_pairs.py`):**
1. E-14 final vs the **science RL final** (started from the science SFT) = the answer to "is the science SFT needed" (primary).
2. E-14 final vs **mixed best** (its own start; = alpha 0), vs **science SFT alpha 0.5** (the executed run's start), vs **B1 final** (Hub `stage-b1-final`, the b256 RL; evaluated on AfriScience for the first time here; benchmark cells from E-20). *The owner's phrase "the b256 rl" is ambiguous; B1 (b256) is the stage-B1 RL, mixed best is the b64+b256 RL, and the executed science RL is the other candidate, so all three comparisons are reported and none is picked afterwards.*
3. Context only: science RL final vs mixed best, science RL final vs B1 final (these exist already; recomputed on the same sentences for the table).

**Decision rule for (1), AfriScience-MT test, macro over 6 languages (fixed now).** Non-inferiority margins: -0.5 chrF and -0.004 held-out AfriCOMET (the margin of the dev early stop). *Science SFT not needed*: the lower 95% bounds of E-14 minus science RL final are above -0.5 chrF AND above -0.004 AfriCOMET. *Science SFT needed*: the upper bound of the chrF difference is below -0.5 or that of AfriCOMET below -0.004 (either metric alone is enough; both are shown). Otherwise *inconclusive*. The SSA-COMET-QE difference is shown but does not enter the rule (reward family). Documents: descriptive (28 papers per language), no rule. NTREX/MAFAND: non-regression versus mixed best = lower bound of the chrF++ difference above -0.5.

**Noise.** One seed. The E-04 seed spread (RL gain at step 100 of the b64 stage) is the only seed information; the science RL's own effect over its start was +0.46 chrF [+0.27, +0.65] on test, i.e. of the size at which seed noise matters. Differences under about 0.5 chrF are described as "not separable from seed noise" even when their sentence-level interval excludes 0.

**Cost note.** The executed run took 17,153 s (4.8 h) for 160 steps; the evaluation about 1 h.

**Not done:** best/final of a second seed; science RL from a different start (SFT only, no RL); long-document RL from this model.

## 1. Execution log and results

**Run (A100, nothing else on the GPU).** Start 2026-11-05 15:18 UTC (first step 15:21), end 20:17 UTC: 160 steps, `stop_reason planned_winddown` (flat to step 150, winddown to step 160 as executed), 14,395 s in the last process. **One crash, diagnosed and restarted:** at about step 38 (16:15 UTC) the COMET reward worker hit CUDA out-of-memory (GPU 79.25 GiB, trainer process 47 GiB + vLLM worker + COMET worker); the launcher's OOM ladder lowered the micro-batch from 12,288 to 6,144 tokens (same 42 prompts x 16 samples per step, same gradient up to summation order) and restarted from the step-20 checkpoint; steps 21-38 were replayed (the prompt schedule is a function of seed and step, the RNG state is restored), so `metrics.jsonl` holds two records for steps 21-38 (use the later one). No restart afterwards; KL guard 0 cuts, median KL 0.026-0.033 throughout; step 25 evaluation and checkpoint save confirmed at 16:01. Config `cfg_e14.yaml` (differences against the executed `runs_385m_sci/final/config.json`: model_name, steps 160, no extension/early stops/dev stop, `planned_winddown_last` 10, hub off, archive off); code state hashes in `runs_e14/code_state.txt`. Archive: `research_archive_2026-11-03/box_final_2026-11-04/e14/` (sha256 checked), tables in `reports/data/e14_science_rl_no_sft_2026-11-05/`.

**Checkpoints.** The trainer wrote no `best/` (it saves a best only when the dev chrF++ rises by 0.2 above the previous best; the dev macro stayed within 43.19-43.50 for the whole run), so **only `final` (step 160) exists**; "best" and "final" are the same question here. Final was pushed after the evaluation to the public `<HF_ORG>/gemma-0.4B-translate-rl-experiments`, branch `e14-science-rl-only-final` (model.safetensors sha256 2435076a6958..., verified against the box copy); box copy deleted.

**Data used:** as the executed science RL (section 0): AfriScience-MT train (Apache-2.0) windows and reference pairs, PMC / arXiv science text, FineWeb-Edu / Wikipedia, IbomMT reference pairs; reward models africomet-qe-stl-1.1 and ssa-comet-mtl (reference-free and reference-based), GlotLID, the science terms, and the NLLB-3.3B pseudo-reference signal (reward only) for kea kbp nus bam kab; start = mixed best. No TranslatePsy.

**Trainer dev (AfriScience-MT dev subset, 200 sentences per language, greedy chrF++ macro):** E-14 43.50 (step 0) / 43.19 (25) / 43.31 (50) / 43.38 (75) / 43.35 (100) / 43.34 (125) / 43.48 (150) / 43.43 (160); executed science RL (start alpha 0.5): 44.62 / 44.79 / 44.91 / 45.19 / 45.06 / 45.20 / 45.05 / 45.13. The science SFT is worth +1.1 on this dev subset (alpha 0.5 vs alpha 0), the RL after it +0.5; the RL started from mixed best moved the dev macro by -0.07.

### 1.1 AfriScience-MT, sentence level, greedy, macro over 6 languages (amh hau lug nso yor zul); TEST then DEV

| model | chrF | chrF++ | AfriCOMET-STL (held out) | SSA-COMET-QE (reward family) | SSA-COMET-MTL ref (reward family) |
|---|---:|---:|---:|---:|---:|
| mixed best (= alpha 0; E-14's start) | 45.40 | 42.44 | 0.6828 | 0.6130 | 0.5123 |
| B1 final (stage-B1 RL, b256) | 45.45 | 42.50 | 0.6846 | 0.6141 | 0.5102 |
| **E-14 final (no science SFT)** | **45.66** | 42.69 | **0.6850** | 0.6153 | 0.5163 |
| science SFT alpha 0.5 (executed run's start) | 46.63 | 43.80 | 0.6879 | 0.6118 | 0.5150 |
| science RL best (from the science SFT) | 46.79 | 43.93 | 0.6893 | 0.6135 | 0.5185 |
| science RL final (from the science SFT) | 47.10 | 44.22 | 0.6932 | 0.6158 | 0.5224 |
| DEV: mixed best / B1 final / **E-14 final** / alpha 0.5 / sci RL best / sci RL final | 45.74 / 45.58 / **46.00** / 47.17 / 47.38 / 47.62 | | 0.6988 / 0.6993 / **0.7020** / 0.7043 / 0.7067 / 0.7098 | 0.6242 / 0.6251 / 0.6268 / 0.6235 / 0.6255 / 0.6277 | |

Paired differences (second minus first, as printed by `e14_pairs.py`; here written as E-14 minus X), TEST; intervals 95%, Poisson bootstrap, independent weights per language, 1,000 resamples:

| E-14 final minus | chrF | chrF++ | AfriCOMET-STL (held out) | SSA-COMET-QE (reward family) |
|---|---|---|---|---|
| **science RL final (started from the science SFT)** | **-1.44 [-1.61, -1.25]** | -1.53 [-1.70, -1.35] | **-0.0082 [-0.0101, -0.0065]** | -0.0006 [-0.0018, +0.0007] |
| science SFT alpha 0.5 | -0.97 [-1.20, -0.75] | -1.11 [-1.33, -0.90] | -0.0030 [-0.0050, -0.0009] | +0.0035 [+0.0022, +0.0047] |
| mixed best (its own start) | +0.26 [+0.12, +0.39] | +0.25 [+0.11, +0.38] | +0.0022 [+0.0007, +0.0036] | +0.0023 [+0.0013, +0.0033] |
| B1 final (the b256 RL) | +0.21 [+0.07, +0.36] | +0.19 [+0.05, +0.33] | +0.0004 [-0.0012, +0.0020] | +0.0012 [+0.0000, +0.0023] |
| (context) science RL final minus mixed best | +1.70 [+1.50, +1.89] | +1.78 | +0.0104 [+0.0086, +0.0123] | +0.0028 |
| (context) science RL final minus B1 final | +1.65 [+1.48, +1.82] | +1.71 | +0.0087 [+0.0068, +0.0104] | +0.0017 |

DEV (same comparisons): E-14 minus science RL final -1.63 [-1.81, -1.44] chrF, -0.0078 [-0.0096, -0.0058] AfriCOMET; minus alpha 0.5 -1.17 [-1.40, -0.94]; minus mixed best +0.25 [+0.10, +0.42]; minus B1 final +0.42 [+0.27, +0.57].

Per language, TEST, E-14 minus science RL final (chrF): amh -1.21 [-1.61, -0.80], hau -1.34 [-1.84, -0.89], lug -1.11 [-1.64, -0.53], nso -2.70 [-3.01, -2.34], yor -0.72 [-1.07, -0.27], zul -1.55 [-1.94, -1.15]: all six significantly lower; AfriCOMET lower in five (yor inside). E-14 minus mixed best per language: lug +0.62 [+0.24, +1.07] and yor +0.23 [+0.02, +0.47] significant, the others inside noise.

### 1.2 Benchmarks, FLORES+ (54) / NTREX (24) / MAFAND (15) = 93 cells, standard harness (vLLM 0.32.0 bf16 greedy 768 tokens, held-out AfriCOMET; contamination-flagged rows removed)

| system | chrF++ all 93 | FLORES+ | NTREX | MAFAND | AfriCOMET FLORES+ | finished | loop (reward flag) | loop (independent) |
|---|---|---|---|---|---|---|---|---|
| SFT | 26.58 | 22.85 | 29.25 | 35.77 | 0.5538 | 0.866 | 0.160 | 0.236 |
| mixed best | 34.55 | 31.66 | 37.16 | 40.78 | 0.6267 | 0.9972 | 0.0297 | 0.0537 |
| B1 final | 34.34 | 31.39 | 36.98 | 40.73 | 0.6250 | 0.9976 | 0.0322 | 0.0620 |
| **E-14 final** | **34.68** | 31.82 | 37.28 | 40.78 | 0.6276 | 0.9984 | 0.0257 | 0.0477 |

E-14 minus mixed best: **+0.13 [+0.10, +0.15]** chrF++ over 93 cells (FLORES+ +0.17 [+0.13, +0.20], NTREX +0.12 [+0.08, +0.16], MAFAND 0.00 [-0.06, +0.06]), AfriCOMET (FLORES+) +0.0009 [+0.0004, +0.0015], reward-flag loop rate -0.004, independent flag -0.006. E-14 minus B1 final: **+0.34 [+0.31, +0.36]** (FLORES+ +0.43, NTREX +0.30, MAFAND +0.06 [-0.01, +0.12]), AfriCOMET FLORES+ +0.0026 [+0.0020, +0.0032]. (The all-cells AfriCOMET intervals in `cmp_bench93.md` are anomalously wide for a few NTREX cells with few cached scores; the per-benchmark values are used here.) NTREX + MAFAND (39 cells) against the executed science RL final (stored predictions): E-14 minus science RL final **-0.17 [-0.21, -0.12]** (NTREX -0.05 [-0.09, +0.01], MAFAND -0.36 [-0.43, -0.29]); science RL final minus mixed best +0.24 [+0.20, +0.29]; E-14 minus alpha 0.5 +0.13 [+0.08, +0.18].

### 1.3 Documents, 168 test documents (28 papers x 6 languages), macro; descriptive (no bootstrap)

| model | whole: chrF / finished / loop | chunked: chrF / loop / AfriCOMET window / SSA-COMET-MTL(ref) window |
|---|---|---|
| mixed best (alpha 0) | 22.8 / 0.71 / 0.86 | 57.29 / 0.14 / 0.6872 / 0.4479 |
| science SFT alpha 0.5 | 24.9 / 0.51 / 0.86 | 58.81 / 0.23 / 0.6924 / 0.4476 |
| **E-14 final** | **25.6 / 0.69 / 0.82** | **57.30 / 0.14 / 0.6893 / 0.4506** |
| science RL final | 27.4 / 0.70 / 0.85 | 58.74 / 0.20 / 0.6956 / 0.4547 |

(The SSA-COMET-QE window score of E-16 was not recomputed for E-14; the reward-family SSA-COMET-MTL(ref) window is shown.)

## 2. Verdict by the pre-registered rule

* **Science SFT needed: YES.** For the rule (AfriScience-MT test, E-14 final minus science RL final): the upper bounds are -1.25 chrF (< -0.5) and -0.0065 AfriCOMET (< -0.004); both metrics fail non-inferiority, in all six languages for chrF. The same on DEV (-1.63 [-1.81, -1.44]). The science RL started from the general RL model reaches 45.66 test chrF against 47.10 with the science SFT in front: **the science SFT, not the science RL, carries the science gain** (alpha 0.5 minus mixed best +1.23 chrF; science RL on top of it +0.46; science RL alone on top of mixed best +0.26 [+0.12, +0.39], AfriCOMET +0.0022).
* **A reward-family caveat that points the other way:** SSA-COMET-QE (the reward family) rates E-14 final level with the science RL final (-0.0006 [-0.0018, +0.0007]) and above alpha 0.5 (+0.0035), while chrF and held-out AfriCOMET rate it 1.4 chrF below and below alpha 0.5. The RL moved the metric it optimises without a matching move in the independent metrics: SSA-COMET-QE is not evidence of quality for these runs (DISCLOSURES section 3).
* **The b256 RL comparison (the owner's phrase is ambiguous; all three reported):** against mixed best (+0.26 chrF) and B1 final (+0.21 chrF [+0.07, +0.36], AfriCOMET +0.0004 [-0.0012, +0.0020]) the E-14 gain is small and about the size at which seed noise matters (E-04: per-language seed differences up to 0.5 chrF++); against the science RL final it is clearly lower. Against B1 final on the 93 benchmark cells E-14 is +0.34 chrF++ [+0.31, +0.36]: on the general benchmarks the science RL from mixed best gives a small positive transfer, with fewer loops (reward flag 0.026 vs 0.032), no regression.
* Non-regression (NTREX / MAFAND against mixed best, lower bound above -0.5): holds (+0.12 / 0.00). The executed science RL final is higher on MAFAND (+0.36) than E-14.
* Documents: no document-level effect of the E-14 run: chunked chrF 57.30 against 57.29 for its start; whole-document chrF +2.8 (25.6 vs 22.8) with finished 0.69 and loop 0.82 (still failing, as in every model without long-document training).

## 3. Caveats / not done
One seed; E-14 had one OOM restart (micro-batch 6,144 after step 20 instead of 12,288; the replayed steps 21-38 can differ from the first pass because vLLM sampling is not bit-reproducible; the resumed state is the step-20 checkpoint); `best/` does not exist; the verdict does not depend on a `best`. The comparison with the science RL final mixes the effect of the science SFT with the interpolation (alpha 0.5, 50% weights of the general model); a science SFT at alpha 1.0 followed by RL was not run. 160 steps, as executed; a longer E-14 run was not tried. Document level: 28 papers per language, descriptive. Seed-noise-sized differences (E-14 vs mixed best / B1 final) are reported with their sentence-level intervals but are single-run results. Not done: evaluation of a second seed, long-document RL from E-14.
