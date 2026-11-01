# Language draw weights as a run control, and a chrF-informed adaptive updater (D-65, agent R13, 2026-10-22)

Owner question (2026-10-22): "can we use the chrF score to inform the sampling weight per language, to see if it affects performance?" Context: Yoruba is expected to rise; today the draw weights are static (`language_weights` in the yaml: 3.0 for ish bin bbj tsc kon nus dag oke, 2.5 for 17 others, 2.0 for hau lug nso zul, 1.0 for 48) plus the D-59 floor (>= 2 prompts per language per 20 steps).

Scope: CPU analysis and code only. The running final run (`/workspace/jobs/r7/runs/final`) was never touched (read-only copies of its metrics); it does **not** contain this code (a Python process does not re-import); see section 7 for how it could be picked up. My tree: `/workspace/jobs/r13/gemma_mt_run` (= `/workspace/jobs/r7/gemma_mt_run` + the D-65 hunks, nothing else). Laptop mirror: `gemma_mt/` of this repo. Nothing pushed.

Files: code `gemma_mt/rl/lang_weights.py` (new), hunks in `gemma_mt/rl/data.py` and `gemma_mt/rl/train_rl.py`, tests `tests/test_lang_weights.py` (27), scripts `scripts/replay_lang_weights.py`, `replay_schedule_draws.py`, `lang_weight_evidence.py`, `lang_weight_experiment_power.py`, `lang_weight_experiment_pairs.py`, patch `patches/D65_lang_weights.patch`, data `reports/adaptive_language_weights_2026-10-22/`.

## 0. Short answer

1. **Yes, technically; whether it helps performance is not answerable from the data we have, and the obvious way of doing it (weight by low QE reward) would be wrong in at least one important case.**
2. *Evidence from the finished 2c run* (section 2): the number of RL samples a language saw is almost perfectly collinear with the 2.5x weight class (r = 0.988; within a class the samples differ by 6-13 %). The best regression (gain ~ SFT level + reward mode + script + log2 samples, 64 languages) gives **+1.4 chrF++ per doubling of RL samples, 95 % bootstrap CI [-0.3, +3.1]**; the same effect read as "being in the 2.5x class" is +2.0 [-0.3, +4.2]. Consistent with a real effect of roughly that size, equally consistent with none. What *is* clear: the SFT level explains most of the variation (-1.6 chrF++ of gain per +10 points of baseline, CI [-2.7, -0.6]; R^2 0.47 for the baseline alone), i.e. headroom, not draw count, is the first-order effect.
3. **2c itself was a plateau.** Over its 1,433 steps (900,384 completions) the 19-language dev macro chrF++ went 41.89 -> 41.88 (31 evaluations, range 41.57-41.98, trend +0.09 per 1,000 steps), no dev language moved by more than 1.4. The SFT -> RL gain was made before (phases 1, 2, 2b-b64). So the *marginal* value of extra samples at the current policy state was ~0 on the dev sets, and a cross-sectional "gain vs cumulative samples" slope says nothing about it.
4. **QE headroom is not chrF headroom.** Greedy hold-out reward (shaped so that 1.0 = the median human reference, 2.0 = human P90): the "strong" languages zul 0.78, xho 0.42, sot 0.57, som 0.38, swh 0.02 sit *below* the human median, with flat chrF++ (small gains explained mostly by their high baseline); Yoruba sits at 1.85 (near human P90, i.e. "QE saturated") while its chrF++ is 27-33 and it has the largest absolute headroom of the 19 dev languages. A rule "weight up where QE reward is low, down where it is high" would boost zul/xho/sot/som/swh and cut Yoruba. Hence the updater is gated by failures, reliability flags and reference-based dev trends, and shrinks a QE-saturated language only mildly unless a *precisely measured* human-reference trend confirms the plateau.
5. **What I built:** (a) `LANG_WEIGHTS.json` run control (read every step, validated, merge/replace/multiplier/freeze, logged as `reward/lang_weights/<iso>`); (b) an automatic updater `adaptive_lang_weights` (default OFF, dry-run mode, treated-subset option, bounded, smoothed, budget-preserving, resume-safe, fault-guarded); (c) offline replay tool; (d) 27 CPU tests; (e) per-language training statistics (`lang/*`) that did not exist before (the 2c and current final metrics have **no per-language training reward at all**).
6. **Expected effect size is small.** Moving weights by +-25 % is 0.3 doublings = ~0.4 chrF++ at the point estimate (1.0 at the upper CI), below anything the dev sets can see (section 5). The updater is therefore best judged by its mechanism (dry run first), and the causal question needs a designed contrast (9 pre-registered matched pairs, treated 3.0 vs control 1.0: 74 % power at the point estimate, 25 % at half of it; 3.0 vs 0.5: 97 % / 48 %), which moves 11-15 % of the draw mass and is the owner's call.

## 1. What the weights do now (so that the controller composes correctly)

* `PromptSchedule` draws domain -> length bin -> language (probability ~ pool_size^alpha x weight inside the (domain, bin) pool; `language_alpha: 0` so ~ weight) -> prompt, stateless in (seed, step). The D-59 floor forces 2 x 77 = 154 slots per 20 steps (7.7 of the 42 prompts per step) to named languages; the other ~34.3 slots follow the weights.
* Expected prompts per language over the 800-step final run = 80 (floor) + 27,440 x w / sum(w) with sum(w) = 122.5: w 1.0 -> 304, 2.0 -> 543, 2.5 -> 640, 3.0 -> 752 (matches `language_coverage_800.json`: 304 / 544 / 639 / 745). A weight of 0.5 gives 192, 1.6 gives 438, 4.5 gives 1,088. The floor means no language can be starved whatever the controller does.
* Weights are relative inside each pool, so only ratios matter; the updater keeps the **sum** of the treated languages' weights equal to the sum of their base weights ("bounded total draw budget").

## 2. Evidence from the finished 2c run

Data: `reports/adaptive_language_weights_2026-10-22/` (`per_language.csv`, `regression.json`, `saturation.csv`, `lang_draws_by_phase.json`, `power.json`, `experiment_pairs.json`, `replay_2c_trajectory.csv`), produced by `scripts/replay_schedule_draws.py` and `scripts/lang_weight_evidence.py` (numpy only, laptop or box).

### 2.1 How many samples did each language see?

The metrics of 2c carry no per-language sample counts, but the schedule is stateless in (seed, step, weights), so I replayed it over the real `rl_prompts_v3` pool structure (5,986,200 prompts, exact per-(language, bin) pool sizes). The replay reproduces the **language set of 1,411 of 1,433 logged steps exactly** (checked against the logged `reward/lid/fail_rate/<iso>` key sets; pcm/kon have the LID guard off). It also exposed something the config.json does not say: the weight of **efi changed during 2c** (1e-4 for steps 1-350, absent = 1.0 for 351-450, 2.5 from 451: restarts with edited yaml), so efi saw only ~919 prompts vs ~1,340 for the other 2.5x languages.

| phase | steps | group | prompts | completions | note |
|---|---|---|---|---|---|
| 1 (rl_rtxpro.yaml) | 300 | 8 | 13,840 | 110,720 | approximate: alpha 0.5, the language set of that time is unknown |
| 2 | 497 | 8 | 47,620 | 380,960 | alpha 0, 17 languages x2.5 |
| 2b stage b64 | 488 | 16 | 31,232 | 499,712 | same weights (the diverged b256 stage was discarded) |
| 2c | 1,433 | 16 | 56,274 | 900,384 | exact replay with the efi regimes |

2c prompts per language: 2.5x class (18 languages) mean 1,340 (max 1,399; efi 919), weight-1.0 class (46 languages with reference sets) mean 549 (sd 24, range 500-599): ratio 2.44. **The sample count is the weight class**: corr(log2 samples, class) = 0.988, sd of log2 samples inside a class 0.06 (weight 1.0) and 0.13 (2.5), between classes 1.28 doublings.

### 2.2 SFT -> RL gain vs samples (64 RL languages with human-reference rows)

Outcome: chrF++ gain RL-final (2c step 1433) minus SFT-final on `no_pair` rows, precision-weighted over the benchmarks a language has (FLORES+ devtest, NTREX, MAFAND, SmolSent, SALT, IbomMT, GlotStoryBook; table `reports/final_model_per_language_wide.csv`). Covariates: SFT level (same benchmarks), reward mode of the 2c calibration (qe 32 / blend 19 / consensus 13 among these), script (Ethiopic = amh, tir), log2(completions). 95 % intervals: bootstrap over languages, 4,000 resamples. Between-language SD after controlling for baseline is 2.7 chrF++ (measurement SE of a language's gain: 0.36).

| model (OLS) | R^2 | SFT level (per +10) | log2 samples (per doubling) | notes |
|---|---|---|---|---|
| A: baseline only | 0.47 | -1.94 [-2.60, -1.33] | | |
| B: + mode | 0.51 | -1.59 [-2.53, -0.89] | | blend +1.9 [-0.3, +3.8], consensus +0.9 [-1.4, +3.0] vs qe |
| C: + script + log2 samples (2c) | 0.58 | -1.63 [-2.66, -0.59] | **+1.38 [-0.28, +3.11]** | non-Latin script -5.7 [-7.6, 0.0] |
| D: C with cumulative samples (phases 2, 2b, 2c) | 0.57 | -1.69 [-2.75, -0.72] | +1.14 [-0.58, +2.80] | |
| E: C with the 2.5x indicator instead | 0.58 | -1.57 [-2.62, -0.62] | 2.5x class: **+2.00 [-0.26, +4.19]** | |
| F: C + quadratic SFT term | 0.58 | | +1.42 [-0.71, +3.40] | quadratic term -0.06 [-0.64, +0.51]: nothing |
| row level (143 language x benchmark rows, benchmark fixed effects, language-clustered bootstrap) | | -1.50 [-2.61, -0.73] | +1.67 [-0.31, +3.38] | |

WLS variants and the full coefficient tables are in `regression.json` (same picture). **Model-free check on the confounded comparison** (low-baseline stratum, SFT < 22): 17 languages at 2.5x (2.47x the samples) gained 8.8, 12 at 1.0x gained 7.9: difference +0.96 [-1.35, +3.46]; SFT < 26: +1.11 [-0.78, +3.09].

**What can be concluded**: headroom (SFT level) dominates; a weight/sample effect of order 1-2 chrF++ per doubling is compatible with the data and an effect of zero is too. **What cannot**: (i) dose-response *within* a weight class (the samples differ by 6-13 %: the estimate is meaningless, CI -7..+12 in weight 1.0 and -150..+0.1 in 2.5x); (ii) causality: the 2.5x languages were *chosen* as the weakest by baseline and by QE reliability, they are mostly blend/consensus-mode languages with different reward signals, and baseline control is imperfect; (iii) the marginal value of more samples at the present state (2c was a plateau, see 0.3, and the cumulative count is dominated by earlier phases); (iv) anything about the 13 languages without any human reference.

### 2.3 The saturation hypothesis for afr swh zul xho sot som ibo

| language | SFT chrF++ | gain | residual vs baseline-only line | 2c dev change (first 3 -> last 3 evals) | greedy hold-out reward (1.0 = human median, 2.0 = human P90) |
|---|---|---|---|---|---|
| afr | 61 | +0.8 | +1.5 | -0.1 | no hold-out prompt |
| swh | 54 | +1.7 | +1.2 | +1.4 | 0.02 |
| zul | 45 | +0.9 | -1.3 | -0.1 | 0.78 |
| xho | 43 | +1.4 | -1.3 | +0.8 | 0.42 |
| sot | 44 | +0.7 | -1.9 | -0.3 | 0.57 |
| som | 40 | +1.6 | -1.6 | 0.0 | 0.38 |
| ibo | 39 | +3.6 | +0.2 | -0.3 | 1.16 |

Verdict: **partly supported, but not by the reward.** The small gains are mostly the baseline effect (mean residual -0.6, inside the 2.7 residual SD; afr and swh are *above* the line). In 2c every dev language was flat (macro SD 0.09; per-language change mean +0.07, sd 0.60, range -0.9..+1.4 over 19 languages), weak ones (fon, orm, wol, yor, ewe, twi) as much as strong ones, so 2c cannot discriminate. The shaped-QE anchors do **not** show saturation for these languages: their greedy reward is 0.0-0.8, below the human-median anchor (1.0) and far from t_great (2.0). Across the 23 qe-mode languages with a usable hold-out reward the Spearman correlation between greedy reward and RL chrF++ is -0.23 (with the gain +0.23): the reward level does not rank languages by reference quality. Caveats: the hold-out reward is 1-3 fixed greedy prompts per language (and is -2.0 at every evaluation for a dozen languages whose one prompt always fails: bbj bin dua iso kab lug nus shi sid ven urh kmb), consensus chrF in the logs falls for several languages (fon 47.6 -> 41.5, loz 54 -> 47, nbl 51 -> 46) but the length-bin stage shift (b256 -> b2048) is confounded with it. Both are the reason why the controller bin-adjusts and shrinks.

## 3. Design of the adaptive weights

Principle: **use the signals that exist for every language, trust the reference only where it exists and is measured precisely enough, never boost on a QE level alone, never move without a signal.** Everything bounded, smoothed, reversible with one file.

### 3.1 Signals

| signal | scope | source | used for |
|---|---|---|---|
| mean reward, bin- and domain-adjusted, shrunk to the content-mode mean (48 pseudo-samples) | all 77 | NEW per-step accumulators (`LangWindow`), logged as `lang/reward`, `lang/radj`, `lang/n`, `lang/fail` | qe-mode languages: distance to human parity (1.0); QE saturation |
| failure rate (status != ok), loop rate (degenerate / no end / post-eos / dangling) | all 77 | same | the one signal all reward modes share: boost when above the median |
| flat-group share, within-group reward std | all 77 | same | learnability gate: groups with no gradient teach nothing |
| advantage weight (calibrated reliability), constraint-penalty mean, QE vs consensus agreement (blend) | all / anchors / blend | `out.weights`, `out.anchors_penalty`, `out.qe` x `out.consensus_chrf` | flags (never boosted) |
| mean KL per language | all | **not wired**: the trainer has no per-sequence KL (a ~6-line hunk in the update loop); the controller accepts `kl=` and flags > 3x median | brake |
| dev chrF++ trend | general dev 19, weak dev 13, AfriSci dev 6 (amh hau lug nso yor zul) | `bench/chrf/<iso>`, `bench_aux/<set>/chrf/<iso>` at the eval cadence | marginal-gain gate (rising / flat-confirmed / falling / unknown) |

Reward-mode caveat built in: for blend and consensus languages the reward *level* has no human parity (a consensus value is sampling agreement), so `need_qe` applies to qe-mode languages only; blend/consensus languages move on failures, flags and dev trend.

### 3.2 Rule (defaults = `AdaptiveParams`, all overridable through `adaptive_lang_weights_params`)

```
f_i = clip( u_i * d_i * m_i , min_ratio=0.5, max_ratio=1.6 );  if flagged: f_i <= 1
u_i = 1 + 0.5 * max(need_qe_i, need_fail_i) * reliability_i * learnability_i          # boost for under-performance, max x1.5
      need_qe   = clip((1.0 - R_i) / 3.0, 0, 1)   (qe mode only, R_i shrunk bin-adjusted reward; 1.0 = human median, -2 = floor)
      need_fail = clip((fail_i - max(median fail, 0.05)) / 0.30, 0, 1)
      learnability = min(1, (1 - flat share) / 0.5)
d_i = saturation shrink, qe mode only, R_i >= 1.4, fail <= 5 %, not flagged:
      d = 1 - (1 - floor) * (0.5 + 0.5 * sat),  floor = 0.6 if the dev trend is CONFIRMED flat/falling (se <= 0.5 and |slope| <= 0.3 chrF++ per 100 steps),
                                                  0.85 if no/imprecise reference, 1.0 (no shrink) if the dev trend is rising
m_i = 1.15 if dev trend rising (t >= 1.5), 0.85 if falling (t <= -1.5), else 1
flags (f <= 1): qe-consensus disagree (blend, corr < 0.05), anchor penalty mean > 0.25, loops > 25 %, reliability < 0.3, dev falling, KL > 3x median
target_i = clip(base_i * f_i, [0.5, 1.6] * base_i, [0.25, 4.5]);  new_i = prev_i + 0.5 * (target_i - prev_i), at most +-25 % per update
sum over treated languages rescaled to the sum of their base weights (re-clipped, 4 passes)
```
Update every `adaptive_lang_weights_every` = 50 steps (the dev-benchmark cadence) from step 100 (two windows); a language needs >= 6 groups in the window (the floor alone gives 5 per 50 steps, the draws add ~14 for a weight-1 language), else f = 1 (no signal, no move). EMA of window statistics (0.5). The D-59 floor is untouched.

Behaviour on archetypes (`factors`, background languages at R = 0.9):

| language | f |
|---|---|
| qe laggard (R = -0.5) | 1.22 |
| ... but 95 % of its groups are flat | 1.02 |
| ... with 40 % loops / anchor penalty 0.4 / KL 10x median | 1.00 (flagged) |
| ... with a falling dev trend | 1.00 (flagged) |
| qe near parity (R = 0.9) | 1.02 |
| qe saturated, R = 1.85, no reference (Yoruba-like) | 0.88 |
| ... dev trend confirmed flat | 0.69 |
| ... dev flat but imprecise (se 2.0 chrF++/100 steps: a 40-60 sentence cell) | 0.88 (not "confirmed") |
| ... dev trend rising | 1.15 |
| blend language, QE and consensus disagree inside groups | 1.00 (flagged) |
| consensus language, low reward level | 1.00 (no human parity) |
| consensus language, 60 % failures | 1.50 |
| no signal (2 groups) | 1.00 |

Per update a weight moves at most 25 %; four consecutive updates are needed to reach the 1.6x cap; the cap on a weight-3.0 language is 4.5 (1,088 expected prompts vs 752).

### 3.3 Run control `<out_dir>/LANG_WEIGHTS.json`

```json
{"weights": {"yor": 3.5, "zul": 1.0},    "multipliers": {"hau": 0.8},
 "mode": "merge",  "adaptive": "on",  "note": "free text"}
```
* `weights` absolute (override yaml AND adaptive weight of those languages), `multipliers` relative to the weight that would apply without the file, `mode` merge (default) | replace (unlisted languages = 1.0, yaml and adaptive ignored), `adaptive` on | freeze (keep the last adaptive weights, stop updating) | off (yaml weights). Weights in [0, 20], multipliers [0, 10]. Precedence: yaml -> adaptive -> file.
* Read again every step through `os.stat` (parsed only when the file changed); an invalid file (bad JSON, unknown language such as `nde`, out-of-range value, unknown key, a pool left with no positive weight) is ignored as a whole and the **last valid weights stay in force** (logged once per file version, and the schedule itself rejects a weight vector that would kill a pool); deleting the file clears the override. A weight of 0 is allowed (the floor still forces 2 prompts per 20 steps).
* Write atomically: `echo '{...}' > LANG_WEIGHTS.json.tmp && mv LANG_WEIGHTS.json.tmp LANG_WEIGHTS.json`.

### 3.4 Anti-hacking and failure-mode safeguards

* No boost from a low QE level alone: qe-mode only, shrunk, learnability- and reliability-weighted, capped at x1.5 and x1.6, and **any flag caps f at 1**. Blend languages must show QE and consensus agreeing inside groups (the property D-43 validated). Rising anchor penalties, loops, low calibrated reliability, a falling reference or a KL outlier all freeze the boost.
* A QE-saturated language is not cut hard without a confirmed reference plateau (0.85 floor vs 0.6), and not at all while its reference is rising. No language can go below 0.5 x base or 0.25 absolute; the floor of the schedule remains.
* Budget: the sum of the treated weights is invariant, so the updater cannot raise the total share of "hard" languages at the expense of the production objective unless someone says so.
* Reference noise is respected: the trend t-statistic uses the measured paired-chrF noise (sd 14.7 / sqrt(n) per cell, section 5), so a 40-60 sentence cell never produces "rising" or "falling" over a few evaluations; only 6-8 evaluations of 200 sentences or more can.
* Observation errors cannot stop a 2-day run: every public method is guarded (log once with traceback, carry on with the weights in force; after 5 errors the updater switches itself off, file control stays).

### 3.5 State, resume

`<out_dir>/LANG_WEIGHTS.state.json` (atomic): history of the weights in force, window accumulators, EMAs and dev histories, written at every trainer checkpoint (`save_every`) and right after each adaptive update. On resume the history is cut at the resumed step; when the sidecar was written at exactly that step the window continues, otherwise the weights come from the history and the next update is skipped ("window incomplete", logged). `weights_at(step)` reproduces any past draw. **Keep `adaptive_lang_weights_every` a multiple of `save_every`** (50 and 50 in the final yaml; the controller warns otherwise). A fresh run in a directory with an old sidecar ignores it (`start_step == 0`).

## 4. Implementation: keys, commands, hunks

### 4.1 yaml keys (all default OFF; nothing changes without them)

```yaml
lang_weights_file: true                 # <out_dir>/LANG_WEIGHTS.json run control (needs run_controls: true)
lang_stats_log: true                    # log lang/{n,reward,radj,fail}/<iso> every step for the languages drawn (always on with the updater)
lang_weights_log_every: 50              # reward/lang_weights/<iso> (all 77) at every change and every 50 steps
adaptive_lang_weights: true             # the automatic updater
adaptive_lang_weights_dry_run: true     # RECOMMENDED FIRST: only logs reward/lang_weights_dry/<iso> and "lang weights (dry run)" lines; the weights in force stay as in the yaml
adaptive_lang_weights_every: 50
adaptive_lang_weights_first_step: 100
adaptive_lang_weights_languages: []     # [] = all 77; else only these move (the rest are the static control group)
adaptive_lang_weights_params: {}        # AdaptiveParams overrides, e.g. {eta: 0.3, max_ratio: 1.4}; an unknown key stops the run at start-up
```
CLI overrides work too (`... --config configs/rl_science_final.yaml adaptive_lang_weights=true adaptive_lang_weights_dry_run=true lang_weights_file=true lang_stats_log=true`).

Metrics added: `reward/lang_weights/<iso>` (weights in force), `reward/lang_weights/adaptive_updates`, `reward/lang_weights/file_active`, `reward/lang_weights_dry/<iso>`, `lang/{n,reward,radj,fail}/<iso>`; `summary.json["lang_weights"]`. Log lines: `[gemma_mt] lang weights adaptive update @ N: yor 2.50->2.71, ...`, `LANG_WEIGHTS.json read / ignored / removed`.

### 4.2 Commands

```bash
# run control (the process picks it up at the next step)
echo '{"weights": {"yor": 3.5}, "note": "owner: Yoruba catch-up"}' > $OUT/LANG_WEIGHTS.json.tmp && mv $OUT/LANG_WEIGHTS.json.tmp $OUT/LANG_WEIGHTS.json
echo '{"adaptive": "freeze"}' > ...   # stop the updater, keep its weights;   rm $OUT/LANG_WEIGHTS.json   # back to yaml / adaptive

# tests (CPU)
cd /workspace/jobs/r13/gemma_mt_run && CUDA_VISIBLE_DEVICES= python -m pytest tests/test_lang_weights.py -q          # 27 passed (includes a CPU end-to-end smoke)

# offline replay of an adaptive trajectory from a metrics file
PYTHONPATH=. python scripts/replay_lang_weights.py --metrics runs/rl2c/metrics.jsonl --calib calibration/shaping_flores_dev_v8.json \
    --config runs/rl2c/config.json --draws reports/adaptive_language_weights_2026-10-22/lang_draws_by_phase.json --out-csv traj.csv
PYTHONPATH=. python scripts/replay_lang_weights.py --metrics runs/final/metrics.jsonl --calib ... --base-json '{...}' --every 50 --first-step 100   # a run started with lang_stats_log

# evidence / experiment arithmetic
python scripts/replay_schedule_draws.py --prompts $RL_PROMPTS_V3 --out lang_draws_by_phase.json --check runs/rl2c/metrics.jsonl
python scripts/lang_weight_evidence.py --draws ... --wide reports/data/final_model_per_language_wide.csv --calib calibration/shaping_flores_dev_v8.json --metrics runs/rl2c/metrics.jsonl --out DIR
python scripts/lang_weight_experiment_power.py --evidence DIR --out DIR/power.json;  python scripts/lang_weight_experiment_pairs.py --evidence DIR --out DIR/experiment_pairs.json
```

### 4.3 Hunks (for the owners of the neighbouring changes: R8 reward aggregation, R9 anchors edit `reward.py`, `anchors*.py`, `grpo.py`, and in `train_rl.py` the D-61 `hard_fail_mask` hunks; mine do not overlap them)

`gemma_mt/rl/data.py` (one hunk, `PromptSchedule`): `_build` now calls the new `_p(langs, by_lang)`; new method `set_language_weights(weights)` recomputing the probability vectors of all pools; raises `ValueError` and changes nothing if a pool would have no positive weight. The default path (no call) is bit-for-bit unchanged.

`gemma_mt/rl/train_rl.py` (10 hunks, line numbers of the r13 tree = r7 + D-65):
1. l.39 import `build_controller`.
2. l.168-180 `RLConfig`: 9 new fields (`lang_weights_file`, `lang_stats_log`, `lang_weights_log_every`, `adaptive_lang_weights*`).
3. l.230-240 `validate_config`: params keys, cadence, language names, `lang_weights_file` needs `run_controls`.
4. l.701-708 after `controls = RunControls(...)`: build the controller (modes from `mode_of`), dev-set sizes from `bench_rows` / `aux_sets`.
5. l.721 loop top, before `schedule.indices`: `lw_rec = lwc.update(step)`.
6. l.758-764 after `t_sample = ...`: `lwc.observe(...)` (+ `step_record` when logging stats). Reads `out.rewards, out.status, out.qe, out.weights, out.consensus_chrf, out.anchors_penalty`, `prompts.iso / length_bin / domain`: independent of how R8 aggregates the reward.
7. l.854 `rec.update(lw_rec)`.
8. l.900 after `logger.log(step, ev)`: `lwc.observe_eval(step, ev)`.
9. l.944 after `save_last()`: `lwc.save_state(step)`.
10. l.956 `summary["lang_weights"]`.

`patches/D65_lang_weights.patch` contains exactly these hunks (+ the new files); it applies cleanly (`patch -p1 --dry-run`) to the unmodified r7 tree. Neighbouring suites pass on the r13 tree: test_rl_data, test_language_floor, test_run_controls, test_configs, test_science_run, test_rl_loop, test_greedy_in_group (52 passed) + test_lang_weights (27).

## 5. What the monitors can and cannot see (noise)

Paired chrF++ delta between two models on the same n sentences: sd per sentence 14.7 (median of 136 language x benchmark cells of the wide table; p10 8.1, p90 19.2), so the standard error of a delta is 14.7 / sqrt(n) and the smallest difference detectable with 80 % power (two-sided 0.05) is 2.8 x SE:

| cell | n | minimum detectable change in one language |
|---|---|---|
| general dev (19 languages) | 40 | 6.5 chrF++ |
| weak dev (13 languages) | 60 | 5.3 |
| AfriScience dev sentences (6 languages) | 200 | 2.9 |
| FLORES+ devtest no_pair | 657-1,012 | 1.3-1.6 |
| NTREX / SmolSent | 863-1,997 | 0.9-1.0 |
| macro of 6 AfriSci languages (1,200 sentences) | | 1.2 |
| macro of the 13 weak or 19 general dev languages (780 / 760 sentences) | | 1.5 |

Sentences needed for one language: 1.0 chrF++ -> 1,703; 2.0 -> 426; 3.0 -> 190. **Too noisy to detect**: any per-language change smaller than ~3 chrF++ on the monitors of the run (40/60-sentence cells: 5-6.5), per-language reward from the 1-3 hold-out prompts, the paper-level (25 documents) chrF++, any trend over fewer than ~6 evaluations (the controller encodes this through the t-statistic and the "confirmed flat" rule). The trainer's `select_min_gain: 0.2` is about half of one SE of the 1,200-sentence macro: selection by it is selection by noise unless the improvement is larger.

## 6. Experiment plan: "does it affect performance?"

Reality check first. The pooled estimate (point +1.4, upper CI +3.1 chrF++ per doubling) implies for a **+-25 % weight change 0.3 doublings = 0.4 chrF++ (upper 1.0)**, below the 1.3-2.9 chrF++ the best outcome sets can resolve for a *single language*. Averaging over languages helps only with enough *matched* languages: after matching on content mode, script and SFT level (+-4 chrF++) the SD of the difference between two languages' gains is still 1.8-2.7 chrF++ (RMS 1.84 over the 9 pairs below, SD 2.65 over 29 pairs without the final-weight restriction).

**Tier 0 (free, mechanism, do first).** Restart the final run once from a checkpoint with the patched tree and `lang_weights_file`, `lang_stats_log`, `adaptive_lang_weights` + `adaptive_lang_weights_dry_run`. Over the first 300-400 steps collect `lang/*` and the dry-run proposals. Pass criteria, all measurable without any dev set: (a) split-half reliability of the factor f_i between consecutive windows, Spearman >= 0.5 (otherwise the signal is noise and the updater is dropped); (b) at most ~20 of 77 languages with |f - 1| > 10 % per update; (c) every proposed boost is on a language with no flag, and no boosted language shows a rising anchor penalty / LID failure / loop rate in the next window; (d) proposals for qe-mode languages are not simply the ranking of mean reward (rank correlation with R < 0.8, i.e. the failure/flag/learnability terms matter); (e) Yoruba is not cut unless the AfriSci dev trend is confirmed flat. Cost: ~0 (a few seconds per step; the observe/update code is ~1 ms).

**Tier 1 (cheap, within the run, low power, honest about it): adaptive vs static on matched pairs.** `adaptive_lang_weights_languages: [treated list]`, the control partners stay static at base. Outcome: change of chrF++ from the step-100 checkpoint to the final one on FLORES+ devtest `no_pair` (657-1,012 sentences, SE 0.5 per language), paired-difference over pairs (bootstrap over pairs). With 9 pairs the minimum detectable difference is 1.7 chrF++, with 29 pairs 1.4; the expected adaptive-vs-static effect is ~0.4: power about 11 % (9 pairs) to 15 % (29). **Do not expect this to show an effect**; it is a sanity check that adaptivity does not hurt (non-inferiority within about 1.5 chrF++) and that the guards hold.

**Tier 2 (the actual causal answer): randomised dose on pre-registered matched pairs.** `experiment_pairs.json`: 9 pairs (candidates: final weight 1.0 and FLORES+ devtest; identical mode and script, SFT within 4 chrF++), a balanced randomisation fixed by sha1 (94 balanced assignments; treated-control difference in past gain -0.22, in SFT level -0.10):

| treated (x3) | control (x1) | SFT chrF++ (t / c) | past 2c gain (t / c) |
|---|---|---|---|
| tum | bem | 25.8 / 25.2 | 7.2 / 7.7 |
| wol | lua | 26.0 / 26.9 | 3.1 / 6.4 |
| run | ewe | 27.8 / 28.2 | 4.4 / 2.7 |
| twi | ssw | 34.2 / 33.6 | 1.4 / 3.6 |
| kin | lin | 34.3 / 34.3 | 7.2 / 5.1 |
| sna | ibo | 35.2 / 38.9 | 1.9 / 3.6 |
| nya | mlg | 39.3 / 38.9 | 3.0 / 1.7 |
| tso | som | 42.0 / 40.1 | 2.9 / 1.6 |
| sot | xho | 43.5 / 42.9 | 0.7 / 1.4 |

Implementation: one `LANG_WEIGHTS.json` at step 100 setting the treated weights (adaptive off, or restricted to the other 59 languages). Outcome as in Tier 1 plus NTREX / SmolSent where available and ANCOVA on the past gain. Standard error of the mean pair difference 0.61 chrF++ (RMS 1.84 / sqrt 9); minimum detectable difference 1.7. **The D-59 floor (2 forced prompts per language per 20 steps) dilutes the contrast**: a x3 weight gives 2.2x the prompts, not 3x. With the pooled coefficient (+1.38 per doubling) and half of it:

| contrast from step 100 | expected prompts treated / control | doublings | effect at point / half estimate (chrF++) | power at point / half | extra draw mass |
|---|---|---|---|---|---|
| 3.0 vs 1.0 | 621 / 279 | 1.15 | 1.6 / 0.8 | 74 % / 25 % | +15 % |
| 4.0 vs 1.0 | 750 / 269 | 1.48 | 2.0 / 1.0 | 92 % / 38 % | +22 % |
| 3.0 vs 0.5 | 638 / 196 | 1.70 | 2.3 / 1.2 | 97 % / 48 % | +11 % |
| 2.0 vs 1.0 | 473 / 291 | 0.70 | 1.0 / 0.5 | 35 % / 12 % | +7 % |
| 3.0 vs 1.0 from step 400 | 485 / 290 | 0.74 | 1.0 / 0.5 | 39 % / 13 % | +15 % |

(If the true effect is the lower CI end, about -0.3 per doubling, there is nothing to find; no design here has power for effects below ~1 chrF++.) Recommendation if the owner wants the answer: **3.0 vs 1.0 is the cheapest design with decent power at the point estimate; 3.0 vs 0.5 is the strongest but takes samples away from 9 languages.** More pairs than 9 are not available among weight-1.0 languages with FLORES+ rows (23 candidates); allowing NTREX/SmolSent-only languages adds pairs but changes the outcome set. Cost: no extra GPU hours (a re-allocation inside the 46.4 h / 800-step run) but 11-22 % of the draw mass moves to 9 languages: a decision for the owner, not for me.

**Tier 3 (not recommended).** A separate short ablation (200 steps = 11.6 h per arm at 209 s/step) from a final-run checkpoint: at the plateau 200 steps move a language by well under 1 chrF++, which no monitor resolves.

Metrics for any tier: primary = paired chrF++ delta on FLORES+ devtest `no_pair` (single pre-registered endpoint to avoid 77-fold multiplicity); secondary = NTREX-128, SmolSent, AfriScience dev (6 languages), spBLEU; guards = finished rate, GlotLID right-language rate, anchors penalty rate, general_dev macro (forgetting), QE-reward vs chrF++ divergence per language (reward up, chrF++ down = hacking), KL. Training-time `lang/reward` is a dense, high-power *secondary* outcome (16 samples x >= 5 groups per language per 50 steps) but it is the quantity being optimised, not the evidence of quality.

## 7. How this could reach the running final run

The supervisor already restarts the trainer from `last/` (D-59 shakedown demonstrated kill -> resume). A restart with the patched tree costs the ~7 min start-up plus the steps since the last 50-step checkpoint (kill right after a checkpoint to lose nothing else). Steps, for the lead: (1) apply `patches/D65_lang_weights.patch` (or copy `lang_weights.py` and the two hunks) to `/workspace/jobs/r7/gemma_mt_run`; (2) add the yaml keys of 4.1 (dry-run first); (3) restart at a checkpoint. Everything is off by default, so a restart without the keys is the old behaviour. Not done by me: the run was not touched, no files in `/workspace/jobs/r7` were written.

## 8. Open points and caveats

* The per-language training reward is new data; all thresholds (parity 1.0, saturation 1.4, fail span, flags) are principled but **uncalibrated on real per-language reward trajectories** (none existed). The dry run is how to calibrate them; the replay on 2c and on the first 5 steps of the final run (which stopped at step 5/5 when I looked) only exercise the failure/penalty/dev terms and the plumbing: on 2c the updater (27 updates) would have moved no weight above 1.17x (shi, step 1250) or below 0.96x its base (largest final increases lug, lua, bem, ssw, hau, shi, driven by failure-rate proxies and the 2-sample hold-out reward; `replay_2c_trajectory.csv`, `replay_2c_stdout.txt`).
* Phases 1-2b are replayed from the yaml files; like 2c they may have been restarted with edited yaml (efi was), so their counts are approximate; this affects model D only.
* KL per language is not wired; group-normalised advantage std equals 1 by construction for non-flat groups, so the informative statistics are the raw within-group std and the flat share (both kept).
* `reward/lang_weights/*` logs all 77 weights on change and every 50 steps (about 80 keys); `lang/*` about 100 keys per step, 2-3 KB per step, ~2 MB over 800 steps.
* The rule's `parity = 1.0` assumes the calibrated shaping (t_mid = human median -> 1.0); with another reward preset the constant must be changed through `adaptive_lang_weights_params`.
