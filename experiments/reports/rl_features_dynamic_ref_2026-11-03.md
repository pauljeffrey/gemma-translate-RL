# RL features for the 385M run: dynamic sampling, reference-anchored reward, advantage std floor, beam-guided member (2026-11-03)

Owner requests of 2026-11-03: (1) dynamic sampling, (2) reference-anchored reward, (3) advantage std floor, (4) beam-guided group member, (5) per-prompt group statistics (for a later "prompt replay by learnability", NOT implemented).
Everything is **default OFF in code** (old configs and the running preflight/stage-A behave exactly as before; `test_off_by_default_changes_nothing` and `test_off_is_bit_for_bit_the_old_loop` check this) and **ON in the stage yamls A / B1 / mixed / final** (not in the preflight, not in F2).
Resume safety: only new config keys with defaults and new log files; `last/trainer_state.pt`, the optimizer state, the KL-guard state and the prompt-schedule sidecar are untouched; `config.json` is write-only, so a stage can be resumed from `last/` with the new yaml.
Decision log: `gemma_mt/DECISIONS.md` D-69 (dynamic sampling), D-70 (reference term). VERIFIED = run by me; ESTIMATED = not measured.

## 1. What is in the patch (`gemma_mt/` tree)

| file | change |
|---|---|
| `gemma_mt/rl/dynamic_sampling.py` (new) | zero-variance detection, row selection / concatenation of rollouts and reward outputs, the refill loop, per-language counters |
| `gemma_mt/rl/train_rl.py` | config keys (below), dynamic-sampling step, guide handling, group statistics, monitoring on all generated groups |
| `gemma_mt/rl/grpo.py` | `adv_std_floor` (`_denominator`), `drop` mask (a hard-failed guide leaves its group), `cap_member_share` |
| `gemma_mt/rl/reward.py` | `reward.reference_weight`, `Sample.reference`, `Sample.guide`, `RewardOutput.ref_chrf` |
| `gemma_mt/rl/data.py` | `reference` / `ref_source` / `ref_licence` columns, `RLPrompts.reference`, `PromptSchedule.replacements` |
| `gemma_mt/rl/sampling.py` | `generate_beam_completions` (HF beam search on the trainer's policy) |
| `scripts/clean_rerun_385m_make_configs.py` | `--features ds,adv,log,guide,ref` (default all) writes the new keys into the stage yamls |
| `scripts/clean_rerun_tools.py` | validator: replay with domains (it was broken by the news domain), the new keys, F2 hub check (stale after the rename) |
| `scripts/build_rl_prompts_ref.py`, `scripts/group_std_percentiles.py` (new) | reference pool builder, percentiles of `group_stats.jsonl` |
| `configs/clean_rerun_385m/rl_{A_b64,B1_b256,C_mixed,F_final,F2_long}.yaml` | regenerated with all features (the preflight yaml is byte-identical to before) |
| `tests/test_dynamic_sampling.py`, `test_reference_reward.py`, `test_adv_floor_guide.py`, `test_dynamic_ref_loop.py` (new), `test_clean_rerun_385m.py` | CPU tests |

## 2. Config keys (RLConfig unless marked `reward.`)

| key | code default | stage yamls A / B1 / mixed / final | meaning |
|---|---|---|---|
| `dynamic_sampling` | false | true (F2 false) | draw ceil(prompts x oversample) prompts, drop flat groups, refill, update on the first `prompts` informative groups |
| `dynamic_sampling_oversample` | 1.0 | 1.1 | |
| `dynamic_sampling_max_refills` | 2 | 2 | extra generation rounds per step (cost bound: 1 + 2 generation calls) |
| `dynamic_sampling_min_std` | 0.01 | 0.01 | a group whose reward std (sampled members only with `greedy_stats: sampled`) is below this is flat; floor 1e-6 |
| `adv_std_floor` | 0.0 | 0.02 | advantage = (r - mean) / max(std, floor) instead of (r - mean) / (std + 1e-4) |
| `log_group_stats` | false | true | `<out_dir>/group_stats.jsonl` (one line per generated group: step, prompt index, language, bin, domain, std, mean, n_fail, guide) and `group_std/p05..p75` in the step record |
| `guide_fraction` | 0.0 | 0.25 (F2 0.0) | share of the prompts of a step that get a beam-guided member |
| `guide_beams` | 4 | 4 | |
| `guide_adv_share_cap` | 0.0 | 0.25 | the guides' share of the step's token-weighted \|advantage\| is limited to this |
| `reward.reference_weight` | 0.0 | 0.35 | w in r = (1 - w) r_existing + w consensus_value(chrF++ vs the human reference) |
| `domain_weights` | - | `{general: 0.595, news: 0.255, ref: 0.15}` (A, B1, mixed) | the `ref` domain needs the reference pool files at `<prompts_path>/ref/` |

**Why 0.02 for the std floor.** The preflight (`runs_385m/preflight/metrics.jsonl`, 49 training steps) records only the pooled `reward_std` (0.55-0.77) and `flat_group_frac` (median 0.016, max 0.078), NOT the within-group std, so the 10th percentile could not be read from it. I did not guess a high value: a floor far above the typical group std would shrink every advantage (a hidden learning-rate cut). 0.02 reward units is ~0.005 AfriCOMET-QE units (the shaping slope is ~4 reward units per QE unit) = noise level; it damps only groups that are nearly flat (advantage magnitude std / 0.02 < 1) and does nothing to ordinary groups. `log_group_stats` now records the real distribution: after ~20 steps run `python scripts/group_std_percentiles.py <run_dir> --by-language` and, if the 10th percentile is clearly different, re-set `adv_std_floor` at a checkpoint boundary (and `dynamic_sampling_min_std` to about half of it).

## 3. Dynamic sampling (D-69)

* Informative = reward std >= max(min_std, 1e-6) over the SAMPLED members (greedy member excluded when `greedy_stats: sampled`); a hard-failed guide is not part of its group; a guide that is not failed counts, so **a group that is informative only because of the guide is kept** (test).
* Replacement prompts keep the language, domain and length bin of the dropped prompt (`PromptSchedule.replacements`, a pure function of (seed, step, round, slots)), so the schedule's language mixture is kept in expectation (test: the kept set has the same language shares although the filter removed one language unevenly). Weak languages are therefore not starved: they are redrawn. Per-language counters: `<out_dir>/dynamic_sampling_by_language.json` (written every 10 steps), per step `ds_flat/<iso>` for the worst five.
* **Monitoring is on all generated groups**, the update on the kept ones: reward, reward_std, hard_fail_frac, finished_rate (the early-stop / plateau input), greedy_finished / loop rate, the D-65 per-language window, the translation archive. A kept-only finished rate would look better than the policy is (the filter removes exactly the all-failed groups). `completion_len`, `adv_*`, `weight_mean`, the greedy advantage share and the update use the kept groups.
* All flat step: the first `prompts` groups of round 0 are kept (zero advantage; `ds/fallback`). Fewer informative groups than `prompts` after the refills: the step trains on those (`ds/short_batch`).
* Step accounting: one optimizer step = one update on the kept groups; `save_every`, KL guard, warm-up, extension and patience logic are unchanged. The token loss is normalised by the kept tokens (flat groups no longer dilute it): when many groups are flat the effective step is a little larger; the KL guard watches it.
* **Cost / risk.** Generation work is ~(1 + flat share + 0.1 oversample surplus); the preflight's flat share was 0.00-0.06 (median 0.016), higher in weak / consensus languages, so expect +10-15 % generation and QE-scoring time per step (ESTIMATED). Surplus informative groups are discarded (counted in `ds/wasted_token_frac`). In stages with very few informative groups the refill cap bounds the cost.

## 4. Advantage std floor

`grpo._denominator`: with `adv_std_floor` > 0 the divisor is `max(std, floor)` (all three advantage functions); flat groups (std < 1e-6) stay at 0; 0 = the old `std + std_eps` (bit-identical, tested).

## 5. Reference-anchored reward (D-70)

r = (1 - w) * r_existing + w * consensus_value(chrF++(translation, reference)), w = 0.35; chrF++ = `consensus.chrf_sentence` (== sacrebleu `CHRF(word_order=2)`), `consensus_value` = the fixed linear map [0, 100] -> [-2, +2.25] already used for the consensus signal (offset cancels inside a group; slope 0.0425 reward units per chrF++ point; the mix stays inside the reward range). Applied only to samples with status ok whose reward is above the failure penalty, in training steps only (`with_reference`); failures, guards and every penalty are untouched. Metrics `reward/ref/n|chrf_mean|reward_shift_mean`.

**Pool (release 1, `scripts/build_rl_prompts_ref.py`, domain `ref`, 12,000 rows, 8 languages), every pair checked on both sides against the protected index (6 IbomMT pairs dropped; the AfriScience clean-train file was already filtered; 201 duplicate English sentences, 27 pairs failing the length/format rules):**

| language | rows | b64 | b256 | sources (rows) | licence |
|---|---:|---:|---:|---|---|
| amh | 1669 | 1169 | 500 | afriscience_mt_train 1196, afriscience_mt_train_window 473 | apache-2.0 |
| efi | 993 | 953 | 40 | ibom_mt_train 993 | cc-by-nc-4.0 |
| hau | 1668 | 1168 | 500 | afriscience_mt_train 1196, afriscience_mt_train_window 472 | apache-2.0 |
| ibb | 998 | 958 | 40 | ibom_mt_train 998 | cc-by-nc-4.0 |
| lug | 1668 | 1168 | 500 | afriscience_mt_train 1196, afriscience_mt_train_window 472 | apache-2.0 |
| nso | 1667 | 1167 | 500 | afriscience_mt_train 1199, afriscience_mt_train_window 468 | apache-2.0 |
| yor | 1668 | 1168 | 500 | afriscience_mt_train 1200, afriscience_mt_train_window 468 | apache-2.0 |
| zul | 1669 | 1169 | 500 | afriscience_mt_train 1199, afriscience_mt_train_window 470 | apache-2.0 |

Sources are HUMAN translations only: AfriScience-MT train (professional translators; sentences for b64, 2-6 consecutive sentences of one paper joined to 65-250 source tokens for b256) and IbomMT train. **Left out and why:** MT560 / OPUS (web-mined and religious text, no row-level provenance), Pontoon / Weblate (UI strings), sudoping01 Wolof (provenance undocumented), the Bible (the 25 % cap would apply; none in release 1), MAFAND train (owner), **Vukuzenzele eval / test** (CC-BY-4.0, SFT read train only) because the corpus is **lower-cased on both sides** (0 of 4,000+ rows contain an upper-case letter): a case-sensitive chrF++ against it would reward lower-cased output (the builder supports it with `--vukuzenzele DIR`, off by default).
**Overlap with the SFT (caveat).** IbomMT train was in the SFT data (once, owner exception). AfriScience train was not a pair source of the A0 / stage-B packs (not among the `source_family` values of `data_pipeline/sft_clean/add_family.py`; the science SFT stage has not run), but overlap through other sources is possible and was NOT measured (the packed shards hold tokens only; measuring needs a decode + hash scan of the shards). **Coverage:** 8 of the 84 RL languages; none of the weak tail languages (kbp nus kea kam bam mos umb kmb ...) has a non-religious human source, so they get no anchor; with 15 % of the draws on 8 languages those get ~1.9 % each (B1: efi / ibb only 40 b256 rows each). Bucket: `<S3_BUCKET>/clean_rerun_2026-11-01/rl_prompts_ref/` (parquet per language + `stats_ref.json`); repo copy `gemma_mt/artifacts/rl_prompts_ref/`.

## 6. Beam-guided group member

* For `guide_fraction` of the prompts (a fixed-size random subset, function of (seed, step-key)), `generate_beam_completions` (HF `generate`, `num_beams=4`, no sampling, same prompt ids, stop ids and `max_new_tokens` as the sampling) runs on the **trainer's policy** (= the current policy: no weight-sync question) and the result **replaces the last sampled member** of the group (the group keeps its size 16: 14 samples, guide, greedy; `Sample.guide`). It is NOT an extra 17th row because the group tensors, the consensus grouping and the advantage code assume fixed-size consecutive groups; the group mean / std / advantage include it exactly as if it were extra.
* **Why HF and not vLLM beam search:** the vLLM worker protocol (`vllm_engine.py`) has no beam command, and CPU-only testing of a new engine op is not possible here; the HF path is exactly the current policy, needs no change of the engine and is covered by CPU tests. Cost = one beam search per guided prompt on the training GPU (4x the greedy cost for those prompts, sequential with the vLLM sampling); b64 ESTIMATED +5-10 % step time, b256 +15-25 %. **Measured on the run:** `guide/seconds`, `guide/overhead_frac` (share of the step) in every step record. Switch to a vLLM op later if it exceeds ~15 %.
* Off-policy handling: the beam output is not a sample of the policy. Its loss uses the log-probs of the **current policy** (teacher forcing, like every row) with the ratio fixed at 1 (`num_updates_per_batch: 1`; no importance weight): a REINFORCE term on an expert-style sequence with the group baseline. The sampler log-prob check (`vllm_check_every`) skips guide rows; the confidence rules skip them (`Sample.logp` stays None).
* **KL:** mirrors the greedy member: the loop has NO KL exclusion for the greedy member (its tokens are in the k3 term and in the logged `kl`), so the guide rows are treated the same way (verified in `train_rl.py` / `grpo_token_loss`: no per-row mask). Their share of the tokens is small (<= 0.25 x 1/16).
* Guards, penalties, the reference term and the per-language weight apply to the guide as to any member (it goes through the same reward call). **A guide with a hard failure is dropped from its group** (`grpo.group_advantages*(drop=...)`: left out of the mean / std, advantage 0; with <2 members left the group is flat). `guide_fraction > 0` requires `hard_fail_mask: off`.
* **Cap:** `guide_adv_share_cap` (0.25): share = sum over guide rows of |A| x length / the same over all rows (the weighting each token gets in the token loss; same as `greedy_adv_token_share`); above the cap the guides' advantages are scaled by the factor that brings it to the cap (`guide/adv_share`, `guide/adv_cap_factor`). With 25 % of the groups guided the uniform share is ~1.6 %, the cap is a safety net.
* **Logged** (step record): `guide/n`, `guide/failed_rate`, `guide/reward`, `guide/sampled_reward`, `guide/greedy_reward`, `guide/mbr_reward`, `guide/better_frac` (guide above the mean of the other members), `guide/best_frac` (guide is the best member; average over steps = share of groups), `guide/finished_rate`, `guide/adv_mean`, `guide/adv_share`, `guide/adv_cap_factor`, `guide/seconds`, `guide/overhead_frac`. **Per language**, running table `<out_dir>/guide_by_language.json` (every 10 steps): n, win_rate, best_rate, mean rewards of the guide / the **MBR pick among the group's own samples** (chrF++ agreement = the consensus utility; NOT added to the group, it already is a member) / the greedy member / the group mean, and how often each (guide, MBR, greedy) is the best member. Purpose: the decode study (greedy 35.00, beam-4 37.53, MBR-16 38.29) shows MBR beating beam mostly in weak languages (orm +7.3, ewe +4.6, lug +4.3): the table tells which language should get which guide.

## 7. Per-prompt group statistics (prompt replay by learnability: NOT implemented)

`group_stats.jsonl` has, per generated group, the prompt index, language, bin, domain, group std, group mean, number of hard failures and the guide flag. A replay of high-variance prompts can be built on it (e.g. a `PromptSchedule` that revisits prompts whose std was in the top decile); nothing in the schedule reads it yet.

## 8. How to apply (launch agent; the box tree `/workspace/clean_rerun/tree/gemma_mt` was byte-identical to the base on 2026-11-03)

Package on the box: `/workspace/clean_rerun/features_2026-11-03/` = `rl_features_dynamic_ref_2026-11-03.patch`, `BASE_MD5.txt`, `apply_features.sh`, `rl_prompts_ref/` (the pool), `FEATURES_READY` (sha256 of the patch).

```bash
cd /workspace/clean_rerun/features_2026-11-03 && bash apply_features.sh            # checks BASE_MD5, patch --dry-run, patch -p1, regenerates the yamls with FEATURES (default ds,adv,log,guide,ref), copies the pool to <prompts_path>/ref/
```

* **Resume**: stop the stage at a checkpoint boundary (`last/` is written every 20 steps), apply, start the same launcher command: the trainer resumes from `last/` (the new keys only change what happens from the next step). With `FEATURES=ds,adv,log,guide` (no `ref`) the yaml keeps the 70 / 30 general / news mixture and needs no pool; add `ref` later (copy the pool to `<prompts_path>/ref/`: without the files a yaml with `ref: 0.15` stops at start-up with "no prompt carries that domain label").
* First checks in the step records: `ds/flat_frac`, `ds/refill_rounds`, `ds/wasted_token_frac`; `guide/seconds`, `guide/overhead_frac`, `guide/better_frac`; `reward/ref/n` (> 0 only with the pool); `group_std/p10`; `finished_rate` (now on all generated groups). Then `python scripts/group_std_percentiles.py <run_dir> --by-language`.
* If `guide/overhead_frac` > 0.2: `guide_fraction: 0.15`. If `ds/short_batch` is frequent: raise `dynamic_sampling_oversample`.

## 9. Tests

* Laptop (no torch): `tests/test_dynamic_sampling.py` (12; 1 needs torch), `test_reference_reward.py` (7; 1 needs torch), `test_clean_rerun_385m.py` (yaml settings, `--features`): all pass.
* Box CPU (torch, isolated copy `/workspace/ds_test`, nothing of the running preflight touched): `test_adv_floor_guide.py` (std floor, drop mask, share cap, beam completions, loops with floor + group stats + guide + cap, loop with guide + greedy member + dynamic sampling + floor, off-by-default), `test_dynamic_ref_loop.py` (4), `test_rl_loop.py`, `test_grpo.py`, `test_greedy_in_group.py`, `test_hard_fail_mask.py`: all pass; `clean_rerun_tools.py validate --plan b256_385m --extra-stages F2`: all stage yamls load strictly, schedules replay, invariants hold (Stage A replay: ref 3,823 / general 15,296 / news 6,481 prompts of 25,600).
* Full suite on the box before the guide/floor work: 73 failed vs 76 failed on the original files, 1107 vs 1080 passed; the failures are environmental (no `reports/` / `artifacts/` in the isolated copy, missing sacrebleu, old tests), none new (`comm` of the failing sets: nothing only in the new run except a randomly named tmp path). The result of the final full run after all changes is in the hand-over message.
* NOT testable on CPU: wall-clock cost of the beam search on the A100, vLLM interplay of a mid-run patch (the engine is rebuilt by the restart), real QE scores of reference rows.

---

# Patch 2 (second request of 2026-11-03): MBR guide for the weak languages + early stop on the dev metrics

Signal: `/workspace/clean_rerun/FEATURES_READY_2`. Applies on top of patch 1 (the box tree was byte-identical to the patch-1 result when this patch was cut). Resumable like patch 1: new config keys with defaults, the checkpoint only gains an optional `state.dev_patience` (a checkpoint without it loads fine). Yaml additions only in **B1, mixed, final** (stage A had run); the preflight and A yamls are unchanged, F2 gets `early_stop_dev: false`.

## 11. MBR guide (D-74), `rl/guide.py`

| key | code default | B1 / mixed / final yaml |
|---|---|---|
| `guide_mode_by_language` | `{}` (all beam) | the 29 languages of the x2.5 weight list (tir dyu amh kbp fon nus mos kam bam umb kea kmb vmw kab seh yor wol ewe luo tum nyn ach kon pcm shi efi tiv bin ish) + **orm lug** (ewe and tir were in the list already) = 31 x `mbr`; all other languages use the beam guide |
| `guide_mbr_extra_samples` | 16 | 16 (0 = the MBR languages use the beam guide) |
| `guide_mbr_temperature` / `guide_mbr_top_p` | 0.7 / 0.95 | 0.7 / 0.95 |
| `guide_mbr_max_overhead_frac` | 0.20 | 0.20 |

* `guide_fraction` 0.25 (patch 1) picks the guided prompts; the guide of an MBR-language prompt is the **MBR winner over a larger pool**: the group's members (samples + greedy) + the beam-4 output + 16 extra samples from the vLLM sampler (T 0.7, top-p 0.95, for those prompts only; `sampler.generate`, HF path in the CPU tests), utility = mean sentence chrF++ against all other candidates (`consensus` utility). Only candidates that pass the cheap hard checks (finished, nothing after the end token, tag present, non-empty, not a source copy, not repetitive) take part.
* The winner replaces the last sampled member exactly like the beam guide (ratio 1, hard-failure drop from the group, `guide_adv_share_cap` unchanged). **If the winner is one of the group's own members** (it would only be a copy of a member) **the beam output stays the guide for that group** (`mbr_member->beam`); the cases are `Sample.guide_src` = `beam` | `mbr_beam` (the beam output won) | `mbr_extra` (an extra sample won) | `mbr_member->beam` | `mbr_few->beam` (< 3 valid candidates) | `mbr_skipped->beam` (budget).
* **Budget:** the running share of the step time spent in guides over the last 10 steps (beam + extra samples + MBR computation) is compared with `guide_mbr_max_overhead_frac` before each step; above it the extra samples are skipped for that step (`guide/mbr_skipped`), the MBR languages fall back to the beam guide. The beam guides themselves are not throttled (lower `guide_fraction` if `guide/running_overhead` stays high).
* **Cost.** Extra generation: 16 samples x (0.25 x prompts of MBR languages) per step through vLLM (batched with nothing else: one extra sampler call per step), e.g. b256 stage: 42 prompts x 0.25 = ~10 guided prompts of which ~half MBR languages: ~80 extra completions per step, ESTIMATED <10 % of the 15 x 42 = 630 regular completions' time. The chrF++ matrix (numpy, equal to the reference implementation pair by pair, tested): **33 candidates x 40 prompts = 0.2 s (15-word texts) / 0.5 s (60 words) / 1.2 s (150 words)** on the laptop CPU, MEASURED (`tests/test_guide_mbr.py`); the real per-step overhead is logged: `guide/seconds`, `guide/beam_seconds`, `guide/extra_seconds`, `guide/mbr_compute_seconds`, `guide/overhead_frac`, `guide/running_overhead`.
* **Logged per language** (`<out_dir>/guide_by_language.json`, every 10 steps): `mode` (mbr / beam), `src` (counts of the cases above), `candidates_mean`, `reward_guide` (the winner's reward) vs `reward_group_mean` / `reward_greedy` / `reward_mbr` (MBR pick among the group's own samples, patch 1), `win_rate`, `best_rate`, `best_rate_guide|mbr|greedy`. Compare `reward_guide` against `reward_group_mean` per language to see where the MBR guide helps.

## 12. Early stop on the dev metrics (D-73), `run_controls.DevPatience`

| key | code default | B1 / mixed / final yaml |
|---|---|---|
| `early_stop_dev` | false | true (F2: false) |
| `early_stop_dev_k` | 3 | 3 (consecutive DEV evaluations without improvement) |
| `early_stop_dev_eps` | 0.15 | 0.15 chrF++ |
| `early_stop_dev_min_steps` | 75 | 75 |
| `early_stop_dev_min_finished` | 0.98 | 0.98 (dev finished rate AND the greedy-evaluation finished rate of the stage's bin) |
| `early_stop_dev_winddown` | 10 | 10 |
| `early_stop_dev_aux` | `weak_dev` | `weak_dev` (the aux set whose `chrf_macro` is the second metric; absent: macro only) |
| `benchmark_eval_every` | (existing) | **25** (was 50): K counts DEV evaluations, so k = 3 is 75 steps only with a dev evaluation every 25 steps |

Rule: an anchor per metric (dev macro = the `benchmark_chrf` that selects checkpoints; weak_dev) moves only on a gain >= eps over the anchor; `stale` = consecutive dev evaluations with no anchor move; fire when stale >= k, steps in stage >= 75, finished rates >= 0.98, last stage of the run, and no KL-guard event inside the stale window (from the evaluation before the k stale ones); **no learning-rate condition**, an older guard event does not block. Then the same WINDDOWN as `plateau_stop` (`controls.request_winddown(10, "dev_plateau")`, last/ saved, latest + best pushed, `summary.stop_reason = "dev_plateau"`, `summary.early_stop_dev`, `dev_plateau_checks.jsonl`, state in the checkpoint). Cost of the finer cadence: one dev evaluation (720 + weak 60 x ~19 sentences, greedy) per 25 steps instead of per 50.

**Stage A replayed (the real series, `tests/test_dev_patience.py`).** dev macro / weak_dev / dev finished / greedy-eval finished: 50: 39.04 / 23.92 / .968 / .914; 100: 40.14 / 26.02 / .988 / .945; 150: 40.13 / 26.28 / .989 / .961; 200: 40.47 / 26.45 / .993 / .977; 250: 40.43 / 26.80 / .997 / .984; 300: 40.46 / 26.82 / .997 / .984; 350: 40.47 / 26.81 / .999 / .984; 400: 40.50 / 26.72 / .997 / .992. The macro improved by >= 0.15 at 100 and 200, the weak set at 100, 150, 200, 250; from 250 on neither did. **The rule as specified (k 3, eps 0.15) therefore fires at 400 only at the stage-A cadence of 50 steps** (stale 1 @300, 2 @350, 3 @400); k = 2 would fire at 350; the three KL-guard cuts of steps 226-247 lie inside the windows of any earlier candidate and block them (`guard_event`). So the rule is conservative on this series; with the 25-step cadence of the new yamls k = 3 is a 75-step window: IF the unobserved points between the 50-step evaluations are as flat as the observed ones, the rule would fire 75 steps after the last improvement (250), i.e. at ~325 (the guard cuts of 226-247 are older than that window). If the owner wants it to catch "flat from ~100" it needs a larger eps (the 100 -> 400 macro gain is +0.36, the weak gain +0.7: eps 0.4 with k 2 fires at 350, without the guard rule at 300) or a different measure; the dev-set sd (0.09-0.10 macro) limits how small eps can be.

**Did the training-reward `early_stop` ever fire or could it have?** No. Stage A (`early_stop_checks.jsonl`, 15 checks every 25 steps): 5 x `min_steps`, 10 x `improving`; it tracks three series (smoothed training reward, finished rate, greedy loop rate) and a new best of ANY resets the counter: reward made new bests at 251 and 345 (+0.02 each), the greedy loop rate at 287, the finished rate at 214. The longest stale stretch in the per-step record was **57 steps (at step 344)**, i.e. below the stage-A patience of 150 and also below the B1 patience of 80, so it would not have fired with patience 80 either. `plateau_stop` was 'not evaluated' at the 375 check (`lr_low`: the guard cuts at steps 226 / 236 / 247 brought the lr scale to 0.125).

## 13. Apply (second package)

`/workspace/clean_rerun/features_2026-11-03/apply_features_2.sh` (patch `rl_features_2_mbr_devstop_2026-11-03.patch`, `BASE_MD5_2.txt`): checks the base, patches, regenerates the yamls with `FEATURES` (default `ds,adv,log,guide,ref,mbr,devstop`; the stage-A yaml is not changed by the new features), validates. Restart from a 20-step checkpoint as before. First checks after the restart: `guide/extra_seconds`, `guide/running_overhead`, `guide/mbr_skipped`, `guide_by_language.json` (`src` counts of the MBR languages), and `dev check @ N` lines in the log (stale evals k/3).

Tests: `tests/test_guide_mbr.py` (9), `tests/test_dev_patience.py` (11, incl. the real stage-A series and the loop stopping through WINDDOWN with `stop_reason dev_plateau`), `test_clean_rerun_385m.py` (yaml settings), plus patch 1's suites: all pass on the box CPU; the plan validator loads all stage yamls strictly.

---

# Patch 3 (third request of 2026-11-03): held-out AfriCOMET monitor, reward-hacking alarm, combined dev stop, per-language dev history, dev predictions

Signals: `/workspace/clean_rerun/FEATURES_READY_3` (full patch, for the MIXED stage) and `FEATURES_READY_3A` (dev-prediction saving only, for B1). Applies on top of patch 2 (the box tree was byte-identical to the patch-2 result when this was cut; B1 was running at step ~40). Defaults OFF in code; ON in `rl_C_mixed` and `rl_F_final`; `save_dev_predictions` also in `rl_B1_b256`; F2 and stage A / the preflight unchanged or off. Decision log D-75 to D-77.

## 14. What it does (keys; code default -> mixed / final yaml)

| feature | keys |
|---|---|
| held-out AfriCOMET (D-75) | `heldout_comet` false -> true; `heldout_comet_model` `masakhane/africomet-stl-1.1`; `heldout_comet_batch_size` 64; `heldout_comet_keep_loaded` false; `heldout_comet_bootstrap` 200 |
| hacking alarm (D-76) | `hacking_alarm` false -> true; `_k` 3; `_eps_chrf` 0.3; `_eps_comet` 0.005; `_winddown` 10 |
| combined stop (D-76) | `early_stop_dev_require_comet` false -> true; `early_stop_dev_eps_comet` 0.004 |
| per-language history (D-77) | `lang_stall_log` false -> true; `lang_stall_k` 3; `lang_stall_eps_chrf` 0.5; `lang_stall_eps_comet` 0.01 |
| dev predictions (D-77) | `save_dev_predictions` false -> true in B1 / mixed / final |

1. **Held-out AfriCOMET.** Hub id verified with the Hub API: `masakhane/africomet-stl-1.1` (apache-2.0, `checkpoints/model.ckpt` 2.32 GB) = the reference-based AfriCOMET-STL (`rl/comet.py` `REF_MODEL_ID`). It is a different checkpoint from `masakhane/africomet-qe-stl-1.1` and `McGill-NLP/ssa-comet-mtl` that the policy trains on. At every dev evaluation the greedy dev translations of the bench set and the aux sets (`weak_dev`) are scored against the human dev references for the languages in `QE_LANGUAGES` that the dev set contains. Logged: `bench/comet_macro`, `bench/comet/<iso>`, `bench/comet_n/<iso>` (n per language, ~40-60), `bench/comet_macro_sd` (bootstrap sd of the macro over sentences; ~0.005 for 19 languages x 45 sentences, tested), `bench_aux/weak_dev/comet_macro|comet/<iso>|...`, `heldout_comet/seconds|load_seconds|score_seconds|n|languages`, and in the eval print line. **Where it runs:** a separate comet worker (`WorkerScorer(reference_based=True)`, same python / pythonpath as the QE scorer spec), not the existing QE worker: `--reference-based` is a flag of the whole worker process and the QE checkpoints are reference-free. It is started for the evaluation and released afterwards (`keep_loaded: false`). A failure (download, OOM, timeout) is logged, counted in `summary.heldout_comet.failures`, and the evaluation continues without it; the alarm and the combined stop then do not apply (patch-2 behaviour).
2. **Hacking alarm.** Over the last k + 1 = 4 dev evaluations of the stage: hold-out QE reward (`eval/qe_mean_scored`) has a positive least-squares slope AND dev chrF++ macro fell by > 0.3 AND held-out AfriCOMET macro fell by > 0.005 (first to last) -> `controls.request_winddown(10, "hacking_alarm")`, `stop_reason hacking_alarm`, `hacking_alarm_checks.jsonl`. `best/` and the Hub best (by the existing best-selection = dev chrF++) are untouched: continue the chain from the best checkpoint (`HANDOFF=best`), not from `final/` (which is the degraded one). Noise: from the stage-A series the dev macro moved by <= 0.04 between consecutive evaluations after step 250 and the evaluation sd is ~0.1 (yaml note: 0.09-0.10), so a 3-evaluation window difference has sd ~0.14 and 0.3 is ~2 sd; the alarm also needs the AfriCOMET fall and a rising QE, so false alarms need three independent coincidences (the unit test replays the real stage-A tail: it does not fire). For AfriCOMET there is no series yet: the macro sd by sentence sampling is ~0.005 (logged), the evaluation-to-evaluation noise on the same sentences is smaller; 0.005 is a first value to re-set with the first mixed series.
3. **Combined stop.** With `early_stop_dev_require_comet` the AfriCOMET macro has an anchor (eps_comet 0.004) and an improvement of it resets the stale count of the patch-2 rule: a `dev_plateau` stop needs the chrF++ metrics AND the held-out AfriCOMET flat over the same k dev evaluations. An evaluation without an AfriCOMET value does not block (patch-2 rule).
4. **Per-language history.** `dev_by_language.jsonl` (one line per dev evaluation: per-language chrF++ / AfriCOMET / n of the bench set and the aux sets, stale counts, `stalled` list) and `dev_stalled/<iso>`, `dev_stalled_count` in the record. A language is stalled after k = 3 dev evaluations without a gain >= 0.5 chrF++ or >= 0.01 AfriCOMET over its anchor. Logging only; no weight changes.
5. **Dev predictions.** NOT saved by stage A or by B1 so far (the run dirs hold no dev dumps; `archive.sqlite` holds training prompts' best translations only): the dev chrF++ numbers of `metrics.jsonl` are all that exists, so **stage A cannot be re-scored retroactively with the AfriCOMET** (only its `best/`, `final/`, `last/` weights exist, not the per-evaluation translations). From now on `save_dev_predictions` writes `<out_dir>/dev_predictions/step_<N>.jsonl` (set, iso, id, src, hyp, ref, finished, step; ~1,900 rows, ~0.5 MB per evaluation). `scripts/dev_comet_retro.py RUN_DIR` scores those files offline with the reference-based AfriCOMET and replays the patch-2 dev patience, the combined stop and the hacking alarm on the series (with the hold-out QE and finished rates of `metrics.jsonl`), printing the step at which each would have stopped; without saved predictions it says so.

## 15. Cost

* **GPU memory (ESTIMATED, not run: the GPU belongs to B1):** the extra worker is one ~560M-parameter encoder: fp16 weights 1.1 GB + CUDA context 0.5-1 GB + activations at batch 64 (~1 GB) = **2.3-3 GB** while loaded (the QE pair measures 2.7 GB resident / 6.7 GB peak at batch 128, `reports/rl_gpu_ram_385m_2026-11-01.md`); the box had ~10 GB free during B1 (69.6 of 81.9 GB used). Released after each evaluation (`keep_loaded: false`).
* **Time (ESTIMATED):** ~1,900 pairs per dev evaluation (bench 720 + weak_dev ~1,140): model load 15-30 s (page-cached checkpoint) + scoring ~10 s = **30-45 s per dev evaluation** (+ the bootstrap, < 0.1 s), i.e. +4-6 % of a 25-step block at ~12 min per block; resident (`keep_loaded: true`) ~10-15 s. The run logs `heldout_comet/seconds|load_seconds|score_seconds`.
* **Disk — needs action before the mixed stage.** The checkpoint is **2.32 GB** in the HF cache (`HF_HOME=/workspace/clean_sft/hf_home`, from `rl_env.sh`). The box disk (`/workspace`, 68 GB overlay) had 5.6 GB free (92 %). `apply_features_3.sh` pre-downloads it only if >= 6 GiB are free, otherwise it warns and the first dev evaluation downloads it into whatever is left (a failure is harmless but loses the first AfriCOMET point). Free >= 3 GB first (candidates, not touched by me: `/workspace/eval_385m` 13 GB, `/workspace/.hf_home` 12 GB, `runs_385m/A` 1.5 GB). The run's `last/` checkpoint lives in `/dev/shm` (50 GB free), so it is not what fills the disk. **Incident:** while testing the apply script I let it pre-download the checkpoint into `/workspace/.hf_home` of my shell for ~2 minutes (free space 3.4 GB), then deleted it; free space is back at 5.6 GB. The free-space check is now exact (KiB).

## 16. Apply

`/workspace/clean_rerun/features_2026-11-03/apply_features_3.sh`:
* **B1, same boundary as patch 2 (optional, tiny): `MODE=devpred bash apply_features_3.sh`** applies only `rl_features_3a_devpred_2026-11-03.patch` (train_rl.py: the `save_dev_predictions` key and the file writing; no yaml change). Then pass `save_dev_predictions=true` as a key=value override of the trainer command (like the other overrides). It changes nothing else (`tests/test_dev_predictions.py`: identical rewards with and without). Without it, the first dev predictions exist only from the mixed stage on.
* **Mixed stage: `bash apply_features_3.sh`** applies the full patch (if 3a is in the tree it applies the 3b difference), regenerates the yamls (default `FEATURES=ds,adv,log,guide,ref,mbr,devstop,devpred,monitor`; the monitor lines only enter rl_C_mixed / rl_F_final), pre-downloads the checkpoint if there is room, validates. Tested on copies of the live tree: 3a then 3b and 3 directly give identical code and yamls; 37 + 106 tests pass.
* First checks: `held-out AfriCOMET macro ...` in the eval print line, `heldout_comet/seconds`, `dev check @ N` (now with `comet`), `dev_by_language.jsonl`, `dev_predictions/`.

Tests: `tests/test_dev_monitor.py` (13: alarm cases incl. the real stage-A tail, combined stop, language stall, monitor scoring / failure / bootstrap sd, a CPU loop with fake worker, predictions, alarm checks, language history), `tests/test_dev_predictions.py` (1), `tests/test_dev_comet_retro.py` (2), `tests/test_dev_patience.py` (11), `tests/test_clean_rerun_385m.py` (yaml settings). NOT run: the real checkpoint (no disk / GPU headroom on the box), so the exact load and scoring time and the real macro sd are unmeasured.

## 17. Mixed-stage settings decided by the owner (2026-11-03, applied by the launch agent)

* Start model: **B1 best** (`HANDOFF=best`, step 50, dev chrF++ 40.743), not B1 final (step 135, 40.56). B1 ended by `dev_plateau` at step 135 (dev chrF++ 40.74 / 40.74 / 40.72 / 40.48 at steps 50 / 75 / 100 / 125 against 40.50 at the SFT start; weak_dev about 27.0). Gate PASS_WITH_WARNINGS.
* Patch 3 (full) applied to the tree before the mixed launch (held-out AfriCOMET `masakhane/africomet-stl-1.1`, hacking alarm, combined stop, per-language history, dev predictions).
* Combined dev stop: `early_stop_dev_k` 2 (50 steps without improvement = 2 dev evaluations at `benchmark_eval_every` 25), `early_stop_dev_require_comet` true, `early_stop_dev_eps_comet` 0.004 (unchanged), `early_stop_dev_min_steps` 75, `early_stop_dev_min_finished` 0.98.
* Hacking alarm loosened: `hacking_alarm_eps_chrf` 0.3 -> **0.75** (owner asked 0.5-1.0); `hacking_alarm_eps_comet` 0.005 -> **0.03** (the owner first wrote 0.1-0.5, which on the 0-1 AfriCOMET scale would never fire; the owner then confirmed the range 0.02-0.05 and 0.03 was set; an intermediate value 0.02 was in the yaml for a few minutes before the first mixed restart). `hacking_alarm_k` 3 and `hacking_alarm_winddown` 10 unchanged. Applied through a clean restart of the mixed stage at its first (step-20) checkpoint (`logs/pool_swap_mixed.log`); the other settings of the mixed yaml: `dynamic_sampling_oversample` 1.0, `kl_guard_threshold` 0.25, `guide_fraction` 0.25 (B1 guide overhead median 36%, below the 40% bar for 0.15), ref domain on (general 0.595 / news 0.255 / ref 0.15, 7 reference languages), the 81-language pool without the flagged windows.
* B1 facts relevant to the guide: the MBR guide's extra samples were skipped on 99% of the B1 steps after patch 2 (running guide overhead 0.27-0.47 against `guide_mbr_max_overhead_frac` 0.20), so B1 effectively ran the beam guide only; B1 step time median 109 s at b256.
* Launch incident: the mixed stage was auto-started at 18:43:25 UTC from B1/final with the patch-2 yaml (by a process outside the launch agent) while the post-B1 evaluation was starting; it had done no step and was stopped at 18:44 (STOP file kept as `STOP.stopped_by_launch_agent_1843`) and relaunched at 18:59:54 from B1/best with patch 3.
* Disk: `/workspace/.hf_home/hub/models--masakhane--africomet-stl-1.1` (2.2 GB) was created by `apply_features_3.sh` (its default HF_HOME) at 18:51 although the run's cache `/workspace/clean_sft/hf_home` already holds the same snapshot; the delete was refused by the permission layer, so the duplicate is still there (safe to remove). Hub: the repo still reports `private`, so the B1 final push (step 135) and the `latest`/`stage-b1-final` branches fail with 403 "Private storage limit reached for org <HF_ORG>"; B1 best (step 50) is on the Hub as `stage-b1-b256`, B1 final exists only in `/dev/shm/rl_B1_ckpt/final` and `runs_385m/B1/hub_stage_latest`.
