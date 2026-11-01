# Final science + general GRPO run: integrated tree, start state, launcher, SFT-stage result, shakedown (D-59, agent R7, 2026-10-21/22)

Status: **tree, yaml, launcher, start-state script and tests are final for the CURRENT reward (R1 ensemble + R2 constraint penalties + R3 lexicon). The real run was NOT started.** The lead has since delayed the launch for R8 (long-document reward v2), R9 (anchors v2), R10 (extra weak-language data), R11 (SFT mixture v2), R12 (sampling study); those are not merged here. Nothing was pushed to the Hub or the bucket. Data/evidence: `reports/final_run_integration_2026-10-21/` (also under `gemma_mt/reports/`).

## 1. What was merged, from where

* **Run tree `/workspace/jobs/r7/gemma_mt_run`** (= repo `gemma_mt/`, mirrored; not committed). Base `/workspace/sft_build/gemma_mt_2c`. I found that `/workspace/jobs/b1/gemma_mt`, `/workspace/jobs/b2/gemma_mt` and the laptop `gemma_mt/` were already one consolidated tree (md5-identical code, differing from 2c only in the expected files: `rl/reward.py`, `rl/train_rl.py`, `rl/data.py`, `rl/comet.py`, `rl/comet_worker.py`, `train_sft.py`, `rl/confidence.py` (newlang, not used by the trainer) plus the new modules `qe_ensemble.py`, `joint_budget.py`, `anchors*.py`, `run_controls.py`, `sci_anchor.py`, `lexicon.py`). Verification instead of re-merging: R2's two diffs reverse-apply cleanly onto it, R1's patch hunks are present, A4's code is present, and the suites of all four (`test_qe_ensemble` 31, `test_anchors`, `test_run_controls`, `test_science_run`, `test_greedy_in_group`) pass in the tree. Base for the tree = `jobs/b2/gemma_mt`; R3's final `lexicon.py`, `build_lexicons.py`, `validate_lexicons.py`, `tests/test_lexicon.py` (16 tests) and the 44 `lexicon/*.json` come from the laptop (md5-identical to `/workspace/jobs/b3/lexicon_final`).
* **Stable assets `/workspace/jobs/r7/assets`** (171 MB + 11 MB lexicons; nothing depends on other agents' folders): `rl_prompts_sci_v1/` (77 parquet; 60,901 science + 46,021 general replay rows), `calibration/{shaping_flores_dev_v7,v8,anchors_v1,anchors_v1_lex,science_anchor_v1}.json`, `lexicon/` (44 files, only 10 `validated=true`), `dev/{bench_afrisci_dev_sent,bench_afrisci_dev_doc,bench_dev_clean,bench_weak_flores_dev}.jsonl`, `afs_raw/` (AfriScience corpus + published chrF csv). Outside the assets dir and still needed: `/workspace/sft_build/runs/final_1433` (KL reference), `runs/rl2c/last/trainer_state.pt` (only to rebuild the start state), the HF caches in `/workspace/sft_build/hf` (GlotLID, AfriCOMET, Afro-XLM-R, and **`McGill-NLP/ssa-comet-mtl`, which I downloaded there (2.2 GB): R1's copy lived in b1's HF cache, which had been deleted**), `venv_gemma`, `venv_vllm`, `pylibs*`, `.hf_write_token`.
* **New in this tree (all default-off in code, on in the yaml):** (1) language FLOOR in `PromptSchedule` (`language_floor_steps/prompts`); (2) lexicon wiring (R3 via R2's hook): `anchors_lexicon_per_language/weight_cap/guard/dir`, hook returns the density-guarded overlap, per-language weight = min(0.25, validated weight); `scripts/add_lexicon_calibration.py`; (3) `greedy_adv_scale`; (4) `scripts/prepare_initial_state.py`, `scripts/launch_final.sh`, `scripts/language_coverage.py`, `configs/rl_science_final.yaml`; `dry_check_rl.py` understands the ensemble scorer, the floor and the lexicon; (5) tests `test_language_floor` (5), `test_prepare_initial_state` (1, end to end: prepared state -> trainer resumes at step 0 with the moments continuing), lexicon/greedy additions to `test_anchors`/`test_greedy_in_group`.

## 2. Tests

Full suite on the box, CPU: **866 passed, 11 skipped, 0 failed** (3 min). The 5 known failures are resolved, not just documented: `test_lexicon` x4 (R3's newer test + `lexicon.py` from the laptop), `test_b2048_bin::test_staged_config...` (the staged yaml legitimately carries `kl_estimator: k3c` since 2026-10-20; the test now allows exactly that difference), `test_stratified::test_pusher_is_disabled...` (the test now also hides the token files `/workspace/sft_build/.hf_write_token` / `env.sh` that the pusher re-reads). `dry_check_rl.py configs/rl_science_final.yaml --real-lid` -> DRY CHECK OK. `launch_final.sh check` -> preflight OK. Supervisor semantics (HOLD waits, GO consumes HOLD+GO, STOP stops restarts but never kills a trainer, second supervisor refused, WINDDOWN/LR_SCALE helpers) tested with a dummy trainer.

## 3. SFT stage (R4, D-58): result and start model

Run on the shared card as the GPU allowed (`MIN_FREE_GIB` lowered to 10-15 GiB for the 6.5 GiB trainer; nothing pushed). Baseline `final_1433` re-evaluated with R4's harness; `train` at lr 3e-6 (293 steps, 17 min): AfriScience dev loss 1.767 -> 1.204 (fell by 0.56 nats, so no 5e-6 rerun), replay val macro 1.426 -> 1.261, drift 0.003 %. Sweep (dev, greedy):

| model | sent. chrF++ | paper chrF | paper finished | general dev ALL | weak-dev blend (19) | weak-dev consensus (13) | control (7) | weak-dev ALL (63) |
|---|---|---|---|---|---|---|---|---|
| final_1433 | 45.65 | 36.00 | 74.7 % | 40.96 | 27.20 | 23.21 | 42.87 | 32.31 |
| alpha 0.3 | 47.06 | 42.79 | 88.0 % | +0.19 | **-0.38 [-0.60,-0.16]** | +0.22 | +0.73 | +0.07 |
| alpha 0.5 | 47.75 | 45.04 | 89.3 % | +0.26 | **-0.63 [-0.92,-0.34]** | +0.12 | +0.51 | -0.02 |
| alpha 0.7 | 48.09 | 46.21 | 92.0 % | +0.16 | **-1.04 [-1.39,-0.68]** | -0.14 | +0.60 | -0.21 |
| alpha 1.0 | 47.88 | 45.41 | 89.3 % | -0.37 | **-2.34 [-2.93,-1.75]** | -1.19 | -0.59 | -1.31 |
| L2-SP 0.3, alpha 0.5 | 47.27 | 43.56 | 90.0 % | +0.23 | **-0.53 [-0.82,-0.25]** | +0.02 | +0.52 | -0.04 |
| L2-SP 0.3, alpha 1.0 | 47.51 | 46.44 | 91.3 % | -0.21 | **-2.07 [-2.52,-1.58]** | -1.02 | +0.15 | -1.01 |

(deltas against final_1433, 95 % bootstrap over languages; full tables `sft_stage_sweep_tables.md`, `sft_stage_l2sp_sweep_tables.md`.) The SFT gives large science gains (paper-level chrF +7..+10, finished 75 % -> 88-92 %, sentence +1.4..+2.4 chrF++) and no forgetting on the general dev, but **the weak-blend languages (the ones the RL improved with its own consensus signal) lose 0.4-2.3 chrF++ at every alpha, and L2-SP does not fix it** (as R4 feared: replay of human references pulls back RL gains). R4's rule (every weak group not lower by more than 0.5 and not significantly below 0) fails for every alpha, plain and L2-SP -> **start model = `final_1433` (alpha 0).** Alternative if the owner values the paper-level gain more: alpha 0.3 (`run_sci_sft_stage.sh finalize 0.3`, then `prepare_initial_state.py --start-model models/sci_sft_alpha_0.3 --force`; `model_name` stays final_1433). The checkpoint `/workspace/jobs/r4/runs/sci_sft/last` (1.1 GB, mix_v1) is kept for that. **R11's mix_v2 READY marker does not exist yet: the SFT stage was NOT repeated; if R11 delivers, re-run `train`/`sweep` on it (30 min + 10 min per alpha) and re-apply the rule.**

## 4. Start state (built and verified)

`/workspace/jobs/r7/runs/final/last/trainer_state.pt` (3.2 GB, PREPARED.json beside it): weights = final_1433 (so identical to the 2c policy, relative L2 0.0), the 2c Adam moments (236 tensors, step counts 1433, all finite), 2c RNG, trainer step 0, best_select -inf, bench_path None (-> the trainer scores the start policy on the AfriScience dev first and that is the bar for selection). KL reference and engine start = `model_name: runs/final_1433`. Verified by loading into a fresh model + AdamW and by the end-to-end CPU test. To rebuild (or switch start model):
```
cd /workspace/jobs/r7/gemma_mt_run && source ../env.sh && CUDA_VISIBLE_DEVICES= $PY scripts/prepare_initial_state.py --start-model /workspace/sft_build/runs/final_1433 --optimizer-from /workspace/sft_build/runs/rl2c/last/trainer_state.pt --out-dir /workspace/jobs/r7/runs/final --verify --force
```
(bug found on the way: cloning the state dict un-tied embed/lm_head and made the file 0.7 GB bigger; fixed.)

## 5. Config `configs/rl_science_final.yaml` (derived from rl_science.yaml; key choices)

* Reward: `qe_ensemble: true` (delta 1.0, calibration **v8**, `scorer.ensemble_model_id: McGill-NLP/ssa-comet-mtl`, batch 128), consensus kept, anchors on (entity restricted to 4 languages by the calibration, repetition + boilerplate on all 45), **lexicon on for the 10 validated languages** (`sag kam fon kab kmb kea umb vmw mos kbp`; weights 0.25 .. 0.08 = min(0.25, R3's weight), stuffing guard on; the other 31 lexicons and dyu/nus/bam have no detector), `science_anchor_penalty 0.5`, `word_ratio_penalty 0.5`, k3c, `reliability_from: same_source`.
* **Lexicon calibration caveat:** the 3 % human quantile of the lexicon overlap is 0 for all 10 languages (9-32 % of human FLORES+ references have no covered hit), so R2's rule would disable the detector. I fitted the threshold at the human MEDIAN instead (`anchors_v1_lex.json`, `add_lexicon_calibration.py --alpha 0.5`): deficit = (median - overlap)/median below the median, weight <= 0.25. It is therefore a graded, small term (mean penalty of a human reference 0.02-0.08, of the final model's output 0.03-0.11, highest for kam/kea/kbp); in the shakedown it judged 15 % of chunks and cost 0.0017 reward on average. This is the "coarse recall signal with small weight" R3 describes; set `anchors_lexicon_weight: 0` to remove it.
* Data/schedule: 70/30 science/general, bins as D-52 (science b1024/b2048 70 %), **MIXED from step 0** (`curriculum_steps: {b64: 0}`, step counter restarts at 0), 42 prompts x group 16 per step, `greedy_in_group: true`, language weights: ish bin bbj tsc kon nus dag oke 3.0 (lead, 2026-10-22), the other 17 of the 18 weakest 2.5, hau lug nso zul 2.0.
* **Language floor** `language_floor_steps: 20, language_floor_prompts: 2`: every one of the 77 languages gets >= 2 prompts in every aligned block of 20 steps (the random draw gives an ordinary language only 0.36 prompts per step, so ~50 % of steps lack it; a hard per-step floor is impossible with 42 prompts for 77 languages). 18 % of the slots are forced, the rest drawn as before. Tested (`test_language_floor`), checked by `dry_check_rl.py` (min per block 3-4).
* **LR 1.0e-6** (lower end of 1e-6..2e-6), warmup 20. Evidence: 2c ran 945 steps at 2e-6 with mean KL plateauing at 0.02-0.03 and grad norm 0.65. On the real state the carried-over Adam moments see coherent gradients: at 1.5e-6 (no ramp) the mean KL was 0.0058 / 0.0090 / 0.0137 / 0.0184 at steps 4 / 5 / 7 / 8 and still rising ~0.002/step (2c needed 150 steps for 0.02), so the same lr moves the policy ~3x faster than in 2c. KL guard 0.15 (halves the lr, min 1/8) kept as the safety net; `LR_SCALE` is the live lever.
* **`greedy_adv_scale: 0.1` (new, my addition; 1.0 = plain recipe).** Measured on the real start state, 4-8 steps each, same batches: greedy member at full weight -> pre-clip grad norm **8-19** (clip 1.0 then crushes the signal of the 15 sampled members, which alone give 0.5); no greedy member -> 0.51-0.56; greedy advantage x0.1 -> 0.9-1.7. The greedy completion of a looping long document is one sequence of up to 7,168 tokens that all push the same way. The scale keeps the anti-loop signal (the greedy-in-group effect was visible already: paper finished rate 0.773 -> 0.860 after 10 steps) at a weight comparable to the rest. Cost of the greedy member: ~35-50 s per step.
* Monitoring/selection as D-52 (selection by AfriScience dev sentence chrF++, `select_min_gain` 0.2, reset at start): `benchmark_aux` = general_dev, weak_dev, afrisci_dev_doc (`every: 100`), cadence 50 steps; `eval_every 25`, save every 50, Hub: private `<HF_ORG_LEGACY>/gemma-science-translate`, main = best, branch `latest` every 25 steps, token re-read from `/workspace/sft_build/.hf_write_token` at every push (token present; **the push path itself was not exercised, no pushes allowed**). Archive `/workspace/jobs/r7/runs/final/archive/archive.sqlite`.
* **Steps = 800** (below).

## 6. Expected samples per language (800 steps x 42 prompts x 16 = 537,600 sampled completions; simulation of the real sampler, `language_coverage_800.json`)

| group | languages | prompts per language (min / mean / max) | sampled completions (mean) |
|---|---|---|---|
| weight 3.0 | ish bin bbj tsc kon nus dag oke | 694 / 745 / 775 | ~11,900 |
| weight 2.5 | 17 (tir amh kbp kam bam mos umb kmb dyu kea luo kik orm fon yor vmw efi) | 602 / 639 / 685 | ~10,200 |
| weight 2.0 | hau lug nso zul | 497 / 544 / 606 | ~8,700 |
| weight 1.0 | the other 48 (incl. very-low-resource ones such as kri, wes, bas, dua, kua, her) | **275** / 304 / 333 | ~4,900 (min 4,400) |

~70 % of every language's prompts are science. Without the floor the minimum is 0 prompts in some 20-step blocks; with it every language has >= 2 prompts (32 completions) in every 20 steps, i.e. >= 1,280 per language over the run, plus the random share. Status of the weights vs R10's later recommendation: replace by theirs in `language_weights` and re-run `scripts/language_coverage.py`.

## 7. Shakedown (real GPU, real config on the real start state; scratch dir, no Hub, no live archive)

15 steps in three phases (`shakedown_metrics.jsonl`; 10 steps, kill, supervisor restart + resume at step 10 with `LR_SCALE 0.5`, `WINDDOWN 3`, normal finish at step 15) plus three 4-8 step probes (the greedy/LR evidence above). Config of the shakedown = the final yaml of that moment (lr 1.5e-6 with the 20-step warm-up, greedy_adv_scale 1.0, weights before the 3.0 change) with eval/save every 10. Results:

* Start: resumed from the prepared state at step 0, "selection reset", baseline on the new monitors **46.47 / 32.39 (paper, finished 0.773) / 41.88 / 25.34** (AfriScience dev sentences / paper level / general dev / weak dev) = R2's and A4's numbers for final_1433 (so state, engine and monitors are right). Start-up 6.5 min (vLLM 43 s, 2-model QE worker 75 s, baselines).
* Ensemble: `reward/qe_ens/min_share` 0.095-0.182, mean **0.137** (expected 0.10-0.15), abs_diff 0.54, joint_split_rate 0, both models usable 100 %; QE seconds per step: AfriCOMET ~9.5 + SSA ~13 = ~22 s. Anchors: applied to 57 % of samples, mean penalty 0.044, per-language `anchors/penalty/<lang>` logged, repetition penalised 1-3 %, boilerplate silent (buffer warm-up, as designed). Lexicon: judged 15 % of chunks, penalised 86 % of those (deficit 0.43), mean penalty 0.0017. Fail classes: degenerate 3 %, no_end_of_turn 1 %, wrong_language 0.06 %.
* **Step time (dedicated card, steps 2-15): mean 205 s, median 203 s (range 170-244)**; components: generation 20 s (sampled), sampling+reward+greedy 100 s, training ~100 s. With greedy 35-50 s dearer than without (probe: 139-180 s). Eval cycle (reward eval 192 prompts + dev + all monitors incl. paper level) 111 s; real cadence adds ~4 s/step. **Estimate 209 s/step -> 48 h = 827 steps; set 800 (46.4 h + 7 min start-up, 1.4 h margin).** The lead's ~1,280 would be 74 h. Time will fall as greedy loops (paper finished rate 0.773 -> 0.86 after 10 steps) disappear; extend later with `steps=N` and the same launch command.
* GPU memory (nvidia-smi, process level): trainer 42 GB, vLLM 20 GB (0.20 of the card), QE worker 3-7 GB: **65 GB total**; launcher default preflight requires 70 GiB free. Disk: state 3.2 GB (+3.2 GB tmp while saving) + hub staging 1.1 GB.
* Controls: `LR_SCALE 0.5` -> lr 1.5e-6 x 11/20 x 0.5 = 4.13e-7 exactly as logged; `WINDDOWN 3` -> logged "decays linearly over 3 steps from step 12; the run ends at step 15", final eval, save, `summary.json`, file renamed `WINDDOWN.done`, supervisor logged "RL finished" and removed its pidfile. Checkpoint save at step 10, kill (rc 143), supervisor restart after 20 s, "resumed RL from step 10", same bench path -> no second baseline. Monitor sets all wrote metrics (`bench_aux/{afrisci_dev_doc,general_dev,weak_dev}/{chrf_macro,bleu_macro,finished_rate,seconds}`, weak per-language chrF for 13 languages, `eval/reward_domain/*`, `eval/reward_bin/*`). Abort rules never triggered (KL guard cut 0 times, no NaN, no reward collapse).
* Bugs found and fixed by it: the supervisor passed an empty `''` argument to the trainer when no extra args were given (every start crashed with "override '' must look like key=value").
* NOT verified: the Hub push, the first 20 steps with a ramp to full lr 1e-6 (the KL trajectory beyond step 15 is unknown; at 1.0e-6 I expect it below the 1.5e-6 probe), R8/R9 features.

## 8. Launch

```
ssh -o BatchMode=yes -i $SSH_KEY -p $BOX_PORT root@$BOX_HOST
cd /workspace/jobs/r7/gemma_mt_run
bash scripts/launch_final.sh check          # preflight: state, assets, disk >= 8 GiB, GPU >= 70 GiB free, yaml loads
bash scripts/launch_final.sh start          # setsid nohup supervisor (OOM ladder 24576/128 .. 3072/16, 8 starts/h limit); extra key=value (e.g. steps=900) go to the trainer
bash scripts/launch_final.sh status         # pid, control files, last metrics, disk, GPU
#   log /workspace/jobs/r7/logs/rl_FINAL.log, launcher log logs/launch_FINAL.log, metrics /workspace/jobs/r7/runs/final/metrics.jsonl
bash scripts/launch_final.sh winddown 50    # graceful end (lr -> 0 over 50 steps, final eval/save/push)
bash scripts/launch_final.sh lr_scale 0.5   # live multiplier;   hold | go | stop = touch HOLD | GO | STOP (checked only between trainer processes; a running trainer is never killed)
```
The old training tab must be gone before `start` (GPU preflight).

## 9. Recommended WINDDOWN criteria (all from `metrics.jsonl`; evaluated every 50 steps)

1. **Plateau:** `bench/chrf_macro` (AfriScience dev sentences; start 46.47, noise +-0.4) has not beaten its best by >= 0.2 for 6 consecutive evaluations (300 steps), and the paper-level `bench_aux/afrisci_dev_doc/chrf_macro` (start 32.39) is flat or down over the same window -> `winddown 50`.
2. **Forgetting (act at once, first `lr_scale 0.5`, winddown if it repeats):** `bench_aux/general_dev/chrf_macro` more than 1.0 below its step-0 value 41.88 in 2 consecutive evaluations; or `bench_aux/weak_dev/chrf_macro` more than 1.5 below 25.34 (2 consecutive) or its `finished_rate` < 0.98; or `bench/finished_rate` < 0.97.
3. **Stability:** 5-step mean `kl` > 0.10 for 10 steps -> `lr_scale 0.5`; `kl_guard_lr_scale` <= 0.25 or any NaN/inf -> `winddown`; pre-clip `grad_norm` > 5 for 20 steps (expected ~1 with greedy_adv_scale 0.1) -> look at the greedy members (`eval/finished_rate`, repetition).
4. **Reward hacking:** `reward/qe_ens/min_share` > 0.30 for 20 steps or `min_share_ssa_high` > 0.15 (expected 0.14 / 0.07), `reward/anchors/penalised_rate` rising monotonically for 100 steps, `reward/fail/degenerate` > 0.10.
5. **No progress in training reward:** 100-step mean of `reward` rises < 0.05 over 300 steps while criteria 2-4 are quiet -> winddown (the policy has stopped learning, the remaining time only adds drift).

## 10. Risks

1. KL/stability beyond step 15 unobserved; coherent Adam updates move the policy ~3x faster than in 2c (hence 1.0e-6, `greedy_adv_scale`, guard 0.15, LR_SCALE).
2. `greedy_adv_scale: 0.1` is my deviation from the plain "greedy true" recipe; 1.0 reproduces it (grad norm 10-17, clip dominates).
3. The SFT stage was not adopted (weak-blend regression); the science benefit it showed (paper level +7..+10 chrF) is left to the RL to find. R11's v2 mixture might change that.
4. Step time: 205 s measured with the current reward; R8 (tail-sensitive aggregation, more penalties) and R9 will add CPU time; re-measure with a 20-step shakedown after merging.
5. Lexicon term thresholded at the human median (coarse; small weight); can be switched off.
6. Hub push untested; disk is tight (11 GB free at the end; a save needs 3.2 GB of tmp + 1.1 GB staging).
7. The boilerplate penalty is silent for the first ~40 steps after every restart (buffer in memory).
8. Weak-dev monitor covers 13 of the 45 weak languages with 60 sentences (+-3 chrF per language): it finds collapses, not +0.5 changes.

## 11. Disk: state and cleanup candidates (not touched unless stated)

Deleted (mine): shakedown/probe out_dirs (3 x 3.2 GB hard links + the shakedown checkpoint 3.0 GB), the L2-SP checkpoint (1.1 GB), supervisor test dirs, CPU test scratch; `/workspace` now 11 GB free (was 5-8 GB). Candidates for the lead / owners (sizes): `/workspace/sft_build/runs/sci` 3.0 GB (A4's byte-identical copy of `runs/rl2c/last`; the new run does not use it), `/workspace/sft_build/gemma/rl_prompts_v2` 3.2 GB (superseded by v3 and sci_v1; check that 2c resume does not need it), `/workspace/sft_build/runs/rl_2b_stage_b64` 1.1 GB (old stage model), `/workspace/jobs/a2` 6.9 GB (A2 generation data/models), `/workspace/.hf_home` 1.6 GB (unknown owner), `/workspace/jobs/a1` 0.8 GB, `/workspace/jobs/a4` 0.4 GB (its prompts/dev files are copied into r7/assets; its code is not needed), `/workspace/jobs/b3` 0.36 GB and `/workspace/jobs/b1,b2` 0.13 GB (merged, finished), `/workspace/jobs/r4` 1.5 GB (keep `runs/sci_sft/last` 1.1 GB if alpha-0.3 or a re-run is wanted), `/workspace/datagen/models` 22 GB (HOLD, not touched). Do NOT delete: `runs/final_1433`, `runs/rl2c/last`, `/workspace/jobs/r7`.

---
# ADDENDUM 2026-10-22: SFT stage round 2 (R11 mixtures v2), chosen start model, pushes

All numbers: AfriScience-MT dev (6 languages), general dev (19), weak-language dev (FLORES+ dev 60 sentences / pool 60, 63 languages) with R4's unchanged harness (same code md5 as before, so the earlier `final_1433` baseline was reused: 45.65 sentence chrF++, 36.00 paper chrF, 74.7 % paper finished). Tables: `reports/final_run_integration_2026-10-21/sft_round2/` (`guardrails_*.md`, `compare_*.md`, `SFT_STATUS.txt`). Trainer = R11's configs, R4's hyper-parameters (peak lr 3e-6, 65,536 tokens/step, cosine, 6 % warm-up), start `final_1433`, one epoch; four runs, concurrently on the shared card.

| run (mixture) | steps | val replay macro (start 1.4263) | val_sci loss (start 1.7669) |
|---|---|---|---|
| A (mix_v2A) | 330 | 1.2656 | 1.2034 |
| B_v1 (A + R10 commercial-OK extras) | 362 | 1.2662 | 1.1984 |
| A2W (A2 + unfiltered self-distilled anchors + weak human replay x0.5) | 318 | 1.2811 | 1.2037 |
| B_v2_ibom (B_v2 + IbomMT train; NC) | 514 | 1.2738 | 1.1881 |

The science dev loss fell by 0.56-0.58 nats in all four (rule: rerun at 5e-6 only below 0.15), so no 5e-6 rerun. Guardrail rule (lead): best sentence dev chrF++ with general-dev macro, each weak group and (science6, consensus, blend, control, other) within 0.5 of final_1433 and paper finished rate not lower.

| model | sent. chrF++ | paper chrF | paper fin. % | general ALL d | weak-blend d [95 % CI] | weak-consensus d | guardrails |
|---|---|---|---|---|---|---|---|
| final_1433 | 45.65 | 36.00 | 74.7 | 0 | 0 | 0 | - |
| A a0.3 / 0.5 / 0.7 / 1.0 | 47.04 / 47.67 / 48.09 / 48.01 | 42.36 / 46.35 / 47.34 / 49.04 | 90.7 / 94.0 / 92.0 / 96.7 | +0.26 / +0.27 / +0.21 / -0.16 | -0.25 / **-0.50 [-0.76,-0.26]** / -0.72 / -1.49 | +0.20 / +0.21 / +0.10 / -0.46 | PASS / FAIL(blend) / FAIL / FAIL |
| B_v1 a0.3 / 0.5 / 0.7 / 1.0 | 46.98 / 47.63 / 48.02 / 47.82 | 40.79 / 45.17 / 45.98 / 46.12 | 82.7 / 88.0 / 89.3 / 82.0 | +0.11 / +0.28 / +0.11 / -0.21 | -0.25 / -0.50 / -0.93 / -2.06 | +0.10 / -0.04 / -0.33 / -1.23 | PASS / PASS (borderline) / FAIL / FAIL |
| **A2W a0.3 / 0.5 / 0.7 / 1.0** | 47.05 / 47.65 / **47.97** / 47.96 | 42.98 / 46.10 / **47.81** / 48.87 | 85.3 / 90.0 / **92.0** / 93.3 | +0.20 / +0.28 / **+0.18** / -0.07 | -0.23 / -0.36 / **-0.36 [-0.58,-0.16]** / -0.85 | +0.22 / +0.21 / **+0.38** / +0.06 | PASS / PASS / **PASS** / FAIL(blend) |
| B_v2_ibom a0.3 / 0.5 / 0.7 / 1.0 | 47.01 / 47.63 / 47.93 / 47.61 | 42.43 / 43.83 / 44.43 / 44.21 | 82.7 / 80.7 / 78.7 / 72.0 | +0.24 / +0.20 / +0.13 / -0.19 | -0.22 / -0.38 / -0.99 / -2.41 | +0.12 / -0.16 / -0.39 / -1.73 | PASS / PASS / FAIL / FAIL |

Answer to the lead's question: **yes, the new data/regime fixes most of the mix_v1 weak-blend regression**: weak-blend delta at alpha 0.7 went from -1.04 (mix_v1) to -0.36 (A2W; halving the weak-language human replay + self-distilled anchors) and at alpha 1.0 from -2.34 to -0.85 (A2W) / -1.49 (A); R10's extras alone (B_v1) barely changed it (-2.06 at 1.0). The IbomMT variant does not beat A2W. Paper-level quality is where all of them gain: +7 to +13 chrF and finished 75 % -> 90-97 %.

**Chosen by the rule: A2W alpha 0.7** (best sentence chrF++ among the passing models: 47.97 vs 47.63 B_v1 a0.5, 47.63 B_v2_ibom a0.5, 47.04 A a0.3). Its tables against final_1433 (95 % CI over languages):
* AfriScience dev, per language sentence chrF++ (base -> chosen): amh 23.63 -> 24.68, hau 59.26 -> 62.02, lug 42.55 -> 45.36, nso 55.98 -> 59.53, yor 37.03 -> 39.15, zul 55.45 -> 57.06; paper level chrF++: amh 20.51 -> 22.75, hau 46.16 -> 61.21, lug 27.77 -> 38.25, nso 36.86 -> 54.56, yor 20.92 -> 33.71, zul 40.32 -> 54.31; paper finished %: amh 92 -> 92, hau 76 -> 100, lug 48 -> 84, nso 88 -> 100, yor 60 -> 76, zul 84 -> 100.
* **AfriScience TEST (read once, chosen model and baseline): sentence chrF++ 44.90 -> 47.13, paper chrF 33.05 -> 45.76, paper finished 72 % -> 94 %.**
* General dev (19): +0.18 [+0.04, +0.32]; weak dev groups: science6 +0.85 [+0.58, +1.10], consensus +0.38 [-0.15, +1.09], blend -0.36 [-0.58, -0.16], control +0.42 [-0.02, +0.92], other -0.06 [-0.30, +0.19], ALL +0.08 [-0.11, +0.29].
* Ge'ez / weak languages (weak dev, chrF++): amh sentence +1.05 (dev), paper +2.24; tir 13.60 -> 13.73; ibb 21.57 -> 21.43; efi 28.99 -> 28.63; nus 20.05 -> 19.83; kon 16.69 -> 16.57; dag 25.83 -> 26.19; kri 24.63 -> 24.97; tsc 18.61 -> 22.15 (full per-language table: `compare_A2W_a0.7.md`).
* **IbomMT TEST (FLORES+ devtest English, 1,012 sentences each; R11's audit: 0 of the 329 IbomMT train windows hit it, exact sentence or word 8-gram; the dev side was already excluded by R10), chrF++ ibb / efi:** final_1433 **22.27 / 31.27**; A2W a0.7 22.07 / 30.79 (this mixture has no IbomMT train); B_v2_ibom a0.5 21.92 / 30.66; a0.7 21.58 / 30.02; a1.0 20.69 / 27.55. So the 329 IbomMT train windows brought **no** test gain for ibb/efi in this setting, and the SFT replay lowers efi. If the owner still wants the IbomMT variant, B_v2_ibom a0.5 passes the guardrails (47.63) but has lower paper finished (80.7 %) and costs 0.3 chrF++ on the sentence level versus A2W a0.7; B_v2_ibom is NC and carries the owner's code-switch subset.

**Outputs**
* HF model dir (fp32, tokenizer + chat template + generation_config from final_1433): `/workspace/jobs/r7/models/sci_sft_A2W_alpha0.7`.
* **RL start state rebuilt** from it: `/workspace/jobs/r7/runs/final/last/trainer_state.pt` (verified; relative L2 distance to the 2c policy 2.4e-5; step 0; 2c Adam moments; KL reference unchanged `runs/final_1433`). Revert: `prepare_initial_state.py --start-model /workspace/sft_build/runs/final_1433 ... --force`. NOTE: the shakedown/probe numbers of the main report were measured from the final_1433 start; the first RL steps from this start have not been run.
* Kept (1.0 GB each): `/workspace/jobs/r7/runs/sft_A2W/last`, `/workspace/jobs/r7/runs/sft_B_v2_ibom/last`. Deleted (lost the comparison, numbers recorded): `sft_A/last`, `sft_B_v1/last`, `/workspace/jobs/r4/runs/sci_sft/last` (mix_v1). Freed earlier on authority: `jobs/a2/models/rl2c` (sha-identical to final_1433), `jobs/a2/models/africomet_ref` (sha-identical to the HF cache copy), `sft_build/gemma/rl_prompts_v2` (in the bucket as `data/gemma_rl_prompts_v2.tar.zst`; only default paths of old scripts mention it). `.hf_home` is the live HF_HOME of instance services and was left alone. Free disk at the end: 5.2 GB.
* New code: `scripts/sft_stage_eval.py` suite `ibom` (IbomMT test), `scripts/sft_guardrails.py` (the rule above), `scripts/push_sft_variant.py`, `prepare_initial_state.py` disk check (needs state size + 1 GiB, not 2x).

**Hub pushes: NOT DONE.** The owner-authorised push (via your message) was refused by the permission classifier ("External System Writes") when I ran `push_sft_variant.py`, and I did not work around it. Nothing was uploaded; no commit shas to report. The script is ready and dry-run-safe: it stages `final_1433 + run weights` at the chosen alpha (1.1 GB), writes a model card (variant, mixture token table from R11's `meta.json`, hyper-parameters, steps, start model, validation numbers from the training log, optional `--notes` file with the guardrail results, licence note, date), uploads with `upload_folder(path_in_repo="sft/<folder>", allow_patterns=model files)` and no delete patterns to main, creates the repo private if needed, appends repo/folder/sha to SFT_STATUS.txt. Commands (run by someone with the permission, e.g. after an explicit allow rule; the A2W run dir is kept; A2W is in the main repo, B_v2_ibom must go to the -acs repo):
```
cd /workspace/jobs/r7/gemma_mt_run && source ../env.sh && export CUDA_VISIBLE_DEVICES=
$PY scripts/push_sft_variant.py --variant A2W --mix mix_v2A2W --repo <HF_ORG_LEGACY>/gemma-science-translate --folder A2W_alpha0.7 --alpha 0.7 --run /workspace/jobs/r7/runs/sft_A2W --log /workspace/jobs/r7/logs/sft_A2W.log --config /workspace/jobs/r11/configs/sft_science_v2_A2W.yaml --notes /workspace/jobs/r7/eval/guardrails_A2W.md
$PY scripts/push_sft_variant.py ... --folder A2W --alpha 1.0     # the raw trained model (same command with alpha 1.0)
$PY scripts/push_sft_variant.py --variant B_v2_ibom --mix mix_v2B_v2_ibom --repo <HF_ORG_LEGACY>/gemma-science-translate-acs --folder B_v2_ibom_alpha0.5 --alpha 0.5 --run /workspace/jobs/r7/runs/sft_B_v2_ibom --log /workspace/jobs/r7/logs/sft_B_v2_ibom.log --config /workspace/jobs/r11/configs/sft_science_v2_B_v2_ibom.yaml --notes /workspace/jobs/r7/eval/guardrails_B_v2_ibom.md
```
(variants A, B_v1 and their weights were deleted before the authorisation arrived, to free disk; they can be retrained in ~35-60 min each if wanted.)
