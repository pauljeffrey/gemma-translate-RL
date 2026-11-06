# E-11: RL component ablations, b64 recipe, 100 steps (2026-11-05)

Executor: the E-02/E-04/E-11/E-14 agent. Scripts as in `reports/e04_rl_seeds_2026-11-05.md` (same `e04_cfg.py` / `e04_run.sh` / `e_eval20.sh` / `ecompare.py`). Section 0 is the pre-registration, written before the first ablation run existed and not edited afterwards.

## 0. Pre-registration (written 2026-11-05 05:30 UTC)

**Question (M10).** Stage A added four features at step 41 together (dynamic sampling, advantage std floor 0.02, beam-guided group member, reference domain) and the KL slope rose when they started. Which of them, if any, helps? Without ablations the paper must call them design choices, not contributions. This experiment is limited to the b64 recipe: stage A, 100 steps, 64 prompts x 16 samples, one run per arm.

**Arms** (all start from the SFT final with the executed stage-A recipe; see E-04 for the full recipe):
| arm | steps 1-40 | steps 41-100 | switch |
|---|---|---|---|
| (a) no-feature baseline | features off | features off | `base` config for all 100 steps |
| (b) full feature set | features off | features on (executed stage A) | **not rerun**: = the E-04 seed runs s2 and s3 (and stage A seed 1 at step 100, dev metrics only) |
| (c) no reference-anchored reward | features off | full minus the reference domain: domain weights general 0.7 / news 0.3 (the steps 1-40 mixture), `reward.reference_weight` 0 | removes both the 15% reference prompts and the chrF++-against-human-reference term, the feature as it was shipped |
| (d) no beam guide | features off | full with `guide_fraction` 0 and `guide_adv_share_cap` 0 | |
| (e) KL guard 0.25 | features off | full with `kl_guard_threshold` 0.25 | the executed stage A used 0.15, so arm "0.15" IS arm (b) |
All four switches exist as config keys (nothing had to be implemented). Dynamic sampling and the std floor are not ablated separately (stage A showed flat-group fraction 0 and group-std p10 0.128 against the floor 0.02: they never acted); the base arm (a) switches them off together with the guide and the reference domain.

**Design: shared 40-step prefix (a deviation from "one independent run per arm", made to save about 2.5 GPU-h and to make the comparison tighter).** Steps 1-40 are identical in every arm (features off). The E-04 run with seed 20261105 (s2) saves its step-40 state (`last/` incl. optimiser, RNG, KL-guard state, plus metrics up to step 40). Arms (a), (c), (d), (e) are forks of that state: they start at step 41 with their own config and the same seed. So an arm differs from s2 only through its switch after step 40 (sampling noise of vLLM still makes the trajectories diverge); s3 is an independent full run. Consequences stated in advance: (i) arm minus s2 is a tight paired comparison, arm minus s3 includes the full seed noise; (ii) the ablation effects are compared with the *full* E-04 seed spread (s2 vs s3), which is conservative for the forks; (iii) all arms share one realisation of steps 1-40, so an unlucky/lucky prefix affects all arms and s2 alike.

**Arm (e) rule (fixed now).** The KL guard (threshold 0.15) acts only if the 5-step median KL exceeds 0.15; in stage A the median KL at step 100 was about 0.12 and the first cut was at step 226. Arm (e) is run **only if** the guard acts (a `kl_guard_cut` record, or a 5-step median KL above 0.15) in s2 or s3 within 100 steps (`decide_e.py`); otherwise it is **not run** because the 0.15 and 0.25 guards are then identical in effect (no cut with either threshold) and the arm would be a repeat of (b). The evidence (max median KL of s2, s3, a, c, d) is printed in the report. The question the owner presumably had (the guard cutting the learning rate eight-fold after step 226 and never recovering) needs a run of at least 250 steps and is not testable in 100.

**Evaluation (identical for every arm and for the E-04 seeds):** as E-04 sections 1-3: in-training dev (720 sentences, steps 0/50/100; macro numbers; sentence-level paired comparison among the new runs from the saved dev predictions) and the standard harness on the step-100 model, 20 languages x FLORES+ devtest / NTREX / MAFAND, vLLM 0.32.0 bf16, greedy, 768 tokens, held-out AfriCOMET, plain chrF/chrF++, finished rate, loops with both detectors. **Pre-declared step 100** (final model); best-by-dev reported separately.

**Decision rules (fixed now).** For arm X: d_X = X - s2 (paired, identical sentences), also X - s3 and X - mean(s2, s3); chrF++ macro over the 20 languages, FLORES+ and the all-cells macro, and held-out AfriCOMET. An effect is **real** only if (1) the paired 95% interval of X - s2 excludes 0 for chrF++, (2) |point difference| exceeds the observed seed difference |s2 - s3| on the same metric, and (3) the held-out AfriCOMET difference does not have an interval excluding 0 in the opposite direction. A feature "helps" if removing it makes things real-ly worse; "hurts" if removing it makes things real-ly better; otherwise "not distinguishable from seed noise". The same reading is applied to the in-training dev macro (reference: the three-seed range) and to the finished/loop rates (reported, not part of the rule, because the loop flag is the reward's own guard). KL and step-time effects (the KL slope after step 41, the generation overhead of the guide) are read from `metrics.jsonl` and described, not tested.

**Reading rules for the paper.** SSA-COMET-QE and the QE members are reward family and are not used as evidence here; chrF/chrF++ and held-out AfriCOMET are. One run per arm: all ablation results are single-run results relative to the E-04 spread.

**Not done:** separate arms for dynamic sampling and the std floor (inactive in stage A), multi-seed arms, runs longer than 100 steps, the KL-guard question beyond step 100.

## 1. Execution log and results

**What ran (A100, one job at a time).** Fork parent = the E-04 seed-2 run (seed 20261105): its step-40 trainer state (weights, Muon state, RNG, KL-guard state; 43 metric records) was copied at 09:19 UTC. Arms started from that state with the seed unchanged: (a) `base` 12:08-12:44 (steps 41-100 with the features off, 31.5 s per step), (c) `noref` 13:03-13:57, (d) `noguide` 14:18-15:00; each followed by the standard harness evaluation (about 15 min). Arm (e): **not run** (rule of section 0: the KL guard did not act in s2 or s3: max 5-step median KL 0.1256 and 0.1204 against the threshold 0.15, 0 cuts; also not in a, c, d: 0.1198, 0.1267, 0.1239; so a 0.25 threshold changes nothing within 100 steps). Weights of the three arms (step 100 = final; no separate best) were pushed and verified (sha256 against the box copy) to the public `<HF_ORG>/gemma-0.4B-translate-rl-experiments`, branches `e04-a_base-step100`, `e04-c_noref-step100`, `e04-d_noguide-step100`, box copies deleted. Run configs: `cfg_*_phase1.yaml` (the arm's config is the file of the run), logs, metrics, dev predictions, per-sentence harness outputs and tables are in `research_archive_2026-11-03/box_final_2026-11-04/e11/` (sha256 checked) and `reports/data/e11_rl_ablations_2026-11-05/`. Code state hashes in each `runs_*/code_state.txt`. Heartbeats every 8 minutes: no restarts in any arm, root disk 5.3-5.7 GB free, /dev/shm 39-42 GB free.

**Data used:** as E-04 (no TranslatePsy; NLLB-3.3B pseudo-references only as a reward signal for kea kbp nus bam kab). Arm (c) and (a) drop the reference domain (AfriScience-MT train / IbomMT train pairs) from steps 41 on; arm (d) and (a) drop the beam guide (the policy's own beam-4 output as a group member).

### Results at the pre-declared step 100 (20 languages, 49 cells, standard harness, identical sentences; 95% paired-bootstrap intervals)

| run | chrF++ all cells | FLORES+ | NTREX | MAFAND | AfriCOMET (held out) | finished | loop (reward flag) | loop (independent) | length ratio |
|---|---|---|---|---|---|---|---|---|---|
| (b) full, s2 (seed 20261105; the fork parent) | 38.51 [38.30, 38.69] | 38.46 | 37.59 | 39.89 | 0.6312 | 0.9896 | 0.0122 | 0.0279 | 1.131 |
| (b) full, s3 (seed 20261106) | 38.46 [38.26, 38.64] | 38.40 | 37.48 | 39.93 | 0.6304 | 0.9891 | 0.0133 | 0.0289 | 1.138 |
| (a) no features | 38.19 [37.99, 38.38] | 38.09 | 37.34 | 39.57 | 0.6285 | 0.9872 | 0.0151 | 0.0331 | 1.165 |
| (c) no reference domain | 38.35 [38.16, 38.54] | 38.31 | 37.47 | 39.67 | 0.6296 | 0.9889 | 0.0137 | 0.0295 | 1.144 |
| (d) no beam guide | 38.42 [38.23, 38.61] | 38.28 | 37.53 | 39.92 | 0.6308 | 0.9891 | 0.0131 | 0.0290 | 1.140 |

| arm minus (b) | chrF++ vs s2 (paired, same prefix and seed) | chrF++ vs s3 (independent run) | AfriCOMET vs s2 | AfriCOMET vs s3 | finished vs s2 | loop (reward) vs s2 | dev chrF++ (720 sentences, step 100) vs s2 |
|---|---|---|---|---|---|---|---|
| **(a) no features** | **-0.32 [-0.41, -0.23]** | -0.26 [-0.35, -0.18] | -0.0028 [-0.0033, -0.0022] | -0.0019 [-0.0025, -0.0014] | -0.0024 [-0.0034, -0.0014] | +0.0030 [+0.0020, +0.0041] | -0.14 [-0.51, +0.25] |
| **(c) no reference domain** | **-0.16 [-0.25, -0.08]** | -0.10 [-0.19, -0.02] | -0.0017 [-0.0023, -0.0011] | -0.0008 [-0.0014, -0.0003] | -0.0007 [-0.0016, +0.0002] | +0.0016 [+0.0006, +0.0025] | -0.23 [-0.64, +0.19] |
| **(d) no beam guide** | -0.09 [-0.17, -0.01] | -0.04 [-0.11, +0.04] | -0.0005 [-0.0010, +0.0001] | +0.0004 [-0.0002, +0.0010] | -0.0005 [-0.0014, +0.0004] | +0.0009 [0.0000, +0.0019] | -0.07 [-0.41, +0.30] |
| seed spread s2 - s3 (E-04) | +0.06 [-0.02, +0.14] | | +0.0008 | | | | |

Per benchmark vs s2 (chrF++): (a) FLORES+ -0.38 [-0.54, -0.23], NTREX -0.26, MAFAND -0.31; (c) FLORES+ -0.15 [-0.32, -0.01], NTREX -0.13, MAFAND -0.21; (d) FLORES+ -0.19 [-0.33, -0.06], NTREX -0.07 [-0.17, +0.04], MAFAND +0.03 [-0.12, +0.18]. Dev macro at step 100 (arms vs s2, same dev sentences): 40.14, 40.04, 40.21 against 40.28; weak_dev 25.93, 25.93, 26.08 against 26.14; none differs significantly.

**Training behaviour (`summ_*.json`).** Median step time over the 100 steps: (a) 31.5 s, (d) 32.4 s, s2 40.3 s, s3 40.7 s, (c) 43.2 s (c runs with the beam guide; its slower median than s2/s3 is not explained by any logged quantity, the guide overhead logged by the trainer is 27-29% of the step in s2, s3 and c): the beam guide costs about 8 to 9 s per step (+25 to 28%). **KL**: median KL of steps 81-90 is 0.116 (a, no features), 0.122 (c), 0.120 (d), 0.123 (s2), 0.120 (s3); of steps 51-60 0.075 / 0.0745 / 0.073 / 0.072 / 0.073. The no-feature arm has the same KL trajectory.

## 2. Verdict (pre-registered rule: real = paired interval excludes 0 AND |difference| > observed seed difference 0.06 AND no opposite AfriCOMET interval)

* **The four features together help a little: +0.32 chrF++ [+0.23, +0.41] (+0.0028 AfriCOMET) over 60 steps** (the no-feature arm is below s2 by that much and below s3 by 0.26), real by the rule, about five times the observed seed difference, and consistent on all three benchmarks and on AfriCOMET and the loop flag (+0.003). It is 5% of the 5.96 chrF++ gain and does not show on the in-training dev macro (-0.14 [-0.51, +0.25]).
* **Reference domain (c): -0.16 [-0.25, -0.08] when removed** (vs s3 -0.10 [-0.19, -0.02]); real by the rule against s2 (2.7x the seed difference), AfriCOMET -0.0017 in the same direction: a small positive contribution, about half of the feature set's.
* **Beam guide (d): -0.09 [-0.17, -0.01] when removed vs s2 but -0.04 [-0.11, +0.04] vs the independent run s3, AfriCOMET difference inside noise**: formally it passes the rule against s2 (point 0.09 > 0.06, interval just excludes 0) but not against s3 and not on AfriCOMET; the honest reading is "not distinguishable from seed noise; if anything at most +0.1 chrF++", for a cost of about +25% step time. Dynamic sampling and the std floor were never active (flat fraction 0, group-std p10 0.128 against the floor 0.02) and are part of arm (a) only.
* The sum of the single removals (c + d = -0.25) is close to the all-off arm (-0.32). The registered comparison with the KL slope (M10): **the KL rise is not caused by the features**: the no-feature arm reaches the same KL (0.116 against 0.123 at steps 81-90); the earlier inference that the beam guide drives the KL slope (STA, ledger M10) is not supported by this run (the rise from about 0.03 to 0.12 over steps 1-90 is there without any feature; the step-41 acceleration seen in stage A is therefore at most a small effect).
* Arm (e) not run (rule of section 0; the KL guard threshold is irrelevant within 100 steps).
* Reading for the paper: the four features are design choices with a **small** benefit at 100 steps (+0.3 chrF++ together, mostly the reference-anchored term), not the source of the RL gain (+5.9), and the beam guide is not shown to be worth its 25% generation time. All effects are single-run, at step 100, 20 languages; arms (a), (c), (d) share steps 1-40 with s2 (so arm minus s2 is the tight paired comparison; arm minus s3 carries the full seed noise and gives the same signs, weaker intervals).

## 3. Caveats / not done
One run per arm; shared 40-step prefix with s2; the seed spread (0.06) comes from one pair of seeds, so the "real" rule is lenient and the effects of 0.1 to 0.2 chrF++ should be read as small and suggestive; 100 steps only (the benchmark gain continues to grow to step 400: E-04); 20 languages (the reference domain covers amh efi hau lug nso yor zul, of which 5 are in these 20, the other languages benefit, if at all, by transfer); the dev macro at 100 steps cannot resolve effects of this size (interval about +-0.4). Arm (e), dynamic sampling and the std floor alone, and multi-seed arms were not run.
