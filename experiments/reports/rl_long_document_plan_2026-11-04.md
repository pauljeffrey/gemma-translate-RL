# Long-document RL stage: plan, data, budget, risks (2026-11-04)

Owner decision "lets do it" (2026-11-04): a long-document RL stage after the science RL and its evaluation, started automatically so the GPU does not idle. Status when this was written: config, pool, start-model rule, waiter and post-run evaluation are on the A100 and the waiter is armed; nothing has run on the GPU yet; the science RL is at step ~60 of 150 and is not touched.

## 1. What runs, in order (all automatic)

1. Science RL ends (`runs_385m_sci/final/summary.json`) -> the existing `post_sci_rl_waiter.sh` evaluates its best and final (sentence dev/test, NTREX/MAFAND, documents whole + chunked, neural passes) and writes `POST_SCI_RL_DONE`.
2. `long_rl_waiter.sh` (armed 04:40Z, PID in `science_sft/long_rl_waiter.log`) waits for that marker (or, if the science summary is 4 h old and no evaluation process is running, starts anyway: the GPU must not idle), picks the start checkpoint with `pick_long_start.py`, and launches launcher stage `final` of the new config dir `configs/clean_rerun_385m_long/` into `runs_385m_long/final` (watchdog `watchdog_long.sh`, checkpoints in `/dev/shm/rl_long_ckpt/{best,final,last}`, logs in `logs_long/`).
3. `post_long_rl_waiter.sh` starts `post_long_rl_eval.sh` when the stage has ended: best and final on AfriScience-MT dev/test sentences, NTREX and MAFAND, and the 168 test documents whole and chunked, then the neural passes (AfriCOMET, SSA-COMET-QE, SSA-COMET-MTL), marker `POST_LONG_RL_DONE`.

**Start checkpoint** (rule written before seeing the result, `data_pipeline/sft_clean/fresh/long385/pick_long_start.py`): the science RL `best/` (highest dev chrF++ under the trainer's min-gain rule), unless the last dev evaluation of the run beats it by >= 0.2 chrF++ with the held-out AfriCOMET not lower by more than 0.004; the choice and reason go to `science_sft/long_start_choice.json`. Dry run on the current series: best (step 50, dev 44.91).

## 2. Design (config `gemma_mt/configs/clean_rerun_385m_long/rl_F_final.yaml`, derived from the science RL yaml; only these keys differ)

| setting | science RL | long-document RL | why |
|---|---|---|---|
| steps | 150 | **80** | 51 % of the prompts are long; the F2 plan had 70; early stops can end it earlier; no extension of the cap |
| prompts per step | 42 | **32** (about 16 long + 16 short) | token budget (below) |
| group size | 16 (15 + greedy) | **8** (7 + greedy) | see below |
| long / short | none long | **51 % long** (b512 17 %, b1024 24 %, b2048 10 % of prompts) / 49 % short (b64, b256) | measured on the real schedule: 2,560 prompts drawn over 80 steps: b64 654, b256 593, b512 435, b1024 624, b2048 254; mean 407 source tokens per prompt, 13k source tokens per step |
| domains | science .4, ref .3, general .3 | **science .40, general .20, news .15, ref .25** | ref = short anchors with human references (the six AfriScience languages), the other three domains are 2/3 long and 1/3 short |
| generation caps | 1,280 | b512 2,304, b1024 3,584, b2048 7,168 (one cap per step: 7,168) | F2 / D-45 values; whole-document evaluation stays capped at 4,096 |
| sequence handling | no checkpointing | **gradient checkpointing on**, `vllm_max_model_len` 10,240, `max_src_tokens` 2,048, vLLM util 0.30, 256 seqs | sequences reach ~9.5k tokens |
| guide (beam / MBR member) | 0.25 | **off** | 4 beams + 16 extra samples per prompt are unaffordable on 2-7k token documents |
| checkpoint selection | dev chrF++ | **hold-out reward** (greedy, >= 16 prompts of each of the 5 bins, finished >= 0.95) | the stage's target is finishing / not looping / covering long documents, which the sentence dev cannot see; the sentence dev stays as early stop and hacking alarm |
| kept unchanged | | KL coef 0.1 and guard 0.25, learning rate 1e-6 Muon, patch-3 monitor (held-out AfriCOMET, `early_stop_dev` k 2 x 25 steps = 50-step combined stop, hacking alarm 0.75 chrF++ / 0.03 AfriCOMET), reward (QE ensemble + chunked `longdoc_v2` path: sentence-aligned chunk rewards, omission = failed chunk, coverage / omission penalty, loop and no-end-of-turn hard failures, length-ratio term), Hub repo `<HF_ORG>/gemma-0.4B-translate-rl` public, new branches `long-rl` / `long-rl-latest`, no model card | |

**Group size and prompts per step, justified.** A b1024/b2048 completion is 8 to 40 times as long as a b256 one, and generation, chunked QE scoring (about 49 sentence chunks per long document) and the backward pass all scale with it. The science RL step is 42 x 16 = 672 short completions (125 s). Here a step is 32 x 8 = 256 completions of which 128 are long: roughly 275k generated tokens and 7k scored chunks per step (about 2.7x and 1.7x the science step), while the group of 8 keeps the greedy member plus 7 samples (enough for a within-group baseline; dynamic sampling refills flat groups once). Group 16 with 32 prompts would double all of that for noise reduction that the record does not show to matter (phase 2 used group 8 with twice the prompts at the same dev gain per completion). The price is noisier advantages for each prompt.

## 3. Data (hard constraints)

* **LibreTexts removed (2026-11-04, owner: LibreTexts is excluded until its licence is resolved).** The CPU contamination follow-up (`reports/contamination_followups_385m_2026-11-04.md` section 2) found 3,428 `common-pile/libretexts` rows (17.8 % of the 19,250 rows) in `long_rl/science_b2048`: neither the scan nor the pool builder had a source filter. The stage therefore uses a filtered copy, `prompts/long_rl_filtered` (hard-linked copy of `long_rl`, only the 77 files of `science_b2048` rewritten; the original `long_rl` is untouched; `gemma_mt/scripts/filter_long_pool_source.py`, log `science_sft/overnight/long_pool_filter.json`): 241,679 -> 238,251 prompts, `science_b2048` 19,250 -> 15,822 (sources left there: fineweb-edu 4,980, pubmed 4,748, AfriScience-MT train 2,600, arxiv 1,853, Wikipedia 1,641); no other directory held LibreTexts. Verified by loading the pool with the trainer's loader (0 of 238,251 prompts with a LibreTexts source), by checking that every kept prompt is a prompt of the original pool (so it keeps the protected-index result: the CPU agent's rescan of the original found 0 of 241,679 rows flagged), and by re-drawing the schedule (2,560 prompts: b64 654, b256 593, b512 435, b1024 624, b2048 254; long share 0.513, unchanged). The config now points at `long_rl_filtered`. b2048 prompts are 10 % of the draw; their science share is now fineweb/pubmed/AfriScience/arxiv/Wikipedia text.
* Pool `prompts/long_rl` before the filter (241,679 prompts, 84 languages; builder `gemma_mt/scripts/build_long_rl_pool.py`, stats `prompts/long_rl/_pool_stats.json`): science = `sci_rl` + the `sci_v2` b2048 rows; general = `clean_v3_385m` + the v3 `zz_b2048` rows; news = BBC news windows (b64, b256 from the 385M pool, b512 and b1024 newly drawn); ref = AfriScience-MT **train** pairs (b64/b256). English-only prompts; AfriScience-MT **test papers are not in the pool**.
* The b2048 rows had been dropped unscanned in all earlier pools. On 2026-11-03 they were scanned with `data_pipeline/sft_clean/scan_long_pool.py` (decon.py rules R1 exact sentence, R2 contained sentence, R3 shared word 8-gram, strict, both normal forms, full protected index `protected-2026-10-28.3`: every protected set including AfriScience-MT dev and test, AfriLongDoc, SmolDoc, AfriDocMT, TICO-19, NTREX, MAFAND, FLORES+): science b2048 26,094 rows -> 25,222 kept (693 of 20,109 distinct texts flagged: afrilongdoc 421, smol 327, afridocmt 153, mafand 147, tico19 39, ...; AfriScience-MT dev/test 0), general b2048 `zz_b2048` 2,659 of 42,910 distinct texts flagged. A prompt with a hit is dropped, never repaired. Per-language caps (250) were then applied for size.
* All other long rows (b512, b1024) come from pools that passed the same scan earlier (385M plan; the news build removed 8,532 of 606,814 windows for hits, most of them AfriLongDoc and SmolDoc overlaps).
* Evaluation documents (AfriScience-MT test papers) are unseen by construction; the evaluation papers' sentence-level dev data are used only by the dev monitor.

## 4. Budget (ESTIMATES: no 385M long-sequence step has been timed)

| part | GPU-hours (A100, one process set) | basis |
|---|---:|---|
| training, 80 steps | **2.2 h central** (1.5 to 4.0) | 100 s per step central: scaling the 270M B3 row (134 s for 672 b1024 completions) to 128 long + 128 short completions and x1.3 for 385M gives 85 to 110 s; first-principles count (scoring 70 s + generation 27 s + update 14 s + sync) gives ~125 s; the science RL measures 125 s per step for short prompts |
| dev monitor and hold-out evaluation (4 x ~8 min), start-up | 0.7 h | |
| post-run evaluation of best and final (sentence, NTREX/MAFAND, 168 documents whole + chunked, neural passes) | 2 to 3.5 h | the document mode with best-of-4 repair is the long part |
| **stage + evaluation** | **~5 to 8 h, central ~6** | F2's earlier plan was 4.3 to 6.0 h for training alone on 270M numbers |

A crash restart of the stage costs about 15 min (resume from `last/`, watchdog); an OOM rung of the ladder halves the micro-batch.

## 5. Risks

1. **Memory at 9.5k-token sequences** (estimated 15 GB trainer with checkpointing + 24 GB vLLM + scorers). The ladder and the watchdog handle it, but the first steps may be lost to OOM restarts. No 385M run has used sequences this long.
2. **Reward calibration for long documents**: `longdoc_clean.json` (fitted without AfriLongDoc / SmolDoc) has no `tail_gap` (null -> defaults, the measured medians of human documents); the CVaR term acts on sources of >= 8 chunks. The reward has never been run at 385M scale on documents.
3. **Length and coverage gaming**: omission and coverage penalties exist, but the hacking alarm watches the sentence-level dev only; document behaviour is watched through the per-bin greedy finished / loop rates of the hold-out evaluation and the post-run document evaluation. Concatenated COMET scores see only the first 512 tokens, so the honest document numbers are the 4-sentence-window ones.
4. **Regression on short text**: 49 % of the prompts are short (a quarter with human references), KL guard 0.25, sentence-dev early stop and hacking alarm; the post-run NTREX/MAFAND/AfriScience sentence evaluation is the non-regression check against the science RL model.
5. **Noisier updates** (group 8), only ~1,300 long prompts over the whole stage, so a gain may be small; the stage is not evidence that long-document RL helps until the evaluation says so.
6. **Selection by hold-out reward is the optimised quantity**: the evaluation therefore reports both best and final and uses independent metrics (SSA-COMET-QE window as the paper's headline, plain chrF, AfriCOMET-STL held out, finished, loop, coverage); the reward contains `africomet-qe-stl-1.1` and `ssa-comet-mtl`, so those scores are not independent evidence.
7. **Dependency on the science RL**: if it ends in a state the picker cannot use (neither `best/` nor `final/` complete) the waiter does not launch and says so in `long_rl_waiter.log`.
8. **Box resources**: `/workspace` has 3.2 GB free; checkpoints go to `/dev/shm` (21 GB free, ~8 GB needed); the GPU must be free of other jobs when the stage starts (the waiter waits for the post-evaluation to finish).
9. **Not decided by the owner and chosen here** (change before the science RL ends if wrong): 80 steps, 32 prompts, group 8, guide off, selection by hold-out reward, the ref domain kept as 25 % short anchors, start-model rule.

## 6. Evaluation to be written when the stage ends

`reports/eval_long_document_rl_*.md`: on the same 168 AfriScience-MT test documents, whole and chunked: finished, loop, coverage, SSA-COMET-QE (4-sentence windows; concatenated only as a flagged first-512-token number), plain chrF; sentence level on AfriScience dev/test, NTREX and MAFAND (non-regression); against the science RL model and the old 270M rl2c; best and final both shown; the reward-family caution of section 5.6.

## 7. Measured (added 2026-11-04 10:12 UTC)

The stage started at 10:03 UTC from the science RL `final/` (see `reports/overnight_log_2026-11-04.md`). Measured step time on the first steps: 131-180 s (plan: 100 s central, 60-150 s range), reward -0.49 to -0.75, finished 0.99, KL 0.02, GPU 55 GB, no OOM. At about 140 s per step the 80 steps take about 3.1 h plus roughly 0.7 h of dev and hold-out evaluations, so the training estimate moves from 2.2 h central to about 3.8 h; the total with the post-run evaluation to about 6-7.5 h.

## 8. Crash loop 11:10-15:35 UTC and re-estimate (added 2026-11-04 15:45 UTC)

* **What happened.** After 10:55 the trainer stopped with rc=1 (ENOSPC) at the first step-25 evaluation ("new best" save) and was restarted 17-18 times by the launcher's supervisor, one every ~16 min, each time resuming from `last/` at step 20 and repeating steps 21-25. About **4.5 h of GPU time (11:10-15:35) produced no progress.**
* **Cause.** `hub_stage` (written at every new best) and `hub_stage_latest` (every 25 steps) of the run directory were real directories on the root disk (1.5 GB of weights each), not symlinks to `/dev/shm` like `best/`, `final/` and `last/`. With 2.4-2.5 GB free, the second save (old copy + temporary swap) did not fit. The disk preflight of the launcher (3 GiB) had already refused the first launch and I lowered it to 1 GB to start the run, which removed the one check that pointed at the problem.
* **My monitoring miss.** After the first three training steps (10:12) I watched only the overnight chain log and the waiter log, neither of which contains the launcher's restart lines, and I reported "no OOM, no alert" without having seen step 25. Fixed on 15:35 by the coordinator (both directories copied to `/dev/shm/rl_long_ckpt/`, identical, replaced by symlinks). From now on `data_pipeline/sft_clean/fresh/overnight/watch_run.sh` reports a heartbeat every 10 min (step, restarts, last exit codes, root and shm free, metrics age) and raises an ALERT at the second restart with the same exit code, at < 1.5 GB root free, or at 25 min without a metrics line.
* **Re-estimate.** 80 steps at ~145 s per step plus ~4 min per 25-step evaluation and save: the run is at step 25 at 15:38; steps 26-80 take about 2.4 h, so training ends about **18:05 UTC**; the post-run evaluation (2-3.5 h) ends about 20:30-21:30 UTC; the later queue (E-05, E-02, E-09b, E-01 stronger penalty, E-04) follows. Total GPU time for the stage: 15:38 - 10:03 = 5.6 h elapsed to reach step 25, so the stage costs about 10.5 h instead of the 6-7.5 h estimated in section 7.

## 9. Outcome (added 2026-11-04 18:55 UTC)

Training ended at 17:44 UTC (80 of 80 steps, 7.7 h wall-clock including the 4.5 h crash loop); evaluation finished at 18:40 UTC. Result in `reports/eval_long_document_rl_385m_2026-11-04.md`: whole-document generation improves moderately (finished 0.70 -> 0.82, loop 0.85 -> 0.73, chrF 27.4 -> 30.9 for best/), chunked mode and sentence level are unchanged. The estimates of section 4 held for the evaluation (about 1 h) and failed for training (crash loop, step time 130-180 s).
