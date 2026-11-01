# Science RL run: preparation, tests, baseline (D-52, agent A4, 2026-10-21)

Continue the 2c RL (policy AND Adam state, step 1433) for 500 (default) to 1000 more steps on long scientific text, 77 languages, evaluate on AfriScience-MT, push to the private repo `<HF_ORG_LEGACY>/gemma-science-translate`. **Nothing has been launched, nothing was pushed to the Hub or the bucket.** Research on the benchmark itself: `reports/afriscience_mt_2026-10-21.md`.

## 1. Where everything is

| what | where |
|---|---|
| run config | `gemma_mt/configs/rl_science.yaml` (repo) = `/workspace/jobs/a4/gemma_mt_sci/configs/rl_science.yaml` (box) |
| code used by the run | `/workspace/jobs/a4/gemma_mt_sci/` = a copy of `/workspace/sft_build/gemma_mt_2c` plus the D-52 patches below (the 2c tree is untouched; the same files are in the repo's `gemma_mt/`) |
| supervisor | `gemma_mt/scripts/sup_science.sh` (box: `/workspace/jobs/a4/gemma_mt_sci/scripts/sup_science.sh`), same OOM ladder as `sup.sh`, run dir `runs/sci` |
| prompts | `/workspace/jobs/a4/sci/rl_prompts_sci_v1/` (77 parquet files, 154 MB) + `/workspace/jobs/a4/sci/rl_prompts_sci_v1_stats.json` (copy: `rl_prompts_sci_v1_stats.json` here) |
| resume state | `/workspace/sft_build/runs/sci/last/trainer_state.pt` (3.2 GB, byte-identical copy of `runs/rl2c/last`, step 1433); KL reference / engine start model: `/workspace/sft_build/runs/final_1433` (HF snapshot of `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c@latest`, `training_state.json` says step 1433) |
| dev / benchmark files | `/workspace/jobs/a4/afs_raw/` : `corpus/{train,dev,test}-00000.parquet`, `bench_afrisci_dev_sent.jsonl` (5,058 rows), `bench_afrisci_dev_doc.jsonl` (150 papers), `published_chrf_test_eng2x.csv` |
| science-anchor thresholds | `gemma_mt/calibration/science_anchor_v1.json` (box: `gemma_mt_sci/calibration/`) |
| baseline eval | `reports/afriscience_science_run_2026-10-21/baseline_final_1433/afriscience_eval.{json,md}` (box: `/workspace/jobs/a4/eval_final1433/`, with per-row predictions) |
| logs | `dry_check_science.log` (27 checks, OK), `gpu_smoke_2steps.log` (2 real GPU steps), `anchor_probe.json` |

My box footprint: `/workspace/jobs/a4` 0.4 GB + `runs/sci` 3.2 GB + `runs/final_1433` 1.1 GB (about 4.7 GB; the GPU smoke copy of the trainer state was deleted).

## 2. AfriScience-MT in one paragraph

Six languages (amh hau lug nso yor zul, all in our 77), English -> X is the primary direction, 230 open-access papers in 11 domains turned into 250-350 word plain-language summaries by science communicators and translated by professionals; 7,605 English sentences, train / dev / test = 5,792 / 843 / 970 sentences (177 / 25 / 28 papers), Apache-2.0 (dataset card), arXiv:2605.29741. Scored with SSA-COMET (primary) and chrF, BLEU released; sentence level and whole-paper level. Details, citations and the published numbers: `reports/afriscience_mt_2026-10-21.md`.

## 3. Prompt set `rl_prompts_sci_v1`

Format of `rl_prompts_v3` (english, iso, language, src_afr_tokens, length_bin, n_sentences, source_dataset, licence) plus `domain` (science | general), `field`, `doc_id`, `window_id`. English windows are contiguous whole sentences of one document joined by single spaces (D-23), measured in Davlan/afro-xlmr-large-76L tokens and binned b64..b2048 like v3. Builder: `gemma_mt/scripts/build_science_prompts.py` (sub-commands windows / holdout / general / assemble; network sources are streamed and cut off, nothing but the final parquet is stored).

**106,922 rows, 77 languages.** Science 60,901 rows (39,684 unique English windows), general replay 46,021 rows.

| domain / bin | b64 | b256 | b512 | b1024 | b2048 |
|---|---|---|---|---|---|
| science rows | 6,231 | 9,240 | 11,550 | 16,940 | 16,940 |
| general rows | 9,233 | 9,229 | 9,213 | 9,192 | 9,154 |
| per language (science / general) | 80 / 120 | 120 / 120 | 150 / 120 | 220 / 120 | 220 / 120 |

Every (domain, bin) cell has all 77 languages (checked by `dry_check_rl.py`). Science rows by source (rows, licence): AfriScience-MT train English side 13,327 (apache-2.0; 1,941 unique windows incl. the 177 whole papers; the six benchmark languages draw 50 % of each cell from it, the others 20 %), PubMed Central OA 20,291 (CC BY or CC0, per document), LibreTexts science books 15,085 (CC BY), arXiv full text 7,572 (CC BY or CC0), arXiv abstracts 4,626 (CC0 metadata). Sources were streamed only partly (PMC 220 MB, arXiv 220 MB, LibreTexts all 164 MB, abstracts 80 MB of compressed data; 155,820 candidate windows after the hold-out). General replay: 120 rows per language and bin sampled from `rl_prompts_v3` (fineweb-edu odc-by 33,073 rows, Wikipedia share-alike 12,948 rows), without the 2c hold-out prompts (the 600 smallest-hash English texts of every bin are excluded, a superset of the 128 + 16-per-bin prompts the 2c eval used).

**Hold-out** (exact + 8-gram, applied to science AND general rows): AfriScience-MT dev (25 papers) and test (28 papers), AfriLongDoc `hard/` (135 documents) and `core/` (450 documents): 23,664 held-out sentences and 480,156 word 8-grams. A window is dropped when one of its sentences equals a held-out sentence (NFC, lower case) or when it shares any 8-gram: dropped 71 AfriScience-train, 590 LibreTexts, 252 PMC, 379 arXiv, 11 abstract windows, and 179 general rows. No train-prompt text can be a dev / test / hard sentence; this does NOT protect against paraphrases or against the benchmark papers' own sources being in PMC/arXiv under other wording.

**Mixing 70 / 30.** `PromptSchedule` got an optional domain mixture (minimal, tested, default off): every prompt of a step draws its domain by `domain_weights` (science 0.7, general 0.3), then its bin by `domain_bin_weights[domain]` (science b64 0.05, b256 0.10, b512 0.15, b1024 0.35, b2048 0.35 because the benchmark papers are 600-1,800 AfriXLM-R tokens; general uniform over the five bins like the 2c mixed stage), then the language (alpha 0 x `language_weights`, inside that domain+bin pool), then the prompt. It is stateless in (seed, step) like the rest, so resume needs only the step counter. Measured on the real set, 100 steps x 42 prompts: science 68.9 % (wanted 70), science bins b1024 1,031 / b2048 1,009 / b512 430 / b256 285 / b64 139, general bins ~255 each, all 77 languages drawn (24..111 draws each).

**Language weighting.** Same as 2c (uniform, x2.5 on the 17 weakest languages and efi) plus x2.0 for hau, lug, nso, zul (amh and yor already have 2.5): my judgement is that the six benchmark languages should be seen about twice as often as an ordinary language (hau 72, amh 95, yor 89 draws in 4,200) without starving the other 71; the AfriScience windows are also given to every language, so the other languages also get that vocabulary. Change `language_weights` to remove it.

## 4. Code changes (all optional, off by default; 14 new tests in `tests/test_science_run.py`, 5 in `tests/test_eval_afriscience.py`; 23 pass together with `test_greedy_in_group.py`, and the earlier 82 related tests passed)

* `rl/data.py`: `domain` column (`RLPrompts.domain`, `domain_of`), `PromptSchedule(domain=, domain_weights=, domain_bin_weights=)`. Without `domain_weights` the stream is bit-identical (tested).
* `rl/train_rl.py`: config fields `domain_weights`, `domain_bin_weights`, `resume_reset_selection`, `benchmark_aux`; `Sample.domain` is passed to the reward; eval logs `eval/reward_domain/<domain>`; `evaluate_aux_benchmarks` (monitor-only dev sets, logged as `bench_aux/<name>/...`); the checkpoint stores the benchmark path it was selected on, so `resume_reset_selection` re-baselines the resumed policy on the NEW dev set only once (the first start); a crash restart keeps the running best (tested end to end on the CPU smoke).
* `rl/reward.py` + new `rl/sci_anchor.py`: **science anchor** (see 6).
* new scripts: `build_science_prompts.py`, `prepare_afriscience.py`, `build_science_anchor_calibration.py`, `eval_afriscience.py`, `sup_science.sh`; `dry_check_rl.py` got a D-52 block.
* `tests/test_b2048_bin.py::test_staged_config_loads_strictly_and_the_stage_arithmetic_holds` fails on the box copy and in the repo: it compares `rl_rtxpro_phase2c.yaml` with the b2048 yaml and trips on `kl_estimator` / `reward` differences that predate this work (not touched).

## 5. Run configuration (`configs/rl_science.yaml`)

Copy of the live 2c yaml (`rl_rtxpro_phase2c_mixed.yaml`) with these differences (every other key, including lr 2e-6, group 16, kl 0.05 k3c, KL guard 0.15 median, prompts per step 42 in the mixed stage, max_new_tokens 7168, vLLM 0.20 / max_model_len 10240, is identical; a test asserts it):

* `model_name: /workspace/sft_build/runs/final_1433` = KL reference and engine start weights; `resume: true` loads the POLICY and the optimizer from `runs/sci/last/trainer_state.pt`. **Verified on the GPU:** the first resumed step reports `kl 0.0000` against the reference (policy == final model), then 0.0028 after one update.
* `out_dir: runs/sci`, `steps: 1933` (= 1433 + 500). For 1000 steps: `steps: 2433` (or the launch override `steps=2433`), same command; the supervisor resumes from `runs/sci/last`. `curriculum_steps` is unchanged, so every resumed step (>= 1284) is in the MIXED stage.
* `prompts_path` = the set above; domain mixture; `eval_prompts: 192`; `eval_min_per_bin: 16`.
* selection: `benchmark_eval_path` = AfriScience dev sentences (6 languages x 200 sentences, greedy, 512 new tokens, macro chrF++), `select_by: benchmark_chrf`, `select_min_gain: 0.2`, `resume_reset_selection: true` (the resumed policy is scored first on this set, only a better checkpoint goes to Hub main). Monitor only: `afrisci_dev_doc` (25 papers x 6 languages, up to 7,168 new tokens) and `general_dev` (the 2c dev set, forgetting check).
* Hub: `hub_repo_id: <HF_ORG_LEGACY>/gemma-science-translate`, private, main = best by dev chrF++ (tag `rl-step-N`), branch `latest` every 25 steps; no stage branches. `CheckpointPusher` creates the repo (private) at the first push.
* reward = live reward + `science_anchor_penalty: 0.5` (science prompts only).
* Option, off: `greedy_in_group` (see risks).

Checks run: `scripts/dry_check_rl.py configs/rl_science.yaml --real-lid` OK (`dry_check_science.log`); 2 real GPU steps with the real config at small batch (8 prompts x group 8, vLLM 0.09 of the card, real AfriCOMET-QE worker, real GlotLID, aux sets, no Hub): resumed from the trainer state, baseline scored (60 dev rows chrF++ 51.18, aux doc 38.81 / general 42.93 on tiny samples), 2 steps, eval, exit 0 (`gpu_smoke_2steps.log`). Timing of that smoke is not representative.

### Time and cost per 100 steps (estimate from measured 2c step times, not from this config)

Measured in the 2c metrics (`runs/rl2c/metrics.jsonl`): mixed stage (steps 1285-1433, 42 prompts x 16, mean completion 800 Gemma tokens) 120.7 s per step (median 118.7, p90 145.5, max 177); b1024 stage (42 prompts) 132.9 s; b2048 stage (21 prompts) 142.8 s; step time ~ 17.5 + 0.129 x mean completion length in the mixed stage (R2 0.62). The science mix is longer than the 2c mix (b1024 + b2048 = 70 % of the science prompts against 40 % in 2c): my estimate is **125-150 s per step, 135 s central**, plus about 4 % for the evals (reward eval every 25 steps, dev benchmark + whole-paper aux every 50) and 3 s per step for the science anchor (0.33 ms per chunk, measured).

| steps | GPU hours (range) | at $1.034/h (planning rate of `gemma_rl_cost_estimate_2026-10-17.md`; I do not know the current rental rate) |
|---|---|---|
| 100 | 3.9 (3.5-4.4) | $4.0 |
| 500 | 19.5 (17.5-22) | $20 ($18-23) |
| 1000 | 39 (35-44) | $40 ($36-45) |

The first 50-100 steps can be slower: long-paper generations that run into the 7,168-token cap are costly (greedy decoding loops at baseline, section 7; sampled rollouts finished 120 of 120 in my probe).

## 6. Science anchor (`rl/sci_anchor.py`)

Per chunk (sentence pair of the existing chunk plan), five features from the hard-split metrics (`eval/longdoc_hard.py`): number recall and precision (integers 0-10 excluded), number+unit pair recall, protected-token recall (acronyms, gene / chemical names, binomials, drug stems, citation markers; Latin-script outputs only, so Amharic / Tigrinya transliteration is not punished), formula recall. Deficit of a feature = max(0, t - f) / t with t = the human 10th percentile (AfriScience-MT train human references, 34,752 pairs; pooled p10 = 1.0 for all four features that occur, per-language entries for the six languages, amh numbers/units p10 0.0 so skipped); chunk deficit = mean over the features present; reward -= 0.5 x deficit (floor: the failure penalty); penalty only, never a bonus; only prompts with `domain == science`.

Measured: on the AfriScience test the human references have mean sentence deficit 0.001-0.019 and the final model 0.004-0.054 (sentence level keeps 99 % of the numbers), but on whole-paper outputs the model drops numbers: number recall 0.74 (hau), 0.54 (zul), 0.27 (yor) against 1.00 / 1.00 / 0.98 for the human references (finished papers only, whole-document metric), and on 120 sampled rollouts of science windows (temperature 1, final model) the mean chunk deficit was 0.288 (b2048 0.41, b1024 0.26, b256 0.02; PMC 0.36, LibreTexts 0.15): the anchor will average about 0.14 reward units on long science text at weight 0.5 and at most 0.5 per chunk. Part of that deficit is chunk misalignment (the chunked recall is lower than the whole-document recall), which the QE term punishes too.

## 7. Baseline: final general model (step 1433) on AfriScience-MT, greedy

Model `<HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c@latest` (= `runs/final_1433`), vLLM greedy, 512 new tokens per sentence, 7,168 per paper, `scripts/eval_afriscience.py`, English -> X. chrF = sacrebleu defaults (the paper's chrF is assumed to be this; its implementation is not stated), chrF++ word order 2, BLEU 13a, spBLEU flores200. Published numbers are the paper's released test metrics (`afriscience_mt_2026-10-21.md` section 4). No SSA-COMET (not run).

**Test, sentence level (970 sentences per language)**

| | amh | hau | lug | nso | yor | zul | macro |
|---|---|---|---|---|---|---|---|
| chrF | 25.28 | 59.43 | 46.19 | 55.65 | 37.41 | 63.39 | **47.89** |
| chrF++ | 22.80 | 57.23 | 41.88 | 53.40 | 36.22 | 57.66 | 44.86 |
| BLEU | 4.91 | 33.13 | 14.88 | 29.12 | 14.46 | 23.61 | 20.02 |
| spBLEU | 11.58 | 34.08 | 18.91 | 31.71 | 19.52 | 35.01 | 25.13 |
| mean per-sentence chrF | 25.72 | 59.19 | 46.45 | 55.97 | 37.08 | 63.45 | 47.98 |
| finished rate | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| published chrF, NLLB-1.3B fine-tuned | 45.5 | 67.7 | 51.5 | 64.3 | 17.7 | 67.2 | 52.3 |
| published chrF, GPT-5.4 10-shot | 43.8 | 63.0 | 56.4 | 59.5 | 28.1 | 68.9 | 53.3 |
| published chrF, Gemma2-9B LoRA | 39.7 | 58.8 | 50.1 | 61.6 | 41.6 | 50.2 | 50.3 |

**Test, whole papers (28 papers per language)**

| | amh | hau | lug | nso | yor | zul | macro |
|---|---|---|---|---|---|---|---|
| chrF | 22.09 | 49.10 | 30.82 | 34.21 | 21.78 | 40.75 | **33.13** |
| chrF++ | 19.11 | 45.82 | 25.30 | 31.09 | 20.04 | 34.14 | 29.25 |
| BLEU | 1.96 | 14.45 | 1.14 | 6.57 | 2.39 | 3.56 | 5.01 |
| spBLEU | 6.68 | 16.45 | 2.74 | 8.02 | 4.69 | 7.70 | 7.71 |
| finished rate | 0.96 | 0.86 | 0.43 | 0.82 | 0.54 | 0.79 | **0.73** |
| hyp / ref length | 0.88 | 1.31 | 1.96 | 1.23 | 1.58 | 1.62 | 1.43 |
| published chrF, GPT-5.4 1-shot | 42.8 | 66.8 | 59.5 | 64.9 | 28.7 | 68.6 | 55.2 |
| published chrF, Gemini-3.1-FL 1-shot | 45.1 | 64.3 | 50.8 | 55.8 | 17.5 | 62.9 | 49.4 |

Dev: sentence level chrF macro 48.59 (chrF++ 45.65), whole papers 36.00 (finished 0.75); full tables in `baseline_final_1433/afriscience_eval.md`. Reading: at sentence level the 270M model is within 5 chrF of fine-tuned NLLB-1.3B and GPT-5.4 and far above them on Yoruba (37.4 against 17.7 / 28.1), but on Amharic it is 20 points lower; **at paper level it is 22 points below GPT-5.4 and 27 % of the papers do not finish** (greedy decoding runs into the 7,168-token cap in repetition loops, e.g. "B. B. B. ..."; the unfinished papers are scored as they are, which depresses the macro).

## 8. Risks and decisions for the coordinator

1. **Greedy decoding of long text loops, sampled rollouts do not.** 2c's own eval reward on b2048 prompts (greedy) was -1.74 .. -2.0 at each of the five evals I looked at (steps 1275, 1300, 1350, 1400, 1433; finished rate 0.88-0.94 on the whole eval set), while the training rollouts (temperature 1) finish 99.99 % of the time. The RL signal comes from sampled completions only, so a science run can raise the sampled quality and still leave greedy paper-level evaluation (this benchmark) at ~73 % finished. `greedy_in_group: true` (last member of each group greedy, as in phase 2b) is the lever that punishes greedy loops directly; it is OFF in the yaml because it changes the 2c recipe and costs one more generate call per step (+30-60 s while loops are common). My recommendation: switch it on (`greedy_in_group=true` on the launch line) if the owner's target is the greedy paper-level score. Not tested on the GPU in this config (the CPU test exists, `test_greedy_in_group.py`).
2. Domain shift: the science windows are number- and acronym-dense; early rewards on b1024 / b2048 science prompts will be much lower than on general text (the greedy eval reward by domain is logged as `eval/reward_domain/*`). Flat groups (all -2) give no gradient; the KL guard (median, 0.15) is the safety net, nothing is cancelled automatically.
3. Benchmark is lay-summary text and small (28 test papers): one paper moves the paper-level chrF by ~1 point; the sentence-level dev selection set (1,200 sentences) has noise of about +-0.4 chrF++, hence `select_min_gain 0.2`. Selection is on SENTENCE-level dev chrF++, the paper-level dev set is only monitored.
4. Contamination: the hold-out is exact + 8-gram only. The benchmark's source papers are open access (AfricArXiv, Zenodo) and could appear in PMC / arXiv under their original wording; the English sentences of the benchmark are lay summaries written for the dataset, so a verbatim overlap is unlikely, not excluded.
5. Disk: the box had 8.5 GB free at my last check (other agents); a 500-step run needs ~3.2 GB for `last/` (+ a 3.2 GB temporary `.tmp` during every save) and the Hub staging dirs; check `df` before launching.
6. The coordinator's rule "GPU only after the other evaluations": the run needs the whole card as in 2c (vLLM 0.20 of 96 GB + trainer); the supervisor's ladder only lowers the micro-batch.
7. Code lives in `/workspace/jobs/a4/gemma_mt_sci` (outside `sft_build`): do not delete `/workspace/jobs/a4` while the run is up (the prompts, the dev files, the calibration and the code are there). Move them and edit three paths in the yaml if wanted.
8. SSA-COMET (the paper's primary metric) is not computed anywhere in our pipeline; the comparison with the paper is on chrF only.

## 9. Commands

Launch (the supervisor refuses to start without `runs/sci/last/trainer_state.pt`; it deletes `runs/sci/summary.json` first, so it can be re-run to extend the run):

```bash
ssh -o BatchMode=yes -i $SSH_KEY -p $BOX_PORT root@$BOX_HOST \
  'cd /workspace/sft_build && nohup /workspace/jobs/a4/gemma_mt_sci/scripts/sup_science.sh /workspace/jobs/a4/gemma_mt_sci/configs/rl_science.yaml SCI > /dev/null 2>&1 &'
# log: /workspace/sft_build/logs/rl_SCI.log   metrics: /workspace/sft_build/runs/sci/metrics.jsonl
# 1,000 steps instead of 500: set `steps: 2433` in the yaml before launching (or edit it later and re-run the same command: the supervisor resumes from runs/sci/last)
# greedy_in_group: the supervisor passes only micro-batch overrides, so switch it on by editing `greedy_in_group: true` in the yaml
```

Evaluate a checkpoint (final model or the science model; GPU <= 14 GB; dev + test, sentence + paper level, about 10 minutes):

```bash
source /workspace/jobs/a4/pyenv.sh   # PY + PYTHONPATH
PYTHONPATH=/workspace/jobs/a4/gemma_mt_sci:/workspace/sft_build/pylibs /workspace/venv_gemma/bin/python /workspace/jobs/a4/gemma_mt_sci/scripts/eval_afriscience.py \
  --model <HF_ORG_LEGACY>/gemma-science-translate --revision main --splits dev,test --levels sentence,document \
  --corpus-dir /workspace/jobs/a4/afs_raw/corpus --published /workspace/jobs/a4/afs_raw/published_chrf_test_eng2x.csv \
  --vllm-python /workspace/venv_vllm/bin/python --vllm-gpu-util 0.15 --out /workspace/jobs/a4/eval_science_main
# other decoding: --decoding "sample:temperature=0.3,top_p=0.95"  or  --decoding @decoding.json
```
