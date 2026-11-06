# RL preflight, 385M grown model (2026-11-03)

Run: A100 80 GB, `/workspace/clean_rerun/runs_385m/preflight` (config `rl_00_preflight.yaml`, 49 steps, b64 only, `GEMMA_MT_TAG_MODE=single_token`, init = final SFT `g15_385m_sft_final`). Log `/workspace/clean_rerun/logs/rl_preflight.log`, metrics `metrics.jsonl`, `summary.json`. All numbers below are read from those files (VERIFIED) unless marked.

Setup facts: the preflight ran on the general pool only (`clean_v3_385m`, 3,030,964 prompts, 128 eval rows). It was started before the news domain and before arz was removed, so its language list still has 85 languages including arz and NO news prompts; its config.json has no `domain_weights`. The GPU was shared with the evaluation's MBR decode job for the whole run (pid 1282937, about 7.6 GB), so step time is an upper bound.

## Does it run?
* vLLM loads the 39-layer config and samples (worker up in 25 s). Policy, vLLM and the two QE workers coexist. Tag mode single_token works: sampled text starts with the language tag for almost all languages (see bad_prefix below for the exceptions).
* No crash, no NaN, no OOM. Hard-failure classes are logged per language.

## Speed and memory
* 49 steps in 2,777 s including two evals: about 52-57 s per step at 42-64 prompts x group 16 (b64), generation 2,053 tokens/s on average (range 858-5,033).
* GPU memory (nvidia-smi during the run): policy process 46.3 GB, vLLM 16.8 GB, QE workers 2.7 GB = 65.8 GB, plus the other job's 7.6 GB = 73.5 GB of 80. Running alone it fits with about 14 GB spare. Host RAM 73 GB used of 503 GB (shared box).

## Learning signal
* Dev chrF++ macro (720 rows, 16 languages): SFT start 36.81 -> step 25 37.50 -> step 49 38.83 (BLEU 12.46 -> 13.59). Greedy finished rate on the 128-row eval 0.93 -> 0.91, loop rate 0.11 (b256 rows 0.17, b64 rows 0.07); training-sample finished rate 0.99.
* weak_dev aux set chrF++ 20.76 at the start (end value in summary.json).
* KL 0.0 -> 0.063 (mean 0.037), grad norm 2.9-3.8, clip fraction 0, KL guard never fired (lr scale 1.0), flat-group fraction 2% (0-8%).
* Reward (training) mean -1.02 (range -1.18 to -0.80): the scale is dominated by penalties and the hard-failure share, so only its trend matters. Eval reward -0.28 (step 25) -> -0.20 (step 49).

## Failure classes (share of samples, mean over 49 steps)
* hard failures 9.5% (4-15%): wrong_language 4.1%, bad_prefix 3.3%, no_end_of_turn 1.0%, stuffed_output 0.6%, length_ratio 0.3%, degenerate 0.1%, copy_source about 0 (pcm 88%: the model copies English, a Pidgin-reward question), empty 0, special tokens 0.
* wrong_language is high for a few languages: iso 44%, bem 37%, tum 29%, umb 26%, lua 25%, arz 23%, sna 19% (GlotLID sibling confusions; the label sets for orm, bem, tum, umb, lua are the v5 ones, `lid_labels_v6.json` only fixes other languages). Oromo samples are labelled `gaz_Latn`, which is accepted (status ok); its wrong_language rate is 15% (up to 50%).
* `mos` samples sometimes contain Arabic-script garbage; the diacritic guard scores them -2 (working as intended).

## Problem found: three RL languages the SFT never saw
`dag`, `bbj` and `ibb` show bad_prefix 93-95% (output does not start with the language tag) and eval reward -2.0. They are absent from the SFT token table and the shard LID measurement (no SFT data in the clean rerun). With every sample failing, their groups are flat and give no gradient, so they only waste about 3.5% of the prompts. Recommendation: drop dag, bbj, ibb from the stage A language list (as for arz) and bring them back only after SFT data exists. Other partial cases: bin 35%, lub 44%, tll 25%, zne 25% bad_prefix (new languages with very little or no A0 data; they have stage-B data and may improve with RL).

## Recommendation on the reward flags
Keep confidence, fluency and spelling OFF for stage A. The preflight gives no sign that the base reward is being gamed (finished rate up, chrF++ up, KL small, no collapse); the flags are untested on this model and the LID-confusion languages above are a larger problem than fluency. Revisit after stage A, using per-language logs.
