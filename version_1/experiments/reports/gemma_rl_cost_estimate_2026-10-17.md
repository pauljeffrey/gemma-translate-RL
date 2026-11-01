# Cost estimate: GRPO RL for gemma-3-270m-it en->African translation on an RTX PRO 6000 Blackwell (96 GB) at $1.034/h


> **Note added 2026-10-17 (evening, D-32):** the RL set now has **42 languages** (22 added: amh tir lin lug mlg run bem tso nso tum mos kik luo bam kmb ssw lua umb kab kam kbp nus; 20,000 prompts each, 5,000 per bin, in `rl_prompts_v2`). Everything below that says "20 languages" describes the old set. The per-step workload and the cost per step do NOT change
> (`prompts_per_step_by_bin` is per step, not per language); what changes is who is drawn: with `language_alpha 1.0` an added language gets 0.38% of the draws (~14 of the 3,600 prompts of a 300-step run). Details and the new draw shares: `reports/rl_reward_explained_2026-10-17_v2.md` section 19.

Date: 2026-10-17. Author: analysis agent (read-only analysis; no GPU used anywhere, nothing pushed, no running job touched). All numbers below are produced by one small Python model (inputs and formulas in sections 1 and 4) so that every assumption can be changed and the table re-derived.

**Evidence labels used throughout:** [M] = measured in this task from the repo or the data on the old box; [L] = taken from an existing log/report of the owner's earlier measurements; [A] = assumption (no measurement possible without a GPU run of this code). The RL path has never run on a real GPU, so every speed number for sampling is [A].

## 0. Headline

| scenario | prompts / language x bin | optimizer steps | prompts | completions (x8) | V0 as shipped (HF generate): central [low - high] | V2 vLLM rollouts+eval: central [low - high] |
|---|---|---|---|---|---|---|
| MIN | 16 (12 staged + 4 mixed) | 64 | 1,280 | 10,240 | **$2.43** / 2.4 GPU-h [$1.29 - $4.36] | **$0.59** / 0.6 GPU-h [$0.37 - $1.18] |
| BASE | 48 (32 staged + 16 mixed) | 192 | 3,840 | 30,720 | **$7.59** / 7.3 GPU-h [$4.01 - $13] | **$1.38** / 1.3 GPU-h [$0.89 - $2.63] |
| GENEROUS | 160 (100 staged + 60 mixed) | 640 | 12,800 | 102,400 | **$25** / 24.3 GPU-h [$13 - $44] | **$4.14** / 4.0 GPU-h [$2.71 - $7.71] |

Plus a one-off **first-run shake-down** that the estimate does not hide: the RL path has never executed on a GPU, so budget about 3 GPU-h (range 1 - 8 h) = **$3 (range $1 - $8)** for calibration (`calibrate_shaping`), a smoke run, fixing the first crash and measuring the three numbers in section 9. Disk/network fees of the rental are not included.

**Reading the table.** The money is small in every scenario; the interesting quantities are wall-clock and whether the prompts are enough to learn something. In V0, HF `generate` (training rollouts plus the two evaluation passes, which are also generation) is 80 - 90 % of GPU time because every batch waits for its longest sequence; vLLM removes most of that. QE scoring is about 1 % of the time. The single biggest uncertainty is the **per-token latency of HF `generate` for this model and its long-tail behaviour (one runaway sample holds the whole batch until `max_new_tokens`)**: nothing in the repo or on the old box measures it, and it moves V0 by a factor of about 3 between the low and high columns.

## 1. What was verified (code and data) and the anchors

### 1.1 The RL path as it exists (read from `gemma_mt/gemma_mt/rl/*` and `configs/rl.yaml`) [M]
* **Sampling = HF `model.generate`** (`sampling.py`: `model.generate(..., do_sample=True, num_return_sequences=G)`), fp32 weights under bf16 autocast, one call per optimizer step for all `prompts_per_step x G` sequences, left-padded prompts. No vLLM anywhere in `gemma_mt/`. A batch stops only when every sequence has produced a stop id or hit `max_new_tokens_by_bin` (b64 384, b256 1280, b512 2304, b1024 3584, mixed 3584).
* **Update**: one on-policy update per batch (`num_updates_per_batch: 1`), group size 8, AdamW lr 1e-6 (warmup 10), KL 0.05 to a frozen fp32 reference. Per step: policy never runs a no-grad pass; the reference runs one forward, the policy one forward+backward over `[prompt | completion]` with chunked LM-head log-probs (`loss_chunk_tokens` 512, each chunk checkpointed). `gradient_checkpointing: false`. Micro-batch = `min(micro_batch_seqs=16, micro_batch_tokens=24576 // longest)` sequences.
* **Reward**: AfriCOMET-QE `masakhane/africomet-qe-stl-1.1`, in-process `CometScorer`, fp16, batch 32 sorted by length, **sentence-chunked** (`chunking: sentences`): every sample becomes about `n_sentences` (source, translation) segments, so one segment = one encoder pass over (source sentence, output sentence) = two encoder sequences. Python-side reward logic measured here on the laptop with the fake scorer: 0.6 / 2.4 / 7.0 ms per sample at 5 / 13 / 25 sentences (negligible).
* **Schedule as shipped** (`rl.yaml`, 300 steps): b64 40 steps x 32 prompts, b256 60 x 16, b512 60 x 8, b1024 60 x 4, then 80 mixed steps x 8 prompts = **3,600 prompts, 28,800 completions**; language drawn with probability proportional to pool size (`language_alpha 1.0`).
* **Evaluation**: every 25 steps a greedy pass over 128 held-out prompts with the full QE reward, batches of `prompts_per_step_by_bin` (32/16/8/4 per call by bin, so 1/2/4/8 calls per bin of 32 prompts); every 50 steps (+ step 0 baseline + final step) greedy dev chrF++ on 40 sentences x 19 languages = 760 sentences, batch 32, `max_new_tokens` 256. A `last/` checkpoint (model + AdamW, fp32, about 3.2 GB) every 50 steps; the selected model is pushed to the Hub in a background thread.
* `load_rl_prompts` reads the whole `prompts_path` (the full set is 4.8 M windows, several GB of text); give the run a small per-scenario subset file (section 6.4).

### 1.2 Prompt pools on the old box (`gemma/rl_prompts_v2`, `rl_prompts_v2_tok/report.json`) [M]
The pool is **not 120 k per language**: 11 languages have 120 k (afr hau ibo kin sna som swh tsn xho yor zul), 8 have 360 k (efi ewe fon nya orm sot twi wol), pcm has 600 k; **total 4,800,000**, uniform over the four bins within a language. Even GENEROUS uses 0.27 % of it, so prompts never repeat. English-window statistics are the same for every language (same windows):

| bin (AfroXLM tokens of the English window) | mean Gemma tokens of the English window | p95 | mean chat-prompt tokens | mean sentences | mean AfroXLM tokens of the window |
|---|---|---|---|---|---|
| b64 | 35 | 53 | 52 | 1.6 | 40 |
| b256 | 125 | 205 | 142 | 5.3 | 139 |
| b512 | 313 | 425 | 330 | 12.7 | 348 |
| b1024 | 613 | 834 | 630 | 24.5 | 685 |

### 1.3 Expansion factor: Gemma tokens of the output (`<iso> ` tag + translation + `<end_of_turn>`) / Gemma tokens of the English [M]
Two independent samples: (a) **pairs_final**, up to 2,000 random sentence pairs per language (1,177 - 2,000 where a language has fewer pairs); (b) **windows_sft**, 40 random windows per language x bin (3,200 windows). Window ratios are used in the cost model because RL prompts are windows.

| language | pairs ratio (n) | windows ratio b64 / b256 / b512 / b1024 | pooled windows ratio | AfroXLM tgt/src (windows) |
|---|---|---|---|---|
| afr | 1.60 (2000) | 1.58 / 1.48 / 1.46 / 1.48 | 1.48 | 1.13 |
| efi | 2.00 (1564) | 1.75 / 1.99 / 2.01 / 2.02 | 2.01 | 1.90 |
| ewe | 2.04 (2000) | 2.23 / 2.24 / 2.10 / 2.14 | 2.14 | 2.01 |
| fon | 2.51 (2000) | 2.56 / 2.47 / 2.44 / 2.50 | 2.48 | 2.49 |
| hau | 1.68 (2000) | 1.79 / 1.72 / 1.73 / 1.70 | 1.71 | 1.39 |
| ibo | 2.13 (2000) | 2.61 / 2.12 / 2.18 / 2.12 | 2.14 | 2.23 |
| kin | 1.73 (1177) | 1.90 / 1.76 / 1.73 / 1.70 | 1.72 | 1.53 |
| nya | 1.80 (1726) | 1.97 / 1.89 / 1.80 / 1.83 | 1.83 | 1.51 |
| orm | 2.01 (1817) | 2.13 / 2.34 / 2.18 / 2.11 | 2.15 | 1.75 |
| pcm | 1.17 (2000) | 1.62 / 1.36 / 1.25 / 1.25 | 1.27 | 1.18 |
| sna | 1.74 (1545) | 1.76 / 1.72 / 1.70 / 1.73 | 1.72 | 1.44 |
| som | 1.90 (1886) | 1.65 / 1.64 / 1.60 / 1.61 | 1.61 | 1.16 |
| sot | 1.98 (1816) | 2.10 / 2.49 / 2.08 / 2.06 | 2.09 | 1.85 |
| swh | 1.62 (2000) | 1.86 / 1.76 / 1.61 / 1.56 | 1.60 | 1.14 |
| tsn | 2.15 (2000) | 2.09 / 2.24 / 2.04 / 2.03 | 2.05 | 1.76 |
| twi | 1.83 (2000) | 1.61 / 1.76 / 1.73 / 1.75 | 1.74 | 1.70 |
| wol | 1.64 (2000) | 1.75 / 1.72 / 1.67 / 1.65 | 1.66 | 1.54 |
| xho | 2.01 (2000) | 2.22 / 2.16 / 2.16 / 2.09 | 2.12 | 1.67 |
| yor | 2.45 (2000) | 2.33 / 2.48 / 2.18 / 2.27 | 2.26 | 2.25 |
| zul | 1.93 (2000) | 2.04 / 2.05 / 1.99 / 1.92 | 1.95 | 1.54 |

Mean over the 20 languages: **1.89x** (range 1.27 pcm ... 2.48 fon); the `rl.yaml` comment "2 - 2.5x" is right for yor/fon/ibo/ewe/orm/sot/xho/tsn but too high for the average. Per bin (mean over languages) the ratio is almost flat: b64 1.98, b256 1.97, b512 1.88, b1024 1.88; the maximum over languages (fon/sot/ewe, the worst case that sets how long a 20-language batch runs) is b64 2.61, b256 2.49, b512 2.44, b1024 2.50.

Resulting completion lengths used by the model (mean of the 20 languages; "natural longest" = highest-expansion language x p95 English window, i.e. how long a 160-sample batch runs if nothing derails):

| bin | prompt tokens | mean completion tokens | natural longest completion | `rl.yaml` cap | cap / mean | QE segments per completion | mean QE tokens per segment pair |
|---|---|---|---|---|---|---|---|
| b64 | 52 | 69 | 138 | 384 | 5.6x | 1.6 | 69 |
| b256 | 142 | 246 | 511 | 1280 | 5.2x | 5.3 | 72 |
| b512 | 330 | 589 | 1035 | 2304 | 3.9x | 12.7 | 75 |
| b1024 | 630 | 1149 | 2088 | 3584 | 3.1x | 24.5 | 76 |

### 1.4 Speed anchors
* **SFT on the RTX 5070 12 GB** [L]: 6,100 tokens/s planned; the live `sft_unimax` log at 2026-10-17 shows 5,820 - 5,822 tokens/s, 21.4 s/step (131 k tokens/step, micro 2 x accum 16, gradient checkpointing). The model uses 5,900 as the anchor. One SFT token costs about 1.61 GFLOP in this implementation (100 M-parameter body: fwd + recompute + bwd = 0.8 GFLOP; 168 M-parameter tied LM head, computed only on the ~60 % target tokens and checkpointed per chunk: 1.34 GFLOP x 0.6). One RL token (no body checkpointing, but the reference forward, and ~65 % completion tokens) costs about 1.9 GFLOP, so **RL training tokens/s = SFT tokens/s / 1.18** at the same hardware efficiency [M-derived].
* **AfriCOMET-QE fp16 on the 5070** [L, from `project_gemma_mt` notes and `logs/qe_*.log`]: about 290 - 350 pairs/s with one scoring process (CPU/dataloader bound), about 480 pairs/s aggregate with three processes at 0.30 GPU fraction each (about 159 each); later runs show 140 - 300 pairs/s per process depending on how many ran concurrently. The population scored in those runs [M, 20,000 random pairs from `gemma/scored`]: mean 34.6 + 52.9 = **87.5 AfroXLM tokens per pair** (median 57, p95 198), so the 5070 processed about 25 - 42 k encoder tokens/s. RL segments (sentences of the windows) average 69 - 76 tokens per pair, i.e. very similar. CPU-only numbers (0.5 - 0.85 pairs/s, `ssa_cmp`) are irrelevant to the GPU case.
* **Nothing measured** for: HF `generate` latency of this model (no log of any generation timing exists on the old box), vLLM, or any speed on the RTX PRO 6000.
* **Hardware ratio PRO 6000 Max-Q / 5070** [A from published specs; spec sheets for the PRO 6000 Max-Q: 24,064 CUDA cores, 110 TFLOPS FP32 peak, 96 GB GDDR7, 1,792 GB/s, 300 W (Lenovo/Puget spec pages); 5070 figures are from memory, not re-verified: about 6,144 cores, about 31 TFLOPS FP32, 12 GB GDDR7, 672 GB/s]. Cores x3.9, FP32 x3.5, bandwidth x2.7. A 300 W part sustains lower clocks than the 5070 boost and a 640-wide model does not fill 188 SMs, so: **low-cost 3.8x, central 3.0x, high-cost 2.0x** for training; QE uses 0.87x of that (its tokenizer/batching runs on one CPU thread). HF decode is launch/CPU bound and is *not* scaled by this ratio.

## 2. Definition of the "minimum prompts per language x bin"

**Unit.** A *cell* is (language, length bin): 20 x 4 = 80 cells. Each prompt is used once (fresh draw, no repeats; `num_updates_per_batch: 1` means a prompt can only ever give one update anyway). A prompt yields one group of G = 8 completions and one advantage vector.

**Schedule used by all three scenarios** (a small change to `PromptSchedule`, see section 6.4): *language-stratified*, **one prompt per language per optimizer step (P = 20 prompts = 160 completions per step)**, stages b64 -> b256 -> b512 -> b1024 as in `rl.yaml`, then a *mixed* stage in which the bin rotates round-robin over (step + language) so that each cell receives exactly `mixed_steps / 4` prompts. Then a cell's prompt count equals the number of optimizer steps in which that language-bin is present: `prompts per cell = steps per stage + mixed steps / 4`.

**Why these minima (heuristic arguments, labelled; none is a measurement):**
1. *What the shipped sampler gives a plain language.* `language_alpha 1.0` draws a language with probability proportional to its pool: 2.5 % for the 11 plain languages, 7.5 % for the 8 boosted ones, 12.5 % for pcm. In the 300-step plan a plain-language cell therefore receives on average 36 / 28 / 16 / 10 prompts (b64 / b256 / b512 / b1024, staged + mixed share) and a random Poisson spread around that (b1024 stage alone: mean 6, P(fewer than 2) = 1.7 %). So the shipped plan is already about MIN-sized for plain languages in b512/b1024 and below BASE everywhere, and boosted languages/pcm get 3x/5x that. A "minimum" defined as a per-cell quota cannot be met exactly by random draws; it needs the stratified schedule (section 6.4), which is cheap to write.
2. *Updates per cell.* A cell must be present in enough consecutive optimizer steps for its stage to mean something: the stage must be longer than the LR warm-up (10 steps; use 4 in MIN) and long enough that the stage-end evaluation compares different policies. **12 steps** is the smallest stage that satisfies both with 3x margin on a 4-step warm-up.
3. *Reward noise / learning signal.* The independent unit of evidence is the prompt, not the completion (the 8 completions share one source). With the mean-std advantage (`advantage_norm: mean_std`) each group contributes a unit-scale signal as long as its 8 rewards are not all equal. Groups that are all failures (no `<end_of_turn>` -> penalty -2) are *flat* and contribute exactly zero; early in b512/b1024 this can be a large share, so the effective number of informative prompts per cell is plausibly 50 - 90 % of the nominal one [A; `flat_group_frac` in the log measures it]. The standard error of a cell's mean reward falls as sigma/sqrt(prompts); with an assumed between-prompt sigma of 0.5 reward units [A; `reward_std` in the log measures it] a cell mean has SE 0.125 at 16 prompts, 0.07 at 48 and 0.04 at 160. Standard error of the mean reward at each aggregation level (sigma 0.5): pooled over all cells MIN 0.014 / BASE 0.008 / GENEROUS 0.004; per language (4 bins) 0.063 / 0.036 / 0.020; per cell 0.125 / 0.072 / 0.040. A before/after comparison needs a change of several SE, so **MIN can show a pooled reward trend and little else; per-language trends need BASE or more; per-cell statements need GENEROUS.** The dev chrF++ check is the same for all scenarios (760 sentences; `rl.yaml` itself notes that differences below about 0.3 macro chrF++ are noise), so no scenario resolves gains smaller than that.
4. *A cell must receive at least one full eval-sized sample of completions:* 16 prompts x 8 = 128 completions = one `rl.yaml` optimizer step's worth of samples, the smallest number for which a cell's reward/finish-rate/failure-class mix is not anecdotal.

**Scenarios (exact counts).** G = 8, P = 20 prompts/step, `eval_every` as stated.

| | MIN | BASE | GENEROUS | `rl.yaml` as shipped (reference) |
|---|---|---|---|---|
| steps per stage b64 / b256 / b512 / b1024 | 12 / 12 / 12 / 12 | 32 / 32 / 32 / 32 | 100 / 100 / 100 / 100 | 40 / 60 / 60 / 60 |
| mixed steps | 16 | 64 | 240 | 80 |
| prompts per language x bin: staged + from mixed = total | 12 + 4 = **16** | 32 + 16 = **48** | 100 + 60 = **160** | random and uneven: a plain language gets on average 36 / 28 / 16 / 10 (b64 / b256 / b512 / b1024), a boosted one 3x, pcm 5x |
| per-language total (4 bins) | 64 | 192 | 640 | 90 (plain) / 270 (boosted) / 450 (pcm), on average |
| total optimizer steps | **64** | **192** | **640** | 300 |
| total prompts (80 cells) | **1,280** | **3,840** | **12,800** | 3,600 |
| total completions (x 8) | **10,240** | **30,720** | **102,400** | 28,800 |
| LR warm-up | 4 steps (was 10) | 10 | 10 | 10 |
| QE-reward eval / dev-chrF++ eval cadence | every 32 / every 32 (+ baseline) | every 25 / every 50 (+ baseline) | every 25 / every 50 (+ baseline) | same |
| what it can support | pipeline works, reward and KL move, a pooled trend | per-language reward trends | per-cell trends, more checkpoints to select from | |

**Exact prompts (completions) for each of the 20 languages x each bin** (identical for afr efi ewe fon hau ibo kin nya orm pcm sna som sot swh tsn twi wol xho yor zul; the last column is the sum over the 4 bins):

| scenario | b64 | b256 | b512 | b1024 | per language | all 20 languages |
|---|---|---|---|---|---|---|
| MIN | 16 (128) | 16 (128) | 16 (128) | 16 (128) | 64 (512) | 1,280 (10,240) |
| BASE | 48 (384) | 48 (384) | 48 (384) | 48 (384) | 192 (1,536) | 3,840 (30,720) |
| GENEROUS | 160 (1,280) | 160 (1,280) | 160 (1,280) | 160 (1,280) | 640 (5,120) | 12,800 (102,400) |

 Owner boost (8 low-resource languages x2, pcm x3, i.e. 30 language-units instead of 20) multiplies prompts and completions by 1.5 (sensitivity table). MIN at lr 1e-6 for 64 steps moves the policy only a little (the sum of Adam updates is at most about 6e-5 per weight); it answers "does the reward move and does the pipeline run", not "is the model better".

## 3. Per-step workload by bin (mean of 20 languages, G = 8, 160 completions) [M inputs, derived]

| stage | completion tokens generated per step | real tokens trained per step (prompt + completion) | QE segments per step | decode horizon (steps) HF, q-weighted | V0 central: sample / QE / train / other = step s | V2 central: sample / QE / train / other = step s |
|---|---|---|---|---|---|---|
| b64 | 10,969 | 19,227 | 247 | 212 | 5 / 0.2 / 1.7 / 1.6 = **9 s** | 2 / 0.2 / 1.5 / 1.6 = **5 s** |
| b256 | 39,412 | 62,137 | 818 | 896 | 25 / 0.7 / 5.6 / 2.0 = **34 s** | 3 / 0.7 / 4.8 / 2.0 = **11 s** |
| b512 | 94,199 | 146,960 | 1,967 | 1,923 | 61 / 1.7 / 13.2 / 2.6 = **79 s** | 6 / 1.7 / 11.2 / 2.6 = **22 s** |
| b1024 | 183,856 | 284,583 | 3,805 | 3,285 | 122 / 3.4 / 25.6 / 3.7 = **155 s** | 11 / 3.4 / 21.8 / 3.7 = **40 s** |
| mixed | 82,109 | 128,227 | 1,709 | 3,285 | 122 / 1.5 / 11.5 / 2.5 = **138 s** | 7 / 1.5 / 9.8 / 2.5 = **21 s** |

(mixed = 20 prompts spread over the four bins in ONE `generate` call, so the call runs as long as the b1024 prompts need.)

## 4. Cost model: formulas and every assumption

Per optimizer step with N = 8 x P sequences, bin b, parameters from the table below:

* **Sampling, HF** (`T_samp = S_b x (t0 + 2 x N x ctx x KVB / BW)`): `S_b = min(cap_b, q_b x cap_b + (1 - q_b) x L_nat_b)` decode steps (the batch runs until its longest sample finishes; with probability q_b at least one of the N samples never emits `<end_of_turn>` and runs to the cap); `ctx` = longest prompt p95 + S_b/2; `KVB` = 18 layers x 2 x 1 KV head x 256 x 2 B = 18,432 B/token; `t0` = launch/Python-bound latency of one HF decode step of an 18-layer model [A: 15 / 25 / 40 ms].
* **Sampling, vLLM colocated or as a sidecar process**: `T_samp = S_b x t_floor + N x (C_b + rho x cap_b) x c_tok + t_sync` with `t_floor` per-decode-step latency at small batch (CUDA graphs), `c_tok` marginal cost per generated token, `rho` = fraction of runaway samples, `t_sync` = copying the new policy weights (fp32 -> bf16, 0.54 GB) into the engine [A: 2.5 / 3.5 / 6 ms; 4 / 8 / 20 us; 0.2 / 1.0 / 3.0 s; rho 0.3 / 1 / 3 %].
* **QE** `T_qe = N x n_seg x 0.97 x tau_b / (QE_tps x r_qe)`; `tau_b` = encoder tokens per segment pair (src + mt + 4 specials) from the AfroXLM window lengths x (1 + 1.575 mt/src ratio) / n_seg; 3 % of samples fail the guards and are not scored. `QE_tps` = 5070 encoder tokens/s [L: 42 k / 33 k / 25 k = 480 (three processes) / 380 / 290 (one process) pairs/s x 87.5 tokens].
* **Policy + reference forward/backward** `T_train = N x (P_b + C_b) x pad / ((5,900 / 1.18) x r)`: `pad` = padding waste of left-padded prompt + right-padded completion in micro-batches that are not length-sorted [A: 1.15 / 1.35 / 1.6; V1/V2 sort micro-batches by length: x0.85].
* **Other per step** `T_ov = ov + 2 x 0.28 ms x N x n_seg` (batch assembly, AdamW, clip, logging, Python reward logic; `ov` [A] 1.0 / 1.5 / 3.0 s).
* **Evaluation**: `T_QEeval` = greedy pass over 128 held-out prompts (HF: per call `steps x t0`, calls per bin as in `rl.yaml`; V1/V2: one call of 32 per bin) + QE scoring; `T_bench` = 24 calls of 32 sentences x (q x 256 + (1 - q) x 150) steps x t0 + 10 s of chrF++/BLEU on CPU (V2: one vLLM call of 760).
* **Checkpoints and overhead**: `last/` every 50 steps (3.2 GB torch.save, 10 / 20 / 45 s [A]), plus about 8 s per Hub staging and 60 s final wait for the background upload; **start-up** 5 / 10 / 25 min [A] (load policy + reference + QE, read prompt subset, first CUDA graphs).
* **Total** = sum over steps of `T_samp + T_qe + T_train + T_ov` + `n_QEeval x T_QEeval` + `n_bench x T_bench` + checkpoints + start-up. GPU-hours x $1.034 = dollars; wall-clock days = GPU-hours / 24 (continuous, no restarts, GPU not shared).

| symbol | low-cost | central | high-cost | status |
|---|---|---|---|---|
| speed ratio r (training), PRO 6000 Max-Q / 5070 | 3.8 | 3.0 | 2.0 | [A] from specs |
| speed ratio r_qe | 3.3 | 2.6 | 1.7 | [A] |
| 5070 QE encoder tokens/s | 42,000 | 33,000 | 25,000 | [L] 480 / ~380 / 290 pairs/s |
| SFT tokens/s anchor on 5070 | 5,900 | 5,900 | 5,900 | [L/M] |
| RL token cost vs SFT token | 1.18 | 1.18 | 1.18 | [M-derived] |
| HF decode step t0 | 15 ms | 25 ms | 40 ms | [A] no measurement |
| KV-copy effective bandwidth | 1.4 TB/s | 1.2 TB/s | 1.0 TB/s | [A] (1.79 TB/s peak) |
| q_cap (runaway reaches cap) b64 / b256 / b512 / b1024 / mixed | .1 / .2 / .3 / .4 / .4 | .3 / .5 / .7 / .8 / .8 | .6 / .9 / 1 / 1 / 1 | [A] a 160-sample batch at temperature 1.0; with a 1 % per-sample derail rate q = 80 % |
| padding factor | 1.15 | 1.35 | 1.6 | [A] |
| step overhead | 1.0 s | 1.5 s | 3.0 s | [A] |
| vLLM t_floor / c_tok / t_sync | 2.5 ms / 4 us / 0.2 s | 3.5 ms / 8 us / 1.0 s | 6 ms / 20 us / 3.0 s | [A] |
| checkpoint write / start-up | 10 s / 5 min | 20 s / 10 min | 45 s / 25 min | [A] |

Variants: **V0** as shipped (HF generate, caps and eval batching as in `rl.yaml`, un-sorted micro-batches); **V1** HF tuned by config plus a few lines of code (caps 1.25 x natural longest = 170 / 640 / 1,290 / 2,610 tokens instead of 384 / 1280 / 2304 / 3584, length-sorted micro-batches, one eval call per bin); **V2** V1 + vLLM for rollouts and for both evals (engineering about 1 - 2 days, section 6.4).

## 5. Results
### 5.1 Total GPU time and cost ($1.034/h), GPU not shared
| scenario | variant | low-cost h / $ | central h / $ | high-cost h / $ | wall-clock days (central) |
|---|---|---|---|---|---|
| MIN | V0 as shipped (HF generate) | 1.2 / $1.29 | **2.4 / $2.43** | 4.2 / $4.36 | 0.10 |
| MIN | V1 HF tuned | 0.9 / $0.93 | **1.5 / $1.55** | 2.6 / $2.68 | 0.06 |
| MIN | V2 vLLM rollouts+eval | 0.4 / $0.37 | **0.6 / $0.59** | 1.1 / $1.18 | 0.02 |
| BASE | V0 as shipped (HF generate) | 3.9 / $4.01 | **7.3 / $7.59** | 12.9 / $13 | 0.31 |
| BASE | V1 HF tuned | 2.6 / $2.69 | **4.3 / $4.41** | 7.1 / $7.32 | 0.18 |
| BASE | V2 vLLM rollouts+eval | 0.9 / $0.89 | **1.3 / $1.38** | 2.5 / $2.63 | 0.06 |
| GENEROUS | V0 as shipped (HF generate) | 12.8 / $13 | **24.3 / $25** | 42.2 / $44 | 1.01 |
| GENEROUS | V1 HF tuned | 8.6 / $8.91 | **14.0 / $15** | 23.0 / $24 | 0.59 |
| GENEROUS | V2 vLLM rollouts+eval | 2.6 / $2.71 | **4.0 / $4.14** | 7.5 / $7.71 | 0.17 |

### 5.2 Where the time goes (central, hours)
| scenario | variant | sampling | QE | policy+ref train | per-step other | evals | ckpt + start-up | total |
|---|---|---|---|---|---|---|---|---|
| MIN | V0 | 1.26 | 0.03 | 0.21 | 0.04 | 0.62 (2 QE + 3 dev) | 0.20 | **2.35** |
| MIN | V2 | 0.10 | 0.03 | 0.17 | 0.04 | 0.02 (2 QE + 3 dev) | 0.20 | **0.57** |
| BASE | V0 | 4.08 | 0.08 | 0.62 | 0.13 | 2.22 (8 QE + 5 dev) | 0.21 | **7.34** |
| BASE | V2 | 0.32 | 0.08 | 0.52 | 0.13 | 0.06 (8 QE + 5 dev) | 0.21 | **1.33** |
| GENEROUS | V0 | 14.12 | 0.27 | 2.05 | 0.44 | 7.12 (26 QE + 14 dev) | 0.27 | **24.27** |
| GENEROUS | V2 | 1.08 | 0.27 | 1.74 | 0.44 | 0.20 (26 QE + 14 dev) | 0.27 | **4.00** |

Total volumes (central model, V0 schedule): MIN generates 5.3 M tokens, trains on 8.2 M real tokens and scores 113 k segments; BASE 15.8 M / 24.6 M / 338 k; GENEROUS 52.5 M / 82.1 M / 1128 k. Sanity check against the SFT anchor: BASE trains on 24.6 M real tokens x 1.35 padding = 33 M padded tokens, i.e. about 1.8 h at the RL-equivalent 5070 rate (5,900 / 1.18 = 5,000 tokens/s) and 0.6 h at 3x. The model's training line (0.5 - 0.6 h) is consistent with that; everything else is sampling and evaluation.

Shipped `rl.yaml` as it stands (300 steps, 3,600 prompts, uneven bins, 12 QE evals + 7 dev evals): V0 central 9.2 (low 4.6, high 16.5) h, V2 central 1.9 (low 1.1, high 3.8) h.

### 5.3 GPU shared with the other job (about 70 GB used, about 26 GB free)
Assumptions [A]: while both jobs are active the GPU is time-sliced. Throughput-bound phases (QE, train) take 1.5 / 2.0 / 2.9 x longer (low / central / high); the many tiny kernels of decode wait for the other process's time slice, so sampling and evals take 1.5 / 2.5 / 3.5 x (HF) or 1.5 / 2 / 3 x (vLLM); CPU, I/O and start-up unchanged. If the other job's compute use is light, the factors fall toward 1; if it is a full-load training run (like the 5070 SFT), they are realistic.

| scenario | variant | alone (central) | shared low-cost | shared central | shared high-cost | central shared wall-clock days |
|---|---|---|---|---|---|---|
| MIN | V0 | 2.4 h / $2.43 | 1.8 h / $1.86 | **5.4 h / $5.58** | 13.2 h / $14 | 0.22 |
| MIN | V2 | 0.6 h / $0.59 | 0.5 h / $0.48 | **0.9 h / $0.93** | 2.3 h / $2.41 | 0.04 |
| BASE | V0 | 7.3 h / $7.59 | 5.7 h / $5.90 | **17.5 h / $18** | 42.5 h / $44 | 0.73 |
| BASE | V2 | 1.3 h / $1.38 | 1.2 h / $1.21 | **2.3 h / $2.40** | 6.1 h / $6.33 | 0.10 |
| GENEROUS | V0 | 24.3 h / $25 | 19.0 h / $20 | **58.5 h / $60** | 141.7 h / $147 | 2.44 |
| GENEROUS | V2 | 4.0 h / $4.14 | 3.7 h / $3.81 | **7.3 h / $7.54** | 19.4 h / $20 | 0.30 |

Dollar reading: if the rental runs anyway for the other job, the marginal cost of training concurrently is about zero and the only cost is wall-clock; if the rental is billed for this job alone, wall-clock hours x $1.034 applies (table). Waiting for the other job to end and then running alone costs the "alone" column but needs the job's end time (unknown here). Memory in the 26 GB that is free: section 6.2. Running concurrently is attractive only for MIN/BASE under V2; GENEROUS under V0 would run for days.

## 6. How to max out the 96 GB (and what fits in 26 GB)
### 6.1 Principles from the numbers above
1. **HF `generate` time is nearly independent of the number of sequences** until the batch is a few hundred (a decode step is launch-bound: about 25 ms whether N is 16 or 400; the KV-copy term grows only to ~12 ms at N = 160, ctx 2,500). So under V0/V1, *more sequences per generate call is almost free*: doubling the prompts per step (k = 2: P = 40, N = 320) costs about +27 % in total (BASE 7.3 -> 9.3 h, section 7) instead of +100 %. The limit is KV memory: 18 KB/token x N x (prompt + cap).
2. **Gradient checkpointing stays off** (already `false` in `rl.yaml`): activations cost about 0.7 MB per token (estimate [A]: 18 layers x ~40 KB; fp32 residual stream, bf16 matmul copies) so a 49 k-token micro-batch needs about 37 GB. Checkpointing would cost +25 % of training FLOPs for no benefit on 96 GB.
3. **The 262 k-vocabulary logits are the memory trap**: never materialise `[tokens x 262,144]`. `completion_logprobs` already chunks the LM head per `loss_chunk_tokens` and checkpoints each chunk: peak per chunk = chunk x 262,144 x (2 bf16 + 4 fp32 + 4 log-sum-exp temp + 6 backward) B = chunk x 4.2 MB: 512 -> 2.1 GB, 2,048 -> 8.6 GB. A micro-batch of 49 k completion tokens un-chunked would be 51 GB of fp32 logits alone. HF `generate` holds `[N x 262,144]` fp32 per step (160 -> 168 MB, 512 -> 537 MB); check that the installed transformers uses `logits_to_keep=1` at prefill (older versions would build `[N x prompt x 262,144]`: 71 GB at N = 160, prompt 850).
4. **`micro_batch_seqs: 16` is the real limiter** (it caps sequences per forward at 16, i.e. only 2 k tokens in b64 and 29 k in b1024): raise it so `micro_batch_tokens` governs.

### 6.2 Recommended settings (all other keys as in `rl.yaml`)
| key | shipped | 96 GB dedicated, V2 | 96 GB dedicated, V0/V1 | ~26 GB shared, V2 |
|---|---|---|---|---|
| `prompts_per_step_by_bin` (all bins) | 32/16/8/4, mixed 8 | 20 (one per language), mixed 20 | 40 (k = 2) all bins, mixed 40 | 20 |
| `group_size` | 8 | 8 (16 is affordable under V2: +51 % cost) | 8 | 8 |
| `micro_batch_seqs` | 16 | 320 | 320 | 64 |
| `micro_batch_tokens` | 24,576 | 49,152 (try 65,536) | 49,152 | 8,192 |
| `loss_chunk_tokens` | 512 | 2,048 | 2,048 | 512 |
| `gradient_checkpointing` | false | false | false | false |
| `param_dtype` / `ref_dtype` / `autocast_dtype` | fp32 / fp32 / bf16 | same (keep: the step-0 KL assertion needs equal dtypes) | same | same |
| `scorer.batch_size` | 32 | 256 (encoder activations < 3 GB; sorted by length) | 256 | 64 |
| `max_new_tokens_by_bin` | 384/1280/2304/3584 | 192/640/1344/2624 | same | same |
| `eval` calls | 1/2/4/8 per bin | one call of 32 per bin (vLLM) | one call of 32 per bin | same |
| LR warm-up | 10 | 10 (MIN: 4) | same | same |
| vLLM | none | `gpu_memory_utilization` 0.20 (19 GB), `max_model_len` 4,096, prefix caching on | none | 0.07 (6.5 GB) |

### 6.3 Memory budget (GiB-order, [A] for activations/KV, [M] for parameters: 268 M parameters)
| item | 96 GB dedicated, V2 | 96 GB, V0 (HF generate N = 320, k = 2) | ~26 GB shared, V2 |
|---|---|---|---|
| policy fp32 + grads + AdamW m,v | 4.3 | 4.3 | 4.3 |
| frozen reference fp32 | 1.1 | 1.1 | 1.1 |
| AfriCOMET-QE fp16 + activations | 1.1 + 2.5 | 1.1 + 2.5 | 1.1 + 1.0 |
| training activations (micro-batch tokens x ~0.75 MB) | 37 (49 k tokens) | 37 | 6 (8 k tokens) |
| LM-head chunk (logits, temps, backward) | 8.6 (chunk 2,048) | 8.6 | 2.1 |
| vLLM engine (0.54 GB weights + KV) | 19 budget (about 5 GB actually used: 3 of 18 layers are global, the rest windowed at 512) | - | 6.5 |
| HF KV cache for generation (18 KB/token x 320 x (850 + 2,624)) | - | 20 (+ torch.cat transient) | - |
| CUDA contexts, fragmentation | 4 | 4 | 3 |
| **total** | **about 78 GB** (18 GB margin) | **79 GB if every pool were live at once; about 56 GB typical peak** because the caching allocator reuses the generation KV blocks for training activations (call `torch.cuda.empty_cache()` between phases) | **about 25 GB of 26: no margin; start with `micro_batch_tokens` 4,096** |

### 6.4 Engineering needed (not done here)
* **Stratified/quota schedule** (`PromptSchedule`: new mode "one prompt per language per step, bin rotation in the mixed stage"; plus a per-scenario prompt subset file drawn with a fixed seed from `rl_prompts_v2_tok`, excluding the hash-selected eval prompts): about 2 h, CPU-testable with the existing tests.
* **vLLM rollouts** (V2): load the policy into a vLLM engine in the same process (needs `transformers` compatible with `unbabel-comet`, which pins `transformers < 5`; if the environments cannot be merged run vLLM as a sidecar process in `/workspace/venv_vllm` and push each step's bf16 weights through `/dev/shm`, which is what `t_sync = 1 s` assumes), sample with temperature 1.0, `top_p 1`, `top_k -1`, then keep the existing HF teacher-forced log-probs for the loss (the loss never uses the sampler's log-probs: ratio = 1 by construction). Caveat: the sampling distribution (bf16 vLLM kernels) differs slightly from the fp32-weights/bf16-autocast HF scoring pass; the estimator is then very slightly off-policy (the repo's `check_sampler_logprobs` gap check would measure it on a few steps). Also move `evaluate_policy` / `generate_translations` onto the same engine. Estimate 1 - 2 days including the weight-sync test. What it saves (V0 -> V2, central): MIN $2.43 -> $0.59, BASE 7.3 -> 1.3 h ($7.59 -> $1.38), GENEROUS 24.3 -> 4.0 h ($25 -> $4.14), i.e. about 85 - 95 % of the sampling and evaluation time.
* Whether it pays for itself: in rental dollars vLLM saves about $2 / $6 / $21 (MIN / BASE / GENEROUS), which does not cover 1 - 2 days of engineering; it pays in wall-clock (GENEROUS 1 day -> 4 hours) and iteration speed. Under V1 (no engineering beyond config) BASE costs $4.41 and GENEROUS $15.

## 7. Sensitivity (central, scenario BASE unless stated; V0 | V2; [low - high])

**Group size** (same prompts and steps):
| case | steps | prompts | completions | V0 central [low - high] | V2 central [low - high] | note |
|---|---|---|---|---|---|---|
| G = 4 | 192 | 3,840 | 15,360 | 6.4 h / $6.57 [$3.36 - $12] | 1.0 h / $1.02 [$0.64 - $2.00] | HF sampling time per step barely changes with G; train/QE scale with G |
| G = 8 | 192 | 3,840 | 30,720 | 7.3 h / $7.59 [$4.01 - $13] | 1.3 h / $1.38 [$0.89 - $2.63] | HF sampling time per step barely changes with G; train/QE scale with G |
| G = 16 | 192 | 3,840 | 61,440 | 9.3 h / $9.62 [$5.30 - $16] | 2.0 h / $2.08 [$1.37 - $3.89] | HF sampling time per step barely changes with G; train/QE scale with G |

**Prompts per language x bin** (staged and mixed steps scaled together, P = 20):
| case | steps | prompts | completions | V0 central [low - high] | V2 central [low - high] | note |
|---|---|---|---|---|---|---|
| 24 per cell | 96 | 1,920 | 15,360 | 3.8 h / $3.91 [$2.07 - $6.91] | 0.8 h / $0.78 [$0.50 - $1.54] |  |
| 48 per cell (BASE) | 192 | 3,840 | 30,720 | 7.3 h / $7.59 [$4.01 - $13] | 1.3 h / $1.38 [$0.89 - $2.63] |  |
| 96 per cell | 384 | 7,680 | 61,440 | 14.5 h / $15 [$7.89 - $26] | 2.5 h / $2.56 [$1.67 - $4.81] |  |
| 192 per cell | 768 | 15,360 | 122,880 | 28.4 h / $29 [$16 - $51] | 4.8 h / $4.92 [$3.22 - $9.16] |  |

**Prompts per language per step k** with the same prompts per cell (fewer, larger steps) | with the same number of steps (more prompts):
| case | steps | prompts | completions | V0 central [low - high] | V2 central [low - high] | note |
|---|---|---|---|---|---|---|
| k = 1 (P = 20/step), same prompts | 192 | 3,840 | 30,720 | 7.3 h / $7.59 [$4.01 - $13] | 1.3 h / $1.38 [$0.89 - $2.63] | fewer updates: statistically weaker per prompt |
| k = 2 (P = 40/step), same prompts | 96 | 3,840 | 30,720 | 4.8 h / $4.92 [$2.71 - $8.48] | 1.1 h / $1.14 [$0.74 - $2.17] | fewer updates: statistically weaker per prompt |
| k = 4 (P = 80/step), same prompts | 48 | 3,840 | 30,720 | 3.5 h / $3.59 [$2.07 - $6.08] | 1.0 h / $1.02 [$0.67 - $1.95] | fewer updates: statistically weaker per prompt |
| k = 2 (P = 40/step), same steps | 192 | 7,680 | 61,440 | 9.3 h / $9.62 [$5.30 - $16] | 2.0 h / $2.08 [$1.37 - $3.89] | more evidence per update |
| k = 4 (P = 80/step), same steps | 192 | 15,360 | 122,880 | 13.2 h / $14 [$7.90 - $23] | 3.4 h / $3.50 [$2.34 - $6.42] | more evidence per update |

**Bins included** (48 prompts per remaining cell):
| case | steps | prompts | completions | V0 central [low - high] | V2 central [low - high] | note |
|---|---|---|---|---|---|---|
| all four bins | 192 | 3,840 | 30,720 | 7.3 h / $7.59 [$4.01 - $13] | 1.3 h / $1.38 [$0.89 - $2.63] |  |
| drop b1024 (stages b64-b512, mixed over 3) | 144 | 2,880 | 23,040 | 3.9 h / $4.04 [$2.01 - $7.56] | 0.8 h / $0.78 [$0.49 - $1.55] | saves 47 % of V0 cost |
| drop b512 and b1024 | 96 | 1,920 | 15,360 | 2.0 h / $2.05 [$1.03 - $3.98] | 0.4 h / $0.45 [$0.27 - $0.95] |  |
| b64 only (96 steps) | 96 | 1,920 | 15,360 | 1.6 h / $1.62 [$0.82 - $3.09] | 0.4 h / $0.37 [$0.22 - $0.82] | cheapest, but trains nothing on paragraphs |

**Owner boost** (pcm x3, 8 low-resource languages x2, rest x1 = 30 language-units):
| case | steps | prompts | completions | V0 central [low - high] | V2 central [low - high] | note |
|---|---|---|---|---|---|---|
| BASE with boost | 192 | 5,760 | 46,080 | 8.3 h / $8.60 [$4.66 - $15] | 1.7 h / $1.73 [$1.13 - $3.26] | +50 % prompts |

Observations: (i) under V0 cost follows the number of *steps* (every step pays the decode horizon of the longest sequence, almost independent of how many sequences share it), so with HF it is cheaper to use fewer, larger steps (k = 2 or 4) if the statistical argument allows; under V2 cost follows completions plus a fixed per-step tail. (ii) Dropping b1024 saves about half of V0 cost; b1024 is also where the flat-group (all-failed) problem is worst. (iii) Group size 16 costs +27 % (V0) / +51 % (V2) over 8; group 4 saves 13 % / 26 %.

## 8. SFT remainder on the same card (`runs/sft_unimax`)
Live state at 2026-10-17: the log shows **step 13,910 of 27,638** (the step counter continues from the 4,750 stop of the previous run, so this run is 22,888 steps in total), 21.4 s/step, 5,822 tokens/s [M, `logs/sft_unimax.log` / `runs/sft_unimax/metrics.jsonl`]. The task text assumed about 7,500 done (20,138 left); that is out of date, so both are given.

Per-step time on the 96 GB card: 22 s x 0.875 (no body recompute: gradient checkpointing can be off, micro-batch 8 x 4096 fits) / speed ratio r: **5.1 s (r = 3.8) / 6.4 s (3.0) / 9.6 s (2.0)**.

| remaining steps | on the 5070 (21.4 - 22 s) | PRO 6000: low-cost | central | high-cost |
|---|---|---|---|---|
| 13,728 (observed: 27,638 - 13,910) | 84 h (3.5 d) | 19 h / $20 | 24 h / $25 | 37 h / $38 |
| 20,138 (27,638 - 7,500, as stated in the task) | 123 h (5.1 d) | 28 h / $29 | 36 h / $37 | 54 h / $56 |

So finishing the SFT on the rental would cost about $25 (13,728 steps) - $37 (20,138 steps) central ($20 - $56 over the range) and take 1.0 - 1.5 days instead of 3.5 - 5.1 days on the 5070. RL needs the SFT result as its starting policy, so the two are sequential, not concurrent. Muon vs AdamW optimizer time is neglected; this assumes the whole 22 s/step is GPU-bound and scales with the same ratio r as RL training, which is the weakest part of the SFT estimate.

## 9. Measured vs assumed, and what to measure first
**Measured/derived from files or data in this task:** code behaviour of the RL path; prompt pool sizes and window statistics; expansion factors (pairs and windows, 20 languages); QE pair-length population; reward CPU time; SFT live throughput and step counter; engine/hardware specs of the PRO 6000 Max-Q (spec pages).
**From the owner's earlier logs:** SFT 6,100 -> 5,820 tokens/s; QE 290 - 350 pairs/s single / 480 pairs/s three processes (5070, fp16).
**Assumed:** every GPU speed on the PRO 6000, HF decode latency and tail probability, vLLM latencies, padding, activations memory, GPU time-slicing factors, reward sigma, number of flat groups, start-up/IO times.

**A 30-minute measurement plan on the rented GPU (before committing to a scenario), cost about $0.50:**
1. `generate` of 160 sequences, b1024 prompts, SFT checkpoint, temperature 1.0, record seconds per decode step and the length distribution / fraction hitting 3,584: replaces `t0`, `q_cap`.
2. `CometScorer` on 3,000 sentence pairs from `gemma/scored` at batch 32 and 256: replaces `QE_tps x r_qe`.
3. One fwd+bwd+ref pass of a 49 k-token micro-batch: replaces `r`, `pad` and the activation memory estimate (`torch.cuda.max_memory_allocated`).
Then update the table in section 4 and rerun the model; the tables above change only through those parameters.

## 10. Caveats
* The cost is cheap enough that the real risks are not money: (a) the never-run code path (budget the shake-down), (b) 16 prompts per cell cannot show a per-language gain, (c) QE-reward reliability (D-24: 0.47 chance-level on appended text; the length/lexical penalties are in the reward) means the dev chrF++ selection, not the cost, decides whether RL is kept.
* The 4.8 M-prompt pool, the contamination-filtered dev set (`bench_dev_clean.jsonl`: 10,043 rows, confirmed) and the Hub repo names are untouched by this report.
* Sources for the PRO 6000 Max-Q specification: https://lenovopress.lenovo.com/lp2364-thinksystem-nvidia-rtx-pro-6000-blackwell-max-q-workstation-edition-gpu and https://www.pugetsystems.com/parts/Video-Card/NVIDIA-RTX-PRO-6000-Blackwell-Max-Q-Workstation-Edition-96GB-PCI-E-16222/ (the 5070 figures were not re-checked).
* All low/high columns combine the extremes of every parameter at once and are therefore an envelope, not a confidence interval.
