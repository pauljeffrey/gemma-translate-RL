# vLLM rollouts for the Gemma-3-270m-it en->African GRPO: what was built, what was measured (2026-10-17)

Decision log entry: `gemma_mt/DECISIONS.md` D-28. Code: `gemma_mt/gemma_mt/rl/vllm_engine.py`, `vllm_sampler.py`, plumbing in `train_rl.py`, `eval/eval_translation.py`; keys in `configs/rl.yaml`; GPU test `gemma_mt/scripts/vllm_smoke.py`; CPU tests `gemma_mt/tests/test_vllm_sampler.py` (17 tests, fake / HF-backed engines and a real worker process over a real socket).

**Evidence labels:** [M] measured on the rental (RTX PRO 6000 Blackwell Max-Q, sm_120, driver 595.71 / CUDA 13), [C] checked by reading code, [A] assumption.
**Conditions of every GPU number:** the GPU is SHARED with a foreign training job (about 70 GB, 100 % utilisation) and I was limited to about 10 GB: all throughput numbers are **lower bounds** for an idle card, for HF and for vLLM alike. Tests used the base Gemma-3-270m-it architecture (the SFT checkpoint `<HF_ORG_LEGACY>/gemma3-270m-translate` is private: 401 from the new instance; `google/gemma-3-270m-it` is gated: the ungated mirror `unsloth/gemma-3-270m-it` was used; same architecture and config, and the engine's weights are overwritten by the first sync anyway). Nothing was trained, nothing was pushed, nothing was copied to the laptop.

## 0. Result

| question | answer |
|---|---|
| Does vLLM 0.31 run Gemma-3-270m on sm_120? | **Yes** [M]. vLLM picks `FLASH_ATTN` (FlashAttention 2) among FLASH_ATTN / FLASHINFER / TRITON_ATTN / FLEX_ATTENTION, runs the 5:1 sliding-window/global layer pattern through its hybrid KV-cache manager (6 groups of 3 layers), 262k vocab, head_dim 256, V2 model runner. `TRITON_ATTN` also runs and gives the same sync figures (section 3.4). FlashInfer *sampling* is unavailable on sm_12x with CUDA 13 (vLLM falls back to its own sampler; harmless). |
| Where do trainer and vLLM live? | Both layouts work. **Default: `vllm_mode: worker`** = engine in a child process of `/workspace/venv_vllm/bin/python` (torch 2.13.0+cu130, vllm 0.31.0), trainer unchanged in `venv_gemma` (torch 2.14). **`inprocess`** = one interpreter that has both: works in `venv_vllm` (+ pandas/pyarrow from a `--target` dir), identical outputs to worker mode, sync 30x faster (0.03 s vs 0.8-1.1 s). Why worker is the default: section 2. |
| Weight sync | **Correct** [M]: mean abs difference of per-token log-probs engine vs HF = 0.010-0.021 (p99 0.14-0.24, max 0.44), which is the size of HF's *own* autocast-vs-fp32 difference (0.009-0.014 / p99 0.13-0.16). Two consecutive syncs of different weights, a third with the same prompts (prefix cache dropped), a 5x-stronger perturbation, a config written by transformers 5.19 (`rope_parameters`): all consistent. |
| Sync time | worker **0.7-1.5 s** (median ~1.1 s) per step [M]; in-process **0.03 s** [M]. |
| Speedup vs HF `generate` | **8.5-10x wall time per rollout call** on the same prompt sets (bins b64..b1024), up to ~25x tokens/s when HF hits a runaway sample that holds its whole batch to `max_new_tokens` [M, lower bound, shared GPU]. Section 4. |
| Memory | engine process **4.6 GiB** with a 2 GiB KV cache (weights 0.53 GiB, graphs/activations/context the rest); sleeping: 1.5 GiB; the largest batch (160 x b1024) fits in that cache (it queues, does not fail). Peak of engine + my test trainer 10.1 GiB. [M] |
| Stop / truncation / `post_stop_tokens` | as HF [M]: stop id included, `finished` iff a stop id is present, `max_new_tokens` truncation reported, `post_stop_tokens == 0`. Section 5. |
| Old log-probs from the sampler? | **Not needed** [C+M]: the loss recomputes log p with the trainer. Section 6. |
| Failure behaviour | engine killed / start-up cannot fit / prompt too long -> `SamplerError`, run stops; `sampler: hf` restores the old path. Section 7. |

## 1. What was built

* `rl/vllm_engine.py` (stdlib at import time): `EngineConfig`, `SamplingSpec`, `RawCompletion`, `SamplerError`; `VLLMEngine` (vLLM `LLM` in this process, prompts as token ids, `skip_tokenizer_init`, `generation_config="vllm"`, `stop_token_ids`, per-prompt seeds, in-place `load_weights` through `LLM.apply_model`, completeness check, prefix-cache reset, optional level-2 sleep); `WorkerEngine` (child process = `python -m gemma_mt.rl.vllm_engine`, unix-socket pickles, weights through one mmap'd shared-memory file on /dev/shm, dead / hung / failed worker -> `SamplerError` with the tail of `vllm_worker.log`).
* `rl/vllm_sampler.py`: `VLLMSampler` (versioned `sync`, `generate` -> the same `Rollout` as `generate_completions`, contract checks), `HFSampler` (old path), `derive_seeds`, `policy_bf16_weights`, `HFBackedEngine` (CPU stand-in used by the tests and `train_rl --smoke-vllm`), `build_vllm_sampler`.
* `train_rl.py`: `sampler`/`vllm_*` keys, engine started BEFORE the policy/reference/optimizer are loaded, `sync_policy()` before each rollout/eval whose weights changed (version = optimizer steps), eval passes (held-out reward AND dev chrF++ via `generate_translations(sampler=)`) on the same engine at temperature 0, `t_sync`/`t_gen`/`gen_tokens_per_s`/`sampler_logp_gap_mean` metrics, the consistency check (section 6), `train_rl(..., vllm_sampler=)` injection for tests.
* `configs/rl.yaml` (`sampler: vllm`, `vllm_mode: worker`, `vllm_python`, `vllm_gpu_memory_utilization: 0.25`, `vllm_kv_cache_gib`, `vllm_max_model_len: 8192`, `vllm_max_num_seqs: 256`, `vllm_enforce_eager`, `vllm_sleep`, `vllm_attention_backend`, `vllm_sync_dir`, timeouts, `vllm_check_every: 25`, `vllm_logp_gap_tol: 0.25`), `scripts/vast_train_rl.sh` (cache dirs on disk, `VLLM_PYTHON`).
* Tests: the CPU suite is green for everything of mine (`cd gemma_mt && ../.venv-comet/bin/python tests/run_tests.py`: 247 pass, 0 fail, 3 env-gated skips; an intermediate run showed one failure in `test_reward_shaping` from the concurrent reward/calibration work, gone in the final run).

## 2. Environment: (a) in-process vs (b) worker, with evidence

| | (a) `inprocess` | (b) `worker` (default) |
|---|---|---|
| trainer interpreter | must be `venv_vllm` (torch 2.13, transformers 5.17, numpy 2.3.5) | `venv_gemma`, untouched |
| what had to be added | pandas 3.0.6 + pyarrow 26 via `uv pip install --target /workspace/sft_build/pylibs_vllm --no-deps ...` (215 MB), comet/sacrebleu/lightning through a symlink farm to `pylibs` (shadowing is why `pylibs` cannot simply be put on the PYTHONPATH of a foreign venv) | nothing |
| CPU smoke (`train_rl --smoke`) in that env | passes [M] | n/a |
| GPU smoke (`vllm_smoke.py`) | passes, numerically identical to (b) (same log-prob statistics, same samples) [M] | passes [M] |
| sync per step | 0.03 s | 0.7-1.5 s (GPU -> shm 0.3-0.6 s, load in the worker ~0.7 s) |
| extra GPU memory | none (one CUDA context) | ~0.5 GB (second context) |
| failure domain | engine and trainer share one process | engine failure leaves the trainer able to print a clean error with the engine's log |
| risk | the trainer's environment is someone else's read-only venv with different torch/transformers than the one the project is developed on | one more process, one socket |

Choice: (b). It costs ~1 s per step (under 5 % of every step except the b64 stage, where generation itself takes ~2 s) and avoids running the trainer in a foreign environment. Switch with `vllm_mode=inprocess` once a single environment exists (then also set `scorer.type: worker` unless comet works there, next paragraph).

**Finding that affects the whole RL path, independent of this work:** `comet` 2.2.7 (`pylibs/comet/encoders/xlmr.py:95`) fails with `ValueError: not enough values to unpack (expected 3, got 2)` on transformers 5.x, in `venv_gemma` (5.19.0) and in `venv_vllm` (5.17.0) alike [M, CPU run of `CometScorer(...)` on two pairs]. `scorer.type: comet` therefore cannot run in either environment as they are today; `scorer.type: worker` needs an interpreter with transformers < 5 (or a patched comet). Not touched (not my files), but it blocks the first real RL run.

Attention backend: vLLM chose `FLASH_ATTN` by itself on sm_120; no override needed. Notes from the logs: `FlashInfer top-p/top-k sampling unavailable: SM 12.x requires CUDA >= 12.9` (falls back; irrelevant for top_p = 1 / top_k = 0), `Triton kernel JIT compilation during inference: _topk_log_softmax_kernel` once, the first time log-probs are requested (one-off latency spike).

## 3. Weight-sync correctness [M]

`vllm_smoke.py --only sync`: policy = base weights + gaussian noise (0.01 x each parameter's std, NOT cumulative: W0 + noise_k). Sync k pushes the bf16 copy; 16 prompts from `rl_prompts_v2/yor.parquet` (bin b64) built with `fmt.build_prompt`; the engine decodes greedily 48 tokens and returns the log-prob of each token; HF then scores the engine's token ids teacher-forced with `losses.completion_logprobs` (the trainer's own function) in three precisions.

| weights | engine vs HF fp32 (mean / p99 / max abs dlogp) | vs HF fp32 + bf16 autocast (the trainer) | vs HF pure bf16 | greedy sequences identical to the trainer-path HF greedy (48 tokens) |
|---|---|---|---|---|
| sync 1 | 0.0154 / 0.185 / 0.41 | 0.0191 / 0.219 / 0.41 | 0.0210 / 0.237 / 0.44 | 8/16 |
| sync 2 (different noise) | 0.0097 / 0.144 / 0.36 | 0.0136 / 0.230 / 0.35 | 0.0121 / 0.157 / 0.30 | 8/16; 16/16 completions differ from sync 1 |
| sync 3, same 16 prompts as before | - | 0.0144 / 0.148 / 0.24 | - | 16/16 differ from sync 2 (a stale prefix cache would reproduce the old text) |
| 5x noise (nearly garbage model, huge logits) | 0.117 / 0.74 / 1.03 | - | 0.115 / 0.59 / 1.61 | - |
| reference: HF autocast vs HF fp32, same ids | 0.0140 / 0.156 / 0.24 and 0.0094 / 0.131 / 0.32 | | | |

Reading: a stale, partial or mis-mapped engine would give differences of the size of the weights' effect (several nats); the measured 0.01-0.02 is bf16 arithmetic, the same size as the trainer's own autocast-vs-fp32 difference. Half of the greedy sequences differ somewhere in 48 tokens (median first divergence at token 22 / 36 for syncs 1 / 2): bf16 near-ties, not a sync error, since the per-token log-prob differences stay at the HF-autocast-vs-fp32 level. The same figures with a `config.json` written by transformers 5.19 (per-layer `rope_parameters`, run `--engine-model .../cfg519`) are the same to 3 digits (0.0185 / 0.0135), so vLLM reads the new rope format correctly (global theta 1e6, local 1e4).
Sync time per call (worker): 1.43 / 0.99 s in this run; across 8 consecutive steps 0.93-1.16 s; with sleep 0.6-1.5 s including the wake-up. In-process: 0.03 s.

### 3.1 Sampling distribution (no hidden truncation) [M]
4,096 first-token samples (`max_tokens=1`) per case:
* T = 2.0 (high entropy): mass beyond HF's 64 most likely tokens 0.3975, fraction of vLLM samples beyond them 0.4036 (sigma 0.0076): no top-k; 1,411 distinct tokens were drawn. `top_k=0` / `top_p=1` mean the full distribution.
* T = 1.0 (peaked): empirical frequency vs the engine's own reported probabilities: max |z| 0.3 and 2.1 over the tokens with >= 30 expected draws. Distance of the empirical distribution to HF's softmax: TV 0.018 / 0.055 (fp32), 0.038 / 0.040 (autocast), 0.086 / 0.049 (pure bf16), while HF fp32 vs HF pure bf16 are themselves 0.068 / 0.020 apart: the engine's sampling distribution differs from the trainer's scoring distribution by bf16 logit noise.

### 3.2 Determinism [M]
Same `key` (= optimizer step) -> identical ids; another key -> different ids; no two completions of a group are identical (seed + 0..7 per prompt, prompts 1024 apart). Worker and in-process engines produced the same samples and log-prob statistics. Bitwise reproducibility across different batch compositions or GPUs is not promised by vLLM.

### 3.3 Real reward code on the engine's output [M]
`rollout_and_reward` (real tokenizer, real `TranslationReward`, fake QE scorer) with the engine: 64 samples in 1.8 s vs 14.4 s with HF; status mix identical (base model never writes the iso tag -> `bad_prefix`; 2/64 `no_end_of_turn`), rewards finite, `post_stop_tokens` 0. Dev-benchmark path (`generate_translations(sampler=)`, greedy, 16 rows): 9/16 identical hypotheses to the HF path, same number finished.

### 3.4 Backends
`FLASH_ATTN` is vLLM's own choice (run with `vllm_attention_backend: null`). Forced `TRITON_ATTN` (`--attention-backend TRITON_ATTN`, log: "Using AttentionBackendEnum.TRITON_ATTN backend"): engine starts in 33 s, sync figures 0.0226 / 0.0115 (mean abs dlogp vs HF autocast, syncs 1 / 2), p99 0.25 / 0.14, 16/16 completions change at the second sync, third sync with the same prompts 0.0156: a working fallback if FLASH_ATTN ever breaks on a new vLLM. FLASHINFER and FLEX_ATTENTION were not tried.

## 4. Throughput [M] (shared GPU, lower bounds)

Setup: G = 8 completions per prompt, temperature 1, `max_new_tokens` per bin from `rl.yaml` (b64 384, b256 1280, b512 2304, b1024 3584), prompts drawn from the yor pool for the given bin, weights re-synced before every call (as in training), the base model (so lengths are those of an untrained translator: mean 118 / 230 / 355-420 / 674-874 tokens for b64 / b256 / b512 / b1024; the SFT model's lengths will differ), engine KV cache 2 GiB, HF limited to 8.5 GiB of torch memory.

| bin x prompts | samples | vLLM: tokens / seconds / tok/s | HF: tokens / seconds / tok/s | wall-time ratio | notes |
|---|---|---|---|---|---|
| b64 x 20 | 160 | 18,856 / 1.9 / 9,961 | 19,545 / 18.7 / 1,046 | 9.8x | earlier run 1.7 s vs 16.0 s (9.4x) |
| b256 x 10 | 80 | 16,630 / 6.1 / 2,735 | 19,268 / 52.5 / 367 | 8.6x | |
| b256 x 20 | 160 | 36,838 / 6.7 / 5,462 | OOM inside the 8.5 GiB cap | - | HF needs > 8.5 GiB torch memory for 160 x 1,280 tokens (KV of all 18 layers for all rows) |
| b512 x 8 | 64 | 23,083 / 5.0 / 4,571 | 24,024 / 46.9 / 512 | 9.4x | earlier run 9.0 s vs 80.6 s (9.0x, both drew a runaway to 2,304) |
| b512 x 20 | 160 | 56,803 / 9.7 / 5,836 | not run | - | |
| b1024 x 4 | 32 | 21,531 / 4.0 / 5,347 | 24,492 / 115.9 / 211 | 29x (25x tok/s) | HF drew a sample that never stopped (3,584 tokens) and held its batch for 116 s; vLLM's draw had none (max 815). Earlier run with a runaway on both sides: 13.7 s vs 117.5 s (8.6x) |
| b1024 x 20 | 160 | 139,812 / 23.6 / 5,920 | not run | - | KV-limited (2 GiB): rows queue; still no failure |

* Summary: **~9x faster per rollout call on equal prompts; HF's cost per call is set by its longest sample, vLLM's by the total tokens**, so the gap widens when a policy sometimes does not stop (exactly what the `no_end_of_turn` class punishes). vLLM sustains 5-10 k tokens/s on a card that is 100 % busy with someone else's training.
* **Steady state of the loop** (8 x [new weights -> sync -> 20 prompts x 8 at b64]): sync 0.93-1.16 s, generation 1.9-3.0 s, process memory constant (engine 4,634 MiB, test trainer 5,706 MiB, no growth).
* Start-up: ~70 s the very first time (Triton/compile caches empty), **14-30 s** afterwards (caches in `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR` on disk). First request of a process pays a one-off JIT (0.1 s when the sections before it already warmed the kernels; the cold probe took 75 s with `enforce_eager` and empty caches).
* Memory [M, nvidia-smi per pid]: engine process 4,634-4,668 MiB with `kv_cache_gib=2.0` and CUDA graphs; **1,472 MiB after level-2 sleep** (`vllm_sleep: true`: sleep 0.05 s, wake-up 0.13-0.26 s, sync after wake-up included in the 0.6-1.5 s above; closing a sleeping engine used to abort inside vLLM's cumem allocator, fixed by waking it first); HF phase: 8.4-9.4 GiB total for 160 x 384 up to 32 x 3,584 tokens. Peak of the whole test (engine + test trainer that also keeps an fp32 policy and bf16 reference copies): 10.1 GiB.
* Effect on the cost estimate (`gemma_rl_cost_estimate_2026-10-17.md`, which called every sampling number [A]): the HF figure that report could not measure is now measured: 1,046 / 367 / 512 / 211 tokens/s at b64 / b256 / b512 / b1024 for 160 / 80 / 64 / 32 samples on a busy card; vLLM is ~9x faster at the same prompt sets. The shipped schedule (32/16/8/4 prompts per bin) and the stratified one (20 prompts per bin) both fit the engine.

## 5. Stop-token handling [M]

`stop_token_ids = [<end_of_turn>=106, <eos>=1]`, `ignore_eos` off. Checked on 128 samples per budget:
* budget 64: 50 finished, all end in exactly one stop id (106 for every one of them), none has a stop id earlier; the 78 unfinished have exactly 64 tokens and no stop id. budget 6: 0 finished, all 6 tokens. Truncation is `finish_reason == "length"` and is reported as `finished = False`, which is what the reward's `no_end_of_turn` class counts.
* An additional stop id (the most frequent generated token, 506) stops 31 completions and is included as the last id, like HF.
* The stop id is part of `token_ids` in vLLM 0.31 (`stop_token_ids` keep the token). If a vLLM version reported it only in `stop_reason`, the sampler appends it; if it is in neither place the sampler raises (the reward would count it as truncation and the loss would lack the target).
* A stop id exactly at the last allowed position counts as finished (like HF); vLLM 0.31 checks stop before length [C].
* `post_stop_tokens` is computed with the HF rule and is 0 in every sample: the engine stops at the stop id, so D-26's `post_eos_rubbish` stays unreachable with this sampler, as with HF.
* Token ids are returned exactly as generated: prompts enter as ids (`prompt_token_ids`), `skip_tokenizer_init=True` means the engine has no tokenizer and never decodes; the trainer's `assemble_batch` / `completion_logprobs` consume these ids directly.

## 6. Where the trainer gets old log-probs [C+M]

`train_rl.py`: with `num_updates_per_batch == 1` no old log-probs are computed at all: `grpo_token_loss(..., old_logp=None)` uses `logp.detach()` (ratio == 1; the loop asserts mean ratio == 1 to 1e-5 every step). With more than one update per batch the loop computes `old_lp` itself with `completion_logprobs(policy, ...)` before the first update. The reference log-probs come from the frozen fp32 copy. So the sampler's log-probs are never used in the loss; they are used only for `check_sampler_logprobs` (HF) and, with vLLM, for the consistency check: every `vllm_check_every` steps (default 25) the engine returns the log-prob of every sampled token (`logprobs=0`), the trainer compares them with the recomputed ones and **raises** when the mean absolute gap exceeds `vllm_logp_gap_tol` (0.25; healthy ~0.015). The CPU test `test_cpu_rl_loop_through_the_vllm_code_path` runs the whole loop with this check on at every step (gap < 0.05, ratio 1, engine synced exactly once per policy version: 5 pushes in a 4-step run with evals).

Because the sampling distribution is vLLM-bf16 and the scoring distribution is fp32-under-autocast, the rollouts are on-policy only up to bf16 noise (section 3). No importance-sampling correction is applied (it would be a one-line change in `grpo_token_loss` if the check ever trips).

## 7. Failure behaviour [M]

| event | result |
|---|---|
| prompt + `max_new_tokens` > `vllm_max_model_len` | `SamplerError` naming the longest prompt; nothing truncated |
| worker process killed with SIGKILL during the run | next call raises `SamplerError` after 0.3 s ("connection lost", tail of `vllm_worker.log`); later calls raise too |
| engine cannot start (KV cache 0.005 GiB) | `SamplerError` after 14 s with vLLM's traceback; the same for a missing checkpoint (unit test) |
| `gpu_memory_utilization` x total > FREE memory (seen on the shared card: 24 GiB free vs 0.92 x 95) | vLLM refuses to start: `ValueError ... Free memory ... less than desired GPU memory utilization` -> `SamplerError` |
| engine answers inconsistently (finish reason vs ids, missing stop id, too many tokens, wrong count) | `SamplerError` (unit tests) |
| `generate` before the first `sync` / after `sleep` | `SamplerError` |
| trainer exits or crashes | worker sees EOF on the socket and exits; `atexit` terminates it; the /dev/shm directory is removed |
| `sampler: hf` | the old code path, untouched |

There is no code path from a vLLM failure to HF generation.

## 8. Commands

Environment on a fresh rental (not re-run here: `venv_vllm` is 8.0 GB and the box had 7.5 GB free; the versions are those found in the existing `venv_vllm`): 
```bash
uv venv --python 3.12 /workspace/venv_vllm
uv pip install --python /workspace/venv_vllm/bin/python vllm==0.31.0 --torch-backend cu130     # torch 2.13.0+cu130, flashinfer-python 0.7.0.post1, triton 3.7.1, transformers 5.17.0
```
The trainer environment is unchanged (`venv_gemma`: torch 2.14, transformers 5.19). **/dev/shm is mounted noexec** on this host: vLLM/Triton compile caches must be on a real disk (the shm file is only data):
```bash
source /workspace/sft_build/vllm_test/envv.sh      # = export VLLM_CACHE_ROOT=.../cache/vllm TRITON_CACHE_DIR=.../cache/triton TORCHINDUCTOR_CACHE_DIR=... XDG_CACHE_HOME=... HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 VLLM_NO_USAGE_STATS=1
```
GPU smoke test (worker mode; every section; ~15 min on the busy card; the model dir must hold config + tokenizer + weights):
```bash
cd /workspace/sft_build/vllm_test/runN && source ../envv.sh && export PYTHONPATH=/workspace/sft_build/pylibs:/workspace/sft_build/gemma_mt
/workspace/venv_gemma/bin/python /workspace/sft_build/gemma_mt/scripts/vllm_smoke.py \
    --model /workspace/sft_build/vllm_test/gemma-3-270m-it --mode worker --vllm-python /workspace/venv_vllm/bin/python \
    --prompts /workspace/sft_build/gemma/rl_prompts_v2/yor.parquet --out smoke.json --kv-cache-gib 2.0
#   --only sync,stop,dist,reward,loop,throughput,failure   sections;  --sleep  level-2 sleep;  --attention-backend TRITON_ATTN
#   --engine-model DIR   engine reads its config from another dir;  --bins / --hf-bins  throughput sets;  --no-hf-throughput
```
In-process variant (one interpreter):
```bash
uv pip install --python /workspace/venv_vllm/bin/python --target /workspace/sft_build/pylibs_vllm --no-deps pandas==3.0.6 pyarrow==26.0.0 python-dateutil tzdata six   # 215 MB
PYTHONPATH=/workspace/sft_build/gemma_mt:/workspace/sft_build/pylibs_vllm /workspace/venv_vllm/bin/python /workspace/sft_build/gemma_mt/scripts/vllm_smoke.py --mode inprocess --model ... --prompts ... --no-hf-throughput
```
Training (after the COMET scorer problem of section 2 is solved):
```bash
SFT=<checkpoint> PROMPTS=... CAL=... VLLM_PYTHON=/workspace/venv_vllm/bin/python bash scripts/vast_train_rl.sh            # sampler: vllm (rl.yaml)
bash scripts/vast_train_rl.sh sampler='"hf"'                                                                                  # old sampler
# on a shared GPU lower what the engine claims:  vllm_gpu_memory_utilization=0.10 vllm_kv_cache_gib=2 ;  in-process: vllm_mode='"inprocess"'
```
CPU checks: `cd gemma_mt && ../.venv-comet/bin/python tests/run_tests.py` (247 pass), `python -m gemma_mt.rl.train_rl --smoke-vllm` (whole loop through the vLLM code path with the stand-in engine).

## 9. Remaining risks

1. **COMET scorer does not run on transformers 5.x** (section 2): blocks any RL run regardless of the sampler.
2. **bf16 sampler vs autocast scorer**: mean |dlogp| ~0.015, p99 ~0.2, first-token TV up to 0.04-0.09 vs pure bf16; unbiased only up to that. Guard: the periodic consistency check; fix if it ever trips: importance weights or `vllm_dtype` float32 (not tried: needs a non-flash backend).
3. **Not tested with the real SFT checkpoint** (private repo): same architecture, config format checked with a transformers-5.19 `config.json`; a checkpoint with different `generation_config` is harmless (ignored). The first real run should start with `vllm_check_every=1` for a few steps.
4. **Long-tail memory**: 160 x b1024 sequences at 3,584 new tokens queue inside a 2 GiB cache; on the exclusive card the default (0.25 of 96 GB) gives >20 GiB of cache. On the shared card the engine will not start unless `vllm_gpu_memory_utilization` x 95 GiB is free.
5. **Foreign environment**: `venv_vllm` belongs to someone else; if it changes (vLLM or torch upgrade), `EngineConfig` options (`attention_backend`, `kv_cache_memory_bytes`, `generation_config`, `apply_model`, `reset_prefix_cache`) are vLLM-version-specific. `pip freeze` of the working set is in section 8.
6. **Level-2 sleep** works but each wake-up reallocates; with the unified-memory allocator an abort at close was seen once (fixed); keep `vllm_sleep: false` unless memory is tight.
7. **Numbers are lower bounds and are for the base model**: the SFT model's length distribution (and its `no_end_of_turn` rate) decides the HF/vLLM ratio in production; measure `t_gen`, `gen_tokens_per_s`, `t_sync` in the first RL log lines.
8. **Disk**: this work left 0.64 GB on the rental (`/workspace/sft_build/vllm_test`: base model copy 545 MB, compile caches 97 MB, small logs); the 8 GB `venv_vllm` is not mine.

## 10. State left on the rental (<box>) and on the laptop

* Synced into `/workspace/sft_build/gemma_mt` (mine only): `gemma_mt/rl/vllm_engine.py`, `vllm_sampler.py`, `train_rl.py`, `eval/eval_translation.py`, `scripts/vllm_smoke.py`, `scripts/vast_train_rl.sh`, `tests/test_vllm_sampler.py`, `configs/rl.yaml`. **`train_rl.py` is shared with the reward work**: the laptop copy now also prints `out.diacritic_deficit` (needs the newer `reward.py`, which I did not sync); on the rental I removed that print so the file matches the older `reward.py` there. When the reward files are synced, sync the laptop `train_rl.py` again (it contains my edits).
* `/workspace/sft_build/vllm_test` (0.64 GB): `gemma-3-270m-it` (unsloth mirror of the base model, 545 MB), `cache/` (vLLM/Triton compile caches, 97 MB), `envv.sh`, `cfg519/` (a config.json written by transformers 5.19 + symlink to the weights), `run1..run10/` (logs and `smoke.json` results, < 100 KB each). `pylibs_vllm` and `pylibs_comet` (in-process experiments) were removed again to spare the disk (recreate with the command of section 8). `/dev/shm` holds nothing of mine. No process of mine is running. The foreign job and `venv_vllm` were not touched (read-only use; no signals sent to foreign processes; one `pkill -f` early on matched only my own just-started smoke command).
* Laptop: `gemma_mt/DECISIONS.md` (D-28 appended), the files above, `reports/gemma_rl_vllm_2026-10-17.md`. Nothing pushed to the Hub/S3, no data copied to the laptop.
