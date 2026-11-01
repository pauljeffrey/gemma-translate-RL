# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage 2: GRPO on top of the SFT checkpoint with a shaped AfriCOMET-QE reward.

    python -m gemma_mt.rl.train_rl --config configs/rl.yaml
    python -m gemma_mt.rl.train_rl --smoke          # CPU, tiny random Gemma3, fake tokenizer, fake scorer

One optimizer step (everything on-policy, one update per sampled batch by default):
  1. draw `prompts_per_step` prompts (language-balanced by `language_alpha`, deterministic in the step number);
  2. sample `group_size` completions per prompt with the CURRENT policy (temperature `temperature`, top_p = 1);
  3. reward(sample) = shaped AfriCOMET-QE of the translation, penalty for failures (gemma_mt.rl.reward);
  4. advantages = group-normalised rewards x per-language weight (gemma_mt.rl.grpo);
  5. recompute log-probs of the sampled tokens with grad (teacher forcing, same temperature), add the k3 KL to the
     frozen reference, token-level loss normalised by the number of completion tokens of the WHOLE step,
     clip the gradient, AdamW step.
Checks inside the loop (they raise, they do not warn): every tensor shape, finite rewards/advantages/log-probs,
completion mask counts, KL == 0 at step 0, ratio == 1 for the single-update case.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gemma_mt import format as fmt
from gemma_mt.languages import SUPPORTED_RL
from gemma_mt.rl import grpo
from gemma_mt.languages import canonical_iso
from gemma_mt.rl.data import LENGTH_BINS, MIXED, PromptSchedule, RLPrompts, load_rl_prompts
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
from gemma_mt.rl.optim import build_rl_optimizer, optimizer_kind_of_state
from gemma_mt.rl.run_controls import KLGuard, PatienceStop, PlateauStop, RunControls, extension_decision
from gemma_mt.rl.lang_weights import build_controller as build_lang_weight_controller
from gemma_mt.rl.sampling import Rollout, assemble_batch, generate_completions, merge_greedy_into_groups
from gemma_mt.utils import (MetricsLogger, dataclass_from_dict, get_rng_state, load_yaml, parse_overrides,
                            set_rng_state, set_seed)


@dataclass
class RLConfig:
    model_name: str = "runs/sft/best"               # the SFT checkpoint (policy start AND frozen reference)
    tokenizer_name: Optional[str] = None
    prompts_path: str = "data/rl_prompts"
    eval_prompts_path: Optional[str] = None         # default: a hash-selected hold-out of the prompts
    out_dir: str = "runs/rl"
    attn_implementation: str = "sdpa"
    param_dtype: str = "float32"
    ref_dtype: str = "float32"
    autocast_dtype: str = "bfloat16"
    # rollout
    steps: int = 200
    prompts_per_step: int = 16
    group_size: int = 8
    # Recommended for EVERY stage of the clean rerun (b64 .. b1024, mixed, final), 2026-10-28: greedy_in_group true, greedy_stats sampled, greedy_adv_scale 0.3, greedy_adv_cap 3 (D-67 form: grad norm 0.60, greedy |A| 0.32 vs
    # sampled 0.57; the unscaled greedy member with lr 3e-6 and no std division diverged in phase 2b), greedy_share_warn 0.35, group_size >= 4 (one greedy + >= 3 sampled; the greedy member is the LAST row of every group).
    # Monitoring: step records greedy_finished_rate / greedy_loop_rate / greedy_reward_mean / sampled_reward_mean / greedy_adv_token_share; evaluation records eval/greedy_{finished,loop}_rate[_bin/<bin>] (evaluation is greedy).
    greedy_in_group: bool = False                   # 2026-10-19: the LAST member of each group is the greedy (temperature 0) completion, the other group_size-1 are sampled
    hard_fail_mask: str = "off"                     # D-61: off (default: a failed sample, reward -2, keeps a strong NEGATIVE advantage) | zero | exclude: advantage of the hard-failed samples (status != ok) set to 0 (exclude: also taken out of the group baseline / std). See grpo.group_advantages_masked
    greedy_stats: str = "all"                       # D-67: greedy_in_group only. all = group baseline / std over all members (as before); sampled = over the SAMPLED members only, the greedy member's advantage relative to them (grpo.group_advantages_sampled_baseline)
    greedy_adv_cap: float = 0.0                     # D-67: with greedy_stats: sampled, |advantage of the greedy member| <= this before greedy_adv_scale (0 = no cap)
    greedy_share_warn: float = 0.0                  # 2026-10-28: greedy_in_group only. > 0: print a warning (every 10 steps at most) when the greedy members' share of the step's token-weighted |advantage| (greedy_adv_token_share, logged every step) exceeds this. Uniform share is 1/group_size; suggested 0.35
    greedy_adv_scale: float = 1.0                   # D-59: multiplier of the GREEDY member's advantage when greedy_in_group (1 = as before). The greedy completion of a long looping document dominates the clipped gradient
    micro_batch_seqs: int = 16
    num_updates_per_batch: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    allow_truncated_sampling: bool = False          # must be True to use top_p < 1 / top_k > 0 (biases the estimator)
    # Rollout sampler (D-28): "hf" = HF `generate` on the training model (the original path); "vllm" = a vLLM engine that gets a bf16 copy of the
    # policy after EVERY optimizer step (on-policy sampling is kept). No silent fallback: a failing engine stops the run.
    sampler: str = "hf"
    vllm_mode: str = "worker"                       # worker: engine in a child process of `vllm_python` | inprocess: engine in this interpreter
    vllm_python: Optional[str] = None               # worker mode: interpreter of the environment that has vLLM (null = this interpreter)
    vllm_gpu_memory_utilization: float = 0.25       # fraction of the device memory the engine may use (weights + activations + KV cache)
    vllm_kv_cache_gib: Optional[float] = None       # fixed KV-cache size; takes precedence over vllm_gpu_memory_utilization for the cache
    vllm_max_model_len: int = 8192                  # >= longest prompt + the largest max_new_tokens
    vllm_max_num_seqs: int = 256
    vllm_max_num_batched_tokens: Optional[int] = None
    vllm_enforce_eager: bool = False
    vllm_sleep: bool = False                        # free the engine's GPU memory (level-2 sleep) while the policy is being trained
    vllm_attention_backend: Optional[str] = None    # null = vLLM's choice
    vllm_sync_dir: str = "/dev/shm"                 # worker mode: tmpfs for the weight file (~0.54 GB, deleted after each sync)
    vllm_startup_timeout: float = 1800.0
    vllm_request_timeout: float = 3600.0
    vllm_extra: Dict[str, Any] = field(default_factory=dict)   # extra keyword arguments for vllm.LLM(...)
    vllm_check_every: int = 25                      # every N steps compare vLLM's log-probs of its own samples with the trainer's recomputed ones (0 = never)
    vllm_logp_gap_tol: float = 0.25                 # ... and stop the run when the mean absolute gap is larger (a weight sync that did not work shows up here)
    max_new_tokens: int = 256
    max_src_tokens: int = 256                       # AfriCOMET-encoder tokens of the source
    # Input-length curriculum (owner 2026-10-12): bins b64 / b256 / b512 / b1024 (b2048) are trained in stages, in the order of
    # `curriculum_steps`, then in a random mixture of all bins ("mixed"). Without `curriculum_steps` the bins are cycled.
    # Per-bin overrides keep the long bins within memory / time.
    length_bins: bool = False
    curriculum_steps: Dict[str, int] = field(default_factory=dict)             # e.g. {b64: 40, b256: 60, b512: 60, b1024: 60}
    prompts_per_step_by_bin: Dict[str, int] = field(default_factory=dict)      # also key "mixed"
    max_new_tokens_by_bin: Dict[str, int] = field(default_factory=dict)        # also key "mixed"
    micro_batch_tokens: int = 0                     # > 0: sequences per forward = max(1, micro_batch_tokens // longest sequence)
    language_alpha: float = 1.0
    language_weights: Dict[str, float] = field(default_factory=dict)     # {fon: 2.0, ...}: extra sampling weight per language
    # Domain mixture (D-52, science run; default off): `domain` column of the prompt files ("science" / "general"). Every prompt of a step first draws its domain by these shares,
    # then (mixed stage) its bin by `domain_bin_weights[domain]` ({bin: weight}, default uniform over the bins the domain has), then the language. See PromptSchedule.
    domain_weights: Dict[str, float] = field(default_factory=dict)       # e.g. {science: 0.7, general: 0.3}
    domain_bin_weights: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # D-59 language FLOOR (needs domain_weights): every language gets >= language_floor_prompts prompts in every aligned block of language_floor_steps steps (0 = off). See PromptSchedule.
    language_floor_steps: int = 0
    language_floor_prompts: int = 1
    # optimisation
    learning_rate: float = 1.0e-6
    warmup_steps: int = 10
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    advantage_norm: str = "mean_std"                # mean_std (repo) | mean
    std_eps: float = 1.0e-4
    kl_coef: float = 0.05
    # KL guard (owner 2026-10-20, after phase 2b's KL ran away to 2.3): when the mean KL of the last `kl_guard_window` steps exceeds the threshold the learning rate is
    # multiplied by `kl_guard_factor` (not below `kl_guard_min_scale`), at most once per `kl_guard_cooldown` steps. 0 = off. Nothing is stopped or cancelled.
    kl_guard_threshold: float = 0.0
    kl_guard_factor: float = 0.5
    kl_guard_min_scale: float = 0.125
    kl_guard_window: int = 5
    kl_guard_cooldown: int = 10
    # 2026-10-28 (clean rerun): (a) the guard state (scale, last cut, KL history) is stored in `last/trainer_state.pt` and restored on resume (`kl_guard_restore`; set it false at a deliberate fresh-reference restart);
    # every guard event goes to <out_dir>/kl_guard_events.jsonl and to the step record. (b) OPTIONAL recovery (`kl_guard_recover`, default off): after a cut, when the median KL of the last `kl_guard_recover_window` steps
    # (all of them after the last guard event, >= cooldown steps after it) is below `kl_guard_recover_below` x threshold, the scale is multiplied by `kl_guard_recover_factor`, never above 1.0 (the planned lr is never exceeded).
    kl_guard_restore: bool = True
    kl_guard_recover: bool = False
    kl_guard_recover_factor: float = 1.25
    kl_guard_recover_below: float = 0.5
    kl_guard_recover_window: int = 20
    # Plateau / saturation stop (default off). At evaluation boundaries (needs eval_every > 0, run_controls true): when the stage has >= min_steps steps (`plateau_stop_min_steps_by_bin[bin]` overrides it), the
    # 95 % bootstrap CI over `plateau_stop_block`-step block means of the OLS slope of the training reward over the last `plateau_stop_window` steps contains 0, the training finished rate of the window and the greedy evaluation
    # finished rate of the stage's bin are >= `plateau_stop_min_finished`, the window holds no guard event and no learning-rate change, and the lr multiplier is >= `plateau_stop_min_lr_scale`, the trainer asks for a
    # WINDDOWN of `plateau_stop_winddown` steps; the run then ends normally (last/ saved, latest + best pushed, summary.json: stop_reason "plateau_stop" + the evidence). Only evaluated in the LAST stage of the run; `steps` stays the hard cap.
    plateau_stop: bool = False
    plateau_stop_min_steps: int = 150
    plateau_stop_min_steps_by_bin: Dict[str, int] = field(default_factory=dict)
    plateau_stop_window: int = 100
    plateau_stop_block: int = 20
    plateau_stop_min_finished: float = 0.99
    plateau_stop_winddown: int = 10
    plateau_stop_min_lr_scale: float = 0.5
    plateau_stop_bootstrap: int = 2000
    # Patience-based early stop (2026-10-29, Amendment 4; default off), evaluated at EVERY step in the LAST stage of the run: the smoothed (mean of `early_stop_block` steps) training reward, training finished rate and
    # greedy loop rate (greedy_in_group) are tracked against their best values; when none has set a new best by more than its epsilon for `early_stop_patience` steps, after `early_stop_min_steps` steps in the stage,
    # the run winds down over `early_stop_winddown` steps (the WINDDOWN mechanism). Never inside a window with a KL-guard event, a changed or low (< early_stop_min_lr_scale) lr multiplier or a finished rate < early_stop_min_finished.
    # It works next to plateau_stop (the slope-CI rule, which needs a 100-step window); whichever fires first ends the run. See run_controls.PatienceStop.
    early_stop: bool = False
    early_stop_patience: int = 60
    early_stop_min_steps: int = 80
    early_stop_block: int = 10
    early_stop_eps_reward: float = 0.02
    early_stop_eps_finished: float = 0.002
    early_stop_eps_loop: float = 0.005
    early_stop_min_finished: float = 0.99
    early_stop_min_lr_scale: float = 0.5
    early_stop_winddown: int = 10
    # Surgical extension (default off; needs early_stop): at the decision point, if the improvement criterion above fired within the last `extend_recent` steps, the cap `steps` is raised by `extend_steps`; repeated at most
    # until the total extension is `extend_max_frac` x `steps` (+50 %). Every decision (extend or not, with the reason) goes to <out_dir>/extensions.jsonl and summary.json. The decision point is the cap itself, or, with
    # `planned_winddown_last` N > 0, N steps before it (the trainer then also starts the planned final wind-down of N steps there, instead of an external watcher, so that an extension moves it).
    extend_if_improving: bool = False
    extend_steps: int = 0
    extend_max_frac: float = 0.5
    extend_recent: int = 25
    planned_winddown_last: int = 0
    # Optimizer (2026-10-29, Amendment 4): adamw (default, unchanged) | muon (Muon on the 2-D decoder matrices, AdamW on embeddings / norms / 1-D, update RMS matched to AdamW: same convention as train_sft.py, one lr).
    optimizer: str = "adamw"
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    # Evaluation generation must use vLLM (Amendment 4): True = every evaluation path (hold-out reward, dev benchmark, aux sets) raises instead of falling back to HF `generate` (needs sampler: vllm).
    eval_vllm_only: bool = False
    hub_allow_public: bool = False                  # hub_repo_id with hub_private: false is refused unless this is true (owner 2026-10-29: models are private)
    plateau_stop_ci_high_max: float = 0.0           # 0 = off. > 0: also require the upper end of the slope CI (reward units per 100 steps) to be <= this, i.e. the window can exclude a gain larger than this (power guard)
    kl_estimator: str = "k3"                        # k3 (GRPO/repo) | k2 (exact reverse-KL gradient)
    clip_eps: Optional[float] = None                # only matters when num_updates_per_batch > 1
    clip_eps_high: Optional[float] = None
    loss_agg: str = "token"                         # token | sequence
    loss_chunk_tokens: int = 512
    gradient_checkpointing: bool = False
    # reward
    reward: Dict[str, Any] = field(default_factory=dict)       # RewardConfig fields
    scorer: Dict[str, Any] = field(default_factory=lambda: {"type": "comet", "model_id":
                                                              "masakhane/africomet-qe-stl-1.1", "batch_size": 32})
    # bookkeeping
    eval_every: int = 25
    eval_prompts: int = 128
    eval_min_per_bin: int = 0                       # D-45: hold out at least this many prompts of EVERY length bin (topped up with that bin's smallest-hash prompts, on top of the `eval_prompts` hash selection, which is never changed); 0 = off
    save_every: int = 50
    log_every: int = 1
    check_sampler_logprobs: bool = False            # extra sanity check; stores per-step scores: tiny vocab only
    seed: int = 1234
    # Directories appended to sys.path before anything else is imported (pure-python helpers the policy environment lacks: sacrebleu for the dev chrF++ evaluation lives in
    # /workspace/sft_build/pylibs on the rental, D-38 audit; they are APPENDED, so the environment's own packages always win)
    extra_libs: List[str] = field(default_factory=list)
    device: Optional[str] = None
    resume: bool = True
    wandb_project: Optional[str] = None
    run_name: Optional[str] = None
    # Reference-based model selection (owner 2026-10-14): greedy translations of a held-out DEV benchmark sample (FLORES+ dev / MAFAND dev, human
    # references) are scored with chrF++/BLEU; the checkpoint that goes to the Hub is the best by `select_by`, and only if it beats the
    # starting SFT model (evaluated at step 0). The QE reward on held-out prompts stays as a diagnostic and a guard.
    benchmark_eval_path: Optional[str] = None       # jsonl/parquet: english, reference, iso, benchmark, id  (DEV splits only, never the test sets)
    benchmark_eval_per_language: int = 40
    benchmark_eval_every: int = 0                   # 0 = every `eval_every` steps
    benchmark_max_new_tokens: int = 256
    benchmark_batch_size: int = 32
    select_by: str = "reward"                       # reward (QE reward of the held-out prompts) | benchmark_chrf (macro chrF++ on the dev sample)
    select_min_gain: float = 0.0                    # a checkpoint must beat the best so far by this much (chrF++ points / reward units)
    select_min_finished_rate: float = 0.95          # candidates whose greedy outputs stop (<end_of_turn>) less often than this are never selected
    # D-52: a run that RESUMES with a different benchmark / selection metric than the checkpoint was selected on (the science run resumes the general model's state): True = ignore the
    # checkpoint's best_select / base_chrf, score the resumed policy on `benchmark_eval_path` first (the "baseline" the selection has to beat) and log it. False = the original behaviour.
    resume_reset_selection: bool = False
    # D-52: monitor-only dev sets next to the selection benchmark (logged as bench_aux/<name>/chrf_macro|bleu_macro|finished_rate|chrf/<iso>, never used for selection or pushes).
    # Each entry: {name, path, per_language (default 40), max_new_tokens (default benchmark_max_new_tokens), batch_size (default benchmark_batch_size)}; same cadence as the benchmark.
    benchmark_aux: List[Dict[str, Any]] = field(default_factory=list)
    # D-56: run controls read from files in out_dir at every step (rl/run_controls.py): WINDDOWN ("50": the learning rate decays linearly to 0 over the next N steps, then the run ends normally:
    # last/ saved, latest + best pushed, summary.json) and LR_SCALE (a float multiplier on the learning rate). Nothing happens without those files. aux entries may carry `every: N` (steps).
    export_steps: List[int] = field(default_factory=list)   # AB1: also write <out_dir>/snap_step<N> (HF directory, export_dtype) at these steps and at the last step
    export_dtype: str = "bfloat16"                  # AB1: bfloat16 (0.54 GB, what the evaluations load) | float32
    run_controls: bool = True
    # D-65 (rl/lang_weights.py): language DRAW WEIGHTS as a run control. `lang_weights_file`: <out_dir>/LANG_WEIGHTS.json is read at every step (needs run_controls) and overrides / merges `language_weights`.
    # `adaptive_lang_weights` (default OFF): every `adaptive_lang_weights_every` steps from `adaptive_lang_weights_first_step` the weights of `adaptive_lang_weights_languages` (empty = all)
    # are recomputed from the per-language training statistics of the window and the dev trends (AdaptiveParams overrides in `adaptive_lang_weights_params`); `..._dry_run`: only log what it would do.
    # `lang_stats_log`: log the per-language step statistics lang/{n,reward,fail}/<iso> (always on with the adaptive updater). All off = the run is bit-for-bit the one without this feature.
    lang_weights_file: bool = False
    lang_stats_log: bool = False
    lang_weights_log_every: int = 50
    adaptive_lang_weights: bool = False
    adaptive_lang_weights_dry_run: bool = False
    adaptive_lang_weights_every: int = 50
    adaptive_lang_weights_first_step: int = 100
    adaptive_lang_weights_languages: List[str] = field(default_factory=list)
    adaptive_lang_weights_params: Dict[str, Any] = field(default_factory=dict)
    # Hub: push the BEST checkpoint (by held-out eval reward) as soon as it is found; nothing else is uploaded
    hub_repo_id: Optional[str] = None               # None = never push
    hub_branch: Optional[str] = None                # e.g. "rl" to keep the SFT model on main; None = main of hub_repo_id
    hub_latest_branch: Optional[str] = None         # owner 2026-10-18: besides the BEST checkpoint (hub_branch / main, tag rl-step-N) also keep the LATEST one on this branch
    hub_latest_every: int = 0                       # push the latest checkpoint every this many steps (and at the last step); 0 = never
    hub_stage_branches: bool = False                # owner 2026-10-18: push the checkpoint at the END of every curriculum stage to branch `stage-<bin>` (b64, b256, ...) and the one after the
                                                    # final random-mixture stage to `stage-mixed` (the mixed-stage model counters forgetting of the single-bin stages)
    hub_private: bool = True
    hub_squash: bool = True                         # owner 2026-10-19 (small Hub quota): after each push squash the history of the pushed branch (best = main, latest, stage-*) ...
    hub_prune_tags: bool = True                     # ... and delete the older rl-step-N tags when a newer best is pushed (the repo holds ~7 model copies: best, latest, 5 stage branches)
    keep_single_checkpoint: bool = False            # local disk: only last/ (resumable); no best/ or final/ copies
    # Best-translation archive (D-41, owner 2026-10-19): per source (key = sha1(iso + NFC source)[:16]) the best eligible translation seen so far, written by a
    # background thread to one SQLite file (rl/archive.py). Eligible = status ok, no failed chunk, LID passed, diacritic deficit 0, reward >= the threshold of the
    # language's content mode. Read-only with respect to training. Review: python -m gemma_mt.rl.archive stats|export|review PATH
    archive: bool = False
    archive_path: Optional[str] = None              # null = <out_dir>/archive/archive.sqlite
    archive_min_reward_by_mode: Dict[str, float] = field(default_factory=lambda: {"qe": 1.0, "blend": 1.0, "consensus": 0.5})   # +1.0 = the language's median-human QE anchor t_mid; keys are MERGED over these defaults
    archive_min_consensus_chrf: float = 50.0        # blend languages: the group consensus (mean chrF++ vs the valid siblings) must be at least this (consensus languages: reward 0.5 already means chrF++ >= 58.8)
    archive_flush_seconds: float = 20.0             # the writer commits at least this often (a hard kill loses at most this much)
    archive_queue_size: int = 64                    # steps waiting for the writer; a full queue drops the step's rows (counted as archive/dropped_queue), the training step never waits


def _dtype(name: str):
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def validate_config(cfg: RLConfig) -> None:
    if cfg.select_by not in ("reward", "benchmark_chrf"):
        raise ValueError("select_by must be 'reward' or 'benchmark_chrf'")
    if cfg.select_by == "benchmark_chrf" and not cfg.benchmark_eval_path:
        raise ValueError("select_by=benchmark_chrf needs benchmark_eval_path (a dev-split file with human references)")
    if cfg.eval_min_per_bin < 0:
        raise ValueError("eval_min_per_bin must be >= 0")
    if cfg.group_size < 2:
        raise ValueError("group_size must be >= 2 (the baseline is the group mean)")
    if cfg.greedy_in_group and cfg.group_size < 3:
        raise ValueError("greedy_in_group needs group_size >= 3 (at least two sampled members next to the greedy one)")
    if cfg.hard_fail_mask is False:                 # YAML reads an unquoted `off` as the boolean False
        cfg.hard_fail_mask = "off"
    if cfg.hard_fail_mask not in ("off", "zero", "exclude"):
        raise ValueError("hard_fail_mask must be off | zero | exclude")
    if cfg.greedy_stats not in ("all", "sampled") or (cfg.greedy_stats == "sampled" and not cfg.greedy_in_group) or cfg.greedy_adv_cap < 0:
        raise ValueError("greedy_stats must be 'all' or 'sampled' (sampled needs greedy_in_group); greedy_adv_cap >= 0")
    if cfg.greedy_stats == "sampled" and cfg.hard_fail_mask != "off":
        raise ValueError("greedy_stats: sampled is not combined with hard_fail_mask")
    if not (0.0 <= cfg.greedy_adv_scale <= 1.0) or (cfg.greedy_adv_scale != 1.0 and not cfg.greedy_in_group):
        raise ValueError("greedy_adv_scale must be in [0, 1] and can differ from 1 only with greedy_in_group")
    if not (0.0 <= cfg.greedy_share_warn < 1.0) or (cfg.greedy_share_warn > 0 and not cfg.greedy_in_group):
        raise ValueError("greedy_share_warn must be in [0, 1) and > 0 only with greedy_in_group")
    if cfg.kl_guard_threshold < 0:
        raise ValueError("kl_guard_threshold must be >= 0 (0 = off)")
    if cfg.kl_guard_threshold > 0 and (not 0.0 < cfg.kl_guard_factor < 1.0 or not 0.0 < cfg.kl_guard_min_scale <= 1.0 or cfg.kl_guard_window < 1 or cfg.kl_guard_cooldown < 0):
        raise ValueError("kl_guard: need 0 < kl_guard_factor < 1, 0 < kl_guard_min_scale <= 1, kl_guard_window >= 1, kl_guard_cooldown >= 0")
    if cfg.kl_guard_recover:
        if cfg.kl_guard_threshold <= 0:
            raise ValueError("kl_guard_recover needs kl_guard_threshold > 0")
        if not cfg.kl_guard_recover_factor > 1.0 or not 0.0 < cfg.kl_guard_recover_below <= 1.0 or cfg.kl_guard_recover_window < 1:
            raise ValueError("kl_guard_recover: need kl_guard_recover_factor > 1, 0 < kl_guard_recover_below <= 1 (fraction of the threshold), kl_guard_recover_window >= 1")
    if cfg.plateau_stop:
        if not cfg.run_controls:
            raise ValueError("plateau_stop needs run_controls: true (it ends the run through the WINDDOWN mechanism)")
        if cfg.eval_every < 1 or cfg.eval_prompts < 1:
            raise ValueError("plateau_stop is evaluated at evaluation boundaries: it needs eval_every >= 1")
        if cfg.plateau_stop_block < 1 or cfg.plateau_stop_window < 4 * cfg.plateau_stop_block:
            raise ValueError("plateau_stop_window must hold at least 4 blocks of plateau_stop_block steps")
        if cfg.plateau_stop_min_steps < cfg.plateau_stop_window or any(int(v) < cfg.plateau_stop_window for v in cfg.plateau_stop_min_steps_by_bin.values()):
            raise ValueError("plateau_stop_min_steps (and every plateau_stop_min_steps_by_bin value) must be >= plateau_stop_window")
        bad = set(cfg.plateau_stop_min_steps_by_bin) - set(LENGTH_BINS) - {MIXED}
        if bad:
            raise ValueError(f"plateau_stop_min_steps_by_bin: unknown bin(s) {sorted(bad)}")
        if not 0.0 < cfg.plateau_stop_min_finished <= 1.0 or cfg.plateau_stop_winddown < 1 or not 0.0 <= cfg.plateau_stop_min_lr_scale <= 1.0 or cfg.plateau_stop_bootstrap < 200 or cfg.plateau_stop_ci_high_max < 0:
            raise ValueError("plateau_stop: need 0 < plateau_stop_min_finished <= 1, plateau_stop_winddown >= 1, 0 <= plateau_stop_min_lr_scale <= 1, plateau_stop_bootstrap >= 200, plateau_stop_ci_high_max >= 0")
    if cfg.optimizer not in ("adamw", "muon"):
        raise ValueError("optimizer must be 'adamw' or 'muon'")
    if cfg.muon_momentum < 0 or cfg.muon_momentum >= 1 or cfg.muon_ns_steps < 1:
        raise ValueError("muon_momentum must be in [0, 1) and muon_ns_steps >= 1")
    if cfg.eval_vllm_only and cfg.sampler != "vllm":
        raise ValueError("eval_vllm_only needs sampler: vllm")
    if cfg.hub_repo_id and not cfg.hub_private and not cfg.hub_allow_public:
        raise ValueError("hub_repo_id with hub_private: false is refused (the models are private); set hub_allow_public: true to override")
    if cfg.early_stop:
        if not cfg.run_controls:
            raise ValueError("early_stop needs run_controls: true (it ends the run through the WINDDOWN mechanism)")
        if cfg.early_stop_block < 2 or cfg.early_stop_patience < cfg.early_stop_block or cfg.early_stop_min_steps < cfg.early_stop_block:
            raise ValueError("early_stop: need early_stop_block >= 2, early_stop_patience >= early_stop_block, early_stop_min_steps >= early_stop_block")
        if min(cfg.early_stop_eps_reward, cfg.early_stop_eps_finished, cfg.early_stop_eps_loop) < 0 or not 0.0 < cfg.early_stop_min_finished <= 1.0 or not 0.0 <= cfg.early_stop_min_lr_scale <= 1.0 or cfg.early_stop_winddown < 1:
            raise ValueError("early_stop: epsilons >= 0, 0 < early_stop_min_finished <= 1, 0 <= early_stop_min_lr_scale <= 1, early_stop_winddown >= 1")
    if cfg.extend_if_improving:
        if not cfg.early_stop:
            raise ValueError("extend_if_improving needs early_stop: true (its improvement criterion is the extension criterion)")
        if cfg.extend_steps < 1 or not 0.0 < cfg.extend_max_frac <= 0.5 + 1e-9 or cfg.extend_recent < 1:
            raise ValueError("extend_if_improving: need extend_steps >= 1, 0 < extend_max_frac <= 0.5 (hard maximum +50 %), extend_recent >= 1")
    if cfg.planned_winddown_last < 0 or (cfg.planned_winddown_last > 0 and (not cfg.run_controls or cfg.planned_winddown_last >= cfg.steps)):
        raise ValueError("planned_winddown_last must be >= 0, < steps, and needs run_controls: true")
    if cfg.planned_winddown_last > 0 and cfg.extend_if_improving and cfg.extend_steps <= cfg.planned_winddown_last:
        raise ValueError("extend_steps must exceed planned_winddown_last (the decision point moves N steps before the new cap)")
    if (cfg.top_p < 1.0 or cfg.top_k > 0) and not cfg.allow_truncated_sampling:
        raise ValueError("top_p < 1 / top_k > 0 make the sampled distribution differ from the scored one and bias the "
                         "policy gradient; set allow_truncated_sampling=true to accept that")
    if cfg.temperature <= 0:
        raise ValueError("temperature must be > 0")
    for k, v in cfg.language_weights.items():
        if canonical_iso(k) is None or float(v) <= 0:
            raise ValueError(f"language_weights: unknown language {k!r} or non-positive weight {v!r}")
    if cfg.adaptive_lang_weights or cfg.lang_weights_file:
        from gemma_mt.rl.lang_weights import AdaptiveParams

        AdaptiveParams.from_dict(cfg.adaptive_lang_weights_params)                        # unknown key -> ValueError at start-up, not at step 100
        if cfg.adaptive_lang_weights_every < 1 or cfg.adaptive_lang_weights_first_step < 0:
            raise ValueError("adaptive_lang_weights_every must be >= 1 and adaptive_lang_weights_first_step >= 0")
        bad_l = [x for x in cfg.adaptive_lang_weights_languages if canonical_iso(x) is None]
        if bad_l:
            raise ValueError(f"adaptive_lang_weights_languages: unknown language(s) {bad_l}")
    if cfg.lang_weights_file and not cfg.run_controls:
        raise ValueError("lang_weights_file needs run_controls: true")
    for name in ("prompts_per_step_by_bin", "max_new_tokens_by_bin", "curriculum_steps"):
        bad = set(getattr(cfg, name)) - set(LENGTH_BINS) - {MIXED}
        if bad:
            raise ValueError(f"{name}: unknown length bin(s) {sorted(bad)}; known: {LENGTH_BINS} and {MIXED!r}")
    if MIXED in cfg.curriculum_steps:
        raise ValueError("curriculum_steps lists the staged bins; the mixture stage always follows them")
    if cfg.domain_weights:
        if not cfg.length_bins:
            raise ValueError("domain_weights needs length_bins: true (the domain mixture draws domain -> bin -> language)")
        if any(float(v) < 0 for v in cfg.domain_weights.values()) or sum(float(v) for v in cfg.domain_weights.values()) <= 0:
            raise ValueError("domain_weights must be non-negative with a positive sum")
    for d, bw in cfg.domain_bin_weights.items():
        bad = set(bw) - set(LENGTH_BINS)
        if bad or any(float(v) < 0 for v in bw.values()):
            raise ValueError(f"domain_bin_weights[{d!r}]: unknown bin(s) {sorted(bad)} or a negative weight")
    for a in cfg.benchmark_aux:
        if not isinstance(a, dict) or not a.get("name") or not a.get("path"):
            raise ValueError("benchmark_aux entries need `name` and `path`")
    if cfg.num_updates_per_batch > 1 and cfg.clip_eps is None:
        raise ValueError("num_updates_per_batch > 1 needs clip_eps (otherwise the update is not trust-region bounded)")
    if cfg.kl_estimator not in grpo.KL_ESTIMATORS:
        raise ValueError(f"kl_estimator must be one of {sorted(grpo.KL_ESTIMATORS)}")
    if cfg.sampler not in ("hf", "vllm"):
        raise ValueError("sampler must be 'hf' or 'vllm'")
    if cfg.sampler == "vllm":
        if cfg.vllm_mode not in ("worker", "inprocess"):
            raise ValueError("vllm_mode must be 'worker' or 'inprocess'")
        if cfg.vllm_max_model_len < max([cfg.max_new_tokens, cfg.benchmark_max_new_tokens] + list(cfg.max_new_tokens_by_bin.values())) + 512:
            raise ValueError("vllm_max_model_len must exceed the largest max_new_tokens by the prompt length (prompt + completion must fit: set it >= the largest max_new_tokens + ~2048)")
        if cfg.vllm_kv_cache_gib is None and not 0.0 < cfg.vllm_gpu_memory_utilization < 1.0:
            raise ValueError("vllm_gpu_memory_utilization must be in (0, 1)")
    if cfg.archive:
        from gemma_mt.rl.archive import MODES as ARCHIVE_MODES

        bad = set(cfg.archive_min_reward_by_mode) - set(ARCHIVE_MODES)
        if bad:
            raise ValueError(f"archive_min_reward_by_mode: unknown mode(s) {sorted(bad)}; known: {ARCHIVE_MODES}")
        if any(not isinstance(v, (int, float)) or not math.isfinite(float(v)) for v in cfg.archive_min_reward_by_mode.values()):
            raise ValueError("archive_min_reward_by_mode: thresholds must be finite numbers")
        if cfg.archive_flush_seconds <= 0 or cfg.archive_queue_size < 1:
            raise ValueError("archive_flush_seconds must be > 0 and archive_queue_size >= 1")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError("the RL stage is single-GPU (DECISIONS.md D-14); run it without torchrun")


def build_scorer(spec: Dict[str, Any], device: Optional[str] = None):
    t = spec.get("type", "comet")
    if spec.get("ensemble_model_id"):          # D-55: QE ensemble (reward.qe_ensemble): two checkpoints, one worker process (rl/qe_ensemble.py)
        from gemma_mt.rl.qe_ensemble import build_pair_scorer

        return build_pair_scorer(spec, device)
    if t == "fake":
        from gemma_mt.rl.comet import FakeScorer

        return FakeScorer()
    if t == "worker":
        from gemma_mt.rl.comet import WorkerScorer

        return WorkerScorer(python=spec.get("python", sys.executable), model_id=spec.get("model_id"),
                            batch_size=spec.get("batch_size", 32), fake=spec.get("fake", False),
                            pythonpath=spec.get("pythonpath"))
    from gemma_mt.rl.comet import CometScorer

    return CometScorer(spec.get("model_id", "masakhane/africomet-qe-stl-1.1"), batch_size=spec.get("batch_size", 32),
                       device=device)


def split_eval_prompts(prompts: RLPrompts, n_eval: int, seed: int, min_per_bin: int = 0):
    """Deterministic hold-out: the n_eval prompts with the smallest hash of their English text (so the same sentence
    is never in both sets). Returns (train_indices, eval_indices).

    `min_per_bin` (D-45, default 0 = the original behaviour): a length bin that has fewer than `min_per_bin` prompts in that selection (the new b2048 bin holds ~0.5 % of the
    rows, so the plain selection would give it ~0-1 hold-out prompts) is topped up with ITS smallest-hash prompts (and every other copy of the same English text). The `n_eval` selection itself is unchanged, so the hold-out
    of an earlier run is always a subset of the new one."""
    from gemma_mt.data.prepare_sft import split_fraction

    order = sorted(range(len(prompts)), key=lambda i: (split_fraction(prompts.english[i], seed), i))
    ev = order[:n_eval]
    evset = set(ev)
    if min_per_bin > 0 and prompts.length_bin:
        have: Dict[str, int] = {}
        for i in ev:
            have[prompts.length_bin[i]] = have.get(prompts.length_bin[i], 0) + 1
        for b in sorted(set(prompts.length_bin)):
            need = min_per_bin - have.get(b, 0)
            if need > 0:
                cand = [i for i in order if prompts.length_bin[i] == b and i not in evset]
                k = need
                while k < len(cand) and prompts.english[cand[k]] == prompts.english[cand[k - 1]]:
                    k += 1                      # a window used for several languages has one hash: all its copies are held out together (English never on both sides)
                add = cand[:k]
                ev = ev + add
                evset.update(add)
    return [i for i in range(len(prompts)) if i not in evset], sorted(ev)


def _subset(p: RLPrompts, idx: List[int]) -> RLPrompts:
    return RLPrompts([p.english[i] for i in idx], [p.iso[i] for i in idx], [p.source_dataset[i] for i in idx],
                     [p.src_afr_tokens[i] for i in idx], dict(p.dropped), p.name_mismatches,
                     [p.length_bin[i] for i in idx] if p.length_bin else [], [p.domain[i] for i in idx] if p.domain else [])


def rollout_and_reward(policy, tokenizer, reward: TranslationReward, prompts: RLPrompts, indices: List[int],
                       cfg: RLConfig, stop_ids, pad_id, device, autocast, greedy: bool = False,
                       return_logprobs: bool = False, max_new_tokens: Optional[int] = None, sampler=None, key: int = 0):
    """`sampler`: a gemma_mt.rl.vllm_sampler sampler (already synced to `policy`); None = HF `generate` on `policy` (the original path)."""
    prompt_ids = [fmt.build_prompt(prompts.english[i], prompts.iso[i], tokenizer) for i in indices]
    G = 1 if greedy else cfg.group_size
    kw = dict(group_size=cfg.group_size, max_new_tokens=cfg.max_new_tokens if max_new_tokens is None else max_new_tokens,
              temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k, stop_ids=stop_ids, pad_id=pad_id, greedy=greedy,
              return_logprobs=return_logprobs)
    def _gen(**over):
        if sampler is not None:
            return sampler.generate(prompt_ids, key=key, **dict(kw, **over))
        return generate_completions(policy, prompt_ids, device=device, autocast=autocast, **dict(kw, **over))

    if cfg.greedy_in_group and not greedy:
        roll = merge_greedy_into_groups(_gen(group_size=cfg.group_size - 1), _gen(greedy=True, group_size=1), cfg.group_size - 1)
    else:
        roll = _gen()
    samples = [Sample(roll.completions[r], prompts.iso[indices[r // G]], prompts.english[indices[r // G]],
                      roll.post_stop_tokens[r] if roll.post_stop_tokens else 0, prompts.domain_of(indices[r // G]))
               for r in range(len(roll.completions))]
    if reward.cfg.confidence and roll.sampler_logp is not None and len(roll.sampler_logp) == len(samples):          # D-47: mean token log-prob of each sampled completion (content tokens only)
        from gemma_mt.rl.confidence import completion_mean_logp

        for r, smp in enumerate(samples):
            smp.logp = completion_mean_logp(roll.sampler_logp[r], roll.completions[r], _tag_len(tokenizer, roll.completions[r], smp.iso), stop_ids)
    out = reward(samples, group_size=G)       # the consensus signal needs to know which consecutive rows form a GRPO group (greedy evaluation: G = 1)
    return roll, samples, out


def _tag_len(tokenizer, token_ids: List[int], iso: str) -> int:
    """Number of leading tokens of a completion that make up the language tag: 1 with the dedicated single tag token (`<unusedK>` or the renamed `<yor>`), 1-3 text pieces before
    (growth 1 of 2026-11-01 switched the data to single tokens). Falls back to 1."""
    from gemma_mt import tags
    from gemma_mt.rl.confidence import tag_prefix_len

    iso = canonical_iso(iso) or iso
    cands = {iso, f"<{iso}>"}
    try:
        cands.add(tags.tag_token(iso))
    except ValueError:
        pass
    return tag_prefix_len(lambda ids: tokenizer.decode(ids, skip_special_tokens=False), token_ids, cands)


def _text_loops(reward: TranslationReward, source: str, text: str, iso: str) -> bool:
    """The reward's own repetition / degeneration verdict on a decoded completion (reward.py, status "degenerate"): the length-calibrated severity >= 1 for a long source
    (D-60), otherwise `guards.is_repetitive` and, where the reward uses it, `guards.is_self_repetitive`."""
    from gemma_mt.rl import guards, longdoc
    from gemma_mt.rl.text import nfc

    cfg = reward.cfg
    src = nfc(source) if cfg.nfc else source
    if reward.longdoc_cal is not None and cfg.longdoc_repetition and len(src.split()) >= cfg.longdoc_min_src_words:
        sev, _ = reward.longdoc_cal.severity(longdoc.repetition_profile(text), len(src.split()), reward._src_profile(src))
        return bool(sev >= 1.0)
    if cfg.repetition_guard and guards.is_repetitive(text):
        return True
    mode = reward.content_spec(canonical_iso(iso))[0]
    return bool(cfg.selfrep_guard != "off" and (cfg.selfrep_guard == "all" or mode != "qe") and guards.is_self_repetitive(text))


_LOOP_CHECK_WARNED = [False]


def loop_flags(reward: TranslationReward, samples: List[Sample], status: List[str], decoded: List[str], finished: List[bool]) -> List[bool]:
    """Per completion: does it LOOP? (2026-10-28, owner: score the greedy output that loops at inference.) True when the reward classified it "degenerate", or when it was cut at max_new_tokens
    (never finished) and the reward's repetition check fires on its text (a loop that runs into the length cap gets the status no_end_of_turn first, so the status alone would miss exactly
    the failure that matters). A fault in this monitoring helper never stops a run: it falls back to the status and says so once."""
    flags = [st == "degenerate" for st in status]
    try:
        for i, sm in enumerate(samples):
            if not flags[i] and not finished[i] and status[i] == "no_end_of_turn":
                flags[i] = _text_loops(reward, sm.source, decoded[i], sm.iso)
    except Exception as e:                                                                  # monitoring only
        if not _LOOP_CHECK_WARNED[0]:
            _LOOP_CHECK_WARNED[0] = True
            print(f"[gemma_mt] WARNING: loop check on unfinished completions failed ({type(e).__name__}: {e}); using the reward status 'degenerate' only", flush=True)
    return flags


def evaluate_policy(policy, tokenizer, reward, eval_prompts: RLPrompts, cfg: RLConfig, stop_ids, pad_id, device,
                    autocast, sampler=None) -> Dict[str, float]:
    """Greedy (temperature 0) completions of the held-out prompts. Besides the rewards: `eval/greedy_finished_rate[_bin/<b>]` and `eval/greedy_loop_rate[_bin/<b>]` (loop = `loop_flags`), the
    quantities that decide whether greedy decoding loops at inference (a bin mean is a mean over that bin's hold-out prompts, `eval/greedy_n_bin/<b>`)."""
    if cfg.eval_vllm_only and sampler is None:
        raise RuntimeError("eval_vllm_only: the evaluation would fall back to HF generate (no vLLM sampler given)")
    rewards, qes, fin, fails = [], [], [], 0
    loops: List[bool] = []
    per_bin_fin: Dict[str, List[bool]] = {}
    per_bin_loop: Dict[str, List[bool]] = {}
    per_lang: Dict[str, List[float]] = {}
    per_bin: Dict[str, List[float]] = {}
    per_dom: Dict[str, List[float]] = {}
    # batches never mix bins (padding to the longest prompt would waste the long bins' memory); one bin when unbinned
    groups: List[tuple] = []
    if cfg.length_bins and eval_prompts.length_bin:
        for b in LENGTH_BINS:
            ii = [i for i, x in enumerate(eval_prompts.length_bin) if x == b]
            if ii:
                groups.append((b, ii))
    else:
        groups.append((None, list(range(len(eval_prompts)))))
    work = []
    for b, ii in groups:
        bs = max(1, cfg.prompts_per_step_by_bin.get(b, cfg.micro_batch_seqs) if b else cfg.micro_batch_seqs)
        mnt = cfg.max_new_tokens_by_bin.get(b, cfg.max_new_tokens) if b else cfg.max_new_tokens
        work += [(b, ii[s: s + bs], mnt) for s in range(0, len(ii), bs)]
    for b, chunk, mnt in work:
        roll, samples, out = rollout_and_reward(policy, tokenizer, reward, eval_prompts, chunk, cfg, stop_ids, pad_id,
                                                device, autocast, greedy=True, max_new_tokens=mnt, sampler=sampler)
        lf = loop_flags(reward, samples, out.status, out.decoded, roll.finished)
        if b:
            per_bin.setdefault(b, []).extend(out.rewards)
            per_bin_fin.setdefault(b, []).extend(roll.finished)
            per_bin_loop.setdefault(b, []).extend(lf)
        rewards += out.rewards
        qes += [q for q in out.qe if q is not None]
        fin += roll.finished
        loops += lf
        for sm, r in zip(samples, out.rewards):
            per_lang.setdefault(sm.iso, []).append(r)
            if sm.domain:
                per_dom.setdefault(sm.domain, []).append(r)
    res = {"eval/reward": sum(rewards) / max(len(rewards), 1),
           "eval/qe_mean_scored": sum(qes) / len(qes) if qes else float("nan"),
           "eval/finished_rate": sum(fin) / max(len(fin), 1), "eval/n": len(rewards),
           "eval/greedy_finished_rate": sum(fin) / max(len(fin), 1), "eval/greedy_loop_rate": sum(loops) / max(len(loops), 1)}
    for k, v in per_bin_fin.items():
        res[f"eval/greedy_finished_rate_bin/{k}"] = sum(v) / len(v)
        res[f"eval/greedy_loop_rate_bin/{k}"] = sum(per_bin_loop[k]) / len(v)
        res[f"eval/greedy_n_bin/{k}"] = float(len(v))
    for k, v in per_lang.items():
        res[f"eval/reward/{k}"] = sum(v) / len(v)
    for k, v in per_bin.items():
        res[f"eval/reward_bin/{k}"] = sum(v) / len(v)
    for k, v in per_dom.items():
        res[f"eval/reward_domain/{k}"] = sum(v) / len(v)
    return res


def load_benchmark_dev(path: str, languages, per_language: int, seed: int = 0) -> List[Dict[str, str]]:
    """Deterministic stratified sample of the dev rows: `per_language` per language (smallest hash of the English text first)."""
    from gemma_mt.data.prepare_sft import split_fraction
    from gemma_mt.eval import benchmarks as B

    rows = [r for r in B.load_normalised_file(path) if r["iso"] in set(languages)]
    by: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by.setdefault(r["iso"], []).append(r)
    out: List[Dict[str, str]] = []
    for iso in sorted(by):
        rs = sorted(by[iso], key=lambda r: (split_fraction(r["english"], seed), r["english"]))
        out += rs[:per_language] if per_language > 0 else rs
    return out


def evaluate_benchmark(policy, tokenizer, rows: List[Dict[str, str]], cfg: RLConfig, stop_ids, pad_id, device, autocast,
                       sampler=None) -> Dict[str, float]:
    """Greedy translations of the dev rows -> macro chrF++ / BLEU (unweighted mean over languages) + per-language chrF++."""
    from gemma_mt.eval import metrics as M
    from gemma_mt.eval.eval_translation import generate_translations

    if cfg.eval_vllm_only and sampler is None:
        raise RuntimeError("eval_vllm_only: the dev-benchmark generation would fall back to HF generate (no vLLM sampler given)")
    outs = generate_translations(policy, tokenizer, rows, stop_ids=stop_ids, pad_id=pad_id, device=device,
                                 batch_size=cfg.benchmark_batch_size, max_new_tokens=cfg.benchmark_max_new_tokens,
                                 autocast=autocast, sampler=sampler, require_vllm=cfg.eval_vllm_only)
    by: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by.setdefault(r["iso"], []).append(i)
    res: Dict[str, float] = {}
    chrf, bleu = [], []
    for iso, idx in sorted(by.items()):
        h = [outs[i]["hyp"] for i in idx]
        ref = [rows[i]["reference"] for i in idx]
        c, b = M.chrf_pp(h, ref), M.bleu(h, ref)
        res[f"bench/chrf/{iso}"] = c
        res[f"bench/finished/{iso}"] = sum(bool(outs[i].get("finished", True)) for i in idx) / max(len(idx), 1)
        chrf.append(c)
        bleu.append(b)
    res["bench/chrf_macro"] = sum(chrf) / max(len(chrf), 1)
    res["bench/bleu_macro"] = sum(bleu) / max(len(bleu), 1)
    res["bench/finished_rate"] = sum(bool(o.get("finished", True)) for o in outs) / max(len(outs), 1)
    res["bench/n"] = len(rows)
    return res


def evaluate_aux_benchmarks(aux: List[Dict[str, Any]], policy, tokenizer, cfg: RLConfig, stop_ids, pad_id, device, autocast, sampler=None, step: Optional[int] = None,
                            last_step: Optional[int] = None) -> Dict[str, float]:
    """D-52: the monitor-only dev sets (`cfg.benchmark_aux`): greedy, macro chrF++ / BLEU per set, keys `bench_aux/<name>/...` (+ `finished/<iso>`, `seconds`). `aux` = [{name, rows,
    max_new_tokens, batch_size, every}]. D-56: an entry with `every: N` is evaluated only when `step` is a multiple of N or the last step (a costly set - whole papers - can run less often);
    `step=None` evaluates everything."""
    res: Dict[str, float] = {}
    t_all = time.time()
    for a in aux:
        ev = int(a.get("every") or 0)
        if step is not None and ev > 0 and step % ev != 0 and step != last_step:
            continue
        t0 = time.time()
        sub = dataclasses.replace(cfg, benchmark_max_new_tokens=a["max_new_tokens"], benchmark_batch_size=a["batch_size"])
        r = evaluate_benchmark(policy, tokenizer, a["rows"], sub, stop_ids, pad_id, device, autocast, sampler=sampler)
        res.update({k.replace("bench/", f"bench_aux/{a['name']}/", 1): v for k, v in r.items()})
        res[f"bench_aux/{a['name']}/seconds"] = time.time() - t0
    res["bench_aux/seconds_total"] = time.time() - t_all
    return res


def build_prompt_schedule(cfg: RLConfig, prompts: RLPrompts) -> PromptSchedule:
    """The trainer's prompt schedule (deterministic in (seed, step)). A single-bin stage that runs to its cap keeps its bin through an `extend_if_improving` extension (the last curriculum entry grows by the
    maximum extension; the draw of steps < cap is unchanged). Also used by scripts/clean_rerun_nllb_draws.py to replay which prompts a stage draws."""
    curriculum = dict(cfg.curriculum_steps)
    max_ext = int(cfg.steps * cfg.extend_max_frac + 1e-9) if cfg.extend_if_improving else 0
    if max_ext and curriculum and sum(int(v) for v in curriculum.values()) >= cfg.steps:
        curriculum[list(curriculum)[-1]] = int(curriculum[list(curriculum)[-1]]) + max_ext
    return PromptSchedule(prompts.iso, cfg.language_alpha, cfg.seed,
                          length_bin=prompts.length_bin if cfg.length_bins else None,
                          curriculum=curriculum or None, language_weights=cfg.language_weights or None,
                          domain=prompts.domain or None, domain_weights=cfg.domain_weights or None, domain_bin_weights=cfg.domain_bin_weights or None,
                          floor_steps=cfg.language_floor_steps, floor_prompts=cfg.language_floor_prompts)


def train_rl(cfg: RLConfig, tokenizer: Any = None, scorer: Any = None, prompts: Optional[RLPrompts] = None,
             eval_prompts: Optional[RLPrompts] = None, lid: Any = None, vllm_sampler: Any = None) -> Dict[str, Any]:
    """`vllm_sampler`: a ready VLLMSampler to use instead of building one from the `vllm_*` fields (tests / CPU smoke)."""
    import torch

    validate_config(cfg)
    for d in cfg.extra_libs:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.append(d)
    set_seed(cfg.seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name or cfg.model_name)
    fmt.assert_template_matches_reference(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # The vLLM engine is created BEFORE the policy / reference / optimizer take their memory (it sizes its cache against what is free);
    # a start-up failure then stops the run before anything expensive was loaded. HF sampler: built after the policy below.
    vsampler = vllm_sampler
    if cfg.sampler == "vllm" and vsampler is None:
        from gemma_mt.rl.vllm_sampler import build_vllm_sampler

        os.makedirs(cfg.out_dir, exist_ok=True)
        vsampler = build_vllm_sampler(cfg, cfg.model_name, log_dir=cfg.out_dir)
    if vsampler is not None:
        set_seed(cfg.seed)                          # engine start-up re-seeds the global RNGs (in-process mode)
        print(f"[gemma_mt] vLLM sampler up ({cfg.vllm_mode}): {getattr(vsampler.engine, 'worker_info', None) or vsampler.engine.info()}", flush=True)
    try:
        return _train_rl(cfg, tokenizer, scorer, prompts, eval_prompts, lid, device, pad_id, vsampler)
    finally:
        if vsampler is not None:
            vsampler.close()


def _train_rl(cfg: RLConfig, tokenizer, scorer, prompts, eval_prompts, lid, device, pad_id, vsampler) -> Dict[str, Any]:
    import torch

    from gemma_mt.losses import completion_logprobs
    from gemma_mt.modeling import load_model, save_model

    policy = load_model(cfg.model_name, attn_implementation=cfg.attn_implementation, dtype=_dtype(cfg.param_dtype),
                        device=str(device))
    stop = fmt.resolve_stop_tokens(tokenizer, policy.generation_config)
    for w in stop.warnings:
        print("[gemma_mt] WARNING:", w, flush=True)
    fmt.ensure_generation_config(policy.generation_config, stop, pad_id=pad_id)
    if cfg.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    policy.config.use_cache = False

    ref = None
    if cfg.kl_coef > 0:
        ref = load_model(cfg.model_name, attn_implementation=cfg.attn_implementation, dtype=_dtype(cfg.ref_dtype),
                         device=str(device))
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    if scorer is None:
        scorer = build_scorer(cfg.scorer, str(device))
    token_counter = getattr(scorer, "token_lengths", None)
    # ---- prompts
    if prompts is None:
        prompts = load_rl_prompts(cfg.prompts_path, max_src_tokens=cfg.max_src_tokens, token_counter=token_counter)
    if eval_prompts is None:
        if cfg.eval_prompts_path:
            eval_prompts = load_rl_prompts(cfg.eval_prompts_path, max_src_tokens=cfg.max_src_tokens,
                                           token_counter=token_counter)
        else:
            tr_idx, ev_idx = split_eval_prompts(prompts, min(cfg.eval_prompts, len(prompts) // 10), cfg.seed, cfg.eval_min_per_bin)
            prompts, eval_prompts = _subset(prompts, tr_idx), _subset(prompts, ev_idx)
    if len(prompts) == 0:
        raise ValueError("no usable RL prompts")
    schedule = build_prompt_schedule(cfg, prompts)
    max_ext = int(cfg.steps * cfg.extend_max_frac + 1e-9) if cfg.extend_if_improving else 0
    if cfg.domain_weights and not prompts.domain:
        raise ValueError(f"domain_weights is set but the prompts at {cfg.prompts_path!r} have no `domain` column")
    if cfg.length_bins:                              # D-45: a staged bin without prompts used to be skipped silently (e.g. b2048 rows dropped by max_src_tokens < 1025)
        empty = [b for b, n in (cfg.curriculum_steps or {}).items() if int(n) > 0 and b not in schedule.cycle]
        if empty:
            raise ValueError(f"curriculum_steps lists {empty} but the loaded prompts have no such rows (prompts_path {cfg.prompts_path!r}, max_src_tokens {cfg.max_src_tokens}, "
                             f"drops {prompts.dropped}): the stage would be skipped silently")

    reward_cfg = dataclass_from_dict(RewardConfig, cfg.reward)
    reward = TranslationReward(reward_cfg, scorer, tokenizer, stop.ids, lid=lid, pad_id=pad_id)   # builds + self-tests GlotLID when reward.lid_enabled (D-37)
    lo, hi = reward.reward_range
    mode_of = {l: reward.content_spec(l) for l in prompts.languages()}
    lid_on = [l for l in prompts.languages() if reward_cfg.lid_enabled and reward.lid_spec(l)[2]]
    print(f"[gemma_mt] reward content modes over {len(mode_of)} prompt languages: "
          + ", ".join(f"{m}={sum(1 for v in mode_of.values() if v[0] == m)}" for m in ("qe", "blend", "consensus"))
          + f"; GlotLID guard {'on for %d languages' % len(lid_on) if reward_cfg.lid_enabled else 'off'}; consensus {'on' if reward_cfg.consensus else 'off'}"
          + (f"; confidence rules on (gate {reward_cfg.confidence_gate}, trust {reward_cfg.confidence_trust}, cap {reward_cfg.confidence_cap}, modes {reward_cfg.confidence_modes})" if reward_cfg.confidence else ""), flush=True)
    unscored = sorted(l for l in prompts.languages() if reward.shaper.reliability(l) <= 0.0)
    if unscored:
        print(f"[gemma_mt] languages with advantage weight 0 (their groups teach nothing): {unscored}", flush=True)

    # ---- optimiser
    optimizer = build_rl_optimizer(policy, cfg, device)          # AdamW (default) | Muon on the decoder matrices + AdamW on the rest (rl/optim.py); always FRESH at the start of a stage

    os.makedirs(cfg.out_dir, exist_ok=True)
    archive = None
    if cfg.archive:                                  # D-41: opened here, so a bad path stops the run before step 1; later write failures never stop it
        from gemma_mt.rl.archive import TranslationArchive

        archive = TranslationArchive(cfg.archive_path or os.path.join(cfg.out_dir, "archive", "archive.sqlite"),
                                     min_reward_by_mode=cfg.archive_min_reward_by_mode, min_consensus_chrf=cfg.archive_min_consensus_chrf,
                                     flush_seconds=cfg.archive_flush_seconds, queue_size=cfg.archive_queue_size)
        print(f"[gemma_mt] translation archive: {archive.path} ({archive.size} rows), thresholds {archive.min_reward_by_mode}, blend consensus >= {archive.min_consensus_chrf}", flush=True)
    last_path = os.path.join(cfg.out_dir, "last", "trainer_state.pt")
    guard = KLGuard(cfg.kl_guard_threshold, cfg.kl_guard_factor, cfg.kl_guard_min_scale, cfg.kl_guard_window, cfg.kl_guard_cooldown, recover=cfg.kl_guard_recover,
                    recover_factor=cfg.kl_guard_recover_factor, recover_below=cfg.kl_guard_recover_below, recover_window=cfg.kl_guard_recover_window)
    plateau = (PlateauStop(cfg.plateau_stop_window, cfg.plateau_stop_block, cfg.plateau_stop_min_finished, cfg.plateau_stop_min_lr_scale, cfg.plateau_stop_bootstrap, cfg.seed, cfg.plateau_stop_ci_high_max)
               if cfg.plateau_stop else None)
    patience = (PatienceStop(cfg.early_stop_patience, cfg.early_stop_block, cfg.early_stop_eps_reward, cfg.early_stop_eps_finished, cfg.early_stop_eps_loop,
                             cfg.early_stop_min_finished, cfg.early_stop_min_lr_scale) if cfg.early_stop else None)
    cap_now = cfg.steps                              # the current hard cap (raised by extend_if_improving)
    extensions: List[Dict[str, Any]] = []
    guard_events_path = os.path.join(cfg.out_dir, "kl_guard_events.jsonl")
    guard_events_run: List[Dict[str, Any]] = []

    def log_guard_event(ev: Dict[str, Any]) -> None:
        """Every guard event (cut / recover / restore) is appended to <out_dir>/kl_guard_events.jsonl: the effective learning rate per step is a primary result of the run."""
        try:
            with open(guard_events_path, "a") as f:
                f.write(json.dumps(dict(ev, time=time.time())) + "\n")
        except OSError as e:
            print(f"[gemma_mt] WARNING: cannot write {guard_events_path}: {e}", flush=True)

    restored_guard: Optional[Dict[str, Any]] = None
    restored_controls: Optional[Dict[str, Any]] = None
    step, best_eval, resumed = 0, -float("inf"), False
    best_select = -float("inf")                     # best value of the selection metric so far (cfg.select_by)
    reset_selection = False                         # D-52: resumed from a checkpoint that was selected on another benchmark -> the resumed policy is scored first, see below
    if cfg.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        policy.load_state_dict(ck["model"])
        if optimizer_kind_of_state(ck["optimizer"]) != cfg.optimizer:
            raise ValueError(f"the checkpoint {last_path} holds a {optimizer_kind_of_state(ck['optimizer'])} optimizer state but optimizer={cfg.optimizer}: a new stage starts in a NEW out_dir with a fresh optimizer; "
                             "a crash-restart must keep the optimizer")
        optimizer.load_state_dict(ck["optimizer"])
        set_rng_state(ck["rng"])
        step, best_eval, resumed = ck["state"]["step"], ck["state"]["best_eval"], True
        best_select = ck["state"].get("best_select", best_select)
        if cfg.kl_guard_restore:                    # 2026-10-28: the guard / operator state survives a restart (a checkpoint written before this feature has no such keys: nothing changes)
            if guard.enabled:
                restored_guard = guard.load_state_dict(ck["state"].get("kl_guard"))
            restored_controls = ck["state"].get("controls")
            if plateau is not None:
                plateau.load_state_dict(ck["state"].get("plateau"))
            if patience is not None:
                patience.load_state_dict(ck["state"].get("patience"))
        if reward.conf is not None:                 # D-47: the running per-language confidence state survives a restart (a checkpoint without it: the tracker starts empty)
            n_conf = reward.load_conf_state(ck["state"].get("confidence"))
            print(f"[gemma_mt] confidence tracker: {n_conf} languages restored from the checkpoint", flush=True)
        if cfg.extend_if_improving and ck["state"].get("caps"):          # the extended cap and the decisions so far survive a restart
            cap_now = min(max(int(ck["state"]["caps"].get("cap_now", cfg.steps)), cfg.steps), cfg.steps + max_ext)
            extensions = list(ck["state"]["caps"].get("extensions", []))
        reset_selection = bool(cfg.resume_reset_selection and ck["state"].get("bench_path") != cfg.benchmark_eval_path)
        if reset_selection:
            best_select = -float("inf")
        print(f"[gemma_mt] resumed RL from step {step}" + (" (selection reset: the checkpoint was selected on another benchmark)" if reset_selection else ""), flush=True)

    logger = MetricsLogger(os.path.join(cfg.out_dir, "metrics.jsonl"), cfg.wandb_project, cfg.run_name,
                           dataclasses.asdict(cfg))
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump({"config": dataclasses.asdict(cfg), "reward_range": [lo, hi], "n_prompts": len(prompts),
                   "n_eval": len(eval_prompts), "languages": prompts.languages(),
                   "stop_ids": stop.ids, "prompt_drops": prompts.dropped}, f, indent=2)
    print(f"[gemma_mt] rl: {len(prompts)} prompts ({len(eval_prompts)} eval), languages={prompts.languages()}, "
          f"reward range [{lo}, {hi}], preset={reward_cfg.preset}", flush=True)

    ac_dtype = None if cfg.autocast_dtype in (None, "none", "None") else _dtype(cfg.autocast_dtype)

    def autocast():
        if ac_dtype is None:
            return torch.autocast(device_type=device.type, enabled=False)
        return torch.autocast(device_type=device.type, dtype=ac_dtype)

    def save_last():
        os.makedirs(os.path.dirname(last_path), exist_ok=True)
        tmp = last_path + ".tmp"
        torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "rng": get_rng_state(),
                    "state": {"step": step, "best_eval": best_eval, "best_select": best_select, "base_chrf": base_chrf,
                              "bench_path": cfg.benchmark_eval_path, "kl_guard": guard.state_dict(),
                              "controls": {"lr_file_scale": float(ctl.file_scale) if ctl is not None else 1.0},
                              "plateau": plateau.state_dict() if plateau is not None else None,
                              "patience": patience.state_dict() if patience is not None else None,
                              "optimizer_kind": cfg.optimizer, "caps": {"cap_now": cap_now, "extensions": extensions},
                              "confidence": reward.conf_state()}}, tmp)
        os.replace(tmp, last_path)

    from gemma_mt.hub.push_checkpoint import CheckpointPusher

    pusher = CheckpointPusher(cfg.hub_repo_id, private=cfg.hub_private, stage="rl", branch=cfg.hub_branch, tag_prefix="rl-step",
                              squash=cfg.hub_squash, prune_tags=cfg.hub_prune_tags)
    stage_ends: Dict[int, str] = {}                          # step -> stage name, from the cumulative curriculum lengths; the last step is the end of the mixed stage
    if cfg.hub_repo_id and cfg.hub_stage_branches:
        acc = 0
        for name, n in (cfg.curriculum_steps or {}).items():
            acc += int(n)
            if acc <= cfg.steps:
                stage_ends[acc] = name
        stage_ends[cfg.steps] = "mixed" if cfg.steps not in stage_ends else stage_ends[cfg.steps]
    stage_pushers: list = []
    latest_pusher = (CheckpointPusher(cfg.hub_repo_id, private=cfg.hub_private, stage="rl", branch=cfg.hub_latest_branch, tag_prefix=None, squash=cfg.hub_squash)
                     if cfg.hub_repo_id and cfg.hub_latest_branch and cfg.hub_latest_every else None)
    if cfg.hub_repo_id and not pusher.enabled:
        print("[gemma_mt] WARNING: hub_repo_id is set but no write token is available: the best RL checkpoint will NOT be pushed", flush=True)

    policy_version = [0]                              # optimizer steps taken in this process; the engine's weights are tagged with it

    def sync_policy() -> float:
        """vLLM only: make the engine's weights those of the current policy (no-op when they already are). Seconds spent."""
        if vsampler is None:
            return 0.0
        return vsampler.sync(policy, policy_version[0])

    bench_rows: List[Dict[str, str]] = []
    base_chrf = float("nan")
    if cfg.benchmark_eval_path:
        bench_rows = load_benchmark_dev(cfg.benchmark_eval_path, prompts.languages(), cfg.benchmark_eval_per_language, cfg.seed)
        if not bench_rows:
            raise ValueError(f"no dev benchmark rows for the RL languages in {cfg.benchmark_eval_path}")
        if not resumed or reset_selection:           # the SFT starting point (or, D-52, the resumed policy on a new benchmark) is the bar every RL checkpoint has to clear
            sync_policy()
            b0 = evaluate_benchmark(policy, tokenizer, bench_rows, cfg, stop.ids, pad_id, device, autocast(), sampler=vsampler)
            if vsampler is not None:
                vsampler.sleep()
            base_chrf = b0["bench/chrf_macro"]
            logger.log(step, {"baseline": 1.0, **b0})
            print(f"[gemma_mt] {'resumed policy' if resumed else 'SFT starting point'} on {len(bench_rows)} dev rows: chrF++ macro {base_chrf:.2f} "
                  f"BLEU {b0['bench/bleu_macro']:.2f} finished {b0['bench/finished_rate']:.2f}", flush=True)
            if cfg.select_by == "benchmark_chrf":
                best_select = base_chrf
        else:
            base_chrf = float(ck["state"].get("base_chrf", float("nan")))

    aux_sets: List[Dict[str, Any]] = []
    for a in cfg.benchmark_aux:                      # D-52: monitor-only dev sets
        rows_a = load_benchmark_dev(a["path"], prompts.languages(), int(a.get("per_language", 40)), cfg.seed)
        if not rows_a:
            raise ValueError(f"benchmark_aux {a['name']!r}: no dev rows for the RL languages in {a['path']}")
        aux_sets.append({"name": str(a["name"]), "rows": rows_a, "max_new_tokens": int(a.get("max_new_tokens", cfg.benchmark_max_new_tokens)),
                         "batch_size": int(a.get("batch_size", cfg.benchmark_batch_size)), "every": int(a.get("every", 0) or 0)})
    if aux_sets and (not resumed or reset_selection):
        sync_policy()
        a0 = evaluate_aux_benchmarks(aux_sets, policy, tokenizer, cfg, stop.ids, pad_id, device, autocast(), sampler=vsampler)
        if vsampler is not None:
            vsampler.sleep()
        logger.log(step, {"baseline": 1.0, **a0})
        print("[gemma_mt] aux dev sets at the start: " + ", ".join("%s chrF++ %.2f" % (a_["name"], a0["bench_aux/%s/chrf_macro" % a_["name"]]) for a_ in aux_sets), flush=True)

    t_start = time.time()
    summary: Dict[str, Any] = {"steps": step}
    controls = RunControls(cfg.out_dir, cfg.steps) if cfg.run_controls else None      # D-56: WINDDOWN / LR_SCALE files in out_dir
    if restored_guard is not None:
        print(f"[gemma_mt] KL guard state restored from the checkpoint: learning-rate scale {guard.lr_scale:.4f}, last cut at step {guard.last_cut if guard.last_cut > -10 ** 8 else 'never'}, "
              f"{restored_guard['kl_hist_len']} KL values in the window, {guard.n_cuts} cuts / {guard.n_recoveries} recoveries so far", flush=True)
        log_guard_event(dict(restored_guard, step=step, event="restore", scale_new=guard.lr_scale))
    elif resumed and guard.enabled and cfg.kl_guard_restore:
        print("[gemma_mt] KL guard: the checkpoint has no guard state (written before it was persisted): the guard starts at the full learning rate", flush=True)
    if restored_controls and controls is not None and controls.restore_lr_scale(float(restored_controls.get("lr_file_scale", 1.0))):
        print(f"[gemma_mt] WARNING: the LR_SCALE file was missing but the checkpoint recorded an operator multiplier {float(restored_controls['lr_file_scale']):g}: the file was written again "
              f"(write 1 into it to undo)", flush=True)
        log_guard_event({"step": step, "event": "restore_lr_scale_file", "scale_new": float(restored_controls["lr_file_scale"])})
    if controls is not None:
        controls.total = cap_now
    if cap_now != cfg.steps:
        print(f"[gemma_mt] cap restored from the checkpoint: {cap_now} steps (base {cfg.steps}, {len(extensions)} extension decision(s) so far)", flush=True)
    lr_mult_sum, lr_mult_n, lr_mult_min = 0.0, 0, float("inf")
    greedy_warn_last = -10 ** 9
    schedule_staged = bool(cfg.length_bins and schedule.curriculum and schedule.cycle)

    def stage_info(done: int):
        """(name, first step, is the last stage of the run) of the stage the step just completed (`done` steps done) belongs to; one stage 'run' without a curriculum."""
        if not schedule_staged:
            return "run", 0, True
        name = schedule.bin_of_step(done - 1)
        start = dict(schedule.stage_boundaries()).get(name, 0)
        return name, start, name == schedule.bin_of_step(max(cfg.steps - 1, 0))

    lwc = build_lang_weight_controller(cfg, schedule, prompts.languages(), {l: v[0] for l, v in mode_of.items()}, start_step=step, canon=canonical_iso)   # D-65: LANG_WEIGHTS.json + adaptive updater (None = all off)
    if lwc is not None:
        dev_sizes: Dict[Any, int] = {}
        for name_, rows_ in [("bench", bench_rows)] + [(a_["name"], a_["rows"]) for a_ in aux_sets]:
            for row_ in rows_:
                k_ = (name_, canonical_iso(row_.get("iso") or row_.get("language") or ""))
                dev_sizes[k_] = dev_sizes.get(k_, 0) + 1
        lwc.set_dev_sizes(dev_sizes)
    end_step = cap_now
    ctl = None
    ext_path = os.path.join(cfg.out_dir, "extensions.jsonl")
    es_path = os.path.join(cfg.out_dir, "early_stop_checks.jsonl")

    def _append(path: str, row: Dict[str, Any]) -> None:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(dict(row, time=time.time()), default=str) + "\n")
        except OSError as e:
            print(f"[gemma_mt] WARNING: cannot write {path}: {e}", flush=True)

    while step < end_step:
        t0 = time.time()
        if controls is not None:
            ctl = controls.update(step)
            end_step = ctl.end_step
            if step >= end_step:                       # a wind-down that was already complete when this (restarted) run began
                break
        lr_scale_used = guard.lr_scale
        lr_mult_eff = lr_scale_used * (ctl.lr_mult if ctl is not None else 1.0)         # guard scale x operator file x wind-down: the part of the learning rate that is not the plan
        lr = cfg.learning_rate * lr_scale_used * min(1.0, (step + 1) / max(cfg.warmup_steps, 1)) * (ctl.lr_mult if ctl is not None else 1.0)
        for g in optimizer.param_groups:
            g["lr"] = lr
        lw_rec = lwc.update(step) if lwc is not None else {}          # D-65: LANG_WEIGHTS.json / adaptive weights -> schedule.set_language_weights (before this step's draw)
        step_bin = schedule.bin_of_step(step) if cfg.length_bins else None
        n_step = cfg.prompts_per_step_by_bin.get(step_bin, cfg.prompts_per_step) if step_bin else cfg.prompts_per_step
        mnt = cfg.max_new_tokens_by_bin.get(step_bin, cfg.max_new_tokens) if step_bin else cfg.max_new_tokens
        if step_bin == MIXED and MIXED not in cfg.max_new_tokens_by_bin and cfg.max_new_tokens_by_bin:
            mnt = max(cfg.max_new_tokens_by_bin.values())
        idx = schedule.indices(step, n_step)
        # ---------------- 1. sample + reward (no grad)
        t_sync = sync_policy()
        check_lp = cfg.check_sampler_logprobs or (vsampler is not None and cfg.vllm_check_every > 0 and step % cfg.vllm_check_every == 0)
        roll, samples, out = rollout_and_reward(policy, tokenizer, reward, prompts, idx, cfg, stop.ids, pad_id, device,
                                                autocast(), return_logprobs=check_lp or reward_cfg.confidence, max_new_tokens=mnt,
                                                sampler=vsampler, key=step)
        t_gen = vsampler.last_generate_seconds if vsampler is not None else float("nan")
        if vsampler is not None:
            vsampler.sleep()                        # only with vllm_sleep: the engine's memory goes back to the trainer for the update
        N, G = len(roll.completions), cfg.group_size
        assert N == len(idx) * G == len(out.rewards) == len(out.weights), "rollout / reward length mismatch"
        assert all(len(c) >= 1 for c in roll.completions), "empty completion"
        rewards = torch.tensor(out.rewards, dtype=torch.float32)
        assert torch.isfinite(rewards).all(), "non-finite reward"
        assert rewards.min() >= lo - 1e-5 and rewards.max() <= hi + 1e-5, \
            f"reward outside the configured range [{lo}, {hi}]: {rewards.min().item()}..{rewards.max().item()}"
        t_archive = 0.0
        if archive is not None:                     # D-41: nothing is recomputed, the fields of `out` are read off; the writer thread does the rest
            from gemma_mt.rl.archive import archive_step

            t_a = time.time()
            archive_step(archive, reward, samples, out, step + 1, step_bin or "",
                         [prompts.length_bin[i] for i in idx] if prompts.length_bin else None, G)
            t_archive = time.time() - t_a
        if cfg.greedy_in_group and cfg.greedy_stats == "sampled":
            adv, flat_frac = grpo.group_advantages_sampled_baseline(rewards, G, cfg.advantage_norm, cfg.std_eps, greedy_cap=cfg.greedy_adv_cap)
        else:
            adv, flat_frac = grpo.group_advantages(rewards, G, cfg.advantage_norm, cfg.std_eps)
        hard_failed = torch.tensor([st != "ok" for st in out.status], dtype=torch.bool)
        if cfg.hard_fail_mask != "off":             # D-61 (default off): the hard-failed samples leave the policy gradient
            adv, flat_frac = grpo.group_advantages_masked(rewards, G, hard_failed, cfg.hard_fail_mask, cfg.advantage_norm, cfg.std_eps)
        adv = adv * torch.tensor(out.weights, dtype=torch.float32)
        if cfg.greedy_in_group and cfg.greedy_adv_scale != 1.0:
            adv = grpo.scale_last_member(adv, G, cfg.greedy_adv_scale)
        assert adv.shape == (N,) and torch.isfinite(adv).all(), "bad advantages"
        greedy_rec: Dict[str, float] = {}
        if cfg.greedy_in_group:                      # 2026-10-28: the greedy (temperature 0) member of every group is what loops at inference: monitor it directly
            gi = list(range(G - 1, N, G))
            g_fin = [roll.finished[i] for i in gi]
            g_loop = loop_flags(reward, [samples[i] for i in gi], [out.status[i] for i in gi], [out.decoded[i] for i in gi], g_fin)
            tok_w = adv.abs() * torch.tensor([float(len(c)) for c in roll.completions], dtype=torch.float32)
            tok_w = tok_w.view(-1, G)
            g_share = float(tok_w[:, -1].sum() / tok_w.sum().clamp(min=1e-12))
            rv = rewards.view(-1, G)
            greedy_rec = {"greedy_finished_rate": sum(g_fin) / len(g_fin), "greedy_loop_rate": sum(g_loop) / len(g_loop), "greedy_reward_mean": float(rv[:, -1].mean()),
                          "sampled_reward_mean": float(rv[:, :-1].mean()), "greedy_adv_token_share": g_share, "greedy_len_mean": sum(len(roll.completions[i]) for i in gi) / len(gi)}
        t_sample = time.time() - t0
        if lwc is not None:                         # D-65: per-language statistics of this step (the updater's window; metrics only with lang_stats_log / adaptive)
            lw_iso = [prompts.iso[i] for i in idx]
            lw_keys = [f"{prompts.length_bin[i] if prompts.length_bin else ''}|{prompts.domain_of(i)}" for i in idx]
            lwc.observe(step, lw_iso, lw_keys, out.rewards, out.status, G, qe=out.qe, adv_w=out.weights, cons=getattr(out, "consensus_chrf", None) or None,
                        pen=getattr(out, "anchors_penalty", None) or None)
            if lwc.log_stats:
                lw_rec = dict(lw_rec, **lwc.step_record(lw_iso, lw_keys, out.rewards, out.status, G))

        # ---------------- 2. log-probs + update
        total_tokens = float(sum(len(c) for c in roll.completions))
        normaliser = total_tokens if cfg.loss_agg == "token" else float(N)
        mb = cfg.micro_batch_seqs
        if cfg.micro_batch_tokens > 0:
            longest = max(len(roll.prompts[r]) + len(roll.completions[r]) for r in range(N))
            mb = max(1, min(cfg.micro_batch_seqs, cfg.micro_batch_tokens // max(longest, 1)))
        micro = [list(range(s, min(s + mb, N))) for s in range(0, N, mb)]
        batches = []
        for rows in micro:
            b = assemble_batch([roll.prompts[r] for r in rows], [roll.completions[r] for r in rows], pad_id, device)
            assert int(b["completion_mask"].sum()) == sum(len(roll.completions[r]) for r in rows), "mask count"
            batches.append((rows, b))
        old_lp, ref_lp = {}, {}
        policy.eval()
        with torch.no_grad(), autocast():
            for k, (rows, b) in enumerate(batches):
                if cfg.num_updates_per_batch > 1:
                    old_lp[k] = completion_logprobs(policy, b["input_ids"], b["attention_mask"], b["prompt_width"],
                                                    b["completion_mask"], cfg.temperature, cfg.loss_chunk_tokens)
                if ref is not None:
                    ref_lp[k] = completion_logprobs(ref, b["input_ids"], b["attention_mask"], b["prompt_width"],
                                                    b["completion_mask"], cfg.temperature, cfg.loss_chunk_tokens)
        policy.train()
        metrics_acc = {"pg": 0.0, "kl": 0.0, "clip": 0.0, "ratio": 0.0, "tok": 0.0}
        gnorm = float("nan")
        sampler_gap = float("nan")
        gap_sum, gap_n = 0.0, 0
        for upd in range(cfg.num_updates_per_batch):
            optimizer.zero_grad(set_to_none=True)
            acc = {"pg": 0.0, "kl": 0.0, "clip": 0.0, "ratio": 0.0, "tok": 0.0}
            for k, (rows, b) in enumerate(batches):
                with autocast():
                    lp = completion_logprobs(policy, b["input_ids"], b["attention_mask"], b["prompt_width"],
                                             b["completion_mask"], cfg.temperature, cfg.loss_chunk_tokens)
                cm = b["completion_mask"]
                assert lp.shape == cm.shape and torch.isfinite(lp).all(), "non-finite / misshaped log-probs"
                assert bool((lp[cm == 0] == 0).all()), "log-probs leaked outside the completion mask"
                if check_lp and roll.sampler_logp is not None and upd == 0:
                    diffs = []
                    for j, r in enumerate(rows):
                        s = torch.tensor(roll.sampler_logp[r], dtype=lp.dtype)
                        d = (lp[j, : len(s)].detach().cpu() - s).abs()
                        diffs.append(d.max().item())
                        gap_sum += float(d.sum())
                        gap_n += len(s)
                    sampler_gap = max([sampler_gap if sampler_gap == sampler_gap else 0.0] + diffs) if diffs else sampler_gap
                res = grpo.grpo_token_loss(
                    lp, adv[rows].to(device), cm, old_logp=old_lp.get(k), ref_logp=ref_lp.get(k), kl_coef=cfg.kl_coef,
                    kl_estimator=cfg.kl_estimator, clip_eps=cfg.clip_eps, clip_eps_high=cfg.clip_eps_high,
                    loss_agg=cfg.loss_agg, normaliser=normaliser)
                assert torch.isfinite(res["loss"]), "non-finite loss"
                res["loss"].backward()
                for key, name in (("pg", "pg_sum"), ("kl", "kl_sum"), ("clip", "clip_sum"), ("ratio", "ratio_sum"),
                                  ("tok", "n_tokens")):
                    acc[key] += float(res[name])
            if upd == 0 and cfg.num_updates_per_batch == 1:
                ratio_mean = acc["ratio"] / max(acc["tok"], 1.0)
                assert abs(ratio_mean - 1.0) < 1e-5, f"on-policy ratio is {ratio_mean}, expected exactly 1"
            if step == 0 and not resumed and ref is not None and cfg.param_dtype == cfg.ref_dtype and upd == 0:
                kl0 = acc["kl"] / max(acc["tok"], 1.0)
                assert kl0 < 1e-3, f"policy and reference disagree at step 0 (mean KL {kl0}): the reference must be " \
                                    "an exact copy of the starting policy"
            gnorm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip))
            assert math.isfinite(gnorm), "non-finite gradient norm"
            optimizer.step()
            policy_version[0] += 1                  # the engine's copy is now stale: the next rollout / eval syncs first
            metrics_acc = acc
        step += 1
        sampler_gap_mean = gap_sum / gap_n if gap_n else float("nan")
        if vsampler is not None and gap_n and sampler_gap_mean > cfg.vllm_logp_gap_tol:
            raise RuntimeError(f"vLLM sampler and trainer disagree: mean |log p_vllm - log p_trainer| = {sampler_gap_mean:.3f} over {gap_n} tokens "
                               f"(tolerance {cfg.vllm_logp_gap_tol}; max {sampler_gap:.2f}). The engine's weights are not the policy's, or the "
                               "sampling distribution differs from the scored one: stopping instead of training on off-policy samples")

        tok = max(metrics_acc["tok"], 1.0)
        r_t = rewards
        rec = {"reward": float(r_t.mean()), "reward_std": float(r_t.std(unbiased=False)), "flat_group_frac": flat_frac, "hard_fail_frac": float(hard_failed.float().mean()),
               "adv_abs_mean": float(adv.abs().mean()), **greedy_rec, **({"adv_abs_mean_greedy": float(adv.view(-1, G)[:, -1].abs().mean()), "adv_abs_mean_sampled": float(adv.view(-1, G)[:, :-1].abs().mean())} if cfg.greedy_in_group else {}), "kl": metrics_acc["kl"] / tok,
               "pg_loss": metrics_acc["pg"] / tok, "clip_frac": metrics_acc["clip"] / tok,
               "ratio_mean": metrics_acc["ratio"] / tok, "grad_norm": gnorm, "lr": lr,
               "completion_len": total_tokens / N, "finished_rate": sum(roll.finished) / N,
               "weight_mean": float(sum(out.weights) / N), "sampler_logp_gap": sampler_gap,
               "t_sample": t_sample, "t_step": time.time() - t0}
        if vsampler is not None:
            rec.update({"sampler_logp_gap_mean": sampler_gap_mean, "t_sync": t_sync, "t_gen": t_gen,
                        "gen_tokens_per_s": vsampler.last_generate_tokens / max(t_gen, 1e-9)})
        rec.update({f"reward/{k}": v for k, v in out.stats.items()})
        rec.update(out.confidence)                      # D-47: `confidence/...` keys (empty unless reward.confidence)
        rec.update(lw_rec)
        rec["lr_mult_effective"] = lr_mult_eff
        lr_mult_sum, lr_mult_n, lr_mult_min = lr_mult_sum + lr_mult_eff, lr_mult_n + 1, min(lr_mult_min, lr_mult_eff)
        if cfg.greedy_share_warn > 0 and greedy_rec and greedy_rec["greedy_adv_token_share"] > cfg.greedy_share_warn and step - greedy_warn_last >= 10:
            greedy_warn_last = step
            print(f"[gemma_mt] WARNING: the greedy members carry {greedy_rec['greedy_adv_token_share']:.2f} of the token-weighted |advantage| of step {step} (uniform {1.0 / G:.2f}, limit {cfg.greedy_share_warn}): "
                  f"greedy loop rate {greedy_rec['greedy_loop_rate']:.2f}, greedy length {greedy_rec['greedy_len_mean']:.0f} vs mean {rec['completion_len']:.0f} tokens; lower greedy_adv_scale if this persists", flush=True)
        if ctl is not None and (ctl.file_scale != 1.0 or ctl.winddown):
            rec["lr_file_scale"] = ctl.file_scale
            rec["winddown_factor"] = ctl.winddown_factor
            rec["winddown_steps_left"] = float(max(end_step - step, 0))
        if guard.enabled:
            gev = guard.update(step, rec["kl"])
            if gev is not None:
                gev = dict(gev, lr_before=lr, lr_planned=cfg.learning_rate, lr_next_full=cfg.learning_rate * guard.lr_scale * (ctl.lr_mult if ctl is not None else 1.0))
                guard_events_run.append(gev)
                log_guard_event(gev)
                if plateau is not None:
                    plateau.note_guard_event(step)
                if gev["event"] == "cut":
                    print(f"[gemma_mt] KL guard: median KL of the last {gev['window']} steps {gev['kl_median']:.3f} > {cfg.kl_guard_threshold}: "
                          f"learning rate scale -> {guard.lr_scale:.4f}", flush=True)
                else:
                    print(f"[gemma_mt] KL guard: median KL of the last {gev['window']} steps {gev['kl_median']:.3f} < {cfg.kl_guard_recover_below} x {cfg.kl_guard_threshold}: "
                          f"learning rate scale {gev['scale_old']:.4f} -> {guard.lr_scale:.4f} (recovery, planned lr is the ceiling)", flush=True)
            rec["kl_guard_lr_scale"] = guard.lr_scale
            rec["kl_guard_median"] = guard.last_median
            rec["kl_guard_cut"] = 1.0 if gev is not None and gev["event"] == "cut" else 0.0
            rec["kl_guard_recover"] = 1.0 if gev is not None and gev["event"] == "recover" else 0.0
        if plateau is not None:
            plateau.observe(step, rec["reward"], rec["finished_rate"], lr_mult_eff, stage_info(step)[0])
        if patience is not None:
            st_name, st_start, st_last = stage_info(step)
            patience.observe(step, rec["reward"], rec["finished_rate"], lr_mult_eff, st_name, rec.get("greedy_loop_rate"), rec.get("greedy_finished_rate"))
            since = patience.steps_since_improve(step)
            rec["early_stop/steps_since_improve"] = float(since) if since is not None else -1.0
            for k_, v_ in (patience.smoothed or {}).items():
                if v_ is not None:
                    rec[f"early_stop/smooth_{k_}"] = float(v_)
            if step < end_step and not controls.winddown_active():
                dec = patience.check(step, st_name, st_start, cfg.early_stop_min_steps, st_last)
                if dec["fire"] and controls.request_winddown(cfg.early_stop_winddown, "early_stop", dec):
                    patience.triggered = dec
                    _append(es_path, dec)
                    print(f"[gemma_mt] EARLY STOP: no new best of the smoothed reward / finished rate / greedy loop rate for {dec['steps_since_improve']} steps (patience {cfg.early_stop_patience}, "
                          f"last improvement by {dec['last_improve_by']}); winding down over {cfg.early_stop_winddown} steps, the run ends at step {min(end_step, step + cfg.early_stop_winddown)}", flush=True)
                elif step % 25 == 0:
                    _append(es_path, dec)
        # ---- decision point: extend the cap while still improving, and/or start the planned final wind-down (a wind-down already requested wins: nothing is extended)
        dp = cap_now - cfg.planned_winddown_last
        if step == dp and controls is not None and (cfg.extend_if_improving or cfg.planned_winddown_last > 0) and not controls.winddown_active():
            extended = False
            if cfg.extend_if_improving:
                d_ = extension_decision(step, cfg.steps, cap_now, cfg.extend_steps, cfg.extend_max_frac, patience.steps_since_improve(step), cfg.extend_recent)
                _append(ext_path, d_)
                extensions.append(d_)
                print(f"[gemma_mt] extension decision @ {step}: {'EXTEND by %d to %d' % (d_['extend_steps'], d_['new_cap']) if d_['extend'] else 'no extension'} ({d_['reason']}; steps since the last improvement "
                      f"{d_['steps_since_improve']}, recent window {cfg.extend_recent}, used {d_['extension_used']} of {int(cfg.steps * cfg.extend_max_frac + 1e-9)})", flush=True)
                if d_["extend"]:
                    extended = True
                    cap_now = d_["new_cap"]
                    controls.total = cap_now
                    end_step = cap_now
                    rec["cap"] = float(cap_now)
            if not extended and cfg.planned_winddown_last > 0:
                controls.request_winddown(cfg.planned_winddown_last, "planned_winddown")
        if archive is not None:
            rec.update(archive.counters())
            rec["archive/hook_ms"] = 1000.0 * t_archive
        if step % cfg.log_every == 0 or step == 1:
            logger.log(step, rec)
            print(f"[gemma_mt] rl step {step}/{end_step} reward {rec['reward']:.3f} kl {rec['kl']:.4f} "
                  f"fin {rec['finished_rate']:.2f} len {rec['completion_len']:.1f} flat {flat_frac:.2f} "
                  f"gnorm {gnorm:.2f}" + (f" archive {int(rec['archive/size'])} (+{int(rec['archive/inserted'])} ~{int(rec['archive/improved'])} drop {int(rec['archive/dropped_queue'])})" if archive is not None else ""), flush=True)
            if step == 1 or step % max(cfg.log_every, 1) == 0:
                j = 0
                print(f"[gemma_mt]   sample [{samples[j].iso}] {samples[j].source[:60]!r} -> {out.decoded[j][:80]!r} "
                      f"status={out.status[j]} qe={out.qe[j]} reward={out.rewards[j]:.2f}"
                      + (f" diacritic_deficit={out.diacritic_deficit[j]:.2f}" if out.diacritic_deficit and out.diacritic_deficit[j] is not None else "")
                      + (f" lid={out.lid_label[j]}" if out.lid_label and out.lid_label[j] is not None else "")
                      + (f" consensus_chrf={out.consensus_chrf[j]:.1f}" if out.consensus_chrf and out.consensus_chrf[j] is not None else "")
                      + (f" mode={out.content_mode[j]}" if out.content_mode else ""),
                      flush=True)

        if cfg.eval_every and (step % cfg.eval_every == 0 or step == end_step) and len(eval_prompts):
            sync_policy()
            ev = evaluate_policy(policy, tokenizer, reward, eval_prompts, cfg, stop.ids, pad_id, device, autocast(), sampler=vsampler)
            bench = None
            b_every = cfg.benchmark_eval_every or cfg.eval_every
            if bench_rows and (step % b_every == 0 or step == end_step):
                bench = evaluate_benchmark(policy, tokenizer, bench_rows, cfg, stop.ids, pad_id, device, autocast(), sampler=vsampler)
                ev.update(bench)
                if aux_sets:
                    ev.update(evaluate_aux_benchmarks(aux_sets, policy, tokenizer, cfg, stop.ids, pad_id, device, autocast(), sampler=vsampler, step=step, last_step=end_step))
            if vsampler is not None:
                vsampler.sleep()
            logger.log(step, ev)
            if lwc is not None:
                lwc.observe_eval(step, ev)
            print(f"[gemma_mt] eval @ {step}: reward {ev['eval/reward']:.3f} finished {ev['eval/finished_rate']:.2f}"
                  + (f" | dev chrF++ macro {bench['bench/chrf_macro']:.2f} (SFT start {base_chrf:.2f}) BLEU {bench['bench/bleu_macro']:.2f}"
                     if bench else ""), flush=True)
            best_eval = max(best_eval, ev["eval/reward"])
            metric, fin = None, None
            if cfg.select_by == "benchmark_chrf":
                if bench is not None:
                    metric, fin = bench["bench/chrf_macro"], bench["bench/finished_rate"]
            else:
                metric, fin = ev["eval/reward"], ev["eval/finished_rate"]
            if metric is not None and fin >= cfg.select_min_finished_rate and metric > best_select + cfg.select_min_gain:
                best_select = metric
                print(f"[gemma_mt] new best ({cfg.select_by}) {metric:.3f} @ step {step}", flush=True)
                if not cfg.keep_single_checkpoint:
                    save_model(policy, tokenizer, os.path.join(cfg.out_dir, "best"), stop_ids=stop.ids)
                if cfg.hub_repo_id:
                    stage_dir = os.path.join(cfg.out_dir, "hub_stage")
                    save_model(policy, tokenizer, stage_dir, stop_ids=stop.ids)
                    state = {"selected_by": cfg.select_by, "selection_value": metric, "eval_reward": ev["eval/reward"],
                             "eval_finished_rate": ev.get("eval/finished_rate"), "kl": rec.get("kl"), "reward_range": [lo, hi],
                             "sft_start_dev_chrf": base_chrf if bench_rows else None, "config": dataclasses.asdict(cfg)}
                    if bench is not None:
                        state.update({k: v for k, v in bench.items()})
                    pusher.push(stage_dir, step, state, as_best=False, cleanup=cfg.keep_single_checkpoint, block=False)
            summary["last_eval"] = ev
            if plateau is not None and step < end_step and step % cfg.eval_every == 0:
                st_name, st_start, st_last = stage_info(step)
                ev_fin = ev.get(f"eval/greedy_finished_rate_bin/{st_name}", ev["eval/greedy_finished_rate"] if st_name in ("run", MIXED) else None)
                dec = plateau.check(step, st_name, st_start, cfg.plateau_stop_min_steps_by_bin.get(st_name, cfg.plateau_stop_min_steps), st_last, ev_fin)
                try:
                    with open(os.path.join(cfg.out_dir, "plateau_checks.jsonl"), "a") as f:
                        f.write(json.dumps(dict(dec, time=time.time()), default=str) + "\n")
                except OSError:
                    pass
                if "slope_per_100_steps" in dec:
                    print(f"[gemma_mt] plateau check @ {step} (stage {st_name}, {dec['steps_in_stage']} steps): slope {dec['slope_per_100_steps']:+.4f}/100 steps, 95% CI [{dec['ci_low_per_100_steps']:+.4f}, "
                          f"{dec['ci_high_per_100_steps']:+.4f}] -> {'PLATEAU' if dec['fire'] else 'still moving'}", flush=True)
                else:
                    print(f"[gemma_mt] plateau check @ {step} (stage {st_name}, {dec['steps_in_stage']} steps): not evaluated ({dec['blocked']})", flush=True)
                if dec["fire"]:
                    if controls.request_winddown(cfg.plateau_stop_winddown, "plateau_stop", dec):
                        plateau.triggered = dec
                        print(f"[gemma_mt] PLATEAU STOP: the training reward has no trend over the last {cfg.plateau_stop_window} steps and the finished rate is {dec['finished_train_window']:.4f}; "
                              f"winding down over {cfg.plateau_stop_winddown} steps, the run ends at step {min(cfg.steps, step + cfg.plateau_stop_winddown)}", flush=True)
                    else:
                        print("[gemma_mt] plateau detected but a wind-down is already requested: nothing changed", flush=True)
                logger.log(step, {"plateau/fire": float(dec["fire"]), "plateau/steps_in_stage": float(dec["steps_in_stage"]),
                                  **({"plateau/slope_per_100": dec["slope_per_100_steps"], "plateau/ci_low_per_100": dec["ci_low_per_100_steps"],
                                      "plateau/ci_high_per_100": dec["ci_high_per_100_steps"]} if "slope_per_100_steps" in dec else {})})
        if latest_pusher is not None and latest_pusher.enabled and (step % cfg.hub_latest_every == 0 or step == end_step):
            latest_dir = os.path.join(cfg.out_dir, "hub_stage_latest")
            save_model(policy, tokenizer, latest_dir, stop_ids=stop.ids)
            latest_pusher.push(latest_dir, step, {"role": "latest", "reward": rec.get("reward"), "kl": rec.get("kl"), "best_selection_value": best_select,
                                                  "selected_by": cfg.select_by, "config": dataclasses.asdict(cfg)},
                               as_best=False, cleanup=True, block=(step == end_step))
        if step in stage_ends:
            sp = CheckpointPusher(cfg.hub_repo_id, private=cfg.hub_private, stage="rl", branch=f"stage-{stage_ends[step]}", tag_prefix=None, squash=cfg.hub_squash)
            if sp.enabled:
                stage_dir_ = os.path.join(cfg.out_dir, f"hub_stage_{stage_ends[step]}")
                save_model(policy, tokenizer, stage_dir_, stop_ids=stop.ids)
                sp.push(stage_dir_, step, {"role": f"end of the {stage_ends[step]} stage", "stage": stage_ends[step], "reward": rec.get("reward"), "kl": rec.get("kl"),
                                           "best_selection_value": best_select, "selected_by": cfg.select_by, "config": dataclasses.asdict(cfg)},
                        as_best=False, cleanup=True, block=False)
                stage_pushers.append(sp)
        if step in cfg.export_steps or (cfg.export_steps and step == end_step):
            save_model(policy, tokenizer, os.path.join(cfg.out_dir, f"snap_step{step}"), stop_ids=stop.ids,
                       dtype={"bfloat16": torch.bfloat16, "float32": None}[cfg.export_dtype])
            print(f"[gemma_mt] AB1 snapshot written: {cfg.out_dir}/snap_step{step} ({cfg.export_dtype})", flush=True)
        if (cfg.save_every and step % cfg.save_every == 0) or step == end_step:
            save_last()
            if lwc is not None:
                lwc.save_state(step)

    pusher.wait()
    if latest_pusher is not None:
        latest_pusher.wait()
    for sp in stage_pushers:
        sp.wait()
    if not cfg.keep_single_checkpoint:
        save_model(policy, tokenizer, os.path.join(cfg.out_dir, "final"), stop_ids=stop.ids)
    summary["cap"] = {"base": cfg.steps, "final": cap_now, "extensions": extensions}
    summary.update({"steps": step, "best_eval_reward": best_eval, "seconds": round(time.time() - t_start, 1),
                    "reward_range": [lo, hi]})
    if lwc is not None:
        summary["lang_weights"] = lwc.summary()
    wd = None
    if controls is not None:
        wd = controls.finish(step)
        if wd is not None:
            summary["winddown"] = wd
    # why the run ended (2026-10-28): plateau_stop (with the evidence) | winddown (operator) | max_steps
    summary["stop_reason"] = ("plateau_stop" if plateau is not None and plateau.triggered is not None and wd is not None and wd["completed"]
                              else "early_stop" if patience is not None and patience.triggered is not None and wd is not None and wd["completed"]
                              else "planned_winddown" if cfg.planned_winddown_last > 0 and wd is not None and wd["completed"] else "winddown" if wd is not None and wd["completed"]
                              else "max_steps" if step >= cfg.steps else "ended_early")
    if plateau is not None:
        summary["plateau_stop"] = {"triggered": plateau.triggered is not None, "evidence": plateau.triggered, "last_check": plateau.last_check}
    if patience is not None:
        summary["early_stop"] = {"triggered": patience.triggered is not None, "evidence": patience.triggered, "best": dict(patience.best), "best_step": dict(patience.best_step),
                                 "last_improve_step": patience.last_improve_step, "last_improve_by": patience.last_improve_by}
    if guard.enabled:
        summary["kl_guard"] = {"final_scale": guard.lr_scale, "cuts_total": guard.n_cuts, "recoveries_total": guard.n_recoveries, "events_this_process": guard_events_run,
                               "events_file": guard_events_path}
    if lr_mult_n:
        summary["lr_mult_effective"] = {"mean": lr_mult_sum / lr_mult_n, "min": lr_mult_min, "steps_this_process": lr_mult_n}
    if archive is not None:
        summary["archive"] = archive.close()
    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.close()
    return summary


# ----------------------------------------------------------------------------------------------- smoke / CLI
def run_smoke(workdir: Optional[str] = None, sft_steps: int = 150, benchmark: bool = False, vllm_fake: bool = False,
              overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """CPU end to end: a few SFT steps on synthetic data give a tiny model that knows the format; then GRPO with a
    fake scorer. Everything the real run does is exercised, including all in-loop assertions."""
    import tempfile

    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.smoke import ENGLISH, write_jsonl
    from gemma_mt.train_sft import SFTConfig, train as train_sft
    from gemma_mt.data.prepare_sft import PrepareConfig, prepare
    from gemma_mt.smoke import synthetic_rows

    work = workdir or tempfile.mkdtemp(prefix="gemma_mt_smoke_rl_")
    tok = FakeGemmaTokenizer()
    raw = write_jsonl(os.path.join(work, "raw.jsonl"), synthetic_rows(120, seed=2))
    prepare(PrepareConfig(input=raw, out_dir=os.path.join(work, "packed"), seq_len=256, val_fraction=0.1, seed=1),
            tokenizer=tok)
    sft = train_sft(SFTConfig(model_name="tiny-random", data_dir=os.path.join(work, "packed"),
                              out_dir=os.path.join(work, "sft"), autocast_dtype="none", micro_batch_size=4,
                              grad_accum_steps=1, epochs=1000, max_steps=sft_steps, learning_rate=3e-3,
                              warmup_steps=5, eval_every=0, save_every=0, log_every=50, early_stopping_patience=0,
                              device="cpu", seed=0), tokenizer=tok)
    langs = ["yor", "hau", "ibo"]
    rows = [{"english": ENGLISH[i % len(ENGLISH)], "iso": langs[i % 3], "language": "", "src_afr_tokens": 12,
             "source_dataset": "smoke"} for i in range(40)]
    ppath = write_jsonl(os.path.join(work, "rl_prompts.jsonl"), rows)
    cfg = RLConfig(model_name=os.path.join(work, "sft", "final"), prompts_path=ppath, out_dir=os.path.join(work, "rl"),
                   autocast_dtype="none", steps=4, prompts_per_step=3, group_size=4, micro_batch_seqs=5,
                   max_new_tokens=48, learning_rate=1e-4, warmup_steps=1, kl_coef=0.05, eval_every=2, eval_prompts=4,
                   save_every=2, log_every=1, check_sampler_logprobs=True, device="cpu", seed=0, resume=False,
                   reward={"max_comet_tokens": 510}, scorer={"type": "fake"})
    if benchmark:                       # dev sample with human references -> chrF++-based selection
        from gemma_mt.smoke import fake_translation

        dev = [{"english": ENGLISH[i % len(ENGLISH)], "reference": fake_translation(ENGLISH[i % len(ENGLISH)], langs[i % 3]),
                "iso": langs[i % 3], "benchmark": "dev", "id": str(i)} for i in range(18)]
        cfg.benchmark_eval_path = write_jsonl(os.path.join(work, "dev.jsonl"), dev)
        cfg.benchmark_eval_per_language = 4
        cfg.benchmark_max_new_tokens = 24
        cfg.select_by = "benchmark_chrf"
        cfg.select_min_finished_rate = 0.0
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    # the tiny SFT checkpoint is saved by `save_model`; reload it as a local path
    vs = None
    if vllm_fake:                       # the whole vLLM code path with a CPU stand-in engine (a private model copy fed through push_weights)
        from gemma_mt.rl.vllm_sampler import HFBackedEngine, VLLMSampler

        cfg.sampler, cfg.vllm_check_every = "vllm", 1
        vs = VLLMSampler(HFBackedEngine(cfg.model_name), seed=cfg.seed)
    summary = train_rl(cfg, tokenizer=tok, scorer=build_scorer(cfg.scorer) if cfg.scorer.get("ensemble_model_id") else FakeScorer(), vllm_sampler=vs)
    if vs is not None:
        summary["engine_pushes"], summary["engine_generate_calls"] = vs.engine.pushes, vs.engine.generate_calls
    summary["workdir"] = work
    summary["sft_best_val"] = sft["best_val_loss"]
    return summary


def apply_overrides(d: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """`key=value` overrides on top of the yaml. The dict-valued blocks `reward`, `scorer` and `archive_min_reward_by_mode` are MERGED key by key (D-29..D-32 follow-up): a shallow
    `reward='{"calibration_path": ...}'` used to REPLACE the whole block of rl.yaml and silently switched chunking, the length / lexical / diacritic
    penalties, the omission detector and the end-of-turn discipline back to their dataclass defaults (off). Every other key is replaced as before."""
    out = dict(d)
    for k, v in overrides.items():
        if k in ("reward", "scorer", "archive_min_reward_by_mode") and isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-vllm", action="store_true", help="--smoke through the vLLM sampler code path with a CPU stand-in engine")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    if a.smoke or a.smoke_vllm:
        print(json.dumps(run_smoke(vllm_fake=a.smoke_vllm), indent=2, default=str))
        return 0
    if not a.config:
        ap.error("--config is required (or --smoke)")
    d = apply_overrides(load_yaml(a.config), parse_overrides(a.overrides))
    train_rl(dataclass_from_dict(RLConfig, d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
