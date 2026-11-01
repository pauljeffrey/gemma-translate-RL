#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Stage 2: GRPO with the shaped AfriCOMET-QE reward. Single GPU only.
#   SFT=runs/sft/best PROMPTS=/workspace/rl_prompts OUT=runs/rl [CAL=calibration/other.json] \
#     bash scripts/vast_train_rl.sh [key=value ...]
# Preconditions: reward.calibration_path of configs/rl.yaml (calibration/shaping_flores_dev_v4.json, 42 languages) or $CAL exists; the SFT checkpoint's generation_config lists
# <end_of_turn>. Do a dry run first:  python -m gemma_mt.rl.train_rl --smoke   (CPU, seconds).
# On an 11 GB GPU lower micro_batch_seqs (8) and scorer.batch_size via:  scorer='{"type":"comet","batch_size":16}'
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
# vLLM sampler (configs/rl.yaml `sampler: vllm`, DECISIONS.md D-28): the engine runs in a child process of VLLM_PYTHON (the interpreter of the
# environment that has vllm; null/unset = this one). Its compile caches must live on a real disk (/dev/shm is mounted noexec on the rentals).
# Fall back to the old sampler with:  bash scripts/vast_train_rl.sh sampler='"hf"'
VLLM_CACHE="${VLLM_CACHE:-$HOME/.cache/gemma_mt_vllm}"
export VLLM_CACHE_ROOT="$VLLM_CACHE/vllm" TRITON_CACHE_DIR="$VLLM_CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$VLLM_CACHE/inductor" \
       XDG_CACHE_HOME="$VLLM_CACHE/xdg" VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
if [ -n "${VLLM_PYTHON:-}" ]; then set -- "vllm_python=\"$VLLM_PYTHON\"" "$@"; fi
SFT="${SFT:-runs/sft/best}"
OUT="${OUT:-runs/rl}"
PROMPTS="${PROMPTS:-data/rl_prompts}"
# The reward block (chunking, penalties, omission detector, EOS discipline, calibration file) comes from configs/rl.yaml. CAL / PRESET only override those two
# keys: train_rl merges a `reward='{...}'` override into the yaml block (a shallow override used to wipe chunking and every penalty, D-32 follow-up).
REWARD_OVR=""
[ -n "${CAL:-}" ] && REWARD_OVR="\"calibration_path\":\"$CAL\""
[ -n "${PRESET:-}" ] && REWARD_OVR="${REWARD_OVR:+$REWARD_OVR,}\"preset\":\"$PRESET\""
if [ -n "$REWARD_OVR" ]; then set -- "reward={$REWARD_OVR}" "$@"; fi
python -m gemma_mt.rl.train_rl --config configs/rl.yaml \
  model_name="\"$SFT\"" out_dir="\"$OUT\"" prompts_path="\"$PROMPTS\"" "$@"
