#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Total RL reward of a system's document translations, with the real reward code (D-44 addendum). Thin wrapper: see gemma_mt/eval/longdoc_reward.py (`--help`).

    python scripts/longdoc_score_reward.py --preds preds.jsonl --docs source_set/core/docs.jsonl \\
        --config /workspace/sft_build/gemma_mt_2c/configs/rl_rtxpro_phase2c.yaml --calibration /workspace/sft_build/gemma_mt_2c/calibration/shaping_flores_dev_v6.json \\
        --scorer worker --out out/reward_scores.jsonl          # CPU: CUDA_VISIBLE_DEVICES="" ; GPU: leave it visible
    python scripts/longdoc_score_reward.py ... --scorer fake --tokenizer fake      # plumbing check without any model
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma_mt.eval.longdoc_reward import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
