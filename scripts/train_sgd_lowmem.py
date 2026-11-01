# SPDX-License-Identifier: AGPL-3.0-or-later
"""train_sft with plain SGD (no optimizer state) instead of Muon / AdamW, for equivalence runs on a GPU that is shared and has
only ~4 GB free (fp32 weights + grads = 2.2 GB, an AdamW state would add 2.2 GB). Everything else is the production training loop
(collators, accumulation, loss module, evaluation, logging); only the optimizer differs. NOT for real training.

    python scripts/train_sgd_lowmem.py --config configs/sft_unimax.yaml learning_rate=0.02 attn_implementation=varlen ..."""
import sys

import torch

import gemma_mt.train_sft as T


def _sgd(model, cfg):
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate, momentum=0.0)


T.build_optimizer = _sgd

if __name__ == "__main__":
    sys.exit(T.main())
