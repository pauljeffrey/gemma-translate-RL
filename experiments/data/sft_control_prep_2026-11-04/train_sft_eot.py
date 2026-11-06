#!/usr/bin/env python
"""Launcher for the end-of-turn-emphasis arm of the continued-SFT control (E-03). It does NOT edit gemma_mt: it replaces `gemma_mt.losses.sft_nll_sum`
(imported inside `make_loss_module` at call time) by a version that multiplies the NLL of every `<end_of_turn>` label by EOT_LOSS_WEIGHT. The validation loss
(`evaluate` uses `sft_selected_logp`) stays unweighted, so validation numbers of the two arms are comparable. Normalisation is unchanged (divided by the number of
labelled tokens), so with weight 1.0 the run is bit-identical to `python -m gemma_mt.train_sft`.

    EOT_LOSS_WEIGHT=5 EOT_TOKEN_IDS=106 python train_sft_eot.py --config control_eot.yaml [key=value ...]
"""
import os, sys

import torch
import gemma_mt.losses as L

W = float(os.environ.get("EOT_LOSS_WEIGHT", "5.0"))
IDS = [int(x) for x in os.environ.get("EOT_TOKEN_IDS", "106").split(",") if x]


def sft_nll_sum_eot(model, model_kwargs, labels, chunk=512, loss_impl="chunked"):
    logp, sel = L.sft_selected_logp(model, model_kwargs, labels, chunk, loss_impl)
    n = int(sel.sum().item())
    if n == 0:
        return logp.sum(), 0
    tgt = labels[:, 1:][sel]
    w = torch.ones_like(logp)
    for i in IDS:
        w = torch.where(tgt == i, torch.full_like(w, W), w)
    return -(logp * w).sum(), n


L.sft_nll_sum = sft_nll_sum_eot

if __name__ == "__main__":
    from gemma_mt.train_sft import main
    print(f"[control] end-of-turn loss weight {W} on token ids {IDS}", flush=True)
    raise SystemExit(main(sys.argv[1:]))
