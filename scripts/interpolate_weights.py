#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Weight-space interpolation between the model the science-SFT stage started from and the model it produced (WiSE-FT / model soup of two points).

    theta(a) = (1 - a) * theta_init + a * theta_sft          a = 0 -> the initial (RL-trained) model, a = 1 -> the SFT model

    python scripts/interpolate_weights.py --init /workspace/sft_build/runs/final_1433 --sft RUN/last/trainer_state.pt --alpha 0.5 --out DIR
    python scripts/interpolate_weights.py --init ... --sft ... --alpha 0.3 0.5 0.7 1.0 --out-root ROOT          # ROOT/alpha_0.30 ...

`--init` = an HF model directory; `--sft` = a trainer checkpoint (`trainer_state.pt`, key "model") or an HF model directory. The output is a complete HF
directory (weights in float32 by default, tokenizer, chat template, generation_config copied from --init), so `AutoModelForCausalLM.from_pretrained(DIR)`,
vLLM and `scripts/eval_afriscience.py --model DIR` read it as it is. Floating-point tensors are interpolated; any other tensor is taken from --init.
CPU only; ~5 s. Wise-FT reference: Wortsman et al. 2022, "Robust fine-tuning of zero-shot models".
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Dict

SIDE_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
              "added_tokens.json", "chat_template.jinja", "chat_template.json")


def load_sft_state(path: str) -> Dict:
    import torch

    if os.path.isdir(path):
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).state_dict()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["model"] if "model" in ck else ck


def interpolate_state(init_sd: Dict, sft_sd: Dict, alpha: float) -> Dict:
    import torch

    out, missing = {}, []
    for k, v in init_sd.items():
        w = sft_sd.get(k)
        if w is None:
            missing.append(k)
            out[k] = v.clone()
        elif torch.is_floating_point(v):
            if v.shape != w.shape:
                raise ValueError(f"{k}: shape {tuple(v.shape)} vs {tuple(w.shape)}")
            out[k] = ((1.0 - alpha) * v.double() + alpha * w.to(v.device).double()).to(v.dtype)
        else:
            out[k] = v.clone()
    if missing:
        # only a tied lm_head may be absent from one side
        bad = [k for k in missing if "lm_head" not in k]
        if bad:
            raise KeyError(f"{len(bad)} tensors of --init are missing in --sft, e.g. {bad[:3]}")
    return out


def interpolate_to_dir(init_dir: str, sft_path: str, alpha: float, out_dir: str, dtype: str = "float32") -> str:
    import torch
    from transformers import AutoModelForCausalLM

    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(init_dir, torch_dtype=torch.float32)
    sd = interpolate_state(model.state_dict(), load_sft_state(sft_path), alpha)
    model.load_state_dict(sd)
    if dt != torch.float32:
        model = model.to(dt)
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    for f in SIDE_FILES:                     # the init directory's tokenizer / template / generation config (stop ids) are authoritative
        if os.path.exists(os.path.join(init_dir, f)):
            shutil.copy2(os.path.join(init_dir, f), os.path.join(out_dir, f))
    from gemma_mt.format import write_greedy_generation_config      # D-67 (R12): greedy generation_config in every exported directory

    write_greedy_generation_config(out_dir)
    return out_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--alpha", type=float, nargs="+", required=True)
    ap.add_argument("--out", default=None, help="output directory (one alpha)")
    ap.add_argument("--out-root", default=None, help="several alphas: ROOT/alpha_0.30, ...")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args(argv)
    if len(a.alpha) == 1 and a.out:
        print(interpolate_to_dir(a.init, a.sft, a.alpha[0], a.out, a.dtype))
        return 0
    if not a.out_root:
        ap.error("several alphas need --out-root")
    for al in a.alpha:
        print(interpolate_to_dir(a.init, a.sft, al, os.path.join(a.out_root, f"alpha_{al:.2f}"), a.dtype))
    return 0


if __name__ == "__main__":
    sys.exit(main())
