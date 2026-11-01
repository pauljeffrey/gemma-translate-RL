#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""How self-distilled is each component? Mean per-token NLL of the START model (final_1433) on the target tokens of a random sample of every source of mix_v2.

    python scripts/mix2_nll.py --data /workspace/jobs/r11/mix_v2 --n 300 [--max-gib 6]

A target the model already finds likely (low NLL) gives small gradients (little to learn, little to forget); human references have a higher NLL. Reads train/source.npy
(0 science, 1 replay, 2 archive, 3 longdoc, 4 weak_extra). Plain HF forward, one sample at a time (bf16), GPU memory < 3 GB, `nice`.
"""
import argparse, json, os, sys, collections
import numpy as np

SOURCES = ("science", "replay", "archive", "longdoc", "anchor", "r10_commercial", "r14_teacher", "r10_share_alike", "r10_code_switch", "r10_nc")
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="/workspace/sft_build/runs/final_1433")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-len", type=int, default=8192)
ap.add_argument("--out", default="")
a = ap.parse_args()
import torch
from transformers import AutoModelForCausalLM

d = os.path.join(a.data, "train")
tok = np.memmap(os.path.join(d, "tokens.u32.bin"), dtype=np.uint32, mode="r")
off, ln = np.load(os.path.join(d, "sample_offset.npy")), np.load(os.path.join(d, "sample_len.npy"))
pl, tl = np.load(os.path.join(d, "prompt_len.npy")), np.load(os.path.join(d, "target_len.npy"))
src, iso = np.load(os.path.join(d, "source.npy")), np.load(os.path.join(d, "iso_id.npy"))
from gemma_mt.languages import ALL_ISOS

model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda().eval()
rng = np.random.default_rng(a.seed)
res = {}
for s in range(len(SOURCES)):
    idx = np.flatnonzero(src == s)
    if len(idx) == 0:
        continue
    pick = rng.choice(idx, size=min(a.n, len(idx)), replace=False)
    tot_nll, tot_tok, per_sample, by_lang = 0.0, 0, [], collections.defaultdict(lambda: [0.0, 0])
    for i in pick:
        ids = torch.tensor(np.asarray(tok[int(off[i]): int(off[i]) + int(ln[i])]).astype(np.int64))[None].cuda()
        p, t = int(pl[i]), int(tl[i])
        with torch.no_grad():
            h = model.model(input_ids=ids).last_hidden_state[0, p - 1: p + t - 1]
            nll = 0.0
            for c in range(0, t, 512):
                lg = model.lm_head(h[c:c + 512]).float()
                nll += torch.nn.functional.cross_entropy(lg, ids[0, p + c: p + min(c + 512, t)], reduction="sum").item()
        tot_nll += nll
        tot_tok += t
        per_sample.append(nll / t)
        by_lang[ALL_ISOS[int(iso[i])]][0] += nll
        by_lang[ALL_ISOS[int(iso[i])]][1] += t
    ps = np.array(per_sample)
    res[SOURCES[s]] = {"samples": len(pick), "target_tokens": tot_tok, "mean_nll_per_token": tot_nll / tot_tok, "median_sample_nll": float(np.median(ps)),
                       "p10_p90_sample_nll": [float(np.percentile(ps, 10)), float(np.percentile(ps, 90))]}
    print(SOURCES[s], res[SOURCES[s]], flush=True)
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
