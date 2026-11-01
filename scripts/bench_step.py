# SPDX-License-Identifier: AGPL-3.0-or-later
"""fwd+bwd time of one micro-batch of real packs for attention / loss implementations (the part of a training step that
the attention and loss changes touch; the optimizer is not included).

    python scripts/bench_step.py --data gemma/packed_fwd --micro 2 --ckpt 1 --cases sdpa:chunked varlen:chunked varlen:fused flex:chunked
    python scripts/bench_step.py ... --profile varlen:fused           # torch.profiler top CUDA ops of that case

A case is `<attn>:<loss>[:<chunk>]`, attn in sdpa | varlen | flex. Weights fp32, bf16 autocast, as in training."""
import argparse
import json
import time

import numpy as np
import torch

from gemma_mt import attention as attn
from gemma_mt.data.collate import PackedCollator
from gemma_mt.data.packed_dataset import PackedDataset
from gemma_mt.losses import sft_nll_sum
from gemma_mt.modeling import load_model

FLEX_OPTS = {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32, "num_stages": 1, "num_warps": 4}


def build(case_attn, items, dev):
    mode = {"sdpa": "explicit", "varlen": "flash", "flex": "position_ids"}[case_attn]
    impl = {"sdpa": "sdpa", "varlen": "varlen", "flex": "flex_attention"}[case_attn]
    col = PackedCollator("flatten" if mode == "flash" else "padded", pad_id=0)
    b = col(items)
    b = {k: (torch.from_numpy(v).to(dev) if isinstance(v, np.ndarray) else v) for k, v in b.items()}
    return impl, mode, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/sft_build/gemma/packed_fwd")
    ap.add_argument("--split", default="train")
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--ckpt", type=int, default=1)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--first-pack", type=int, default=1000)
    ap.add_argument("--cases", nargs="+", default=["sdpa:chunked", "varlen:chunked"])
    ap.add_argument("--profile", default="")
    ap.add_argument("--rounds", type=int, default=1, help="repeat the whole case list (interleaved), for a shared GPU")
    a = ap.parse_args()
    dev = "cuda"
    ds = PackedDataset(a.data, a.split, pad=True)
    model = load_model("google/gemma-3-270m-it", attn_implementation="sdpa", dtype=torch.float32, device=dev)
    if a.ckpt:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    results = {}
    for r in range(a.rounds):
        for case in a.cases:
            parts = case.split(":")
            ca, cl = parts[0], parts[1]
            chunk = int(parts[2]) if len(parts) > 2 else a.chunk
            items = [ds[a.first_pack + i] for i in range(a.micro)]
            impl, mode, b = build(ca, items, dev)
            if impl == "varlen":
                attn.register_varlen_attention()
            model.set_attn_implementation(impl)
            kw = attn.model_inputs(b, attn_implementation=impl, mask_mode=mode, sliding_window=model.config.sliding_window,
                                   kernel_options=FLEX_OPTS if ca == "flex" else None)
            tokens = int(np.asarray(items[0]["input_ids"]).size) * a.micro

            def step():
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    nll, n = sft_nll_sum(model, kw, b["labels"], chunk=chunk, loss_impl=cl)
                nll.backward()
                return float(nll.detach())

            torch.cuda.reset_peak_memory_stats()
            for _ in range(a.warmup):
                step()
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(a.iters):
                loss = step()
            torch.cuda.synchronize()
            ms = (time.time() - t) / a.iters * 1e3
            results.setdefault(case, []).append(ms)
            print(f"round {r} {case:24s} {ms:8.1f} ms/fwd+bwd  {tokens / ms * 1e3:9.0f} tok/s(padded)  peak {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB  loss_sum {loss:.1f}", flush=True)
            if a.profile == case and r == 0:
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                    step()
                    torch.cuda.synchronize()
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=70))
    print(json.dumps({k: [round(x, 1) for x in v] for k, v in results.items()}))


if __name__ == "__main__":
    main()
