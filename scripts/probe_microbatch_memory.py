#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPU memory / time of ONE trainer micro-batch of the RL update at a given sequence length (D-45: is a 9.3k-token b2048 sequence safe?).

    python scripts/probe_microbatch_memory.py --model /workspace/sft_build/runs/rl_2b_stage_b64 --lengths 4900x5 9300x2 9300x1 --loss-chunk 2048

Each case `LxB` = B random sequences of L tokens (prompt = 2,150 tokens, the rest completion), forward + backward of `completion_logprobs` under bf16 autocast with fp32
weights, `sdpa` attention and no gradient checkpointing, exactly as `train_rl` does it, plus the reference forward. Prints peak allocated / reserved GiB and seconds.
It needs a GPU that is NOT busy with a trainer (the policy + reference + optimiser state of the real run take ~45 GB; this probe needs ~10 GB + the activations).
Never run it next to the live trainer on the rental: written for the day the card is free (or on another card).
"""
from __future__ import annotations

import argparse
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", nargs="+", default=["4900x5", "9300x2", "9300x1"], help="LxB cases: B sequences of L tokens")
    ap.add_argument("--prompt", type=int, default=2150)
    ap.add_argument("--loss-chunk", type=int, default=2048)
    ap.add_argument("--attn", default="sdpa")
    a = ap.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("no CUDA device: nothing to probe")
        return 1
    from gemma_mt.losses import completion_logprobs
    from gemma_mt.modeling import load_model
    from gemma_mt.rl.sampling import assemble_batch

    dev = torch.device("cuda")
    policy = load_model(a.model, attn_implementation=a.attn, dtype=torch.float32, device="cuda")
    policy.config.use_cache = False
    ref = load_model(a.model, attn_implementation=a.attn, dtype=torch.float32, device="cuda").eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    g = torch.Generator().manual_seed(0)
    for case in a.lengths:
        L, B = (int(x) for x in case.split("x"))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        prompts = [torch.randint(1000, 200000, (a.prompt,), generator=g).tolist() for _ in range(B)]
        comps = [torch.randint(1000, 200000, (L - a.prompt,), generator=g).tolist() for _ in range(B)]
        b = assemble_batch(prompts, comps, 0, dev)
        t = time.time()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                completion_logprobs(ref, b["input_ids"], b["attention_mask"], b["prompt_width"], b["completion_mask"], 1.0, a.loss_chunk)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp = completion_logprobs(policy, b["input_ids"], b["attention_mask"], b["prompt_width"], b["completion_mask"], 1.0, a.loss_chunk)
            (-lp.sum()).backward()
            torch.cuda.synchronize()
            print(f"{case}: peak allocated {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB, {time.time() - t:.2f}s", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            print(f"{case}: OUT OF MEMORY ({str(e)[:120]})", flush=True)
        policy.zero_grad(set_to_none=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
