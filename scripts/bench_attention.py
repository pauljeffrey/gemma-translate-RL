# SPDX-License-Identifier: AGPL-3.0-or-later
"""Micro-benchmark of ONE attention call (fwd+bwd), Gemma-3-270m shapes: 4 q heads, 1 kv head, head_dim 256, bf16.

    python scripts/bench_attention.py [--packs 2] [--T 4096] [--window 512] [--samples 6]

Candidates: sdpa-explicit (efficient kernel + [B,1,T,T] bool mask, what the trainer uses by default), sdpa-flash-causal (no
mask; NOT equivalent, a speed ceiling), varlen (torch.nn.attention.varlen.varlen_attn), flex (flex_attention with a document +
causal (+ window) BlockMask and small tiles for sm_120)."""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F


def make_segments(packs, T, samples, seed=0):
    g = np.random.default_rng(seed)
    lens = []
    for _ in range(packs):
        cuts = np.sort(g.choice(np.arange(64, T - 64), samples - 1, replace=False))
        lens.append(np.diff(np.concatenate([[0], cuts, [T]])).tolist())
    return lens


def timeit(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", type=int, default=2)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--only", default="")
    ap.add_argument("--flex-opts", default="BLOCK_M=64,BLOCK_N=64,BLOCK_M1=32,BLOCK_N1=64,BLOCK_M2=64,BLOCK_N2=32,num_stages=1,num_warps=4")
    a = ap.parse_args()
    dev, dt = "cuda", torch.bfloat16
    B, T, H, Hk, D, W = a.packs, a.T, 4, 1, 256, a.window
    lens = make_segments(B, T, a.samples)
    seg = torch.stack([torch.repeat_interleave(torch.arange(len(l)), torch.tensor(l)) for l in lens]).to(dev)   # [B, T]
    flat_lens = [x for l in lens for x in l]
    cu = torch.tensor([0] + list(np.cumsum(flat_lens)), dtype=torch.int32, device=dev)
    maxq = max(flat_lens)
    g = torch.Generator(device=dev).manual_seed(0)
    mk = lambda *s: torch.randn(*s, device=dev, dtype=dt, generator=g, requires_grad=True)
    q, k, v = mk(B, H, T, D), mk(B, Hk, T, D), mk(B, Hk, T, D)
    scale = 256 ** -0.5
    res = {}

    def fwdbwd(fn, *xs):
        go = {}
        def run():
            out = fn()
            if "g" not in go:
                go["g"] = torch.randn_like(out)
            torch.autograd.backward(out, go["g"])
            for x in xs:
                x.grad = None
        return run

    idx = torch.arange(T, device=dev)
    for name, window in (("global", None), ("local", W)):
        causal = idx[None, :] <= idx[:, None]
        if window:
            causal = causal & ((idx[:, None] - idx[None, :]) < window)
        mask = ((seg[:, :, None] == seg[:, None, :]) & causal[None])[:, None]               # [B,1,T,T]
        kx, vx = k.expand(B, H, T, D), v.expand(B, H, T, D)

        if "sdpa" in a.only or not a.only:
            res[f"sdpa-explicit {name}"] = timeit(fwdbwd(lambda: F.scaled_dot_product_attention(q, kx, vx, attn_mask=mask, scale=scale), q, k, v))
        if "flashc" in a.only or not a.only:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            def f():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(q, kx, vx, is_causal=True, scale=scale)
            res[f"sdpa-flash-causal(no seg) {name}"] = timeit(fwdbwd(f, q, k, v))
        if "varlen" in a.only or not a.only:
            from torch.nn.attention.varlen import varlen_attn
            qf = q.transpose(1, 2).reshape(B * T, H, D).detach().requires_grad_()
            kf = k.transpose(1, 2).reshape(B * T, Hk, D).detach().requires_grad_()
            vf = v.transpose(1, 2).reshape(B * T, Hk, D).detach().requires_grad_()
            ws = (window - 1, 0) if window else (-1, 0)
            res[f"varlen {name}"] = timeit(fwdbwd(lambda: varlen_attn(qf, kf, vf, cu, cu, maxq, maxq, scale=scale, window_size=ws, enable_gqa=True), qf, kf, vf))
        if "flex" in a.only or not a.only:
            from torch.nn.attention.flex_attention import create_block_mask, flex_attention
            opts = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.flex_opts.split(",") if kv}
            def mod(b, h, qi, ki, window=window):
                m = (seg[b, qi] == seg[b, ki]) & (qi >= ki)
                if window:
                    m = m & (qi - ki < window)
                return m
            bm = create_block_mask(mod, B, None, T, T, device=dev, BLOCK_SIZE=128)
            cf = torch.compile(flex_attention, dynamic=False)
            try:
                f = lambda: cf(q, k, v, block_mask=bm, scale=scale, enable_gqa=True, kernel_options=opts)
                res[f"flex {name} {a.flex_opts}"] = timeit(fwdbwd(f, q, k, v))
            except Exception as e:
                res[f"flex {name}"] = f"FAILED {type(e).__name__}: {str(e)[:200]}"
    print(f"packs={B} T={T} window={W} samples/pack={a.samples} lens={lens}")
    for k_, v_ in res.items():
        print(f"  {k_:70s} {v_ if isinstance(v_, str) else f'{v_:8.2f} ms'}")


if __name__ == "__main__":
    main()
