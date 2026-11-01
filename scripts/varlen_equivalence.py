# SPDX-License-Identifier: AGPL-3.0-or-later
"""Numerical equivalence of the attention paths (explicit sdpa masks vs varlen vs flex_attention) and of the loss
implementations (chunked vs fused) on the SAME weights and the SAME packs.

    python scripts/varlen_equivalence.py tiny  [--head-dim 256]                    # random tiny Gemma3, window 8 < samples
    python scripts/varlen_equivalence.py real  --data gemma/packed_fwd --packs 0 1 # google/gemma-3-270m-it, real packs
    python scripts/varlen_equivalence.py loss  --data gemma/packed_fwd             # chunked vs fused loss, real model

Ground truth = the explicit-mask sdpa path in fp32 (no autocast). The point of the comparison: the NEW path's error against
that truth in bf16 autocast must not exceed the error of the OLD path's own bf16 against it (numerical noise)."""
import argparse
import json
import sys

import numpy as np
import torch

from gemma_mt import equiv
from gemma_mt.data.collate import PackedCollator


def to_dev(b, dev):
    return {k: (torch.from_numpy(v).to(dev) if isinstance(v, np.ndarray) else v) for k, v in b.items()}


def tiny_items(lens_rows, vocab=500, seed=0):
    items = []
    for r, lens in enumerate(lens_rows):
        g = np.random.default_rng(seed + r)
        n = sum(lens)
        ids = g.integers(4, vocab, n).astype(np.int64)
        lab = ids.copy()
        c = 0
        for L in lens:
            lab[c: c + 4] = -100
            c += L
        pos = np.concatenate([np.arange(L) for L in lens]).astype(np.int64)
        seg = np.concatenate([np.full(L, j + 1) for j, L in enumerate(lens)]).astype(np.int32)
        T = 4096 if sum(lens) > 200 else 80
        pad = T - n
        items.append({"input_ids": np.concatenate([ids, np.zeros(pad, np.int64)]), "labels": np.concatenate([lab, np.full(pad, -100, np.int64)]),
                      "position_ids": np.concatenate([pos, np.arange(pad)]), "segment_ids": np.concatenate([seg, np.zeros(pad, np.int32)]),
                      "cu_seqlens": np.concatenate([[0], np.cumsum(lens), [T]]).astype(np.int32),
                      "token_iso": np.zeros(T, np.int16), "length": n, "n_samples": len(lens)})
    return items


def table(name, rows):
    print(f"\n== {name}")
    keys = sorted({k for r in rows.values() for k in r})
    print("%-34s" % "candidate vs reference" + "".join("%18s" % k for k in keys))
    for lab, r in rows.items():
        print("%-34s" % lab + "".join("%18.3e" % r.get(k, float("nan")) for k in keys))
    sys.stdout.flush()


CHUNK = 512


def candidates(model, padded, flat, ref_fp32, with_flex, flex_opts, want_logits, backward=True, loss_impl="chunked"):
    out = {}
    cands = [("explicit bf16", "sdpa", "explicit", padded, torch.bfloat16),
             ("varlen bf16", "varlen", "flash", flat, torch.bfloat16),
             ("varlen fp32 (reference path)", "varlen", "flash", flat, None)]
    if with_flex:
        cands.append(("flex bf16", "flex_attention", "position_ids", padded, torch.bfloat16))
    runs = {}
    for lab, impl, mode, b, ac in cands:
        if impl == "flex_attention":
            runs[lab] = equiv.run_path_kw(model, b, impl=impl, mode=mode, autocast=ac, logits=want_logits, backward=backward,
                                          kernel_options=flex_opts, loss_impl=loss_impl, chunk=CHUNK)
        else:
            runs[lab] = equiv.run_path(model, b, impl=impl, mode=mode, autocast=ac, logits=want_logits, backward=backward, loss_impl=loss_impl, chunk=CHUNK)
    return runs


def report(runs, ref):
    rows = {f"{k} vs explicit fp32": equiv.compare(ref, v) for k, v in runs.items()}
    if "explicit bf16" in runs:
        for k in runs:
            if k != "explicit bf16":
                rows[f"{k} vs explicit bf16"] = equiv.compare(runs["explicit bf16"], runs[k])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tiny", "real", "loss"])
    ap.add_argument("--head-dim", type=int, default=16)
    ap.add_argument("--data", default="gemma/packed_fwd")
    ap.add_argument("--split", default="val")
    ap.add_argument("--packs", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--flex", action="store_true")
    ap.add_argument("--flex-opts", default='{"BLOCK_M":32,"BLOCK_N":32,"BLOCK_M1":32,"BLOCK_N1":32,"BLOCK_M2":32,"BLOCK_N2":32,"num_stages":1,"num_warps":4}')
    ap.add_argument("--json", default="")
    ap.add_argument("--init-from", default="", help="trainer_state.pt whose weights replace google/gemma-3-270m-it (a trained model)")
    ap.add_argument("--chunk", type=int, default=512, help="loss chunk (rows); lower it when the GPU is shared")
    a = ap.parse_args()
    global CHUNK
    CHUNK = a.chunk
    dev = "cuda"
    flex_opts = json.loads(a.flex_opts)
    result = {}
    if a.what == "tiny":
        from gemma_mt.modeling import build_tiny_model
        if a.head_dim == 16:
            kw, rows, window = {}, [[20, 31, 14], [40, 9], [25, 25, 12]], 8
        else:                           # the 270M attention geometry: 4 q heads, 1 kv head, head_dim 256, window smaller than samples
            kw = dict(hidden_size=128, num_attention_heads=4, num_key_value_heads=1, head_dim=a.head_dim, query_pre_attn_scalar=a.head_dim,
                      sliding_window=100, max_position_embeddings=4096)
            rows, window = [[900, 1700, 300, 1196], [4096 - 100, 100], [250, 1000, 2000, 846]], 100
        items = tiny_items(rows)
        padded = to_dev(PackedCollator("padded", pad_id=0)(items), dev)
        flat = to_dev(PackedCollator("flatten")(items), dev)
        model = build_tiny_model(seed=3, attn_implementation="sdpa", **kw).to(dev).train()
        ref = equiv.run_path(model, padded, impl="sdpa", mode="explicit", autocast=None, logits=True, chunk=CHUNK)
        runs = candidates(model, padded, flat, ref, a.flex, flex_opts, True)
        t = report(runs, ref)
        table(f"tiny model head_dim={model.config.head_dim} window={model.config.sliding_window} (loss sum {ref['nll']:.3f}, n={ref['n']}, gnorm {ref['gnorm']:.4g})", t)
        result = t
    elif a.what == "real":
        from gemma_mt.data.packed_dataset import PackedDataset
        from gemma_mt.modeling import load_model
        ds = PackedDataset(a.data, a.split, pad=True)
        items = [ds[i] for i in a.packs]
        padded = to_dev(PackedCollator("padded", pad_id=ds.pad_id)(items), dev)
        flat = to_dev(PackedCollator("flatten")(items), dev)
        print("packs", a.packs, "padded", tuple(padded["input_ids"].shape), "flat", tuple(flat["input_ids"].shape),
              "samples", [int(x) for x in np.diff(flat["cu_seqlens"].cpu().numpy())], "targets", padded["num_target_tokens"])
        model = load_model("google/gemma-3-270m-it", attn_implementation="sdpa", dtype=torch.float32, device=dev)
        if a.init_from:
            sd = torch.load(a.init_from, map_location="cpu", weights_only=False)["model"]
            print("loaded weights from", a.init_from, model.load_state_dict(sd, strict=False))
            del sd
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.train()
        print("config:", {k: getattr(model.config, k) for k in ("num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
                                                                 "sliding_window", "query_pre_attn_scalar", "attn_logit_softcapping",
                                                                 "final_logit_softcapping")})
        ref = equiv.run_path(model, padded, impl="sdpa", mode="explicit", autocast=None, chunk=CHUNK)
        runs = candidates(model, padded, flat, ref, a.flex, flex_opts, False)
        t = report(runs, ref)
        table(f"REAL gemma-3-270m-it, packs {a.packs} (loss sum {ref['nll']:.3f}, n={ref['n']}, mean {ref['nll']/ref['n']:.4f}, gnorm {ref['gnorm']:.4g})", t)
        for k, v in runs.items():
            print(f"  {k}: loss mean {v['nll']/v['n']:.5f}  gnorm {v['gnorm']:.4f}")
        result = t
    else:
        from gemma_mt.data.packed_dataset import PackedDataset
        from gemma_mt.modeling import load_model
        ds = PackedDataset(a.data, a.split, pad=True)
        items = [ds[i] for i in a.packs]
        flat = to_dev(PackedCollator("flatten")(items), dev)
        model = load_model("google/gemma-3-270m-it", attn_implementation="varlen", dtype=torch.float32, device=dev)
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.train()
        rows = {}
        for ac, lab in ((torch.bfloat16, "bf16 autocast"), (None, "fp32")):
            ch = equiv.run_path(model, flat, impl="varlen", mode="flash", autocast=ac, loss_impl="chunked", chunk=CHUNK)
            fu = equiv.run_path(model, flat, impl="varlen", mode="flash", autocast=ac, loss_impl="fused", chunk=CHUNK)
            rows[f"fused vs chunked ({lab})"] = equiv.compare(ch, fu)
            print(f"  {lab}: chunked loss {ch['nll']:.4f} fused {fu['nll']:.4f} gnorm {ch['gnorm']:.5f} vs {fu['gnorm']:.5f}")
        # noise floor: chunked bf16 vs chunked fp32
        ref = equiv.run_path(model, flat, impl="varlen", mode="flash", autocast=None, loss_impl="chunked", chunk=CHUNK)
        ch = equiv.run_path(model, flat, impl="varlen", mode="flash", autocast=torch.bfloat16, loss_impl="chunked", chunk=CHUNK)
        fu = equiv.run_path(model, flat, impl="varlen", mode="flash", autocast=torch.bfloat16, loss_impl="fused", chunk=CHUNK)
        rows["chunked bf16 vs fp32 (noise floor)"] = equiv.compare(ref, ch)
        rows["fused bf16 vs fp32"] = equiv.compare(ref, fu)
        table("loss implementations, real model, 2 real packs flattened", rows)
        result = rows
    if a.json:
        json.dump(result, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
