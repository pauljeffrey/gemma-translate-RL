#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Depth-upcycle a Gemma-3 text checkpoint: insert new decoder layers that start as an EXACT identity (arm A2-up of the clean rerun).

    python scripts/upcycle_layers.py --model google/gemma-3-270m-it --out ckpt/gemma3-270m-up6 [--n-new 6] [--dtype fp32]
    python scripts/upcycle_check.py  --base google/gemma-3-270m-it --up ckpt/gemma3-270m-up6          # the proof (logits, loading, 20 steps)

What it does (all on the safetensors tensors, no model object):
  * N new layers are inserted at evenly spaced depths: after original layer floor((k + 0.5) * L / N) for k = 0..N-1.
    Gemma-3-270m: L = 18, N = 6 -> after original layers 1, 4, 7, 10, 13, 16 (one new layer in the middle of every group of 3).
  * each new layer is a COPY of the original layer it follows (all of its tensors), except that
        self_attn.o_proj.weight  and  mlp.down_proj.weight  are set to ZERO.
    Gemma-3 applies post_attention_layernorm / post_feedforward_layernorm to the sub-layer OUTPUT before the residual add, and an RMSNorm of an
    all-zero vector is exactly zero, so both residual branches add exactly 0.0: the upcycled model computes the same function as the base.
  * layer_types: every ORIGINAL layer keeps its type (otherwise the function would change); EVERY NEW layer is a LOCAL (sliding-window) layer
    (owner decision 2026-10-29, `--new-layer-type sliding_attention`, the default; `copy` = the type of the layer it copies, `full_attention` = global).
    6 new layers (the "up6" arm): after original layers 1, 4, 7, 10, 13, 16 -> 24 layers = 7 sliding + 1 global, three times: the global layers move from
    indices 5, 11, 17 to 7, 15, 23 and are still the original global layers; `sliding_window_pattern` = 8.
    12 new layers ("up12"): the evenly spaced formula would land after global layers, so the new layers are spread evenly INSIDE the three runs of five sliding
    layers (4 per run, after original layers 0, 1, 3, 4 | 6, 7, 9, 10 | 12, 13, 15, 16) -> 30 layers = 9 sliding + 1 global, three times, pattern 10.
    `sliding_window_pattern` (global iff (i + 1) % P == 0) then describes the new stack, for transformers (explicit layer_types) AND for readers that derive the
    pattern from that integer (older vLLM); the script writes both and refuses a layout that is not periodic unless --allow-aperiodic is given.
  * the vocabulary, embeddings (tied), final norm, rope, sliding window and every other config value are untouched.

Gradient note (read before training): the zeroed o_proj / down_proj still receive gradient (it flows through the post-norm); the other tensors of a new
layer receive none until o_proj / down_proj are non-zero. Because RMSNorm rescales its input (eps = 1e-6), the FIRST update of a zeroed projection is
amplified (up to 1 / sqrt(eps) = 1000x) and the new branches switch on within a few steps instead of growing gently. `--post-norm-gain g` (default: off,
i.e. a pure copy) multiplies the output of the new layers' two post-norms by g (stored as weight = g - 1, Gemma's norm is x * (1 + w)) - an optional
damping that keeps the model exactly identical at init for g > 0; DECISION 2026-10-30: the A2-up6 and A2-up12 arms are built with --post-norm-gain 0.01 (recorded as `post_norm_gain` in upcycle.json); the flag stays optional in the tool.
Measured (CPU, toy data, AdamW 5e-5, 40-step warm-up, 2026-10-30): the pure-copy up6 / up12 models are identical to the base at step 0 but their loss jumps UP in steps 5-10
(8.7 -> 18.7 for up6, -> 12.9 for up12; the base falls smoothly) and the step-1 gradient norm is 83k (up6) / 268k (up12) against 126 for the base; see the report of 2026-10-30.
"""
import argparse, glob, hashlib, json, math, os, shutil, sys

LAYER_KEY = "model.layers."


def plan_positions(n_layers, n_new, layer_types=None):
    """Original-layer indices AFTER which a new layer is inserted: floor((k + 0.5) * L / N). When `layer_types` is given and that formula would insert after a
    global layer, the new layers are instead spread evenly inside the runs of consecutive non-global layers (quota proportional to the run length, largest remainder;
    inside a run of length m with quota q: run_start + floor((j + 0.5) * m / q)), so a new (sliding) layer never separates a global layer from its successor."""
    if not 0 < n_new <= n_layers:
        raise ValueError(f"n_new must be in 1..{n_layers}")
    pos = [int((k + 0.5) * n_layers / n_new) for k in range(n_new)]
    assert len(set(pos)) == n_new and all(0 <= p < n_layers for p in pos), pos
    if layer_types is None or not any(layer_types[p] == "full_attention" for p in pos):
        return pos
    runs, start = [], 0
    for i, t in enumerate(layer_types):
        if t == "full_attention":
            if i > start:
                runs.append((start, i - start))
            start = i + 1
    if n_layers > start:
        runs.append((start, n_layers - start))
    total = sum(m for _, m in runs)
    raw = [n_new * m / total for _, m in runs]
    quota = [int(x) for x in raw]
    for i in sorted(range(len(runs)), key=lambda i: -(raw[i] - quota[i]))[: n_new - sum(quota)]:
        quota[i] += 1
    out = []
    for (st, m), q in zip(runs, quota):
        if q > m:
            raise ValueError(f"cannot place {q} new layers in a run of {m} layers")
        out += [st + int((j + 0.5) * m / q) for j in range(q)]
    assert len(set(out)) == n_new
    return sorted(out)


def layer_map(n_layers, after):
    """-> list of (source original index, is_new) in the order of the new stack."""
    after = sorted(after)
    out = []
    for i in range(n_layers):
        out.append((i, False))
        if i in after:
            out.append((i, True))
    return out


def sliding_pattern(layer_types):
    """P with 'global iff (i + 1) % P == 0' for every layer, or None when the stack is not periodic in that sense."""
    g = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    if not g:
        return None
    p = g[0] + 1
    ok = all((t == "full_attention") == ((i + 1) % p == 0) for i, t in enumerate(layer_types))
    return p if ok else None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve_dir(model):
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download      # needs a read token for the gated Gemma repo (HF_TOKEN in the environment; never printed)
    return snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt"])


def load_tensors(d):
    from safetensors.torch import load_file
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    if not files:
        raise SystemExit(f"no *.safetensors in {d}")
    sd = {}
    for f in files:
        sd.update(load_file(f))
    return sd, files


def upcycle(model_dir, out_dir, n_new=6, positions=None, dtype="fp32", post_norm_gain=None, allow_aperiodic=False, source_name=None,
            new_layer_type="sliding_attention"):
    import torch
    from safetensors.torch import save_file
    cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    L = int(cfg["num_hidden_layers"])
    types = list(cfg.get("layer_types") or [])
    if len(types) != L:
        raise SystemExit("config.json has no layer_types of length num_hidden_layers (expected a transformers >= 4.53 Gemma-3 text config)")
    after = sorted(positions) if positions else plan_positions(L, n_new, types)
    if new_layer_type not in ("sliding_attention", "full_attention", "copy"):
        raise SystemExit("new_layer_type must be sliding_attention | full_attention | copy")
    if any(not 0 <= p < L for p in after) or len(set(after)) != len(after):
        raise SystemExit(f"bad positions {after}")
    sd, files = load_tensors(model_dir)
    if any(k.startswith("language_model.") or k.startswith("vision") for k in sd):
        raise SystemExit("multimodal checkpoint: this tool takes the text-only Gemma3ForCausalLM layout (model.layers.N...)")
    lm = layer_map(L, after)
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    out, new_types, rows = {}, [], []
    for k, v in sd.items():
        if not k.startswith(LAYER_KEY):
            out[k] = v.to(torch.float32).to(dt).contiguous()
    for new_idx, (src, is_new) in enumerate(lm):
        new_types.append(types[src] if (not is_new or new_layer_type == "copy") else new_layer_type)
        prefix_src = f"{LAYER_KEY}{src}."
        for k, v in sd.items():
            if not k.startswith(prefix_src):
                continue
            name = k[len(prefix_src):]
            t = v.to(torch.float32).clone()
            if is_new:
                if name in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
                    t = torch.zeros_like(t)
                elif post_norm_gain is not None and name in ("post_attention_layernorm.weight", "post_feedforward_layernorm.weight"):
                    t = torch.full_like(t, float(post_norm_gain) - 1.0)
            out[f"{LAYER_KEY}{new_idx}.{name}"] = t.to(dt).contiguous()
        rows.append({"new_index": new_idx, "source_original_index": src, "new": bool(is_new), "layer_type": new_types[-1]})
    pat = sliding_pattern(new_types)
    if pat is None and not allow_aperiodic:
        raise SystemExit(f"the new layer types are not periodic ({new_types}); readers that derive the pattern from an integer would mis-assign "
                         "global layers. Choose other --positions or pass --allow-aperiodic (then rely on layer_types only).")
    cfg2 = dict(cfg)
    cfg2["num_hidden_layers"] = len(lm)
    cfg2["layer_types"] = new_types
    if pat is not None:
        cfg2["_sliding_window_pattern"] = pat
        cfg2["sliding_window_pattern"] = pat
    cfg2["dtype"] = "float32" if dtype == "fp32" else "bfloat16"
    cfg2.pop("torch_dtype", None)
    os.makedirs(out_dir, exist_ok=True)
    save_file(out, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    json.dump(cfg2, open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8"), indent=2)
    for fn in ("generation_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
               "added_tokens.json", "chat_template.jinja"):
        src = os.path.join(model_dir, fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out_dir, fn))
    n_base = sum(v.numel() for v in sd.values())
    n_up = sum(v.numel() for v in out.values())
    info = {"rule": "new layer = copy of the original layer it follows, self_attn.o_proj.weight and mlp.down_proj.weight = 0 (exact identity at init; Gemma's "
                    "post-attention / post-feedforward RMSNorm of a zero vector is zero)",
            "source": source_name or model_dir, "source_files": {os.path.basename(f): sha256_file(f) for f in files},
            "n_original_layers": L, "n_new_layers": len(after), "insert_after_original_layers": after, "layers": rows,
            "layer_types": new_types, "new_layer_type": new_layer_type, "sliding_window_pattern": pat, "post_norm_gain": post_norm_gain, "dtype": dtype,
            "parameters_base": int(n_base), "parameters_upcycled": int(n_up), "parameters_added": int(n_up - n_base),
            "parameters_per_layer": int((n_up - n_base) // max(1, len(after))),
            "tied_embeddings": not any(k.endswith("lm_head.weight") for k in out),
            "output_sha256": sha256_file(os.path.join(out_dir, "model.safetensors"))}
    json.dump(info, open(os.path.join(out_dir, "upcycle.json"), "w", encoding="utf-8"), indent=1)
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local dir (config.json + model.safetensors) or a Hub id (needs HF_TOKEN for gated repos)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-new", type=int, default=6)
    ap.add_argument("--positions", help="comma list of ORIGINAL layer indices after which a new layer is inserted (default: evenly spaced)")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--post-norm-gain", type=float, default=None, help="optional damping, see the module docstring (default off = pure copy)")
    ap.add_argument("--allow-aperiodic", action="store_true")
    ap.add_argument("--new-layer-type", choices=["sliding_attention", "full_attention", "copy"], default="sliding_attention",
                    help="attention type of the inserted layers (default: local / sliding window; 'copy' = the type of the layer copied)")
    a = ap.parse_args(argv)
    d = resolve_dir(a.model)
    pos = [int(x) for x in a.positions.split(",")] if a.positions else None
    info = upcycle(d, a.out, a.n_new, pos, a.dtype, a.post_norm_gain, a.allow_aperiodic, source_name=a.model, new_layer_type=a.new_layer_type)
    print(json.dumps({k: info[k] for k in ("source", "n_original_layers", "n_new_layers", "insert_after_original_layers", "sliding_window_pattern",
                                            "parameters_base", "parameters_upcycled", "parameters_added", "parameters_per_layer", "output_sha256")}, indent=1))
    g = [i for i, t in enumerate(info["layer_types"]) if t == "full_attention"]
    print("layer_types: global layers at", g, "of", len(info["layer_types"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
