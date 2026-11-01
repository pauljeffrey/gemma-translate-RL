#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Proof that an upcycled checkpoint (scripts/upcycle_layers.py) is the base model at init, loads everywhere, and trains.  CPU only.

    python scripts/upcycle_check.py --base <base dir> --up <upcycled dir> [--seq 700] [--train-steps 20] [--optimizer muon|adamw|both] [--tol 1e-5]

Checks (any failure exits non-zero):
  1. tensors: every original layer of the base reappears unchanged at its new index; every new layer equals the layer it follows except o_proj / down_proj == 0;
     everything outside the layer stack is identical; config layer_types / num_hidden_layers / sliding_window_pattern agree with upcycle.json
     (global iff (i + 1) % pattern == 0, the rule older vLLM versions use; vLLM itself is NOT run here - config check only).
  2. HF transformers (AutoModelForCausalLM, fp32, sdpa and eager): full logits of base and upcycled on one random sequence longer than the sliding window.
  3. the repo's own path (gemma_mt.modeling.load_model + gemma_mt.attention.model_inputs, a PACKED two-sample batch with block-diagonal sliding / global masks +
     gemma_mt.losses.final_hidden): last hidden state and logits of selected positions, base vs upcycled.
  4. training: `--train-steps` optimizer steps of the trainer's own optimizer (gemma_mt.train_sft.build_optimizer, all parameters trainable) on random packed
     batches; loss, gradient norm and weights must stay finite; the step-1 loss must equal the base model's.
"""
import argparse, glob, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def _fail(msg):
    print("FAIL:", msg, flush=True)
    raise SystemExit(1)


def check_tensors(base, up, info):
    from safetensors.torch import load_file
    import torch
    b = {}
    for f in sorted(glob.glob(os.path.join(base, "*.safetensors"))):
        b.update(load_file(f))
    u = {}
    for f in sorted(glob.glob(os.path.join(up, "*.safetensors"))):
        u.update(load_file(f))
    cfg_b, cfg_u = json.load(open(os.path.join(base, "config.json"))), json.load(open(os.path.join(up, "config.json")))
    if cfg_u["num_hidden_layers"] != len(info["layers"]) or len(cfg_u["layer_types"]) != cfg_u["num_hidden_layers"]:
        _fail("config.json layer count disagrees with upcycle.json")
    for k, v in cfg_b.items():
        if k not in ("num_hidden_layers", "layer_types", "_sliding_window_pattern", "sliding_window_pattern", "dtype", "torch_dtype") and cfg_u.get(k) != v:
            _fail(f"config key {k} changed: {v} -> {cfg_u.get(k)}")
    pat = info["sliding_window_pattern"]
    if pat is not None:
        for i, t in enumerate(cfg_u["layer_types"]):
            if (t == "full_attention") != ((i + 1) % pat == 0):
                _fail(f"layer {i}: type {t} contradicts sliding_window_pattern={pat}")
        if cfg_u.get("_sliding_window_pattern") != pat or cfg_u.get("sliding_window_pattern") != pat:
            _fail("sliding_window_pattern missing in config.json")
    seen_new = 0
    for row in info["layers"]:
        n, s = row["new_index"], row["source_original_index"]
        want = cfg_b["layer_types"][s] if (not row["new"] or info.get("new_layer_type", "copy") == "copy") else info["new_layer_type"]
        if cfg_u["layer_types"][n] != want:
            _fail(f"layer {n}: type {cfg_u['layer_types'][n]} but expected {want}")
        if row["new"] and info.get("new_layer_type") == "sliding_attention" and cfg_u["layer_types"][n] != "sliding_attention":
            _fail(f"new layer {n} is not a local (sliding-window) layer")
        for k, v in b.items():
            if not k.startswith(f"model.layers.{s}."):
                continue
            name = k[len(f"model.layers.{s}."):]
            w = u[f"model.layers.{n}.{name}"]
            if row["new"] and name in ("self_attn.o_proj.weight", "mlp.down_proj.weight"):
                if float(w.abs().max()) != 0.0:
                    _fail(f"new layer {n}: {name} is not zero")
                seen_new += 1
            elif row["new"] and info.get("post_norm_gain") is not None and name in ("post_attention_layernorm.weight", "post_feedforward_layernorm.weight"):
                pass
            elif not torch.equal(v.float(), w.float()):
                _fail(f"layer {n} ({'new' if row['new'] else 'original'} copy of {s}): {name} differs from the base")
    if seen_new != 2 * info["n_new_layers"]:
        _fail("zeroed projections not found for every new layer")
    for k, v in b.items():
        if not k.startswith("model.layers.") and not torch.equal(v.float(), u[k].float()):
            _fail(f"{k} differs")
    n_up = sum(v.numel() for v in u.values())
    if n_up != info["parameters_upcycled"]:
        _fail("parameter count differs from upcycle.json")
    print(f"[1] tensors OK: {sum(v.numel() for v in b.values()):,} -> {n_up:,} parameters ({info['parameters_added']:,} added, {info['parameters_per_layer']:,} per layer); "
          f"global layers at {[i for i, t in enumerate(cfg_u['layer_types']) if t == 'full_attention']}", flush=True)


def expected_tensor_shapes(cfg):
    """Names and shapes of a text-only Gemma3ForCausalLM checkpoint (tied embeddings), from config.json alone."""
    H, I, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    out = {"model.embed_tokens.weight": (V, H), "model.norm.weight": (H,)}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        out.update({p + "self_attn.q_proj.weight": (nh * hd, H), p + "self_attn.k_proj.weight": (nkv * hd, H), p + "self_attn.v_proj.weight": (nkv * hd, H),
                    p + "self_attn.o_proj.weight": (H, nh * hd), p + "self_attn.q_norm.weight": (hd,), p + "self_attn.k_norm.weight": (hd,),
                    p + "mlp.gate_proj.weight": (I, H), p + "mlp.up_proj.weight": (I, H), p + "mlp.down_proj.weight": (H, I),
                    p + "input_layernorm.weight": (H,), p + "post_attention_layernorm.weight": (H,),
                    p + "pre_feedforward_layernorm.weight": (H,), p + "post_feedforward_layernorm.weight": (H,)})
    return out


def check_vllm_config(d, verbose=True):
    """CONFIG-ONLY check (no vLLM import, header-only read of the safetensors) of what vLLM's Gemma3 text loader relies on: architecture / model_type, a layer_types list of
    num_hidden_layers entries that agrees with the integer sliding_window_pattern some vLLM versions derive the local / global split from, the rope parameters, tied embeddings,
    and EXACTLY the tensor names and shapes of a Gemma3ForCausalLM with that many layers. Returns the parameter count."""
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(d, "config.json")))
    if cfg.get("architectures") != ["Gemma3ForCausalLM"] or cfg.get("model_type") != "gemma3_text":
        _fail(f"{d}: architectures {cfg.get('architectures')} / model_type {cfg.get('model_type')} (vLLM text path expects Gemma3ForCausalLM / gemma3_text)")
    lt, n = cfg.get("layer_types"), cfg["num_hidden_layers"]
    if not lt or len(lt) != n or any(t not in ("sliding_attention", "full_attention") for t in lt):
        _fail(f"{d}: layer_types missing or not num_hidden_layers long")
    pat = cfg.get("sliding_window_pattern", cfg.get("_sliding_window_pattern"))
    if pat is not None and any((t == "full_attention") != ((i + 1) % pat == 0) for i, t in enumerate(lt)):
        _fail(f"{d}: layer_types contradicts sliding_window_pattern={pat} (a vLLM version that derives the split from the integer would mis-assign layers)")
    if not cfg.get("sliding_window"):
        _fail(f"{d}: no sliding_window")
    if not (cfg.get("rope_parameters") or (cfg.get("rope_theta") and cfg.get("rope_local_base_freq"))):
        _fail(f"{d}: no rope parameters")
    if cfg.get("tie_word_embeddings") is False:
        _fail(f"{d}: untied embeddings are not what the Gemma3 loader expects")
    exp, got, total = expected_tensor_shapes(cfg), {}, 0
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                got[k] = tuple(sf.get_slice(k).get_shape())
    if set(got) != set(exp):
        _fail(f"{d}: tensor names differ from a {n}-layer Gemma3ForCausalLM: missing {sorted(set(exp) - set(got))[:3]}, extra {sorted(set(got) - set(exp))[:3]}")
    bad = [k for k in exp if exp[k] != got[k]]
    if bad:
        _fail(f"{d}: shape mismatch {bad[:3]}: {exp[bad[0]]} vs {got[bad[0]]}")
    for v in got.values():
        c = 1
        for x in v:
            c *= x
        total += c
    if verbose:
        print(f"[5] vLLM config check (config + safetensors headers only; vLLM itself not run) {d}: Gemma3ForCausalLM, {n} layers, global layers at "
              f"{[i for i, t in enumerate(lt) if t == 'full_attention']}, pattern {pat}, {len(got)} tensors with the expected names and shapes, {total:,} parameters", flush=True)
    return total


def check_hf(base, up, seq, tol):
    import torch
    from transformers import AutoModelForCausalLM
    ids = torch.randint(4, 200000, (1, seq), generator=torch.Generator().manual_seed(0))
    worst = 0.0
    for impl in ("sdpa", "eager"):
        outs = []
        for d in (base, up):
            m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32, attn_implementation=impl).eval()
            with torch.no_grad():
                outs.append(m(input_ids=ids, use_cache=False).logits)
            del m
        diff = float((outs[0] - outs[1]).abs().max())
        worst = max(worst, diff)
        print(f"[2] HF transformers ({impl}): max |logit(base) - logit(upcycled)| over {seq} positions x vocab = {diff:.3e} (logit scale {float(outs[0].abs().max()):.1f})", flush=True)
        del outs
    if worst > tol:
        _fail(f"HF logits differ by {worst} > {tol}")
    return worst


def check_repo(base, up, seq, tol):
    import torch
    from gemma_mt import attention as attn
    from gemma_mt.losses import final_hidden
    from gemma_mt.modeling import load_model, sliding_window_of
    n1 = seq // 2
    ids = torch.randint(4, 200000, (1, seq), generator=torch.Generator().manual_seed(1))
    seg = torch.cat([torch.full((n1,), 1), torch.full((seq - n1,), 2)])[None].int()
    pos = torch.cat([torch.arange(n1), torch.arange(seq - n1)])[None].long()
    batch = {"input_ids": ids, "position_ids": pos, "segment_ids": seg}
    sel = torch.tensor([0, 5, n1 - 1, n1, n1 + 7, seq - 1])
    worst = 0.0
    for impl in ("sdpa", "eager"):
        res = []
        for d in (base, up):
            m = load_model(d, attn_implementation=impl).eval()
            kw = attn.model_inputs(batch, attn_implementation=impl, mask_mode="explicit", sliding_window=sliding_window_of(m))
            with torch.no_grad():
                h = final_hidden(m, **kw)
                lg = m.lm_head(h[:, sel])
            res.append((h, lg))
            del m
        dh = float((res[0][0] - res[1][0]).abs().max())
        dl = float((res[0][1] - res[1][1]).abs().max())
        worst = max(worst, dh, dl)
        print(f"[3] repo path ({impl}, packed 2 samples, block-diagonal sliding/global masks): max |hidden diff| = {dh:.3e}, max |logit diff| = {dl:.3e}", flush=True)
    if worst > tol:
        _fail(f"repo-path outputs differ by {worst} > {tol}")
    return worst


def check_training(base, up, steps, optimizer, seq=192):
    import torch
    from gemma_mt import attention as attn
    from gemma_mt.losses import sft_nll_sum
    from gemma_mt.modeling import load_model, sliding_window_of
    from gemma_mt.train_sft import SFTConfig, build_optimizer
    g = torch.Generator().manual_seed(2)
    batches = []
    for _ in range(steps):
        ids = torch.randint(4, 200000, (2, seq), generator=g)
        lab = ids.clone(); lab[:, : seq // 3] = -100
        seg = torch.ones(2, seq, dtype=torch.int32); pos = torch.arange(seq)[None].repeat(2, 1)
        batches.append((ids, lab, seg, pos))

    def first_loss(d):
        m = load_model(d).eval()
        ids, lab, seg, pos = batches[0]
        kw = attn.model_inputs({"input_ids": ids, "position_ids": pos, "segment_ids": seg}, attn_implementation="sdpa", mask_mode="explicit", sliding_window=sliding_window_of(m))
        with torch.no_grad():
            nll, n = sft_nll_sum(m, kw, lab)
        return float(nll) / n

    l_base, l_up0 = first_loss(base), first_loss(up)
    if abs(l_base - l_up0) > 1e-5:
        _fail(f"step-1 loss differs: base {l_base} vs upcycled {l_up0}")
    m = load_model(up)
    m.train()
    cfg = SFTConfig(optimizer=optimizer, learning_rate=5e-5, weight_decay=0.01)
    opt = build_optimizer(m, cfg)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in m.parameters())
    if n_train != n_all:
        _fail("not all parameters are trainable")
    losses, gnorms = [], []
    new_idx = {}
    for step, (ids, lab, seg, pos) in enumerate(batches):
        for g_ in opt.param_groups:
            g_["lr"] = 5e-5 * (step + 1) / 5 if step < 5 else 5e-5
        kw = attn.model_inputs({"input_ids": ids, "position_ids": pos, "segment_ids": seg}, attn_implementation="sdpa", mask_mode="explicit", sliding_window=sliding_window_of(m))
        opt.zero_grad(set_to_none=True)
        nll, n = sft_nll_sum(m, kw, lab)
        (nll / n).backward()
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0))
        if not (math.isfinite(float(nll)) and math.isfinite(gn)):
            _fail(f"non-finite loss/gradient at step {step + 1}: {float(nll)} {gn}")
        opt.step()
        losses.append(float(nll) / n); gnorms.append(gn)
    bad = [k for k, p in m.named_parameters() if not bool(torch.isfinite(p).all())]
    if bad:
        _fail(f"non-finite weights after training: {bad[:3]}")
    print(f"[4] {steps} steps with {optimizer}: all parameters trainable ({n_train:,}); step-1 loss base {l_base:.6f} == upcycled {l_up0:.6f}; "
          f"loss {losses[0]:.3f} -> {losses[-1]:.3f}; grad norm step 1 {gnorms[0]:.3f}, max {max(gnorms):.3f}; weights finite", flush=True)
    return {"optimizer": optimizer, "losses": losses, "grad_norms": gnorms}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base"); ap.add_argument("--up")
    ap.add_argument("--seq", type=int, default=700, help="longer than the sliding window (512) so that the window matters")
    ap.add_argument("--train-steps", type=int, default=20); ap.add_argument("--optimizer", choices=["muon", "adamw", "both"], default="both")
    ap.add_argument("--tol", type=float, default=1e-5); ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--vllm-config-only", metavar="DIR", help="only run the config-level vLLM check on a model directory (e.g. a downloaded google/gemma-3-1b-it)")
    a = ap.parse_args(argv)
    if a.vllm_config_only:
        check_vllm_config(a.vllm_config_only); return 0
    import torch
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    if not a.base or not a.up:
        ap.error("--base and --up are required")
    info = json.load(open(os.path.join(a.up, "upcycle.json")))
    check_tensors(a.base, a.up, info)
    if not a.skip_hf:
        check_hf(a.base, a.up, a.seq, a.tol)
    check_repo(a.base, a.up, a.seq, a.tol)
    check_vllm_config(a.base); check_vllm_config(a.up)
    out = []
    if a.train_steps > 0:
        for o in (["muon", "adamw"] if a.optimizer == "both" else [a.optimizer]):
            out.append(check_training(a.base, a.up, a.train_steps, o))
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
