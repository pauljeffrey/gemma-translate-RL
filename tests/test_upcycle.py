# SPDX-License-Identifier: AGPL-3.0-or-later
"""Depth upcycling (scripts/upcycle_layers.py, arm A2-up) and the S2 start checkpoint check (scripts/upcycle_s2_start_check.py). CPU, tiny models."""
import importlib.util
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load("upcycle_layers")
CHECK = _load("upcycle_check")

CFG_270M = dict(vocab_size=262144, hidden_size=640, intermediate_size=2048, num_hidden_layers=18, num_attention_heads=4, num_key_value_heads=1, head_dim=256,
                sliding_window=512, max_position_embeddings=32768, query_pre_attn_scalar=256, rope_local_base_freq=10000.0, rope_theta=1000000.0)


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    from gemma_mt.modeling import build_tiny_model
    d = tmp_path_factory.mktemp("tiny_base")
    m = build_tiny_model(seed=3, attn_implementation="sdpa", dtype=torch.float32)
    m.save_pretrained(str(d), safe_serialization=True)
    return str(d)


def _up(tiny_base, tmp_path, name="up", **kw):
    out = str(tmp_path / name)
    kw.setdefault("n_new", 2)
    info = U.upcycle(tiny_base, out, **kw)
    return out, info


def test_positions_and_pattern_arithmetic():
    assert U.plan_positions(18, 6) == [1, 4, 7, 10, 13, 16]
    assert U.plan_positions(6, 2) == [1, 4]
    lm = U.layer_map(18, [1, 4, 7, 10, 13, 16])
    assert len(lm) == 24 and [i for i, (s, new) in enumerate(lm) if new] == [2, 6, 10, 14, 18, 22]
    types = ["sliding_attention"] * 5 + ["full_attention"]
    types = types * 3
    new = [types[s] for s, _ in lm]
    assert [i for i, t in enumerate(new) if t == "full_attention"] == [7, 15, 23] and U.sliding_pattern(new) == 8
    assert U.sliding_pattern(types) == 6
    assert U.sliding_pattern(["sliding_attention"] * 4) is None
    with pytest.raises(ValueError):
        U.plan_positions(4, 5)
    # up12: the evenly spaced formula would insert after global layers (5, 11, 17); the new layers go inside the three runs of five sliding layers instead
    t18 = (["sliding_attention"] * 5 + ["full_attention"]) * 3
    assert U.plan_positions(18, 6, t18) == [1, 4, 7, 10, 13, 16], "up6 keeps the plain formula"
    p12 = U.plan_positions(18, 12, t18)
    assert p12 == [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16] and not any(t18[p] == "full_attention" for p in p12)
    t30 = [("sliding_attention" if s_ != "g" else "full_attention") for s_ in [t18[x] for x, _ in U.layer_map(18, p12)]]
    t30 = [t18[x] if not new else "sliding_attention" for x, new in U.layer_map(18, p12)]
    assert len(t30) == 30 and [i for i, t in enumerate(t30) if t == "full_attention"] == [9, 19, 29] and U.sliding_pattern(t30) == 10


def test_exact_parameter_count_of_the_real_270m_architecture():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    base_types = (["sliding_attention"] * 5 + ["full_attention"]) * 3
    new_types = [base_types[s] for s, _ in U.layer_map(18, U.plan_positions(18, 6))]
    with torch.device("meta"):
        base = Gemma3ForCausalLM(Gemma3TextConfig(layer_types=base_types, **CFG_270M))
        up = Gemma3ForCausalLM(Gemma3TextConfig(layer_types=new_types, **{**CFG_270M, "num_hidden_layers": 24}))
    nb, nu = sum(p.numel() for p in base.parameters()), sum(p.numel() for p in up.parameters())
    assert nb == 268_098_176 and nu == 301_539_968 and (nu - nb) == 6 * 5_573_632
    t12 = [base_types[s] if not new else "sliding_attention" for s, new in U.layer_map(18, U.plan_positions(18, 12, base_types))]
    with torch.device("meta"):
        up12 = Gemma3ForCausalLM(Gemma3TextConfig(layer_types=t12, **{**CFG_270M, "num_hidden_layers": 30}))
    assert sum(p.numel() for p in up12.parameters()) == 334_981_760 == nb + 12 * 5_573_632


CFG_1B = dict(vocab_size=262144, hidden_size=1152, intermediate_size=6912, num_hidden_layers=26, num_attention_heads=4, num_key_value_heads=1, head_dim=256,
              sliding_window=512, max_position_embeddings=32768, query_pre_attn_scalar=256, rope_local_base_freq=10000.0, rope_theta=1000000.0)


def test_gemma3_1b_architecture_is_supported_by_the_repo_code_and_the_expected_tensor_list():
    """google/gemma-3-1b-it is a text-only Gemma3ForCausalLM (model_type gemma3_text) like the 270M: same loader, same loss path, same tensor naming."""
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    types = (["sliding_attention"] * 5 + ["full_attention"]) * 4 + ["sliding_attention"] * 2
    cfg = Gemma3TextConfig(layer_types=types, **CFG_1B)
    with torch.device("meta"):
        m = Gemma3ForCausalLM(cfg)
    n = sum(p.numel() for p in m.parameters())
    assert n == 999_885_952, n
    shapes = {k: tuple(v.shape) for k, v in m.state_dict().items() if k != "lm_head.weight"}
    assert shapes == CHECK.expected_tensor_shapes(CFG_1B), "expected_tensor_shapes matches the HF module (names and shapes) for the 1B"
    from gemma_mt.modeling import sliding_window_of
    assert sliding_window_of(m) == 512 and hasattr(m, "lm_head") and hasattr(m.model, "layers") and m.lm_head.weight is m.model.embed_tokens.weight
    assert [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"] == [5, 11, 17, 23]


def test_upcycled_model_is_the_base_function_hf_and_repo_paths(tiny_base, tmp_path):
    from transformers import AutoModelForCausalLM
    out, info = _up(tiny_base, tmp_path)
    assert info["insert_after_original_layers"] == [1, 4] and info["sliding_window_pattern"] == 8 and info["parameters_added"] == 2 * info["parameters_per_layer"]
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["num_hidden_layers"] == 8 and len(cfg["layer_types"]) == 8
    assert [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"] == [7] and cfg["_sliding_window_pattern"] == 8 == cfg["sliding_window_pattern"]
    CHECK.check_tensors(tiny_base, out, info)                                        # copies, zeros, config, parameter count
    ids = torch.randint(4, 500, (1, 40), generator=torch.Generator().manual_seed(0))
    for impl in ("sdpa", "eager"):
        ms = [AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32, attn_implementation=impl).eval() for d in (tiny_base, out)]
        with torch.no_grad():
            a, b = ms[0](input_ids=ids, use_cache=False).logits, ms[1](input_ids=ids, use_cache=False).logits
        assert float((a - b).abs().max()) < 1e-6, impl
    # the repo's own masks (packed, block-diagonal, sliding window 8 < sequence length)
    from gemma_mt import attention as attn
    from gemma_mt.losses import final_hidden
    from gemma_mt.modeling import load_model, sliding_window_of
    seg = torch.cat([torch.full((17,), 1), torch.full((23,), 2)])[None].int()
    pos = torch.cat([torch.arange(17), torch.arange(23)])[None].long()
    hs = []
    for d in (tiny_base, out):
        m = load_model(d).eval()
        kw = attn.model_inputs({"input_ids": ids, "position_ids": pos, "segment_ids": seg}, attn_implementation="sdpa", mask_mode="explicit", sliding_window=sliding_window_of(m))
        with torch.no_grad():
            hs.append(final_hidden(m, **kw))
    assert float((hs[0] - hs[1]).abs().max()) < 1e-6
    assert not any(k.endswith("lm_head.weight") for k in __import__("safetensors.torch", fromlist=["load_file"]).load_file(os.path.join(out, "model.safetensors"))), "embeddings stay tied"


def test_new_layers_train_and_the_function_moves_away_from_the_base(tiny_base, tmp_path):
    from gemma_mt import attention as attn
    from gemma_mt.losses import sft_nll_sum
    from gemma_mt.modeling import load_model, sliding_window_of
    from gemma_mt.train_sft import SFTConfig, build_optimizer
    out, _ = _up(tiny_base, tmp_path)
    ids = torch.randint(4, 500, (2, 48), generator=torch.Generator().manual_seed(1))
    lab = ids.clone(); lab[:, :10] = -100
    seg = torch.ones(2, 48, dtype=torch.int32); pos = torch.arange(48)[None].repeat(2, 1)
    for opt_name in ("adamw", "muon"):
        if opt_name == "muon" and not hasattr(torch.optim, "Muon"):
            continue
        m = load_model(out); m.train()
        opt = build_optimizer(m, SFTConfig(optimizer=opt_name, learning_rate=1e-3))
        assert all(p.requires_grad for p in m.parameters())
        w0 = {n: p.detach().clone() for n, p in m.named_parameters()}
        losses = []
        for _ in range(20):
            kw = attn.model_inputs({"input_ids": ids, "position_ids": pos, "segment_ids": seg}, attn_implementation="sdpa", mask_mode="explicit", sliding_window=sliding_window_of(m))
            opt.zero_grad(set_to_none=True)
            nll, n = sft_nll_sum(m, kw, lab)
            (nll / n).backward()
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            assert torch.isfinite(nll) and torch.isfinite(gn), (opt_name, float(nll), float(gn))
            opt.step()
            losses.append(float(nll) / n)
        assert all(np.isfinite(losses)) and losses[-1] < losses[0] - 0.05
        assert all(bool(torch.isfinite(p).all()) for p in m.parameters())
        for i in (2, 6):                                                              # the inserted layers (indices in the 8-layer stack) did train
            assert float((m.model.layers[i].self_attn.o_proj.weight - w0[f"model.layers.{i}.self_attn.o_proj.weight"]).abs().max()) > 0
            assert float((m.model.layers[i].mlp.down_proj.weight - w0[f"model.layers.{i}.mlp.down_proj.weight"]).abs().max()) > 0


def test_post_norm_gain_keeps_the_identity_and_is_off_by_default(tiny_base, tmp_path):
    from transformers import AutoModelForCausalLM
    out, info = _up(tiny_base, tmp_path, name="gain", post_norm_gain=0.05)
    CHECK.check_tensors(tiny_base, out, info)
    ids = torch.randint(4, 500, (1, 30), generator=torch.Generator().manual_seed(2))
    a, b = [AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()(input_ids=ids, use_cache=False).logits for d in (tiny_base, out)]
    assert float((a - b).abs().max()) < 1e-6
    from safetensors.torch import load_file
    sd = load_file(os.path.join(out, "model.safetensors"))
    assert torch.allclose(sd["model.layers.2.post_attention_layernorm.weight"], torch.full_like(sd["model.layers.2.post_attention_layernorm.weight"], 0.05 - 1.0))
    plain = load_file(os.path.join(_up(tiny_base, tmp_path, name="plain")[0], "model.safetensors"))
    assert torch.equal(plain["model.layers.2.post_attention_layernorm.weight"], plain["model.layers.1.post_attention_layernorm.weight"]), "default: a pure copy"


def test_new_layers_are_local_even_when_they_copy_a_global_layer_and_aperiodic_layouts_are_refused(tiny_base, tmp_path):
    # default: every new layer is sliding-window, whatever it copies (here: the global layer 5)
    out, info = _up(tiny_base, tmp_path, name="local", positions=[5])
    cfg = json.load(open(os.path.join(out, "config.json")))
    assert info["new_layer_type"] == "sliding_attention" and cfg["layer_types"] == ["sliding_attention"] * 5 + ["full_attention", "sliding_attention"]
    CHECK.check_tensors(tiny_base, out, info)
    out2, info2 = _up(tiny_base, tmp_path, name="copytype", positions=[5], new_layer_type="copy", allow_aperiodic=True)
    assert json.load(open(os.path.join(out2, "config.json")))["layer_types"][-2:] == ["full_attention", "full_attention"]
    # a GLOBAL new layer at depth 1 makes the layout aperiodic: refused (the integer pattern cannot describe it)
    with pytest.raises(SystemExit, match="not periodic"):
        U.upcycle(tiny_base, str(tmp_path / "bad"), positions=[0], new_layer_type="full_attention")
    out3, info3 = _up(tiny_base, tmp_path, name="ok_aperiodic", positions=[0], new_layer_type="full_attention", allow_aperiodic=True)
    assert info3["sliding_window_pattern"] is None


def test_up12_layout_on_a_tiny_three_group_model_and_vllm_config_check(tmp_path):
    """3 groups of (3 sliding + 1 global) = 12 layers; 6 new layers (4 -> quota 2 per run of 3): periodic, new layers local, identity, config-level vLLM check."""
    from gemma_mt.modeling import build_tiny_model
    from transformers import AutoModelForCausalLM
    m = build_tiny_model(seed=5, attn_implementation="sdpa", dtype=torch.float32, num_hidden_layers=12, layer_types=(["sliding_attention"] * 3 + ["full_attention"]) * 3, sliding_window_pattern=4)
    base = str(tmp_path / "b"); m.save_pretrained(base, safe_serialization=True)
    out = str(tmp_path / "u")
    info = U.upcycle(base, out, n_new=6)
    assert all(info["layer_types"][r["new_index"]] == "sliding_attention" for r in info["layers"] if r["new"]) and 3 not in info["insert_after_original_layers"]
    assert info["sliding_window_pattern"] is not None and info["n_new_layers"] == 6
    CHECK.check_tensors(base, out, info)
    CHECK.check_vllm_config(base); total = CHECK.check_vllm_config(out)
    assert total == info["parameters_upcycled"]
    ids = torch.randint(4, 500, (1, 33), generator=torch.Generator().manual_seed(4))
    a, b = [AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()(input_ids=ids, use_cache=False).logits for d in (base, out)]
    assert float((a - b).abs().max()) < 1e-6
    # the vLLM config check catches a layer_types / pattern contradiction and a wrong tensor list
    cfg = json.load(open(os.path.join(out, "config.json"))); cfg2 = dict(cfg); cfg2["sliding_window_pattern"] = cfg["sliding_window_pattern"] + 1
    os.makedirs(tmp_path / "w", exist_ok=True)
    for f in os.listdir(out):
        os.link(os.path.join(out, f), tmp_path / "w" / f) if f != "config.json" else None
    json.dump(cfg2, open(tmp_path / "w" / "config.json", "w"))
    with pytest.raises(SystemExit):
        CHECK.check_vllm_config(str(tmp_path / "w"))


# ------------------------------------------------------------------------------------------------ S2 start checkpoint
ART = os.path.join(REPO, "gemma_mt", "artifacts", "tokswap_2026-10-25", "control_geez_N3000")
SWAP = os.path.join(REPO, "reports", "data", "tokswap_2026-10-25", "tokenizers", "geez_N3000")


@pytest.mark.skipif(not (os.path.exists(os.path.join(ART, "model.safetensors")) and os.path.exists(os.path.join(SWAP, "swap_map.json"))),
                    reason="needs the local tokswap artifacts (stand-in base) and the swapped tokenizer")
def test_s2_start_build_and_check_roundtrip_and_a_bad_map_is_refused(tmp_path):
    import gzip
    import shutil
    import subprocess
    import sys
    swap = tmp_path / "swap"; swap.mkdir()
    for f in ("swap_map.json", "tokenizer_config.json"):
        shutil.copy(os.path.join(SWAP, f), swap / f)
    for f in ("chat_template.jinja", "special_tokens_map.json"):                      # as scripts/upcycle_s2_start.sh: copied from the original tokenizer folder
        if os.path.exists(os.path.join(ART, f)):
            shutil.copy(os.path.join(ART, f), swap / f)
    with gzip.open(os.path.join(SWAP, "tokenizer.json.gz")) as g, open(swap / "tokenizer.json", "wb") as o:
        shutil.copyfileobj(g, o)
    start = tmp_path / "start"
    sw = os.path.join(SCRIPTS, "tokswap_swap_vocab.py")
    subprocess.check_call([sys.executable, sw, "surgery", "--model", ART, "--swap", str(swap), "--orig-tok", os.path.join(ART, "tokenizer.json"), "--dtype", "fp32", "--out", str(start)],
                          stdout=subprocess.DEVNULL)
    chk = os.path.join(SCRIPTS, "upcycle_s2_start_check.py")
    r = subprocess.run([sys.executable, chk, "--base", ART, "--start", str(start), "--swap-map", str(swap / "swap_map.json")], capture_output=True, text=True)
    assert r.returncode == 0 and "S2 START CHECKPOINT OK" in r.stdout, r.stdout + r.stderr
    sm = json.load(open(swap / "swap_map.json"))
    sm["new_tokens"][5]["id"] = 262144 + 5                                             # outside the unused tail
    json.dump(sm, open(swap / "bad_map.json", "w"))
    r = subprocess.run([sys.executable, chk, "--base", ART, "--start", str(start), "--swap-map", str(swap / "bad_map.json")], capture_output=True, text=True)
    assert r.returncode == 1 and "outside" in r.stdout
    sm = json.load(open(swap / "swap_map.json")); sm["orig_tokenizer_sha256"] = "0" * 64
    json.dump(sm, open(swap / "bad_map2.json", "w"))
    r = subprocess.run([sys.executable, chk, "--base", ART, "--start", str(start), "--swap-map", str(swap / "bad_map2.json")], capture_output=True, text=True)
    assert r.returncode == 1 and "original tokenizer" in r.stdout


def test_gain_001_model_is_bit_identical_to_the_base_at_init_and_records_the_gain(tiny_base, tmp_path):
    from transformers import AutoModelForCausalLM
    out, info = _up(tiny_base, tmp_path, name="g001", post_norm_gain=0.01)
    ids = torch.randint(4, 500, (2, 41), generator=torch.Generator().manual_seed(7))
    a, b = [AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()(input_ids=ids, use_cache=False).logits for d in (tiny_base, out)]
    assert torch.equal(a, b) and info["post_norm_gain"] == 0.01 and json.load(open(os.path.join(out, "upcycle.json")))["post_norm_gain"] == 0.01
