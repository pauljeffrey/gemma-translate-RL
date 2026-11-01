# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-67 (R12): exported model directories carry a greedy generation_config.json."""
import json
import os

import pytest


def test_write_greedy_generation_config_rewrites_a_sampling_config(tmp_path):
    from gemma_mt.format import write_greedy_generation_config

    d = tmp_path / "m"
    d.mkdir()
    json.dump({"cache_implementation": "hybrid", "do_sample": True, "eos_token_id": [106, 1], "pad_token_id": 0, "top_k": 64, "top_p": 0.95, "transformers_version": "5.19.0"}, open(d / "generation_config.json", "w"))
    write_greedy_generation_config(str(d))
    g = json.load(open(d / "generation_config.json"))
    assert g["do_sample"] is False and g["eos_token_id"] == [1, 106] and g["max_new_tokens"] == 320 and g["cache_implementation"] == "hybrid" and g["pad_token_id"] == 0
    assert not any(k in g for k in ("top_k", "top_p", "min_p", "temperature"))
    write_greedy_generation_config(str(d), stop_ids=[7, 9])
    assert json.load(open(d / "generation_config.json"))["eos_token_id"] == [7, 9]


def test_save_model_and_interpolate_export_greedy_configs(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import GenerationConfig

    from gemma_mt.modeling import load_model, save_model

    m = load_model("tiny-random", dtype=torch.float32, device="cpu")
    m.generation_config.do_sample, m.generation_config.top_k, m.generation_config.top_p = True, 64, 0.95
    save_model(m, None, str(tmp_path / "a"), stop_ids=[1, 106])
    g = json.load(open(tmp_path / "a" / "generation_config.json"))
    assert g["do_sample"] is False and g["eos_token_id"] == [1, 106] and "top_k" not in g
    gc = GenerationConfig.from_pretrained(str(tmp_path / "a"))
    assert gc.do_sample is False and gc.max_new_tokens == 320
    save_model(m, None, str(tmp_path / "b"), stop_ids=[1, 106], greedy_generation_config=False)
    assert json.load(open(tmp_path / "b" / "generation_config.json")).get("top_k") == 64          # opt-out keeps the model's own config
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
    from interpolate_weights import interpolate_to_dir

    interpolate_to_dir(str(tmp_path / "b"), os.path.join(str(tmp_path / "a"), "model.safetensors") if False else str(tmp_path / "a"), 0.5, str(tmp_path / "c"))
    g = json.load(open(tmp_path / "c" / "generation_config.json"))
    assert g["do_sample"] is False and "top_k" not in g and g["max_new_tokens"] == 320
