# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import os

import pytest

from gemma_mt.hub import dataset_card, model_card, push_dataset, push_model


def test_model_card_has_exact_format_and_license():
    c = model_card.build_model_card((os.environ['HF_ORG_LEGACY'] + "/gemma3-270m-en2african-sft"))
    assert c.startswith("---\nlicense: gemma\n")
    assert "base_model: google/gemma-3-270m-it" in c
    assert "Translate the text to {african_language}:  {english_text}" in c          # two spaces
    assert "<bos><start_of_turn>user\nTranslate the text to Yoruba:  Good morning, how are you today?<end_of_turn>\n" \
           "<start_of_turn>model\n" in c
    assert "apply_chat_template" in c and 'pipeline("text-generation"' in c and "do_sample=False" in c
    assert "Gemma Terms of Use" in c and "Pidgin" in c and "Zulu" in c
    assert "sabiyarn" not in c.lower()
    assert "Additional terms for data generated with this model" in c and "closed or commercial models" in c
    assert "Nothing in these terms obliges you to release your own models" in c
    custom = model_card.build_model_card("x/y", license_id="other-license", stage="rl")
    assert custom.startswith("---\nlicense: other-license\n") and "grpo" in custom


def test_dataset_card_licence_and_schema():
    rl = dataset_card.build_dataset_card((os.environ['HF_ORG_LEGACY'] + "/gemma3-270m-en2african-rl-prompts"), "rl")
    assert "license: cc-by-4.0" in rl and "src_afr_tokens" in rl and "source_dataset" in rl
    assert "No reference translations" in rl
    sft = dataset_card.build_dataset_card("x/sft", "sft")
    assert "`target`" in sft
    with pytest.raises(ValueError):
        dataset_card.build_dataset_card("x/y", "other")


def test_default_repo_names_and_privacy():
    assert push_model.DEFAULT_REPOS == {"sft": (os.environ['HF_ORG_LEGACY'] + "/gemma3-270m-translate"),
                                        "rl": (os.environ['HF_ORG_LEGACY'] + "/gemma3-270m-en2african-rl")}
    assert push_dataset.DEFAULT_REPOS == {"sft": (os.environ['HF_ORG_LEGACY'] + "/gemma3-270m-en2african-sft-data"),
                                          "rl": (os.environ['HF_ORG_LEGACY'] + "/gemma3-270m-en2african-rl-prompts")}


def _ckpt(tmp_path, eos=(106, 1)):
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x")
    (d / "config.json").write_text("{}")
    (d / "generation_config.json").write_text(json.dumps({"eos_token_id": list(eos)}))
    return str(d)


def test_push_model_refuses_without_yes_and_on_bad_checkpoint(tmp_path):
    d = _ckpt(tmp_path)
    assert push_model.main(["--model-dir", d, "--stage", "sft"]) == 2            # dry run, nothing uploaded
    assert not os.path.exists(os.path.join(d, "README.md")), "a dry run must not write into the model dir"
    assert push_model.check_checkpoint(d) == []
    d2 = tmp_path / "bad"
    d2.mkdir()
    (d2 / "model.safetensors").write_bytes(b"x")
    (d2 / "config.json").write_text("{}")
    (d2 / "generation_config.json").write_text(json.dumps({"eos_token_id": 1}))
    probs = push_model.check_checkpoint(str(d2))
    assert any("end_of_turn" in p for p in probs)
    assert push_model.main(["--model-dir", str(d2), "--stage", "sft", "--yes"]) == 3


def test_push_dataset_dry_run(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "train.jsonl").write_text("{}\n")
    assert push_dataset.main(["--data-dir", str(d), "--kind", "rl"]) == 2
    assert not (d / "README.md").exists()


def test_importing_hub_modules_has_no_side_effects():
    import importlib

    for name in ("gemma_mt.hub.push_model", "gemma_mt.hub.push_dataset", "gemma_mt.hub.model_card"):
        importlib.import_module(name)          # would raise / print / upload if anything ran at import time


def test_model_card_reverse_direction_section_only_when_bidirectional():
    one = model_card.build_model_card("x/y")
    two = model_card.build_model_card("x/y", directions=("en2x", "x2en"))
    assert "Reverse direction" not in one and "only translates **from English**" in one
    assert "Reverse direction" in two and "Translate the text to English:  E kaaro" in two and "eng " in two
