# SPDX-License-Identifier: AGPL-3.0-or-later
import os

import pytest

from gemma_mt.utils import dataclass_from_dict, load_yaml, parse_overrides

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_shipped_configs_load_strictly():
    pytest.importorskip("yaml")
    pytest.importorskip("torch")
    from gemma_mt.rl.reward import RewardConfig
    from gemma_mt.rl.train_rl import RLConfig, validate_config
    from gemma_mt.train_sft import SFTConfig

    sft = dataclass_from_dict(SFTConfig, load_yaml(os.path.join(ROOT, "configs", "sft.yaml")))
    assert sft.micro_batch_size * sft.grad_accum_steps * 2048 == 131072
    rl = dataclass_from_dict(RLConfig, load_yaml(os.path.join(ROOT, "configs", "rl.yaml")))
    validate_config(rl)
    rc = dataclass_from_dict(RewardConfig, rl.reward)
    assert rc.preset == "repo" and rc.prefix_policy == "require"
    # whole-text scoring truncates the translation at COMET's 510 tokens, so a long generation budget needs sentence chunking
    assert rc.chunking == "sentences" or rl.max_new_tokens <= 510
    assert rc.chunking in ("off", "sentences") and rc.chunk_agg in ("mean", "source_length")


def test_unknown_keys_are_rejected_and_overrides_parse():
    from gemma_mt.utils import dataclass_from_dict as dfd
    import dataclasses

    @dataclasses.dataclass
    class C:
        a: int = 1

    with pytest.raises(ValueError):
        dfd(C, {"b": 2})
    assert parse_overrides(["lr=1e-5", "name=x", "flag=true"]) == {"lr": 1e-5, "name": "x", "flag": True}


def test_long_sft_config_keeps_the_token_budget_per_step():
    pytest.importorskip("yaml")
    pytest.importorskip("torch")
    from gemma_mt.train_sft import SFTConfig

    sft = dataclass_from_dict(SFTConfig, load_yaml(os.path.join(ROOT, "configs", "sft_long.yaml")))
    assert sft.micro_batch_size * sft.grad_accum_steps * 4096 == 131072


def test_the_gemma_rl_yaml_has_chunking_eos_nfc_omission_and_diacritics_on():
    """Owner 2026-10-17: chunked scoring and the EOS penalty must be ON for the Gemma run, and the trainer builds its reward from this block."""
    pytest.importorskip("yaml")
    pytest.importorskip("torch")
    from gemma_mt.rl.reward import RewardConfig, TranslationReward
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
    from gemma_mt.rl.train_rl import RLConfig

    rl = dataclass_from_dict(RLConfig, load_yaml(os.path.join(ROOT, "configs", "rl.yaml")))
    rc = dataclass_from_dict(RewardConfig, rl.reward)                       # exactly what train_rl.py does (line `dataclass_from_dict(RewardConfig, cfg.reward)`)
    assert rc.chunking == "sentences" and rc.chunk_agg == "source_length" and rc.extra_output_penalty
    assert rc.eos_discipline is True and rc.eos_truncated_penalty is None and rc.eos_post_end_penalty is None and rc.eos_dangling_penalty is None
    assert rc.nfc and rc.omission_detector and rc.diacritic_penalty == 1.0 and rc.length_penalty == 1.0 and rc.lexical_penalty == 1.0
    assert rc.calibration_path == "calibration/shaping_flores_dev_v5.json" and rc.fail_floor == 0.0
    tok = FakeGemmaTokenizer()
    from gemma_mt.rl.lid import FakeLID

    r = TranslationReward(rc, FakeScorer(), tok, [SPECIALS["<end_of_turn>"]], shaper=None, lid=FakeLID())      # loads the shipped calibration from the repo root
    assert r.cfg.chunking == "sentences" and r.cfg.eos_discipline


def test_the_shipped_calibration_covers_every_rl_language():
    from gemma_mt.languages import QE_VALIDATED
    import json as _json

    path = os.path.join(ROOT, "calibration", "shaping_flores_dev_v4.json")
    if not os.path.exists(path):
        pytest.skip("calibration/shaping_flores_dev_v4.json not shipped yet")
    cal = _json.load(open(path, encoding="utf-8"))
    langs = {k for k in cal if not k.startswith("_")}
    assert langs == set(QE_VALIDATED) and len(langs) == 42          # v4 is the frozen 42-language file; v5 adds the 35 languages of D-37 / D-38
    assert cal["efi"]["usable"] is False                                   # no reference set
    for k in langs - {"efi"}:
        e = cal[k]
        assert len(e["anchors"]) == 3 and e["anchors"][0] < e["anchors"][1] < e["anchors"][2]
        assert e["len_ratio"]["corpus"] > 0 and e["len_ratio"]["s2"] > 0 and "diacritics" in e
        if k != "pcm":
            assert e["n"] == 997                                           # FLORES+ dev, all sentences


def test_a_reward_override_is_merged_into_the_yaml_block_not_substituted():
    """scripts/vast_train_rl.sh passes reward='{"calibration_path": ...}': a shallow dict.update replaced the whole `reward:` block of rl.yaml and silently
    reset chunking, the penalties, the omission detector and the end-of-turn discipline to the dataclass defaults (off)."""
    pytest.importorskip("yaml")
    pytest.importorskip("torch")
    from gemma_mt.rl.train_rl import RLConfig, apply_overrides

    d = load_yaml(os.path.join(ROOT, "configs", "rl.yaml"))
    out = apply_overrides(d, parse_overrides(['reward={"calibration_path":"calibration/other.json","preset":"repo"}', "steps=5", 'scorer={"batch_size":16}']))
    assert out["steps"] == 5 and out["reward"]["calibration_path"] == "calibration/other.json"
    assert out["reward"]["chunking"] == "sentences" and out["reward"]["eos_discipline"] is True and out["reward"]["diacritic_penalty"] == 1.0
    assert out["scorer"]["batch_size"] == 16 and out["scorer"]["model_id"] == "masakhane/africomet-qe-stl-1.1"
    rl = dataclass_from_dict(RLConfig, out)
    assert rl.reward["omission_detector"] is True
    assert apply_overrides({"a": {"x": 1}}, {"a": {"y": 2}}) == {"a": {"y": 2}}          # other dict-valued keys are replaced as before
