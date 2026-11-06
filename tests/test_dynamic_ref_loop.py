# SPDX-License-Identifier: AGPL-3.0-or-later
"""End to end on CPU (tiny random Gemma3, fake tokenizer, fake scorer): the training loop with dynamic sampling and with reference-anchored prompts (owner 2026-11-03)."""
import json
import math

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


def _rows(path):
    return [json.loads(l) for l in open(path)]


def _train(rows):
    return [r for r in rows if "reward" in r]


def test_dynamic_sampling_loop_with_everything_flat_refills_then_falls_back():
    from gemma_mt.rl.train_rl import run_smoke

    s = run_smoke(sft_steps=60, overrides={"dynamic_sampling": True, "dynamic_sampling_min_std": 100.0, "dynamic_sampling_max_refills": 2, "dynamic_sampling_oversample": 1.0})
    assert s["steps"] == 4
    train = _train(_rows(s["workdir"] + "/rl/metrics.jsonl"))
    assert len(train) == 4
    for r in train:
        assert r["ds/fallback"] == 1 and r["ds/refill_rounds"] == 2 and r["ds/kept_groups"] == 3 and r["ds/generated_groups"] == 9 and r["ds/dropped_flat"] == 6
        assert math.isfinite(r["reward"]) and abs(r["ratio_mean"] - 1.0) < 1e-5 and r["flat_group_frac"] == 1.0
    table = json.load(open(s["workdir"] + "/rl/dynamic_sampling_by_language.json"))
    assert sum(v["generated"] for v in table.values()) >= 9 and all(v["flat_rate"] <= 1.0 for v in table.values())


def test_dynamic_sampling_loop_oversample_keeps_exactly_the_target_and_monitors_all_groups():
    from gemma_mt.rl.train_rl import run_smoke

    s = run_smoke(sft_steps=60, overrides={"dynamic_sampling": True, "dynamic_sampling_min_std": 0.0, "dynamic_sampling_oversample": 5 / 3})
    train = _train(_rows(s["workdir"] + "/rl/metrics.jsonl"))
    assert len(train) == 4
    for r in train:
        assert r["ds/generated_groups"] >= 3 and r["ds/kept_groups"] <= 3
        assert r["ds/kept_groups"] + r["ds/surplus"] + r["ds/dropped_flat"] == r["ds/generated_groups"]
        assert abs(r["ratio_mean"] - 1.0) < 1e-5 and 0.0 <= r["finished_rate"] <= 1.0
    assert abs(train[0]["kl"]) < 1e-6


def test_off_is_bit_for_bit_the_old_loop():
    from gemma_mt.rl.train_rl import run_smoke

    a = run_smoke(sft_steps=60)
    b = run_smoke(sft_steps=60, overrides={"dynamic_sampling": False})
    ra, rb = _train(_rows(a["workdir"] + "/rl/metrics.jsonl")), _train(_rows(b["workdir"] + "/rl/metrics.jsonl"))
    assert [r["reward"] for r in ra] == [r["reward"] for r in rb] and not any(k.startswith("ds/") for k in ra[0])


def test_reference_rows_get_the_chrf_term_in_training_only(tmp_path):
    from gemma_mt.smoke import ENGLISH, fake_translation, write_jsonl
    from gemma_mt.rl.train_rl import run_smoke

    langs = ["yor", "hau", "ibo"]
    rows = [{"english": ENGLISH[i % len(ENGLISH)], "iso": langs[i % 3], "language": "", "src_afr_tokens": 12, "source_dataset": "smoke", "length_bin": "b64", "domain": "general"} for i in range(30)]
    rows += [{"english": ENGLISH[i % len(ENGLISH)], "iso": langs[i % 3], "language": "", "src_afr_tokens": 12, "source_dataset": "smoke", "length_bin": "b64", "domain": "ref",
              "reference": fake_translation(ENGLISH[i % len(ENGLISH)], langs[i % 3]), "ref_source": "synthetic", "ref_licence": "test"} for i in range(30)]
    path = write_jsonl(str(tmp_path / "prompts.jsonl"), rows)
    s = run_smoke(sft_steps=120, overrides={"prompts_path": path, "length_bins": True, "domain_weights": {"general": 0.4, "ref": 0.6}, "reward": {"max_comet_tokens": 510, "reference_weight": 0.35},
                                            "steps": 6, "eval_every": 3})
    rec = _rows(s["workdir"] + "/rl/metrics.jsonl")
    train = _train(rec)
    assert len(train) == 6 and any(r.get("reward/ref/n", 0) > 0 for r in train), "the reference term never fired"
    assert all(0.0 <= r["reward/ref/chrf_mean"] <= 100.0 for r in train if "reward/ref/chrf_mean" in r)
    ev = [r for r in rec if "eval/reward" in r]
    assert ev and all(math.isfinite(r["eval/reward"]) for r in ev)
