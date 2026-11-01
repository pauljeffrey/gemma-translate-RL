# SPDX-License-Identifier: AGPL-3.0-or-later
import collections

import pytest

from gemma_mt.rl.data import PromptSchedule, load_rl_prompts
from gemma_mt.smoke import write_jsonl


def _rows():
    rows = []
    for i in range(30):
        rows.append({"english": f"sentence {i}", "iso": "yor", "language": "Yoruba", "src_afr_tokens": 10, "source_dataset": "a"})
    for i in range(10):
        rows.append({"english": f"another {i}", "iso": "hau", "language": "Hausa", "src_afr_tokens": 12, "source_dataset": "b"})
    rows += [
        {"english": "too long", "iso": "yor", "language": "Yoruba", "src_afr_tokens": 300, "source_dataset": "a"},
        {"english": "north ndebele", "iso": "nde", "language": "North Ndebele", "src_afr_tokens": 5, "source_dataset": "a"},
        {"english": "bad", "iso": "zzz", "language": "?", "src_afr_tokens": 5, "source_dataset": "a"},
        {"english": " ", "iso": "yor", "language": "Yoruba", "src_afr_tokens": 5, "source_dataset": "a"},
        {"english": "name differs", "iso": "pcm", "language": "Nigerian Pidgin", "src_afr_tokens": 5, "source_dataset": "c"},
    ]
    return rows


def test_loader_filters_and_needs_no_reference(tmp_path):
    p = write_jsonl(str(tmp_path / "p.jsonl"), _rows())
    pr = load_rl_prompts(p)
    assert len(pr) == 41
    assert pr.dropped == {"source_too_long": 1, "not_rl_supported": 1, "unknown_iso": 1, "empty_english": 1}
    assert pr.name_mismatches == 1
    assert set(pr.languages()) == {"yor", "hau", "pcm"}
    assert pr.item(0)["source_dataset"] == "a"
    only = load_rl_prompts(p, languages=["hau"])
    assert set(only.iso) == {"hau"}


def test_missing_token_counts_are_filled_by_a_counter(tmp_path):
    rows = [{"english": "short", "iso": "yor"}, {"english": "long " * 50, "iso": "yor"}]
    p = write_jsonl(str(tmp_path / "p.jsonl"), rows)
    pr = load_rl_prompts(p, token_counter=lambda texts: [len(t.split()) * 6 for t in texts])
    assert pr.english == ["short"] and pr.src_afr_tokens == [6]
    assert pr.dropped["source_too_long"] == 1


def test_schedule_is_deterministic_and_language_balanced():
    iso = ["yor"] * 900 + ["hau"] * 100
    nat = PromptSchedule(iso, alpha=1.0, seed=1)
    uni = PromptSchedule(iso, alpha=0.0, seed=1)
    assert nat.indices(5, 64) == nat.indices(5, 64) and nat.indices(5, 64) != nat.indices(6, 64)
    cn, cu = collections.Counter(), collections.Counter()
    for step in range(60):
        cn.update(iso[i] for i in nat.indices(step, 64))
        cu.update(iso[i] for i in uni.indices(step, 64))
    assert 0.05 < cn["hau"] / sum(cn.values()) < 0.16          # ~10% natural
    assert 0.42 < cu["hau"] / sum(cu.values()) < 0.58          # ~50% uniform over languages
    a = nat.indices(3, 8, rank=0, world=2)
    b = nat.indices(3, 8, rank=1, world=2)
    assert len(a) == len(b) == 8
    assert sorted(a + b) == sorted(nat.indices(3, 16, rank=0, world=1)), "ranks must split ONE global draw"


def test_length_bin_schedule_curriculum_stages_then_mixture_and_is_stateless():
    from gemma_mt.rl.data import LENGTH_BINS, MIXED, PromptSchedule, bin_of_tokens

    assert [bin_of_tokens(n) for n in (10, 64, 65, 256, 257, 512, 513, 1024, 1025, 2048, 5000)] == \
        ["b64", "b64", "b256", "b256", "b512", "b512", "b1024", "b1024", "b2048", "b2048", "b2048"]
    iso = ["yor", "hau"] * 60
    lb = [LENGTH_BINS[i % 4] for i in range(120)]            # b64 b256 b512 b1024 ...
    cur = {"b64": 2, "b256": 3, "b512": 2, "b1024": 1}
    sch = PromptSchedule(iso, alpha=1.0, seed=3, length_bin=lb, curriculum=cur)
    seq = [sch.bin_of_step(s) for s in range(11)]
    assert seq == ["b64", "b64", "b256", "b256", "b256", "b512", "b512", "b1024", MIXED, MIXED, MIXED]
    assert sch.stage_boundaries() == [("b64", 0), ("b256", 2), ("b512", 5), ("b1024", 7), (MIXED, 8)]
    for step in range(8):                                      # staged steps draw only their bin
        idx = sch.indices(step, 6)
        assert {lb[i] for i in idx} == {sch.bin_of_step(step)}
        assert idx == sch.indices(step, 6)                      # stateless / deterministic
    seen = set()
    for step in range(8, 20):                                  # the mixture stage uses every bin
        seen |= {lb[i] for i in sch.indices(step, 12)}
    assert seen == set(LENGTH_BINS[:4])
    # a bin without prompts is skipped; no curriculum = plain cycle; no bins = old behaviour
    sch2 = PromptSchedule(iso, 1.0, 3, length_bin=["b64"] * 60 + ["b512"] * 60, curriculum={"b64": 1, "b256": 5, "b512": 1})
    assert [sch2.bin_of_step(s) for s in range(4)] == ["b64", "b512", MIXED, MIXED]
    assert [PromptSchedule(iso, 1.0, 3, length_bin=lb).bin_of_step(s) for s in range(5)] == ["b64", "b256", "b512", "b1024", "b64"]
    assert PromptSchedule(iso, 1.0, 3).bin_of_step(0) is None


def test_language_weights_raise_the_draw_probability_of_a_language():
    from gemma_mt.rl.data import PromptSchedule

    iso = ["yor"] * 500 + ["fon"] * 500
    plain = PromptSchedule(iso, alpha=1.0, seed=1)
    boosted = PromptSchedule(iso, alpha=1.0, seed=1, language_weights={"fon": 3.0})
    f = lambda s: sum(iso[i] == "fon" for step in range(40) for i in s.indices(step, 20)) / 800
    assert 0.4 < f(plain) < 0.6 and f(boosted) > 0.68
