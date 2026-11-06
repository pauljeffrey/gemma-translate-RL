# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dynamic sampling (owner 2026-11-03; rl/dynamic_sampling.py): zero-variance detection, row selection / concatenation, the refill loop (cap, language bookkeeping, fallback),
the replacement draw of the schedule, config validation."""
import json
import os
from collections import Counter

import pytest

from gemma_mt.rl.data import PromptSchedule, load_rl_prompts
from gemma_mt.rl.dynamic_sampling import LanguageCounters, concat_rollout_output, dynamic_rollout, informative_groups, subset_rollout_output
from gemma_mt.rl.reward import RewardOutput, Sample
from gemma_mt.rl.sampling import Rollout

G = 4
ISO = {0: "yor", 1: "hau", 2: "ibo", 3: "yor", 4: "hau", 5: "ibo", 6: "yor", 7: "hau"}


def make_gen(flat_prompts, calls=None):
    """generate(indices, round): a prompt in `flat_prompts` gets four identical rewards (-2: all failed), every other prompt a spread; completion k of prompt p has length p + k + 1."""
    def gen(ix, rnd):
        if calls is not None:
            calls.append((list(ix), rnd))
        comps, rewards, status, samples = [], [], [], []
        for p in ix:
            for k in range(G):
                comps.append([7] * (p + k + 1))
                flat = p in flat_prompts
                rewards.append(-2.0 if flat else -1.0 + 0.5 * k + 0.01 * p)
                status.append("no_end_of_turn" if flat else "ok")
                samples.append(Sample(comps[-1], ISO.get(p, "yor"), f"src {p}"))
        n = len(comps)
        roll = Rollout([[1, 2]] * n, comps, [s == "ok" for s in status], None, [0] * n)
        out = RewardOutput(rewards, [1.0] * n, [None] * n, status, [""] * n, [""] * n, {"fail/no_end_of_turn": sum(s != "ok" for s in status) / n}, consensus_chrf=[None] * n)
        return roll, samples, out
    return gen


def redraw_fn(log):
    def redraw(dropped, rnd):
        log.append((list(dropped), rnd))
        return [100 * rnd + 10 + k for k in range(len(dropped))]            # a fresh, informative prompt per dropped one
    return redraw


def iso_of(p):
    return ISO.get(p, "yor")


def test_informative_groups_flat_tiny_spread_sampled_only_and_exclude():
    r = [1, 1, 1, 1,  0, 0.5, 1, 1.5,  -2, -2, -2, -2,  0.5, 0.5, 0.5, 0.504]
    assert informative_groups(r, 4, 0.01) == [False, True, False, False]            # 0.0017 of std is jitter, not a signal
    assert informative_groups(r, 4, 0.0)[0] is False and informative_groups(r, 4, 0.0)[3] is True
    # sampled_only: the group's greedy (last) member alone does not make it informative
    assert informative_groups([1, 1, 1, -2], 4, 0.01, sampled_only=False) == [True]
    assert informative_groups([1, 1, 1, -2], 4, 0.01, sampled_only=True) == [False]
    # exclude: the spread is measured on the non-failed members, fewer than 2 survivors = uninformative
    failed = [True, False, True, True,  False, False, True, True]
    assert informative_groups([-2, 1, -2, -2,  0.0, 1.0, -2, -2], 4, 0.01, failed=failed, mask_mode="exclude") == [False, True]
    assert informative_groups([-2, 1, -2, -2,  0.0, 1.0, -2, -2], 4, 0.01) == [True, True]
    with pytest.raises(ValueError):
        informative_groups([1, 2, 3], 2, 0.01)


def test_a_guide_makes_a_group_informative_unless_it_failed_and_is_dropped():
    # four members, the guide is member index 2 (the greedy member is last); samples agree, the guide differs: informative BECAUSE of the guide (sampled members = first three)
    assert informative_groups([1, 1, 2, 1], 4, 0.01, sampled_only=True) == [True]
    # the same group with a hard-failed guide that is dropped: the others are flat, the group is not informative
    assert informative_groups([1, 1, -2, 1], 4, 0.01, sampled_only=True, drop=[False, False, True, False]) == [False]
    assert informative_groups([1, 1, -2, 1], 4, 0.01, sampled_only=False, drop=[False, False, True, False]) == [False]
    assert informative_groups([1, 0, -2, 2], 4, 0.01, sampled_only=False, drop=[False, False, True, False]) == [True]
    # a group left with fewer than two members is not informative
    assert informative_groups([1, 0, -2, 2], 4, 0.01, sampled_only=True, drop=[True, True, False, False]) == [False]


def test_subset_and_concat_keep_every_per_row_field_aligned():
    roll, samples, out = make_gen(set())([0, 1, 2], 0)
    s_roll, s_samples, s_out = subset_rollout_output(roll, samples, out, [2, 0], G)
    assert len(s_roll.completions) == len(s_out.rewards) == len(s_samples) == 2 * G
    assert [len(c) for c in s_roll.completions[:G]] == [3, 4, 5, 6] and s_samples[0].source == "src 2" and s_samples[G].source == "src 0"
    assert s_out.rewards[:G] == out.rewards[2 * G:3 * G] and s_out.status[G:] == out.status[:G]
    assert s_out.consensus_chrf == [None] * (2 * G) and s_out.decoded == [""] * (2 * G)          # optional columns follow
    assert s_out.texts == [""] * (2 * G) and s_out.diacritic_deficit == []                       # an empty (feature off) column stays empty
    a = make_gen(set())([0], 0)
    b = make_gen({5})([5, 6], 1)
    r2, s2, o2 = concat_rollout_output(a, b)
    assert len(r2.completions) == 3 * G == len(o2.rewards) == len(s2) == len(r2.finished) == len(o2.status)
    assert o2.rewards[:G] == a[2].rewards and o2.rewards[G:] == b[2].rewards
    assert o2.stats["fail/no_end_of_turn"] == pytest.approx((0 * G + (G / (2 * G)) * (2 * G)) / (3 * G))      # sample-weighted mean of the two calls


def test_no_refill_when_the_oversampled_draw_already_holds_enough_informative_groups():
    calls, log = [], []
    res = dynamic_rollout([0, 1, 2, 3, 4], 4, G, 2, 0.01, make_gen({1}, calls), redraw_fn(log), iso_of)
    assert len(calls) == 1 and not log                                       # one generation call, nothing redrawn
    assert res.informative == [True, False, True, True, True] and res.keep == [0, 2, 3, 4]
    assert res.stats["ds/dropped_flat"] == 1 and res.stats["ds/kept_groups"] == 4 and res.stats["ds/refill_rounds"] == 0 and res.stats["ds/fallback"] == 0
    assert res.by_lang["hau"] == {"generated": 2, "dropped_flat": 1, "surplus": 0, "kept": 1}      # prompts 1 and 4 are hau; 1 is flat


def test_surplus_informative_groups_are_not_used_but_counted_as_waste():
    res = dynamic_rollout([0, 2, 3, 4, 5], 3, G, 2, 0.01, make_gen(set()), redraw_fn([]), iso_of)
    assert res.keep == [0, 1, 2] and res.stats["ds/surplus"] == 2 and res.stats["ds/dropped_flat"] == 0
    assert sum(d["surplus"] for d in res.by_lang.values()) == 2
    assert res.stats["ds/wasted_completion_tokens"] == sum(len(c) for c in res.roll.completions[3 * G:])


def test_refill_replaces_each_dropped_prompt_once_and_respects_the_cap():
    calls, log = [], []
    res = dynamic_rollout([0, 1, 2, 3], 4, G, 2, 0.01, make_gen({1, 2}, calls), redraw_fn(log), iso_of)
    assert log == [([1, 2], 1)]                                              # both dropped prompts redrawn in ONE extra round (the deficit is 2)
    assert [c[1] for c in calls] == [0, 1] and calls[1][0] == [110, 111]
    assert res.prompt_idx == [0, 1, 2, 3, 110, 111] and res.keep == [0, 3, 4, 5] and res.stats["ds/refill_rounds"] == 1 and res.stats["ds/kept_groups"] == 4
    # the redrawn prompts are flat again (and the next ones too): the second round replaces only those (never the first dropped ones twice), then the cap (2 rounds) stops it
    calls2, log2 = [], []
    res2 = dynamic_rollout([0, 1, 2, 3], 4, G, 2, 0.01, make_gen({1, 2, 110, 111, 210, 211}, calls2), redraw_fn(log2), iso_of)
    assert [rnd for _, rnd in log2] == [1, 2] and log2[1][0] == [110, 111]
    assert res2.stats["ds/refill_rounds"] == 2 and res2.stats["ds/kept_groups"] == 2 and res2.stats["ds/short_batch"] == 1 and len(res2.keep) == 2
    assert len(calls2) == 3                                                  # cost bound: 1 + max_refills generation calls


def test_all_flat_step_falls_back_to_the_first_groups_instead_of_an_empty_batch():
    res = dynamic_rollout([0, 1, 2], 2, G, 1, 0.01, make_gen({0, 1, 2, 110, 111}), redraw_fn([]), iso_of)
    assert res.stats["ds/fallback"] == 1 and res.keep == [0, 1] and res.stats["ds/kept_groups"] == 2 and res.stats["ds/flat_frac"] == 1.0
    assert res.stats["ds/kept_groups"] + res.stats["ds/surplus"] + res.stats["ds/dropped_flat"] == res.stats["ds/generated_groups"]


def test_redraw_must_return_one_prompt_per_dropped_group():
    with pytest.raises(ValueError):
        dynamic_rollout([0, 1], 2, G, 2, 0.01, make_gen({1}), lambda dropped, rnd: [], iso_of)


def test_language_counters_accumulate_and_report_the_worst_languages():
    c = LanguageCounters()
    c.add({"yor": {"generated": 10, "dropped_flat": 5, "surplus": 0, "kept": 5}, "hau": {"generated": 10, "dropped_flat": 0, "surplus": 1, "kept": 9}})
    c.add({"yor": {"generated": 10, "dropped_flat": 1, "surplus": 0, "kept": 9}})
    t = c.table()
    assert t["yor"]["generated"] == 20 and t["yor"]["flat_rate"] == pytest.approx(0.3) and t["hau"]["kept_rate"] == pytest.approx(0.9)
    m = c.step_metrics({"yor": {"generated": 4, "dropped_flat": 2, "surplus": 0, "kept": 2}, "hau": {"generated": 4, "dropped_flat": 0, "surplus": 0, "kept": 4}})
    assert m == {"ds_flat/yor": 0.5}


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _pool(tmp_path):
    root = str(tmp_path / "pool")
    for iso in ("yor", "hau", "ibo"):
        for b, ntok in (("b64", 12), ("b256", 100)):
            _write(f"{root}/{iso}_{b}.jsonl", [{"english": f"General {b} {i} for {iso}.", "iso": iso, "src_afr_tokens": ntok, "length_bin": b} for i in range(50)])
            _write(f"{root}/news/{iso}_{b}.jsonl", [{"english": f"News {b} {i} for {iso}.", "iso": iso, "src_afr_tokens": ntok, "length_bin": b, "domain": "news"} for i in range(50)])
    _write(f"{root}/ref/yor_b64.jsonl", [{"english": f"Ref {i} for yor.", "iso": "yor", "src_afr_tokens": 12, "length_bin": "b64", "domain": "ref", "reference": f"yor ref {i}", "ref_source": "x"} for i in range(30)])
    return root


def test_replacements_keep_language_domain_and_bin_and_are_deterministic(tmp_path):
    p = load_rl_prompts(_pool(tmp_path))
    sch = PromptSchedule(p.iso, 0.0, 3, length_bin=p.length_bin, curriculum={"b64": 0}, domain=p.domain, domain_weights={"general": 0.5, "news": 0.5})
    slots = [("yor", "news", "b64"), ("hau", "general", "b256"), ("ibo", "news", "b256")] * 5
    a = sch.replacements(7, 1, slots)
    assert a == sch.replacements(7, 1, slots) and a != sch.replacements(7, 2, slots) and a != sch.replacements(8, 1, slots)
    for i, (iso, dom, b) in zip(a, slots):
        assert (p.iso[i], p.domain[i], p.length_bin[i]) == (iso, dom, b)
    # a language the (domain, bin) pool lacks: replaced by an ordinary draw of that pool (here "ref" has only yor)
    sch2 = PromptSchedule(p.iso, 0.0, 3, length_bin=p.length_bin, curriculum={"b64": 0}, domain=p.domain, domain_weights={"general": 0.5, "ref": 0.5})
    got = sch2.replacements(0, 1, [("hau", "ref", "b64")] * 6)
    assert all(p.domain[i] == "ref" and p.iso[i] == "yor" for i in got)


def test_replacements_without_bins_or_domains(tmp_path):
    p = load_rl_prompts(_pool(tmp_path))
    sch = PromptSchedule(p.iso, 1.0, 3)
    got = sch.replacements(1, 1, [("hau", "", ""), ("yor", "", "")])
    assert [p.iso[i] for i in got] == ["hau", "yor"]


def test_replacement_draw_keeps_the_language_mixture_of_the_step(tmp_path):
    """Whatever the filter drops, the replacements come from the same languages: the expected language counts of the kept set equal those of an unfiltered draw."""
    p = load_rl_prompts(_pool(tmp_path))
    sch = PromptSchedule(p.iso, 0.0, 5, length_bin=p.length_bin, curriculum={"b64": 0}, domain=p.domain, domain_weights={"general": 0.5, "news": 0.5})
    kept = Counter()
    for step in range(100):
        idx = sch.indices(step, 30)
        pos = [k for k, i in enumerate(idx) if p.iso[i] == "yor"][:5]       # the filter removes yor groups unevenly
        drop = [idx[k] for k in pos]
        keep = [i for k, i in enumerate(idx) if k not in set(pos)]
        new = sch.replacements(step, 1, [(p.iso[i], p.domain[i], p.length_bin[i]) for i in drop])
        kept.update(p.iso[i] for i in keep + new)
    n = sum(kept.values())
    assert n == 100 * 30 and all(abs(kept[l] / n - 1 / 3) < 0.04 for l in ("yor", "hau", "ibo")), kept


def test_config_validation_and_defaults_off():
    pytest.importorskip("torch")
    from gemma_mt.rl.train_rl import RLConfig, validate_config

    cfg = RLConfig()
    assert cfg.dynamic_sampling is False and cfg.dynamic_sampling_oversample == 1.0 and cfg.dynamic_sampling_max_refills == 2 and cfg.dynamic_sampling_min_std == 0.01
    validate_config(cfg)
    for bad in ({"dynamic_sampling_oversample": 0.9}, {"dynamic_sampling_max_refills": -1}, {"dynamic_sampling_min_std": -0.1}):
        with pytest.raises(ValueError):
            validate_config(RLConfig(dynamic_sampling=True, **bad))
    validate_config(RLConfig(dynamic_sampling=True, dynamic_sampling_oversample=1.2))
