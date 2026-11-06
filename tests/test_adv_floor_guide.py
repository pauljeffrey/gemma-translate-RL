# SPDX-License-Identifier: AGPL-3.0-or-later
"""Owner 2026-11-03: advantage std floor, per-prompt group statistics, beam-guided group member (all default OFF). Needs torch (CPU, tiny random Gemma3)."""
import json
import math
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gemma_mt.rl import grpo


def _rows(path):
    return [json.loads(l) for l in open(path)]


def test_floor_off_is_the_old_advantage_and_on_damps_only_groups_below_the_floor():
    r = torch.tensor([0.0, 0.1, 0.2, 0.3,   0.0, 0.001, 0.002, 0.003,   1.0, 1.0, 1.0, 1.0])
    old, _ = grpo.group_advantages(r, 4)
    same, _ = grpo.group_advantages(r, 4, std_floor=0.0)
    assert torch.equal(old, same)
    new, flat = grpo.group_advantages(r, 4, std_floor=0.05)
    mean = r.view(3, 4).mean(1, keepdim=True)
    std = r.view(3, 4).std(1, unbiased=False, keepdim=True)
    assert torch.allclose(new.view(3, 4)[0], ((r.view(3, 4) - mean) / std)[0], atol=1e-3)          # std 0.112 > floor: the standardised advantage (no eps)
    assert torch.allclose(new.view(3, 4)[1], (r.view(3, 4) - mean)[1] / 0.05, atol=1e-6)            # near-flat group: divided by the floor, magnitude ~ std / floor
    assert new.view(3, 4)[1].abs().max() < 0.05 and old.view(3, 4)[1].abs().max() > 1.0              # the old rule blew the same jitter up to |A| > 1
    assert torch.equal(new.view(3, 4)[2], torch.zeros(4)) and flat == pytest.approx(1 / 3)           # exactly flat stays 0


def test_floor_in_the_sampled_baseline_and_masked_variants():
    r = torch.tensor([0.0, 0.001, 0.002, 5.0,   0.0, 1.0, 2.0, 3.0])
    a, _ = grpo.group_advantages_sampled_baseline(r, 4, std_floor=0.1)
    s = r.view(2, 4)[:, :-1]
    assert torch.allclose(a.view(2, 4)[0], (r.view(2, 4)[0] - s[0].mean()) / 0.1, atol=1e-5)
    failed = torch.tensor([False, False, False, True, False, False, False, False])
    z, _ = grpo.group_advantages_masked(r, 4, failed, "zero", std_floor=0.1)
    assert z[3] == 0.0 and z.view(2, 4)[1].abs().max() > 0
    e, _ = grpo.group_advantages_masked(r, 4, failed, "exclude", std_floor=0.1)
    assert e[3] == 0.0 and torch.isfinite(e).all()


def test_dropped_members_leave_the_group_statistics_and_get_no_advantage():
    r = torch.tensor([0.0, 1.0, 2.0, -2.0,   0.0, 1.0, 2.0, 3.0])
    drop = torch.tensor([False, False, False, True, False, False, False, False])
    a, _ = grpo.group_advantages(r, 4, drop=drop)
    ref, _ = grpo.group_advantages(torch.tensor([0.0, 1.0, 2.0]), 3)
    assert a[3] == 0.0 and torch.allclose(a[:3], ref, atol=1e-4)                                   # the others: exactly the advantages of a group without the dropped member
    base, _ = grpo.group_advantages(r, 4)
    assert torch.equal(a[4:], base[4:]) and not torch.allclose(a[:3], base[:3])                    # the untouched group is unchanged, the touched one is not
    assert torch.equal(grpo.group_advantages(r, 4, drop=torch.zeros(8, dtype=torch.bool))[0], base)   # an all-False mask is the old code path
    # sampled baseline (greedy last, guide at index 2 dropped): statistics over members 0 and 1 only, the greedy member relative to them, dropped member 0
    rr = torch.tensor([0.0, 1.0, -2.0, 4.0])
    b, _ = grpo.group_advantages_sampled_baseline(rr, 4, drop=torch.tensor([False, False, True, False]))
    mean, std = 0.5, 0.5
    assert b[2] == 0.0 and abs(float(b[0]) - (0.0 - mean) / (std + 1e-4)) < 1e-5 and abs(float(b[3]) - (4.0 - mean) / (std + 1e-4)) < 1e-4
    one, flat = grpo.group_advantages(torch.tensor([1.0, 2.0, 3.0, 4.0]), 4, drop=torch.tensor([True, True, True, False]))
    assert torch.equal(one, torch.zeros(4)) and flat == 1.0                                          # fewer than two members left: flat


def test_guide_share_cap_scales_only_when_the_share_is_above_the_cap():
    adv = torch.tensor([1.0, -1.0, 1.0, 3.0, -1.0, 1.0])
    lengths = torch.tensor([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    members = torch.tensor([False, False, False, True, False, False])
    same, share, factor = grpo.cap_member_share(adv, lengths, members, 0.6)
    assert torch.equal(same, adv) and factor == 1.0 and share == pytest.approx(3.0 / 8.0)
    out, share, factor = grpo.cap_member_share(adv, lengths, members, 0.25)
    w = out.abs() * lengths
    assert share == pytest.approx(0.375) and float(w[members].sum() / w.sum()) == pytest.approx(0.25) and torch.equal(out[~members], adv[~members]) and factor < 1.0
    assert torch.equal(grpo.cap_member_share(adv, lengths, members, 0.0)[0], adv) and torch.equal(grpo.cap_member_share(adv, lengths, torch.zeros(6, dtype=torch.bool), 0.1)[0], adv)
    # token weighting: a long guide counts for its length
    out2, share2, _ = grpo.cap_member_share(adv, torch.tensor([10.0, 10.0, 10.0, 40.0, 10.0, 10.0]), members, 0.25)
    assert share2 > share and float((out2.abs() * torch.tensor([10.0, 10.0, 10.0, 40.0, 10.0, 10.0]))[members].sum() / (out2.abs() * torch.tensor([10.0, 10.0, 10.0, 40.0, 10.0, 10.0])).sum()) == pytest.approx(0.25)


def test_config_validation_of_the_new_keys():
    from gemma_mt.rl.train_rl import RLConfig, validate_config

    cfg = RLConfig()
    assert cfg.adv_std_floor == 0.0 and cfg.log_group_stats is False and cfg.guide_fraction == 0.0 and cfg.guide_beams == 4
    validate_config(RLConfig(adv_std_floor=0.02, log_group_stats=True, guide_fraction=0.25, group_size=8))
    assert cfg.guide_adv_share_cap == 0.0
    validate_config(RLConfig(guide_fraction=0.25, guide_adv_share_cap=0.25, group_size=16, greedy_in_group=True, greedy_stats="sampled"))
    for bad in ({"adv_std_floor": -0.1}, {"guide_fraction": 1.5}, {"guide_fraction": 0.25, "guide_beams": 1}, {"guide_fraction": 0.25, "group_size": 3}, {"guide_adv_share_cap": 1.0},
                {"guide_fraction": 0.25, "group_size": 8, "hard_fail_mask": "zero"}):
        with pytest.raises(ValueError):
            validate_config(RLConfig(**bad))


def test_beam_completions_are_one_per_prompt_cut_at_stop_and_deterministic():
    from gemma_mt import format as fmt
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.modeling import build_tiny_model
    from gemma_mt.rl.sampling import generate_beam_completions

    tok = FakeGemmaTokenizer()
    model = build_tiny_model(seed=11, attn_implementation="sdpa").eval()
    prompts = [fmt.build_prompt("Good morning", "yor", tok), fmt.build_prompt("Water is life, and we all need it.", "hau", tok)]
    a = generate_beam_completions(model, prompts, num_beams=3, max_new_tokens=10, stop_ids=[106, 1], pad_id=0, device="cpu")
    b = generate_beam_completions(model, prompts, num_beams=3, max_new_tokens=10, stop_ids=[106, 1], pad_id=0, device="cpu")
    assert len(a.completions) == 2 and a.completions == b.completions and a.sampler_logp is None
    for c, f in zip(a.completions, a.finished):
        assert 1 <= len(c) <= 10 and f == (c[-1] in (106, 1)) and not any(t in (106, 1) for t in c[:-1])


def test_loop_with_floor_group_stats_and_guides_runs_all_in_loop_assertions():
    from gemma_mt.rl.train_rl import run_smoke

    s = run_smoke(sft_steps=60, overrides={"adv_std_floor": 0.02, "log_group_stats": True, "guide_fraction": 0.7, "guide_beams": 2, "group_size": 4, "guide_adv_share_cap": 0.01})
    rows = _rows(s["workdir"] + "/rl/metrics.jsonl")
    train = [r for r in rows if "reward" in r]
    assert len(train) == 4
    for r in train:
        assert math.isfinite(r["reward"]) and abs(r["ratio_mean"] - 1.0) < 1e-5            # on-policy ratio stays exactly 1 with a guide in the batch
        assert r["guide/n"] >= 1 and 0.0 <= r["guide/better_frac"] <= 1.0 and math.isfinite(r["guide/reward"]) and math.isfinite(r["group_std/p10"])
        assert 0.0 <= r["guide/best_frac"] <= 1.0 and r["guide/seconds"] > 0 and 0.0 < r["guide/overhead_frac"] < 1.0 and 0.0 < r["guide/adv_cap_factor"] <= 1.0 and r["guide/adv_share"] >= 0.0
    # (whether the cap bites in this tiny model depends on its groups being flat or not; the arithmetic is covered by test_guide_share_cap_scales_only_when_the_share_is_above_the_cap)
    table = json.load(open(s["workdir"] + "/rl/guide_by_language.json"))
    assert table and all(0.0 <= v["win_rate"] <= 1.0 and v["n"] >= 1 for v in table.values())
    for v in table.values():                         # guide vs MBR pick of the samples vs greedy vs group mean, and how often each is the best member
        assert all(0.0 <= v[k] <= 1.0 for k in ("best_rate", "best_rate_guide", "best_rate_mbr", "best_rate_greedy")) and v["n_mbr"] <= v["n"]
        assert all(math.isfinite(v[k]) for k in ("reward_guide", "reward_mbr", "reward_greedy", "reward_group_mean"))
    gs = _rows(s["workdir"] + "/rl/group_stats.jsonl")
    assert len(gs) == 4 * 3 and {"step", "prompt", "iso", "std", "mean", "n_fail", "guide"} <= set(gs[0]) and any(x["guide"] for x in gs)


def test_loop_with_guides_greedy_member_dynamic_sampling_and_everything_else_on():
    from gemma_mt.rl.train_rl import run_smoke

    s = run_smoke(sft_steps=60, overrides={"guide_fraction": 0.5, "guide_beams": 2, "group_size": 4, "greedy_in_group": True, "greedy_stats": "sampled", "greedy_adv_scale": 0.3,
                                           "dynamic_sampling": True, "dynamic_sampling_oversample": 1.5, "dynamic_sampling_min_std": 0.0, "adv_std_floor": 0.02, "log_group_stats": True})
    train = [r for r in _rows(s["workdir"] + "/rl/metrics.jsonl") if "reward" in r]
    assert len(train) == 4 and all(abs(r["ratio_mean"] - 1.0) < 1e-5 and r["ds/kept_groups"] <= 3 for r in train)
    assert os.path.exists(s["workdir"] + "/rl/group_stats.jsonl") and abs(train[0]["kl"]) < 1e-6


def test_off_by_default_changes_nothing():
    from gemma_mt.rl.train_rl import run_smoke

    a = run_smoke(sft_steps=60)
    b = run_smoke(sft_steps=60, overrides={"adv_std_floor": 0.0, "guide_fraction": 0.0, "log_group_stats": False})
    ra = [r["reward"] for r in _rows(a["workdir"] + "/rl/metrics.jsonl") if "reward" in r]
    rb = [r["reward"] for r in _rows(b["workdir"] + "/rl/metrics.jsonl") if "reward" in r]
    assert ra == rb and not os.path.exists(a["workdir"] + "/rl/group_stats.jsonl")
