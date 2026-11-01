import pytest

from gemma_mt.rl.sampling import Rollout, merge_greedy_into_groups
from gemma_mt.rl.train_rl import RLConfig, validate_config


def _roll(rows, per, tag):
    n = rows
    return Rollout(prompts=[[i // per] for i in range(n)], completions=[[tag, i] for i in range(n)], finished=[True] * n,
                   sampler_logp=[[-0.1 * (i + 1)] * 2 for i in range(n)], post_stop_tokens=[0] * n)


def test_greedy_is_the_last_row_of_every_group():
    s, g = _roll(6, 3, 1), _roll(2, 1, 9)
    m = merge_greedy_into_groups(s, g, 3)
    assert len(m.completions) == 8
    assert [c[0] for c in m.completions] == [1, 1, 1, 9, 1, 1, 1, 9]
    assert [c[1] for c in m.completions] == [0, 1, 2, 0, 3, 4, 5, 1]
    assert [p[0] for p in m.prompts] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert len(m.sampler_logp) == 8 and len(m.post_stop_tokens) == 8


def test_missing_logprobs_stay_none():
    s, g = _roll(4, 2, 1), _roll(2, 1, 9)
    s.sampler_logp = None
    assert merge_greedy_into_groups(s, g, 2).sampler_logp is None


def test_wrong_row_count_raises():
    with pytest.raises(ValueError):
        merge_greedy_into_groups(_roll(5, 3, 1), _roll(2, 1, 9), 3)


def test_config_needs_group_of_three():
    cfg = RLConfig(group_size=2, greedy_in_group=True, select_by="reward")
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_greedy_advantage_scale_touches_only_the_last_member_and_is_validated():
    import torch

    from gemma_mt.rl import grpo

    adv = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5])
    assert torch.equal(grpo.scale_last_member(adv, 3, 1.0), adv)
    out = grpo.scale_last_member(adv, 3, 0.1)
    assert torch.allclose(out, torch.tensor([1.0, -1.0, 0.2, -2.0, 0.5, -0.05]))
    assert torch.equal(adv, torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5, -0.5]))            # input not modified
    with pytest.raises(ValueError):
        grpo.scale_last_member(adv, 4, 0.1)
    validate_config(RLConfig(group_size=8, greedy_in_group=True, greedy_adv_scale=0.1))
    with pytest.raises(ValueError):
        validate_config(RLConfig(group_size=8, greedy_in_group=False, greedy_adv_scale=0.1))
    with pytest.raises(ValueError):
        validate_config(RLConfig(group_size=8, greedy_in_group=True, greedy_adv_scale=1.5))


def test_sampled_baseline_advantages_keep_the_sampled_members_untouched_by_a_looping_greedy_member():
    import torch

    from gemma_mt.rl import grpo

    # one group of 5: four sampled members + a greedy member that loops (reward -2)
    r = torch.tensor([0.5, 0.7, 0.3, 0.6, -2.0])
    a_all, _ = grpo.group_advantages(r, 5)
    a_s, flat = grpo.group_advantages_sampled_baseline(r, 5)
    a_ref, _ = grpo.group_advantages(r[:4], 4)                         # the group WITHOUT the greedy member
    assert torch.allclose(a_s[:4], a_ref, atol=1e-6) and flat == 0.0
    assert a_all[:4].abs().mean() < a_s[:4].abs().mean() * 0.8          # the all-member std shrinks the sampled advantages (R8)
    assert a_s[4] < -5                                                  # greedy member relative to the sampled statistics: (-2 - 0.525) / 0.14
    capped, _ = grpo.group_advantages_sampled_baseline(r, 5, greedy_cap=3.0)
    assert capped[4].item() == -3.0 and torch.equal(capped[:4], a_s[:4])
    # flat sampled members -> the whole group is flat, including the greedy member
    f, fr = grpo.group_advantages_sampled_baseline(torch.tensor([0.5, 0.5, 0.5, 0.5, -2.0]), 5)
    assert torch.count_nonzero(f) == 0 and fr == 1.0
    # mean norm, two groups, greedy better than the sampled mean
    m, _ = grpo.group_advantages_sampled_baseline(torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]), 4, norm="mean")
    assert torch.allclose(m, torch.tensor([-1.0, 0.0, 1.0, 2.0] * 2))
    with pytest.raises(ValueError):
        grpo.group_advantages_sampled_baseline(torch.tensor([1.0, 2.0]), 2)
    validate_config(RLConfig(group_size=8, greedy_in_group=True, greedy_stats="sampled", greedy_adv_cap=3.0, greedy_adv_scale=0.2))
    with pytest.raises(ValueError):
        validate_config(RLConfig(group_size=8, greedy_in_group=False, greedy_stats="sampled"))
    with pytest.raises(ValueError):
        validate_config(RLConfig(group_size=8, greedy_in_group=True, greedy_stats="bogus"))
