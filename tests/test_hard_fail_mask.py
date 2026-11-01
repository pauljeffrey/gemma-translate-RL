# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-61: the OFF-by-default hard_fail_mask of the GRPO trainer (grpo.group_advantages_masked) and what the standard rule does to a -2 sample."""
import pytest
import torch

from gemma_mt.rl import grpo


def test_a_failed_sample_keeps_a_strong_negative_advantage_under_the_standard_rule():
    r = torch.tensor([-2.0, 0.4, 0.5, 0.6, 0.5, 0.3, 0.7, 0.5])
    adv, flat = grpo.group_advantages(r, 8)
    assert adv[0] < -2.0 and (adv[1:] > -0.5).all() and flat == 0.0                    # not removed: the failure is pushed down hardest


def test_zero_mode_zeroes_failed_members_and_keeps_the_survivors_advantages():
    r = torch.tensor([-2.0, 0.4, 0.5, 0.6, 0.5, 0.3, 0.7, 0.5])
    failed = torch.tensor([True] + [False] * 7)
    std, _ = grpo.group_advantages(r, 8)
    adv, flat = grpo.group_advantages_masked(r, 8, failed, "zero")
    assert adv[0] == 0.0 and torch.equal(adv[1:], std[1:])


def test_exclude_mode_recomputes_baseline_and_std_over_the_survivors():
    r = torch.tensor([-2.0, 0.4, 0.5, 0.6, 0.5, 0.3, 0.7, 0.5])
    failed = torch.tensor([True] + [False] * 7)
    adv, flat = grpo.group_advantages_masked(r, 8, failed, "exclude")
    assert adv[0] == 0.0
    surv, _ = grpo.group_advantages(r[1:].repeat(1)[:7].new_tensor([0.4, 0.5, 0.6, 0.5, 0.3, 0.7, 0.5]), 7)
    assert torch.allclose(adv[1:], surv, atol=1e-6) and abs(float(adv[1:].sum())) < 1e-5


def test_groups_with_fewer_than_two_survivors_are_flat_and_other_groups_are_untouched():
    r = torch.tensor([-2.0, -2.0, -2.0, 0.5, 0.1, 0.9, 0.3, 0.6])
    failed = torch.tensor([True, True, True, False, False, False, False, False])
    adv, flat = grpo.group_advantages_masked(r, 4, failed, "exclude")
    assert torch.equal(adv[:4], torch.zeros(4)) and adv[4:].abs().sum() > 0 and flat == pytest.approx(0.5)


def test_bad_mode():
    with pytest.raises(ValueError):
        grpo.group_advantages_masked(torch.zeros(4), 4, torch.zeros(4, dtype=torch.bool), "nope")
