# SPDX-License-Identifier: AGPL-3.0-or-later
import math

import pytest

torch = pytest.importorskip("torch")

from gemma_mt.rl import grpo


def test_advantages_known_numbers():
    r = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 10.0, 10.0])
    adv, flat = grpo.group_advantages(r, 4, "mean")
    assert adv[:4].tolist() == pytest.approx([-1.5, -0.5, 0.5, 1.5])
    assert adv[4:].tolist() == [0.0] * 4
    assert flat == 0.5
    adv, _ = grpo.group_advantages(r, 4, "mean_std", std_eps=0.0)
    std = math.sqrt(1.25)                                    # population std of 1,2,3,4
    assert adv[:4].tolist() == pytest.approx([-1.5 / std, -0.5 / std, 0.5 / std, 1.5 / std], rel=1e-6)
    assert adv[:4].sum().abs() < 1e-6, "advantages are zero-mean inside a group"
    with pytest.raises(ValueError):
        grpo.group_advantages(torch.ones(5), 4)


def test_groups_are_consecutive_rows():
    r = torch.tensor([0.0, 1.0, 5.0, 7.0])
    adv, _ = grpo.group_advantages(r, 2, "mean")
    assert adv.tolist() == [-0.5, 0.5, -1.0, 1.0]


def test_kl_estimator_values():
    lp = torch.tensor([-1.0, -2.0])
    ref = torch.tensor([-1.5, -1.0])
    d = ref - lp
    assert torch.allclose(grpo.kl_k3(lp, ref), torch.exp(d) - d - 1)
    assert grpo.kl_k3(lp, lp).abs().max() == 0
    assert (grpo.kl_k3(lp, ref) >= 0).all()
    assert torch.allclose(grpo.kl_k2(lp, ref), 0.5 * d ** 2)


def test_k3_is_unbiased_for_reverse_kl_and_gradients_have_the_documented_meaning():
    """Exact expectations over a 5-token vocabulary (no sampling noise)."""
    torch.manual_seed(0)
    z = torch.randn(5, requires_grad=True)             # policy logits
    ref_logp = torch.log_softmax(torch.randn(5), -1)
    logp = torch.log_softmax(z, -1)
    p = logp.detach().exp()
    rev = (p * (logp.detach() - ref_logp)).sum()       # KL(pi || ref)
    # value: E_{x~pi}[k3(x)] == KL(pi || ref)
    k3_val = (p * grpo.kl_k3(logp.detach(), ref_logp)).sum()
    assert abs(float(k3_val - rev)) < 1e-6
    # gradient of E_{x~pi (held fixed)}[k] w.r.t. z
    def grad_of(est):
        z.grad = None
        lp = torch.log_softmax(z, -1)
        loss = (p * est(lp, ref_logp)).sum()           # samples (weights p) are NOT differentiated
        loss.backward()
        return z.grad.clone()

    z.grad = None
    kl_rev = (torch.log_softmax(z, -1).exp() * (torch.log_softmax(z, -1) - ref_logp)).sum()
    kl_rev.backward()
    g_rev = z.grad.clone()
    z.grad = None
    kl_fwd = (ref_logp.exp() * (ref_logp - torch.log_softmax(z, -1))).sum()
    kl_fwd.backward()
    g_fwd = z.grad.clone()
    g_k2, g_k3 = grad_of(grpo.kl_k2), grad_of(grpo.kl_k3)
    assert torch.allclose(g_k2, g_rev, atol=1e-6), "k2's gradient is the reverse-KL gradient"
    assert torch.allclose(g_k3, g_fwd, atol=1e-6), "k3's gradient is the FORWARD-KL gradient (GRPO/DeepSeekMath)"


def test_loss_masking_and_normalisation():
    logp = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -0.5, -0.5]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    adv = torch.tensor([2.0, -1.0])
    out = grpo.grpo_token_loss(logp, adv, mask, normaliser=3.0)
    # pg = -A * ratio, ratio == 1 -> sum over masked tokens: -(2*1 + 2*1 + -1*1) = -3; /3 = -1
    assert float(out["loss"]) == pytest.approx(-1.0)
    out["loss"].backward()
    assert logp.grad[0].tolist() == pytest.approx([-2.0 / 3, -2.0 / 3, 0.0])     # d(-A*lp)/dlp = -A, / normaliser
    assert logp.grad[1].tolist() == pytest.approx([1.0 / 3, 0.0, 0.0])
    assert float(out["n_tokens"]) == 3 and float(out["ratio_sum"]) == pytest.approx(3.0)


def test_sequence_aggregation_and_kl_term():
    logp = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    ref = torch.tensor([[-1.0, -1.0], [-2.0, -2.0]])
    mask = torch.ones(2, 2)
    out = grpo.grpo_token_loss(logp, torch.zeros(2), mask, ref_logp=ref, kl_coef=0.1, loss_agg="sequence", normaliser=2.0)
    k = grpo.kl_k3(logp, ref)                                       # row 1: exp(-1)+1-1 = e^-1
    want = 0.1 * (k[0].mean() + k[1].mean()) / 2
    assert float(out["loss"]) == pytest.approx(float(want))
    off = grpo.grpo_token_loss(logp, torch.zeros(2), mask, ref_logp=ref, kl_coef=0.0, normaliser=4.0)
    assert float(off["loss"]) == 0.0


def test_ratio_is_exactly_one_for_the_single_update_case_and_clipping_only_when_needed():
    logp = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    m = torch.ones(1, 2)
    out = grpo.grpo_token_loss(logp, torch.tensor([1.0]), m, clip_eps=0.2, normaliser=2.0)
    assert float(out["ratio_sum"]) == pytest.approx(2.0) and float(out["clip_sum"]) == 0.0
    old = torch.tensor([[-1.0, -4.0]])            # second token moved a lot: ratio e^2
    out = grpo.grpo_token_loss(logp, torch.tensor([1.0]), m, old_logp=old, clip_eps=0.2, normaliser=2.0)
    assert float(out["clip_sum"]) == 1.0
    out["loss"].backward()
    assert logp.grad[0, 1].item() == 0.0            # clipped token carries no gradient for A > 0
