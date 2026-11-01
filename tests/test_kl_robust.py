import torch

from gemma_mt.rl import grpo


def test_clamped_k3_equals_k3_in_the_normal_range():
    logp = torch.tensor([-1.0, -2.0, -0.5, -3.0])
    ref = logp + torch.tensor([0.3, -0.7, 2.0, 4.0])
    assert torch.allclose(grpo.kl_k3_clamped(logp, ref), grpo.kl_k3(logp, ref))


def test_clamped_k3_is_bounded_and_gradient_free_beyond_the_clamp():
    logp = torch.tensor([-30.0, -1.0], requires_grad=True)
    ref = torch.tensor([-2.0, -1.0])                     # d = 28 for the first token: plain k3 would be exp(28)
    k = grpo.kl_k3_clamped(logp, ref)
    assert float(k[0]) < 200 and float(grpo.kl_k3(logp.detach(), ref)[0]) > 1e11
    k.sum().backward()
    assert float(logp.grad[0]) == 0.0 and float(logp.grad.abs().max()) < 1e-6


def test_estimator_is_registered():
    assert "k3c" in grpo.KL_ESTIMATORS and "k3" in grpo.KL_ESTIMATORS
