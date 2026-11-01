# SPDX-License-Identifier: AGPL-3.0-or-later
"""loss_impl="fused" (D-36): fused LM-head log-prob vs the reference chunked implementation. CPU tests run the same
autograd Function with torch ops instead of the Triton kernels (algebra of forward / backward / chunking); the CUDA tests
(skipped without a GPU) run the Triton kernels."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gemma_mt.fused_loss import fused_linear_logp
from gemma_mt.losses import sft_nll_sum, token_logprobs_chunked
from gemma_mt.modeling import build_tiny_model


def _data(N=37, H=16, V=50, device="cpu", dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(N, H, generator=g).to(device, dtype).requires_grad_()
    w = torch.randn(V, H, generator=g).to(device, dtype).requires_grad_()
    t = torch.randint(0, V, (N,), generator=g).to(device)
    return h, w, t


def _grads(fn, h, w, t, weights):
    h.grad = w.grad = None
    out = fn(h, w, t)
    (out * weights).sum().backward()
    return out.detach(), h.grad.clone(), w.grad.clone()


@pytest.mark.parametrize("chunk", [1, 5, 37, 1000])
def test_fused_equals_reference_fp32_for_any_chunk(chunk):
    h, w, t = _data()
    weights = torch.linspace(0.2, 2.0, 37)                     # a non-uniform upstream gradient (policy-gradient-like)
    a = _grads(lambda h, w, t: token_logprobs_chunked(h, w, t, chunk=7), h, w, t, weights)
    b = _grads(lambda h, w, t: fused_linear_logp(h, w, t, chunk=chunk), h, w, t, weights)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-5, rtol=1e-5)


def test_fused_temperature_and_softcap_fallback():
    h, w, t = _data(N=6, H=8, V=20)
    a = token_logprobs_chunked(h, w, t, temperature=0.5)
    b = token_logprobs_chunked(h, w, t, temperature=0.5, impl="fused")
    assert torch.allclose(a, b, atol=1e-5)
    c = token_logprobs_chunked(h, w, t, softcap=3.0)
    d = token_logprobs_chunked(h, w, t, softcap=3.0, impl="fused")                 # silently the reference path
    assert torch.equal(c, d)
    with pytest.raises(ValueError):
        token_logprobs_chunked(h, w, t, impl="nope")


def test_fused_empty_and_no_grad():
    h, w, t = _data(N=0)
    assert fused_linear_logp(h, w, t).shape == (0,)
    h, w, t = _data()
    with torch.no_grad():
        assert torch.allclose(fused_linear_logp(h, w, t, chunk=8), token_logprobs_chunked(h, w, t), atol=1e-5)


def test_fused_under_cpu_bf16_autocast_is_close_to_the_reference():
    h, w, t = _data(N=40, H=32, V=300)
    weights = torch.ones(40)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        a = _grads(lambda h, w, t: token_logprobs_chunked(h, w, t, chunk=16), h, w, t, weights)
        b = _grads(lambda h, w, t: fused_linear_logp(h, w, t, chunk=16), h, w, t, weights)
    assert (a[0] - b[0]).abs().max() < 2e-2
    for x, y in zip(a[1:], b[1:]):
        assert (x - y).norm() / x.norm() < 3e-2


def test_sft_nll_sum_fused_equals_chunked_on_a_model_with_gradients():
    torch.manual_seed(0)
    ids = torch.randint(4, 500, (2, 30))
    labels = ids.clone()
    labels[:, :6] = -100
    res = []
    for impl in ("chunked", "fused"):
        m = build_tiny_model(seed=3, attn_implementation="sdpa").train()
        nll, n = sft_nll_sum(m, {"input_ids": ids, "use_cache": False}, labels, chunk=11, loss_impl=impl)
        nll.backward()
        res.append((float(nll), n, m.model.embed_tokens.weight.grad.clone(), m.model.layers[0].self_attn.q_proj.weight.grad.clone()))
    assert res[0][1] == res[1][1] and abs(res[0][0] - res[1][0]) < 1e-3
    for x, y in zip(res[0][2:], res[1][2:]):
        assert (x - y).abs().max() < 1e-4 * max(1.0, x.abs().max().item())


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")


def test_triton_kernels_match_reference_on_gpu():
    _need_cuda()
    h, w, t = _data(N=130, H=64, V=50021, device="cuda", dtype=torch.float32)      # V not a multiple of the Triton block
    weights = torch.linspace(0.1, 2.0, 130, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        a = _grads(lambda h, w, t: token_logprobs_chunked(h, w, t, chunk=64), h, w, t, weights)
        b = _grads(lambda h, w, t: fused_linear_logp(h, w, t, chunk=48), h, w, t, weights)
    assert (a[0] - b[0]).abs().max().item() < 2e-2
    for x, y in zip(a[1:], b[1:]):
        assert ((x - y).norm() / x.norm()).item() < 2e-2
    # fp32 path (no autocast): the kernels read fp32 logits
    a = _grads(lambda h, w, t: token_logprobs_chunked(h, w, t, chunk=64), h, w, t, weights)
    b = _grads(lambda h, w, t: fused_linear_logp(h, w, t, chunk=48), h, w, t, weights)
    assert (a[0] - b[0]).abs().max().item() < 1e-4
    for x, y in zip(a[1:], b[1:]):
        assert ((x - y).norm() / x.norm()).item() < 1e-4
