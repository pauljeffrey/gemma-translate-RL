# SPDX-License-Identifier: AGPL-3.0-or-later
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gemma_mt.losses import completion_logprobs, sft_nll_sum, token_logprobs_chunked
from gemma_mt.modeling import build_tiny_model


def test_chunked_logprobs_equal_full_and_gradients_match_for_any_chunk():
    torch.manual_seed(0)
    h = torch.randn(37, 16, requires_grad=True)
    w = torch.randn(50, 16, requires_grad=True)
    t = torch.randint(0, 50, (37,))
    ref = torch.log_softmax(h @ w.T, -1).gather(-1, t[:, None]).squeeze(-1)
    ref.sum().backward()
    gh, gw = h.grad.clone(), w.grad.clone()
    for chunk in (1, 5, 37, 1000):
        h.grad = w.grad = None
        lp = token_logprobs_chunked(h, w, t, chunk=chunk)
        assert torch.allclose(lp, ref.detach(), atol=1e-5)
        lp.sum().backward()
        assert torch.allclose(h.grad, gh, atol=1e-4) and torch.allclose(w.grad, gw, atol=1e-4)


def test_temperature_and_softcap():
    h, w, t = torch.randn(6, 8), torch.randn(20, 8), torch.randint(0, 20, (6,))
    a = token_logprobs_chunked(h, w, t, temperature=0.5)
    b = torch.log_softmax((h @ w.T) / 0.5, -1).gather(-1, t[:, None]).squeeze(-1)
    assert torch.allclose(a, b, atol=1e-5)
    c = token_logprobs_chunked(h, w, t, softcap=3.0)
    d = torch.log_softmax(torch.tanh((h @ w.T) / 3.0) * 3.0, -1).gather(-1, t[:, None]).squeeze(-1)
    assert torch.allclose(c, d, atol=1e-5)


def test_sft_loss_matches_hf_loss_on_labelled_tokens():
    model = build_tiny_model(seed=1).eval()
    torch.manual_seed(1)
    ids = torch.randint(4, 500, (2, 17))
    labels = ids.clone()
    labels[:, :5] = -100
    labels[1, 12:] = -100
    with torch.no_grad():
        hf = model(input_ids=ids, labels=labels, use_cache=False).loss
        nll, n = sft_nll_sum(model, {"input_ids": ids, "use_cache": False}, labels, chunk=9)
    assert n == int((labels[:, 1:] != -100).sum())
    assert math.isclose(float(nll) / n, float(hf), rel_tol=1e-4, abs_tol=1e-5)


def test_completion_logprobs_alignment_with_left_padded_prompts():
    """Each row of a left-padded / right-padded batch must give the same per-token log-probs as running that
    sequence alone, without any padding. Catches off-by-one and padding leakage."""
    model = build_tiny_model(seed=2).eval()
    torch.manual_seed(2)
    prompts = [torch.randint(4, 500, (n,)).tolist() for n in (5, 9, 3)]
    comps = [torch.randint(4, 500, (n,)).tolist() for n in (4, 2, 6)]
    from gemma_mt.rl.sampling import assemble_batch

    b = assemble_batch(prompts, comps, pad_id=0)
    with torch.no_grad():
        lp = completion_logprobs(model, b["input_ids"], b["attention_mask"], b["prompt_width"], b["completion_mask"], 1.0, chunk=4)
        for r, (p, c) in enumerate(zip(prompts, comps)):
            ids = torch.tensor([p + c])
            logits = model(input_ids=ids, use_cache=False).logits[0]
            ref = torch.log_softmax(logits, -1)
            for j, tok in enumerate(c):
                want = ref[len(p) + j - 1, tok].item()
                assert abs(lp[r, j].item() - want) < 2e-5, (r, j)
            assert (lp[r, len(c):] == 0).all()
    # completion mask counts
    assert b["completion_mask"].sum().item() == sum(len(c) for c in comps)
    assert b["completion_mask"].tolist()[0] == [1, 1, 1, 1, 0, 0]


def test_completion_logprobs_temperature():
    model = build_tiny_model(seed=3).eval()
    p, c = [[2, 10, 11, 12]], [[20, 21, 22]]
    from gemma_mt.rl.sampling import assemble_batch

    b = assemble_batch(p, c, 0)
    with torch.no_grad():
        a = completion_logprobs(model, b["input_ids"], b["attention_mask"], b["prompt_width"], b["completion_mask"], 0.7)
        logits = model(input_ids=torch.tensor([p[0] + c[0]]), use_cache=False).logits[0] / 0.7
        ref = torch.log_softmax(logits, -1)
    for j, tok in enumerate(c[0]):
        assert abs(a[0, j].item() - ref[len(p[0]) + j - 1, tok].item()) < 2e-5
