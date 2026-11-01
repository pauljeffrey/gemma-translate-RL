# SPDX-License-Identifier: AGPL-3.0-or-later
"""THE key test: a packed row of samples A, B, C gives the same logits / loss / gradients as A, B, C run
separately, for every masking path, with a sliding window SMALLER than the samples; and without masking it does not."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gemma_mt import attention as A
from gemma_mt.losses import sft_nll_sum
from gemma_mt.modeling import build_tiny_model, sliding_window_of

LENS = [20, 31, 14]            # all longer than the sliding window (8) except none
TOL = 2e-5


def _samples(seed=0, vocab=500):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(4, vocab, (L,), generator=g) for L in LENS]


def _packed(samples, pad_to=None):
    ids = torch.cat(samples)
    pos = torch.cat([torch.arange(len(s)) for s in samples])
    seg = torch.cat([torch.full((len(s),), i + 1, dtype=torch.int32) for i, s in enumerate(samples)])
    if pad_to and pad_to > len(ids):
        n = pad_to - len(ids)
        ids = torch.cat([ids, torch.zeros(n, dtype=torch.long)])
        pos = torch.cat([pos, torch.arange(n)])
        seg = torch.cat([seg, torch.zeros(n, dtype=torch.int32)])
    return {"input_ids": ids[None], "position_ids": pos[None], "segment_ids": seg[None]}


def _max_err(model, impl, mode, samples, pad_to=None):
    sw = sliding_window_of(model)
    batch = _packed(samples, pad_to)
    kw = A.model_inputs(batch, attn_implementation=impl, mask_mode=mode, sliding_window=sw)
    with torch.no_grad():
        packed = model(**kw).logits[0]
        sep = [model(input_ids=s[None], use_cache=False).logits[0] for s in samples]
    errs, c = [], 0
    for s, ref in zip(samples, sep):
        errs.append((packed[c: c + len(s)] - ref).abs().max().item())
        c += len(s)
    return max(errs)


def test_window_is_smaller_than_samples_so_the_test_is_meaningful():
    m = build_tiny_model()
    assert sliding_window_of(m) < min(LENS)
    assert "sliding_attention" in m.config.layer_types and "full_attention" in m.config.layer_types


@pytest.mark.parametrize("impl,mode", [("eager", "explicit"), ("sdpa", "explicit"), ("eager", "position_ids"),
                                       ("sdpa", "position_ids")])
def test_packed_equals_separate(impl, mode):
    model = build_tiny_model(seed=1, attn_implementation=impl).eval()
    assert _max_err(model, impl, mode, _samples()) < TOL


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_padding_tail_does_not_change_results(impl):
    model = build_tiny_model(seed=2, attn_implementation=impl).eval()
    assert _max_err(model, impl, "explicit", _samples(1), pad_to=96) < TOL


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_without_masking_the_samples_contaminate_each_other(impl):
    model = build_tiny_model(seed=3, attn_implementation=impl).eval()
    err = _max_err(model, impl, "none", _samples(2))
    assert err > 1e-2, f"negative control should differ, got {err}"


def test_reset_positions_alone_are_not_enough():
    """Positions restarted but a plain causal mask: later samples still see earlier ones."""
    model = build_tiny_model(seed=4, attn_implementation="sdpa").eval()
    samples = _samples(3)
    batch = _packed(samples)
    T = batch["input_ids"].shape[1]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))[None, None]
    with torch.no_grad():
        packed = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"],
                       attention_mask={"full_attention": causal, "sliding_attention": causal}, use_cache=False).logits[0]
        ref = model(input_ids=samples[1][None], use_cache=False).logits[0]
    off = LENS[0]
    assert (packed[off: off + LENS[1]] - ref).abs().max().item() > 1e-2


def test_cache_must_be_off_for_position_id_derived_masks():
    """Documented trap: with use_cache=True transformers silently skips the packed-sequence detection."""
    model = build_tiny_model(seed=5, attn_implementation="sdpa").eval()
    samples = _samples(4)
    batch = _packed(samples)
    with torch.no_grad():
        on = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"], use_cache=True).logits[0]
        ref = model(input_ids=samples[1][None], use_cache=False).logits[0]
    assert (on[LENS[0]: LENS[0] + LENS[1]] - ref).abs().max().item() > 1e-2
    kw = A.model_inputs(batch, attn_implementation="sdpa", mask_mode="position_ids", sliding_window=8)
    assert kw["use_cache"] is False


def test_packed_loss_and_gradients_equal_sum_of_separate():
    model = build_tiny_model(seed=6, attn_implementation="sdpa")
    model.train()
    samples = _samples(5)
    # label every token after the first 4 of each sample
    labels_each = []
    for s in samples:
        l = s.clone()
        l[:4] = -100
        labels_each.append(l)
    batch = _packed(samples)
    labels = torch.cat(labels_each)[None]
    kw = A.model_inputs(batch, attn_implementation="sdpa", mask_mode="explicit", sliding_window=8)
    model.zero_grad()
    nll, n = sft_nll_sum(model, kw, labels, chunk=7)
    nll.backward()
    g_packed = model.model.embed_tokens.weight.grad.clone()
    model.zero_grad()
    tot, n_sep = 0.0, 0
    for s, l in zip(samples, labels_each):
        nl, nn = sft_nll_sum(model, {"input_ids": s[None], "use_cache": False}, l[None], chunk=7)
        nl.backward()
        tot += float(nl)
        n_sep += nn
    assert n == n_sep
    assert abs(float(nll) - tot) < 1e-3 * max(1.0, abs(tot))
    g_sep = model.model.embed_tokens.weight.grad
    assert (g_packed - g_sep).abs().max().item() < 1e-4 * max(1.0, g_sep.abs().max().item())


def test_flash_path_arguments_match_transformers_helper():
    fa = pytest.importorskip("transformers.modeling_flash_attention_utils")
    batch = _packed(_samples(6))
    cu = torch.tensor([0] + list(np.cumsum(LENS)), dtype=torch.int32)
    batch.update({"cu_seqlens": cu, "max_seqlen": max(LENS)})
    kw = A.model_inputs(batch, attn_implementation="flash_attention_2", mask_mode="flash", sliding_window=8)
    (cq, ck), (mq, mk) = fa.prepare_fa_kwargs_from_position_ids(batch["position_ids"])
    assert torch.equal(kw["cu_seq_lens_q"].cpu(), cq.cpu().to(torch.int32))
    assert kw["max_length_q"] == int(mq) == max(LENS)
    assert kw["attention_mask"] is None and kw["use_cache"] is False
    import inspect
    params = inspect.signature(fa._flash_attention_forward).parameters
    for name in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "position_ids", "sliding_window"):
        assert name in params, f"installed transformers does not accept {name}"
    with pytest.raises(ValueError):
        A.model_inputs({**batch, "input_ids": torch.cat([batch["input_ids"]] * 2)}, attn_implementation="flash_attention_2",
                       mask_mode="flash", sliding_window=8)
