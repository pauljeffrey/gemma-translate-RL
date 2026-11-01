# SPDX-License-Identifier: AGPL-3.0-or-later
"""attn_implementation="varlen" (D-35): the flattened-stream attention that replaces the explicit [T, T] masks.

CPU tests exercise the torch reference path of `varlen_attention_forward` (same plumbing: cu_seq_lens, sliding window,
scaling, GQA, padding removal by the flatten collator); the CUDA tests (skipped without a GPU) run the real
`torch.nn.attention.varlen.varlen_attn` FlashAttention kernel and compare it, in fp32 and under bf16 autocast, with the
explicit-mask path."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gemma_mt import attention as A
from gemma_mt.data.collate import PackedCollator
from gemma_mt.losses import sft_nll_sum
from gemma_mt.modeling import build_tiny_model, sliding_window_of

def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")


def _pack_item(lens, seed, vocab=500, pad_to=None, pad=True):
    """One dataset-style pack (what PackedDataset.__getitem__ returns) with samples of the given lengths."""
    g = np.random.default_rng(seed)
    ids = g.integers(4, vocab, sum(lens)).astype(np.int64)
    labels = ids.copy()
    c = 0
    for L in lens:
        labels[c: c + 4] = -100                     # prompt part
        c += L
    pos = np.concatenate([np.arange(L) for L in lens]).astype(np.int64)
    seg = np.concatenate([np.full(L, j + 1) for j, L in enumerate(lens)]).astype(np.int32)
    tiso = np.zeros(sum(lens), np.int16)
    cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    total = sum(lens)
    if pad and pad_to and total < pad_to:
        n = pad_to - total
        ids = np.concatenate([ids, np.zeros(n, np.int64)])
        labels = np.concatenate([labels, np.full(n, -100, np.int64)])
        pos = np.concatenate([pos, np.arange(n)])
        seg = np.concatenate([seg, np.zeros(n, np.int32)])
        tiso = np.concatenate([tiso, np.full(n, -1, np.int16)])
        cu = np.concatenate([cu, np.asarray([pad_to], np.int32)])
    return {"input_ids": ids, "labels": labels, "position_ids": pos, "segment_ids": seg, "cu_seqlens": cu,
            "token_iso": tiso, "length": total, "n_samples": len(lens)}


def _torchify(batch, device="cpu"):
    return {k: (torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v) for k, v in batch.items()}


ROWS = [[20, 31, 14], [40, 9], [25, 25, 12]]          # sample lengths per pack; the sliding window of the tiny model is 8
PAD_TO = 80


def _batches(device="cpu"):
    items = [_pack_item(r, i, pad_to=PAD_TO) for i, r in enumerate(ROWS)]
    padded = _torchify(PackedCollator("padded", pad_id=0).__call__(items), device)
    flat = _torchify(PackedCollator("flatten")(items), device)
    return padded, flat


def _losses(padded, flat, device="cpu", autocast=None, seed=11):
    """(logits-free) NLL sum and embedding / lm-head gradient of the explicit sdpa path and of the varlen path with the
    same random weights."""
    out = {}
    for name, impl, mode, b in (("explicit", "sdpa", "explicit", padded), ("varlen", "varlen", "flash", flat)):
        m = build_tiny_model(seed=seed, attn_implementation=impl).to(device).train()
        kw = A.model_inputs(b, attn_implementation=impl, mask_mode=mode, sliding_window=sliding_window_of(m))
        m.zero_grad()
        ctx = torch.autocast(device, dtype=autocast) if autocast else torch.autocast(device, enabled=False)
        with ctx:
            nll, n = sft_nll_sum(m, kw, b["labels"], chunk=13)
        nll.backward()
        out[name] = (float(nll.detach()), n, {k: p.grad.detach().clone() for k, p in m.named_parameters() if p.grad is not None})
    return out


def test_window_smaller_than_every_sample():
    assert sliding_window_of(build_tiny_model()) == 8 and min(min(r) for r in ROWS) > 8


def test_flatten_collator_removes_padding_and_covers_all_tokens():
    padded, flat = _batches()
    assert padded["input_ids"].shape == (3, PAD_TO)
    n = sum(sum(r) for r in ROWS)
    assert flat["input_ids"].shape == (1, n)
    cu = flat["cu_seqlens"]
    assert int(cu[0]) == 0 and int(cu[-1]) == n and cu.dtype == torch.int32
    assert [int(x) for x in np.diff(cu.numpy())] == [L for r in ROWS for L in r]       # each sample AND each pack boundary
    assert int(flat["max_seqlen"]) == 40
    assert flat["num_target_tokens"] == padded["num_target_tokens"]


def test_flatten_handles_full_packs_without_padding_and_unpadded_datasets():
    full = _pack_item([30, 30, 20], 5, pad_to=80)                    # exactly 80 tokens: no padding segment in cu_seqlens
    part = _pack_item([30, 17], 6, pad_to=80)
    unpadded = _pack_item([30, 17], 6, pad=False)                    # PackedDataset(pad=False) rows
    for items in ([full, part], [part, full], [unpadded, full]):
        flat = PackedCollator("flatten")(items)
        n = sum(int(it["length"]) for it in items)
        assert flat["input_ids"].shape == (1, n) and int(flat["cu_seqlens"][-1]) == n
        assert int(flat["max_seqlen"]) == 30 and len(flat["cu_seqlens"]) == 1 + sum(it["n_samples"] for it in items)
        assert (np.diff(flat["cu_seqlens"]) > 0).all()


def test_model_inputs_varlen_gives_cu_seqlens_and_no_mask():
    _, flat = _batches()
    kw = A.model_inputs(flat, attn_implementation="varlen", mask_mode="flash", sliding_window=8)
    assert kw["attention_mask"] == {"full_attention": None, "sliding_attention": None}
    assert kw["use_cache"] is False and kw["max_length_q"] == 40 and torch.equal(kw["cu_seq_lens_q"], kw["cu_seq_lens_k"])
    assert A.default_mask_mode("varlen") == "flash" and A.collator_mode_for("varlen") == "flatten"
    assert A.default_mask_mode("flex_attention") == "position_ids" and A.collator_mode_for("flex_attention") == "padded"
    assert A.default_mask_mode("sdpa") == "explicit" and A.collator_mode_for("sdpa") == "padded"
    with pytest.raises(ValueError):
        A.model_inputs(flat, attn_implementation="sdpa", mask_mode="flash", sliding_window=8)
    with pytest.raises(ValueError):
        A.model_inputs(flat, attn_implementation="varlen", mask_mode="explicit", sliding_window=8)


def test_varlen_without_cu_seqlens_fails_loudly():
    m = build_tiny_model(seed=1, attn_implementation="varlen").eval()
    with pytest.raises(ValueError, match="cu_seq_lens_q"):
        m(input_ids=torch.randint(4, 500, (1, 12)), use_cache=False, attention_mask={"full_attention": None, "sliding_attention": None})


def test_varlen_equals_separate_samples_cpu():
    m = build_tiny_model(seed=2, attn_implementation="varlen").eval()
    _, flat = _batches()
    kw = A.model_inputs(flat, attn_implementation="varlen", mask_mode="flash", sliding_window=8)
    with torch.no_grad():
        packed = m(**kw).logits[0]
        c, errs = 0, []
        for r in ROWS:
            for L in r:
                ids = flat["input_ids"][0, c: c + L][None]
                ref = m(input_ids=ids, use_cache=False, attention_mask={"full_attention": None, "sliding_attention": None},
                        cu_seq_lens_q=torch.tensor([0, L], dtype=torch.int32), cu_seq_lens_k=torch.tensor([0, L], dtype=torch.int32),
                        max_length_q=L, max_length_k=L).logits[0]
                errs.append((packed[c: c + L] - ref).abs().max().item())
                c += L
    assert max(errs) < 2e-5


def test_varlen_equals_explicit_masks_cpu_loss_and_gradients():
    padded, flat = _batches()
    r = _losses(padded, flat)
    (le, ne, ge), (lv, nv, gv) = r["explicit"], r["varlen"]
    assert ne == nv
    assert abs(le - lv) < 1e-4 * max(1.0, abs(le))
    for k in ge:
        scale = max(1.0, ge[k].abs().max().item())
        assert (ge[k] - gv[k]).abs().max().item() < 1e-4 * scale, k


def test_varlen_reference_matches_a_hand_written_softmax():
    """The reference path itself against explicit math: GQA (4 q heads, 1 kv head), scaling, window 5, two segments."""
    from gemma_mt.attention import _reference_attention

    torch.manual_seed(0)
    T, H, D = 17, 4, 8
    q, k, v = torch.randn(T, H, D), torch.randn(T, 1, D), torch.randn(T, 1, D)
    cu = torch.tensor([0, 9, 17], dtype=torch.int32)
    out = _reference_attention(q, k, v, cu, 0.3, 5)
    ref = torch.zeros(T, H, D)
    for t in range(T):
        lo = 0 if t < 9 else 9
        keys = [s for s in range(lo, t + 1) if t - s < 5]
        for h in range(H):
            sc = torch.stack([(q[t, h] * k[s, 0]).sum() * 0.3 for s in keys]).softmax(0)
            ref[t, h] = sum(w * v[s, 0] for w, s in zip(sc, keys))
    assert (out - ref).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------------------------------- CUDA
def test_kernel_matches_reference_attention_on_gpu():
    _need_cuda()
    from gemma_mt.attention import _reference_attention, varlen_attention_forward

    torch.manual_seed(0)
    T, H, Hk, D = 600, 4, 1, 64
    q = torch.randn(1, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, Hk, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, Hk, T, D, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, 200, 450, 600], dtype=torch.int32, device="cuda")

    class _M:                      # what the function reads from the module
        is_causal = True
        attn_logit_softcapping = None

    for window in (None, 70):
        out, _ = varlen_attention_forward(_M(), q, k, v, None, scaling=0.0625, sliding_window=window, cu_seq_lens_q=cu,
                                          cu_seq_lens_k=cu, max_length_q=250, max_length_k=250)
        ref = _reference_attention(q[0].transpose(0, 1).float(), k[0].transpose(0, 1).float(), v[0].transpose(0, 1).float(),
                                   cu, 0.0625, window)
        assert out.shape == (1, T, H, D) and out.dtype == torch.bfloat16
        assert (out[0].float() - ref).abs().max().item() < 3e-2


def test_varlen_kernel_equals_explicit_on_gpu_and_is_not_noisier_than_explicit_bf16():
    """Tiny random Gemma3 (window 8 < samples, GQA), bf16 autocast, the REAL kernel. Ground truth: explicit-mask sdpa in
    fp32. The varlen path's error against it must be of the size of the explicit path's own bf16 error."""
    _need_cuda()
    from gemma_mt import equiv

    padded, flat = _batches("cuda")
    m = build_tiny_model(seed=11, attn_implementation="sdpa").to("cuda").train()
    ref = equiv.run_path(m, padded, impl="sdpa", mode="explicit", autocast=None, logits=True)
    exp16 = equiv.run_path(m, padded, impl="sdpa", mode="explicit", autocast=torch.bfloat16, logits=True)
    var16 = equiv.run_path(m, flat, impl="varlen", mode="flash", autocast=torch.bfloat16, logits=True)
    var32 = equiv.run_path(m, flat, impl="varlen", mode="flash", autocast=None, logits=True)
    d_exp, d_var, d32 = equiv.compare(ref, exp16), equiv.compare(ref, var16), equiv.compare(ref, var32)
    assert var16["n"] == ref["n"]
    # fp32 (reference path of the varlen function): plumbing is exact
    assert d32["logits_max_abs"] < 1e-4 and d32["rel_l2"] < 1e-4
    # bf16: not worse than the explicit path's own bf16-vs-fp32 noise (factor 2 slack + a floor)
    for k in ("logits_max_abs", "logp_max_abs", "rel_l2"):
        assert d_var[k] < 2.0 * d_exp[k] + 1e-3, (k, d_var[k], d_exp[k])
    assert d_var["loss_rel"] < 1e-3
