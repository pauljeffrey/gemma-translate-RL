# Review of the varlen attention + fused loss speed-up (stopped agent's code, reviewed 2026-10-19)

Status: implemented in the laptop repo (NOT on the rental's live `gemma_mt`, NOT used by the running SFT). The agent was stopped before it wrote its report and its DECISIONS entry (D-35 is referenced in the code, but not written).

What it is
* `attn_implementation="varlen"` (`gemma_mt/attention.py`): our own attention function registered with transformers' `AttentionInterface`; it calls `torch.nn.attention.varlen.varlen_attn` (FlashAttention kernel inside torch, no flash-attn package) on a flattened `[1, N]` stream with `cu_seq_lens`, `window_size=(sliding_window-1, 0)` on the local layers, no [T, T] masks. Needs `mask_mode: flash` and the `flatten` collator. fp32 / CPU use a per-sample pure-torch reference path.
* `loss_impl: fused` (`gemma_mt/fused_loss.py`): chunked cross-entropy over the 262k vocabulary without materialising fp32 logit tensors.
* Tests: `tests/test_varlen_attention.py`, `tests/test_fused_loss.py`; whole suite 288 pass / 7 skipped on the laptop (GPU tests skip).

Evidence I reproduced on the old box (RTX 5070, shared with other sessions' jobs, real google/gemma-3-270m-it, real packs 0 and 1 of gemma/packed_fwd, 5,430 target tokens)
| candidate vs explicit-mask fp32 truth | gnorm rel. error | max abs logp error | mean abs logp error | loss rel. error |
|---|---|---|---|---|
| explicit bf16 (what we train with today) | 8.1e-2 | 2.00 | 0.0446 | 2.2e-4 |
| varlen bf16 | 4.0e-3 | 1.27 | 0.0416 | 2.7e-4 |
| varlen fp32 reference path | 3.8e-7 | 1.1e-4 | 4.8e-6 | 0 |
=> in fp32 the varlen path is the same function as the explicit-mask path (1e-6); in bf16 its error against the fp32 truth is not larger than the explicit path's own bf16 noise.

Speed (fwd+bwd of micro 2, gradient checkpointing on; the card was shared, so absolute times are inflated, ratios are what counts)
| case | ms | tok/s | peak GiB |
|---|---|---|---|
| sdpa explicit masks + chunked loss (today) | 2737 | 2,993 | 4.20 |
| varlen + chunked loss | 1443 | 5,678 | 4.14 |
| varlen + fused loss | 899 | 9,118 | 2.73 |
=> 1.9x from the attention alone, 3.0x with the fused loss; lower memory.

Not verified: a full 30-step training comparison of loss curves, speed on the 96 GB card at micro 16, behaviour of an exact resume from the sdpa checkpoint with the new kernels (different bf16 rounding, same data order), the tiny-model GPU test (it ran out of memory because other jobs were using the card).
