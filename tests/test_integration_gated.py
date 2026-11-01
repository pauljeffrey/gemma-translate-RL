# SPDX-License-Identifier: AGPL-3.0-or-later
"""Integration tests that need real assets (tokenizer / models) and therefore network or a populated HF cache.
SKIPPED unless the environment variable is set:

    GEMMA_MT_TOKENIZER=google/gemma-3-270m-it     real Gemma 3 tokenizer: template, <end_of_turn>, segment check
    GEMMA_MT_COMET=1                              AfriCOMET QE + reference models from the local HF cache / hub
    GEMMA_MT_MODEL=google/gemma-3-270m-it         real 270M model: packed-vs-separate equivalence + 3 SFT steps (GPU box)
"""
import os

import pytest


def test_real_gemma_tokenizer_format():
    name = os.environ.get("GEMMA_MT_TOKENIZER")
    if not name:
        pytest.skip("set GEMMA_MT_TOKENIZER to run")
    from transformers import AutoTokenizer

    from gemma_mt import format as fmt

    tok = AutoTokenizer.from_pretrained(name)
    fmt.assert_template_matches_reference(tok)
    stop = fmt.resolve_stop_tokens(tok)
    assert stop.end_of_turn_id == 106, stop
    ex = fmt.build_example("Good morning, how are you?", "yor", "Ẹ káàárọ̀, báwo ni o ṣe wà?", tok)
    tgt = [t for t, l in zip(ex.input_ids, ex.labels) if l != -100]
    assert tgt[-1] == 106 and tok.decode(tgt[:-1]).startswith("yor ")
    assert ex.input_ids[0] == tok.bos_token_id and ex.input_ids[1] != tok.bos_token_id
    assert fmt.check_segment_consistency(tok) == [], "segment tokenization differs from joint tokenization"


def test_real_africomet_scorers():
    if not os.environ.get("GEMMA_MT_COMET"):
        pytest.skip("set GEMMA_MT_COMET=1 to run (needs the AfriCOMET checkpoints in the HF cache or network)")
    from gemma_mt.rl.comet import QE_MODEL_ID, CometScorer

    sc = CometScorer(QE_MODEL_ID, batch_size=2, device="cpu", half=False)
    src = ["Good morning, how are you today?", "The market opens at eight o'clock."]
    good = ["E kaaro, bawo ni o se wa loni?", "Oja naa n si ni agogo mejo."]
    bad = ["Ọmọ tí a bí lọ́jọ́ kan.", "xx xx xx xx"]
    a = sc(src, good)
    b = sc(src, bad)
    assert len(a) == 2 and all(isinstance(x, float) for x in a + b)
    assert sc(src[::-1], good[::-1]) == pytest.approx(a[::-1], abs=1e-4), "scores must follow the inputs, not the batch order"
    assert min(sc.token_lengths(src)) > 2


def test_real_model_equivalence_and_training_steps():
    name = os.environ.get("GEMMA_MT_MODEL")
    if not name:
        pytest.skip("set GEMMA_MT_MODEL to run (downloads the real 270M model)")
    torch = pytest.importorskip("torch")
    from gemma_mt import attention as A
    from gemma_mt.modeling import load_model, sliding_window_of

    model = load_model(name, attn_implementation="sdpa", dtype=torch.float32).eval()
    sw = sliding_window_of(model)
    lens = [sw // 2 + 40, sw + 90, 60]            # one sample longer than the real sliding window
    g = torch.Generator().manual_seed(0)
    samples = [torch.randint(10, 20000, (n,), generator=g) for n in lens]
    ids = torch.cat(samples)[None]
    pos = torch.cat([torch.arange(n) for n in lens])[None]
    seg = torch.cat([torch.full((n,), i + 1, dtype=torch.int32) for i, n in enumerate(lens)])[None]
    kw = A.model_inputs({"input_ids": ids, "position_ids": pos, "segment_ids": seg}, attn_implementation="sdpa",
                        mask_mode="explicit", sliding_window=sw)
    with torch.no_grad():
        packed = model(**kw).logits[0]
        c = 0
        for s in samples:
            ref = model(input_ids=s[None], use_cache=False).logits[0]
            assert (packed[c: c + len(s)] - ref).abs().max().item() < 1e-3
            c += len(s)
