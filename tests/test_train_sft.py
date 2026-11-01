# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gemma_mt.modeling import build_tiny_model
from gemma_mt.losses import sft_nll_sum
from gemma_mt.utils import lr_at


def test_lr_schedule_shape():
    peak, total, warm = 1e-3, 100, 10
    assert lr_at(0, total, warm, peak) == pytest.approx(peak / warm)
    assert lr_at(9, total, warm, peak) == pytest.approx(peak)
    xs = [lr_at(s, total, warm, peak, 0.1) for s in range(10, 100)]
    assert all(a >= b - 1e-12 for a, b in zip(xs, xs[1:])), "cosine decay must be monotone"
    assert xs[0] <= peak and xs[-1] >= 0.1 * peak - 1e-12
    assert lr_at(100, total, warm, peak, 0.1) == pytest.approx(0.1 * peak)


def test_token_weighted_accumulation_equals_one_big_batch_and_differs_from_mean_of_means():
    torch.manual_seed(0)
    model = build_tiny_model(seed=7, attn_implementation="sdpa")
    model.train()
    ids = torch.randint(4, 500, (2, 24))
    lab_a = ids[:1].clone(); lab_a[:, :20] = -100          # 3 labelled tokens
    lab_b = ids[1:].clone(); lab_b[:, :4] = -100           # 19 labelled tokens

    def grad_of(fn):
        model.zero_grad()
        fn().backward()
        return model.model.embed_tokens.weight.grad.clone()

    # (1) one big batch: sum of all token losses / total tokens
    nll_all, n_all = sft_nll_sum(model, {"input_ids": ids, "use_cache": False}, torch.cat([lab_a, lab_b]), chunk=5)
    g_big = grad_of(lambda: nll_all / n_all)

    # (2) the trainer's rule: sum of micro-batch SUMS divided by the step's total token count
    def accumulated():
        na, ca = sft_nll_sum(model, {"input_ids": ids[:1], "use_cache": False}, lab_a)
        nb, cb = sft_nll_sum(model, {"input_ids": ids[1:], "use_cache": False}, lab_b)
        return (na + nb) / (ca + cb)

    g_acc = grad_of(accumulated)
    # (3) the wrong rule: mean of micro-batch means
    def mean_of_means():
        na, ca = sft_nll_sum(model, {"input_ids": ids[:1], "use_cache": False}, lab_a)
        nb, cb = sft_nll_sum(model, {"input_ids": ids[1:], "use_cache": False}, lab_b)
        return 0.5 * (na / ca + nb / cb)

    g_mom = grad_of(mean_of_means)
    assert (g_acc - g_big).abs().max() < 1e-6
    assert (g_mom - g_big).abs().max() > 1e-4, "the control must differ, otherwise the test proves nothing"


def test_early_stopper_respects_min_epochs_patience_and_delta():
    from gemma_mt.train_sft import EarlyStopper

    s = EarlyStopper(patience=2, min_delta=0.01, min_step=10)
    assert s.update(2, 1.0) == (True, False)
    assert s.update(4, 1.0) == (False, False)      # no improvement, but before min_step: not counted
    assert s.update(6, 1.5) == (False, False)
    assert s.bad == 0
    assert s.update(10, 1.5) == (False, False) and s.bad == 1
    assert s.update(12, 0.995) == (False, True)   # 0.995 is not 0.01 better than 1.0: counts as bad -> patience hit
    s2 = EarlyStopper(patience=2, min_delta=0.0, min_step=0)
    s2.update(1, 1.0)
    assert s2.update(2, 1.1) == (False, False)
    assert s2.update(3, 1.2) == (False, True)
    assert s2.update(4, 0.5) == (True, False) and s2.bad == 0
    off = EarlyStopper(patience=0, min_delta=0.0, min_step=0)
    for i in range(10):
        assert off.update(i, 1.0 + i)[1] is False
    st = s2.state()
    s3 = EarlyStopper(2, 0.0, 0)
    s3.load(st)
    assert s3.best == s2.best


def _prepare(tmp_path):
    from gemma_mt.data.prepare_sft import PrepareConfig, prepare
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.smoke import synthetic_rows, write_jsonl

    tok = FakeGemmaTokenizer()
    raw = write_jsonl(str(tmp_path / "raw.jsonl"), synthetic_rows(80, seed=11))
    prepare(PrepareConfig(input=raw, out_dir=str(tmp_path / "packed"), seq_len=128, val_fraction=0.2, seed=1), tokenizer=tok)
    return tok, str(tmp_path / "packed")


def _cfg(tmp_path, packed, out, **kw):
    from gemma_mt.train_sft import SFTConfig

    base = dict(model_name="tiny-random", data_dir=packed, out_dir=str(tmp_path / out), autocast_dtype="none",
                micro_batch_size=2, grad_accum_steps=2, epochs=50, max_steps=6, learning_rate=2e-3, warmup_steps=2,
                eval_every=3, save_every=3, log_every=1, early_stopping_patience=0, device="cpu", seed=3)
    base.update(kw)
    return SFTConfig(**base)


def test_exact_resume_matches_uninterrupted_run(tmp_path):
    from gemma_mt.train_sft import train

    tok, packed = _prepare(tmp_path)
    full = train(_cfg(tmp_path, packed, "full"), tokenizer=tok)
    part = train(_cfg(tmp_path, packed, "split", stop_after_steps=3), tokenizer=tok)
    assert part["steps"] == 3
    rest = train(_cfg(tmp_path, packed, "split"), tokenizer=tok)         # resumes from split/last
    assert rest["steps"] == 6 and full["steps"] == 6
    a = torch.load(tmp_path / "full" / "last" / "trainer_state.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "split" / "last" / "trainer_state.pt", weights_only=False)["model"]
    assert set(a) == set(b)
    worst = max((a[k].float() - b[k].float()).abs().max().item() for k in a)
    assert worst == 0.0, f"resume is not exact (max diff {worst})"
    rec = [json.loads(l) for l in open(tmp_path / "full" / "metrics.jsonl")]
    rec2 = [json.loads(l) for l in open(tmp_path / "split" / "metrics.jsonl")]
    l1 = {r["step"]: r["train_loss"] for r in rec if "train_loss" in r}
    l2 = {r["step"]: r["train_loss"] for r in rec2 if "train_loss" in r}
    assert all(abs(l1[s] - l2[s]) < 1e-9 for s in l1 if s in l2)


def test_validation_split_is_reported_per_language_and_best_checkpoint_saved(tmp_path):
    from gemma_mt.train_sft import train

    tok, packed = _prepare(tmp_path)
    out = train(_cfg(tmp_path, packed, "run", max_steps=4, eval_every=2), tokenizer=tok)
    assert os.path.exists(tmp_path / "run" / "best" / "model.safetensors")
    gc = json.load(open(tmp_path / "run" / "final" / "generation_config.json"))
    assert 106 in gc["eos_token_id"], "saved generation_config must stop on <end_of_turn>"
    ev = out["last_eval"]
    assert ev["val_loss"] == ev["val_loss"] and ev["val_tokens"] > 0
    assert set(ev["val_loss_by_language"]) <= {"yor", "hau", "ibo"}


def test_smoke_run_end_to_end():
    from gemma_mt.train_sft import run_smoke

    s = run_smoke()
    assert s["last_loss"] < s["first_loss"]


def test_bidirectional_training_reports_both_directions(tmp_path):
    from gemma_mt.data.prepare_sft import PrepareConfig, prepare
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.smoke import synthetic_rows, write_jsonl
    from gemma_mt.train_sft import train

    tok = FakeGemmaTokenizer()
    raw = write_jsonl(str(tmp_path / "raw.jsonl"), synthetic_rows(80, seed=11))
    prepare(PrepareConfig(input=raw, out_dir=str(tmp_path / "packed"), seq_len=128, val_fraction=0.25, seed=1,
                          directions=("en2x", "x2en")), tokenizer=tok)
    out = train(_cfg(tmp_path, str(tmp_path / "packed"), "bidir", max_steps=2, eval_every=2), tokenizer=tok)
    ev = out["last_eval"]
    assert ev["val_loss_en2x"] == ev["val_loss_en2x"] and ev["val_loss_x2en"] == ev["val_loss_x2en"]
    assert set(ev["val_loss_by_language_x2en"]) <= set(ev["val_loss_by_language"]) | {"yor", "hau", "ibo"}
    # overall loss is the token-weighted mean of the two directions
    assert min(ev["val_loss_en2x"], ev["val_loss_x2en"]) - 1e-6 <= ev["val_loss"] <= max(ev["val_loss_en2x"], ev["val_loss_x2en"]) + 1e-6


def test_language_stratified_training_resumes_exactly_and_reports_macro(tmp_path):
    from gemma_mt.data.prepare_sft import PrepareConfig, prepare
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.smoke import synthetic_rows, write_jsonl
    from gemma_mt.train_sft import train

    tok = FakeGemmaTokenizer()
    raw = write_jsonl(str(tmp_path / "raw.jsonl"), synthetic_rows(120, seed=21))
    prepare(PrepareConfig(input=raw, out_dir=str(tmp_path / "packed"), seq_len=128, val_fraction=0.25, seed=1,
                          pack_by_language=True), tokenizer=tok)
    kw = dict(language_alpha=0.5, max_language_epochs=0.0, min_language_share=0.05, min_val_tokens_macro=1, hub_repo_id="me/none")
    full = train(_cfg(tmp_path, str(tmp_path / "packed"), "full", **kw), tokenizer=tok)
    train(_cfg(tmp_path, str(tmp_path / "packed"), "split", stop_after_steps=3, **kw), tokenizer=tok)
    rest = train(_cfg(tmp_path, str(tmp_path / "packed"), "split", **kw), tokenizer=tok)
    assert rest["steps"] == full["steps"] == 6
    a = torch.load(tmp_path / "full" / "last" / "trainer_state.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "split" / "last" / "trainer_state.pt", weights_only=False)["model"]
    assert max((a[k].float() - b[k].float()).abs().max().item() for k in a) == 0.0
    ev = full["last_eval"]
    assert ev["val_macro_languages"] >= 1 and ev["val_loss_macro"] == ev["val_loss_macro"]


def test_keep_single_checkpoint_leaves_only_the_resumable_state(tmp_path):
    from gemma_mt.train_sft import train

    tok, packed = _prepare(tmp_path)
    train(_cfg(tmp_path, packed, "one", max_steps=4, eval_every=2, save_every=2, keep_single_checkpoint=True), tokenizer=tok)
    run = tmp_path / "one"
    assert (run / "last" / "trainer_state.pt").exists()
    assert not (run / "best").exists() and not (run / "final").exists() and not (run / "hub_stage").exists()


def test_quick_eval_logs_per_language_without_touching_stopping(tmp_path):
    from gemma_mt.data.prepare_sft import PrepareConfig, prepare
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.smoke import synthetic_rows, write_jsonl
    from gemma_mt.train_sft import train

    tok = FakeGemmaTokenizer()
    raw = write_jsonl(str(tmp_path / "raw.jsonl"), synthetic_rows(120, seed=21))
    prepare(PrepareConfig(input=raw, out_dir=str(tmp_path / "packed"), seq_len=128, val_fraction=0.25, seed=1,
                          pack_by_language=True), tokenizer=tok)
    out = train(_cfg(tmp_path, str(tmp_path / "packed"), "q", max_steps=5, eval_every=5, quick_eval_every=2,
                     quick_eval_packs_per_language=1, min_val_tokens_macro=1), tokenizer=tok)
    recs = [json.loads(l) for l in open(tmp_path / "q" / "metrics.jsonl")]
    q = [r for r in recs if "quick_val_loss_macro" in r]
    assert [r["step"] for r in q] == [2, 4]
    assert any(k.startswith("quick_val_loss/") for k in q[0])
    assert out["last_eval"]["val_loss_macro"] == out["last_eval"]["val_loss_macro"]


def test_early_stop_can_only_propose(tmp_path):
    from gemma_mt.train_sft import train

    tok, packed = _prepare(tmp_path)
    # patience 1 with a huge lr makes the validation loss stop improving quickly; in "propose" mode the run must still reach max_steps
    out = train(_cfg(tmp_path, packed, "prop", max_steps=8, eval_every=2, early_stopping_patience=1, min_epochs=0.0,
                     learning_rate=5e-2, early_stop_action="propose", keep_single_checkpoint=True), tokenizer=tok)
    assert out["steps"] == 8 and not out["stopped_early"]
    assert (tmp_path / "prop" / "STOP_PROPOSED").exists() or out["best_step"] == 8


def test_previous_checkpoint_is_kept_as_a_hard_link_and_resume_falls_back_to_it(tmp_path):
    from gemma_mt.train_sft import train

    tok, packed = _prepare(tmp_path)
    run = tmp_path / "keep"
    train(_cfg(tmp_path, packed, "keep", max_steps=6, eval_every=0, save_every=2, keep_single_checkpoint=True, keep_prev_checkpoint=True), tokenizer=tok)
    last, prev = run / "last" / "trainer_state.pt", run / "last_prev" / "trainer_state.pt"
    assert last.exists() and prev.exists()
    assert torch.load(prev, weights_only=False)["state"]["step"] < torch.load(last, weights_only=False)["state"]["step"]
    # corrupt the latest: a resumed run must continue from the previous one instead of failing
    last.write_bytes(b"not a checkpoint")
    out = train(_cfg(tmp_path, packed, "keep", max_steps=8, eval_every=0, save_every=2, keep_single_checkpoint=True, keep_prev_checkpoint=True), tokenizer=tok)
    assert out["steps"] == 8


def test_new_phase_muon_unimax_from_a_checkpoint_and_exact_resume(tmp_path):
    from gemma_mt.data.packed_dataset import PackedDataset
    from gemma_mt.data.prepare_sft import PrepareConfig, prepare
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.languages import ALL_ISOS
    from gemma_mt.smoke import synthetic_rows, write_jsonl
    from gemma_mt.train_sft import train

    tok = FakeGemmaTokenizer()
    raw = write_jsonl(str(tmp_path / "raw.jsonl"), synthetic_rows(120, seed=21))
    packed = str(tmp_path / "packed")
    prepare(PrepareConfig(input=raw, out_dir=packed, seq_len=128, val_fraction=0.25, seed=1, pack_by_language=True), tokenizer=tok)
    first = train(_cfg(tmp_path, packed, "p1", max_steps=4, eval_every=2, save_every=2, min_val_tokens_macro=1, min_epochs=0.0), tokenizer=tok)
    init = str(tmp_path / "p1" / "last" / "trainer_state.pt")
    ds = PackedDataset(packed, "train")
    n_packs = len(ds)
    per_step = 2 * 2                                                 # _cfg: micro 2 x accum 2
    budget_tokens = float(int(1.5 * n_packs) // per_step * per_step * 128)     # 1.5 natural passes
    prio = [ALL_ISOS[int(i)] for i in sorted(set(ds.pack_languages().tolist()))[:1]]
    kw = dict(init_from=init, optimizer="muon", unimax_budget_tokens=budget_tokens, unimax_min_epochs=1.0, unimax_max_epochs=2.0,
              unimax_priority_languages=prio, unimax_priority_weight=3.0, eval_every=3, save_every=3, min_val_tokens_macro=1,
              min_epochs=0.0, learning_rate=1e-3, max_steps=0)
    full = train(_cfg(tmp_path, packed, "p2full", **kw), tokenizer=tok)
    n_run = int(budget_tokens // (per_step * 128))
    assert full["steps"] == first["steps"] + n_run                   # the step counter continues, the phase has its own length
    train(_cfg(tmp_path, packed, "p2split", stop_after_steps=first["steps"] + 3, **kw), tokenizer=tok)
    rest = train(_cfg(tmp_path, packed, "p2split", **kw), tokenizer=tok)
    assert rest["steps"] == full["steps"]
    a = torch.load(tmp_path / "p2full" / "last" / "trainer_state.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "p2split" / "last" / "trainer_state.pt", weights_only=False)["model"]
    assert max((a[k].float() - b[k].float()).abs().max().item() for k in a) == 0.0


def test_weights_only_checkpoint_resumes_with_a_fresh_optimizer_and_lr_rewarmup(tmp_path):
    """save_optimizer=False writes a slim checkpoint; resuming from it (or from a full one with resume_optimizer=False) restarts the optimizer and ramps the LR."""
    import os
    from gemma_mt.train_sft import train

    tok, packed = _prepare(tmp_path)
    part = train(_cfg(tmp_path, packed, "slim", stop_after_steps=3, save_optimizer=False), tokenizer=tok)
    assert part["steps"] == 3
    ck = torch.load(tmp_path / "slim" / "last" / "trainer_state.pt", weights_only=False)
    assert "optimizer" not in ck and {"model", "rng", "state"} <= set(ck)
    rest = train(_cfg(tmp_path, packed, "slim", save_optimizer=False, optimizer_rewarmup_steps=2), tokenizer=tok)
    assert rest["steps"] == 6
    recs = [json.loads(l) for l in open(tmp_path / "slim" / "metrics.jsonl")]
    lrs = {r["step"]: r["lr"] for r in recs if "lr" in r and "train_loss" in r}
    full = train(_cfg(tmp_path, packed, "fullopt", stop_after_steps=3), tokenizer=tok)
    assert os.path.getsize(tmp_path / "fullopt" / "last" / "trainer_state.pt") > os.path.getsize(tmp_path / "slim" / "last" / "trainer_state.pt")
    # a FULL checkpoint resumed with resume_optimizer=False also works
    again = train(_cfg(tmp_path, packed, "fullopt", resume_optimizer=False, optimizer_rewarmup_steps=2), tokenizer=tok)
    assert again["steps"] == 6 and full["steps"] == 3
    # the rewarm ramp lowers the first resumed learning rates below the schedule value
    ref = train(_cfg(tmp_path, packed, "ref"), tokenizer=tok)
    ref_lrs = {r["step"]: r["lr"] for r in (json.loads(l) for l in open(tmp_path / "ref" / "metrics.jsonl")) if "lr" in r and "train_loss" in r}
    first = min(s_ for s_ in lrs if s_ > 3)
    assert lrs[first] < ref_lrs[first] and ref["steps"] == 6, (lrs, ref_lrs)
    assert all(abs(lrs[s_] - ref_lrs[s_]) < 1e-12 for s_ in lrs if s_ <= 3 and s_ in ref_lrs)


def _losses_and_eval(tmp_path, name):
    rec = [json.loads(l) for l in open(tmp_path / name / "metrics.jsonl")]
    tr = {r["step"]: r["train_loss"] for r in rec if "train_loss" in r}
    ev = {r["step"]: r["val_loss"] for r in rec if "val_loss" in r}
    return tr, ev


def test_varlen_attention_trains_like_the_explicit_masks_incl_evaluation(tmp_path):
    """D-35: attn_implementation=varlen (flattened [1, N] batches, torch reference kernel on CPU) gives the same
    loss curve and validation loss as sdpa + explicit masks (padded rows with several samples each)."""
    from gemma_mt.train_sft import train

    tok, packed = _prepare(tmp_path)
    kw = dict(max_steps=6, eval_every=3, micro_batch_size=2)
    train(_cfg(tmp_path, packed, "base", **kw), tokenizer=tok)
    s = train(_cfg(tmp_path, packed, "vl", attn_implementation="varlen", **kw), tokenizer=tok)
    assert s["mask_mode"] == "flash"
    a, ea = _losses_and_eval(tmp_path, "base")
    b, eb = _losses_and_eval(tmp_path, "vl")
    assert set(a) == set(b) and ea.keys() == eb.keys() and ea
    for k in a:
        assert abs(a[k] - b[k]) < 2e-3 * max(1.0, abs(a[k])), (k, a[k], b[k])
    for k in ea:
        assert abs(ea[k] - eb[k]) < 2e-3 * max(1.0, abs(ea[k])), (k, ea[k], eb[k])


def test_fused_loss_trains_like_the_chunked_loss(tmp_path):
    from gemma_mt.train_sft import train

    tok, packed = _prepare(tmp_path)
    kw = dict(max_steps=5, eval_every=5)
    train(_cfg(tmp_path, packed, "ch", **kw), tokenizer=tok)
    train(_cfg(tmp_path, packed, "fu", loss_impl="fused", **kw), tokenizer=tok)
    a, ea = _losses_and_eval(tmp_path, "ch")
    b, eb = _losses_and_eval(tmp_path, "fu")
    for k in a:
        assert abs(a[k] - b[k]) < 2e-3 * max(1.0, abs(a[k])), (k, a[k], b[k])
    assert abs(ea[5] - eb[5]) < 2e-3


def test_attention_config_is_validated_before_any_work():
    from gemma_mt.train_sft import SFTConfig, resolve_mask_mode, validate_attention_config

    validate_attention_config(SFTConfig())
    validate_attention_config(SFTConfig(attn_implementation="varlen"))
    validate_attention_config(SFTConfig(attn_implementation="flex_attention", flex_kernel_options={"BLOCK_M": 32}))
    assert resolve_mask_mode(SFTConfig(attn_implementation="flex_attention")) == "position_ids"
    for bad in (dict(attn_implementation="flex_attention", mask_mode="explicit"), dict(attn_implementation="flash"), dict(attn_implementation="varlen", mask_mode="explicit"),
                dict(attn_implementation="sdpa", mask_mode="flash"), dict(loss_impl="triton"),
                dict(attn_implementation="sdpa", flex_kernel_options={"BLOCK_M": 32})):
        with pytest.raises(ValueError):
            validate_attention_config(SFTConfig(**bad))
