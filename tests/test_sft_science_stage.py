# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests of the short science-SFT stage (D-58): window cutting, UniMax budgets, the streaming extraction, L2-SP, drift tracking, weight interpolation."""
import io
import os
import random
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import build_sft_science_mix as B  # noqa: E402


# ------------------------------------------------------------------------------------------------ windows
def test_segment_covers_every_sentence_exactly_once_and_respects_the_mode_sizes():
    for mode, (lo, hi) in {"S": (1, 1), "A": (2, 4), "B": (5, 12)}.items():
        for n in (1, 7, 33, 55):
            w = B.segment(n, mode, random.Random(n))
            assert [a for a, _ in w][0] == 0 and w[-1][1] == n
            assert all(w[k][1] == w[k + 1][0] for k in range(len(w) - 1))          # consecutive, no gap, no overlap
            assert all(b - a <= hi for a, b in w[:-1]) and all(b - a >= lo for a, b in w[:-1])
    assert B.segment(40, "W", random.Random(0)) == [(0, 40)]
    assert B.segment(0, "S", random.Random(0)) == []


def test_a_paper_never_gets_the_same_granularity_twice_and_modes_are_deterministic():
    seen = set()
    for pid in range(200):
        m = B.paper_modes(pid, 2, 5)
        assert m[0] != m[1]
        assert m == B.paper_modes(pid, 2, 5)
        seen.add(m[0])
    assert seen == set(B.MODES)
    assert len(B.paper_modes(1, 6, 5)) == 6                     # more passes than granularities: the list is refilled


_VOCAB = [f"word{i}" for i in range(400)]


def _toy_corpus(n_papers=6, per_paper=9, langs=("hau", "yor")):
    import pandas as pd

    rows = []
    for pid in range(1, n_papers + 1):
        for sid in range(per_paper):
            for iso in langs:
                r = random.Random(pid * 100 + sid)
                src = f"Item P{pid}S{sid} " + " ".join(r.choice(_VOCAB) for _ in range(12)) + "."
                rows.append({"paper_id": pid, "sentence_id": sid + 1, "domain": "health", "lang_pair": f"eng-{iso}", "source_lang": "eng", "target_lang": iso,
                             "source": src, "target": f"{iso}: " + src})
    return pd.DataFrame(rows)


def test_science_windows_cover_each_sentence_once_per_pass_and_never_cross_papers():
    df = _toy_corpus()
    wins, st = B.science_windows(df, B.HoldOut(), passes=2, seed=3, languages=("hau", "yor"))
    for iso in ("hau", "yor"):
        for p in (0, 1):
            ws = [w for w in wins if w["iso"] == iso and w["pass"] == p]
            assert sum(w["n_sent"] for w in ws) == 6 * 9                     # every sentence exactly once per pass
            for w in ws:
                papers = {int(x) for x in re.findall(r"P(\d+)S\d+", w["english"])}
                assert papers == {w["paper"]}                                # a window stays inside its paper
                assert w["english"].count("Item P") == w["n_sent"]
    # multi-parallel: the cut of a paper is the same for both languages
    cuts = lambda iso: sorted((w["paper"], w["pass"], w["n_sent"]) for w in wins if w["iso"] == iso)
    assert cuts("hau") == cuts("yor")


def test_holdout_sentences_are_removed_and_the_rest_survive():
    df = _toy_corpus(n_papers=2, per_paper=6, langs=("hau",))
    ho = B.HoldOut()
    r = random.Random(1 * 100 + 2)
    banned = "Item P1S2 " + " ".join(r.choice(_VOCAB) for _ in range(12)) + "."
    ho.add("t", [banned], [banned])
    wins, st = B.science_windows(df, ho, passes=1, seed=1, languages=("hau",))
    assert st["dropped_holdout_sentence"] == 1
    assert not any(banned in w["english"] for w in wins)
    assert sum(w["n_sent"] for w in wins) == 11


def test_holdout_8gram_catches_a_long_overlap_but_not_a_short_one():
    ho = B.HoldOut()
    ho.add("t", ["the quick brown fox jumps over the lazy dog near the river bank"], [])
    assert ho.hit("A story: the quick brown fox jumps over the lazy dog today") == "ngram8"
    assert ho.hit("the quick brown fox jumps over") is None


# ------------------------------------------------------------------------------------------------ replay budget + streaming
def test_unimax_budgets_water_filling():
    b = B.unimax_budgets([100, 1000, 5000, 200000], total=3500, max_epochs=1.0)
    assert b.sum() == pytest.approx(3500, rel=1e-6)
    assert b[0] == pytest.approx(100) and b[1] == pytest.approx(1000)                 # small languages are taken whole
    assert b[2] == pytest.approx(b[3])                                                # big ones share the remaining level equally
    assert b[2] == pytest.approx(1200)
    assert np.allclose(B.unimax_budgets([10, 20], 1e9, 2.0), [20, 40])               # budget larger than the data: the cap (max_epochs x data)
    w = B.unimax_budgets([1e6, 1e6], 100, 1.0, weights=[3, 1])
    assert w[0] == pytest.approx(75) and w[1] == pytest.approx(25)


def test_plan_replay_is_per_language_and_deterministic():
    iso = np.repeat(np.arange(3), [50, 5, 500])
    ln = np.full(len(iso), 10, np.int64)
    bud = np.array([100.0, 100.0, 100.0])
    p1 = B.plan_replay(iso, ln, 3, bud, oversample=1.5, min_examples=8, seed=1)
    p2 = B.plan_replay(iso, ln, 3, bud, oversample=1.5, min_examples=8, seed=1)
    assert all((p1[k] == p2[k]).all() for k in p1)
    assert len(p1[0]) == 15 and len(p1[1]) == 5 and len(p1[2]) == 15                  # budget 100 tokens x 1.5 / 10 = 15 samples; language 1 has only 5
    assert set(iso[p1[0]]) == {0} and set(iso[p1[2]]) == {2}


def test_select_from_stream_returns_the_exact_ranges_across_chunk_borders():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, 100_000, dtype=np.uint8).tobytes()
    offs, lens, pos = [], [], 0
    while pos < len(data) - 300:
        pos += int(rng.integers(0, 400))
        n = int(rng.integers(1, 300))
        if pos + n > len(data):
            break
        offs.append(pos); lens.append(n); pos += n
    sel = list(range(0, len(offs), 3))
    got = B.select_from_stream(io.BytesIO(data), [offs[i] for i in sel], [lens[i] for i in sel], chunk=777)
    assert all(g == data[offs[i]: offs[i] + lens[i]] for g, i in zip(got, sel))
    with pytest.raises(EOFError):
        B.select_from_stream(io.BytesIO(data[:50]), [10, 60], [5, 5], chunk=16)


# ------------------------------------------------------------------------------------------------ interpolation + L2-SP + drift (tiny model, CPU)
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _tiny_pair(tmp_path):
    from gemma_mt.modeling import build_tiny_model

    a = build_tiny_model(seed=1)
    b = build_tiny_model(seed=2)
    a.save_pretrained(str(tmp_path / "init"))
    return a, b


def test_interpolation_endpoints_midpoint_and_directory_output(tmp_path):
    import interpolate_weights as W
    from transformers import AutoModelForCausalLM

    a, b = _tiny_pair(tmp_path)
    ck = tmp_path / "trainer_state.pt"
    torch.save({"model": b.state_dict(), "state": {"step": 1}}, str(ck))
    for alpha in (0.0, 0.25, 1.0):
        out = W.interpolate_to_dir(str(tmp_path / "init"), str(ck), alpha, str(tmp_path / f"a{alpha}"))
        m = AutoModelForCausalLM.from_pretrained(out, torch_dtype=torch.float32)
        for k, v in m.state_dict().items():
            want = (1 - alpha) * a.state_dict()[k].double() + alpha * b.state_dict()[k].double()
            assert (v.double() - want).abs().max() < 1e-6, k
    assert os.path.exists(tmp_path / "a0.25" / "config.json")


def _mini_train(tmp_path, **over):
    """A few real optimizer steps of gemma_mt.train_sft on the CPU smoke data with a random tiny model."""
    import json

    from gemma_mt.data.prepare_sft import PrepareConfig, prepare
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.smoke import synthetic_rows, write_jsonl
    from gemma_mt.train_sft import SFTConfig, train

    tok = FakeGemmaTokenizer()
    work = str(tmp_path / ("w" + str(over.get("l2sp_coef", 0))))
    os.makedirs(work, exist_ok=True)
    raw = write_jsonl(os.path.join(work, "raw.jsonl"), synthetic_rows(60, seed=1))
    prepare(PrepareConfig(input=raw, out_dir=os.path.join(work, "packed"), seq_len=256, val_fraction=0.2, seed=3), tokenizer=tok)
    import shutil

    shutil.copytree(os.path.join(work, "packed", "val"), os.path.join(work, "packed", "val_sci"), dirs_exist_ok=True)
    kw = dict(model_name="tiny-random", data_dir=os.path.join(work, "packed"), out_dir=os.path.join(work, "run"),
              attn_implementation="sdpa", autocast_dtype="none", micro_batch_size=2, grad_accum_steps=1, epochs=30, max_steps=12, learning_rate=3e-3,
              warmup_steps=2, eval_every=6, save_every=6, log_every=3, early_stopping_patience=0, device="cpu", seed=0, track_drift=True,
              extra_val_splits=["val_sci"], eval_at_start=True)
    kw.update(over)
    cfg = SFTConfig(**kw)
    train(cfg, tokenizer=tok)
    return [json.loads(l) for l in open(os.path.join(cfg.out_dir, "metrics.jsonl"))]


def test_drift_is_logged_extra_val_and_start_eval_work_and_l2sp_pulls_towards_the_start(tmp_path):
    free = _mini_train(tmp_path)
    pulled = _mini_train(tmp_path, l2sp_coef=50.0)
    d_free = [r["drift_rel"] for r in free if "drift_rel" in r]
    d_pull = [r["drift_rel"] for r in pulled if "drift_rel" in r]
    assert d_free and d_pull and d_free[-1] > 0
    assert d_pull[-1] < 0.6 * d_free[-1], (d_pull[-1], d_free[-1])      # a strong L2-SP keeps the weights much closer to the start
    assert any(r.get("step") == 0 and "val_loss" in r for r in free)      # the baseline evaluation at step 0
    assert any("val_val_sci_loss" in r for r in free)                      # the extra validation split is logged
    assert all("l2sp_penalty" in r for r in free if "drift_rel" in r)
