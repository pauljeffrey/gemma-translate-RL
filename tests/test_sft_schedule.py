# SPDX-License-Identifier: AGPL-3.0-or-later
"""Schedule-file option of the SFT trainer (arms A0 / S1 / S2 of the clean rerun, 2026-10-28) and the embeddings-only warm-up.

CPU, tiny random Gemma3, a toy packed set with one language per pack. The pack indices the trainer actually reads are recorded by wrapping
PackedDataset.__getitem__ (eval is off, so only the train set is read)."""
import importlib.util
import json
import os
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gemma_mt.data.packed_dataset import PackedDataset, balanced_schedule
from gemma_mt.data.prepare_sft import PrepareConfig, prepare
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
from gemma_mt.languages import ALL_ISOS
from gemma_mt.smoke import synthetic_rows, write_jsonl
from gemma_mt.train_sft import (SFTConfig, load_schedule, packed_structure_sha256, train)

ISOS = ("yor", "hau", "ibo", "amh", "fon", "ewe")
SPEC = ("amh", "fon", "ewe")
PER_STEP = 4                                                   # micro_batch_size 2 x grad_accum_steps 2 x world 1
TOK = FakeGemmaTokenizer()
SAMPLER_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "sft_clean", "chain", "sampler.py")


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    d = tmp_path_factory.mktemp("sched_toy")
    raw = write_jsonl(str(d / "raw.jsonl"), synthetic_rows(600, isos=ISOS, seed=5))
    prepare(PrepareConfig(input=raw, out_dir=str(d / "packed"), seq_len=128, val_fraction=0.2, seed=1, pack_by_language=True), tokenizer=TOK)
    return str(d / "packed")


@pytest.fixture
def sampler(monkeypatch):
    if not os.path.exists(SAMPLER_PY):
        pytest.skip("data_pipeline/sft_clean/chain/sampler.py is not next to gemma_mt")
    spec = importlib.util.spec_from_file_location("chain_sampler", os.path.abspath(SAMPLER_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "SPECIALIST", list(SPEC))
    return mod


def _cfg(tmp_path, packed_dir, out, **kw):
    base = dict(model_name="tiny-random", data_dir=packed_dir, out_dir=str(tmp_path / out), autocast_dtype="none",
                micro_batch_size=2, grad_accum_steps=2, epochs=50, max_steps=0, learning_rate=2e-3, warmup_steps=2,
                eval_every=0, save_every=100, log_every=1, early_stopping_patience=0, device="cpu", seed=3)
    base.update(kw)
    return SFTConfig(**base)


class Recorder:
    """Wraps PackedDataset.__getitem__: the sequence of pack indices the trainer reads."""

    def __init__(self, monkeypatch):
        self.seen = []
        orig = PackedDataset.__getitem__
        rec = self

        def wrapped(self_, i):
            rec.seen.append(int(i))
            return orig(self_, i)

        monkeypatch.setattr(PackedDataset, "__getitem__", wrapped)


def _model(tmp_path, out):
    return torch.load(tmp_path / out / "last" / "trainer_state.pt", weights_only=False)["model"]


def _same(a, b):
    assert set(a) == set(b)
    return max((a[k].float() - b[k].float()).abs().max().item() for k in a)


def _write_schedule(path_dir, sched, ds, **sidecar_over):
    os.makedirs(path_dir, exist_ok=True)
    sched = np.ascontiguousarray(sched, dtype=np.int64)
    np.save(os.path.join(path_dir, "schedule.npy"), sched)
    import hashlib
    from gemma_mt.train_sft import pack_lang_sha256
    sc = {"schedule_sha256": hashlib.sha256(sched.tobytes()).hexdigest(), "steps": int(sched.shape[0]) if sched.ndim == 2 else None, "per_step": int(sched.shape[1]) if sched.ndim == 2 else None,
          "n_packs": len(ds), "pack_lang_sha256": pack_lang_sha256(ds.pack_languages()), "packed_structure_sha256": packed_structure_sha256(ds)}
    sc.update(sidecar_over)
    json.dump(sc, open(os.path.join(path_dir, "schedule.json"), "w"))
    return os.path.join(path_dir, "schedule.npy")


# ------------------------------------------------------------------------------------------------ A0: schedule == internal sampler
def test_a0_from_schedule_equals_internal_sampler_step_for_step(tmp_path, packed, monkeypatch):
    ds = PackedDataset(packed, "train")
    kw = dict(language_alpha=0.5, min_language_share=0.05, max_language_epochs=0.0, max_steps=6)
    rec1 = Recorder(monkeypatch)
    train(_cfg(tmp_path, packed, "internal", **kw), tokenizer=TOK)
    internal_seen = list(rec1.seen)
    monkeypatch.undo()
    sched, _ = balanced_schedule(ds.pack_languages(), 6, PER_STEP, alpha=0.5, min_share=0.05, max_epochs=0.0, seed=3)
    f = _write_schedule(str(tmp_path / "A0"), sched, ds)
    rec2 = Recorder(monkeypatch)
    out = train(_cfg(tmp_path, packed, "fromfile", schedule_file=f), tokenizer=TOK)
    assert out["steps"] == out["total_steps"] == 6
    assert rec2.seen == internal_seen == sched.reshape(-1).tolist(), "same packs, same order, step for step"
    assert _same(_model(tmp_path, "internal"), _model(tmp_path, "fromfile")) == 0.0, "same batches -> bit-identical weights"
    # logged losses agree too
    l = lambda o: {r["step"]: r["train_loss"] for r in map(json.loads, open(tmp_path / o / "metrics.jsonl")) if "train_loss" in r}
    assert l("internal") == l("fromfile")


def test_default_behaviour_unchanged_without_schedule_file(tmp_path, packed):
    out = train(_cfg(tmp_path, packed, "plain", max_steps=3), tokenizer=TOK)
    assert out["steps"] == 3 and "schedule" not in out
    assert not os.path.exists(tmp_path / "plain" / "schedule_manifest.json")
    assert json.load(open(tmp_path / "plain" / "config.json"))["schedule"] is None


def test_run_length_and_lr_follow_the_file(tmp_path, packed):
    ds = PackedDataset(packed, "train")
    sched, _ = balanced_schedule(ds.pack_languages(), 7, PER_STEP, seed=1)
    f = _write_schedule(str(tmp_path / "s"), sched, ds)
    out = train(_cfg(tmp_path, packed, "len", schedule_file=f, warmup_steps=0, warmup_ratio=0.3), tokenizer=TOK)
    assert out["total_steps"] == out["steps"] == 7
    lrs = {r["step"]: r["lr"] for r in map(json.loads, open(tmp_path / "len" / "metrics.jsonl")) if "lr" in r}
    from gemma_mt.utils import lr_at
    warm = max(1, int(round(0.3 * 7)))
    assert all(abs(lrs[s] - lr_at(s - 1, 7, warm, 2e-3, 0.1)) < 1e-12 for s in lrs)


def test_flat_array_is_accepted_and_hash_is_shape_independent(tmp_path, packed):
    ds = PackedDataset(packed, "train")
    sched, _ = balanced_schedule(ds.pack_languages(), 5, PER_STEP, seed=2)
    f2 = _write_schedule(str(tmp_path / "two"), sched, ds)
    f1 = _write_schedule(str(tmp_path / "one"), sched.reshape(-1), ds, steps=5, per_step=PER_STEP)   # the flat file's sidecar names the shape
    a2, i2 = load_schedule(f2, ds, PER_STEP)
    a1, i1 = load_schedule(f1, ds, PER_STEP)
    assert np.array_equal(a1, a2) and i1["sha256"] == i2["sha256"] and a1.shape == (5, PER_STEP)


# ------------------------------------------------------------------------------------------------ S1: exact subsequence of A0
def test_s1_is_exact_subsequence_of_a0_and_trains_on_it(tmp_path, packed, sampler, monkeypatch):
    # sampler.plan on the real packed train dir (writes schedule.npy, meta.json and the NEW schedule.json), then sampler.arms
    args = type("A", (), dict(stage="A", packed_dir=os.path.join(packed, "train"), packs_csv=None, out=str(tmp_path / "A0"), seed=3,
                              steps=40, per_step=PER_STEP, alpha=0.5, budget_tokens=0))()
    sampler.plan(args)
    sampler.arms(type("A", (), dict(schedule=str(tmp_path / "A0"), out=str(tmp_path / "arms")))())
    a0 = np.load(tmp_path / "A0" / "schedule.npy")
    ds = PackedDataset(packed, "train")
    lang = ds.pack_languages()
    spec_ids = [ALL_ISOS.index(i) for i in SPEC]
    for arm in ("S1", "S2"):
        s1 = np.load(tmp_path / "arms" / arm / "schedule.npy")
        flat = a0.reshape(-1)
        expect = flat[np.isin(lang[flat], spec_ids)]
        assert np.array_equal(expect[:s1.size], s1.reshape(-1)), "S1 = A0's draws on specialist packs, same order"
        assert 0 <= expect.size - s1.size < PER_STEP and s1.shape[1] == PER_STEP
        pos = np.load(tmp_path / "arms" / arm / "a0_positions.npy")
        assert np.array_equal(flat[pos], s1.reshape(-1)) and np.all(np.diff(pos) > 0), "increasing positions inside A0"
    # sidecars: A0's and S1's, both loadable against the packed set the plan was made from
    assert json.load(open(tmp_path / "A0" / "schedule.json"))["packed_structure_sha256"] == packed_structure_sha256(ds), "sampler and trainer hash agree"
    s1_json = json.load(open(tmp_path / "arms" / "S1" / "schedule.json"))
    assert s1_json["packed_structure_sha256"] == packed_structure_sha256(ds) and s1_json["a0_schedule_sha256"]
    # same passes per specialist language (up to the dropped partial step)
    for i in spec_ids:
        da = int((lang[a0.reshape(-1)] == i).sum()); ds_ = int((lang[np.load(tmp_path / "arms" / "S1" / "schedule.npy").reshape(-1)] == i).sum())
        assert 0 <= da - ds_ < PER_STEP
    # the trainer takes exactly these steps; every pack drawn is a specialist pack
    s1 = np.load(tmp_path / "arms" / "S1" / "schedule.npy")
    rec = Recorder(monkeypatch)
    out = train(_cfg(tmp_path, packed, "s1", schedule_file=str(tmp_path / "arms" / "S1" / "schedule.npy")), tokenizer=TOK)
    assert out["steps"] == s1.shape[0] and rec.seen == s1.reshape(-1).tolist()
    assert set(np.unique(lang[np.array(rec.seen)]).tolist()) <= set(spec_ids)
    man = json.load(open(tmp_path / "s1" / "schedule_manifest.json"))
    assert man["sha256"] == s1_json["schedule_sha256"] == out["schedule"]["sha256"]
    assert man["steps"] == s1.shape[0] and man["draws"] == s1.size
    assert sum(v["draws"] for v in man["draws_per_language"].values()) == s1.size
    assert {k for k, v in man["draws_per_language"].items() if v["draws"]} <= set(SPEC)
    assert man["padded_tokens"] == s1.size * 128 and 0 < man["target_tokens"] < man["real_tokens"] <= man["padded_tokens"]
    assert "packed_structure_sha256" in man["checks_passed"] and "schedule_sha256" in man["checks_passed"]
    assert json.load(open(tmp_path / "s1" / "config.json"))["schedule"]["sha256"] == man["sha256"]


# ------------------------------------------------------------------------------------------------ resume
def test_resume_continues_at_the_right_schedule_position(tmp_path, packed, monkeypatch):
    ds = PackedDataset(packed, "train")
    sched, _ = balanced_schedule(ds.pack_languages(), 6, PER_STEP, seed=4)
    f = _write_schedule(str(tmp_path / "s"), sched, ds)
    rec = Recorder(monkeypatch)
    train(_cfg(tmp_path, packed, "full", schedule_file=f, save_every=3), tokenizer=TOK)
    full_seen = list(rec.seen)
    rec.seen.clear()
    train(_cfg(tmp_path, packed, "split", schedule_file=f, save_every=3, stop_after_steps=3), tokenizer=TOK)
    first = list(rec.seen)
    rec.seen.clear()
    out = train(_cfg(tmp_path, packed, "split", schedule_file=f, save_every=3), tokenizer=TOK)
    second = list(rec.seen)
    assert out["steps"] == 6
    assert first == sched[:3].reshape(-1).tolist() and second == sched[3:].reshape(-1).tolist() and first + second == full_seen
    assert _same(_model(tmp_path, "full"), _model(tmp_path, "split")) == 0.0
    st = torch.load(tmp_path / "split" / "last" / "trainer_state.pt", weights_only=False)["state"]
    assert st["schedule_sha256"] == load_schedule(f, ds, PER_STEP)[1]["sha256"]


def test_resume_with_another_schedule_is_refused(tmp_path, packed):
    ds = PackedDataset(packed, "train")
    s_a, _ = balanced_schedule(ds.pack_languages(), 6, PER_STEP, seed=4)
    s_b, _ = balanced_schedule(ds.pack_languages(), 6, PER_STEP, seed=5)
    fa = _write_schedule(str(tmp_path / "a"), s_a, ds)
    fb = _write_schedule(str(tmp_path / "b"), s_b, ds)
    train(_cfg(tmp_path, packed, "r", schedule_file=fa, stop_after_steps=3), tokenizer=TOK)
    with pytest.raises(ValueError, match="same schedule file"):
        train(_cfg(tmp_path, packed, "r", schedule_file=fb), tokenizer=TOK)
    with pytest.raises(ValueError, match="same schedule file"):
        train(_cfg(tmp_path, packed, "r", max_steps=6), tokenizer=TOK)           # a checkpoint of a scheduled run, resumed without the file


# ------------------------------------------------------------------------------------------------ refusals (all before the model is loaded)
@pytest.fixture
def no_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the model must not be loaded when the schedule is refused")
    monkeypatch.setattr("gemma_mt.modeling.load_model", boom)


def test_index_outside_the_pack_set_fails_loudly(tmp_path, packed, no_model):
    ds = PackedDataset(packed, "train")
    sched, _ = balanced_schedule(ds.pack_languages(), 4, PER_STEP, seed=1)
    for bad in (len(ds), -1, 10 ** 9):
        s = sched.copy(); s[2, 1] = bad
        f = _write_schedule(str(tmp_path / f"bad{bad}"), s, ds)
        with pytest.raises(ValueError, match=r"outside the pack set .*step 2, slot 1"):
            train(_cfg(tmp_path, packed, f"o{bad}", schedule_file=f), tokenizer=TOK)


def test_sidecar_disagreements_fail_loudly(tmp_path, packed, no_model):
    ds = PackedDataset(packed, "train")
    sched, _ = balanced_schedule(ds.pack_languages(), 4, PER_STEP, seed=1)
    cases = {"sha": ({"schedule_sha256": "0" * 64}, "sha256"), "count": ({"n_packs": len(ds) + 1}, "n_packs"),
             "structure": ({"packed_structure_sha256": "1" * 64}, "pack structure"), "langs": ({"pack_lang_sha256": "2" * 64}, "pack languages"),
             "steps": ({"steps": 9}, "steps"), "projected": ({"projected": True}, "PROJECTED")}
    for name, (over, msg) in cases.items():
        f = _write_schedule(str(tmp_path / name), sched, ds, **over)
        with pytest.raises(ValueError, match=msg):
            train(_cfg(tmp_path, packed, "o" + name, schedule_file=f), tokenizer=TOK)


def test_missing_sidecar_is_refused_unless_allowed(tmp_path, packed):
    ds = PackedDataset(packed, "train")
    sched, _ = balanced_schedule(ds.pack_languages(), 3, PER_STEP, seed=1)
    f = _write_schedule(str(tmp_path / "x"), sched, ds)
    os.remove(tmp_path / "x" / "schedule.json")
    with pytest.raises(FileNotFoundError, match="sidecar"):
        load_schedule(f, ds, PER_STEP)
    arr, info = load_schedule(f, ds, PER_STEP, require_sidecar=False)
    assert info["sha256"] and info["checks_skipped"] and arr.shape == (3, PER_STEP)
    out = train(_cfg(tmp_path, packed, "ok", schedule_file=f, schedule_require_sidecar=False), tokenizer=TOK)
    assert out["steps"] == 3


def test_meta_json_of_a_plain_plan_output_serves_as_sidecar(tmp_path, packed):
    ds = PackedDataset(packed, "train")
    sched, _ = balanced_schedule(ds.pack_languages(), 3, PER_STEP, seed=1)
    f = _write_schedule(str(tmp_path / "m"), sched, ds)
    sc = json.load(open(tmp_path / "m" / "schedule.json"))
    os.remove(tmp_path / "m" / "schedule.json")
    json.dump({"schedule_sha256": sc["schedule_sha256"], "steps": 3, "per_step": PER_STEP, "n_packs": len(ds), "source": packed + "/train"},
              open(tmp_path / "m" / "meta.json", "w"))
    _, info = load_schedule(f, ds, PER_STEP)
    assert "schedule_sha256" in info["checks_passed"] and "packed_structure_sha256" in info["checks_skipped"]


def test_shape_dtype_and_option_conflicts(tmp_path, packed, no_model):
    ds = PackedDataset(packed, "train")
    sched, _ = balanced_schedule(ds.pack_languages(), 4, 8, seed=1)                 # 8 packs per step, the trainer takes 4
    f = _write_schedule(str(tmp_path / "w"), sched, ds)
    with pytest.raises(ValueError, match="takes 4 packs per step"):
        train(_cfg(tmp_path, packed, "o1", schedule_file=f), tokenizer=TOK)
    good, _ = balanced_schedule(ds.pack_languages(), 4, PER_STEP, seed=1)
    fg = _write_schedule(str(tmp_path / "g"), good, ds)
    with pytest.raises(ValueError, match="file length is the run length"):
        train(_cfg(tmp_path, packed, "o2", schedule_file=fg, max_steps=3), tokenizer=TOK)
    with pytest.raises(ValueError, match="replaces the internal samplers"):
        train(_cfg(tmp_path, packed, "o3", schedule_file=fg, language_alpha=0.5), tokenizer=TOK)
    ff = _write_schedule(str(tmp_path / "f"), good.astype(np.float64), ds)
    np.save(ff, good.astype(np.float64))
    with pytest.raises(ValueError, match="not an integer"):
        load_schedule(ff, ds, PER_STEP)
    with pytest.raises(ValueError, match="whole number of steps"):
        np.save(tmp_path / "g" / "flat.npy", good.reshape(-1)[:-1]); load_schedule(str(tmp_path / "g" / "flat.npy"), ds, PER_STEP)


def test_structure_hash_ignores_token_ids_but_not_the_packing(tmp_path, packed):
    ds = PackedDataset(packed, "train")
    h = packed_structure_sha256(ds)
    # an S2-like replay: same packs, other token ids -> same structure
    shutil.copytree(packed, tmp_path / "replayed")
    tk = tmp_path / "replayed" / "train" / "tokens.u32.bin"
    arr = np.fromfile(tk, dtype=np.uint32); arr[::7] += 1; arr.tofile(tk)
    assert packed_structure_sha256(PackedDataset(str(tmp_path / "replayed"), "train")) == h
    # other pack order / membership -> other structure
    raw = write_jsonl(str(tmp_path / "raw2.jsonl"), synthetic_rows(600, isos=ISOS, seed=6))
    prepare(PrepareConfig(input=raw, out_dir=str(tmp_path / "other"), seq_len=128, val_fraction=0.2, seed=1, pack_by_language=True), tokenizer=TOK)
    assert packed_structure_sha256(PackedDataset(str(tmp_path / "other"), "train")) != h


# ------------------------------------------------------------------------------------------------ S2: embeddings-only warm-up
def _init_state():
    from gemma_mt.modeling import build_tiny_model
    return {k: v.detach().clone() for k, v in build_tiny_model(attn_implementation="sdpa", dtype=torch.float32).state_dict().items()}


def _is_emb(k):
    return "embed_tokens" in k or "lm_head" in k


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_embed_only_warmup_moves_only_the_embedding(tmp_path, packed, optimizer):
    if optimizer == "muon" and not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon not available")
    ds = PackedDataset(packed, "train")
    sched, _ = balanced_schedule(ds.pack_languages(), 2, PER_STEP, seed=1)
    f = _write_schedule(str(tmp_path / "s"), sched, ds)
    out = train(_cfg(tmp_path, packed, "w", schedule_file=f, embed_only_steps=2, optimizer=optimizer, weight_decay=0.1), tokenizer=TOK)
    assert out["steps"] == 2
    init, now = _init_state(), _model(tmp_path, "w")
    assert all(torch.equal(init[k], now[k]) for k in init if not _is_emb(k)), "no decoder / norm weight may move during the warm-up (weight decay and momentum included)"
    assert not torch.equal(init["model.embed_tokens.weight"], now["model.embed_tokens.weight"])
    flags = [r["embed_only"] for r in map(json.loads, open(tmp_path / "w" / "metrics.jsonl")) if "embed_only" in r]
    assert flags and all(x == 1.0 for x in flags)


def test_embed_only_min_row_updates_only_the_new_rows(tmp_path, packed):
    ds = PackedDataset(packed, "train")
    sched, _ = balanced_schedule(ds.pack_languages(), 3, PER_STEP, seed=1)
    f = _write_schedule(str(tmp_path / "s"), sched, ds)
    init = _init_state()
    v = init["model.embed_tokens.weight"].shape[0]
    cut = v - 16
    train(_cfg(tmp_path, packed, "r", schedule_file=f, embed_only_steps=3, embed_only_min_row=cut), tokenizer=TOK)
    now = _model(tmp_path, "r")
    e0, e1 = init["model.embed_tokens.weight"], now["model.embed_tokens.weight"]
    assert torch.equal(e0[:cut], e1[:cut]), "rows below the cut must not move"
    assert not torch.equal(e0[cut:], e1[cut:]), "the new rows must move (the output softmax gives every row a gradient)"
    assert all(torch.equal(init[k], now[k]) for k in init if not _is_emb(k))


def test_warmup_steps_are_inside_the_schedule_and_the_decoder_trains_afterwards(tmp_path, packed):
    ds = PackedDataset(packed, "train")
    sched, _ = balanced_schedule(ds.pack_languages(), 5, PER_STEP, seed=1)
    f = _write_schedule(str(tmp_path / "s"), sched, ds)
    out = train(_cfg(tmp_path, packed, "a", schedule_file=f, embed_only_steps=2), tokenizer=TOK)
    assert out["steps"] == out["total_steps"] == 5, "the warm-up is part of the schedule, not extra steps"
    init, now = _init_state(), _model(tmp_path, "a")
    assert any(not torch.equal(init[k], now[k]) for k in init if not _is_emb(k)), "the decoder trains after the warm-up"
    flags = {r["step"]: r["embed_only"] for r in map(json.loads, open(tmp_path / "a" / "metrics.jsonl")) if "embed_only" in r}
    assert [flags[s] for s in (1, 2, 3, 4, 5)] == [1.0, 1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="longer than the run"):
        train(_cfg(tmp_path, packed, "b", schedule_file=f, embed_only_steps=6), tokenizer=TOK)


def test_warmup_resume_across_the_boundary_is_exact(tmp_path, packed):
    ds = PackedDataset(packed, "train")
    sched, _ = balanced_schedule(ds.pack_languages(), 6, PER_STEP, seed=1)
    f = _write_schedule(str(tmp_path / "s"), sched, ds)
    kw = dict(schedule_file=f, embed_only_steps=3, save_every=2)
    train(_cfg(tmp_path, packed, "full", **kw), tokenizer=TOK)
    train(_cfg(tmp_path, packed, "split", stop_after_steps=2, **kw), tokenizer=TOK)       # stop INSIDE the warm-up
    train(_cfg(tmp_path, packed, "split", **kw), tokenizer=TOK)
    assert _same(_model(tmp_path, "full"), _model(tmp_path, "split")) == 0.0
