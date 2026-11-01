# SPDX-License-Identifier: AGPL-3.0-or-later
"""Language-stratified sampling, one-language packs, the pack audit, the checkpoint pusher (owner 2026-10-14)."""
import os

import numpy as np
import pytest

from gemma_mt.data.audit_packed import audit
from gemma_mt.data.packed_dataset import PackedDataset, balanced_schedule
from gemma_mt.data.prepare_sft import PrepareConfig, prepare
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
from gemma_mt.languages import ALL_ISOS
from gemma_mt.smoke import synthetic_rows, write_jsonl

TOK = FakeGemmaTokenizer()


def _lang():
    return np.concatenate([np.full(2500, 0), np.full(1000, 1), np.full(120, 2), np.full(30, 3)])


def test_schedule_is_deterministic_and_resumable():
    a, _ = balanced_schedule(_lang(), 300, 8, seed=3)
    b, _ = balanced_schedule(_lang(), 300, 8, seed=3)
    c, _ = balanced_schedule(_lang(), 300, 8, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    # a longer schedule starts with the same steps (the step counter is all a resumed run needs)
    d, _ = balanced_schedule(_lang(), 400, 8, seed=3)
    assert np.array_equal(a, d[:300])


def test_every_language_appears_in_every_interval_and_caps_hold():
    lang = _lang()
    sch, st = balanced_schedule(lang, 200, 16, alpha=0.5, min_share=0.01, max_epochs=0, seed=1)
    for b in range(0, 200, 50):
        blk = lang[sch[b:b + 50].reshape(-1)]
        assert set(np.unique(blk)) == {0, 1, 2, 3}, "every language is drawn in every 50-step interval"
    # alpha=1 -> natural frequency, alpha=0 -> uniform
    nat = balanced_schedule(lang, 200, 16, alpha=1.0, min_share=0.0, max_epochs=0, seed=1)[1]
    uni = balanced_schedule(lang, 200, 16, alpha=0.0, min_share=0.0, max_epochs=0, seed=1)[1]
    assert nat[0]["share"] > 0.6 and 0.2 < uni[3]["share"] < 0.3
    # cap: a 30-pack language is used at most ~4 epochs (+ one step of overshoot)
    capped = balanced_schedule(lang, 400, 16, alpha=0.5, min_share=0.01, max_epochs=4, seed=1)[1]
    assert capped[3]["draws"] <= 4 * 30 + 16


def test_no_pack_repeats_before_the_language_is_exhausted():
    lang = np.concatenate([np.full(40, 0), np.full(10, 1)])
    sch, _ = balanced_schedule(lang, 20, 5, alpha=1.0, min_share=0.0, max_epochs=0, seed=0)
    flat = sch.reshape(-1)
    first = [int(i) for i in flat if lang[i] == 0][:40]
    assert len(set(first)) == len(first), "packs of one language go round without repeats within a pass"


def _prep(tmp_path, rows, **kw):
    raw = write_jsonl(str(tmp_path / "raw.jsonl"), rows)
    cfg = PrepareConfig(input=raw, out_dir=str(tmp_path / "out"), seq_len=400, val_fraction=0.2, **kw)
    meta = prepare(cfg, tokenizer=TOK)
    return meta, str(tmp_path / "out")


def test_pack_by_language_gives_pure_packs_and_keeps_every_sample(tmp_path):
    rows = synthetic_rows(400, isos=("yor", "hau", "ibo", "pcm"), seed=2)
    m_mixed, _ = _prep(tmp_path / "a", rows)
    m_pure, out = _prep(tmp_path / "b", rows, pack_by_language=True)
    for s in ("train", "val"):
        assert m_pure["splits"][s]["n_samples"] == m_mixed["splits"][s]["n_samples"]
        assert m_pure["splits"][s]["n_tokens"] == m_mixed["splits"][s]["n_tokens"]
    ds = PackedDataset(out, "train")
    langs = ds.pack_languages()
    assert len(langs) == ds.n_packs and set(int(x) for x in langs) <= {ALL_ISOS.index(i) for i in ("yor", "hau", "ibo", "pcm")}
    mixed_ds = PackedDataset(str(tmp_path / "a" / "out"), "train")
    with pytest.raises(ValueError):
        mixed_ds.pack_languages()


def test_audit_passes_on_good_data_and_flags_a_broken_pack(tmp_path):
    rows = synthetic_rows(200, isos=("yor", "hau"), seed=3)
    _, out = _prep(tmp_path, rows, pack_by_language=True)
    ds = PackedDataset(out, "train")
    rep = audit(ds, TOK, check_samples=500, expect_single_language=True, decode_examples=1)
    assert rep["n_problems"] == 0, rep["problems"]
    assert rep["samples_token_checked"] > 0 and 0 < rep["pack_fill_mean"] <= 1.0 and rep["examples"]
    # break it: corrupt one target length -> the audit must notice
    np.save(os.path.join(out, "train", "target_len.npy"), np.asarray(ds.target_len) + 1)
    bad = audit(PackedDataset(out, "train"), TOK, check_samples=500)
    assert bad["n_problems"] > 0


def test_decontaminate_drops_train_windows_sharing_a_validation_sentence(tmp_path):
    rows = [{"english": f"Sentence number {i} is about the market.", "target": f"Fake {i}", "iso": "pcm", "id": f"a{i}"} for i in range(60)]
    # the same sentences again inside bigger windows (what the x2 pass does)
    rows += [{"english": f"Sentence number {i} is about the market. Sentence number {i + 1} is about the market.",
              "target": f"Fake {i} Fake {i + 1}", "iso": "pcm", "id": f"b{i}"} for i in range(0, 58, 2)]
    plain, _ = _prep(tmp_path / "p", rows)
    clean, _ = _prep(tmp_path / "c", rows, decontaminate_isos=("pcm",))
    assert clean["dropped"].get("train_overlaps_val", 0) > 0
    assert clean["splits"]["train"]["n_samples"] < plain["splits"]["train"]["n_samples"]
    assert clean["splits"]["val"]["n_samples"] == plain["splits"]["val"]["n_samples"]


class FakeApi:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def create_repo(self, repo_id, **kw):
        self.calls.append(("create", repo_id, kw.get("private")))

    def upload_folder(self, **kw):
        if self.fail:
            raise ConnectionError("network down")
        self.calls.append(("upload", kw.get("path_in_repo"), kw["commit_message"]))

    def create_tag(self, repo_id, **kw):
        self.calls.append(("tag", kw["tag"], kw.get("revision")))

    def create_branch(self, repo_id, **kw):
        self.calls.append(("branch", kw["branch"]))


def test_pusher_commits_tags_and_survives_failures(tmp_path):
    from gemma_mt.hub.push_checkpoint import CheckpointPusher

    d = tmp_path / "stage"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x")
    logs = []
    api = FakeApi()
    p = CheckpointPusher("me/model", api=api, log=logs.append)
    assert p.push(str(d), 250, {"val_loss_macro": 3.1}, as_best=True, block=True)
    kinds = [c[0] for c in api.calls]
    assert kinds == ["create", "upload", "upload", "tag"] and api.calls[0][2] is True      # private repo
    assert ("tag", "step-250", None) in api.calls and any(c[1] == "best" for c in api.calls if c[0] == "upload")
    assert (d / "training_state.json").exists() and (d / "README.md").exists()
    # a failing network must not raise
    bad = CheckpointPusher("me/model", api=FakeApi(fail=True), log=logs.append)
    assert bad.push(str(d), 500, {}, block=True) and bad.last_error and "network down" in bad.last_error


def test_pusher_is_disabled_without_repo_or_token(tmp_path, monkeypatch):
    from gemma_mt.hub import push_checkpoint
    from gemma_mt.hub.push_checkpoint import CheckpointPusher

    monkeypatch.setattr(push_checkpoint, "_token_from_files", lambda: None)       # D-59: the rental has /workspace/sft_build/.hf_write_token and env.sh; the test must not see them
    saved = {k: os.environ.pop(k, None) for k in ("HF_WRITE_TOKEN", "HF_TOKEN", "HF_API_KEY")}
    try:
        d = tmp_path / "s"
        d.mkdir()
        assert CheckpointPusher(None).push(str(d), 1, {}) is False
        logs = []
        assert CheckpointPusher("me/m", log=logs.append).push(str(d), 1, {}) is False and logs
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_pusher_cleanup_removes_only_after_success(tmp_path):
    from gemma_mt.hub.push_checkpoint import CheckpointPusher

    d = tmp_path / "stage"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x")
    ok = CheckpointPusher("me/m", api=FakeApi(), log=lambda *_: None)
    ok.push(str(d), 10, {}, block=True, cleanup=True)
    assert not d.exists(), "the staging copy is deleted after a successful upload"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x")
    bad = CheckpointPusher("me/m", api=FakeApi(fail=True), log=lambda *_: None)
    bad.push(str(d), 11, {}, block=True, cleanup=True)
    assert d.exists(), "kept for a retry when the upload failed"


def test_pusher_can_target_a_branch_with_its_own_tag_prefix(tmp_path):
    from gemma_mt.hub.push_checkpoint import CheckpointPusher

    d = tmp_path / "stage"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"x")
    api = FakeApi()
    p = CheckpointPusher("me/model", api=api, stage="rl", branch="rl", tag_prefix="rl-step", log=lambda *_: None)
    assert p.push(str(d), 40, {"eval_reward": 0.7}, block=True)
    assert ("branch", "rl") in api.calls and ("tag", "rl-step-40", "rl") in api.calls


# ---- UniMax budget schedule
def test_unimax_allocation_constraints():
    from gemma_mt.data.packed_dataset import unimax_allocation

    n = np.array([5000, 1200, 800, 300, 40, 10])
    for budget in (int(n.sum()), int(1.4 * n.sum()), int(2 * n.sum())):
        a = unimax_allocation(n, budget, min_epochs=1.0, max_epochs=2.0)
        assert a.sum() == budget
        assert (a >= n).all() and (a <= 2 * n).all()
    a = unimax_allocation(n, int(1.3 * n.sum()), min_epochs=1.0, max_epochs=2.0)
    assert a[-1] == 2 * n[-1] and a[0] == n[0]                       # small language at its cap, the biggest at its floor
    w = np.array([1, 1, 1, 1, 1, 1.0])
    w2 = np.array([1, 4, 1, 1, 1, 1.0])
    assert unimax_allocation(n, int(1.3 * n.sum()), weights=w2)[1] >= unimax_allocation(n, int(1.3 * n.sum()), weights=w)[1]
    with pytest.raises(ValueError):
        unimax_allocation(n, int(n.sum()) - 1)


def test_unimax_schedule_sees_every_pack_and_is_deterministic():
    from gemma_mt.data.packed_dataset import unimax_schedule

    lang = np.repeat(np.arange(5), [400, 120, 60, 12, 8])
    n_steps, per = 70, 16                                            # 1120 draws of 600 packs: between 1 and 2 epochs
    a, st = unimax_schedule(lang, n_steps, per, min_epochs=1.0, max_epochs=2.0, weights={0: 2.0}, seed=5)
    b, _ = unimax_schedule(lang, n_steps, per, min_epochs=1.0, max_epochs=2.0, weights={0: 2.0}, seed=5)
    assert a.shape == (n_steps, per) and (a == b).all()
    assert len(np.unique(a)) == len(lang)                            # every pack used at least once
    assert sum(v["draws"] for v in st.values()) == n_steps * per
    assert all(1.0 <= v["epochs"] <= 2.0 + 1e-9 for v in st.values())
    first_half = np.unique(lang[a[: n_steps // 2].ravel()])
    assert len(first_half) == 5                                      # every language is present in every half of the run


def test_unimax_unseen_packs_come_first():
    from gemma_mt.data.packed_dataset import unimax_schedule

    lang = np.repeat(np.arange(3), [300, 100, 20])
    seen = np.zeros(len(lang), bool)
    seen[::2] = True                                                 # half of every language was used before
    sch, st = unimax_schedule(lang, 60, 10, min_epochs=1.0, max_epochs=2.0, seen=seen, seed=2)
    flat = sch.ravel()
    for l in range(3):
        a = st[l]["draws"]
        mine = flat[lang[flat] == l][: len(np.flatnonzero(lang == l))]          # the first pass of this language, in draw order
        n_unseen = int((~seen[lang == l]).sum())
        assert not seen[mine[:n_unseen]].any()                       # new packs first
        assert len(np.unique(mine)) == len(mine)                    # and the first pass visits every pack once
    assert len(np.unique(flat)) == len(lang)
