# SPDX-License-Identifier: AGPL-3.0-or-later
"""sampler.py `subset` (A2) and `select` (G0): derived schedules of the A0 schedule, on toy packs; the sidecar / hash logic keeps working."""
import importlib.util
import json
import os

import numpy as np
import pytest

pytest.importorskip("torch")

from gemma_mt.data.packed_dataset import PackedDataset
from gemma_mt.data.prepare_sft import PrepareConfig, prepare
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
from gemma_mt.languages import ALL_ISOS
from gemma_mt.smoke import synthetic_rows, write_jsonl
from gemma_mt.train_sft import load_schedule, packed_structure_sha256

ISOS = ("yor", "hau", "ibo", "amh", "fon", "ewe", "tir")
SAMPLER_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "sft_clean", "chain", "sampler.py")
PER_STEP = 4


@pytest.fixture(scope="module")
def sampler():
    if not os.path.exists(SAMPLER_PY):
        pytest.skip("sampler.py is not next to gemma_mt")
    spec = importlib.util.spec_from_file_location("chain_sampler_derived", os.path.abspath(SAMPLER_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def a0(tmp_path_factory, sampler):
    d = tmp_path_factory.mktemp("derived")
    raw = write_jsonl(str(d / "raw.jsonl"), synthetic_rows(1500, isos=ISOS, seed=9))
    prepare(PrepareConfig(input=raw, out_dir=str(d / "packed"), seq_len=128, val_fraction=0.2, seed=1, pack_by_language=True), tokenizer=FakeGemmaTokenizer())
    args = type("A", (), dict(stage="A", packed_dir=str(d / "packed" / "train"), packs_csv=None, out=str(d / "A0"), seed=7, steps=120, per_step=PER_STEP, alpha=0.5, budget_tokens=0))()
    sampler.plan(args)
    return d


def _ns(**kw):
    return type("A", (), kw)()


def test_subset_is_stratified_seeded_ordered_and_loadable(tmp_path, a0, sampler):
    ds = PackedDataset(str(a0 / "packed"), "train")
    lang = ds.pack_languages()
    A0 = np.load(a0 / "A0" / "schedule.npy")
    flat = A0.reshape(-1)
    out = []
    for seed, name in ((1, "x"), (1, "y"), (2, "z")):
        sampler.subset(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / name), frac=0.2, stratify="language", seed=seed, languages=None, name="A2"))
        out.append(np.load(tmp_path / name / "schedule.npy"))
    assert np.array_equal(out[0], out[1]) and not np.array_equal(out[0], out[2]), "seeded"
    s, pos = out[0], np.load(tmp_path / "x" / "a0_positions.npy")
    assert s.shape[1] == PER_STEP and np.array_equal(flat[pos], s.reshape(-1)) and np.all(np.diff(pos) > 0), "same order as A0, positions inside A0"
    rep = json.load(open(tmp_path / "x" / "subset.json"))
    drawn = {ALL_ISOS[i]: int((lang[flat] == i).sum()) for i in np.unique(lang[flat])}
    for iso, n in drawn.items():
        want = max(1, int(round(0.2 * n)))
        assert rep["per_language"][iso]["parent_draws"] == n
        assert want - PER_STEP < rep["per_language"][iso]["draws"] <= want, (iso, want, rep["per_language"][iso])   # only the dropped partial step can remove draws
    assert 0.18 < s.size / flat.size < 0.22 and rep["dropped_incomplete_step_draws"] < PER_STEP
    # sidecar and trainer-side checks still work (hash of the pack structure inherited from A0)
    sc = json.load(open(tmp_path / "x" / "schedule.json"))
    assert sc["packed_structure_sha256"] == packed_structure_sha256(ds) and sc["frac"] == 0.2 and sc["stratify"] == "language" and sc["a0_schedule_sha256"]
    arr, info = load_schedule(str(tmp_path / "x" / "schedule.npy"), ds, PER_STEP)
    assert arr.shape == s.shape and "packed_structure_sha256" in info["checks_passed"] and "pack_lang_sha256" in info["checks_passed"]


def test_subset_every_language_keeps_a_share_even_when_small(tmp_path, a0, sampler):
    ds = PackedDataset(str(a0 / "packed"), "train")
    lang = ds.pack_languages()
    sampler.subset(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "s"), frac=0.02, stratify="language", seed=3, languages=None, name="A2"))
    rep = json.load(open(tmp_path / "s" / "subset.json"))
    A0 = np.load(a0 / "A0" / "schedule.npy").reshape(-1)
    want = sum(max(1, int(round(0.02 * rep["per_language"][iso]["parent_draws"]))) for iso in rep["per_language"])
    # the floor of one draw per language holds before the incomplete last step is dropped (exactly that many draws were selected)
    assert rep["draws"] + rep["dropped_incomplete_step_draws"] == want and want >= len(rep["per_language"])


def test_subset_stratify_none_and_bad_fraction(tmp_path, a0, sampler):
    rep = sampler.subset(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "n"), frac=0.5, stratify="none", seed=1, languages=None, name="A2"))
    assert abs(rep["draws"] - 0.5 * rep["parent_draws"]) < PER_STEP
    for bad in (0.0, 1.5):
        with pytest.raises(SystemExit):
            sampler.subset(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "b"), frac=bad, stratify="language", seed=1, languages=None, name="A2"))


def test_select_languages_g0_is_the_restriction_of_a0(tmp_path, a0, sampler):
    ds = PackedDataset(str(a0 / "packed"), "train")
    lang = ds.pack_languages()
    A0 = np.load(a0 / "A0" / "schedule.npy").reshape(-1)
    rep = sampler.select(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "G0"), languages="amh,tir", name="G0"))
    g0 = np.load(tmp_path / "G0" / "schedule.npy").reshape(-1)
    ids = [ALL_ISOS.index("amh"), ALL_ISOS.index("tir")]
    expect = A0[np.isin(lang[A0], ids)]
    assert np.array_equal(g0, expect[:g0.size]) and 0 <= expect.size - g0.size < PER_STEP
    assert set(np.unique(lang[g0]).tolist()) == set(ids)
    for iso in ("amh", "tir"):                                             # per-language exposure = A0's (up to the dropped partial step)
        assert 0 <= rep["per_language"][iso]["parent_draws"] - rep["per_language"][iso]["draws"] < PER_STEP
    load_schedule(str(tmp_path / "G0" / "schedule.npy"), ds, PER_STEP)
    with pytest.raises(SystemExit):
        sampler.select(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "none"), languages="zul", name="G0"))


def test_arms_accepts_a_language_list_and_default_is_unchanged(tmp_path, a0, sampler, monkeypatch):
    monkeypatch.setattr(sampler, "SPECIALIST", ["amh", "fon"])
    sampler.arms(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "d"), languages=None))
    sampler.arms(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "e"), languages="amh,fon"))
    assert np.array_equal(np.load(tmp_path / "d" / "S1" / "schedule.npy"), np.load(tmp_path / "e" / "S1" / "schedule.npy"))
    sampler.arms(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "f"), languages="amh"))
    assert np.load(tmp_path / "f" / "S1" / "schedule.npy").size < np.load(tmp_path / "e" / "S1" / "schedule.npy").size


def test_a2_subset_of_a_subset_chain_keeps_the_hash_logic(tmp_path, a0, sampler):
    ds = PackedDataset(str(a0 / "packed"), "train")
    sampler.subset(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "sub"), frac=0.5, stratify="language", seed=1, languages="yor,hau,amh", name="A2"))
    sc = json.load(open(tmp_path / "sub" / "schedule.json"))
    assert sc["languages"] == "yor,hau,amh"
    load_schedule(str(tmp_path / "sub" / "schedule.npy"), ds, PER_STEP)


def test_head_is_the_exact_prefix_of_a0_and_loads(tmp_path, a0, sampler):
    ds = PackedDataset(str(a0 / "packed"), "train")
    rep = sampler.head(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "h"), steps=30, name="A0_head300"))
    A0 = np.load(a0 / "A0" / "schedule.npy")
    h = np.load(tmp_path / "h" / "schedule.npy")
    assert rep["steps"] == 30 and np.array_equal(h, A0[:30]) and np.array_equal(np.load(tmp_path / "h" / "a0_positions.npy"), np.arange(30 * PER_STEP))
    arr, info = load_schedule(str(tmp_path / "h" / "schedule.npy"), ds, PER_STEP)
    assert "packed_structure_sha256" in info["checks_passed"]
    assert sampler.head(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "h2"), steps=10 ** 6, name="x"))["steps"] == A0.shape[0], "capped at the schedule length"


def test_stage_b_plan_with_seen_from_draws_a0_packs_last(tmp_path, a0, sampler):
    ds = PackedDataset(str(a0 / "packed"), "train")
    lang = ds.pack_languages()
    prior = tmp_path / "prior"; prior.mkdir()                                       # an earlier phase that used only the even packs
    np.save(prior / "schedule.npy", np.arange(0, len(ds), 2)[: (len(ds) // 2 // 4) * 4].reshape(-1, 4).astype(np.int64))
    seen = np.zeros(len(ds), bool); seen[np.load(prior / "schedule.npy").reshape(-1)] = True
    assert 0 < seen.sum() < len(ds)
    mk = lambda out, sf: sampler.plan(_ns(stage="B", packed_dir=str(a0 / "packed" / "train"), packs_csv=None, out=str(tmp_path / out), seed=5, steps=40, per_step=PER_STEP,
                                         alpha=0.5, budget_tokens=0, seen_from=sf))
    mk("plain", None)
    mk("seen", str(prior))
    flat = np.load(tmp_path / "seen" / "schedule.npy").reshape(-1)
    assert not np.array_equal(flat, np.load(tmp_path / "plain" / "schedule.npy").reshape(-1))
    # inside every language the first appearance of an unseen pack comes before the first appearance of a seen pack that is not also... (unseen packs are drawn first)
    for l in np.unique(lang[flat]):
        d = flat[lang[flat] == l]
        first = list(dict.fromkeys(d.tolist()))[: int((lang == l).sum())]
        flags = [bool(seen[p]) for p in first]
        unseen_n = flags.count(False)
        assert all(not f for f in flags[:unseen_n]), (ALL_ISOS[l], flags)
    assert json.load(open(tmp_path / "seen" / "schedule.json"))["packed_structure_sha256"] == packed_structure_sha256(ds)


def test_subset_floor_draws_and_systematic_spacing(tmp_path, a0, sampler):
    ds = PackedDataset(str(a0 / "packed"), "train")
    lang = ds.pack_languages()
    A0 = np.load(a0 / "A0" / "schedule.npy").reshape(-1)
    n = {ALL_ISOS[i]: int((lang[A0] == i).sum()) for i in np.unique(lang[A0])}
    floor = int(sorted(n.values())[2]) + 5                                  # larger than 25% of the three smallest languages, smaller than all draws of some
    rep = sampler.subset(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "f"), frac=0.25, floor_draws=floor, stratify="language", method="systematic", seed=11, languages=None, name="A2"))
    pl = rep["per_language"]
    for iso, c in n.items():
        want = max(int(round(0.25 * c)), min(c, floor))
        assert want - PER_STEP < pl[iso]["draws"] <= want, (iso, want, pl[iso])
        assert pl[iso]["floored"] == (want > max(1, int(round(0.25 * c))))
    assert rep["floored_languages"] and rep["n_languages"] == len(n) and rep["padded_tokens"] == rep["steps"] * PER_STEP * 4096
    # systematic: the kept draws of a language are evenly spread over its A0 sequence (no two consecutive kept draws are more than 2 * n/k apart)
    sel = np.load(tmp_path / "f" / "a0_positions.npy")
    for i in np.unique(lang[A0]):
        mine = np.flatnonzero(lang[A0[sel]] == i)
        pos_in_lang = np.searchsorted(np.flatnonzero(lang[A0] == i), sel[mine])
        k, c = len(pos_in_lang), n[ALL_ISOS[i]]
        if k > 2:
            assert np.diff(pos_in_lang).max() <= 2 * np.ceil(c / (k + PER_STEP)) + 2 * PER_STEP, (ALL_ISOS[i], np.diff(pos_in_lang).max(), c, k)
    # one subset file serves every arm: identical bytes and sidecar hash on a rebuild with the same arguments
    sampler.subset(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "g"), frac=0.25, floor_draws=floor, stratify="language", method="systematic", seed=11, languages=None, name="A2"))
    assert json.load(open(tmp_path / "f" / "schedule.json"))["schedule_sha256"] == json.load(open(tmp_path / "g" / "schedule.json"))["schedule_sha256"]
    r2 = sampler.subset(_ns(schedule=str(a0 / "A0"), out=str(tmp_path / "h"), frac=0.25, floor_draws=floor, stratify="language", method="random", seed=11, languages=None, name="A2"))
    assert r2["draws"] == rep["draws"] and json.load(open(tmp_path / "h" / "schedule.json"))["schedule_sha256"] != json.load(open(tmp_path / "f" / "schedule.json"))["schedule_sha256"]
    load_schedule(str(tmp_path / "f" / "schedule.npy"), ds, PER_STEP)
