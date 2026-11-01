# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end test of the inter-African SFT stages on a tiny synthetic corpus (needs pyarrow; the real tokenizer is replaced by the byte-level stub)."""
import json
import os
import sys

import numpy as np
import pytest

from gemma_mt import format as fmt
from gemma_mt.afri2afri import index as ix, packing as PK, prompt as P, select_pairs as SP, windows as W
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
from gemma_mt.languages import ALL_ISOS

TOK = FakeGemmaTokenizer()


def test_encode_batch_equals_build_example_x2x():
    items = [("Ẹ kaaro, bawo ni?", "yor", "hau", "Ina kwana, lafiya?"), ("Ina kwana", "hau", "yor", "Ẹ kaaro"), ("Good morning", "eng", "ibo", "Ụtụtụ ọma")]
    got = PK.encode_batch(TOK, items)
    for (text, src, tgt, tr), (ids, pl, tl) in zip(items, got):
        ex = P.build_example_x2x(text, src, tgt, tr, TOK)
        assert ids == ex.input_ids and pl == ex.prompt_len and tl == ex.target_len


def test_write_split_layout_and_loadable(tmp_path):
    pytest.importorskip("numpy")
    store = PK.RawStore(str(tmp_path / "tmp"))
    rng = np.random.default_rng(0)
    yor, hau = ALL_ISOS.index("yor"), ALL_ISOS.index("hau")
    for k in range(60):
        ids, pl, tl = PK.encode_batch(TOK, [(f"Ẹ kaaro {k}", "yor", "hau", f"Ina kwana {k}" * int(rng.integers(1, 5)))])[0]
        store.add(ids, pl, tl, hau, yor, k % 2)
    meta = PK.write_split(store, str(tmp_path / "train"), seq_len=512, buffer_size=100, seed=0)
    assert meta["n_samples"] == 60 and meta["packing_efficiency"] > 0.5 and "yor>hau" in meta["per_direction"]
    d = tmp_path / "train"
    for f in ("tokens.u32.bin", "sample_offset.npy", "sample_len.npy", "prompt_len.npy", "target_len.npy", "iso_id.npy", "src_iso_id.npy", "prompt_form.npy", "direction.npy", "pack_start.npy"):
        assert (d / f).exists(), f
    assert (np.load(d / "direction.npy") == P.DIRECTION_X2X).all()
    assert meta["prompt_forms"] == {"explicit": 30, "implicit": 30} and set(np.load(d / "prompt_form.npy").tolist()) == {0, 1}
    assert (np.load(d / "src_iso_id.npy") == yor).all() and (np.load(d / "iso_id.npy") == hau).all()
    # the PackedDataset of the en->X pipeline reads it (labels only on the target)
    from gemma_mt.data.packed_dataset import PackedDataset

    (tmp_path / "meta.json").write_text(json.dumps({"format_version": 1, "dtype": "uint32", "seq_len": 512, "iso_table": list(ALL_ISOS),
                                                    "tokenizer": {"fingerprint": "x", "pad_id": 0}}))
    ds = PackedDataset(str(tmp_path), "train", pad=False)
    item = ds[0]
    assert item["n_samples"] >= 1 and (item["labels"] != -100).any()


def test_index_shard_and_pivot_join_on_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rows = [("Hello, World!", "Bawo ni ayé", "yor"), ("hello world", "Sannu duniya", "hau"), ("Another sentence here", "Gbolohun miiran", "yor"),
            ("Another sentence here", "Wani jumla", "hau"), ("Lonely sentence", "Nanu", "ibo")]
    t = pa.table({"english": [r[0] for r in rows], "target": [r[1] for r in rows], "iso": [r[2] for r in rows], "source_dataset": ["a/b"] * 5,
                  "licence": ["cc-by"] * 5, "flipped": [False] * 5, "qe": np.array([0.5] * 5, np.float32)})
    p = str(tmp_path / "pairs-0000.parquet")
    pq.write_table(t, p)
    arr = ix.index_shard(p, {})
    assert arr["key"][0] == arr["key"][1] != arr["key"][4]                               # "Hello, World!" ~ "hello world"
    assert arr["key"][2] == arr["key"][3]
    from gemma_mt.afri2afri import pivot as pv

    best = pv.best_per_key_iso(arr["key"], arr["iso"], arr["qe"])
    st, sz = pv.group_bounds(arr["key"][best])
    assert sorted(sz.tolist()) == [1, 2, 2]


def test_select_records_end_to_end_properties():
    rng = np.random.default_rng(0)
    rows = []
    for g in range(400):
        for l in sorted(rng.permutation(6)[: int(rng.integers(2, 6))]):
            rows.append((g, int(l), int(rng.integers(0, 4)), int(rng.integers(1, 2 ** 62)), g + 1))
    a = np.array(rows, dtype=np.int64)
    group, iso, tier, tkey, key = a[:, 0], a[:, 1].astype(np.uint8), a[:, 2].astype(np.uint8), a[:, 3].astype(np.uint64), a[:, 4].astype(np.uint64)
    rec = SP.records_from_groups(group, iso, tier, tkey, key, np.ones(len(group), bool))
    assert (rec["iso_a"] < rec["iso_b"]).all()
    r = SP.select_records(rec, 6, budget=400, max_epochs=2.0, val_frac=0.05, val_cap=3)
    tr, va = r["train"], r["val"]
    assert not (set(rec["gkey"][tr].tolist()) & set(rec["gkey"][va].tolist()))        # a validation sentence is never trained on with another partner
    pid = r["pid"]
    for p, info in r["pid_info"].items():
        assert int((pid[tr] == p).sum()) == info["distinct"] <= info["available"]
        assert info["alloc"] <= 2.0 * info["available"] + 1e-6
    # direct beats pivot and a better class beats a worse one on identical text pairs
    rec2 = SP.concat_records(rec, {k: v[:5].copy() for k, v in rec.items()})
    rec2["origin"][-5:] = 1
    r2 = SP.select_records(rec2, 6, budget=10 ** 6, max_epochs=1.0, val_frac=0.0)
    assert r2["dropped"]["duplicate_text_pair"] >= 5
    kept = set(r2["train"].tolist())
    assert all((len(rec["iso_a"]) + j) in kept for j in range(5))


def test_mix64_is_deterministic_and_unit_fraction_uniform():
    x = np.arange(1, 5001, dtype=np.uint64)
    f = SP.unit_fraction(x, 3)
    assert (f == SP.unit_fraction(x, 3)).all() and (f != SP.unit_fraction(x, 4)).any()
    assert 0.47 < float(f.mean()) < 0.53 and f.min() >= 0 and f.max() < 1


def _load_script(name):
    import importlib.util

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(here, "scripts"))
    spec = importlib.util.spec_from_file_location(name, os.path.join(here, "scripts", name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def test_add_form_column_is_deterministic_and_half_explicit(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    B = _load_script("build_afri2afri_sft")
    n = 4000
    t = pa.table({"group_key": [f"{i:016x}" for i in range(n)], "src_iso": ["yor"] * n, "tgt_iso": ["hau"] * n})
    for split in ("train", "val"):
        pq.write_table(t, str(tmp_path / f"pairs_{split}.parquet"))
    ns = type("A", (), dict(out=str(tmp_path), explicit_share=0.5))()
    B.cmd_add_form(ns)
    f1 = pq.read_table(str(tmp_path / "pairs_train.parquet"))["prompt_form"].to_pylist()
    B.cmd_add_form(ns)
    f2 = pq.read_table(str(tmp_path / "pairs_train.parquet"))["prompt_form"].to_pylist()
    assert f1 == f2 and set(f1) == {"explicit", "implicit"} and 0.46 < f1.count("explicit") / n < 0.54


def test_declared_licence_rules():
    pytest.importorskip("pyarrow")
    F = _load_script("build_afri2afri_full")
    scan = {"a/b": {"declared": "cc-by-sa-4.0", "private": False}, "p/q": {"declared": None, "private": True}}
    assert F.declared_info(scan, "a/b:cfg") == ("a/b", "cc-by-sa-4.0", False)
    assert F.declared_info(scan, "p/q") == ("p/q", "unknown", True)
    assert F.declared_info(scan, "michsethowusu/english-zulu_sentence-pairs_mt560")[1] == "cc-by-4.0"          # read from the card audit
    assert F.declared_info(scan, "x/y")[1] == "unknown"


def test_replay_samples_match_the_en2x_and_x2en_builders():
    items = [("Good morning", None, "yor", "E kaaro"), ("E kaaro", None, "eng", "Good morning"), ("Good morning", "eng", "yor", "E kaaro"), ("E kaaro", "yor", "eng", "Good morning")]
    got = PK.encode_batch(TOK, items)
    e1 = fmt.build_example("Good morning", "yor", "E kaaro", TOK)
    e2 = fmt.build_example("E kaaro", "eng", "Good morning", TOK)                # the reverse direction of the en->X model: `Translate the text to English:`
    assert got[0][0] == e1.input_ids and got[1][0] == e2.input_ids
    assert "from English to Yoruba" in TOK.decode(got[2][0][: got[2][1]]) and "from Yoruba to English" in TOK.decode(got[3][0][: got[3][1]])


def test_write_split_keeps_per_sample_direction(tmp_path):
    store = PK.RawStore(str(tmp_path / "t"))
    yor = ALL_ISOS.index("yor")
    eng = len(ALL_ISOS)
    for k, (d, src) in enumerate([(0, eng), (1, yor), (2, ALL_ISOS.index("hau"))] * 5):
        ids, pl, tl = PK.encode_batch(TOK, [(f"t{k}", None, "yor", f"u{k}")])[0]
        store.add(ids, pl, tl, yor, src, 1, direction=d)
    meta = PK.write_split(store, str(tmp_path / "o"), seq_len=256, buffer_size=50, seed=0)
    assert meta["directions"] == {"en2x": 5, "x2en": 5, "x2x": 5}
    assert set(np.load(tmp_path / "o" / "direction.npy").tolist()) == {0, 1, 2}
    assert {"eng>yor", "yor>eng", "hau>yor"} <= set(meta["per_direction"])
