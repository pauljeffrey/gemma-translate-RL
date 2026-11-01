# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import os

import numpy as np
import pytest

from gemma_mt import format as fmt
from gemma_mt.data.collate import PackedCollator, collate_flatten, collate_padded
from gemma_mt.data.packed_dataset import PackedDataset, batch_indices, steps_per_epoch
from gemma_mt.data.prepare_sft import PrepareConfig, normalise_row, prepare, split_fraction, tokenizer_fingerprint
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
from gemma_mt.languages import ALL_ISOS
from gemma_mt.smoke import synthetic_rows, write_jsonl

TOK = FakeGemmaTokenizer()


def _build(tmp_path, rows, **kw):
    raw = write_jsonl(str(tmp_path / "raw.jsonl"), rows)
    cfg = PrepareConfig(input=raw, out_dir=str(tmp_path / "out"), seq_len=kw.pop("seq_len", 200),
                        val_fraction=kw.pop("val_fraction", 0.2), seed=kw.pop("seed", 0), **kw)
    meta = prepare(cfg, tokenizer=TOK)
    return meta, str(tmp_path / "out")


def test_normalise_row_reasons():
    assert normalise_row({"english": "a", "target": "b", "iso": "yor"})[0] == ("a", "b", "yor", None)
    assert normalise_row({"english": "a", "target": "b", "iso": "SWA"})[0][2] == "swh"
    assert normalise_row({"english": "", "target": "b", "iso": "yor"})[1] == "empty_text"
    assert normalise_row({"english": "a", "target": "b", "iso": "xx1"})[1] == "unknown_iso"
    assert normalise_row({"english": "a", "iso": "yor"})[1] == "missing_field"


def test_prepare_counts_split_and_no_leakage(tmp_path):
    rows = synthetic_rows(300, seed=5)
    rows += [{"english": "", "target": "x", "iso": "yor"}, {"english": "a", "target": "b", "iso": "qqq"}]
    rows += rows[:5]                                          # exact duplicates
    meta, out = _build(tmp_path, rows)
    assert meta["dropped"]["empty_text"] == 1 and meta["dropped"]["unknown_iso"] == 1
    assert meta["dropped"]["duplicate"] >= 5
    tr, va = meta["splits"]["train"], meta["splits"]["val"]
    assert tr["n_samples"] > 0 and va["n_samples"] > 0
    # english sentences are split by hash of the english text: no sentence in both splits
    ds_tr, ds_va = PackedDataset(out, "train"), PackedDataset(out, "val")
    def english_of(ds):
        out_set = set()
        for i in range(len(ds)):
            item = ds[i]
            text = TOK.decode(item["input_ids"][: item["length"]])
            for part in text.split("<bos>")[1:]:
                out_set.add(part.split("Translate the text to ")[1].split(":  ", 1)[1].split("<end_of_turn>")[0])
        return out_set
    assert english_of(ds_tr).isdisjoint(english_of(ds_va))
    assert ds_tr.n_samples == tr["n_samples"] and ds_va.n_samples == va["n_samples"]


def test_split_is_deterministic_and_order_independent(tmp_path):
    rows = synthetic_rows(120, seed=1)
    m1, _ = _build(tmp_path, rows)
    p2 = tmp_path / "second"
    p2.mkdir()
    m2, _ = _build(p2, list(reversed(rows)))
    assert m1["splits"]["train"]["n_samples"] == m2["splits"]["train"]["n_samples"]
    assert split_fraction("hello", 0) == split_fraction("hello", 0)
    assert split_fraction("hello", 0) != split_fraction("hello", 1)


def test_packed_sequences_labels_positions_segments(tmp_path):
    meta, out = _build(tmp_path, synthetic_rows(150, seed=3), seq_len=256)
    ds = PackedDataset(out, "train", pad=True)
    assert meta["splits"]["train"]["packing_efficiency"] > 0.5
    for i in range(len(ds)):
        it = ds[i]
        L = len(it["input_ids"])
        assert L == 256 == len(it["labels"]) == len(it["position_ids"]) == len(it["segment_ids"])
        n = it["length"]
        seg = it["segment_ids"]
        pos = it["position_ids"]
        # padding: segment 0, label -100, pad id
        assert (seg[n:] == 0).all() and (it["labels"][n:] == -100).all() and (it["input_ids"][n:] == ds.pad_id).all()
        # positions restart at 0 at every sample start and count up inside it
        cu = it["cu_seqlens"]
        for a, b in zip(cu[:-1], cu[1:]):
            assert (pos[a:b] == np.arange(b - a)).all()
        # every sample starts with <bos> and its only labelled tokens end with <end_of_turn>
        for s, (a, b) in enumerate(zip(cu[:-1], cu[1:]), start=1):
            if a >= n:
                continue
            assert it["input_ids"][a] == TOK.bos_token_id
            assert (seg[a:b] == s).all()
            lab = it["labels"][a:b]
            idx = np.nonzero(lab != -100)[0]
            assert len(idx) > 0 and lab[idx[-1]] == TOK.convert_tokens_to_ids("<end_of_turn>")
            assert (idx == np.arange(idx[0], idx[-1] + 1)).all(), "labelled tokens must be one contiguous span"


def test_loader_reproduces_build_example(tmp_path):
    rows = synthetic_rows(40, seed=9)
    meta, out = _build(tmp_path, rows, seq_len=512, val_fraction=0.0)
    ds = PackedDataset(out, "train")
    got = []
    for i in range(len(ds)):
        it = ds[i]
        cu = it["cu_seqlens"]
        for a, b in zip(cu[:-1], cu[1:]):
            if a < it["length"]:
                got.append(tuple(it["input_ids"][a:b].tolist()))
    want = {tuple(fmt.build_example(r["english"], r["iso"], r["target"], TOK).input_ids) for r in rows}
    assert set(got) == want


def test_too_long_samples_are_dropped_not_truncated(tmp_path):
    rows = synthetic_rows(30, seed=2)
    rows.append({"english": "word " * 400, "target": "x " * 400, "iso": "yor"})
    meta, out = _build(tmp_path, rows, seq_len=256, val_fraction=0.0)
    assert meta["dropped"]["too_long"] == 1


def test_tokenizer_fingerprint_guard(tmp_path):
    meta, out = _build(tmp_path, synthetic_rows(30), seq_len=256)
    PackedDataset(out, "train", expect_tokenizer_fingerprint=tokenizer_fingerprint(TOK))
    with pytest.raises(ValueError):
        PackedDataset(out, "train", expect_tokenizer_fingerprint="0" * 64)
    m = json.load(open(os.path.join(out, "meta.json")))
    assert m["dtype"] == "uint32" and m["tokenizer"]["end_of_turn_id"] == 106
    assert os.path.getsize(os.path.join(out, "train", "tokens.u32.bin")) % 4 == 0


def test_min_val_per_language(tmp_path):
    rows = synthetic_rows(400, isos=("yor",), seed=0)
    rows += [{"english": f"unique sentence number {i} here", "target": f"t{i}", "iso": "hau"} for i in range(300)]
    meta, _ = _build(tmp_path, rows, val_fraction=0.0, min_val_per_language=5)
    assert meta["splits"]["val"]["per_language"].get("hau", {}).get("samples", 0) >= 5


def test_collators_padded_and_flatten(tmp_path):
    meta, out = _build(tmp_path, synthetic_rows(80, seed=4), seq_len=200, val_fraction=0.0)
    ds_pad, ds_raw = PackedDataset(out, "train", pad=True), PackedDataset(out, "train", pad=False)
    k = min(3, len(ds_pad))
    b = collate_padded([ds_pad[i] for i in range(k)], pad_id=0)
    assert b["input_ids"].shape == (k, 200)
    assert b["num_target_tokens"] == int((b["labels"][:, 1:] != -100).sum())
    f = collate_flatten([ds_raw[i] for i in range(k)])
    total = sum(ds_raw[i]["length"] for i in range(k))
    assert f["input_ids"].shape == (1, total)
    assert f["cu_seqlens"][-1] == total and f["cu_seqlens"][0] == 0
    assert (np.diff(f["cu_seqlens"]) > 0).all()
    assert f["max_seqlen"] == int(np.diff(f["cu_seqlens"]).max())
    # position ids restart exactly at every cu_seqlens boundary
    pos = f["position_ids"][0]
    assert (pos[f["cu_seqlens"][:-1]] == 0).all()
    # flatten and padded carry the same labelled tokens (padding carries none)
    assert int((f["labels"] != -100).sum()) == int((b["labels"] != -100).sum())
    # PackedCollator front door
    assert PackedCollator("flatten")([ds_raw[0]])["input_ids"].shape[0] == 1
    with pytest.raises(ValueError):
        PackedCollator("bogus")


def test_batch_schedule_is_pure_function_with_disjoint_ranks():
    n, mb, acc, world = 103, 4, 3, 2
    spe = steps_per_epoch(n, mb, acc, world)
    assert spe == 103 // 24
    seen = set()
    for step in range(spe):
        a = batch_indices(n, step, mb, acc, rank=0, world=world, seed=5)
        b = batch_indices(n, step, mb, acc, rank=1, world=world, seed=5)
        assert len(a) == acc and all(len(x) == mb for x in a)
        flat = [i for row in a + b for i in row]
        assert len(set(flat)) == len(flat), "ranks / micro-batches overlap inside a step"
        assert not (set(flat) & seen), "a pack repeated inside one epoch"
        seen |= set(flat)
    assert batch_indices(n, 0, mb, acc, 0, world, 5) == batch_indices(n, 0, mb, acc, 0, world, 5)
    assert batch_indices(n, spe, mb, acc, 0, world, 5) != batch_indices(n, 0, mb, acc, 0, world, 5)  # next epoch reshuffles


def test_bidirectional_preparation_counts_split_and_direction(tmp_path):
    rows = synthetic_rows(240, seed=11)
    meta, out = _build(tmp_path, rows, directions=("en2x", "x2en"))
    assert meta["directions"] == ["en2x", "x2en"]
    for split in ("train", "val"):
        pd = meta["splits"][split]["per_direction"]
        assert pd["en2x"]["samples"] == pd["x2en"]["samples"] > 0          # one reverse sample per pair
    ds = PackedDataset(out, "train")
    n = meta["splits"]["train"]["n_samples"]
    assert ds.n_samples == n and int(np.asarray(ds.direction).sum()) == n // 2
    # token_iso offsets the reverse direction by the number of languages, iso_id keeps the African language
    ids = {int(i) for i in np.asarray(ds.iso_id)}
    seen = set()
    for i in range(len(ds)):
        it = ds[i]
        seen |= {int(x) for x in it["token_iso"][: it["length"]]}
    assert any(x >= len(ALL_ISOS) for x in seen) and ids <= set(range(len(ALL_ISOS)))
    # the reverse sample of a pair is in the same split as the forward one: no english sentence on both sides
    def english_set(ds_):
        s = set()
        for i in range(len(ds_)):
            it = ds_[i]
            text = TOK.decode(it["input_ids"][: it["length"]])
            for part in text.split("<bos>")[1:]:
                head, ans = part.split("<start_of_turn>model\n", 1)
                src = head.split(":  ", 1)[1].split("<end_of_turn>")[0]
                tgt = ans.split(" ", 1)[1].split("<end_of_turn>")[0]
                s.add(tgt if head.startswith("<start_of_turn>user\nTranslate the text to English") else src)
        return s
    assert english_set(ds).isdisjoint(english_set(PackedDataset(out, "val")))


def test_x2en_fraction_is_deterministic_and_en2x_only_has_no_reverse(tmp_path):
    rows = synthetic_rows(200, seed=2)
    m_half, _ = _build(tmp_path, rows, directions=("en2x", "x2en"), x2en_fraction=0.5)
    p2 = tmp_path / "again"; p2.mkdir()
    m_half2, _ = _build(p2, rows, directions=("en2x", "x2en"), x2en_fraction=0.5)
    assert m_half["splits"]["train"]["per_direction"] == m_half2["splits"]["train"]["per_direction"]
    pd = m_half["splits"]["train"]["per_direction"]
    assert 0.25 < pd["x2en"]["samples"] / pd["en2x"]["samples"] < 0.75
    p3 = tmp_path / "fwd"; p3.mkdir()
    m_fwd, out = _build(p3, rows)
    assert m_fwd["splits"]["train"]["per_direction"]["x2en"]["samples"] == 0
    with pytest.raises(ValueError):
        PrepareConfig(input="x", out_dir="y", directions=("sideways",))
