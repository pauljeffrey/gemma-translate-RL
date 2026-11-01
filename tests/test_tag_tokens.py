# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tag mode "single_token" (growth 1): format, stripping in both forms, RL decoding, fake tokenizer, pack rewrite."""
import json
import os
import sys

import numpy as np
import pytest

from gemma_mt import format as fmt
from gemma_mt import tags
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS, TAGS
from gemma_mt.languages import ALL_ISOS
from gemma_mt.rl.text import decode_completion

TOK = FakeGemmaTokenizer()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "sft_clean", "chain"))


def test_default_mode_is_text_and_nothing_changes():
    assert fmt.get_tag_mode() == "text"
    assert fmt.build_model_content("yor", "E kaaro") == "yor E kaaro"
    ex = fmt.build_example("Good morning", "yor", "E kaaro", TOK)
    assert TOK.decode([t for t, l in zip(ex.input_ids, ex.labels) if l != fmt.IGNORE_INDEX]) == "yor E kaaro<end_of_turn>"


def test_single_token_example_starts_with_one_tag_token_and_keeps_the_rest():
    text = fmt.build_example("Good morning", "yor", "E kaaro", TOK)
    with fmt.tag_mode("single_token"):
        ex = fmt.build_example("Good morning", "yor", "E kaaro", TOK)
    assert fmt.get_tag_mode() == "text"                                   # the context manager restores the mode
    tgt = [t for t, l in zip(ex.input_ids, ex.labels) if l != fmt.IGNORE_INDEX]
    assert tgt[0] == TAGS[tags.tag_token("yor")] and TOK.decode(tgt) == "<unused0> E kaaro<end_of_turn>"
    assert ex.input_ids[: ex.prompt_len] == text.input_ids[: text.prompt_len]
    # tokens after the tag are byte-identical to the text mode (3 text pieces `yor` became 1 token)
    assert ex.input_ids[ex.prompt_len + 1:] == text.input_ids[text.prompt_len + 3:]
    assert ex.target_len == text.target_len - 2


def test_pivot_tag_stays_text_and_aliases_are_normalised():
    with fmt.tag_mode("single_token"):
        assert fmt.build_model_content("eng", "Good morning") == "eng Good morning"
        assert fmt.build_model_content("SWA", "x") == f"{tags.tag_token('swh')} x"
        ex = fmt.build_example("E kaaro", "eng", "Good morning", TOK)
    assert TOK.decode([t for t, l in zip(ex.input_ids, ex.labels) if l != fmt.IGNORE_INDEX]).startswith("eng Good")


def test_tag_slots_are_the_table_positions_and_unique():
    assert tags.tag_token("yor") == "<unused0>" and tags.tag_token(ALL_ISOS[98]) == "<unused98>" and tags.tag_token(ALL_ISOS[99]) == "<unused6241>" and tags.tag_token(ALL_ISOS[-1]) == f"<unused{6241 - (len(ALL_ISOS) - 1 - 99)}>"
    assert len(set(tags.tag_tokens().values())) == len(ALL_ISOS)
    assert tags.iso_from_tag_token("<unused0>") == "yor" and tags.iso_from_tag_token("<yor>") == "yor"
    assert tags.iso_from_tag_token("<unused6241>") == ALL_ISOS[99] and tags.iso_from_tag_token("<unused500>") is None and tags.iso_from_tag_token("<abc>") is None and tags.iso_from_tag_token("<pad>") is None


@pytest.mark.parametrize("completion,status,text", [
    ("yor E kaaro", "ok", "E kaaro"),                  # text form
    ("<unused0> E kaaro", "ok", "E kaaro"),             # token form (original tokenizer)
    ("<yor> E kaaro", "ok", "E kaaro"),                 # token form (renamed tokenizer)
    ("<unused0>E kaaro", "ok", "E kaaro"),
    ("<unused1> Sannu", "wrong", "Sannu"),
    ("<unused0>", "ok", ""),
    ("E kaaro", "missing", "E kaaro"),
    ("<unused999> E kaaro", "missing", "<unused999> E kaaro"),
])
def test_strip_accepts_both_forms(completion, status, text):
    r = fmt.strip_iso_prefix(completion, "yor")
    assert (r.status, r.text) == (status, text)


def test_rl_decode_keeps_the_leading_tag_token_but_strips_other_unused_strings():
    ids = [TAGS["<unused0>"]] + [256 + b for b in " E kaaro <unused5> x".encode()] + [SPECIALS["<end_of_turn>"]]
    d = decode_completion(TOK, ids, [SPECIALS["<end_of_turn>"], SPECIALS["<eos>"]])
    assert d.text.startswith("<unused0> E kaaro") and "<unused5>" not in d.text and d.finished
    assert d.special_text_stripped                       # the interior <unused5> was typed-out junk
    assert fmt.strip_iso_prefix(d.text, "yor").status == "ok"
    d2 = decode_completion(TOK, [256 + b for b in b"yor E kaaro"] + [SPECIALS["<end_of_turn>"]], [SPECIALS["<end_of_turn>"]])
    assert d2.text == "yor E kaaro" and not d2.special_text_stripped


def test_retag_packs_roundtrip(tmp_path):
    import retag_packs as R
    exs = []
    for k, (en, iso, tr) in enumerate([("Good morning", "yor", "E kaaro"), ("How are you?", "hau", "Yaya kake?"), ("Thanks", "swh", "Asante"), ("Yes", "kmb", "Ehe")] * 3):
        exs.append((iso, fmt.build_example(en, iso, tr, TOK)))
    src = tmp_path / "src"; (src / "train").mkdir(parents=True)
    ids = [ALL_ISOS.index(i) for i, _ in exs]
    sl = np.array([e.n_tokens for _, e in exs], dtype=np.uint32); pl = np.array([e.prompt_len for _, e in exs], dtype=np.uint32); tl = np.array([e.target_len for _, e in exs], dtype=np.uint32)
    so = np.concatenate([[0], np.cumsum(sl.astype(np.int64))[:-1]]).astype(np.uint64)
    ps = np.array([0, 4, 8, 12], dtype=np.uint64)
    for name, v in dict(sample_offset=so, sample_len=sl, prompt_len=pl, target_len=tl, iso_id=np.array(ids, dtype=np.uint16), direction=np.zeros(12, dtype=np.uint8), pack_start=ps).items():
        np.save(src / "train" / (name + ".npy"), v)
    np.concatenate([np.array(e.input_ids, dtype=np.uint32) for _, e in exs]).tofile(src / "train" / "tokens.u32.bin")
    json.dump({"iso_table": list(ALL_ISOS[:85]), "seq_len": 4096, "splits": {"train": {"n_tokens": int(sl.sum()), "target_tokens": int(tl.sum()), "per_language": {"yor": {"target_tokens": 0}}}}}, open(src / "meta.json", "w"))
    out = tmp_path / "out"
    sys.argv = ["retag_packs.py", "--src", str(src), "--out", str(out), "--fake-tokenizer", "--splits", "train", "--verify", "12", "--chunk-tokens", "64"]
    R.main()
    m = json.load(open(out / "meta.json"))
    assert m["tag_mode"] == "single_token" and len(m["tag_map_sha256"]) == 64 and m["tag_map"]["yor"]["token_id"] == TAGS["<unused0>"]
    d = out / "train"
    sl2, tl2, so2 = np.load(d / "sample_len.npy"), np.load(d / "target_len.npy"), np.load(d / "sample_offset.npy")
    assert (sl.astype(np.int64) - sl2 == 2).all() and (tl.astype(np.int64) - tl2 == 2).all()          # every fake tag had 3 byte pieces
    tok2 = np.fromfile(d / "tokens.u32.bin", dtype=np.uint32)
    assert len(tok2) == int(sl2.sum()) and (np.diff(so2.astype(np.int64)) == sl2[:-1]).all()
    # the same samples rendered directly in tag mode equal the rewritten ones
    with fmt.tag_mode("single_token"):
        for i, (en, iso, tr) in enumerate([("Good morning", "yor", "E kaaro"), ("How are you?", "hau", "Yaya kake?"), ("Thanks", "swh", "Asante"), ("Yes", "kmb", "Ehe")] * 3):
            want = fmt.build_example(en, iso, tr, TOK).input_ids
            assert tok2[int(so2[i]): int(so2[i]) + int(sl2[i])].tolist() == want
    for n in ("iso_id", "direction", "pack_start", "prompt_len"):
        assert np.array_equal(np.load(src / "train" / (n + ".npy")), np.load(d / (n + ".npy")))


def test_retag_packs_refuses_a_sample_without_the_tag(tmp_path):
    import retag_packs as R
    ex = fmt.build_example("Good morning", "yor", "E kaaro", TOK)
    src = tmp_path / "src"; (src / "train").mkdir(parents=True)
    ids = list(ex.input_ids); ids[ex.prompt_len] = 300                    # corrupt the first tag piece
    for name, v in dict(sample_offset=np.array([0], dtype=np.uint64), sample_len=np.array([len(ids)], dtype=np.uint32), prompt_len=np.array([ex.prompt_len], dtype=np.uint32), target_len=np.array([ex.target_len], dtype=np.uint32),
                        iso_id=np.array([0], dtype=np.uint16), direction=np.zeros(1, dtype=np.uint8), pack_start=np.array([0, 1], dtype=np.uint64)).items():
        np.save(src / "train" / (name + ".npy"), v)
    np.array(ids, dtype=np.uint32).tofile(src / "train" / "tokens.u32.bin")
    json.dump({"iso_table": list(ALL_ISOS[:85])}, open(src / "meta.json", "w"))
    sys.argv = ["retag_packs.py", "--src", str(src), "--out", str(tmp_path / "o"), "--fake-tokenizer", "--splits", "train", "--verify", "0"]
    with pytest.raises(SystemExit):
        R.main()
