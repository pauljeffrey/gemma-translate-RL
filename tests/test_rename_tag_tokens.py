# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import rename_tag_tokens as R  # noqa: E402
from gemma_mt import tags  # noqa: E402
from gemma_mt.languages import ALL_ISOS  # noqa: E402


def _mini(tmp_path, extra_vocab=None):
    d = tmp_path / "src"; d.mkdir()
    vocab = {"<pad>": 0, "a": 1}
    added = [{"id": 0, "content": "<pad>", "special": True}]
    dec = {"0": {"content": "<pad>", "special": True}}
    at = {}
    for i, iso in enumerate(ALL_ISOS):
        name = tags.tag_token(iso); vocab[name] = 100 + i
        added.append({"id": 100 + i, "content": name, "special": False}); dec[str(100 + i)] = {"content": name, "special": False}; at[name] = 100 + i
    vocab.update(extra_vocab or {})
    json.dump({"added_tokens": added, "model": {"type": "BPE", "vocab": vocab, "merges": []}}, open(d / "tokenizer.json", "w"))
    json.dump({"added_tokens_decoder": dec, "bos_token": "<bos>"}, open(d / "tokenizer_config.json", "w"))
    json.dump(at, open(d / "added_tokens.json", "w"))
    (d / "tokenizer.model").write_bytes(b"spm")
    return d


def test_rename_keeps_ids_and_names_every_slot(tmp_path):
    d = _mini(tmp_path)
    info = R.rename_tokenizer_files(str(d), str(tmp_path / "out"))
    out = tmp_path / "out"
    tj = json.load(open(out / "tokenizer.json")); v = tj["model"]["vocab"]
    assert info["renamed"] == len(ALL_ISOS) and v["<yor>"] == 100 and "<unused0>" not in v and v["<pad>"] == 0 and len(v) == len(json.load(open(d / "tokenizer.json"))["model"]["vocab"])
    assert [a["content"] for a in tj["added_tokens"] if a["id"] == 100] == ["<yor>"] and all(not a["special"] for a in tj["added_tokens"] if a["id"] >= 100)
    tc = json.load(open(out / "tokenizer_config.json")); assert tc["added_tokens_decoder"]["100"]["content"] == "<yor>" and tc["bos_token"] == "<bos>"
    assert json.load(open(out / "added_tokens.json"))["<yor>"] == 100 and (out / "tokenizer.model").read_bytes() == b"spm"


def test_rename_refuses_a_name_clash_and_a_foreign_tokenizer(tmp_path):
    d = _mini(tmp_path, {"<hau>": 7})
    with pytest.raises(SystemExit):
        R.rename_tokenizer_files(str(d), str(tmp_path / "o1"))
    (tmp_path / "foreign").mkdir()
    json.dump({"added_tokens": [], "model": {"vocab": {"a": 0}}}, open(tmp_path / "foreign" / "tokenizer.json", "w"))
    json.dump({"added_tokens_decoder": {}}, open(tmp_path / "foreign" / "tokenizer_config.json", "w"))
    with pytest.raises(SystemExit):
        R.rename_tokenizer_files(str(tmp_path / "foreign"), str(tmp_path / "o2"))
