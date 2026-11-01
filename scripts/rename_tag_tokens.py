#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Publishable tokenizer for the dedicated language-tag tokens (growth 1): rename the 125 `<unusedK>` slots of the Gemma-3 tokenizer to `<yor>`, `<hau>` ... (same ids,
vocabulary size unchanged, nothing else touched). Nothing is pushed to the Hub by this script.

    python scripts/rename_tag_tokens.py --src <dir with tokenizer.json, tokenizer_config.json, added_tokens.json>  --out <new dir>
    python scripts/rename_tag_tokens.py --model google/gemma-3-270m-it --out <new dir>        (downloads the tokenizer files; needs HF_TOKEN, the repo is gated)

Edited: tokenizer.json (the `added_tokens` entries and the key of `model.vocab`), tokenizer_config.json (`added_tokens_decoder`), added_tokens.json. Copied unchanged: every other file.
`tokenizer.model` (the SentencePiece binary used only by the SLOW tokenizer) is copied as it is and still names the slots `<unusedK>`: use the fast tokenizer (AutoTokenizer does).
The tag tokens stay NON-special (`special: false`, like the `<unused*>` they replace), so decode(skip_special_tokens=True) keeps them and format.strip_iso_prefix reads them (`<yor> E kaaro`).
The mapping is gemma_mt.tags.tag_token: language i of gemma_mt.languages.ALL_ISOS <-> `<unused{i}>` -> `<{iso}>`. A new name that already exists in the vocabulary aborts the run.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemma_mt import tags  # noqa: E402
from gemma_mt.languages import ALL_ISOS  # noqa: E402


def rename_map():
    """{ '<unusedK>': '<iso>' } for every row of the language table."""
    return {tags.tag_token(iso): f"<{iso}>" for iso in ALL_ISOS}


def rename_tokenizer_files(src, out):
    m = rename_map()
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):
        if os.path.isfile(os.path.join(src, f)) and f not in ("tokenizer.json", "tokenizer_config.json", "added_tokens.json"):
            shutil.copyfile(os.path.join(src, f), os.path.join(out, f))
    tj = json.load(open(os.path.join(src, "tokenizer.json"), encoding="utf-8"))
    vocab = tj["model"]["vocab"]
    ids_old = {}
    for old in m:
        if old not in vocab:
            raise SystemExit(f"{old} is not in the vocabulary of this tokenizer (is it the original Gemma-3 tokenizer?)")
        ids_old[old] = vocab[old]
    clash = [new for new in m.values() if new in vocab]
    if clash:
        raise SystemExit(f"new tag names already exist in the vocabulary: {clash}")
    tj["model"]["vocab"] = {m.get(k, k): v for k, v in vocab.items()}
    n_added = 0
    for a in tj.get("added_tokens", []):
        if a["content"] in m:
            a["content"] = m[a["content"]]; n_added += 1
    json.dump(tj, open(os.path.join(out, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    tc = json.load(open(os.path.join(src, "tokenizer_config.json"), encoding="utf-8"))
    n_dec = 0
    for k, a in tc.get("added_tokens_decoder", {}).items():
        if a["content"] in m:
            a["content"] = m[a["content"]]; n_dec += 1
    json.dump(tc, open(os.path.join(out, "tokenizer_config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    n_at = 0
    p = os.path.join(src, "added_tokens.json")
    if os.path.exists(p):
        at = json.load(open(p, encoding="utf-8"))
        json.dump({m.get(k, k): v for k, v in at.items()}, open(os.path.join(out, "added_tokens.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        n_at = sum(1 for k in at if k in m)
    assert n_added == len(m) and n_dec == len(m), (n_added, n_dec, len(m))
    return {"renamed": len(m), "ids": {new: ids_old[old] for old, new in m.items()}, "added_tokens.json_entries": n_at}


def check_with_transformers(src, out, ids):
    """Load both tokenizers (needs `transformers`): every tag keeps its id, the old name is gone, `<iso> text` tokenizes to [id, ...] and the rest equals the original's."""
    from transformers import AutoTokenizer
    a, b = AutoTokenizer.from_pretrained(src), AutoTokenizer.from_pretrained(out)
    assert len(a) == len(b)
    for iso in ALL_ISOS:
        old, new = tags.tag_token(iso), f"<{iso}>"
        assert a.convert_tokens_to_ids(old) == b.convert_tokens_to_ids(new) == ids[new], iso
        assert b.convert_tokens_to_ids(old) in (None, b.unk_token_id), iso
    x = a("<unused0> E kaaro<end_of_turn>", add_special_tokens=False)["input_ids"]; y = b("<yor> E kaaro<end_of_turn>", add_special_tokens=False)["input_ids"]
    assert x == y and b.decode(y, skip_special_tokens=True) == "<yor> E kaaro", (x, y)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src"); ap.add_argument("--model"); ap.add_argument("--out", required=True); ap.add_argument("--no-check", action="store_true")
    a = ap.parse_args()
    src = a.src
    if not src:
        from huggingface_hub import snapshot_download
        src = snapshot_download(a.model or "google/gemma-3-270m-it", token=os.environ.get("HF_TOKEN"), allow_patterns=["tokenizer*", "special_tokens_map.json", "added_tokens.json", "chat_template*", "config.json"])
    info = rename_tokenizer_files(src, a.out)
    print(f"renamed {info['renamed']} tag tokens into {a.out}")
    if not a.no_check:
        try:
            print("transformers check:", check_with_transformers(src, a.out, info["ids"]))
        except ImportError:
            print("transformers not installed: check skipped")


if __name__ == "__main__":
    main()
