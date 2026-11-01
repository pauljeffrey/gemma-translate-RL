# SPDX-License-Identifier: AGPL-3.0-or-later
"""Growth 1 helpers (owner 2026-11-01): the freeze patterns of the staged unfreezing and the initialisation of the dedicated language-tag rows.

* `new_layer_indices(upcycle.json)`      indices of the layers scripts/upcycle_layers.py inserted;
* `freeze_regex_new_layers_only(...)`    `freeze_regex` for stage A: every parameter EXCEPT the new layers and the embedding matrix (the embedding is then restricted to the tag rows
                                         with `train_embedding_tags`);
* `tag_token_ids(tokenizer)`             iso -> id of the `<unusedK>` slot of the tag (gemma_mt.tags);
* `init_tag_rows(src_dir, out_dir)`      writes a copy of an HF directory whose tag rows (tied embedding = input and output) are the MEAN of the embeddings of the sub-tokens
                                         the old text tag had (`tokenizer(iso, add_special_tokens=False)`).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from typing import Dict, List, Optional

LAYER_PREFIX = "model.layers."


def new_layer_indices(info: dict) -> List[int]:
    """New (inserted) layer indices of an upcycle.json."""
    return [int(r["new_index"]) for r in info["layers"] if r["new"]]


def freeze_regex_new_layers_only(new_idx: List[int], n_layers: Optional[int] = None) -> str:
    """Pattern (re.search on the parameter name) of everything that stage A freezes: all parameters except `model.layers.<i>.` for i in new_idx and `model.embed_tokens`.
    (`lm_head` is tied to the embedding and has no parameter of its own.)"""
    alt = "|".join(str(i) for i in sorted(new_idx))
    return rf"^(?!model\.layers\.(?:{alt})\.)(?!model\.embed_tokens\.)"


def tag_token_ids(tokenizer) -> Dict[str, int]:
    """iso -> token id of the dedicated tag slot; refuses a tokenizer that lacks a slot or maps two languages to one id."""
    from gemma_mt.languages import ALL_ISOS
    from gemma_mt.tags import tag_string

    unk = getattr(tokenizer, "unk_token_id", None)
    out = {}
    for iso in ALL_ISOS:
        tid = tokenizer.convert_tokens_to_ids(tag_string(iso, tokenizer))
        if tid is None or tid == unk:
            raise ValueError(f"no tag token for {iso}")
        out[iso] = int(tid)
    if len(set(out.values())) != len(out):
        raise ValueError("two languages share a tag token")
    return out


def init_tag_rows(src_dir: str, out_dir: str, tokenizer=None, hf_token: Optional[str] = None) -> dict:
    """Copy `src_dir` (config, tokenizer files, model.safetensors) to `out_dir` with the tag rows of the embedding set to the mean of the old text tag's sub-token rows.
    Only the tag rows change; every other tensor is copied bit for bit. Returns (and writes to out_dir/tags_init.json) the id table."""
    import torch
    from safetensors.torch import load_file, save_file

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(src_dir)
    ids = tag_token_ids(tokenizer)
    files = sorted(glob.glob(os.path.join(src_dir, "*.safetensors")))
    if len(files) != 1:
        raise SystemExit("expected exactly one model.safetensors in " + src_dir)
    sd = load_file(files[0])
    if any(k.endswith("lm_head.weight") for k in sd):
        raise SystemExit("the checkpoint has its own lm_head: this tool initialises the TIED embedding only")
    emb = sd["model.embed_tokens.weight"]
    table = {}
    new_rows = emb.clone()
    for iso, tid in ids.items():
        sub = tokenizer(iso, add_special_tokens=False)["input_ids"]
        if not sub or tid in sub:
            raise ValueError(f"{iso}: sub-token list {sub} is empty or contains the tag id")
        new_rows[tid] = emb[sub].float().mean(0).to(emb.dtype)
        table[iso] = {"token_id": tid, "old_subtoken_ids": [int(x) for x in sub]}
    sd["model.embed_tokens.weight"] = new_rows.contiguous()
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(src_dir):
        p = os.path.join(src_dir, fn)
        if os.path.isfile(p) and not fn.endswith(".safetensors"):
            shutil.copyfile(p, os.path.join(out_dir, fn))
    save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    json.dump(table, open(os.path.join(out_dir, "tags_init.json"), "w"), indent=1)
    return table
