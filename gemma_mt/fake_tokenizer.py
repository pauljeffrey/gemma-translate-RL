# SPDX-License-Identifier: AGPL-3.0-or-later
"""A byte-level stand-in for the Gemma 3 tokenizer, used ONLY by offline tests and `--smoke` runs.

It mimics the parts of the interface this module relies on, with the real Gemma 3 special-token ids:

    <pad>=0  <eos>=1  <bos>=2  <unk>=3  <start_of_turn>=105  <end_of_turn>=106
    byte b  -> id 256 + b          (vocab_size = 512)

Special-token strings inside the text are split out first (as the HF tokenizers do), the rest is UTF-8 bytes, so
`decode(encode(x)) == x` for any string. `apply_chat_template` delegates to `format.reference_render`.
Nothing here is a model of Gemma's real BPE; real-tokenizer behaviour is covered by the env-gated integration test.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from gemma_mt.format import reference_render

SPECIALS: Dict[str, int] = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<start_of_turn>": 105, "<end_of_turn>": 106}
BYTE_BASE = 256
VOCAB_SIZE = 512
# dedicated language-tag tokens (tag mode "single_token"): `<unusedK>` = id 107 + K for the 125 rows of the table (ids 107..231, free in this byte vocabulary). Not special:
# decode(skip_special_tokens=True) keeps them, like the real tokenizer (they are plain added tokens there).
TAG_BASE = 107
TAGS: Dict[str, int] = {f"<unused{k}>": TAG_BASE + k for k in range(125)}
_ID_TO_TAG = {v: k for k, v in TAGS.items()}
_SPLIT = re.compile("(" + "|".join(re.escape(s) for s in list(SPECIALS) + list(TAGS)) + ")")


class FakeGemmaTokenizer:
    def __init__(self, padding_side: str = "left"):
        self.name_or_path = "fake/gemma-byte-tokenizer"
        self.bos_token, self.eos_token, self.pad_token, self.unk_token = "<bos>", "<eos>", "<pad>", "<unk>"
        self.bos_token_id, self.eos_token_id = SPECIALS["<bos>"], SPECIALS["<eos>"]
        self.pad_token_id, self.unk_token_id = SPECIALS["<pad>"], SPECIALS["<unk>"]
        self.padding_side = padding_side
        self.chat_template = "fake-gemma3-reference"
        self.vocab_size = VOCAB_SIZE
        self._id_to_special = {v: k for k, v in SPECIALS.items()}

    # ------------------------------------------------------------------ vocab
    def __len__(self) -> int:
        return VOCAB_SIZE

    def get_vocab(self) -> Dict[str, int]:
        v = dict(SPECIALS); v.update(TAGS)
        for b in range(256):
            v[f"<0x{b:02X}>"] = BYTE_BASE + b
        return v

    def convert_tokens_to_ids(self, tokens: Union[str, Sequence[str]]):
        if isinstance(tokens, str):
            return SPECIALS.get(tokens, TAGS.get(tokens, self.unk_token_id))
        return [self.convert_tokens_to_ids(t) for t in tokens]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            if ids in _ID_TO_TAG:
                return _ID_TO_TAG[ids]
            return self._id_to_special.get(ids, f"<0x{ids - BYTE_BASE:02X}>" if ids >= BYTE_BASE else f"<unused{ids}>")
        return [self.convert_ids_to_tokens(i) for i in ids]

    # ------------------------------------------------------------------ encode / decode
    def _encode_one(self, text: str, add_special_tokens: bool) -> List[int]:
        ids: List[int] = [self.bos_token_id] if add_special_tokens else []
        for piece in _SPLIT.split(text):
            if not piece:
                continue
            if piece in SPECIALS:
                ids.append(SPECIALS[piece])
            elif piece in TAGS:
                ids.append(TAGS[piece])
            else:
                ids.extend(BYTE_BASE + b for b in piece.encode("utf-8"))
        return ids

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self._encode_one(text, add_special_tokens)

    def __call__(self, text, add_special_tokens: bool = True, **kw):
        if isinstance(text, (list, tuple)):
            ids = [self._encode_one(t, add_special_tokens) for t in text]
            return {"input_ids": ids, "attention_mask": [[1] * len(i) for i in ids]}
        ids = self._encode_one(text, add_special_tokens)
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False, **kw) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        out = bytearray()
        parts: List[Any] = []
        for i in ids:
            i = int(i)
            if i >= BYTE_BASE:
                out.append(i - BYTE_BASE)
                continue
            if out:
                parts.append(bytes(out).decode("utf-8", errors="replace"))
                out = bytearray()
            if i in _ID_TO_TAG:
                parts.append(_ID_TO_TAG[i])
            elif not skip_special_tokens and i in self._id_to_special:
                parts.append(self._id_to_special[i])
        if out:
            parts.append(bytes(out).decode("utf-8", errors="replace"))
        return "".join(parts)

    def batch_decode(self, seqs, skip_special_tokens: bool = False, **kw) -> List[str]:
        return [self.decode(s, skip_special_tokens=skip_special_tokens) for s in seqs]

    # ------------------------------------------------------------------ chat template
    def apply_chat_template(self, messages, tokenize: bool = False, add_generation_prompt: bool = False, **kw):
        text = reference_render(messages, add_generation_prompt, bos=self.bos_token)
        if tokenize:
            return self._encode_one(text, add_special_tokens=False)
        return text

    # ------------------------------------------------------------------ persistence (mirrors save_pretrained)
    def save_pretrained(self, path) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "fake_tokenizer.json"), "w") as f:
            json.dump({"specials": SPECIALS, "vocab_size": VOCAB_SIZE}, f)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(SPECIALS, sort_keys=True).encode()).hexdigest()[:16]
