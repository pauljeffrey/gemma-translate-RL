# SPDX-License-Identifier: AGPL-3.0-or-later
"""Windows -> packed uint32 dataset of the inter-African SFT, in the layout of `gemma_mt.data.prepare_sft` (the same files, read by `PackedDataset`):

    tokens.u32.bin  sample_offset.npy  sample_len.npy  prompt_len.npy  target_len.npy  iso_id.npy (TARGET language)  direction.npy  pack_start.npy
plus ONE new array per split:
    src_iso_id.npy  uint16  index into ALL_ISOS of the SOURCE language of the sample (len(ALL_ISOS) means English)
    prompt_form.npy uint8   0 = explicit `from X to Y` prompt, 1 = implicit `to Y` prompt (the 50 / 50 mixture, `prompt.prompt_form_of`)
`direction.npy` is 2 (`prompt.DIRECTION_X2X`) for every sample of this dataset (0 = en2x, 1 = x2en in the en->X dataset); `meta.json` carries
`format_x2x_version` and the explicit-source template. `PackedDataset` loads it unchanged, but the trainer's per-language validation vector has 2 x n_lang
slots (en2x, x2en) and must be sized 3 x n_lang before this dataset is validated (see reports/afri2afri_plan_2026-10-20.md, section "Trainer changes").

`encode_pair` builds the token ids of ONE directional sample without the Jinja template (the pure-python `format.reference_render` produces the same string
as the tokenizer's template -- `format.assert_template_matches_reference` is called by the builder -- and tokenizing the prompt and the target as two
segments is exactly what `format.build_example` does), so a whole window batch goes through the fast tokenizer in one call.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gemma_mt import format as fmt
from gemma_mt.afri2afri import prompt as P
from gemma_mt.data.packing import pack_buffered, packing_stats
from gemma_mt.languages import ALL_ISOS, iso_to_id


def sample_strings(text: str, src: Optional[str], tgt: str, translation: str) -> Tuple[str, str]:
    """(prompt string, target string) of one directional sample; `src=None` renders the implicit en->X-compatible prompt `Translate the text to {tgt}:`."""
    msgs = [{"role": "user", "content": P.build_user_content_x2x(text, src, tgt)}]
    prompt = fmt.reference_render(msgs, add_generation_prompt=True)
    target = fmt.build_model_content(tgt, translation) + fmt.END_OF_TURN
    return prompt, target


def encode_batch(tokenizer: Any, items: Sequence[Tuple[str, Optional[str], str, str]]) -> List[Tuple[List[int], int, int]]:
    """items = (text, src, tgt, translation) -> [(input_ids, prompt_len, target_len)]. The tail "\\n" after <end_of_turn> is appended (label -100)."""
    pairs = [sample_strings(*it) for it in items]
    p_ids = tokenizer([p for p, _ in pairs], add_special_tokens=False)["input_ids"]
    t_ids = tokenizer([t for _, t in pairs], add_special_tokens=False)["input_ids"]
    tail = tokenizer("\n", add_special_tokens=False)["input_ids"]
    eot = tokenizer.convert_tokens_to_ids(fmt.END_OF_TURN)
    out = []
    for p, t in zip(p_ids, t_ids):
        if not t or t[-1] != eot:
            raise ValueError("a target does not end with <end_of_turn>")
        if p[:1] != [tokenizer.bos_token_id] or p[1:2] == [tokenizer.bos_token_id]:
            raise ValueError("prompt must start with exactly one <bos>")
        out.append((list(p) + list(t) + list(tail), len(p), len(t)))
    return out


class RawStore:
    """Append-only on-disk store of tokenized samples before packing."""

    def __init__(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, "raw_tokens.u32")
        self._f = open(self.path, "wb")
        self.length: List[int] = []
        self.prompt: List[int] = []
        self.target: List[int] = []
        self.iso: List[int] = []
        self.src: List[int] = []
        self.form: List[int] = []
        self.direction: List[int] = []
        self.offset: List[int] = []
        self._pos = 0

    def add(self, ids: Sequence[int], prompt_len: int, target_len: int, tgt_iso_id: int, src_iso_id: int, form: int = 0, direction: int = P.DIRECTION_X2X) -> None:
        """`tgt_iso_id` = the AFRICAN language of the pair (the convention of `iso_id.npy`: for x2en samples it is the source); `src_iso_id` = the real source (len(ALL_ISOS) = English)."""
        arr = np.asarray(ids, np.uint32)
        self._f.write(arr.tobytes())
        self.offset.append(self._pos)
        self.length.append(len(arr))
        self.prompt.append(prompt_len)
        self.target.append(target_len)
        self.iso.append(tgt_iso_id)
        self.src.append(src_iso_id)
        self.form.append(form)
        self.direction.append(direction)
        self._pos += len(arr)

    def close(self) -> None:
        self._f.close()

    def __len__(self) -> int:
        return len(self.length)


def write_split(store: RawStore, out_dir: str, seq_len: int, buffer_size: int, seed: int) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    store.close()
    n = len(store)
    lengths = np.asarray(store.length, np.uint32)
    bins = pack_buffered(lengths.tolist(), seq_len, buffer_size, seed) if n else []
    stats = packing_stats(lengths.tolist(), bins, seq_len) if n else {"n_tokens": 0, "efficiency": 0.0, "padding_fraction": 0.0, "mean_samples_per_bin": 0.0}
    order = np.asarray([i for b in bins for i in b], np.int64)
    pack_start = np.zeros(len(bins) + 1, np.uint64)
    if bins:
        pack_start[1:] = np.cumsum([len(b) for b in bins], dtype=np.uint64)
    off_old = np.asarray(store.offset, np.uint64)
    plen, tlen = np.asarray(store.prompt, np.uint32), np.asarray(store.target, np.uint32)
    iso, src = np.asarray(store.iso, np.uint16), np.asarray(store.src, np.uint16)
    form = np.asarray(store.form, np.uint8)
    dirs = np.asarray(store.direction, np.uint8)
    new_len = lengths[order] if n else lengths
    new_off = np.zeros(len(order), np.uint64)
    if len(order):
        new_off[1:] = np.cumsum(new_len[:-1].astype(np.uint64))
    if n:
        raw = np.memmap(store.path, dtype=np.uint32, mode="r")
        with open(os.path.join(out_dir, "tokens.u32.bin"), "wb") as f:
            for i in order:
                o = int(off_old[i])
                f.write(np.asarray(raw[o: o + int(lengths[i])], np.uint32).tobytes())
        del raw
    else:
        open(os.path.join(out_dir, "tokens.u32.bin"), "wb").close()
    np.save(os.path.join(out_dir, "sample_offset.npy"), new_off)
    np.save(os.path.join(out_dir, "sample_len.npy"), new_len.astype(np.uint32))
    np.save(os.path.join(out_dir, "prompt_len.npy"), plen[order] if n else plen)
    np.save(os.path.join(out_dir, "target_len.npy"), tlen[order] if n else tlen)
    np.save(os.path.join(out_dir, "iso_id.npy"), iso[order] if n else iso)
    np.save(os.path.join(out_dir, "src_iso_id.npy"), src[order] if n else src)
    np.save(os.path.join(out_dir, "prompt_form.npy"), form[order] if n else form)             # 0 = explicit `from X to Y`, 1 = implicit `to Y`
    np.save(os.path.join(out_dir, "direction.npy"), dirs[order] if n else dirs)
    np.save(os.path.join(out_dir, "pack_start.npy"), pack_start)
    if os.path.exists(store.path):
        os.remove(store.path)
    # per directed pair statistics
    eng = len(ALL_ISOS)
    tgt_actual = np.where(dirs == 1, eng, iso).astype(np.int64)                       # x2en samples: the target is English
    key = src.astype(np.int64) * 1000 + tgt_actual
    per_dir: Dict[str, Dict[str, int]] = {}
    if n:
        uk, inv = np.unique(key, return_inverse=True)
        samples = np.bincount(inv)
        ttok = np.bincount(inv, weights=tlen.astype(np.float64))
        tok = np.bincount(inv, weights=lengths.astype(np.float64))
        for j, k in enumerate(uk):
            s_i, t_i = int(k // 1000), int(k % 1000)
            s_name = "eng" if s_i == len(ALL_ISOS) else ALL_ISOS[s_i]
            t_name = "eng" if t_i == len(ALL_ISOS) else ALL_ISOS[t_i]
            per_dir[f"{s_name}>{t_name}"] = {"samples": int(samples[j]), "tokens": int(tok[j]), "target_tokens": int(ttok[j])}
    return {"prompt_forms": {"explicit": int((form == 0).sum()), "implicit": int((form == 1).sum())}, "directions": {"en2x": int((dirs == 0).sum()), "x2en": int((dirs == 1).sum()), "x2x": int((dirs == 2).sum())}, "n_samples": n, "n_packs": len(bins), "n_tokens": int(stats["n_tokens"]), "target_tokens": int(tlen.sum()) if n else 0,
            "packing_efficiency": stats["efficiency"], "padding_fraction": stats["padding_fraction"], "mean_samples_per_pack": stats["mean_samples_per_bin"],
            "per_direction": per_dir}
