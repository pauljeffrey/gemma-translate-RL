# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dedicated language-tag tokens (growth 1, owner 2026-11-01): the tag written in front of the translation becomes ONE token instead of 2-3 text pieces.

The tags live in free `<unusedK>` slots of the Gemma-3 vocabulary (6,242 exist). Language i of `languages.ALL_ISOS` takes the slot `<unused{i}>` (ids 0..124 -> `<unused0>` .. `<unused124>`),
so a tag never moves when languages are appended at the END of the table. The vocabulary SIZE does not change (262,144 embedding rows), only the meaning of 125 rows.
Embedding init for a tag row = mean of the embeddings of the sub-tokens the text tag used to have (see the growth builder).
"""
from __future__ import annotations

from typing import Dict

from gemma_mt.languages import ALL_ISOS


def tag_slot(iso: str) -> int:
    return ALL_ISOS.index(iso)


_N_LOW = 99          # <unused0> .. <unused98> are the contiguous ids 6..104 of the Gemma-3 vocabulary
_LAST_UNUSED = 6241  # <unused99> .. <unused6241> are the tail pool (ids 256001 ..) that the Ge'ez vocabulary swap may reuse: tags beyond the first 99 take the FAR end, downward


def unused_index(slot: int) -> int:
    """Index K of the `<unusedK>` token for table row `slot`: rows 0..98 -> K = row; rows >= 99 -> 6241, 6240, ... (away from the front of the tail pool)."""
    return slot if slot < _N_LOW else _LAST_UNUSED - (slot - _N_LOW)


def slot_from_unused_index(k: int) -> int:
    return k if k < _N_LOW else _N_LOW + (_LAST_UNUSED - k)


def tag_token(iso: str) -> str:
    """The token string of the tag of `iso` in the tokenizer (an `<unusedK>` slot until the tokenizer is renamed)."""
    return f"<unused{unused_index(tag_slot(iso))}>"


def tag_tokens() -> Dict[str, str]:
    return {iso: tag_token(iso) for iso in ALL_ISOS}


# --------------------------------------------------------------------------------------------------- tag strings (data / eval side, added with the tag mode of format.py)
import re

_TAG_TOKEN_RE = re.compile(r"^<(?:unused(\d+)|([a-z]{3}))>")
LEADING_TAG_RE = re.compile(r"^\s*(<(?:unused\d+|[a-z]{3})>)")


def tag_string(iso: str, tokenizer=None) -> str:
    """The token STRING that stands for the tag of `iso` in this tokenizer: `<unusedK>` in the original Gemma tokenizer (and in every training pack), `<iso>` in the renamed
    Hub tokenizer (scripts/rename_tag_tokens.py; same ids). Without a tokenizer: the `<unusedK>` form."""
    cands = (tag_token(iso), f"<{iso}>")
    if tokenizer is None:
        return cands[0]
    unk = getattr(tokenizer, "unk_token_id", None)
    for c in cands:
        i = tokenizer.convert_tokens_to_ids(c)
        if i is not None and i != unk:
            return c
    raise ValueError(f"neither {cands[0]} nor {cands[1]} is a token of this tokenizer (is it the Gemma-3 tokenizer?)")


def iso_from_tag_token(token: str):
    """`<unusedK>` -> ALL_ISOS[K], `<yor>` -> "yor" (only for tags of the table); anything else -> None."""
    m = _TAG_TOKEN_RE.match(token or "")
    if not m:
        return None
    if m.group(1) is not None:
        k = slot_from_unused_index(int(m.group(1)))
        return ALL_ISOS[k] if 0 <= k < len(ALL_ISOS) else None
    return m.group(2) if m.group(2) in ALL_ISOS else None
