# SPDX-License-Identifier: AGPL-3.0-or-later
"""README.md (dataset card) generator for the SFT pairs and the RL prompts. Licence: cc-by-4.0 by default (attribution only: reuse in closed or commercial models is allowed).

Not a legal statement about third-party content: the card tells users to check the licences of the source
datasets listed in `source_datasets`, and says that nothing here forces anyone to release data they produce
with a model.
"""

from __future__ import annotations

from typing import Optional, Sequence

from gemma_mt import format as fmt
from gemma_mt.languages import ALL_ISOS, LANGUAGES

SCHEMAS = {
    "sft": [("english", "string", "English source sentence"),
            ("target", "string", "translation in the African language"),
            ("iso", "string", "language tag of `target` (see table)"),
            ("id", "string, optional", "stable sample id"),
            ("source_dataset", "string, optional", "dataset the pair came from"),
            ("licence", "string, optional", "licence class of that source (cc-by, cc0, apache, mit, ...)"),
            ("qe", "float, optional", "AfriCOMET-QE quality estimate used by the quality gate")],
    "rl": [("english", "string", "English source text"),
           ("iso", "string", "target language tag"),
           ("language", "string", "display name of the language, exactly as used in the instruction"),
           ("src_afr_tokens", "int", "length of `english` in Davlan/afro-xlmr-large-76L tokens (<= 256)"),
           ("source_dataset", "string", "dataset the sentence came from")],
}


def build_dataset_card(repo_id: str, kind: str = "sft", *, license_id: str = "cc-by-4.0",
                       languages: Optional[Sequence[str]] = None, description: str = "",
                       source_datasets: Sequence[str] = (), n_rows: Optional[int] = None) -> str:
    if kind not in SCHEMAS:
        raise ValueError("kind must be 'sft' or 'rl'")
    langs = list(languages) if languages else list(ALL_ISOS)
    cols = "\n".join(f"| `{c}` | {t} | {d} |" for c, t, d in SCHEMAS[kind])
    names = "\n".join(f"| `{i}` | {LANGUAGES[i].name} |" for i in langs)
    src = "\n".join(f"- {s}" for s in source_datasets) or "- (list the source datasets here)"
    what = ("English to African-language sentence pairs for supervised fine-tuning of a translation model."
            if kind == "sft" else
            "English source sentences with a target language, used as prompts for reinforcement learning with a "
            "reference-free quality-estimation reward. No reference translations are included.")
    return f"""---
license: {license_id}
language:
- en
{chr(10).join('- ' + i for i in langs)}
task_categories:
- translation
tags:
- african-languages
- machine-translation
size_categories:
- {'1M<n<10M' if (n_rows or 0) > 1_000_000 else '10K<n<1M'}
---
# {repo_id.split('/')[-1]}

{what}
{description}

## Schema

| column | type | meaning |
|---|---|---|
{cols}

## How a row becomes a training prompt

User turn: `{fmt.INSTRUCTION_TEMPLATE}` (two spaces after the colon, literal), where `african_language` is the
`language` name below. Target: `<iso> <translation><end_of_turn>`.

## Languages

| tag | name |
|---|---|
{names}

## Sources and licensing

{src}

This dataset is released under {license_id}: it may be reused, including commercially and for training closed models,
with attribution. Every row carries the licence of the source it came from (`licence`) and the source dataset
(`source_dataset`); those sources keep their own terms, check them before reuse. This card does not oblige anyone to
open-source models trained on the data.
"""
