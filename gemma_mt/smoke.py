# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic data + tiny model helpers for `--smoke` runs and the offline tests (CPU, seconds)."""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Sequence

ENGLISH = [
    "Good morning, how are you today?", "The market opens at eight o'clock.", "Please close the door.",
    "My brother is a teacher.", "We will travel to the city tomorrow.", "The rain has stopped.",
    "Water is life.", "Thank you very much.", "Where is the nearest hospital?", "The children are playing outside.",
    "I would like some food.", "He reads a book every night.", "The river is very wide.", "Come and sit with us.",
    "Education is the key to the future.", "She sells fresh vegetables.", "It is a long way home.",
    "The farmer planted maize in the field.", "Our village has a new school.", "Let us start the meeting now.",
]


def fake_translation(english: str, iso: str) -> str:
    """A deterministic, learnable 'translation' that depends on the language, so the smoke SFT loss can go down."""
    k = sum(map(ord, iso)) % 3
    if k == 0:
        return " ".join(w[::-1] for w in english.split())
    if k == 1:
        return " ".join(w[::-1] for w in english.split()).upper()
    return english[::-1]


def synthetic_rows(n: int, isos: Sequence[str] = ("yor", "hau", "ibo"), seed: int = 0) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        en = ENGLISH[i % len(ENGLISH)]
        iso = isos[rng.randrange(len(isos))]
        rows.append({"english": en, "target": fake_translation(en, iso), "iso": iso, "id": f"s{i}"})
    return rows


def write_jsonl(path: str, rows) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path
