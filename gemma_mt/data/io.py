# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reading raw translation rows from parquet / jsonl (files or directories of files).

Schema (one row per sentence pair):
    english : str   source sentence (English)
    target  : str   translation in the African language
    iso     : str   language tag of `target` (canonical tags in gemma_mt.languages; common aliases are accepted)
    id      : str   OPTIONAL stable id (used only for the validation split when split_key="id")
Extra columns are ignored.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Dict, Iterable, Iterator, List, Sequence

SFT_COLUMNS = ("english", "target", "iso")


def list_input_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = []
        for root, _, names in os.walk(path):
            for n in sorted(names):
                if n.endswith((".parquet", ".jsonl", ".jsonl.gz", ".json")):
                    files.append(os.path.join(root, n))
        if not files:
            raise FileNotFoundError(f"no .parquet/.jsonl files under {path}")
        return sorted(files)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return [path]


def _iter_jsonl(path: str) -> Iterator[Dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: invalid JSON ({e})") from None


def _iter_parquet(path: str, columns: Sequence[str]) -> Iterator[Dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError("reading parquet needs `pyarrow` (pip install pyarrow)") from e
    pf = pq.ParquetFile(path)
    have = set(pf.schema_arrow.names)
    cols = [c for c in columns if c in have]
    for batch in pf.iter_batches(batch_size=50_000, columns=cols):
        d = batch.to_pydict()
        n = len(next(iter(d.values()))) if d else 0
        for i in range(n):
            yield {k: v[i] for k, v in d.items()}


def iter_rows(path: str, columns: Sequence[str] = ("english", "target", "iso", "id", "reference")) -> Iterator[Dict]:
    """Rows of every file under `path`, in file order. Parquet reads only `columns`."""
    for f in list_input_files(path):
        if f.endswith(".parquet"):
            yield from _iter_parquet(f, columns)
        elif f.endswith(".json"):
            with open(f, encoding="utf-8") as fh:
                text = fh.read()
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # JSON LINES in a .json file (masakhane-io/lafand-mt data/json_files/en-*/{dev,test}.json): one object per line
                data = [json.loads(line) for line in text.splitlines() if line.strip()]
            # only ROWS: a directory such as rl_prompts_v2 also holds `stats*.json` (one dict, not rows); iterating a dict yields its keys, which crashed load_rl_prompts (D-38 audit)
            yield from (x for x in data if isinstance(x, dict)) if isinstance(data, list) else ()
        else:
            yield from _iter_jsonl(f)
