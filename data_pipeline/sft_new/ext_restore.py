# SPDX-License-Identifier: AGPL-3.0-or-later
"""Text-restoration tasks (diacritize / clean / both) built by `sft_new.text_restore`, read from LOCAL parquet folders.

text_restore writes `artificial-restore_<task>` folders with the structured/artificial schema (instruction, input, response, language, task,
messages, source_dataset). The earlier artificial sets reach the combined build through the Hub copy of sft-new; these rows are not pushed, so this
converter reads the folders directly and emits the same Rec objects the other converters do (assembly, instruction-language tagging, dedupe and
licence flags then run exactly as for every other source).

    RESTORE_DIRS   colon-separated list of text_restore --out folders (default: every /workspace/sft_build/out_restore_* with a data/ folder)
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from sft_new.compose import A, Rec, Skip
from sft_new.converters import register

PLAN = {"diacritization": ["<diacritize>"], "denoising": ["<clean>"], "diacritization_denoising": ["<clean>", "<diacritize>"]}


def _dirs(env_name: str = "RESTORE_DIRS", pattern: str = "/workspace/sft_build/out_restore_*") -> list[Path]:
    env = os.environ.get(env_name)
    if env:
        return [Path(p) for p in env.split(":") if p]
    return [Path(p) for p in sorted(glob.glob(pattern)) if (Path(p) / "data").is_dir()]


@register("restore_local", "text_restore output folders (diacritize / clean / both for the diacritic languages), read locally -> restoration tasks")
def restore_local(rng, limit=None):
    yield from _read(_dirs(), limit)


@register("restore_extra", "text_restore add-on folders (extra_restore_*: Hausa diacritize/both from snippets that carry hooked letters), read locally")
def restore_extra(rng, limit=None):
    yield from _read(_dirs("RESTORE_DIRS_EXTRA", "/workspace/sft_build/extra_restore_*"), limit)


def _read(dirs, limit):
    import pyarrow.parquet as pq
    n = 0
    for root in dirs:
        for shard in sorted(glob.glob(str(root / "data" / "artificial-restore_*" / "part-*.parquet"))):
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=5000):
                for r in batch.to_pylist():
                    lang, task = r.get("language"), r.get("task")
                    if task not in PLAN:
                        yield Skip(f"unknown_task:{task}")
                        continue
                    if lang not in A.VALID_LANGS:
                        yield Skip(f"language_has_no_tag:{lang}")
                        continue
                    ins, text, resp = str(r.get("instruction") or ""), str(r.get("input") or ""), str(r.get("response") or "")
                    if not ins or not text or not resp:
                        yield Skip("empty_field")
                        continue
                    try:
                        user = next(m["content"] for m in json.loads(r["messages"]) if m["role"] == "user")
                    except Exception:
                        yield Skip("unparseable_messages")
                        continue
                    position = "prepend" if user.startswith(ins) else "append"
                    yield Rec(task, (os.environ['HF_ORG_LEGACY'] + "/pretrain-new (artificially corrupted, text_restore)"), lang, lang, resp, input=text, instruction=ins,
                              position=position, plan=list(PLAN[task]), meta={"origin": str(root.name)})
                    n += 1
                    if limit and n >= limit:
                        return
