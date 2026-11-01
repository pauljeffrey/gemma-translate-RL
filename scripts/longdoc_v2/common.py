# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared helpers of the long-document reward v2 study (agent R8, 2026-10-22). Runs on the rental with venv_gemma;
PYTHONPATH=<gemma_mt tree>:/workspace/sft_build/pylibs. Reads the A2 (R6) artefacts under /workspace/jobs/a2."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

A2 = "/workspace/jobs/a2"
CORE = A2 + "/data/hf/core"
HYP = A2 + "/hyp/rl2c"
SC = A2 + "/sc/rl2c"
NLLB = A2 + "/data/pseudo/nllb/nllb-200-3.3b"
R8 = "/workspace/jobs/r8"
CAL_DEFAULT = None        # set by callers: <tree>/calibration/shaping_flores_dev_v8.json
L6 = ["hau", "yor", "swh", "amh", "zul", "wol"]          # QE languages of R6's B8 that have whole-document systems
L8 = L6 + ["ibo", "xho"]                                  # + human SmolDoc / NLLB only (no whole-document hypotheses)
BINS = [64, 256, 512, 1024, 2048, 4096]


def rd(p: str) -> List[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def wr(p: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(p + ".tmp", p)


def h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def core_docs() -> Dict[str, dict]:
    return {d["doc_id"]: d for d in rd(CORE + "/docs.jsonl")}


def select_doc_ids(docs: Dict[str, dict], per_bin: int = 12, bins: Sequence[int] = BINS) -> Dict[int, List[str]]:
    out = {}
    for b in bins:
        ids = sorted((d for d in docs.values() if d["bin"] == b), key=lambda d: h(d["doc_id"]))
        out[b] = [d["doc_id"] for d in ids[:per_bin]]
    return out


def hyp_rows(system: str) -> Dict[Tuple[str, str], dict]:
    """(doc_id, iso) -> hypothesis row of the core set."""
    out = {}
    for r in rd(f"{HYP}/{system}.jsonl"):
        parts = r["iid"].split("|")
        if parts[0] != "core":
            continue
        out[("|".join(parts[1:-1]), parts[-1])] = r
    return out


def sc_rows(system: str) -> Dict[Tuple[str, str], dict]:
    import pickle

    out = {}
    with open(f"{SC}/{system}.pkl", "rb") as f:
        for r in pickle.load(f):
            parts = r["iid"].split("|")
            if parts[0] != "core":
                continue
            out[("|".join(parts[1:-1]), parts[-1])] = r
    return out


def nllb_refs(iso: str) -> Dict[str, dict]:
    return {r["doc_id"]: r for r in rd(f"{NLLB}/{iso}.refs.jsonl")}


def smol_equal_docs(iso: str) -> List[Tuple[str, List[str], List[str]]]:
    """SmolDoc documents whose human reference has exactly as many sentences as the English side (split with the reward's splitter): (doc_id, src_sents, ref_sents)."""
    from gemma_mt.rl.segments import split_sentences

    src = {d["doc_id"]: d for d in rd(CORE + "/human_smoldoc/docs.jsonl")}
    out = []
    for r in rd(f"{CORE}/human_smoldoc/refs/{iso}.jsonl"):
        d = src[r["doc_id"]]
        s = split_sentences(d["text"])
        if len(s) == len(r["ref_sents"]) and len(s) >= 2 and all(x.strip() for x in r["ref_sents"]):
            out.append((r["doc_id"], s, [" ".join(x.split()) for x in r["ref_sents"]]))
    return out


BIN_WORDS = {64: 26, 256: 110, 512: 405, 1024: 786, 2048: 1579, 4096: 3141}     # median source words of the core documents per bin (R6 data)


def stitched_human(iso: str, target_words: int, n_docs: int, seed: int = 0, pool: Optional[list] = None) -> List[Tuple[List[str], List[str]]]:
    """Long human documents with a KNOWN 1:1 sentence alignment: equal-count SmolDoc documents stitched (random order) until the English side has >= target_words words.
    Returns [(src_sents, ref_sents)]."""
    pool = pool if pool is not None else smol_equal_docs(iso)
    rng = random.Random(f"{seed}-{iso}-{target_words}")
    order = list(range(len(pool)))
    docs = []
    rng.shuffle(order)
    pos = 0
    for _ in range(n_docs):
        s: List[str] = []
        r: List[str] = []
        words = 0
        while words < target_words:
            if pos >= len(order):
                rng.shuffle(order)
                pos = 0
            _, ss, rr = pool[order[pos]]
            pos += 1
            s += ss
            r += rr
            words += sum(len(x.split()) for x in ss)
        docs.append((s, r))
    return docs
