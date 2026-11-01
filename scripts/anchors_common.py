# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared helpers of scripts/build_anchors_calibration.py and scripts/eval_anchors.py (D-56): the human reference sets, pseudo-documents and the synthetic corruptions.

Human text (calibration of THRESHOLDS only, never training data):
    FLORES+ dev                 from the Hub (`openlanguagedata/flores_plus`, dev/<code>.jsonl; loader as in scripts/build_word_ratio_calibration.py), cached under --cache-dir
    smol_sent / ibom_mt / glot_story / salt / flores_moz / ntrex      local copies under --test-sets (the dev-style public sets of the benchmark suite)
    documents                   smol_doc (human translated documents) and, optionally, a jsonl of whole-paper references (AfriScience-MT dev) for the long length buckets
A "unit" is {src, ref, set, id}. Pseudo-documents are windows of k consecutive sentences of one set (ids in numeric order): they give the repetition detector human
reference texts of 50-600 words.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

LOCAL_SETS = ("smol_sent", "ibom_mt", "glot_story", "salt", "flores_moz", "ntrex")
Unit = Dict[str, str]


def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def flores_dev(isos: Sequence[str], cache_dir: Optional[str] = None, log=print) -> Dict[str, List[Unit]]:
    """FLORES+ dev sentence pairs (English source, reference) for every language of `isos` that has a FLORES+ code."""
    from huggingface_hub import hf_hub_download

    from gemma_mt.languages import LANGUAGES

    def hub(code: str) -> List[dict]:
        p = hf_hub_download("openlanguagedata/flores_plus", f"dev/{code}.jsonl", repo_type="dataset", token=os.environ.get("HF_TOKEN"), cache_dir=cache_dir)
        return read_jsonl(p)

    en = {int(r["id"]): r["text"] for r in hub("eng_Latn")}
    out: Dict[str, List[Unit]] = defaultdict(list)
    for iso in isos:
        code = LANGUAGES[iso].flores
        if not code:
            continue
        codes = ["twi_Latn_akua1239", "twi_Latn_asan1239"] if code == "twi_Latn" else [code]
        try:
            for c in codes:
                for r in hub(c):
                    if int(r["id"]) in en and r["text"].strip():
                        out[iso].append({"src": en[int(r["id"])].strip(), "ref": r["text"].strip(), "set": "flores_dev", "id": str(r["id"])})
        except Exception as e:
            log(f"[anchors] FLORES+ dev {iso} ({code}): {str(e)[:100]}")
    return out


def local_sets(isos: Sequence[str], test_sets: str, sets: Sequence[str] = LOCAL_SETS, log=print) -> Dict[str, List[Unit]]:
    from gemma_mt.eval import benchmarks as B

    out: Dict[str, List[Unit]] = defaultdict(list)
    for b in sets:
        path = os.path.join(test_sets, b)
        if not os.path.exists(path):
            log(f"[anchors] {path} missing")
            continue
        for iso in isos:
            if not B.code_for(b, iso):
                continue
            try:
                rows = B.load_benchmark(b, path=path, isos=[iso])
            except Exception as e:
                log(f"[anchors] {b} {iso}: {repr(e)[:100]}")
                continue
            for r in rows:
                if r["iso"] == iso and r["reference"].strip() and r["english"].strip():
                    out[iso].append({"src": r["english"].strip(), "ref": r["reference"].strip(), "set": b, "id": str(r["id"])})
    return out


def documents(isos: Sequence[str], test_sets: str, doc_files: Sequence[str] = (), log=print) -> Dict[str, List[Unit]]:
    """Human documents: smol_doc + optional jsonl files with english / reference / iso (whole papers)."""
    from gemma_mt.eval import benchmarks as B

    out: Dict[str, List[Unit]] = defaultdict(list)
    path = os.path.join(test_sets, "smol_doc")
    if os.path.exists(path):
        for iso in isos:
            if B.code_for("smol_doc", iso):
                try:
                    for r in B.load_benchmark("smol_doc", path=path, isos=[iso]):
                        if r["iso"] == iso and r["reference"].strip():
                            out[iso].append({"src": r["english"].strip(), "ref": r["reference"].strip(), "set": "smol_doc", "id": str(r["id"])})
                except Exception as e:
                    log(f"[anchors] smol_doc {iso}: {repr(e)[:100]}")
    for fp in doc_files:
        for r in read_jsonl(fp):
            if r.get("iso") in isos and r.get("reference") and r.get("english"):
                out[r["iso"]].append({"src": r["english"], "ref": r["reference"], "set": os.path.basename(fp), "id": str(r.get("id", ""))})
    return out


def dedupe(units: Iterable[Unit]) -> List[Unit]:
    seen, out = set(), []
    for u in units:
        k = (" ".join(u["src"].split()).casefold(), " ".join(u["ref"].split()).casefold())
        if k not in seen:
            seen.add(k)
            out.append(u)
    return out


def _idkey(u: Unit):
    try:
        return (u["set"], 0, int(u["id"]))
    except ValueError:
        return (u["set"], 1, u["id"])


def pseudo_documents(units: Sequence[Unit], ks: Sequence[int] = (4, 10, 25)) -> List[Unit]:
    """Non-overlapping windows of k consecutive sentences (same set, numeric id order); src/ref joined with a space."""
    by: Dict[str, List[Unit]] = defaultdict(list)
    for u in sorted(units, key=_idkey):
        by[u["set"]].append(u)
    out: List[Unit] = []
    for s, us in by.items():
        for k in ks:
            for i in range(0, len(us) - k + 1, k):
                w = us[i: i + k]
                out.append({"src": " ".join(x["src"] for x in w), "ref": " ".join(x["ref"] for x in w), "set": f"{s}/w{k}", "id": w[0]["id"]})
    return out


# ------------------------------------------------------------------------------------------------ corruptions of a reference (for the detection rates)
def sentences(text: str) -> List[str]:
    import re

    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def corrupt_half_dropped(ref: str, rng: random.Random) -> str:
    """the second half of the output is missing (sentences when there are several, words otherwise)"""
    ss = sentences(ref)
    if len(ss) >= 2:
        return " ".join(ss[: max(1, len(ss) // 2)])
    w = ref.split()
    return " ".join(w[: max(1, len(w) // 2)])


def corrupt_appended(ref: str, other: str, rng: random.Random) -> str:
    """a sentence about something else is appended (the reference of a different source)"""
    return ref.rstrip() + " " + other.strip()


def corrupt_other_source(other: str) -> str:
    return other


def corrupt_looped(ref: str, rng: random.Random, times: int = 6) -> str:
    """the greedy-decoding loop: the last clause (up to 8 words) repeated until the budget is gone"""
    w = ref.split()
    tail = " ".join(w[-min(8, max(3, len(w) // 3)):])
    return ref.rstrip() + (" " + tail) * times


def corrupt_doubled(ref: str, rng: random.Random) -> str:
    """the whole translation twice"""
    return ref.rstrip() + " " + ref.strip()
