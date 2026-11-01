#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Stage the public benchmark repo (README.md, DATASHEET.md, core/, hard/) in a local directory for scripts/longdoc_push_hub.py (D-44 addendum 2).

    python scripts/longdoc_stage_repo.py --core-dir source_set_v3/core_public --report source_set_v3/SELECTION_REPORT.json --human-dir source_set/human \\
        --hard-dir hard --readme README.md --datasheet DATASHEET.md --out repo

core/docs.jsonl            the public English documents (private rows are not even read: the input is the `core_public` directory)
core/ATTRIBUTION.jsonl     credit line + URL + licence per document
core/SELECTION_REPORT.json funnel and composition (counts only)
core/human_smoldoc/        SmolDoc English documents + the professional translations (CC-BY-4.0, Google SMOL) per language: docs.jsonl, refs/<iso>.jsonl
core/human_stitched/       the same documents stitched into 512 / 1024 / 2048-token documents (derived, same licence)
hard/                      docs.jsonl (public rows only), ATTRIBUTION.jsonl, metric_config.json, SELECTION_REPORT.json, examples.md
Never staged: anything private (core_private, docs_private, *master*), FLORES+ documents (HF-gated, kept out of the public repo).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma_mt.eval.longdoc_release import CANARY  # noqa: E402

SMOL_ATTR = "Google SMOL / SmolDoc (Caswell et al., 2025, arXiv:2502.12301), professionally translated, CC-BY-4.0"
SMOL_URL = "https://huggingface.co/datasets/google/smol"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def stage(a: argparse.Namespace) -> Dict[str, int]:
    out = a.out
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    n: Dict[str, int] = {}
    docs = read_jsonl(os.path.join(a.core_dir, "docs.jsonl"))
    assert all(r.get("split") == "public" for r in docs), "core_dir contains non-public rows"
    n["core/docs.jsonl"] = write_jsonl(os.path.join(out, "core", "docs.jsonl"), docs)
    n["core/ATTRIBUTION.jsonl"] = write_jsonl(os.path.join(out, "core", "ATTRIBUTION.jsonl"), read_jsonl(os.path.join(a.core_dir, "ATTRIBUTION.jsonl")))
    shutil.copy(a.report, os.path.join(out, "core", "SELECTION_REPORT.json"))
    # human subsets
    hdocs = read_jsonl(os.path.join(a.human_dir, "docs_human.jsonl"))
    keep = {"smoldoc": "human_smoldoc", "stitch512": "human_stitched", "stitch1024": "human_stitched", "stitch2048": "human_stitched"}
    ids: Dict[str, str] = {}
    by_sub: Dict[str, List[Dict[str, Any]]] = {"human_smoldoc": [], "human_stitched": []}
    for d in hdocs:
        sub = keep.get(d["doc_id"].split(":")[0])
        if sub is None:
            continue                                       # FLORES+ passages are not re-hosted
        ids[d["doc_id"]] = sub
        by_sub[sub].append({"canary": CANARY, **d, "subset": sub, "split": "public", "source": "smol_doc", "licence": "CC-BY-4.0", "licence_class": "permissive", "url": SMOL_URL,
                            "attribution": SMOL_ATTR})
    for sub, rows in by_sub.items():
        n[f"core/{sub}/docs.jsonl"] = write_jsonl(os.path.join(out, "core", sub, "docs.jsonl"), rows)
    cnt: Dict[str, int] = {}
    for p in sorted(glob.glob(os.path.join(a.human_dir, "refs_human_*.jsonl"))):
        iso = os.path.basename(p)[len("refs_human_"):-len(".jsonl")]
        for sub in by_sub:
            rows = [{"canary": CANARY, **r, "subset": sub} for r in read_jsonl(p) if ids.get(r["doc_id"]) == sub]
            if rows:
                n[f"core/{sub}/refs/{iso}.jsonl"] = write_jsonl(os.path.join(out, "core", sub, "refs", f"{iso}.jsonl"), rows)
                cnt[sub] = cnt.get(sub, 0) + len(rows)
    # hard split (public rows only)
    hd = read_jsonl(os.path.join(a.hard_dir, "docs.jsonl"))
    assert all(r.get("split") == "public" for r in hd), "hard/docs.jsonl contains non-public rows"
    n["hard/docs.jsonl"] = write_jsonl(os.path.join(out, "hard", "docs.jsonl"), hd)
    n["hard/ATTRIBUTION.jsonl"] = write_jsonl(os.path.join(out, "hard", "ATTRIBUTION.jsonl"), read_jsonl(os.path.join(a.hard_dir, "ATTRIBUTION.jsonl")))
    for name in ("metric_config.json", "SELECTION_REPORT.json", "examples.md"):
        p = os.path.join(a.hard_dir, name)
        if os.path.isfile(p):
            shutil.copy(p, os.path.join(out, "hard", name))
    shutil.copy(a.readme, os.path.join(out, "README.md"))
    shutil.copy(a.datasheet, os.path.join(out, "DATASHEET.md"))
    return n


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core-dir", required=True, help="the core_public directory (docs.jsonl, ATTRIBUTION.jsonl)")
    ap.add_argument("--report", required=True)
    ap.add_argument("--human-dir", required=True)
    ap.add_argument("--hard-dir", required=True)
    ap.add_argument("--readme", required=True)
    ap.add_argument("--datasheet", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    n = stage(a)
    print(json.dumps(n, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
