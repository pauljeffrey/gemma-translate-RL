#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Push the staged benchmark repo (README.md, DATASHEET.md, core/, hard/) to a PRIVATE Hugging Face dataset repo and verify it (D-44 addendum 2).

    python scripts/longdoc_push_hub.py --stage longdoc_bench/repo --repo <HF_ORG>/AfriLongDoc --token-file /workspace/sft_build/.hf_write_token [--fallback-namespace <HF_ORG_LEGACY>]

Safety rules enforced BEFORE anything is uploaded (the push is refused otherwise):
  * no file below a directory called `private`, no file whose name contains `private`;
  * no row of any .jsonl file with `"split": "private"`;
  * every .jsonl row that has a text field carries the canary string; README.md carries it too;
  * no token-looking string (hf_..., sk-...) in any staged text file;
  * the repo is created / kept PRIVATE (a repo that is public is refused).
The token file is read, never printed. After the upload the README is downloaded again and compared, and the number of rows of every .jsonl file is re-counted from the
downloaded copy. If a commit is refused for the private-storage quota of the namespace, the push is repeated under --fallback-namespace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma_mt.eval.longdoc_release import CANARY  # noqa: E402

TOKEN_RE = re.compile(r"\b(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,})\b")


def preflight(stage: str) -> Dict[str, Any]:
    """Raises SystemExit with the list of problems; returns {relative path: {bytes, rows}} of the files to upload."""
    problems: List[str] = []
    files: Dict[str, Any] = {}
    for root, _, names in os.walk(stage):
        for n in names:
            p = os.path.join(root, n)
            rel = os.path.relpath(p, stage)
            parts = rel.lower().split(os.sep)
            if any(x == "private" or "private" in x for x in parts):
                problems.append(f"private path: {rel}")
                continue
            if rel.startswith(".cache") or n.startswith("."):
                continue
            info: Dict[str, Any] = {"bytes": os.path.getsize(p)}
            if n.endswith(".jsonl"):
                rows, no_canary = 0, 0
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rows += 1
                        r = json.loads(line)
                        if r.get("split") == "private":
                            problems.append(f"private row in {rel}")
                            break
                        if "text" in r and r.get("canary") != CANARY:
                            no_canary += 1
                if no_canary:
                    problems.append(f"{no_canary} rows without the canary in {rel}")
                info["rows"] = rows
            if n.endswith((".md", ".json", ".jsonl", ".txt", ".yaml", ".yml")) and info["bytes"] < 400_000_000:
                with open(p, encoding="utf-8", errors="ignore") as f:
                    head = f.read(50_000_000)
                if TOKEN_RE.search(head):
                    problems.append(f"token-like string in {rel}")
                if n == "README.md" and CANARY not in head:
                    problems.append("README.md without the canary")
            files[rel] = info
    for need in ("README.md", "DATASHEET.md"):
        if need not in files:
            problems.append(f"missing {need}")
    if problems:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(problems))
    return files


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def push(stage: str, repo: str, token: str, message: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.dataset_info(repo)
    if not info.private:
        raise SystemExit(f"REFUSED: {repo} is not private")
    api.upload_folder(folder_path=stage, repo_id=repo, repo_type="dataset", commit_message=message, ignore_patterns=[".*", "**/.*"])
    return api.dataset_info(repo).sha


def verify(stage: str, repo: str, token: str, files: Dict[str, Any]) -> Dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=token)
    listed = set(api.list_repo_files(repo, repo_type="dataset"))
    missing = [f for f in files if f not in listed]
    out: Dict[str, Any] = {"missing": missing, "extra": sorted(x for x in listed if x not in files and x != ".gitattributes"), "rows": {}, "mismatch": []}
    with tempfile.TemporaryDirectory() as tmp:
        for rel, info in sorted(files.items()):
            if rel == "README.md" or rel.endswith(".jsonl") or rel == "DATASHEET.md":
                got = hf_hub_download(repo, rel, repo_type="dataset", token=token, local_dir=tmp)
                if sha256(got) != sha256(os.path.join(stage, rel)):
                    out["mismatch"].append(rel)
                if rel.endswith(".jsonl"):
                    with open(got, encoding="utf-8") as f:
                        out["rows"][rel] = sum(1 for line in f if line.strip())
                    if out["rows"][rel] != info["rows"]:
                        out["mismatch"].append(f"{rel}: rows {out['rows'][rel]} != {info['rows']}")
                else:
                    with open(got, encoding="utf-8") as f:
                        out[f"{rel}_chars"] = len(f.read())
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--token-file", required=True)
    ap.add_argument("--fallback-namespace", default=None)
    ap.add_argument("--message", default="AfriLongDoc: dataset card, datasheet, core and hard English source documents (no translations or scores yet)")
    ap.add_argument("--check-only", action="store_true", help="run the pre-flight checks and print the file list, upload nothing")
    a = ap.parse_args(argv)
    files = preflight(a.stage)
    print(json.dumps({"files": len(files), "bytes": sum(v["bytes"] for v in files.values()), "rows": {k: v["rows"] for k, v in files.items() if "rows" in v}}, indent=1))
    if a.check_only:
        return 0
    with open(a.token_file, encoding="utf-8") as f:
        token = f.read().strip()
    repo = a.repo
    try:
        sha = push(a.stage, repo, token, a.message)
    except Exception as e:                                               # noqa: BLE001
        msg = str(e).lower()
        if a.fallback_namespace and ("quota" in msg or "storage" in msg or "private" in msg):
            repo = f"{a.fallback_namespace}/{a.repo.split('/', 1)[1]}"
            print(f"push refused ({type(e).__name__}); retrying as {repo}")
            sha = push(a.stage, repo, token, a.message)
        else:
            print(f"push failed: {type(e).__name__}: {str(e)[:300]}")
            return 1
    res = verify(a.stage, repo, token, files)
    print(json.dumps({"repo": repo, "url": f"https://huggingface.co/datasets/{repo}", "commit": sha, **res}, indent=1))
    return 0 if not res["missing"] and not res["mismatch"] else 2


if __name__ == "__main__":
    sys.exit(main())
