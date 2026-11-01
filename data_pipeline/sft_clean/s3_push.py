#!/usr/bin/env python3
"""Push small artifacts of agent S3 to the private Hub dataset <HF_ORG_LEGACY>/sabiyarn-clean-rl-prompts-2026-10-28 (new repo; no existing repo is touched).
    HF_TOKEN=... python s3_push.py --repo <HF_ORG_LEGACY>/sabiyarn-clean-rl-prompts-2026-10-28 --map local_path=repo_path [--map ...] [--message text]
A directory on the left is uploaded recursively below the repo path on the right. One commit."""
import argparse
import os

from huggingface_hub import CommitOperationAdd, HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--map", action="append", required=True)
ap.add_argument("--message", default="S3 artifacts")
a = ap.parse_args()
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
ops = []
for m in a.map:
    src, dst = m.split("=", 1)
    if os.path.isdir(src):
        for root, _, fs in os.walk(src):
            for f in fs:
                if f.startswith(".") or f.endswith(".pyc"):
                    continue
                p = os.path.join(root, f)
                ops.append(CommitOperationAdd(path_in_repo=os.path.join(dst, os.path.relpath(p, src)).replace("\\", "/"), path_or_fileobj=p))
    else:
        ops.append(CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src))
print(len(ops), "files")
for i in range(0, len(ops), 100):
    api.create_commit(a.repo, ops[i:i + 100], repo_type="dataset", commit_message=f"{a.message} ({i // 100 + 1})")
print("pushed")
