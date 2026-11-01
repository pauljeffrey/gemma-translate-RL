#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Upload the `full_pairs/` tree to the dataset repo with the Hub's resumable large-folder uploader, and (optionally) mirror it to the object-storage bucket. DRY RUN unless --yes.

    python scripts/afri2afri_push_full.py --root /workspace/afri2afri/hub_stage --repo-id <HF_ORG_LEGACY>/interlaanguage-translation            # lists what would go
    python scripts/afri2afri_push_full.py --root ... --repo-id ... --yes [--public]                                                                  # Hub
    python scripts/afri2afri_push_full.py --root ... --bucket remote:<S3_BUCKET>/afri2afri --yes --bucket-only                                  # bucket mirror (rclone, creds in the environment)

`--root` is a directory that CONTAINS `full_pairs/` (the repo paths are the paths below root; the uploader has no path_in_repo). Re-running resumes: the uploader keeps its state in
`<root>/.cache/huggingface` and skips what is already committed. The token (HF_WRITE_TOKEN, then HF_TOKEN) is read from the environment and never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", default=(os.environ['HF_ORG_LEGACY'] + "/interlaanguage-translation"))
    ap.add_argument("--bucket", default=None, help=("rclone destination, e.g. remote:" + os.environ['S3_BUCKET'] + "/afri2afri"))
    ap.add_argument("--bucket-only", action="store_true")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    tree = os.path.join(a.root, "full_pairs")
    files = []
    for d, _, fs in os.walk(tree):
        for f in fs:
            if f.endswith((".parquet", ".json")) and not f.startswith("_hf") and "/_hf_cache" not in d:
                files.append(os.path.join(d, f))
    total = sum(os.path.getsize(f) for f in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB under {tree}")
    man = os.path.join(tree, "MANIFEST.json")
    if not os.path.exists(man) or a.yes:
        entries = {}
        for f in sorted(files):
            if f.endswith("MANIFEST.json"):
                continue
            entries[os.path.relpath(f, a.root)] = {"bytes": os.path.getsize(f), "sha256": sha256(f)} if a.yes else {"bytes": os.path.getsize(f)}
        json.dump({"files": entries, "total_bytes": total}, open(man, "w"), indent=0)
    if not a.yes:
        print("DRY RUN: nothing uploaded (add --yes)")
        return 2
    if a.bucket:
        env = dict(os.environ)
        r = subprocess.run(["rclone", "copy", tree, a.bucket.rstrip("/") + "/full_pairs", "--transfers", "4", "--s3-upload-concurrency", "6", "--exclude", "_hf_cache/**", "--exclude", "*.empty",
                            "--stats", "60s", "--stats-one-line"], env=env)
        print("bucket mirror exit code", r.returncode)
        if a.bucket_only:
            return r.returncode
    token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("no HF_WRITE_TOKEN / HF_TOKEN in the environment", file=sys.stderr)
        return 4
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(a.repo_id, repo_type="dataset", private=not a.public, exist_ok=True)
    api.upload_large_folder(repo_id=a.repo_id, repo_type="dataset", folder_path=a.root, allow_patterns=["full_pairs/**/*.parquet", "full_pairs/*.parquet", "full_pairs/MANIFEST.json"],
                            ignore_patterns=["**/_hf_cache/**", "**/*.empty"], num_workers=a.workers)
    print(f"https://huggingface.co/datasets/{a.repo_id}/tree/main/full_pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
