#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Push the staged inter-African dataset to the Hub in two steps. DRY RUN unless --yes. The token is read from the environment (HF_WRITE_TOKEN, then HF_TOKEN) and never printed.

    python scripts/afri2afri_push_hub.py --stage-dir OUT/export --repo-id <HF_ORG_LEGACY>/interlaanguage-translation --step sample            # dry run
    python scripts/afri2afri_push_hub.py --stage-dir OUT/export --repo-id ... --step sample --yes --public                                  # card + samples
    python scripts/afri2afri_push_hub.py --stage-dir OUT/export --repo-id ... --step full --yes --public                                    # full files

`sample`: README.md (the card), MANIFEST.json, direction_counts.csv and `samples/*.parquet` (a few hundred rows per file type).
`full`  : stage/*.parquet at the repo root (sft_pairs_train, sft_pairs_val, rl_prompts). Each file is its own commit, so a failed upload resumes from the next call
          (files already present with the same size are skipped).
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-dir", required=True, help="the `export` directory written by build_afri2afri_sft.py export (contains stage/, sample/, README.md, MANIFEST.json)")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--step", choices=["sample", "full"], required=True)
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    sd = a.stage_dir
    plan = []
    if a.step == "sample":
        for name in ("README.md", "MANIFEST.json", "direction_counts.csv", "source_datasets.csv"):
            p = os.path.join(sd, name)
            if os.path.exists(p):
                plan.append((p, name))
        for fn in sorted(os.listdir(os.path.join(sd, "sample"))):
            plan.append((os.path.join(sd, "sample", fn), "samples/" + fn))
    else:
        for fn in sorted(os.listdir(os.path.join(sd, "stage"))):
            if fn.endswith(".parquet"):
                plan.append((os.path.join(sd, "stage", fn), fn))
    for p, dst in plan:
        print(f"{os.path.getsize(p):>14,}  {dst}")
    print(f"repo {a.repo_id} ({'PUBLIC' if a.public else 'private'}), {len(plan)} files")
    if not a.yes:
        print("DRY RUN: nothing uploaded (add --yes)")
        return 2
    token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("no HF_WRITE_TOKEN / HF_TOKEN in the environment", file=sys.stderr)
        return 4
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(a.repo_id, repo_type="dataset", private=not a.public, exist_ok=True)
    have = {}
    try:
        for f in api.list_repo_tree(a.repo_id, repo_type="dataset", recursive=True):
            have[f.path] = getattr(f, "size", None)
    except Exception:
        pass
    for p, dst in plan:
        if have.get(dst) == os.path.getsize(p) and dst.endswith(".parquet"):
            print("skip (same size already there):", dst)
            continue
        api.upload_file(path_or_fileobj=p, path_in_repo=dst, repo_id=a.repo_id, repo_type="dataset", commit_message=f"{a.step}: {dst}")
        print("uploaded", dst, flush=True)
    print(f"https://huggingface.co/datasets/{a.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
