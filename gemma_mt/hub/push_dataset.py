# SPDX-License-Identifier: AGPL-3.0-or-later
"""Push a prepared dataset directory (parquet/jsonl files) to the Hub with a generated dataset card.

    python -m gemma_mt.hub.push_dataset --data-dir data/sft_parquet --kind sft            # DRY RUN
    python -m gemma_mt.hub.push_dataset --data-dir data/rl_parquet  --kind rl --yes

Default repos (private): <HF_ORG_LEGACY>/gemma3-270m-en2african-sft-data and ...-rl-prompts. Token from HF_TOKEN.
Nothing is uploaded without --yes. Licence of the card: cc-by-4.0 (override with --license).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from gemma_mt.hub.dataset_card import build_dataset_card

DEFAULT_NAMESPACE = os.environ['HF_ORG_LEGACY']
DEFAULT_REPOS = {"sft": f"{DEFAULT_NAMESPACE}/gemma3-270m-en2african-sft-data",
                 "rl": f"{DEFAULT_NAMESPACE}/gemma3-270m-en2african-rl-prompts"}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--kind", choices=["sft", "rl"], required=True)
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--license", default="cc-by-4.0")
    ap.add_argument("--source", action="append", default=[], help="source dataset name (repeatable, for the card)")
    ap.add_argument("--description", default="")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    repo_id = a.repo_id or DEFAULT_REPOS[a.kind]
    files = sorted(f for f in os.listdir(a.data_dir) if f.endswith((".parquet", ".jsonl", ".jsonl.gz")))
    card = build_dataset_card(repo_id, a.kind, license_id=a.license, description=a.description, source_datasets=a.source)
    print(f"repo: {repo_id} ({'public' if a.public else 'private'})  files: {files}")
    if not files:
        print("no .parquet/.jsonl files in the data dir", file=sys.stderr)
        return 3
    if not a.yes:
        print("\nDRY RUN (no --yes): nothing uploaded.\n")
        print("\n".join(card.splitlines()[:30]))
        return 2
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if not token:
        print("set HF_TOKEN in the environment", file=sys.stderr)
        return 4
    from huggingface_hub import HfApi

    with open(os.path.join(a.data_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=not a.public, exist_ok=True)
    api.upload_folder(folder_path=a.data_dir, repo_id=repo_id, repo_type="dataset",
                      allow_patterns=["*.parquet", "*.jsonl", "*.jsonl.gz", "README.md"], commit_message=f"{a.kind} data")
    print(f"uploaded to https://huggingface.co/datasets/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
