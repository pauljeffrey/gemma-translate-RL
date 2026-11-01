#!/usr/bin/env python3
"""Download parts of a Hub repo, resumable, token from HF_TOKEN (never printed).
    python hub_get.py --repo <HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28 --include "data/*" "data_devhalf/*" --out hub
    python hub_get.py --repo Davlan/afro-xlmr-large-76L --type model --include tokenizer.json --out tok_afro"""
import argparse, os
from huggingface_hub import snapshot_download
ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--type", default="dataset"); ap.add_argument("--include", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
p = snapshot_download(a.repo, repo_type=a.type, allow_patterns=a.include, local_dir=a.out, token=os.environ.get("HF_TOKEN"))
print("downloaded to", p)
