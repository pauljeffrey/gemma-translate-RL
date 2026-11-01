#!/usr/bin/env python3
"""Push the packs, schedules, gate outputs and QE scores to a PRIVATE Hub repo (token from HF_TOKEN). Never pushes pair text, windows, or benchmark text.
    python push_packs.py --repo <HF_ORG>/sabiyarn-clean-sft-packs-2026-10-29 --work work [--what packs sched gate scores logs]"""
import argparse, os
from huggingface_hub import HfApi
ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--work", required=True); ap.add_argument("--what", nargs="*", default=["packs", "sched", "gate", "scores", "logs", "code"])
a = ap.parse_args()
api = HfApi(token=os.environ["HF_TOKEN"]); api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
m = {"packs": ("packed", "packed", None), "sched": ("sched", "sched", None), "gate": ("gate", "gate", ["*.csv", "*.json", "keep_*.parquet"]), "scores": ("scores_gpu", "qe_scores_gpu", ["*.parquet"]), "logs": ("../logs", "logs", ["*.log", "*.tsv", "*.md"]), "code": ("../code/sft_clean", "code", ["*.py", "*.sh", "*.md", "*.example"])}
for k in a.what:
    src, dst, pat = m[k]; p = os.path.normpath(os.path.join(a.work, src))
    if os.path.isdir(p):
        api.upload_folder(folder_path=p, path_in_repo=dst, repo_id=a.repo, repo_type="dataset", allow_patterns=pat, commit_message=f"clean sft chain: {k}")
        print("pushed", k, "->", dst)
