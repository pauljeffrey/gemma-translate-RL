#!/usr/bin/env python3
"""Push the final artifacts to the private Hub repo (token from HF_TOKEN, never printed). Benchmark TEXT is never pushed: only hashes (protected_index/) and counts.
    python push_artifacts.py --repo <HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28 --out out --protected protected --scripts .
"""
import argparse, os
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--out"); ap.add_argument("--protected"); ap.add_argument("--scripts"); ap.add_argument("--readme")
a = ap.parse_args()
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
if a.out:
    for sub, pats in (("data", ["final/*.parquet"]), ("stats", ["stats/*"]), ("quarantine_nc", ["quarantine_nc/*"]), ("data_devhalf", ["data_devhalf/*"]), ("specialist", ["specialist/*"])):
        if os.path.isdir(os.path.join(a.out, pats[0].split("/")[0])):
            api.upload_folder(folder_path=os.path.join(a.out, pats[0].split("/")[0]), path_in_repo=sub, repo_id=a.repo, repo_type="dataset", commit_message=f"clean sft: {sub}")
if a.protected:
    api.upload_folder(folder_path=a.protected, path_in_repo="protected_index", repo_id=a.repo, repo_type="dataset",
                      allow_patterns=["protected_index.npz", "protected_manifest.json", "protected_manifest_v1.json", "protected_sentence_hashes.npz", "protected_exemption_devhalf.npz"], commit_message="protected-set hash index (hashes only)")
if a.scripts:
    api.upload_folder(folder_path=a.scripts, path_in_repo="scripts", repo_id=a.repo, repo_type="dataset", allow_patterns=["*.py"], ignore_patterns=["__pycache__/*"],
                      commit_message="scripts")
if a.readme:
    api.upload_file(path_or_fileobj=a.readme, path_in_repo="README.md", repo_id=a.repo, repo_type="dataset")
print("pushed")
