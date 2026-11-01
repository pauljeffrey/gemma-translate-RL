#!/usr/bin/env python3
# SUPERSEDED 2026-10-29 by r4_run_all.py (its push step also deletes the IbomMT-derived Hub files). Do not run: it would re-upload files that may contain IbomMT-derived values.
"""r4: push the NUMBERS of the clean calibration to the private Hub repo <HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28 under calibration/ (token from the environment, never printed).
Only json files of clean/ and the per-sentence QE score files of qe_work/ (ids + scores, no text) and the split v2 file go up. refs/, ibom_halves/, sft_targets*.jsonl, fw_corpus/ never do.
    HF_TOKEN=... python r4_push_calibration.py --work /workspace/calib_r4 --split heldout_split_cal_sel_v2.json
"""
import argparse, os, sys
from huggingface_hub import HfApi

REPO = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-clean-sft-2026-10-28")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--split", required=True)
    a = ap.parse_args()
    api = HfApi(token=os.environ["HF_TOKEN"])
    ops = []
    from huggingface_hub import CommitOperationAdd
    clean = os.path.join(a.work, "clean")
    for fn in sorted(os.listdir(clean)):
        p = os.path.join(clean, fn)
        if os.path.isfile(p) and (fn.endswith(".json") or fn.endswith(".npz")):
            ops.append(CommitOperationAdd(path_in_repo=f"calibration/{fn}", path_or_fileobj=p))
    qe = os.path.join(a.work, "qe_work")
    for m in sorted(os.listdir(qe)) if os.path.isdir(qe) else []:
        for fn in sorted(os.listdir(os.path.join(qe, m))):
            if fn.endswith(".json"):
                ops.append(CommitOperationAdd(path_in_repo=f"calibration/qe_work/{m}/{fn}", path_or_fileobj=os.path.join(qe, m, fn)))
    ops.append(CommitOperationAdd(path_in_repo="calibration/heldout_split_cal_sel_v2.json", path_or_fileobj=a.split))
    ops.append(CommitOperationAdd(path_in_repo="protected_index/heldout_split_cal_sel_v2.json", path_or_fileobj=a.split))
    r = api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops, commit_message="r4: clean calibration refits (numbers only, H_cal v2)")
    print("pushed", len(ops), "files", r.commit_url.split("/")[-1][:12])


if __name__ == "__main__":
    main()
