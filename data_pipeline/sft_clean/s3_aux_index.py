#!/usr/bin/env python3
"""Auxiliary FLAG-ONLY hash index (agent S3, 2026-10-28): the TRAIN split of SSA-MTE (McGill-NLP/SSA-MTE, CC-BY-SA-4.0), which trained / validated the reward model SSA-COMET-MTL and is
the T2 calibration set of the rerun. It is NOT part of the protected index (the train split is a training / calibration resource), so prompts that overlap it are FLAGGED, not removed.

One file at a time: download -> index every string cell -> delete. Only hashes are kept (same format as decon.Index). Usage:
    HF_TOKEN=... python s3_aux_index.py --out aux/ssa_mte_train
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decon as D  # noqa: E402
import fetch_protected as F  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="McGill-NLP/SSA-MTE")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ["HF_TOKEN"]
    api = HfApi(token=tok)
    os.makedirs(a.out, exist_ok=True)
    info = api.dataset_info(a.repo)
    files = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith("train/") and s.rfilename.endswith(".json"))
    B = D.Builder()
    tmp = os.path.join(a.out, "_dl")
    t0 = time.time()
    for f in files:
        p = hf_hub_download(a.repo, f, repo_type="dataset", token=tok, local_dir=tmp)
        for s in F.iter_file(p):
            B.add("ssa_mte:train", s)
        os.remove(p)
        print(f, B.count["ssa_mte:train"], flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    ix = B.finish()
    ix.save(os.path.join(a.out, "ssa_mte_train_index.npz"))
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "repo": a.repo, "commit": info.sha, "licence": "cc-by-sa-4.0 (Hub card)", "files": files, "strings_added": B.count["ssa_mte:train"],
           "digest": ix.digest(), "role": "flag-only auxiliary index: T2 calibration + reward-model training data; not protected, overlaps are reported, not removed",
           "seconds": round(time.time() - t0)}
    json.dump(man, open(os.path.join(a.out, "ssa_mte_train_manifest.json"), "w"), indent=1)
    print("done", json.dumps(man["digest"]))


if __name__ == "__main__":
    main()
