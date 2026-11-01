#!/usr/bin/env python3
"""Download helper of the clean science-stage rebuild (agent C2, 2026-10-28). Reads the Hub token from the environment (HF_TOKEN) and never prints it.

    HF_TOKEN=... python s4_fetch.py --out DIR prompts  [--langs amh,yor,...] [--pools clean_v3,clean_sci_v2] [--afs]      # <HF_ORG_LEGACY>/sabiyarn-clean-rl-prompts-2026-10-28
    HF_TOKEN=... python s4_fetch.py --out DIR sft-index                                                                   # protected_index/* of <HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28
    HF_TOKEN=... python s4_fetch.py --out DIR push --map local=repo_path [--map ...] [--repo ...]                       # upload (default repo = the prompts repo, folder science_stage/)

Files are downloaded ONE AT A TIME (box rule). The prompts repo holds: clean_v3/<iso>.parquet, clean_sci_v2/<iso>.parquet, afriscience_clean_train/afriscience_clean_train.parquet.
"""
import argparse
import os
import sys

PROMPTS_REPO = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-clean-rl-prompts-2026-10-28")
SFT_REPO = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-clean-sft-2026-10-28")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--langs", default="", help="comma separated iso list (default: all files of the pools)")
    p.add_argument("--pools", default="clean_v3,clean_sci_v2")
    p.add_argument("--afs", action="store_true", help="also fetch afriscience_clean_train/")
    sub.add_parser("sft-index")
    q = sub.add_parser("push")
    q.add_argument("--map", action="append", required=True, help="local_path=repo_path (a directory is uploaded recursively)")
    q.add_argument("--repo", default=PROMPTS_REPO)
    q.add_argument("--message", default="C2 science-stage rebuild artifacts")
    a = ap.parse_args()
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    tok = os.environ.get("HF_TOKEN")
    if not tok:
        sys.exit("HF_TOKEN is not set")
    api = HfApi(token=tok)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "prompts":
        want = []
        langs = [x for x in a.langs.split(",") if x]
        for pool in [x for x in a.pools.split(",") if x]:
            for f in api.list_repo_tree(PROMPTS_REPO, repo_type="dataset", path_in_repo=pool):
                if f.path.endswith(".parquet") and (not langs or os.path.basename(f.path)[:-8] in langs):
                    want.append(f.path)
        if a.afs:
            want += ["afriscience_clean_train/afriscience_clean_train.parquet", "afriscience_clean_train/afriscience_stats.json"]
        for w in want:
            p_ = hf_hub_download(PROMPTS_REPO, w, repo_type="dataset", token=tok, local_dir=a.out)
            print("got", w, os.path.getsize(p_), flush=True)
    elif a.cmd == "sft-index":
        for w in ("protected_index/protected_index.npz", "protected_index/protected_manifest.json"):
            p_ = hf_hub_download(SFT_REPO, w, repo_type="dataset", token=tok, local_dir=a.out)
            print("got", w, os.path.getsize(p_), flush=True)
    else:
        ops = []
        for m in a.map:
            src, dst = m.split("=", 1)
            if os.path.isdir(src):
                for root, _, fs in os.walk(src):
                    for f in fs:
                        if f.startswith(".") or f.endswith(".pyc"):
                            continue
                        pth = os.path.join(root, f)
                        ops.append(CommitOperationAdd(path_in_repo=os.path.join(dst, os.path.relpath(pth, src)).replace("\\", "/"), path_or_fileobj=pth))
            else:
                ops.append(CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src))
        api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
        for i in range(0, len(ops), 100):
            api.create_commit(a.repo, ops[i:i + 100], repo_type="dataset", commit_message=f"{a.message} ({i // 100 + 1})")
        print("pushed", len(ops), "files to", a.repo)


if __name__ == "__main__":
    main()
