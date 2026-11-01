#!/usr/bin/env python3
"""Upload every local clean parquet (argv: directories below out/) that is missing in the Hub repo (agent S3). Run from /workspace/clean_rl with HF_TOKEN set."""
import os,sys
from huggingface_hub import HfApi, CommitOperationAdd
api=HfApi(token=os.environ["HF_TOKEN"]); repo=(os.environ['HF_ORG_LEGACY'] + "/sabiyarn-clean-rl-prompts-2026-10-28")
have={f for f in api.list_repo_files(repo, repo_type="dataset")}
ops=[]
for root in sys.argv[1:]:
    for d,_,fs in os.walk(root):
        for f in fs:
            if not f.endswith(".parquet"): continue
            p=os.path.join(d,f); rp=os.path.relpath(p, "out")
            if rp not in have: ops.append(CommitOperationAdd(path_in_repo=rp, path_or_fileobj=p))
print("missing in repo:", len(ops), "of local", sum(len(fs) for _,_,fs in os.walk(sys.argv[1])))
for i in range(0,len(ops),60):
    api.create_commit(repo, ops[i:i+60], repo_type="dataset", commit_message=f"S3 reconcile ({i//60+1})"); print("committed", i+60, flush=True)
