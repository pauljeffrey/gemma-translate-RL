import os, time
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); r = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-clean-sft-2026-10-28")
api.upload_folder(folder_path="/workspace/work/qe_cpu/scores_cpu", path_in_repo="qe_scores_cpu/scores", repo_id=r, repo_type="dataset", allow_patterns=["*.parquet"], commit_message="QE CPU scores (progress)")
for f, dst in (("progress.json", "qe_scores_cpu/progress.json"), ("queue/queue_meta.json", "qe_scores_cpu/queue_meta.json"), ("queue/queue_manifest.json", "qe_scores_cpu/queue_manifest.json"), ("status.md", "qe_scores_cpu/status.md")):
    p = "/workspace/work/qe_cpu/" + f
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=dst, repo_id=r, repo_type="dataset")
print("pushed", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
