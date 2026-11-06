#!/usr/bin/env python
"""Push one run's weights to a branch of the PUBLIC research repo <HF_ORG>/gemma-0.4B-translate-rl-experiments, verify, and (optionally) delete the box copy.
    python hf_push.py MODEL_DIR BRANCH RUN_ID "recipe text" "data text" [--delete]
Never writes to the public gemma-0.4B-translate-rl repo. The token comes from the environment (HF_TOKEN) and is never printed. Verification: the branch listing (files and sizes) and the LFS sha256 of
model.safetensors against the sha256 of the local file. With --delete the model dir is removed ONLY after the verification succeeded (a symlinked dir: its files)."""
import hashlib, os, shutil, sys, tempfile
from huggingface_hub import HfApi
REPO = (os.environ['HF_ORG'] + "/gemma-0.4B-translate-rl-experiments")
assert REPO.endswith("-experiments")
model_dir, branch, run_id, recipe, data = sys.argv[1:6]; delete = "--delete" in sys.argv
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(REPO, private=False, exist_ok=True, repo_type="model")      # owner 2026-11-05: public repo (private repos are not available); NO README / model card is written
if api.model_info(REPO).private:
    api.update_repo_settings(REPO, private=False)
api.create_branch(REPO, branch=branch, exist_ok=True)
sha = hashlib.sha256()
with open(os.path.join(model_dir, "model.safetensors"), "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""): sha.update(c)
local = sha.hexdigest()
local_unused = None
keep = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")
with tempfile.TemporaryDirectory(dir="/dev/shm") as tmp:
    for f in keep:
        p = os.path.join(model_dir, f)
        if os.path.exists(p): os.symlink(os.path.realpath(p), os.path.join(tmp, f))
    api.upload_folder(folder_path=tmp, repo_id=REPO, revision=branch, commit_message=f"{run_id}")
inf = api.model_info(REPO, revision=branch, files_metadata=True)
files = {s.rfilename: (s.size, (s.lfs.sha256 if s.lfs else None)) for s in inf.siblings}
for k, (sz, _) in sorted(files.items()): print("  ", k, sz)
ok = files.get("model.safetensors", (0, None))[1] == local
print("VERIFY", branch, "sha256 match" if ok else "MISMATCH", local[:12], flush=True)
if not ok: sys.exit(2)
if delete:
    for f in os.listdir(model_dir):
        try: os.remove(os.path.join(model_dir, f))
        except IsADirectoryError: shutil.rmtree(os.path.join(model_dir, f))
    print("box copy deleted:", model_dir)
