import os, sys
from huggingface_hub import snapshot_download
repo, name = sys.argv[1], sys.argv[2]
rev = sys.argv[3] if len(sys.argv) > 3 else None
p = snapshot_download(repo, revision=rev, local_dir=f"/dev/shm/lc1_models/{name}", allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt", "*.py"])
print("OK", p)
