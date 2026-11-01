import os, sys
os.environ["HF_HUB_DISABLE_XET"] = "0"
from huggingface_hub import snapshot_download
os.makedirs("/dev/shm/lc1_models", exist_ok=True)
which = sys.argv[1]
if which == "600m":
    p = snapshot_download("facebook/nllb-200-distilled-600M", local_dir="/dev/shm/lc1_models/nllb600m-hf", allow_patterns=["*.json", "*.model", "pytorch_model.bin"])
elif which == "3b3":
    p = snapshot_download("OpenNMT/nllb-200-3.3B-ct2-int8", local_dir="/dev/shm/lc1_models/nllb3p3b-ct2-int8")
print("OK", p)
