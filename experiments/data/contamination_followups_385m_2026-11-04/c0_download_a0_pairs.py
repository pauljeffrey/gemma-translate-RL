"""E-07(c) step 0: download the clean A0 pair text (main output + specialist, pre-gate superset of A0's translation pairs) and the clean
AfriScience-MT train (34,007 rows) from the private Hub datasets (read-only token in env HFT) into a scratch directory (not the repo)."""
import os, sys
from huggingface_hub import snapshot_download, hf_hub_download
dst = sys.argv[1]; tok = os.environ["HFT"]
snapshot_download((os.environ['HF_ORG_LEGACY'] + "/sabiyarn-clean-sft-2026-10-28"), repo_type="dataset", token=tok, local_dir=dst,
                  allow_patterns=["data/*.parquet", "specialist/*.parquet", "data_devhalf/*.parquet", "README.md", "protected_index/protected_manifest*.json", "stats/*.csv", "stats/*.json"])
hf_hub_download((os.environ['HF_ORG_LEGACY'] + "/sabiyarn-clean-rl-prompts-2026-10-28"), "afriscience_clean_train/afriscience_clean_train.parquet", repo_type="dataset", token=tok, local_dir=dst)
print("done")
