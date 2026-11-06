"""E-07(b) step 1: fetch the METADATA columns (iso, src_afr_tokens, length_bin, domain, source_dataset; NOT the English text)
of every file of the stage-A prompt pool from the private Hub datasets (read-only token from env HFT), store as npz/parquet.
General files (74 symlinked + the 7 new-language files come from the local unfiltered_new7 copies) from
<HF_ORG_LEGACY>/sabiyarn-clean-rl-prompts-2026-10-28 clean_v3/; news and ref from <HF_ORG>/gemma-0.4B-rl-prompts clean_v3_385m/{news,ref}/.
"""
import os, sys, json, io
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pool_meta")
os.makedirs(OUT, exist_ok=True)
fs = HfFileSystem(token=os.environ["HFT"])
REPO_ROOT = "/path/to/repo"
syml = [l.split(" -> ")[0].strip() for l in open(f"{REPO_ROOT}/research_archive_2026-11-03/clean_rerun_misc/prompts_symlinks.txt") if " -> " in l]
general = [os.path.basename(s)[:-8] for s in syml]
print(len(general), "symlinked general files", flush=True)
COLS = ["iso", "src_afr_tokens", "length_bin", "domain", "source_dataset"]

def fetch(hubpath, name):
    dst = os.path.join(OUT, name + ".parquet")
    if os.path.exists(dst):
        return
    with fs.open(hubpath, "rb") as fh:
        pf = pq.ParquetFile(fh)
        cols = [c for c in COLS if c in pf.schema_arrow.names]
        t = pf.read(columns=cols)
    pq.write_table(t, dst)
    print(name, t.num_rows, cols, flush=True)

for iso in general:
    fetch(f"datasets/{os.environ['HF_ORG_LEGACY']}/sabiyarn-clean-rl-prompts-2026-10-28/clean_v3/{iso}.parquet", f"general__{iso}")
for kind in ("news", "ref"):
    for f in fs.ls(f"datasets/{os.environ['HF_ORG']}/gemma-0.4B-rl-prompts/clean_v3_385m/{kind}", detail=False):
        iso = os.path.basename(f)[:-8]
        fetch(f, f"{kind}__{iso}")
# the 7 files as they are NOW (filtered), for comparison with the local unfiltered ones
for iso in ("tll", "mfe", "guw", "lub", "crs", "zne", "lue"):
    fetch(f"datasets/{os.environ['HF_ORG']}/gemma-0.4B-rl-prompts/clean_v3_385m/{iso}.parquet", f"filtered7__{iso}")
print("done")
