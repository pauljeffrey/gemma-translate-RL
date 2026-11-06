"""E-07(b) step 1b: the pool's hold-out split (128 eval prompts = smallest blake2b hash of the English text) removes rows from the pool before the schedule indexes it,
so the English text of every loaded row is needed to place the hold-out rows. Fetch the `english` column (+ the row's iso/src_afr_tokens for the filter) of
every pool file into a scratch dir (NOT the repo): argv[1] = out dir.  Read-only token in env HFT."""
import os, sys
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
fs = HfFileSystem(token=os.environ["HFT"])
REPO_ROOT = "/path/to/repo"
syml = [l.split(" -> ")[0].strip() for l in open(f"{REPO_ROOT}/research_archive_2026-11-03/clean_rerun_misc/prompts_symlinks.txt") if " -> " in l]
general = [os.path.basename(s)[:-8] for s in syml]
def fetch(hubpath, name):
    dst = os.path.join(OUT, name + ".parquet")
    if os.path.exists(dst): return
    with fs.open(hubpath, "rb") as fh:
        t = pq.ParquetFile(fh).read(columns=["english"])
    pq.write_table(t, dst, compression="zstd"); print(name, t.num_rows, flush=True)
for iso in general: fetch(f"datasets/{os.environ['HF_ORG_LEGACY']}/sabiyarn-clean-rl-prompts-2026-10-28/clean_v3/{iso}.parquet", f"general__{iso}")
for kind in ("news", "ref"):
    for f in fs.ls(f"datasets/{os.environ['HF_ORG']}/gemma-0.4B-rl-prompts/clean_v3_385m/{kind}", detail=False):
        iso = os.path.basename(f)[:-8]; fetch(f, f"{kind}__{iso}")
print("done")
