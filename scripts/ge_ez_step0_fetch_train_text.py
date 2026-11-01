"""Fetch (read-only) public training-side text for the vocabulary-swap simulation: Wikipedia am/ti/fon (20231101) and FineTranslations amh/tir/fon (ODC-By).
Never FLORES devtest. Writes _cache only; sizes are logged to train_text_inventory.json."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from ge_ez_step0_common import *
from huggingface_hub import hf_hub_download, HfApi
import pyarrow.parquet as pq
tok = load_token(); api = HfApi(token=tok)
inv = {}
for name, repo, fns in [("wiki_am", "wikimedia/wikipedia", ["20231101.am/train-00000-of-00001.parquet"]),
                        ("wiki_ti", "wikimedia/wikipedia", ["20231101.ti/train-00000-of-00001.parquet"]),
                        ("wiki_fon", "wikimedia/wikipedia", ["20231101.fon/train-00000-of-00001.parquet"])]:
    for fn in fns:
        p = hf_hub_download(repo, fn, repo_type="dataset", token=tok, cache_dir=CACHE)
        t = pq.read_table(p); inv[name] = dict(path=os.path.relpath(p, CACHE), rows=t.num_rows, cols=t.column_names, bytes=os.path.getsize(p))
fs = api.list_repo_files("HuggingFaceFW/finetranslations", repo_type="dataset")
for lang in ("amh_Ethi", "tir_Ethi", "fon_Latn"):
    parts = [f for f in fs if f.startswith(f"data/{lang}/")]
    inv["ft_" + lang + "_files"] = parts
    for fn in parts[:1]:
        p = hf_hub_download("HuggingFaceFW/finetranslations", fn, repo_type="dataset", token=tok, cache_dir=CACHE)
        t = pq.read_table(p); inv["ft_" + lang] = dict(path=os.path.relpath(p, CACHE), rows=t.num_rows, cols=t.column_names, bytes=os.path.getsize(p))
json.dump(inv, open(os.path.join(OUT, "train_text_inventory.json"), "w"), indent=1)
print(json.dumps(inv, indent=1)[:3000])
