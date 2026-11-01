"""Download (read-only) the tokenizer files and FLORES+ dev/devtest text used by the Ge'ez Step 0 report."""
import os, sys, json, hashlib
sys.path.insert(0, os.path.dirname(__file__))
from ge_ez_step0_common import *
from huggingface_hub import hf_hub_download, HfApi

tok = load_token()
os.makedirs(CACHE, exist_ok=True)
log = {}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

def grab(repo, fn, rtype="model"):
    try:
        p = hf_hub_download(repo, fn, repo_type=rtype, token=tok, cache_dir=CACHE)
        return p
    except Exception as e:
        log[f"{repo}/{fn}"] = "FAIL " + type(e).__name__ + ": " + str(e)[:120]
        return None

files = {}
for repo in [(os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), "google/gemma-3-1b-it", "google/gemma-3-270m-it", "google/gemma-3-4b-it",
             "facebook/nllb-200-distilled-600M"]:
    for fn in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json", "sentencepiece.bpe.model", "special_tokens_map.json", "config.json"]:
        p = grab(repo, fn)
        if p:
            files[f"{repo}/{fn}"] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
for fn in ["latest/tokenizer.json", "latest/tokenizer_config.json", "latest/config.json", "latest/chat_template.jinja", "latest/generation_config.json"]:
    p = grab((os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), fn)
    if p:
        files[f"{os.environ['HF_ORG_LEGACY']}/gemma-3-270m-translate-rl2c/{fn}"] = {"sha256": sha(p), "bytes": os.path.getsize(p)}
for split in ["dev", "devtest"]:
    for iso, code in FLORES.items():
        p = grab("openlanguagedata/flores_plus", f"{split}/{code}.parquet", "dataset") or grab("openlanguagedata/flores_plus", f"{split}/{code}.jsonl", "dataset")
        if p:
            files[f"flores_plus/{split}/{code}"] = {"path": os.path.relpath(p, CACHE), "bytes": os.path.getsize(p)}
json.dump({"files": files, "failures": log}, open(os.path.join(OUT, "fetch_log.json"), "w"), indent=1)
print(json.dumps({"ok": list(files), "fail": log}, indent=1))
