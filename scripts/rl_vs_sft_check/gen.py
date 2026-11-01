"""Greedy generation of one checkpoint over the pool.  python gen.py POOL.jsonl OUT.jsonl REPO REVISION [BATCH]
HF_TOKEN is read from the environment (never printed)."""
import sys, os, json, time
sys.path.insert(0, "/workspace/rlcheck")
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from gemma_mt.eval.runtime import load_tokenizer_and_stops
from gemma_mt.eval.eval_translation import generate_translations

pool_p, out_p, repo, rev = sys.argv[1:5]
bs = int(sys.argv[5]) if len(sys.argv) > 5 else 96
tok_env = os.environ.get("HF_TOKEN")
path = snapshot_download(repo, revision=rev, token=tok_env,
                         allow_patterns=["config.json", "generation_config.json", "model.safetensors", "model-*.safetensors", "model.safetensors.index.json",
                                         "tokenizer*", "special_tokens_map.json", "added_tokens.json", "chat_template*"])
sha = os.path.basename(os.path.normpath(path))
tok, stop, pad = load_tokenizer_and_stops(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda").eval()
rows = [json.loads(l) for l in open(pool_p, encoding="utf-8")]
t0 = time.time()
res = generate_translations(model, tok, rows, stop_ids=stop.ids, pad_id=pad, device="cuda", batch_size=bs, max_new_tokens=320,
                            log=lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True))
with open(out_p, "w", encoding="utf-8") as f:
    for r, g in zip(rows, res):
        f.write(json.dumps({**r, **g, "repo": repo, "rev": rev, "sha": sha}, ensure_ascii=False) + "\n")
print("done", len(rows), f"{time.time()-t0:.0f}s", sha)
