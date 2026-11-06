#!/usr/bin/env python
"""E-09: engine parity on the 240-sentence set of parity2.py (10 languages x 24 FLORES+ devtest sentences, greedy, 256 new tokens). Modes: hf (bf16), hf32, vllm (bf16), vllm32.
    python parity3.py MODE MODEL OUT.json"""
import os, sys, json
mode, model, out = sys.argv[1:4]
sys.path.insert(0, "/workspace/eval_385m/gemma_mt_tree")
os.environ.setdefault("GEMMA_MT_TAG_MODE", "single_token")
from transformers import AutoTokenizer
from gemma_mt import format as fmt
tok = AutoTokenizer.from_pretrained(model)
D = "/workspace/eval_385m/bench/test_sets/flores_plus/devtest/"
rd = lambda f: [json.loads(l) for l in open(D + f)]
en = rd("eng_Latn.jsonl")
LANGS = {"yor": "yor_Latn", "hau": "hau_Latn", "swh": "swh_Latn", "amh": "amh_Ethi", "zul": "zul_Latn", "ibo": "ibo_Latn", "kin": "kin_Latn", "som": "som_Latn", "xho": "xho_Latn", "wol": "wol_Latn"}
items = []
for iso, f in LANGS.items():
    t = rd(f + ".jsonl")
    for i in range(0, 24 * 11, 11):
        items.append((iso, en[i]["text"], t[i]["text"]))
prompts = []
for iso, s, r in items:
    ids = fmt.build_prompt(s, iso, tok)
    prompts.append(list(ids["input_ids"]) if isinstance(ids, dict) else list(ids))
gens = []
if mode in ("hf", "hf32"):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model, torch_dtype=(torch.float32 if mode == "hf32" else torch.bfloat16)).cuda().eval()
    B = 24
    for b in range(0, len(prompts), B):
        ch = prompts[b:b + B]; L = max(map(len, ch))
        x = torch.tensor([[0] * (L - len(p)) + p for p in ch]).cuda(); am = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in ch]).cuda()
        with torch.no_grad():
            g = m.generate(input_ids=x, attention_mask=am, max_new_tokens=256, do_sample=False, eos_token_id=[1, 106], pad_token_id=0)
        gens += [q[L:].tolist() for q in g]
else:
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, dtype=("float32" if mode == "vllm32" else "bfloat16"), gpu_memory_utilization=0.15, max_model_len=2048, enforce_eager=True)
    o = llm.generate([{"prompt_token_ids": p} for p in prompts], SamplingParams(temperature=0.0, max_tokens=256, stop_token_ids=[1, 106]))
    gens = [list(x.outputs[0].token_ids) for x in o]
def dec(g):
    for i, t in enumerate(g):
        if t in (1, 106, 0): g = g[:i]; break
    return tok.decode(g, skip_special_tokens=True)
txt = [dec(g) for g in gens]
json.dump({"items": items, "hyp": txt, "mode": mode, "model": model}, open(out, "w"), ensure_ascii=False)
print(mode, "done", len(txt))
