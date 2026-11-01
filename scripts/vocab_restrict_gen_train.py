"""Vocabulary restriction study, step 1b: greedy generations of the FINAL model on English TRAIN-side prompts (RL prompt pool v3, FineWeb-edu English, never FLORES/NTREX),
used to build the allowed-token set of each language. Runs vLLM (bf16) on the GPU box. usage: vocab_restrict_gen_train.py OUTDIR LANGS(comma) [N_SHORT=2400] [N_LONG=600] [util=0.12]
Writes OUTDIR/gen_<iso>.npz with the generated token ids (flat) and their lengths. Prompt = the exact training prompt format (gemma_mt.format.build_prompt)."""
import sys, os, json, random
import numpy as np
from transformers import AutoTokenizer
from gemma_mt import format as fmt
out, langs = sys.argv[1], sys.argv[2].split(",")
nshort = int(sys.argv[3]) if len(sys.argv) > 3 else 2400; nlong = int(sys.argv[4]) if len(sys.argv) > 4 else 600; util = float(sys.argv[5]) if len(sys.argv) > 5 else 0.12
M = "/dev/shm/lc1_models/final2"; P = "/workspace/sft_build/gemma/rl_prompts_v3/%s.parquet"
os.makedirs(out, exist_ok=True)
SAMPLE_T = float(os.environ.get("SAMPLE_T", "0")); SAMPLE_N = int(os.environ.get("SAMPLE_N", "1")); PREFIX = "gens" if SAMPLE_T > 0 else "gen"  # SAMPLE_T>0: temperature sampling (component (e) of the set), other prompts (seed offset)
tok = AutoTokenizer.from_pretrained(M)
pool = json.load(open(os.environ.get("POOL_JSON", "/dev/shm/vr/pool.json")))  # {"b64": [...], "b256": [...]}, extracted from the parquet with pandas; English FineWeb-edu text; language independent. Each target language gets its own random draw.
from vllm import LLM, SamplingParams
llm = LLM(M, dtype="bfloat16", gpu_memory_utilization=util, max_model_len=2048, enforce_eager=True, seed=0)
sp = SamplingParams(temperature=SAMPLE_T, n=SAMPLE_N, seed=1, max_tokens=512, stop_token_ids=[1, 106], skip_special_tokens=False)
for iso in langs:
    if os.path.exists(f"{out}/{PREFIX}_{iso}.npz"): continue
    rng = random.Random(4242 + sum(map(ord, iso)) + (1000 if SAMPLE_T > 0 else 0))
    sh = pool["b64"]; lg = pool["b256"]
    texts = rng.sample(sh, nshort) + rng.sample(lg, nlong)
    prompts = [{"prompt_token_ids": fmt.build_prompt(t, iso, tok)} for t in texts]
    res = llm.generate(prompts, sp, use_tqdm=False)
    flat, lens, fin = [], [], []
    for r in res:
        for o in r.outputs: flat += list(o.token_ids); lens.append(len(o.token_ids)); fin.append(o.finish_reason == "stop")
    np.savez(f"{out}/{PREFIX}_{iso}.npz", ids=np.array(flat, dtype=np.int32), lens=np.array(lens), finished=np.array(fin))
    print(iso, len(texts), "gen tokens", len(flat), "finished", float(np.mean(fin)), flush=True)
