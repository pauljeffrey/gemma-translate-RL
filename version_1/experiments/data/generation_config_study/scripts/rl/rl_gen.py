"""R12 RL part 2 (venv_vllm): rollouts of the final model for the RL prompts under several sampling settings, group size 16 (as the trainer), max_new per bin from the 2c yaml.
usage: rl_gen.py name [name ...]   name = T1.0 | T1.0_p0.95 | ... | greedy (n=1). Output rl/gens/<name>.json.gz"""
import gzip, json, os, re, sys, time
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1"); os.environ.setdefault("DO_NOT_TRACK", "1")
sys.path.insert(0, "/workspace/jobs/r12")
R = "/workspace/jobs/r12/rl"; os.makedirs(R + "/gens", exist_ok=True)
MAXNEW = {"b64": 384, "b256": 1280, "b512": 2304, "b1024": 3584}
def parse(name):
    if name == "greedy": return dict(temperature=0.0, n=1)
    kw = dict(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, n=16)
    for i, part in enumerate(name.split("_")):
        if i == 0: kw["temperature"] = float(part[1:]); continue
        m = re.fullmatch(r"(mp|p|k|n)([0-9.]+)", part); key, v = m.group(1), m.group(2)
        if key == "mp": kw["min_p"] = float(v)
        elif key == "p": kw["top_p"] = float(v)
        elif key == "k": kw["top_k"] = int(v)
        elif key == "n": kw["n"] = int(v)
    return kw
names = [n for n in sys.argv[1:] if not os.path.exists(f"{R}/gens/{n}.json.gz")]
if not names: sys.exit(0)
from vllm import LLM, SamplingParams
P = json.load(open(R + "/work/prompts.json"))
llm = LLM(model=("/workspace/sft_build/hf/hub/models--" + os.environ['HF_ORG_LEGACY'] + "--gemma-3-270m-translate-rl2c/snapshots/80baefff3b16b120b3e37e70afeb51f1fc248d96"), dtype="bfloat16", max_model_len=5632, max_num_seqs=int(os.environ.get("MAX_SEQS", "96")), max_num_batched_tokens=int(os.environ.get("MAXBT", "4096")), enforce_eager=os.environ.get("EAGER", "1") == "1", gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.10")),
          skip_tokenizer_init=True, generation_config="vllm", enable_prefix_caching=True, seed=0, disable_log_stats=True)
for name in names:
    kw = parse(name); t0 = time.time()
    ps = [SamplingParams(max_tokens=MAXNEW[p["bin"]], stop_token_ids=[106, 1], detokenize=False, skip_special_tokens=False, seed=(1000 + i) if kw["temperature"] > 0 else None, **kw) for i, p in enumerate(P)]
    outs = llm.generate([{"prompt_token_ids": p["ids"]} for p in P], ps, use_tqdm=False)
    res = [{"pid": p["pid"], "outs": [{"ids": [int(t) for t in c.token_ids], "finish": str(c.finish_reason)} for c in o.outputs]} for p, o in zip(P, outs)]
    ntok = sum(len(c["ids"]) for r in res for c in r["outs"]); dt = time.time() - t0
    with gzip.open(f"{R}/gens/{name}.json.gz.tmp", "wt") as f: json.dump({"name": name, "kw": kw, "res": res, "seconds": dt, "tokens": ntok}, f)
    os.replace(f"{R}/gens/{name}.json.gz.tmp", f"{R}/gens/{name}.json.gz")
    print(f"{name}: {dt:.0f}s {ntok} tokens {ntok/dt:.0f} tok/s", flush=True)
