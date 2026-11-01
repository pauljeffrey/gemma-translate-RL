"""R12 generation-config study, sentence level, vLLM. Config names: T<temp>[_p<top_p>][_k<top_k>][_mp<min_p>][_n<n>][_s<seed offset>]  e.g. T1.0_p0.9, T0.6_mp0.05_n8, greedy.
usage: gen_sent.py name [name ...]   (skips finished ones; output gens/gen_<name>.json.gz = {cands, reasons, seconds, tokens})"""
import gzip, json, os, re, sys, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0"); os.environ.setdefault("VLLM_NO_USAGE_STATS", "1"); os.environ.setdefault("DO_NOT_TRACK", "1")
from vllm import LLM, SamplingParams
R = "/workspace/jobs/r12"
S = json.load(open(R + "/study_set.json"))
rows, stop = S["rows"], S["stop_ids"]
MAXNEW = 320
def parse(name):
    if name == "greedy": return dict(temperature=0.0), 0
    kw = dict(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, n=1); off = 0
    for i, part in enumerate(name.split("_")):
        if i == 0: kw["temperature"] = float(part[1:]); continue
        m = re.fullmatch(r"(mp|p|k|n|s)([0-9.]+)", part)
        key, v = m.group(1), m.group(2)
        if key == "mp": kw["min_p"] = float(v)
        elif key == "p": kw["top_p"] = float(v)
        elif key == "k": kw["top_k"] = int(v)
        elif key == "n": kw["n"] = int(v)
        elif key == "s": off = int(v)
    return kw, off
names = sys.argv[1:]
os.makedirs(R + "/gens", exist_ok=True)
PFX = os.environ.get("NAME_PREFIX", "")
todo = [n for n in names if not os.path.exists(f"{R}/gens/gen_{PFX}{n}.json.gz")]
if not todo: print("nothing to do"); sys.exit(0)
PFX = os.environ.get("NAME_PREFIX", "")
llm = LLM(model=os.environ.get("MODEL_PATH", S["model_path"]), dtype="bfloat16", max_model_len=2048, max_num_seqs=int(os.environ.get("MAX_SEQS", "256")), max_num_batched_tokens=int(os.environ.get("MAXBT", "8192")), enforce_eager=os.environ.get("EAGER", "0") == "1", gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.12")),
          skip_tokenizer_init=True, generation_config="vllm", enable_prefix_caching=False, seed=0, disable_log_stats=True)
prompts = [{"prompt_token_ids": r["prompt"]} for r in rows]
if not os.path.exists(R + "/warm.done"):
    llm.generate(prompts[:256], SamplingParams(max_tokens=MAXNEW, temperature=0.0, stop_token_ids=stop, detokenize=False), use_tqdm=False); open(R + "/warm.done", "w").write("1")
for name in todo:
    kw, off = parse(name)
    ps = [SamplingParams(max_tokens=MAXNEW, stop_token_ids=stop, detokenize=False, skip_special_tokens=False, seed=(1234 + i + 100003 * off) if kw.get("temperature", 0) > 0 else None, **kw) for i in range(len(rows))]
    t0 = time.time(); outs = llm.generate(prompts, ps, use_tqdm=False); dt = time.time() - t0
    cands = [[[int(t) for t in c.token_ids] for c in o.outputs] for o in outs]
    reasons = [[str(c.finish_reason) for c in o.outputs] for o in outs]
    ntok = sum(len(c) for cs in cands for c in cs)
    with gzip.open(f"{R}/gens/gen_{PFX}{name}.json.gz.tmp", "wt") as f:
        json.dump({"cfg": PFX + name, "kw": kw, "seed_off": off, "cands": cands, "reasons": reasons, "seconds": dt, "tokens": ntok, "n_prompts": len(ps), "engine": "vllm"}, f)
    os.replace(f"{R}/gens/gen_{PFX}{name}.json.gz.tmp", f"{R}/gens/gen_{PFX}{name}.json.gz")
    print(f"{name}: {dt:.1f}s {ntok} tokens {ntok/dt:.0f} tok/s", flush=True)
