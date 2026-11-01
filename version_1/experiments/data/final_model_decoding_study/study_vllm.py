"""a1 decoding study, step 2a (venv_vllm): vLLM configs. Writes study/gen_<cfg>.json = {cands: [[ids]*n per row], reasons, seconds, tokens}."""
import json, os, sys, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0"); os.environ.setdefault("VLLM_NO_USAGE_STATS", "1"); os.environ.setdefault("DO_NOT_TRACK", "1")
from vllm import LLM, SamplingParams
D = "/workspace/jobs/a1/study"
S = json.load(open(D + "/study_set.json"))
rows, stop = S["rows"], S["stop_ids"]
MAXNEW = 320
CFGS = {
    "greedy": dict(temperature=0.0),
    "rep1.05": dict(temperature=0.0, repetition_penalty=1.05),
    "rep1.1": dict(temperature=0.0, repetition_penalty=1.1),
    "T0.3": dict(temperature=0.3, top_p=1.0),
    "T0.5": dict(temperature=0.5, top_p=1.0),
    "T0.7": dict(temperature=0.7, top_p=1.0),
    "s8T0.7": dict(temperature=0.7, top_p=1.0, n=8),
    "s8T0.5": dict(temperature=0.5, top_p=1.0, n=8),
}
only = sys.argv[1:] or list(CFGS)
llm = LLM(model=S["model_path"], dtype="bfloat16", max_model_len=2048, max_num_seqs=256, gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.08")),
          skip_tokenizer_init=True, generation_config="vllm", enable_prefix_caching=False, seed=0, disable_log_stats=True)
prompts = [{"prompt_token_ids": r["prompt"]} for r in rows]
def run(name, kw, ps=prompts, tag=None):
    p = [SamplingParams(max_tokens=MAXNEW, stop_token_ids=stop, detokenize=False, skip_special_tokens=False, seed=(1234 + i) if kw.get("temperature", 0) > 0 else None, **kw)
         for i in range(len(ps))]
    t0 = time.time(); outs = llm.generate(ps, p, use_tqdm=False); dt = time.time() - t0
    cands = [[[int(t) for t in c.token_ids] for c in o.outputs] for o in outs]
    reasons = [[str(c.finish_reason) for c in o.outputs] for o in outs]
    ntok = sum(len(c) for cs in cands for c in cs)
    json.dump({"cfg": name, "kw": kw, "cands": cands, "reasons": reasons, "seconds": dt, "tokens": ntok, "n_prompts": len(ps), "engine": "vllm"},
              open(f"{D}/gen_{tag or name}.json", "w"))
    print(f"{name}: {dt:.1f}s {ntok} tokens {ntok/dt:.0f} tok/s", flush=True)
if not os.path.exists(D + "/gen_warmup.done"):
    run("warm", dict(temperature=0.0), prompts[:256], tag="warm"); os.remove(D + "/gen_warm.json"); open(D + "/gen_warmup.done", "w").write("1")
for name in only:
    if os.path.exists(f"{D}/gen_{name}.json"): print("skip", name); continue
    run(name, CFGS[name])
