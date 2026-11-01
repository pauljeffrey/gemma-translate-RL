"""Mixture v2, component B step 2: vLLM generation of the sentence-level requests (adapted from /workspace/jobs/a2/gen_vllm.py, same sampling settings).

    /workspace/venv_vllm/bin/python mix2_gen_vllm.py --req work/ld/req_sent.jsonl --out work/ld/gen --jobs greedy,t07x3 [--gpu-util 0.17]

Jobs: greedy = temperature 0; t07x3 = temperature 0.7, top_p 1, top_k 0, n = 3 (per-request seed from md5(rid): reproducible).
Output OUT/<job>.jsonl.gz rows {rid, outs: [{ids, finish}]}; finished jobs are skipped (restartable). Stop ids 106 / 1 as in the RL run.
"""
import argparse, gzip, hashlib, json, os, sys, time

os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
CONFIGS = {"greedy": dict(temperature=0.0), "t07x3": dict(temperature=0.7, top_p=1.0, top_k=0, n=3)}

ap = argparse.ArgumentParser()
ap.add_argument("--req", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="/workspace/sft_build/runs/final_1433")
ap.add_argument("--jobs", default="greedy,t07x3")
ap.add_argument("--gpu-util", type=float, default=0.17)
ap.add_argument("--max-seqs", type=int, default=384)
ap.add_argument("--chunk", type=int, default=20000)
ap.add_argument("--max-model-len", type=int, default=4096)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
todo = [j for j in a.jobs.split(",") if not os.path.exists(f"{a.out}/{j}.jsonl.gz")]
if not todo:
    print("nothing to do")
    sys.exit(0)
from vllm import LLM, SamplingParams

reqs = [json.loads(l) for l in open(a.req, encoding="utf-8")]
reqs.sort(key=lambda r: -(len(r["ids"]) + r["max_new"]))
t0 = time.time()
llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, max_num_seqs=a.max_seqs, gpu_memory_utilization=a.gpu_util,
          enable_prefix_caching=True, seed=0, skip_tokenizer_init=True, generation_config="vllm", disable_log_stats=True)
print(f"engine up in {time.time() - t0:.0f}s", flush=True)
for job in todo:
    kw = CONFIGS[job]
    ntok, rows, tstart = 0, [], time.time()
    for s in range(0, len(reqs), a.chunk):
        part = reqs[s:s + a.chunk]
        params = [SamplingParams(max_tokens=r["max_new"], stop_token_ids=[106, 1], detokenize=False, skip_special_tokens=False,
                                 seed=(int(hashlib.md5(r["rid"].encode()).hexdigest()[:8], 16) % 2**31) if kw.get("temperature", 0) > 0 else None, **kw) for r in part]
        t1 = time.time()
        outs = llm.generate([{"prompt_token_ids": r["ids"]} for r in part], params, use_tqdm=False)
        for r, o in zip(part, outs):
            oo = []
            for c in o.outputs:
                oo.append({"ids": [int(t) for t in c.token_ids], "finish": str(c.finish_reason)})
                ntok += len(c.token_ids)
            rows.append({"rid": r["rid"], "outs": oo})
        print(f"[{job}] {min(s + a.chunk, len(reqs))}/{len(reqs)} reqs, {ntok} tok, {time.time() - t1:.0f}s this chunk", flush=True)
    wall = time.time() - tstart
    with gzip.open(f"{a.out}/{job}.jsonl.gz.tmp", "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(f"{a.out}/{job}.jsonl.gz.tmp", f"{a.out}/{job}.jsonl.gz")
    json.dump({"job": job, "n_requests": len(reqs), "generated_tokens": ntok, "wall_s": wall, "tok_per_s": ntok / max(wall, 1e-9), "params": kw, "model": a.model},
              open(f"{a.out}/{job}.stats.json", "w"))
    print(f"DONE {job}: {ntok} tok in {wall:.0f}s = {ntok / wall:.0f} tok/s", flush=True)
