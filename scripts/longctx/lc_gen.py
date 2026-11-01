"""vLLM generation for OUR models. venv_vllm.  usage: lc_gen.py NAME MODEL_DIR [--gpu-util 0.07]
Runs greedy whole-document (req_full), greedy sentence units (req_sent), then the repair pass (4 samples, T=0.7) for sentence units that did not finish.
Output OUT/NAME/{full,sent,fix}.jsonl.gz rows {rid, outs:[{ids, finish, stop}]}; finished sets are skipped (resumable)."""
import argparse, gzip, json, os, sys, time
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1"); os.environ.setdefault("DO_NOT_TRACK", "1")
ROOT = "/workspace/jobs/lc1"; STOP = [106, 1]
ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("model_dir")
ap.add_argument("--gpu-util", type=float, default=0.07); ap.add_argument("--max-seqs", type=int, default=256); ap.add_argument("--chunk", type=int, default=3000)
ap.add_argument("--max-model-len", type=int, default=10240); ap.add_argument("--only", default="")
a = ap.parse_args()
from vllm import LLM, SamplingParams
outdir = f"{ROOT}/out/{a.name}"; os.makedirs(outdir, exist_ok=True)
def rdj(p): return [json.loads(l) for l in open(p)]
def have(n): return os.path.exists(f"{outdir}/{n}.jsonl.gz")
want = [x for x in (a.only.split(",") if a.only else ["full", "sent", "fix"]) if not have(x)]
if not want:
    print("nothing to do"); sys.exit(0)
llm = LLM(model=a.model_dir, dtype="bfloat16", max_model_len=a.max_model_len, max_num_seqs=a.max_seqs, gpu_memory_utilization=a.gpu_util, enable_prefix_caching=True, seed=0,
          skip_tokenizer_init=True, generation_config="vllm", disable_log_stats=True)

def run(tag, reqs, make_params, n=1):
    reqs = sorted(reqs, key=lambda r: -(len(r["ids"]) + r["max_new"]))
    rows, ntok, t0 = [], 0, time.time()
    for s in range(0, len(reqs), a.chunk):
        part = reqs[s:s + a.chunk]
        outs = llm.generate([{"prompt_token_ids": r["ids"]} for r in part], [make_params(r) for r in part], use_tqdm=False)
        for r, o in zip(part, outs):
            oo = []
            for c in o.outputs:
                oo.append({"ids": [int(t) for t in c.token_ids], "finish": str(c.finish_reason), "stop": c.stop_reason if isinstance(c.stop_reason, int) else None}); ntok += len(c.token_ids)
            rows.append({"rid": r["rid"], "outs": oo})
        print(f"[{a.name}:{tag}] {min(s + a.chunk, len(reqs))}/{len(reqs)} reqs, {ntok} tok, {time.time() - t0:.0f}s", flush=True)
    p = f"{outdir}/{tag}.jsonl.gz"
    with gzip.open(p + ".tmp", "wt", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    os.replace(p + ".tmp", p)
    json.dump({"tag": tag, "n_requests": len(reqs), "generated_tokens": ntok, "wall_s": time.time() - t0, "model": a.name}, open(f"{outdir}/{tag}.stats.json", "w"))
    return rows

greedy = lambda r: SamplingParams(max_tokens=r["max_new"], temperature=0.0, stop_token_ids=STOP, detokenize=False, skip_special_tokens=False)
if "full" in want:
    run("full", rdj(f"{ROOT}/work/req_full.jsonl"), greedy)
if "sent" in want or "fix" in want:
    sent_reqs = rdj(f"{ROOT}/work/req_sent.jsonl")
    if "sent" in want:
        srows = run("sent", sent_reqs, greedy)
    else:
        srows = [json.loads(l) for l in gzip.open(f"{outdir}/sent.jsonl.gz", "rt")]
    if "fix" in want:
        bad = {r["rid"] for r in srows if not (r["outs"][0]["ids"] and r["outs"][0]["ids"][-1] in STOP)}
        fixreqs = [r for r in sent_reqs if r["rid"] in bad]
        print(f"repair: {len(fixreqs)} unfinished sentence units of {len(sent_reqs)}", flush=True)
        run("fix", fixreqs, lambda r: SamplingParams(max_tokens=r["max_new"], temperature=0.7, top_p=1.0, top_k=0, n=4, seed=abs(hash(r["rid"])) % 2**31,
                                                      stop_token_ids=STOP, detokenize=False, skip_special_tokens=False))
print("DONE", a.name, flush=True)
