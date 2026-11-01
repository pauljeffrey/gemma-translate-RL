"""R12 document-level generation (venv_vllm). usage: gen_doc.py cfg:reqset [cfg:reqset ...]   cfg = greedy | T<temp>[_p<top_p>][_k<top_k>][_mp<min_p>][_n<n>][_s<seed offset>]
reqset = file doc/work/req_<reqset>.jsonl. Output doc/out/rl2c/<cfg>__<reqset>.jsonl.gz (+ .stats.json); finished jobs are skipped."""
import gzip, json, os, re, sys, time, zlib
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1"); os.environ.setdefault("DO_NOT_TRACK", "1")
ROOT = "/workspace/jobs/r12/doc"; MODEL = ("/workspace/sft_build/hf/hub/models--" + os.environ['HF_ORG_LEGACY'] + "--gemma-3-270m-translate-rl2c/snapshots/80baefff3b16b120b3e37e70afeb51f1fc248d96")
def parse(name):
    if name == "greedy": return dict(temperature=0.0), 0
    kw = dict(temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, n=1); off = 0
    for i, part in enumerate(name.split("_")):
        if i == 0: kw["temperature"] = float(part[1:]); continue
        m = re.fullmatch(r"(mp|rp|fp|mt|p|k|n|s)([0-9.]+)", part); key, v = m.group(1), m.group(2)
        if key == "mp": kw["min_p"] = float(v)
        elif key == "p": kw["top_p"] = float(v)
        elif key == "k": kw["top_k"] = int(v)
        elif key == "n": kw["n"] = int(v)
        elif key == "s": off = int(v)
        elif key == "mt": kw["_mt"] = float(v)
        elif key == "rp": kw["repetition_penalty"] = float(v)
        elif key == "fp": kw["frequency_penalty"] = float(v)
    return kw, off
jobs = [j.split(":") for j in sys.argv[1:]]
outdir = f"{ROOT}/out/rl2c"; os.makedirs(outdir, exist_ok=True)
todo = [(c, r) for c, r in jobs if not os.path.exists(f"{outdir}/{c}__{r}.jsonl.gz")]
if not todo: print("nothing to do"); sys.exit(0)
from vllm import LLM, SamplingParams
t0 = time.time()
llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=10240, max_num_seqs=int(os.environ.get("MAX_SEQS", "96")), max_num_batched_tokens=int(os.environ.get("MAXBT", "4096")), enforce_eager=os.environ.get("EAGER", "1") == "1", gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.12")),
          enable_prefix_caching=True, seed=0, skip_tokenizer_init=True, generation_config="vllm", disable_log_stats=True)
print(f"engine up in {time.time()-t0:.0f}s", flush=True)
for cfg, rname in todo:
    reqs = [json.loads(l) for l in open(f"{ROOT}/work/req_{rname}.jsonl")]
    reqs.sort(key=lambda r: -(len(r["ids"]) + r["max_new"]))
    kw, off = parse(cfg); mt = kw.pop("_mt", 0.0); out_path = f"{outdir}/{cfg}__{rname}.jsonl.gz"
    ntok = 0; tstart = time.time(); rows = []; CH = 2500
    for s in range(0, len(reqs), CH):
        part = reqs[s:s + CH]
        params = [SamplingParams(max_tokens=r["max_new"], stop_token_ids=[106, 1], detokenize=False, skip_special_tokens=False, min_tokens=(min(int(mt * r["src_tokens"]), r["max_new"] - 1) if mt else 0),
                                 seed=((zlib.crc32(r["rid"].encode()) + 7919 * off) % 2**31) if kw.get("temperature", 0) > 0 else None, **kw) for r in part]
        t1 = time.time()
        outs = llm.generate([{"prompt_token_ids": r["ids"]} for r in part], params, use_tqdm=False)
        for r, o in zip(part, outs):
            oo = []
            for c in o.outputs:
                oo.append({"ids": [int(t) for t in c.token_ids], "finish": str(c.finish_reason), "stop": c.stop_reason if isinstance(c.stop_reason, int) else None}); ntok += len(c.token_ids)
            rows.append({"rid": r["rid"], "outs": oo})
        print(f"[{cfg}:{rname}] {min(s+CH, len(reqs))}/{len(reqs)} reqs, {ntok} tok, {time.time()-t1:.0f}s", flush=True)
    wall = time.time() - tstart
    with gzip.open(out_path + ".tmp", "wt", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    os.replace(out_path + ".tmp", out_path)
    json.dump({"config": cfg, "reqset": rname, "n_requests": len(reqs), "generated_tokens": ntok, "wall_s": wall, "tok_per_s": ntok / max(wall, 1e-9), "params": kw}, open(out_path.replace(".jsonl.gz", ".stats.json"), "w"))
    print(f"DONE {cfg}:{rname} {ntok} tok in {wall:.0f}s = {ntok/wall:.0f} tok/s", flush=True)
