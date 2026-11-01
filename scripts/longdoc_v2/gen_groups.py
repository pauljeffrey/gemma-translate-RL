# SPDX-License-Identifier: AGPL-3.0-or-later
"""GRPO-style groups from the final policy on long documents (R8, diagnosis (d) and the greedy-in-group assessment). Run with /workspace/venv_vllm/bin/python.

  python gen_groups.py --items items_main.jsonl --out groups --bins 1024 2048 --n 8
For each (document, language) of the study items: n samples at T = 1 (top_p 1, top_k 0: the RL sampler) and the greedy completion, prompts and token budgets of R6
(a2/work/req_full_b.jsonl: max_new = min(3.5 x source tokens, 10240 - prompt)). Output groups/<bin>.jsonl.gz rows {rid, iso, doc_id, kind: s0..s{n-1} | greedy, ids, finish}."""
import argparse, gzip, json, os, sys, time
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1"); os.environ.setdefault("DO_NOT_TRACK", "1")
ap = argparse.ArgumentParser()
ap.add_argument("--items", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--bins", nargs="*", type=int, default=[1024, 2048])
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--model", default="/workspace/jobs/a2/models/rl2c")
ap.add_argument("--req", default="/workspace/jobs/a2/work/req_full_b.jsonl")
ap.add_argument("--gpu-util", type=float, default=0.15)
ap.add_argument("--max-seqs", type=int, default=256)
a = ap.parse_args()
want = {}
for l in open(a.items):
    r = json.loads(l)
    if r["cond"] == "sf" and r["bin"] in a.bins:
        want[f"core|{r['doc_id']}|{r['iso']}"] = r
reqs = [json.loads(l) for l in open(a.req)]
reqs = [r for r in reqs if r["rid"] in want]
print(len(reqs), "requests of", len(want), flush=True)
from vllm import LLM, SamplingParams
llm = LLM(model=a.model, dtype="bfloat16", max_model_len=10240, max_num_seqs=a.max_seqs, gpu_memory_utilization=a.gpu_util, enable_prefix_caching=True, seed=0,
          skip_tokenizer_init=True, generation_config="vllm", disable_log_stats=True)
os.makedirs(a.out, exist_ok=True)
for b in a.bins:
    rb = sorted([r for r in reqs if want[r["rid"]]["bin"] == b], key=lambda r: -(len(r["ids"]) + r["max_new"]))
    path = f"{a.out}/{b}.jsonl.gz"
    if os.path.exists(path) or not rb:
        continue
    n = a.n if b < 4096 else max(2, a.n // 2)
    t0 = time.time()
    rows = []
    CH = 96
    for s in range(0, len(rb), CH):
        part = rb[s: s + CH]
        sp = [SamplingParams(max_tokens=r["max_new"], stop_token_ids=[106, 1], detokenize=False, skip_special_tokens=False, temperature=1.0, top_p=1.0, top_k=0, n=n,
                             seed=abs(hash(r["rid"])) % 2 ** 31) for r in part]
        sg = [SamplingParams(max_tokens=r["max_new"], stop_token_ids=[106, 1], detokenize=False, skip_special_tokens=False, temperature=0.0) for r in part]
        prompts = [{"prompt_token_ids": r["ids"]} for r in part]
        o1 = llm.generate(prompts, sp, use_tqdm=False)
        o2 = llm.generate(prompts, sg, use_tqdm=False)
        for r, x, g in zip(part, o1, o2):
            meta = want[r["rid"]]
            for k, c in enumerate(x.outputs):
                rows.append({"rid": r["rid"], "iso": meta["iso"], "doc_id": meta["doc_id"], "bin": b, "kind": f"s{k}", "ids": [int(t) for t in c.token_ids], "finish": str(c.finish_reason)})
            c = g.outputs[0]
            rows.append({"rid": r["rid"], "iso": meta["iso"], "doc_id": meta["doc_id"], "bin": b, "kind": "greedy", "ids": [int(t) for t in c.token_ids], "finish": str(c.finish_reason)})
        print(f"[b{b}] {min(s + CH, len(rb))}/{len(rb)} prompts, {sum(len(r['ids']) for r in rows)} tokens, {time.time() - t0:.0f}s", flush=True)
    with gzip.open(path + ".tmp", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)
    print(f"DONE b{b}: {len(rows)} completions in {time.time() - t0:.0f}s", flush=True)
