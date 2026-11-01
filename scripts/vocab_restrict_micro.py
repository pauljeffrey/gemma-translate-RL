"""Vocabulary restriction study, step 3a: micro-benchmark of ONE decode step (batch B, KV length ~ L) in PyTorch CPU: transformer body vs output head, full 262,144 rows vs sliced heads.
usage (run under `taskset -c <4 cores> nice`): vocab_restrict_micro.py DTYPE(fp32|bf16) BATCH OUT.csv [reps=3] [steps=48]
Per repetition: the body (model.model, 1 new token, KV cache of the prompt) and the head (hidden[B,640] @ W[K,640]^T + argmax) are timed separately in the SAME loop (so the ratio is not affected by drift);
the heads use real weights (rows of the tied embedding, the first K ids of the allowed set of Swahili padded with the most frequent ids, only the SIZE matters for the speed)."""
import sys, time, json, csv, statistics, os, torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
dtype_s, B, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]; reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3; steps = int(sys.argv[5]) if len(sys.argv) > 5 else 48
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
dt = torch.bfloat16 if dtype_s == "bf16" else torch.float32
M = "/dev/shm/lc1_models/final2"
model = AutoModelForCausalLM.from_pretrained(M, dtype=dt).eval()
W = model.lm_head.weight.data  # [262144, 640] (tied)
Q = json.load(open("/workspace/jobs/c1b/out/Q.json"))
ids = Q[0]["ids"]; ids = (ids * 3)[:64]
Ks = [262144, 80000, 40000, 20000]
heads = {}
for K in Ks:
    idx = torch.arange(K) if K == 262144 else torch.randperm(262144, generator=torch.Generator().manual_seed(1))[:K].sort().values
    heads[K] = W[idx].contiguous().clone()
rows = []
with torch.inference_mode():
    for rep in range(reps):
        inp = torch.tensor([ids] * B)
        cache = DynamicCache()
        out = model.model(input_ids=inp, past_key_values=cache, use_cache=True)
        h = out.last_hidden_state[:, -1]
        tok = torch.full((B, 1), ids[-1])
        tb = 0.0; th = {K: 0.0 for K in Ks}; tfull = 0.0; n = 0
        for s in range(steps + 4):
            t0 = time.perf_counter(); o = model.model(input_ids=tok, past_key_values=cache, use_cache=True); hh = o.last_hidden_state[:, -1]; t1 = time.perf_counter()
            for K in Ks:
                a = time.perf_counter(); lg = hh @ heads[K].T; nxt = lg.argmax(-1); b = time.perf_counter()
                if s >= 4: th[K] += b - a
            if s >= 4: tb += t1 - t0; n += 1
            tok = nxt[:, None] % 1000 + 10
        for K in Ks:
            rows.append({"dtype": dtype_s, "batch": B, "rep": rep, "K": K, "body_ms": round(1000 * tb / n, 3), "head_ms": round(1000 * th[K] / n, 3), "step_ms": round(1000 * (tb + th[K]) / n, 3), "head_share": round(th[K] / (tb + th[K]), 4), "load": open("/proc/loadavg").read().split()[0]})
        print(rep, [(r["K"], r["body_ms"], r["head_ms"]) for r in rows[-4:]], flush=True)
new = not os.path.exists(outp)
with open(outp, "a", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys()))
    if new: w.writeheader()
    w.writerows(rows)
