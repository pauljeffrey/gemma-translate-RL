"""R12 RL part 4 (venv_gemma, GPU <= 3 GB): how far a truncated / tempered sampler is from the policy p the trainer differentiates.
For rollouts of config C (its own samples): per sequence sum_t [log q_C(y_t) - log p(y_t)] (q_C = the sampler's renormalised distribution; this is an unbiased
estimate of KL(q_C || p) per sequence because the samples come from q_C) and the number of tokens. For the plain T=1.0 rollouts (samples from p itself): which sequences contain a
token that a truncating sampler could never emit (outside its kept set) -> rl/res/bias_flags.jsonl joins with the reward. usage: rl_bias.py [max_prompts_per_bin]"""
import gzip, json, os, sys, time
sys.path.insert(0, "/workspace/sft_build/pylibs")
import torch
from gemma_mt.modeling import load_model
R = "/workspace/jobs/r12/rl"
P = {p["pid"]: p for p in json.load(open(R + "/work/prompts.json"))}
TR = {  # name: (T, top_p, top_k, min_p)
    "T1.0": (1.0, 1.0, 0, 0.0), "T1.0_p0.95": (1.0, 0.95, 0, 0.0), "T1.0_p0.9": (1.0, 0.9, 0, 0.0), "T1.0_p0.8": (1.0, 0.8, 0, 0.0), "T1.0_mp0.05": (1.0, 1.0, 0, 0.05), "T1.0_mp0.1": (1.0, 1.0, 0, 0.1),
    "T1.0_k64_p0.95": (1.0, 0.95, 64, 0.0), "T1.0_k20": (1.0, 1.0, 20, 0.0), "T0.8": (0.8, 1.0, 0, 0.0), "T0.7": (0.7, 1.0, 0, 0.0), "T0.5": (0.5, 1.0, 0, 0.0), "T0.3": (0.3, 1.0, 0, 0.0)}
model = load_model(("/workspace/sft_build/hf/hub/models--" + os.environ['HF_ORG_LEGACY'] + "--gemma-3-270m-translate-rl2c/snapshots/80baefff3b16b120b3e37e70afeb51f1fc248d96"), dtype=torch.bfloat16, device="cuda").eval()
def load(name):
    with gzip.open(f"{R}/gens/{name}.json.gz", "rt") as f: return json.load(f)["res"]
def kept_and_logq(l, T, p, k, mp):
    """l (L,V) float32 logits -> (mask kept (L,V) bool, lt log-softmax of tempered logits (L,V), logZ (L,) log mass of the kept set, lse_kept (L,))"""
    lt = l / T; ls = torch.log_softmax(lt, -1); pr = ls.exp(); mask = torch.ones_like(pr, dtype=torch.bool)
    if k > 0: mask &= lt >= lt.topk(k, -1).values[:, -1:]
    if mp > 0: mask &= pr >= mp * pr.max(-1, keepdim=True).values
    if p < 1.0:
        pm = torch.where(mask, pr, torch.zeros_like(pr)); pm = pm / pm.sum(-1, keepdim=True)
        sp, si = pm.sort(-1, descending=True); cs = sp.cumsum(-1); drop = (cs - sp) >= p       # keep the token that crosses p
        dm = torch.zeros_like(mask).scatter(-1, si, drop); mask &= ~dm
    logZ = torch.logsumexp(torch.where(mask, ls, torch.full_like(ls, -1e30)), -1)
    return mask, ls, logZ
@torch.no_grad()
def seq_stats(pid, ids, names):
    pr = P[pid]["ids"]; x = torch.tensor([pr + ids], device="cuda")
    logits = model(input_ids=x).logits[0, len(pr) - 1:-1].float()                   # positions predicting the completion tokens
    y = torch.tensor(ids, device="cuda"); lp = torch.log_softmax(logits, -1).gather(1, y[:, None])[:, 0]
    out = {"ntok": len(ids), "logp": float(lp.sum())}
    for nm in names:
        T, p, k, mp = TR[nm]; ks = []; kls = []
        for a in range(0, len(ids), 128):
            lg = logits[a:a + 128]; mask, ls, logZ = kept_and_logq(lg, T, p, k, mp); yy = y[a:a + 128, None]
            inm = mask.gather(1, yy)[:, 0]
            lq = ls - logZ[:, None]; lp0 = torch.log_softmax(lg, -1); q = torch.where(mask, lq.exp(), torch.zeros_like(lq))
            kl = (q * torch.where(mask, lq - lp0, torch.zeros_like(lq))).sum(-1)          # exact KL(q_t || p_t) at every position (nats)
            ks.append(inm); kls.append(kl)
        ks = torch.cat(ks); kl = torch.cat(kls)
        out[nm] = {"outside": int((~ks).sum()), "kl": float(kl.sum())}
    return out
t0 = time.time(); maxp = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
# (a) own samples of every config (4 samples per prompt, b64 + b256): KL(q||p) per sequence
res = {"own": {}, "t1": []}
for nm in TR:
    if not os.path.exists(f"{R}/gens/{nm}.json.gz"): continue
    G = [g for g in load(nm) if P[g["pid"]]["bin"] in ("b64", "b256")][:maxp]; rows = []
    for g in G:
        for o in g["outs"][:4]:
            if len(o["ids"]) == 0 or o["finish"] != "stop": continue
            s = seq_stats(g["pid"], o["ids"], [nm]); rows.append({"pid": g["pid"], "ntok": s["ntok"], "logp": s["logp"], "kl": s[nm]["kl"], "outside": s[nm]["outside"]})
    ok = rows
    res["own"][nm] = {"n": len(ok), "tokens": sum(r["ntok"] for r in ok), "kl_seq": sum(r["kl"] for r in ok) / max(len(ok), 1), "kl_tok": sum(r["kl"] for r in ok) / max(sum(r["ntok"] for r in ok), 1),
                    "mean_logp_seq": sum(r["logp"] for r in ok) / max(len(ok), 1), "mean_logp_tok": sum(r["logp"] for r in ok) / max(sum(r["ntok"] for r in ok), 1), "share_tokens_outside_kept_set": sum(r["outside"] for r in ok) / max(sum(r["ntok"] for r in ok), 1)}
    print(nm, res["own"][nm], f"{time.time()-t0:.0f}s", flush=True)
# (b) plain T=1 samples (from p): which contain tokens outside a truncating sampler's kept set
names = [n for n in TR if n != "T1.0"]
G = [g for g in load("T1.0") if P[g["pid"]]["bin"] in ("b64", "b256")][:maxp]
with open(R + "/res/bias_flags.jsonl", "w") as f:
    for g in G:
        for j, o in enumerate(g["outs"]):
            if len(o["ids"]) == 0: continue
            s = seq_stats(g["pid"], o["ids"], names); f.write(json.dumps({"pid": g["pid"], "j": j, "ntok": s["ntok"], "logp": s["logp"], "outside": {n: s[n]["outside"] for n in names}}) + "\n")
    print("flags done", f"{time.time()-t0:.0f}s", flush=True)
json.dump(res, open(R + "/res/bias_own.json", "w"))
