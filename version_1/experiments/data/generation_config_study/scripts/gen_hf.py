"""R12: HF generate sampling configs that vLLM 0.31 does not have: typical_p, epsilon_cutoff, eta_cutoff. venv_gemma. usage: gen_hf.py name ... (names in CFGS)."""
import gzip, json, os, sys, time
sys.path.insert(0, "/workspace/sft_build/pylibs")
import torch
from gemma_mt.modeling import load_model
R = "/workspace/jobs/r12"
S = json.load(open(R + "/study_set.json"))
rows, stop, pad = S["rows"], S["stop_ids"], S["pad_id"]
MAXNEW = 320
base = dict(do_sample=True, num_beams=1, temperature=1.0, top_k=0, top_p=1.0)
CFGS = {
    "hf_T1.0": dict(),
    "hf_T1.0_typ0.8": dict(typical_p=0.8), "hf_T1.0_typ0.9": dict(typical_p=0.9), "hf_T1.0_typ0.95": dict(typical_p=0.95),
    "hf_T1.0_eps0.0003": dict(epsilon_cutoff=0.0003), "hf_T1.0_eps0.001": dict(epsilon_cutoff=0.001), "hf_T1.0_eps0.003": dict(epsilon_cutoff=0.003),
    "hf_T1.0_eps0.01": dict(epsilon_cutoff=0.01), "hf_T1.0_eps0.02": dict(epsilon_cutoff=0.02),
    "hf_T1.0_eta0.0003": dict(eta_cutoff=0.0003), "hf_T1.0_eta0.001": dict(eta_cutoff=0.001), "hf_T1.0_eta0.003": dict(eta_cutoff=0.003), "hf_T1.0_eta0.01": dict(eta_cutoff=0.01),
    "hf_T0.7_typ0.9": dict(temperature=0.7, typical_p=0.9), "hf_T0.7_eps0.001": dict(temperature=0.7, epsilon_cutoff=0.001),
    "hf_T0.5_typ0.9": dict(temperature=0.5, typical_p=0.9),
}
names = [n for n in (sys.argv[1:] or list(CFGS)) if not os.path.exists(f"{R}/gens/gen_{n}.json.gz")]
if not names: sys.exit(0)
model = load_model(S["model_path"], dtype=torch.bfloat16, device="cuda").eval()
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"])); BS = 160
for name in names:
    kw = {**base, **CFGS[name]}
    res = [None] * len(rows); torch.cuda.synchronize(); t0 = time.time(); ntok = 0
    for b, s in enumerate(range(0, len(order), BS)):
        idx = order[s:s + BS]; w = max(len(rows[i]["prompt"]) for i in idx)
        ids = torch.full((len(idx), w), pad, dtype=torch.long); att = torch.zeros_like(ids)
        for j, i in enumerate(idx):
            p = rows[i]["prompt"]; ids[j, w - len(p):] = torch.tensor(p); att[j, w - len(p):] = 1
        torch.manual_seed(1234 + b)
        with torch.no_grad():
            g = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=MAXNEW, eos_token_id=list(stop), pad_token_id=pad, use_cache=True, **kw)
        for j, i in enumerate(idx):
            out = g[j, w:].tolist(); k = len(out)
            for q, t in enumerate(out):
                if t in stop: k = q + 1; break
            res[i] = out[:k]; ntok += k
    torch.cuda.synchronize(); dt = time.time() - t0
    with gzip.open(f"{R}/gens/gen_{name}.json.gz.tmp", "wt") as f:
        json.dump({"cfg": name, "kw": {k: v for k, v in kw.items()}, "cands": [[r] for r in res], "reasons": [["stop" if r[-1] in stop else "length"] for r in res],
                   "seconds": dt, "tokens": ntok, "n_prompts": len(rows), "engine": "hf", "batch_size": BS}, f)
    os.replace(f"{R}/gens/gen_{name}.json.gz.tmp", f"{R}/gens/gen_{name}.json.gz")
    print(f"{name}: {dt:.1f}s {ntok} tokens {ntok/dt:.0f} tok/s", flush=True)
