"""a1 decoding study, step 2b (venv_gemma): HF generate configs (greedy cross-check, no_repeat_ngram, beam, output-only repetition penalty)."""
import json, os, sys, time
sys.path.insert(0, "/workspace/sft_build/pylibs")
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from gemma_mt.modeling import load_model
D = "/workspace/jobs/a1/study"
S = json.load(open(D + "/study_set.json"))
rows, stop, pad = S["rows"], S["stop_ids"], S["pad_id"]
MAXNEW = 320
class OutRepPenalty(LogitsProcessor):
    """repetition penalty over the GENERATED tokens only (HF's own class also penalises the prompt, i.e. the English source)."""
    def __init__(self, penalty, plen): self.p, self.plen = penalty, plen
    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.plen:]
        if gen.shape[1] == 0: return scores
        s = torch.gather(scores, 1, gen)
        s = torch.where(s < 0, s * self.p, s / self.p)
        return scores.scatter(1, gen, s)
CFGS = {  # name: (generate kwargs, batch size, out-only penalty)
    "hf_greedy": (dict(do_sample=False, num_beams=1), 160, None),
    "hf_rep1.1": (dict(do_sample=False, num_beams=1, repetition_penalty=1.1), 160, None),
    "rep1.05out": (dict(do_sample=False, num_beams=1), 160, 1.05),
    "rep1.1out": (dict(do_sample=False, num_beams=1), 160, 1.1),
    "nr3": (dict(do_sample=False, num_beams=1, no_repeat_ngram_size=3), 160, None),
    "nr4": (dict(do_sample=False, num_beams=1, no_repeat_ngram_size=4), 160, None),
    "beam2": (dict(do_sample=False, num_beams=2), 80, None),
    "beam4": (dict(do_sample=False, num_beams=4), 40, None),
    "beam8": (dict(do_sample=False, num_beams=8), 20, None),
}
only = sys.argv[1:] or [c for c in CFGS if c != "beam8"]
model = load_model(S["model_path"], dtype=torch.bfloat16, device="cuda").eval()
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prompt"]))
def run(name):
    kw, bs, outpen = CFGS[name]
    res = [None] * len(rows); torch.cuda.synchronize(); t0 = time.time(); ntok = 0
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; w = max(len(rows[i]["prompt"]) for i in idx)
        ids = torch.full((len(idx), w), pad, dtype=torch.long); att = torch.zeros_like(ids)
        for j, i in enumerate(idx):
            p = rows[i]["prompt"]; ids[j, w - len(p):] = torch.tensor(p); att[j, w - len(p):] = 1
        extra = {}
        if outpen: extra["logits_processor"] = LogitsProcessorList([OutRepPenalty(outpen, w)])
        with torch.no_grad():
            g = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=MAXNEW, eos_token_id=list(stop), pad_token_id=pad, use_cache=True,
                               temperature=None, top_p=None, top_k=None, **kw, **extra)
        for j, i in enumerate(idx):
            out = g[j, w:].tolist(); k = len(out)
            for q, t in enumerate(out):
                if t in stop: k = q + 1; break
            res[i] = out[:k]; ntok += k
    torch.cuda.synchronize(); dt = time.time() - t0
    json.dump({"cfg": name, "kw": {**kw, "out_penalty": outpen}, "cands": [[r] for r in res], "reasons": [["stop" if r[-1] in stop else "length"] for r in res],
               "seconds": dt, "tokens": ntok, "n_prompts": len(rows), "engine": "hf", "batch_size": bs}, open(f"{D}/gen_{name}.json", "w"))
    print(f"{name}: {dt:.1f}s {ntok} tokens {ntok/dt:.0f} tok/s", flush=True)
for n in only:
    if os.path.exists(f"{D}/gen_{n}.json"): print("skip", n); continue
    run(n)
