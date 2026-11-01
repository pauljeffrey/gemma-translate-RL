"""Vocabulary restriction study, steps 2 and 3b: greedy translation of the benchmark set with the output layer
  full   : the unmodified model (262,144 logits per step)
  mask   : full head, every logit outside the allowed set of the target language set to -inf (LogitsProcessor; saves NO compute)
  maskdyn: like mask, but the allowed set of every SENTENCE is the language set plus the tokens of its own English source words (with/without leading space, capitalised/lower-case variants): names are copied from the source.
           (in a real implementation this is a gather of ~66 extra rows per sentence on top of the K static rows; here it is evaluated as a mask, which gives exactly the same ids as the slice, see the mask == slice check)
  slice  : lm_head replaced by a Linear with only the allowed rows (K logits per step) and ids mapped back: the decode loop of HF `generate` runs in slice space (argmax index k = allowed[k]);
           the embedding of a 1-token decode step maps k -> allowed[k] (prefill sees real ids), eos/pad are given in slice space, and the generated ids are mapped back afterwards. This really saves compute.
usage: vocab_restrict_gen.py DEVICE(cpu|cuda) DTYPE(fp32|bf16) MODES(comma, run in this order per repetition) BATCH SETFILE(json: list of rows) ALLOWED_DIR OUTPREFIX [REPS=1] [N=0 (all rows; N>0 = fixed stratified subset)]
Rows are batched PER LANGUAGE (each language has its own allowed set), sorted by prompt length. Writes OUTPREFIX.<mode>.jsonl (rep 0 hypotheses) and appends timing records to OUTPREFIX.timing.jsonl."""
import sys, os, json, time, random, collections
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from gemma_mt import format as fmt
dev, dtype_s, modes, B, setfile, adir, outp = sys.argv[1], sys.argv[2], sys.argv[3].split(","), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7]
reps = int(sys.argv[8]) if len(sys.argv) > 8 else 1; N = int(sys.argv[9]) if len(sys.argv) > 9 else 0
MAXNEW = 400; M = "/dev/shm/lc1_models/final2"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
dt = torch.bfloat16 if dtype_s == "bf16" else torch.float32
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, dtype=dt).eval().to(dev)
rows = json.load(open(setfile))
if N:
    rng = random.Random(11); by = collections.defaultdict(list)
    for r in rows: by[r["iso"]].append(r)
    rows = []; per = N // len(by)
    for iso in sorted(by): l = by[iso][:]; rng.shuffle(l); rows += l[:per]
    rows += [r for iso in sorted(by) for r in by[iso] if r not in rows][: N - len(rows)] if len(rows) < N else []
allowed = {iso: torch.from_numpy(np.load(f"{adir}/allowed_{iso}.npy").astype(np.int64)) for iso in {r["iso"] for r in rows}}

class MapEmbed(torch.nn.Module):
    def __init__(self, emb, al): super().__init__(); self.emb = emb; self.register_buffer("al", al)
    def forward(self, ids): return self.emb(self.al[ids] if ids.shape[1] == 1 else ids)  # decode step (1 token) is in slice space, prefill has real ids
import re
class RowMaskProc(LogitsProcessor):
    def __init__(self, neg): self.neg = neg  # [B, V] bool
    def __call__(self, input_ids, scores): return scores.masked_fill(self.neg, float("-inf"))
def source_rows(text):
    words = set(re.findall(r"\w+", text)); ids = set(tok(text, add_special_tokens=False)["input_ids"])
    for w in words: ids |= set(tok(w, add_special_tokens=False)["input_ids"]) | set(tok(" " + w, add_special_tokens=False)["input_ids"]) | set(tok(w.capitalize(), add_special_tokens=False)["input_ids"]) | set(tok(" " + w.lower(), add_special_tokens=False)["input_ids"])
    return sorted(ids)
class MaskProc(LogitsProcessor):
    def __init__(self, ok): self.neg = ~ok
    def __call__(self, input_ids, scores): return scores.masked_fill(self.neg, float("-inf"))

full_emb, full_head = model.model.embed_tokens, model.lm_head
def setup(mode, iso):
    """returns (logits_processors, eos list, pad id, mapper from generated ids to real ids)"""
    model.model.embed_tokens = full_emb; model.lm_head = full_head
    if mode == "full": return None, [1, 106], 0, None
    al = allowed[iso].to(dev)
    if mode == "maskdyn":
        ok = torch.zeros(262144, dtype=torch.bool, device=dev); ok[al] = True
        return "dyn", [1, 106], 0, ok
    if mode == "mask":
        ok = torch.zeros(262144, dtype=torch.bool, device=dev); ok[al] = True
        return LogitsProcessorList([MaskProc(ok)]), [1, 106], 0, None
    head = torch.nn.Linear(640, len(al), bias=False).to(dev, dt); head.weight.data = full_head.weight.data[al].contiguous().clone()
    model.lm_head = head; model.model.embed_tokens = MapEmbed(full_emb, al)
    inv = {int(t): k for k, t in enumerate(al.tolist())}
    return None, [inv[1], inv[106]], inv[0], al.cpu()

def run(mode, collect):
    hyps = {}; tot = 0.0; ngen = 0; nbatch = 0
    for iso in sorted({r["iso"] for r in rows}):
        idx = sorted([i for i, r in enumerate(rows) if r["iso"] == iso], key=lambda i: len(rows[i]["ids"]))
        lp, eos, pad, back = setup(mode, iso)
        ok_static = None
        if lp == "dyn": ok_static, back = back, None
        for k in range(0, len(idx), B):
            b = idx[k:k + B]; L = max(len(rows[i]["ids"]) for i in b)
            ids = torch.tensor([[0] * (L - len(rows[i]["ids"])) + rows[i]["ids"] for i in b]).to(dev)
            att = torch.tensor([[0] * (L - len(rows[i]["ids"])) + [1] * len(rows[i]["ids"]) for i in b]).to(dev)
            if lp == "dyn":
                m = ok_static.unsqueeze(0).repeat(len(b), 1)
                for j, i in enumerate(b): m[j, source_rows(rows[i]["english"])] = True
                lp_b = LogitsProcessorList([RowMaskProc(~m)])
            else: lp_b = lp
            if dev == "cuda": torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.inference_mode():
                g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=MAXNEW, do_sample=False, eos_token_id=eos, pad_token_id=pad, logits_processor=lp_b, use_cache=True)
            if dev == "cuda": torch.cuda.synchronize()
            tot += time.perf_counter() - t; nbatch += 1
            for j, i in enumerate(b):
                o = g[j, L:].tolist(); n = len(o)
                for e in eos:
                    if e in o: n = min(n, o.index(e) + 1)
                o = o[:n]; fin = n < MAXNEW or (len(o) and o[-1] in eos)
                if back is not None: o = [int(back[x]) for x in o]
                hyps[i] = (o, bool(fin)); ngen += len(o)
    if collect:
        with open(f"{outp}.{mode}.jsonl", "w") as f:
            for i, r in enumerate(rows):
                o, fin = hyps[i]; raw = tok.decode(o, skip_special_tokens=True)
                f.write(json.dumps({"iso": r["iso"], "id": r.get("id", r.get("iid")), "hyp": fmt.strip_iso_prefix(raw, r["iso"]).text, "raw": raw, "ids": o, "gen_tokens": len(o), "finished": fin}, ensure_ascii=False) + "\n")
    return tot, ngen

# one warm-up pass on 2 sentences so allocator/threads are warm
for rep in range(reps):
    for mode in modes:
        t, n = run(mode, rep == 0)
        rec = {"device": dev, "dtype": dtype_s, "mode": mode, "rep": rep, "batch": B, "n": len(rows), "threads": torch.get_num_threads(), "total_s": round(t, 2), "gen_tokens": n, "tok_per_s": round(n / t, 2), "sent_per_s": round(len(rows) / t, 3),
               "load": open("/proc/loadavg").read().split()[0], "ts": time.strftime("%FT%T")}
        open(outp + ".timing.jsonl", "a").write(json.dumps(rec) + "\n"); print(json.dumps(rec), flush=True)
