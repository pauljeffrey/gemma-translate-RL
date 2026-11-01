"""T1: run <HF_ORG_LEGACY>/modernbert-large-triage-v7-pangolin (triage_v8.Triage, local copy in /dev/shm/t1_pang) on the unique English chunk texts.
Stores: official outputs (labels via classify(): integer ratings, weighted-mean score, band, english_form_dependent, pipeline, priority) AND the raw continuous
regression outputs (clipped to 1..10) and probabilities. Usage: t1_classify.py IN.jsonl OUT.jsonl"""
import sys, json, os, time
sys.path.insert(0, "/dev/shm/t1_pang")
import numpy as np, torch
torch.cuda.set_per_process_memory_fraction(4.0 / 95.0)
from triage_v8 import Triage
import triage_v7 as TV
inp, outp = sys.argv[1:3]; MAXLEN = int(sys.argv[3]) if len(sys.argv) > 3 else 1536
m = Triage.load("/dev/shm/t1_pang", device="cuda", max_len=MAXLEN)
cache = {}
orig = m.probabilities
def wrapped(items, token_budget=24000):
    P = orig(items, token_budget); cache["P"] = P; return P
m.probabilities = wrapped
rows = [json.loads(l) for l in open(inp)]
done = set()
if os.path.exists(outp):
    for l in open(outp): done.add(json.loads(l)["ekey"])
rows = [r for r in rows if r["ekey"] not in done]
print("to do", len(rows), flush=True)
sc = m.H["regression"]["scale"]
tok = m.tok
t0 = time.time()
BS = 1000
with open(outp, "a") as f:
    for s in range(0, len(rows), BS):
        chunk = rows[s:s + BS]
        items = [{"text": r["text"]} for r in chunk]
        out = m.classify(items, token_budget=12000, with_probabilities=True)
        P = cache["P"]
        ntok = [len(tok(r["text"])["input_ids"]) for r in chunk]
        for i, (r, o) in enumerate(zip(chunk, out)):
            lab = o["_labels_raw"]
            rec = {"ekey": r["ekey"], "n_tok": ntok[i], "n_chars": len(r["text"]), "n_words": len(r["text"].split()),
                   "score_model": o["translation_difficulty_score"], "band_model": o["translation_difficulty"],
                   "efi": bool(o["english_form_dependent"]), "efi_prob": float(P["bool"]["english_form_dependent"][i]),
                   "pipeline": o["recommended_pipeline"], "priority": o["translation_priority"], "tech_depth": lab["technical_depth"], "domain": lab["domain"],
                   "text_type": o.get("text_type"), "doc_quality": o.get("document_quality"), "p_tool": o["probability_needs_tool_or_retrieval"], "hardest": o["hardest_translation_aspects"]}
            for k, d in enumerate(TV.DIMS):
                rec[d + "_int"] = int(lab[d]); rec[d + "_f"] = float(np.clip(P["reg"][i][k], 1, 10))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        print(s + len(chunk), len(rows), round(time.time() - t0), "s  peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
print("done")
