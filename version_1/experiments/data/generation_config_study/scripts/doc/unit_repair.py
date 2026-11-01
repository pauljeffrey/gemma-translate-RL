"""R12 (venv_gemma, in r12/): unit-level evaluation of the repair of unfinished sentence units (greedy output hit the token limit / looped).
For every repair config (<cfg>__sentfail8 in doc/out/rl2c, n candidates) and selection rule: finished share, loop share, chrF++ against the NLLB reference sentence,
length ratio. R6's own repair (t07x4 over all 3,250 units, restricted to the B8 units) is the baseline. Output res/unit_repair.json."""
import glob, gzip, json, os, sys
import numpy as np
sys.path.insert(0, "/workspace/jobs/r12/doc"); sys.path.insert(0, "/workspace/sft_build/pylibs")
import assemble as A
from common import *
from sacrebleu.metrics import CHRF
from gemma_mt.rl import guards
from gemma_mt.eval.longdoc import split_long_sentence
B8 = set("hau yor swh amh zul wol kea sag".split())
REQ = {r["rid"]: r for r in rd(MYWORK + "/req_sentfail8.jsonl")}
# NLLB reference sentence of each unit (only units that are whole sentences of core items)
ref_of = {}
refs = {(r["doc_id"], r["iso"]): r["ref_sents"] for r in rd(WORK + "/refs_nllb.jsonl") if r["iso"] in B8}
for iid, it in A.items.items():
    if it["set"] != "core" or it["iso"] not in B8: continue
    rs = refs.get((it["doc_id"], it["iso"]))
    if not rs or len(rs) != len(it["sents"]): continue
    for s, r in zip(it["sents"], rs):
        if len(s) <= 700: ref_of[(it["iso"], s)] = r
chrf = CHRF(word_order=2)
def evaluate(chosen):
    """chosen: rid -> dec dict"""
    hy, rf, fin, loop, lr = [], [], [], [], []
    for rid, d in chosen.items():
        q = REQ[rid]; r = ref_of.get((q["iso"], q["text"]))
        fin.append(d["finished"]); t = norm_text(d["text"]) if False else d["text"]
        loop.append((not d["finished"]) or guards.is_repetitive(t) or guards.is_self_repetitive(t))
        if r is not None: hy.append(t); rf.append(r); lr.append(len(t) / max(len(r), 1))
    return {"n": len(chosen), "finished": float(np.mean(fin)), "loop_or_unfinished": float(np.mean(loop)), "n_ref": len(hy), "chrf_vs_nllb": float(chrf.corpus_score(hy, [rf]).score) if hy else None, "len_ratio": float(np.mean(lr)) if lr else None}
iso_of = {rid: q["iso"] for rid, q in REQ.items()}
out = {}
greedy = {r["rid"]: r["outs"][0] for r in A.read_out("rl2c", "greedy", "sent") if r["rid"] in REQ}
out["greedy (unrepaired)"] = evaluate({rid: A.dec(iso_of[rid], o) for rid, o in greedy.items()})
def pick(outs, iso, rule, n):
    ds = [A.dec(iso, o) for o in outs[:n]]; fin = [d for d in ds if d["finished"]]
    if not fin: return ds[0]            # still unfinished: keep the first candidate (the greedy output is the fallback in R6; here too)
    if rule == "first" or len(fin) == 1: return fin[0]
    b, _ = A.consensus_pick([d["text"] for d in fin]); return fin[b]
files = sorted(glob.glob(f"{OUT}/rl2c/*__sentfail8.jsonl.gz")) + [f"{OUT}/rl2c/t07x4__sentfail.jsonl.gz"]
for p in files:
    cfg = os.path.basename(p).split("__")[0]
    rows = {r["rid"]: r["outs"] for r in A.read_out("rl2c", cfg, "sentfail8" if "sentfail8" in p else "sentfail") if r["rid"] in REQ}
    if not rows: continue
    nmax = max(len(o) for o in rows.values())
    for rule in (("cons", "first") if nmax > 1 else ("first",)):
        chosen = {}
        for rid in REQ:
            if rid in rows: chosen[rid] = pick(rows[rid], iso_of[rid], rule, nmax)
            else: chosen[rid] = A.dec(iso_of[rid], greedy[rid])
        e = evaluate(chosen); e["n_candidates"] = nmax; e["rule"] = rule; e["units_covered"] = len(rows)
        e["any_candidate_finished"] = float(np.mean([any(A.dec(iso_of[rid], o)["finished"] for o in rows[rid]) for rid in rows]))
        e["mean_cand_finished"] = float(np.mean([np.mean([A.dec(iso_of[rid], o)["finished"] for o in rows[rid]]) for rid in rows]))
        out[f"{cfg} rule={rule}"] = e; print(cfg, rule, e, flush=True)
json.dump(out, open("/workspace/jobs/r12/res/unit_repair.json", "w"), indent=1)
