"""R12 document assembly (venv_gemma; run in doc/): generation outputs -> hyp/rl2c/<name>.jsonl (same row format as R6's assemble.py).
usage:
  asm.py full <cfg>                      whole-document output of <cfg>__docfull  (n=1: first candidate; n>1: also <cfg>_bon<k> = consensus pick among the first k)
  asm.py sent <cfg> [n_pick]             sentence-by-sentence with units from <cfg>__sent8 (n_pick candidates: consensus among the finished ones)
  asm.py fix <base_units_cfg> <repair_cfg> <name> [rule]   greedy (or <base_units_cfg>) units; unfinished ones replaced from <repair_cfg>__sentfail8 by rule: cons (R6: consensus among finished), first (first finished), none"""
import sys, os, json, gzip
sys.path.insert(0, "/workspace/jobs/r12/doc")
import assemble as A
from common import *
from gemma_mt.eval.longdoc import join_sentences, norm_text, split_long_sentence
model = "rl2c"
SEL = set(json.load(open(MYWORK + "/sel_iids.json")))
REQ = {(q["iso"], q["text"]): q["rid"] for q in rd(MYWORK + "/req_sent8.jsonl")}
items = {k: v for k, v in A.items.items() if k in SEL}

def full(cfg, rn="docfull"):
    cm = A.assemble_full(model, cfg, rn)
    A.write(model, cfg, A.single_rows(cm, 0))
    nmax = max(len(c) for c in cm.values())
    if nmax > 1:
        import multiprocessing as mp
        for N in (4, 8):
            if nmax < N: continue
            todo = [(iid, [x["text"] for x in c[:N]]) for iid, c in cm.items()]
            with mp.Pool(8) as pool: res = dict(pool.imap_unordered(A._pick_worker, todo, chunksize=8))
            rows = []
            for iid, c in cm.items():
                b = res[iid][0]; x = c[b]
                rows.append({"iid": iid, "hyp": x["text"], "finished": x["finished"], "new_tokens": sum(y["new_tokens"] for y in c[:N]), "status": x["status"], "dangling": x["dangling"], "pick": b, "n_finished": sum(y["finished"] for y in c[:N])})
            A.write(model, f"{cfg}_bon{N}", rows)

def unit_decs(cfg, rname=None):
    rname = rname or ("sent8" if os.path.exists(f"{OUT}/{model}/{cfg}__sent8.jsonl.gz") else "sent")
    out = {}
    for r in A.read_out(model, cfg, rname):
        out[r["rid"]] = r["outs"]
    return out

def pick_units(outs_by_rid, iso_of, n_pick=1, rule="cons"):
    """rid -> dec dict (chosen candidate)"""
    res = {}
    for rid, outs in outs_by_rid.items():
        if rid not in iso_of: continue
        ds = [A.dec(iso_of[rid], o) for o in outs[:n_pick]]
        if len(ds) == 1: res[rid] = ds[0]; continue
        fin = [d for d in ds if d["finished"]] or ds
        if len(fin) == 1 or rule == "first": res[rid] = fin[0]
        else:
            b, _ = A.consensus_pick([d["text"] for d in fin]); res[rid] = fin[b]
    return res

def build(unit_pick, name, extra_new_tokens=None):
    rows = []
    for iid, it in items.items():
        so, fin, nt, bad = [], True, 0, 0
        for s in it["sents"]:
            ps = [unit_pick[REQ[(it["iso"], p)]] for p in split_long_sentence(s, 700)]
            so.append(norm_text(" ".join(x["text"] for x in ps))); fin &= all(x["finished"] for x in ps); nt += sum(x["new_tokens"] for x in ps); bad += sum(x["status"] != "ok" for x in ps)
        rows.append({"iid": iid, "hyp": join_sentences(so, it["para_starts"]), "hyp_sents": so, "finished": fin, "new_tokens": nt, "status": "ok" if bad == 0 else f"{bad}_bad_units", "dangling": False})
    A.write(model, name, rows)

def main():
    mode = sys.argv[1]
    iso_of = {rid: iso for (iso, _), rid in REQ.items()}
    if mode == "full": full(sys.argv[2])
    elif mode == "full2": full(sys.argv[2], "docfull2")
    elif mode == "sent":
        cfg = sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        up = pick_units(unit_decs(cfg), iso_of, n)
        build(up, f"{cfg}_sent" + (f"_c{n}" if n > 1 else ""))
    elif mode == "fix":
        base, rep, name = sys.argv[2:5]; rule = sys.argv[5] if len(sys.argv) > 5 else "cons"
        up = pick_units(unit_decs(base), iso_of, 1)
        if rule != "none":
            fixes = {}
            for r in A.read_out(model, rep, "sentfail8") if os.path.exists(f"{OUT}/{model}/{rep}__sentfail8.jsonl.gz") else A.read_out(model, rep, "sentfail"):
                fixes[r["rid"]] = r["outs"]
            nfix = nbad = 0
            for rid, outs in fixes.items():
                if rid not in up: continue
                ds = [A.dec(iso_of[rid], o) for o in outs]; fin = [d for d in ds if d["finished"]]
                if not fin: nbad += 1; continue
                if len(fin) == 1 or rule == "first": up[rid] = fin[0]
                else: b, _ = A.consensus_pick([d["text"] for d in fin]); up[rid] = fin[b]
                nfix += 1
            print("repaired", nfix, "still unfinished", nbad)
        build(up, name)
main()
