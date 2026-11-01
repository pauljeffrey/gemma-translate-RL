"""document subset for the unit-level (sentence-by-sentence) sampling experiments: sel2_iids.json + req_sent8s.jsonl (units of those documents) + req_sentfail8 split"""
import json, random, sys, collections
sys.path.insert(0, "/workspace/jobs/r12/doc")
from common import *
from gemma_mt.eval.longdoc import split_long_sentence
SEL = set(json.load(open(MYWORK + "/sel_iids.json")))
items = {r["iid"]: r for r in rd(WORK + "/items.jsonl") if r["iid"] in SEL}
rng = random.Random(12)
CAP = {("core", 256): 30, ("core", 512): 30, ("core", 1024): 30, ("core", 2048): 20}
by = collections.defaultdict(list)
for iid, it in items.items():
    by[(it["set"], it["hbin"], it["doc_id"])].append(iid)
sel2 = set()
docs_by_bin = collections.defaultdict(list)
for (s, b, d) in sorted(by, key=str): docs_by_bin[(s, b)].append(d)
for (s, b), ds in docs_by_bin.items():
    if s == "core" and b not in (256, 512, 1024, 2048): continue
    ds = sorted(set(ds)); rng.shuffle(ds)
    if s == "core": keep = ds[:CAP[(s, b)]]
    else: keep = ds[:20 * 6]            # human: up to 120 documents per bin (<= 20 per language on average)
    for d in keep: sel2 |= set(by[(s, b, d)])
json.dump(sorted(sel2), open(MYWORK + "/sel2_iids.json", "w"))
units = {(it["iso"], p) for iid in sel2 for it in [items[iid]] for s in it["sents"] for p in split_long_sentence(s, 700)}
req = {(r["iso"], r["text"]): r for r in rd(MYWORK + "/req_sent8.jsonl")}
wr(MYWORK + "/req_sent8s.jsonl", [req[u] for u in sorted(units)])
print("items", len(sel2), collections.Counter((items[i]["set"], items[i]["hbin"]) for i in sel2), "units", len(units))

full = [r for r in rd(WORK + "/req_full.jsonl") if r["rid"] in sel2]
wr(MYWORK + "/req_docfull2.jsonl", full); print("docfull2", len(full), sum(r["max_new"] for r in full) / 1e6, "M max tokens")
