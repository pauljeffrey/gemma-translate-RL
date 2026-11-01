import json, sys, os
sys.path.insert(0, "/workspace/jobs/r12/doc")
from common import *
from gemma_mt.eval.longdoc import split_long_sentence
B8 = set("hau yor swh amh zul wol kea sag".split())
items = rd(WORK + "/items.jsonl")
sel = [it for it in items if it["iso"] in B8 and ((it["set"] == "core" and it["hbin"] in (256, 512, 1024, 2048)) or (it["set"] == "human" and not it["contam"]))]
sel_ids = {it["iid"] for it in sel}
json.dump(sorted(sel_ids), open(MYWORK + "/sel_iids.json", "w"))
full = [r for r in rd(WORK + "/req_full.jsonl") if r["rid"] in sel_ids]
wr(MYWORK + "/req_docfull.jsonl", full); print("docfull", len(full))
units = {(it["iso"], p) for it in sel for s in it["sents"] for p in split_long_sentence(s, 700)}
# req_sent.jsonl of R6 no longer exists: rebuild it exactly as prep.py did (rid = index in the sorted unique (iso, piece) list over ALL items)
from transformers import AutoTokenizer
from gemma_mt.format import build_prompt
tok = AutoTokenizer.from_pretrained(MODELS["rl2c"])
ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
uniq = sorted({(it["iso"], p) for it in items for s in it["sents"] for p in split_long_sentence(s, 700)})
rid_of = {u: f"s{k}" for k, u in enumerate(uniq)}
chk = rd(WORK + "/req_sentfail.jsonl")
assert all(rid_of[(r["iso"], r["text"])] == r["rid"] for r in chk), "rid mismatch"
print("rid scheme verified on", len(chk), "R6 sentfail rows; unique units", len(uniq))
sent = []
for (iso, p) in sorted(units):
    ids = build_prompt(p, iso, tok); st = ntok(p); room = CTX - len(ids); want = min(int(3.5 * st) + 16, 1024)
    sent.append({"rid": rid_of[(iso, p)], "ids": ids, "max_new": min(want, room), "capped": want > room, "src_tokens": st, "iso": iso, "text": p})
for r in chk[:50]:   # same ids / budget as R6's rows
    q = next(x for x in sent if x["rid"] == r["rid"]) if r["rid"] in {x["rid"] for x in sent} else None
    if q: assert q["ids"] == r["ids"] and q["max_new"] == r["max_new"], r["rid"]
wr(MYWORK + "/req_sent8.jsonl", sent); print("sent8", len(sent))
sf = [r for r in rd(WORK + "/req_sentfail.jsonl") if r["iso"] in B8]
wr(MYWORK + "/req_sentfail8.jsonl", sf); print("sentfail8", len(sf))
# the same unfinished units restricted to units used by the selected items
sf2 = [r for r in sf if (r["iso"], r["text"]) in units]
wr(MYWORK + "/req_sentfail8u.jsonl", sf2); print("sentfail8u", len(sf2))
