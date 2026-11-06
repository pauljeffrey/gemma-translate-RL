#!/usr/bin/env python
"""Sanity gate between the E-02 smoke test and the full run: exit 1 when beam outputs look broken (eos fix not effective, empty outputs)."""
import json, sys
O = "/workspace/e2405/e02"
d = json.load(open(f"{O}/smoke_sft.json")); n = len(d["items"])
ok = True
for k, v in d["hyp"].items():
    fin = sum(h["finished"] for h in v) / len(v); empty = sum(1 for h in v if not h["hyp"].strip()) / len(v)
    print(k, "finished", round(fin, 3), "empty", round(empty, 3), "mean tokens", round(sum(h["ntok"] for h in v) / len(v), 1))
    if empty > 0.2: ok = False
    if k == "beam4" and fin < 0.9: ok = False
print("engine eos", d["engine_eos_id"], "timing_s", d["timing_s"], "items", n)
if d["engine_eos_id"] != 106: ok = False
sys.exit(0 if ok else 1)
