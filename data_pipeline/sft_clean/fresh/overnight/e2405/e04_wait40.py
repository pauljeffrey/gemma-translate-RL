#!/usr/bin/env python
"""Prints READY when the step-40 train record exists in RUN/metrics.jsonl AND last/trainer_state.pt was (re)written after it (= the step-40 checkpoint), else nothing.  python e04_wait40.py RUN [STEP]"""
import json, os, sys
run = sys.argv[1]; want = int(sys.argv[2]) if len(sys.argv) > 2 else 40
m = os.path.join(run, "metrics.jsonl"); ck = os.path.join(run, "last", "trainer_state.pt")
if not (os.path.exists(m) and os.path.exists(ck)) or os.path.exists(ck + ".tmp"):
    sys.exit(0)
t_rec = None
with open(m, "rb") as f:
    f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 400000)); tail = f.read().decode("utf-8", "ignore").splitlines()
for line in tail:
    try: r = json.loads(line)
    except Exception: continue
    if "reward" in r and r.get("step") == want: t_rec = r.get("time")
if t_rec is not None and os.path.getmtime(ck) > t_rec:
    print("READY")
