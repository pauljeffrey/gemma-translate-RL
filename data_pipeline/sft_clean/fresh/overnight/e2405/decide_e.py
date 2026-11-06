#!/usr/bin/env python
"""E-11 arm (e) rule (pre-registered): run the KL-guard arm (threshold 0.25) only if the KL guard (threshold 0.15, as in stage A) acted in one of the E-04 seed runs within 100 steps, i.e. a record has kl_guard_cut
truthy or a 5-step median KL above 0.15. Prints RUN or SKIP and the evidence.   python decide_e.py RUN_DIR [RUN_DIR ...]"""
import json, sys
run, ev = False, []
for d in sys.argv[1:]:
    mx, cuts = 0.0, 0
    for l in open(d + "/metrics.jsonl"):
        try: r = json.loads(l)
        except Exception: continue
        if "reward" not in r or r.get("step", 0) > 100: continue
        mx = max(mx, float(r.get("kl_guard_median") or 0)); cuts += 1 if r.get("kl_guard_cut") else 0
    ev.append(f"{d}: max 5-step median KL {mx:.4f}, guard cuts {cuts}"); run = run or cuts > 0 or mx > 0.15
print("RUN" if run else "SKIP"); print("\n".join(ev))
