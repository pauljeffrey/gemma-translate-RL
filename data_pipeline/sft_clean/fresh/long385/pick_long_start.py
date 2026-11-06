#!/usr/bin/env python3
"""Start checkpoint of the long-document RL (owner rule 2026-11-04): the science RL `best/` (highest dev chrF++ with the min-gain rule of the trainer), unless the LAST dev evaluation
of the run (`final/`) beats it by the same dev rule: dev chrF++ macro higher by >= 0.2 and the held-out AfriCOMET macro not lower by more than 0.004 (its noise). Written from metrics.jsonl only
(no GPU, no model load), so the long-document stage can start the moment the science RL has ended.   python pick_long_start.py RUN_DIR OUT.json"""
import json, os, sys

run, out = sys.argv[1:3]
L = [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))]
ev = [r for r in L if "bench/chrf_macro" in r and r.get("step", 0) > 0]
ok = lambda d: os.path.exists(os.path.join(run, d, "config.json")) and os.path.exists(os.path.join(run, d, "model.safetensors"))
res = dict(run=run, evals=[(r["step"], r["bench/chrf_macro"], r.get("bench/comet_macro")) for r in ev], best_ok=ok("best"), final_ok=ok("final"))
if not ev:
    pick, why = ("best" if ok("best") else "final"), "no dev evaluation after step 0"
else:
    b = max(ev, key=lambda r: r["bench/chrf_macro"]); f = ev[-1]
    res.update(best_step=b["step"], final_step=f["step"])
    pick, why = "best", f"best dev chrF++ {b['bench/chrf_macro']:.3f} at step {b['step']}"
    if f["step"] > b["step"] and f["bench/chrf_macro"] >= b["bench/chrf_macro"] + 0.2 and (f.get("bench/comet_macro") or 0) >= (b.get("bench/comet_macro") or 0) - 0.004:
        pick, why = "final", f"last dev evaluation (step {f['step']}) chrF++ {f['bench/chrf_macro']:.3f} >= best {b['bench/chrf_macro']:.3f} + 0.2 with held-out AfriCOMET not lower by more than 0.004"
    if not ok(pick):
        other = "final" if pick == "best" else "best"
        if ok(other): pick, why = other, why + f" (but {pick}/ is not a complete model dir: using {other}/)"
        else: raise SystemExit("neither best/ nor final/ is a complete model dir")
res.update(chosen=pick, reason=why, model_dir=os.path.join(run, pick))
# 2026-11-04 refinement (written after the science post-RL evaluation, using DEV data only): the trainer's best/ is the step-75 weights chosen on a 200-sentence-per-language dev subset with a min-gain rule;
# the post-run evaluation scored best/ and final/ on the whole AfriScience-MT DEV set (5,058 sentences, 6 languages) in sci_pairs.json. Same rule, more data: final/ wins when its dev chrF is higher by >= 0.2
# and the held-out AfriCOMET is not lower by more than 0.004; the test split is not used for the choice.
sp = "/workspace/science_sft/sci_pairs.json"
if os.path.exists(sp):
    pr = json.load(open(sp))["pairs"].get("rl_best->rl_final/dev")
    if pr:
        res["full_dev_best_vs_final"] = dict(chrf_diff=pr["chrf"]["diff"], chrf_ci=[pr["chrf"]["lo"], pr["chrf"]["hi"]], afri_diff=pr["afri"]["diff"], afri_ci=[pr["afri"]["lo"], pr["afri"]["hi"]])
        if pr["chrf"]["diff"] >= 0.2 and pr["afri"]["diff"] >= -0.004 and ok("final"):
            res.update(chosen="final", model_dir=os.path.join(run, "final"), reason=f"full AfriScience-MT dev set: final beats best by {pr['chrf']['diff']:+.3f} chrF [{pr['chrf']['lo']:+.3f}, {pr['chrf']['hi']:+.3f}] and {pr['afri']['diff']:+.4f} held-out AfriCOMET (rule: >= +0.2 and not lower by > 0.004); trainer-subset rule alone said: {why}")
        else:
            res["reason"] = res["reason"] + f" (full dev set: final minus best {pr['chrf']['diff']:+.3f} chrF, {pr['afri']['diff']:+.4f} AfriCOMET: below the +0.2 rule, best kept)"
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
