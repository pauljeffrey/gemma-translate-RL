# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU seconds per sample of the v2 pieces vs the v1 plan, by bin (R8 cost report). Single process, one core."""
import os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from gemma_mt.rl import longdoc as LD  # noqa: E402
from gemma_mt.rl.chunk_plan import plan_chunks  # noqa: E402
from gemma_mt.rl.reward_shaping import ContentShaper  # noqa: E402

sh = ContentShaper.from_file("/workspace/jobs/r8/gemma_mt_r8/calibration/shaping_flores_dev_v8.json", reliability_from="same_source")
cal = LD.LongdocCalibration.from_file("/workspace/jobs/r8/gemma_mt_r8/calibration/longdoc_v2.json")
docs = C.core_docs()
items = C.rd(sys.argv[1])
want = {"sf": "complete", "W:t10": "T=1 sample", "W:greedy": "greedy"}
out = {}
for it in items:
    if it["cond"] not in want or it["iso"] not in ("hau", "yor", "amh") or it["bin"] < 256:
        continue
    src = docs[it["doc_id"]]["text"]
    iso = it["iso"]
    lm = sh.length_model(iso, None, None)
    om = {"length_model": lm, "short_frac": 0.7, "short_min": 0.15}
    w_hi = sh.len_window[iso][1]
    t = {}
    t0 = time.perf_counter(); plan_chunks(src, it["out"], w_hi=w_hi, **om); t["plan_v1"] = time.perf_counter() - t0
    t0 = time.perf_counter(); plan_chunks(src, it["out"], w_hi=w_hi, beads_fn=LD.beads_fn(lm), **om); t["plan_v2"] = time.perf_counter() - t0
    t0 = time.perf_counter(); p = LD.repetition_profile(it["out"]); sp = LD.repetition_profile(src); t["profile_out+src"] = time.perf_counter() - t0
    t0 = time.perf_counter(); p = LD.repetition_profile(it["out"]); t["profile_out"] = time.perf_counter() - t0
    out.setdefault((it["bin"], it["cond"]), []).append(t)
print("bin cond n | ms/sample: plan_v1 plan_v2 profile_out profile_out+src")
for (b, c), lst in sorted(out.items()):
    print(b, c, len(lst), *[f"{1000*statistics.mean(x[k] for x in lst):.1f}" for k in ("plan_v1", "plan_v2", "profile_out", "profile_out+src")])
