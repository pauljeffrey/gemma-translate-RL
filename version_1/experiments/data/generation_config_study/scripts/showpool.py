import json
r = json.load(open("/workspace/jobs/r12/res/pools.json"))
for k, v in sorted(r.items(), key=lambda kv: (kv[1]["pool"], kv[1]["method"], kv[1]["n"])):
    if v["method"] == "first": continue
    print(f"{k:34s} {v['chrf_all']:.2f} {v['d_chrf_all']:+.2f} [{v['lo_chrf_all']:+.2f},{v['hi_chrf_all']:+.2f}] sp {v['d_bleu_all']:+.2f} str {v['d_chrf_strong']:+.2f} med {v['d_chrf_medium']:+.2f} weak {v['d_chrf_weak']:+.2f} tok {v['tok_x_greedy']:.1f}x")
