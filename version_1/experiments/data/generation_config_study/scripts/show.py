import json, sys
r = json.load(open("/workspace/jobs/r12/res/grid.json"))
pat = sys.argv[1] if len(sys.argv) > 1 else ""
print(len(r))
for n, v in sorted(r.items(), key=lambda kv: -kv[1]["chrf_all"]):
    if pat and pat not in n: continue
    print(f"{n:22s} {v['chrf_all']:.2f} {v['d_chrf_all']:+.2f} [{v['lo_chrf_all']:+.2f},{v['hi_chrf_all']:+.2f}] sp {v['bleu_all']:.2f} {v['d_bleu_all']:+.2f} loop {v['loop']*100:.2f} num {v['num_kept']*100:.1f} lid {v['lid']*100:.1f} strong {v['d_chrf_strong']:+.2f} med {v['d_chrf_medium']:+.2f} weak {v['d_chrf_weak']:+.2f}")
