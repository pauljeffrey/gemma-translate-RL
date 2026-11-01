"""R12: best-of-n by MBR (chrF++ consensus), oracle, (and QE where scores exist) from n=16 pools; first n = 4, 8, 16 candidates. usage: pool_eval.py pool_name ... ; writes res/pools.json"""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
import sl, pools
pool = Pool(10)
out_p = sl.R + "/res/pools.json"; res = json.load(open(out_p)) if os.path.exists(out_p) else {}
gr = sl.get_row("greedy", pool=pool); base_bs = sl.boot(gr)
gtok = float(sl.get_dec("greedy")["tokens"])
for pname in sys.argv[1:]:
    t0 = time.time(); pd = pools.pool_data(pname, pool); orc = pools.oracle_scores(pname, pool); print(pname, "matrix", f"{time.time()-t0:.0f}s", flush=True)
    for n in (1, 2, 4, 8, 16):
        for method in (["first"] if n in (1, 2) else ["first", "mbr", "oracle"]):
            key = f"{method}{n}:{pname}"
            if key in res: continue
            if method == "oracle": picks = np.array([int(np.argmax(np.where(pd["fin"][k][:n], orc[k, :n], -1))) for k in range(sl.N)])
            else: picks = pools.select(pd, n, method)
            hy = [pd["texts"][k][picks[k]] for k in range(sl.N)]; fn = [bool(pd["fin"][k][picks[k]]) for k in range(sl.N)]; nt = [int(pd["ntok"][k][picks[k]]) for k in range(sl.N)]
            rs = sl.cached_row(key.replace(":", "__"), lambda: sl.row_stats(hy, fn, nt, pool)); bs = sl.boot(rs)
            tok_total = float(pd["ntok"][:, :n].sum())
            s = sl.summary(key, rs, bs, base_bs, None, tok_total, "vllm+cpu"); s["n"] = n; s["method"] = method; s["pool"] = pname; s["tok_x_greedy"] = tok_total / gtok
            s["picked_first"] = float(np.mean(picks == 0)); res[key] = s
            print(f"{key:34s} chrF {s['chrf_all']:.2f} {s['d_chrf_all']:+.2f} [{s['lo_chrf_all']:+.2f},{s['hi_chrf_all']:+.2f}] sp {s['d_bleu_all']:+.2f} loop {s['loop']*100:.2f}", flush=True)
    if pname in ("T0.5_n16", "T0.7_n16", "T1.0_mp0.05_n16", "T0.7_mp0.05_n16"):
        for n in (8, 16):
            key = f"mbrg{n}:{pname}"
            if key in res: continue
            def mk():
                hy, fn, pg = pools.mbr_with_greedy(pd, n, gr["hyp"], gr["fin"]); r = sl.row_stats(hy, fn, [0] * sl.N, pool); r["picked_greedy"] = pg; return r
            rs = sl.cached_row(key.replace(":", "__"), mk); bs = sl.boot(rs)
            tok_total = float(pd["ntok"][:, :n].sum()) + gtok
            s = sl.summary(key, rs, bs, base_bs, None, tok_total, "vllm+cpu"); s["n"] = n; s["method"] = "mbrg"; s["pool"] = pname; s["tok_x_greedy"] = tok_total / gtok; s["picked_first"] = float(rs["picked_greedy"].mean()); res[key] = s
            print(f"{key:34s} chrF {s['chrf_all']:.2f} {s['d_chrf_all']:+.2f} [{s['lo_chrf_all']:+.2f},{s['hi_chrf_all']:+.2f}] picked greedy {s['picked_first']:.2f}", flush=True)
    json.dump(res, open(out_p, "w"))
