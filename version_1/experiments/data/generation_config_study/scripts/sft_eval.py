"""R12: the SFT-final model, sentence level: single samples (base = SFT greedy) and MBR of a 16-sample pool. writes res/sft.json"""
import glob, json, os
from multiprocessing import Pool
import numpy as np
import sl, pools
pool = Pool(8)
names = sorted(os.path.basename(p)[4:-8] for p in glob.glob(sl.R + "/gens/gen_sft_*.json.gz") if "_n16" not in p)
base = sl.get_row("sft_greedy", pool=pool); base_bs = sl.boot(base); res = {}
for n in names:
    rs = sl.get_row(n, 0, pool=pool); d = sl.get_dec(n); res[n] = sl.summary(n, rs, sl.boot(rs), base_bs, d["seconds"], d["tokens"], d["engine"])
    print(f"{n:24s} chrF {res[n]['chrf_all']:.2f} {res[n]['d_chrf_all']:+.2f} [{res[n]['lo_chrf_all']:+.2f},{res[n]['hi_chrf_all']:+.2f}] loop {res[n]['loop']*100:.2f} fin {res[n]['finished']*100:.1f} rep3 {res[n]['rep3']*100:.1f}", flush=True)
gtok = float(sl.get_dec("sft_greedy")["tokens"])
for pname in [os.path.basename(p)[4:-8] for p in glob.glob(sl.R + "/gens/gen_sft_*_n16.json.gz")]:
    pd = pools.pool_data(pname, pool)
    for n in (8, 16):
        for method in ("mbr", "oracle"):
            if method == "oracle": continue
            picks = pools.select(pd, n, method)
            hy = [pd["texts"][k][picks[k]] for k in range(sl.N)]; fn = [bool(pd["fin"][k][picks[k]]) for k in range(sl.N)]; nt = [int(pd["ntok"][k][picks[k]]) for k in range(sl.N)]
            rs = sl.cached_row(f"{method}{n}__{pname}", lambda: sl.row_stats(hy, fn, nt, pool)); s = sl.summary(f"{method}{n}:{pname}", rs, sl.boot(rs), base_bs, None, None, "vllm+cpu")
            res[f"{method}{n}:{pname}"] = s; print(f"{method}{n}:{pname}", f"chrF {s['chrf_all']:.2f} {s['d_chrf_all']:+.2f} [{s['lo_chrf_all']:+.2f},{s['hi_chrf_all']:+.2f}] weak {s['d_chrf_weak']:+.2f}", flush=True)
json.dump(res, open(sl.R + "/res/sft.json", "w"))
