"""R12: score every single-sample generation in gens/ (first candidate), paired bootstrap vs greedy. usage: eval_grid.py [pattern]; writes res/grid.json"""
import glob, json, os, sys, pickle, time
from multiprocessing import Pool
import numpy as np
import sl
os.makedirs(sl.R + "/res", exist_ok=True)
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
names = sorted(os.path.basename(p)[4:-8] for p in glob.glob(f"{sl.R}/gens/gen_{pat}.json.gz"))
out_p = sl.R + "/res/grid.json"
res = json.load(open(out_p)) if os.path.exists(out_p) else {}
pool = Pool(8)
base = sl.get_row("greedy", pool=pool); base_bs = sl.boot(base)
for n in names:
    if n in res and not os.environ.get("REDO"): continue
    t0 = time.time(); d = sl.get_dec(n)
    rs = sl.get_row(n, 0, pool=pool); bs = sl.boot(rs)
    res[n] = sl.summary(n, rs, bs, base_bs, d["seconds"], d["tokens"], d["engine"])
    res[n]["kw"] = d["kw"]
    print(f"{n:24s} chrF {res[n]['chrf_all']:.2f} d {res[n]['d_chrf_all']:+.2f} [{res[n]['lo_chrf_all']:+.2f},{res[n]['hi_chrf_all']:+.2f}] loop {res[n]['loop']*100:.2f}% fin {res[n]['finished']*100:.1f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(out_p, "w"))
