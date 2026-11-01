"""R12: AfriCOMET-QE of every candidate of a pool (18 validated languages) -> picks by QE and by MBR+QE; rows cached under cache/row_<method><n>__<pool>.pkl (the QE table in cache/qecand_<pool>.npy).
usage: qe_pools.py pool:n ..."""
import os, sys
import numpy as np
import sl, pools
from gemma_mt.eval.runtime import CometPair
from gemma_mt.languages import QE_VALIDATED
pair = CometPair(python="/workspace/venv_gemma/bin/python", pythonpath="/workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs", batch_size=64, cache_dir=sl.R + "/comet_cache")
valid = np.array([r["iso"] in QE_VALIDATED for r in sl.rows])
for spec in sys.argv[1:]:
    pname, n = spec.split(":"); n = int(n)
    pd = pools.pool_data(pname); p = f"{sl.R}/cache/qecand_{pname}.npy"
    if os.path.exists(p): Q = np.load(p)
    else:
        Q = np.full((sl.N, 16), np.nan, np.float32); ks = np.where(valid)[0]
        srcs = [sl.rows[k]["english"] for k in ks for a in range(16)]; hs = [pd["texts"][k][a] for k in ks for a in range(16)]
        Q[ks] = np.array(pair.qe(srcs, hs), np.float32).reshape(len(ks), 16); np.save(p, Q)
    for method in ("qe", "mbrqe"):
        key = f"{method}{n}__{pname}"
        picks = pools.select(pd, n, method, qe=np.nan_to_num(Q), qe_mask=valid)
        hy = [pd["texts"][k][picks[k]] for k in range(sl.N)]; fn = [bool(pd["fin"][k][picks[k]]) for k in range(sl.N)]; nt = [int(pd["ntok"][k][picks[k]]) for k in range(sl.N)]
        sl.cached_row(key, lambda: sl.row_stats(hy, fn, nt)); print("row", key, flush=True)
pair.close()
