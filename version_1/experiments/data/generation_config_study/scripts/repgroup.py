import json, numpy as np, sl
out = {}
def stats(h):
    r3 = np.array([sl._rep3(x) and not sl._rep3(sl.REFS[k]) for k, x in enumerate(h)])
    return {g: float(np.mean([r3[sl.IDX[i]].mean() for i in ml])) for g, ml in sl.GM.items()}
for n in ["greedy", "T0.2", "T0.3", "T0.2_mp0.05", "T0.5", "T0.7", "T1.0", "T1.0_p0.9", "T1.0_mp0.05", "T1.0_mp0.1", "T0.7_mp0.05", "mbr8__T0.5_n16", "mbr16__T0.5_n16", "mbr16__T0.7_n16", "mbr16__T0.7_mp0.05_n16", "mbr16__T1.0_mp0.05_n16", "mbrg16__T0.7_n16", "mbrqe16__T0.7_n16", "qe16__T0.7_n16", "oracle16__T0.7_n16"]:
    try: out[n] = stats(sl.find_row(n)["hyp"])
    except Exception as e: print("skip", n, e)
import os
for n in ["beam2", "beam4", "beam8", "nr3", "nr4", "rep1.05", "rep1.1"]:
    p = f"/workspace/jobs/a1/study/gen_{n}.json"
    if os.path.exists(p):
        d = sl.decode_cfg(json.load(open(p))); out["R5:" + n] = stats([sl.nfc(r[0]["hyp"]) for r in d])
    else: print("missing", p)
print({k: {g: round(v * 100, 1) for g, v in d.items()} for k, d in out.items()})
json.dump(out, open(sl.R + "/res/rep_by_group.json", "w"), indent=1)
