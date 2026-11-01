import json, sl, numpy as np
P = json.load(open(sl.R + "/res/pools.json")); gr = sl.get_row("greedy"); base_bs = sl.boot(gr); gtok = float(sl.get_dec("greedy")["tokens"])
for key in ["qe16__T0.7_n16", "qe16__T1.0_mp0.05_n16", "mbrqe16__T0.7_n16", "mbrqe16__T1.0_mp0.05_n16", "qe8__T0.5_n16", "qe8__T0.7_n16", "qe8__T1.0_mp0.05_n16", "mbrqe8__T0.5_n16", "mbrqe8__T0.7_n16", "mbrqe8__T1.0_mp0.05_n16"]:
    try: rs = sl.find_row(key)
    except FileNotFoundError: continue
    m, rest = key.split("__"); n = int("".join(ch for ch in m if ch.isdigit()))
    s = sl.summary(key, rs, sl.boot(rs), base_bs, None, None, "vllm+qe"); s["n"] = n; s["method"] = m.rstrip("0123456789"); s["pool"] = rest; s["tok_x_greedy"] = float(n); s["picked_first"] = 0.0
    P[key] = s; print(f"{key:30s} chrF {s['chrf_all']:.2f} {s['d_chrf_all']:+.2f} [{s['lo_chrf_all']:+.2f},{s['hi_chrf_all']:+.2f}] spBLEU {s['d_bleu_all']:+.2f} str {s['d_chrf_strong']:+.2f} med {s['d_chrf_medium']:+.2f} weak {s['d_chrf_weak']:+.2f} num {s['num_kept']*100:.1f} lid {s['lid']*100:.1f}")
json.dump(P, open(sl.R + "/res/pools.json", "w"))
