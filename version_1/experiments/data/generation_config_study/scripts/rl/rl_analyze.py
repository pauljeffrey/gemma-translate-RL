"""R12 RL part 5 (venv_gemma, CPU): tables from rl/res/*.jsonl (+ bias_own.json, bias_flags.jsonl) -> rl/res/rl_summary.json"""
import collections, json, os, sys
import numpy as np
R = "/workspace/jobs/r12/rl"
P = {p["pid"]: p for p in json.load(open(R + "/work/prompts.json"))}
names = [n for n in ["greedy", "T1.0", "T1.0+g", "T1.0_p0.95", "T1.0_p0.9", "T1.0_p0.8", "T1.0_mp0.05", "T1.0_mp0.1", "T1.0_k64_p0.95", "T0.8", "T0.7", "T0.5", "T0.3"] if os.path.exists(f"{R}/res/{n}.jsonl")]
def load(n): return [json.loads(l) for l in open(f"{R}/res/{n}.jsonl")]
def macro(vals_by_iso): return float(np.mean([np.mean(v) for v in vals_by_iso.values()])) if vals_by_iso else None
out = {"configs": {}}
data = {n: load(n) for n in names}
MODE = {}
for r in data["T1.0"]:
    if r["status"] == "ok": MODE.setdefault(P[r["pid"]]["iso"], r["mode"])
out["language_mode"] = MODE
for n in names:
    rows = data[n]; g = collections.defaultdict(list)
    for r in rows: g[r["pid"]].append(r)
    by_iso = collections.defaultdict(list); by_bin = collections.defaultdict(lambda: collections.defaultdict(list))
    gstd, grange, gbest, gtop4, flat = [], [], [], [], 0
    st = collections.Counter()
    for pid, rs in g.items():
        rw = np.array([r["reward"] for r in rs]); iso, b = P[pid]["iso"], P[pid]["bin"]
        by_iso[iso].append(rw.mean()); by_bin[b][iso].append(rw.mean())
        if len(rs) > 1:
            gstd.append(rw.std()); grange.append(rw.max() - rw.min()); gbest.append(rw.max()); gtop4.append(np.sort(rw)[-4:].mean()); flat += rw.std() < 0.05
        for r in rs: st[r["status"]] += 1
    nrow = len(rows)
    bm = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows: bm[MODE.get(P[r["pid"]]["iso"], "qe")][P[r["pid"]]["iso"]].append(r["reward"])
    out["configs"][n] = {"reward_by_mode": {m: macro(v) for m, v in bm.items()}, "share_mode": {m: sum(len(x) for x in v.values()) / nrow for m, v in bm.items()}, "n": nrow, "reward_macro": macro(by_iso), "reward_by_bin": {b: macro(v) for b, v in by_bin.items()},
        "reward_by_iso": {i: float(np.mean(v)) for i, v in by_iso.items()}, "ok_rate": st.get("ok", 0) / nrow, "finish_stop": float(np.mean([r["finish"] == "stop" for r in rows])),
        "status": {k: v / nrow for k, v in st.most_common(8)}, "group_std": float(np.mean(gstd)) if gstd else None, "group_range": float(np.mean(grange)) if grange else None,
        "group_best": float(np.mean(gbest)) if gbest else None, "group_top4": float(np.mean(gtop4)) if gtop4 else None, "flat_groups": flat / max(len(gstd), 1) if gstd else None,
        "mean_ntok": float(np.mean([r["ntok"] for r in rows])), "qe_mean_ok": float(np.mean([r["qe"] for r in rows if r["qe"] is not None])) if any(r["qe"] is not None for r in rows) else None}
if "T1.0+g" in data:
    d = collections.defaultdict(list)
    for r in data["T1.0+g"]: d[r["pid"]].append(r)
    gr, sm, gtop = [], [], []
    for pid, rs in d.items():
        rw = np.array([r["reward"] for r in sorted(rs, key=lambda r: r["j"])]); gr.append(rw[-1]); sm.append(rw[:-1].mean()); gtop.append(rw[-1] > rw[:-1].mean())
    out["greedy_member"] = {"greedy_reward": float(np.mean(gr)), "sample_mean_reward": float(np.mean(sm)), "greedy_above_group_mean": float(np.mean(gtop)), "mean_adv_greedy_raw": float(np.mean(np.array(gr) - np.array(sm)))}
if os.path.exists(R + "/res/bias_own.json"): out["bias_own"] = json.load(open(R + "/res/bias_own.json"))["own"]
if os.path.exists(R + "/res/bias_flags.jsonl") and "T1.0" in data:
    rw = {(r["pid"], r["j"]): r["reward"] for r in data["T1.0"]}; flags = [json.loads(l) for l in open(R + "/res/bias_flags.jsonl")]
    tr = collections.defaultdict(lambda: {"n": 0, "n_out": 0, "r_in": [], "r_out": [], "tok_out": 0, "tok": 0})
    for f in flags:
        for t, o in f["outside"].items():
            d = tr[t]; d["n"] += 1; d["tok"] += f["ntok"]; d["tok_out"] += o
            if (f["pid"], f["j"]) not in rw: continue
            if o > 0: d["n_out"] += 1; d["r_out"].append(rw[(f["pid"], f["j"])])
            else: d["r_in"].append(rw[(f["pid"], f["j"])])
    out["t1_flags"] = {t: {"share_seq_with_cut_token": d["n_out"] / d["n"], "share_tokens_cut": d["tok_out"] / d["tok"], "reward_inside": float(np.mean(d["r_in"])) if d["r_in"] else None,
                           "reward_with_cut_token": float(np.mean(d["r_out"])) if d["r_out"] else None, "n": d["n"]} for t, d in tr.items()}
json.dump(out, open(R + "/res/rl_summary.json", "w"), indent=1)
for n in names:
    c = out["configs"][n]; print(f"{n:16s} reward {c['reward_macro']:.3f} ok {c['ok_rate']:.3f} fin {c['finish_stop']:.3f} gstd {c['group_std'] if c['group_std'] is None else round(c['group_std'], 3)} best {c['group_best'] if c['group_best'] is None else round(c['group_best'], 3)} flat {c['flat_groups']}")
