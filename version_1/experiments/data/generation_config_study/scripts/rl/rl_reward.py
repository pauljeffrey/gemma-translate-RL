"""R12 RL part 3 (venv_gemma, GPU for the AfriCOMET worker): the REAL RL reward (rl_rtxpro_phase2c.yaml reward + scorer blocks, calibration v7, group consensus with
group_size = the group) of every rollout. usage: rl_reward.py name [name ...]  (+ 'T1.0+g' = 15 samples of T1.0 + the greedy completion as 16th member, the D-59 recipe)"""
import gzip, json, os, sys, time
sys.path.insert(0, "/workspace/sft_build/pylibs")
from gemma_mt.eval.longdoc_reward import build_reward
from gemma_mt.rl.reward import Sample
R = "/workspace/jobs/r12/rl"; os.makedirs(R + "/res", exist_ok=True)
CFG = "/workspace/sft_build/gemma_mt_2c/configs/rl_rtxpro_phase2c.yaml"; CAL = "/workspace/sft_build/gemma_mt_2c/calibration/shaping_flores_dev_v7.json"
P = {p["pid"]: p for p in json.load(open(R + "/work/prompts.json"))}
def load(name):
    with gzip.open(f"{R}/gens/{name}.json.gz", "rt") as f: return json.load(f)
reward, tok, meta = build_reward(CFG, calibration=CAL, scorer_kind="worker", tokenizer_json=("/workspace/sft_build/hf/hub/models--" + os.environ['HF_ORG_LEGACY'] + "--gemma-3-270m-translate-rl2c/snapshots/80baefff3b16b120b3e37e70afeb51f1fc248d96/tokenizer.json"), python="/workspace/venv_gemma/bin/python", pythonpath="/workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs")
print(json.dumps(meta)[:600], flush=True)
def groups_of(name):
    if name == "T1.0+g":
        a, g = load("T1.0"), load("greedy"); gm = {r["pid"]: r["outs"][0] for r in g["res"]}
        return [{"pid": r["pid"], "outs": r["outs"][:15] + [gm[r["pid"]]], "greedy_last": True} for r in a["res"]]
    return [r for r in load(name)["res"] if r["pid"] in P]
for name in sys.argv[1:]:
    out_p = f"{R}/res/{name}.jsonl"
    if os.path.exists(out_p): print("skip", name); continue
    t0 = time.time(); G = groups_of(name); rows = []
    order = sorted(range(len(G)), key=lambda i: (P[G[i]["pid"]]["bin"], P[G[i]["pid"]]["iso"]))
    for s in range(0, len(order), 8):
        idx = order[s:s + 8]; samples, keys = [], []
        for i in idx:
            p = P[G[i]["pid"]]
            for j, o in enumerate(G[i]["outs"]):
                samples.append(Sample(token_ids=o["ids"], iso=p["iso"], source=p["english"])); keys.append((G[i]["pid"], j, o["finish"], len(o["ids"])))
        gs = len(G[idx[0]]["outs"])
        res = reward(samples, group_size=gs)
        for k, (pid, j, fin, nt) in enumerate(keys):
            rows.append({"pid": pid, "j": j, "finish": fin, "ntok": nt, "reward": float(res.rewards[k]), "status": res.status[k], "qe": res.qe[k], "mode": res.content_mode[k] if res.content_mode else None,
                         "cons": res.consensus_chrf[k] if res.consensus_chrf else None, "weight": float(res.weights[k]), "nch": res.n_chunks[k] if res.n_chunks else None,
                         "fch": res.failed_chunks[k] if res.failed_chunks else None, "text": res.texts[k]})
        print(f"[{name}] {s + len(idx)}/{len(G)} prompts {time.time()-t0:.0f}s", flush=True)
    with open(out_p + ".tmp", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(out_p + ".tmp", out_p); print("DONE", name, len(rows), f"{time.time()-t0:.0f}s", flush=True)
