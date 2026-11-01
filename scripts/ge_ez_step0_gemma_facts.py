"""Q3: parameter split, disk sizes and CPU-compute ratio of Gemma-3 270M vs 1B (and 4B text part). Inputs: Hub metadata (gemma_hub_metadata.json, read-only API) and config.json.
Output: gemma_facts.csv. Arithmetic only; CPU speed is an ESTIMATE from the compute per token and from our measured 270M numbers."""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from ge_ez_step0_common import OUT
V = 262144
m = json.load(open(os.path.join(OUT, "gemma_hub_metadata.json")))
rows = []
cfg = {"270M": dict(h=640, L=18, ffn=2048, nh=4, kv=1, hd=256, total=m["google/gemma-3-270m-it"]["params"]),
       "1B": dict(h=1152, L=26, ffn=6912, nh=4, kv=1, hd=256, total=m["google/gemma-3-1b-it"]["params"])}
for k, c in cfg.items():
    c["emb"] = V * c["h"]; c["nonemb"] = c["total"] - c["emb"]
    # check against the config by counting transformer weights: attn q,k,v,o + mlp gate/up/down + 4 norms per layer + final norm
    att = c["h"] * c["nh"] * c["hd"] + 2 * c["h"] * c["kv"] * c["hd"] + c["nh"] * c["hd"] * c["h"]
    mlp = 3 * c["h"] * c["ffn"]
    c["nonemb_from_config"] = c["L"] * (att + mlp + 4 * c["h"] + 2 * c["hd"]) + c["h"]
    c["p_eff"] = c["nonemb"] + c["emb"]
for k, c in cfg.items():
    rows.append(dict(model=k, total_params=c["total"], embedding_params=c["emb"], emb_share=c["emb"] / c["total"], non_embedding=c["nonemb"], non_emb_check_from_config=c["nonemb_from_config"],
                     layers=c["L"], hidden=c["h"], ffn=c["ffn"], head_matmul_share_of_per_token_compute=c["emb"] / c["p_eff"], per_token_params_touched=c["p_eff"]))
d = pd.DataFrame(rows); print(d.round(3).to_string())
r = cfg["1B"]["p_eff"] / cfg["270M"]["p_eff"]
print("per-token compute ratio 1B/270M:", round(r, 2), " layers part:", round(cfg["1B"]["nonemb"] / cfg["270M"]["nonemb"], 2), " head part:", round(cfg["1B"]["emb"] / cfg["270M"]["emb"], 2))
# bytes
b = {"bf16_270M": m["google/gemma-3-270m-it"]["files"][0][1], "bf16_1B": m["google/gemma-3-1b-it"]["files"][0][1], "Q8_0_1B_ggml-org": 1069306368, "Q4_K_M_1B_ggml-org": 806058240,
     "QAT_q4_0_1B_google": 1003541152, "f16_1B_ggml-org": 2006573568, "Q8_0_270M_ggml-org": 291545600, "Q8_0_270M_ours": 299748320}
P = cfg["270M"]
est = {"Q4_0 uniform 4.5 bpw (all weights)": P["total"] * 4.5 / 8, "Q4_K_M by the 1B ratio (806.1/1069.3 x our 299.7 MB)": 299748320 * 806058240 / 1069306368,
       "QAT-style (embedding f16 + transformer Q4_0), as Google's 1B file": P["emb"] * 2 + P["nonemb"] * 4.5 / 8}
est1 = {"1B QAT-style check (embedding f16 + transformer 4.5 bpw)": cfg["1B"]["emb"] * 2 + cfg["1B"]["nonemb"] * 4.5 / 8}
print(json.dumps({k: round(v / 1e6, 1) for k, v in {**b, **est, **est1}.items()}, indent=1))
json.dump({"bytes": b, "estimates": est, "check": est1, "ratio_per_token_compute": r}, open(os.path.join(OUT, "gemma_facts.json"), "w"), indent=1)
d.to_csv(os.path.join(OUT, "gemma_facts.csv"), index=False)
# CPU speed estimate (llama.cpp Q8_0, 4 threads, batch 8, box Xeon): measured 270M 1.85 sent/s (reports/cpu_speed_2026-10-23.md)
for lo, mid, hi in [(1 / 4.0, 1 / r, 1 / 3.3)]:
    print("1B Q8_0 sent/s batch 8: low %.2f central %.2f high %.2f (x slower: 4.0 / %.2f / 3.3)" % (1.85 * lo, 1.85 * mid, 1.85 * hi, r))
