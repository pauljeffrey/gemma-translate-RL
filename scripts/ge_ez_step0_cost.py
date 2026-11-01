"""Q4: compute-scaling estimate of the training cost of repeating (i) the reduced Ge'ez diagnostic and (ii) the full pipeline at Gemma-3-1B and 4B.
EVERYTHING here is an estimate derived from the recorded 270M numbers of paper/sections/15_appendix.md Appendix F. Output: cost_scaling.csv, cost_assumptions.json.

Formula.  P_eff = non-embedding params + head params (262,144 x hidden; the head is tied to the input embedding but is still a matmul per token):
    270M: 100.3 M + 167.8 M = 268.1 M;  1B: 697.9 M + 302.0 M = 999.9 M (ratio 3.73);  4B text: 3,209 M + 671 M = 3,880 M (ratio 14.5)
  Training FLOPs/token ~ 6 P_eff (+ attention, ignored), decode FLOPs/token ~ 2 P_eff.
  Because the 270M run uses the GPU poorly (effective ~33 TFLOP/s by 6 N D, 640-wide model on 188 SMs, see reports/gemma_rl_cost_estimate_2026-10-17.md), step time grows LESS than
  linearly in P_eff for small ratios: multiplier m = low / central / high = 1.8 / 2.7 / 3.7 (1B) and 5 / 9 / 14.5 (4B). High = pure FLOP ratio.
  RL step time T = T_fixed + (T - T_fixed) * m, with T_fixed = reward models (AfriCOMET-QE, SSA-COMET), LID, consensus, weight sync, CPU: phi = 20% / 12% / 5% of the step (low / central / high cost).
  4B memory: fp32 policy + grads + AdamW = 16 B/param = 62 GB + fp32 reference 15.5 GB + vLLM engine > 96 GB: needs a recipe change (bf16 reference, 8-bit optimiser) or a second GPU: factor 1.0 / 1.15 / 1.3.
  Steps (tokens, RL steps) are assumed equal to the 270M run; a larger model may need fewer or more.
"""
import os, sys, json
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from ge_ez_step0_common import OUT

RATE = 1.19
EMB = 262144
cfg = {"270M": dict(h=640, L=18, ffn=2048, total=268_098_176), "1B": dict(h=1152, L=26, ffn=6912, total=999_885_952)}
cfg["270M"]["emb"] = EMB * 640; cfg["1B"]["emb"] = EMB * 1152
cfg["4B"] = dict(h=2560, L=34, ffn=10240, total=None, emb=675e6, nonemb=3209e6)
for k in ("270M", "1B"):
    cfg[k]["nonemb"] = cfg[k]["total"] - cfg[k]["emb"]
for k, v in cfg.items():
    v["p_eff"] = v["nonemb"] + v["emb"]            # transformer + head (tied embedding is also the head)
    v["ratio_vs_270M"] = v["p_eff"] / (cfg["270M"]["nonemb"] + cfg["270M"]["emb"])
M = {"1B": dict(low=1.8, central=2.7, high=3.7), "4B": dict(low=5.0, central=9.0, high=14.5)}
PHI = dict(low=0.20, central=0.12, high=0.05)
MEMF = {"1B": dict(low=1.0, central=1.0, high=1.0), "4B": dict(low=1.0, central=1.15, high=1.3)}
SC = ("low", "central", "high")
def mult_train(size, sc):
    return 1.0 if size == "270M" else M[size][sc] * MEMF[size][sc]
def mult_rl(size, sc):
    return 1.0 if size == "270M" else (PHI[sc] + (1 - PHI[sc]) * M[size][sc]) * MEMF[size][sc]

# recorded 270M quantities (hours)
SFT_H = 27638 * 6.4 / 3600          # 49.1 h
RL2C_H, FINAL_H = 40.5, 28.4
SCI_H = dict(low=0.3, central=0.5, high=0.6)
# RL phases 1, 2, 2b: wall-clock NOT recorded. Bounds from sampled completions / generated tokens (t_rl_budget): proportional to generated tokens -> low; to completions -> high
gen_b = dict(p1=0.055, p2=0.106, p2b=0.037, p2c=0.665); comp = dict(p1=110720, p2=249792, p2b=499712, p2c=900384)
unrec_low = RL2C_H * (gen_b["p1"] + gen_b["p2"] + gen_b["p2b"]) / gen_b["p2c"]
unrec_high = RL2C_H * (comp["p1"] + comp["p2"] + comp["p2b"]) / comp["p2c"]
UNREC = dict(low=unrec_low, central=(unrec_low + unrec_high) / 2 * 0.7, high=unrec_high)   # central a little below the midpoint: b64 steps are short
rows = []
def add(design, item, size, sc, hours, note=""):
    rows.append(dict(design=design, item=item, size=size, scenario=sc, gpu_hours=round(hours, 1), usd=round(hours * RATE, 0), note=note))

# (ii) full pipeline
for size in ("270M", "1B", "4B"):
    for sc in SC:
        sft = SFT_H * mult_train(size, sc)
        rl_unrec = UNREC[sc] * mult_rl(size, sc)
        rl2c = RL2C_H * mult_rl(size, sc)
        fin = FINAL_H * mult_rl(size, sc)
        sci = SCI_H[sc] * mult_train(size, sc)
        for it, h in (("SFT 3.62B tokens", sft), ("RL phases 1,2,2b (unrecorded)", rl_unrec), ("RL phase 2c", rl2c), ("science SFT stage", sci), ("final science RL run", fin)):
            add("full pipeline", it, size, sc, h)
        add("full pipeline", "TOTAL", size, sc, sft + rl_unrec + rl2c + sci + fin)

# (i) reduced Ge'ez diagnostic (proposal Step 1). tokens per arm; step = 131,072 tokens; seconds per step 3.3 (AB2 measured: 367 steps in ~20 min) .. 6.4 (Appendix F)
TOK_STEP = 131072
def sft_hours(tokens, size, sc, sec_per_step):
    return tokens / TOK_STEP * sec_per_step / 3600 * mult_train(size, sc)
arms_r = [("A1 Ge'ez specialist (amh+tir, 28M tokens)", 28e6, "270M"), ("A1f Fon specialist (50M tokens)", 50e6, "270M"),
          ("A2 upweighted multilingual continuation (50M)", 50e6, "270M"), ("A0c control continuation, same schedule (50M)", 50e6, "270M")]
for sc, sps in (("low", 3.3), ("central", 4.8), ("high", 6.4)):
    tot = 0
    for name, tok, size in arms_r:
        h = sft_hours(tok, size, sc, sps); tot += h; add("Step 1 reduced (continuation arms)", name, size, sc, h)
    ev = 4 * 0.17; tot += ev; add("Step 1 reduced (continuation arms)", "evaluation of 4 arms (10 min each)", "270M", sc, ev)
    # 1B arm, reduced mixture of 0.5B tokens from the base (A3-reduced) and the full recipe (A3-full)
    for size in ("1B", "4B"):
        h = sft_hours(0.5e9, size, sc, sps); add("Step 1 reduced (continuation arms)", "A3 multilingual arm, reduced mixture 0.5B tokens", size, sc, h)
        add("Step 1 reduced (continuation arms)", "TOTAL with A3-reduced", size, sc, tot + h + 0.17 * mult_train(size, sc))
        h2 = SFT_H * mult_train(size, sc) * (sps / 6.4); add("Step 1 reduced (continuation arms)", "A3 multilingual arm, FULL recipe 3.62B tokens", size, sc, h2)
        add("Step 1 reduced (continuation arms)", "TOTAL with A3-full", size, sc, tot + h2 + 0.17 * mult_train(size, sc))
    add("Step 1 reduced (continuation arms)", "TOTAL 270M arms only (no A3)", "270M", sc, tot)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "cost_scaling.csv"), index=False)
json.dump(dict(params=cfg, multipliers_train=M, phi=PHI, mem_factor_4B=MEMF["4B"], rate_usd_per_h=RATE, sft_hours_270M=SFT_H,
               rl_unrecorded_hours=UNREC, sci_hours=SCI_H), open(os.path.join(OUT, "cost_assumptions.json"), "w"), indent=1, default=float)
pd.set_option("display.width", 200)
print({k: (round(v["p_eff"] / 1e6, 1), round(v["ratio_vs_270M"], 2)) for k, v in cfg.items()}); print(UNREC)
t = df[df.item.str.startswith("TOTAL")].pivot_table(index=["design", "item", "size"], columns="scenario", values=["gpu_hours", "usd"])
print(t.to_string())
print(df[(df.design == "full pipeline") & (df.size != "270M")].pivot_table(index=["size", "item"], columns="scenario", values="gpu_hours").to_string())
