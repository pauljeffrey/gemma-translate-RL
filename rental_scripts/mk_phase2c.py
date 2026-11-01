import os
import re, yaml
src = "/workspace/sft_build/gemma_mt_2b/configs/rl_rtxpro_phase2b.yaml"
dst = "/workspace/sft_build/gemma_mt_2c/configs/rl_rtxpro_phase2c.yaml"
s = open(src).read()
G = 16
PPS = {"b256": 48, "b512": 42, "b1024": 42, "mixed": 42}
TARGET = {"b256": 300_000, "b512": 300_000, "b1024": 100_000, "mixed": 100_000}
STEPS = {b: round(TARGET[b] / (PPS[b] * G)) for b in TARGET}
total = sum(STEPS.values())
def setkey(s, key, val):
    pat = re.compile(r"^" + re.escape(key) + r":[^\n]*", re.M)
    assert pat.search(s), key
    return pat.sub(lambda m: f"{key}: {val}", s, count=1)
s = setkey(s, "model_name", "/workspace/sft_build/runs/rl_2b_stage_b64")
s = setkey(s, "out_dir", "/workspace/sft_build/runs/rl2c")
s = setkey(s, "steps", str(total))
s = re.sub(r"^greedy_in_group:[^\n]*\n", "", s, flags=re.M)
s = setkey(s, "advantage_norm", "mean_std")
s = setkey(s, "learning_rate", "2.0e-6")
s = setkey(s, "hub_repo_id", (os.environ['HF_ORG'] + "/gemma-3-270m-translate-rl2c"))
s = setkey(s, "archive_path", "/workspace/sft_build/runs/rl2c/archive/archive.sqlite")
s = re.sub(r"(curriculum_steps:[^\n]*\n)  b64: \d+\n  b256: \d+\n  b512: \d+\n  b1024: \d+\n",
           lambda m: m.group(1) + "".join(f"  {b}: {STEPS[b]}\n" for b in ("b256", "b512", "b1024")), s, count=1)
s = s.replace("learning_rate: 2.0e-6", "learning_rate: 2.0e-6\nkl_guard_threshold: 0.15            # KL guard: mean KL of the last 5 steps above this halves the learning rate (min 1/8), at most once per 10 steps\nkl_guard_factor: 0.5\nkl_guard_min_scale: 0.125", 1)
hdr = ("# PHASE 2c of the Gemma RL (owner 2026-10-20): phase 2b's b256 stage diverged (KL 0.04 -> 2.3 from step ~500, greedy eval reward 0.24 -> -0.5). Restart from the 2b stage-b64 model (step 488, dev chrF++ ~41.9-42.0).\n"
       f"# Remaining completions: b256 300k, b512 300k, b1024 100k, mixed 100k = {sum(STEPS[b]*PPS[b]*G for b in STEPS):,} in {total} steps. Group 16 (all sampled), advantage std-normalised (as phase 2), lr 2e-6, KL guard on.\n")
open(dst, "w").write(hdr + s)
c = yaml.safe_load(open(dst))
print({k: c.get(k) for k in ("model_name", "out_dir", "steps", "group_size", "greedy_in_group", "advantage_norm", "learning_rate", "kl_guard_threshold", "hub_repo_id", "archive_path")})
print(c["curriculum_steps"], c["prompts_per_step_by_bin"], STEPS)
