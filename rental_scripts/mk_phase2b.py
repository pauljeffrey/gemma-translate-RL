import os
import re, yaml
src = "/workspace/sft_build/gemma_mt/configs/rl_rtxpro_phase2.yaml"
dst = "/workspace/sft_build/gemma_mt_2b/configs/rl_rtxpro_phase2b.yaml"
s = open(src).read()
# completions per phase (owner 2026-10-19): b64 500k, b256 300k, b512 300k, b1024 100k, mixed 100k; group 16 (15 sampled + 1 greedy)
G = 16
PPS = {"b64": 64, "b256": 48, "b512": 42, "b1024": 42, "mixed": 42}          # prompts per step -> 1024 / 768 / 672 / 672 / 672 completions per step
TARGET = {"b64": 500_000, "b256": 300_000, "b512": 300_000, "b1024": 100_000, "mixed": 100_000}
STEPS = {b: round(TARGET[b] / (PPS[b] * G)) for b in TARGET}
def setkey(s, key, val):
    pat = re.compile(r"^" + re.escape(key) + r":[^\n]*", re.M)
    assert pat.search(s), key
    return pat.sub(lambda m: f"{key}: {val}", s, count=1)
total = sum(STEPS.values())
s = setkey(s, "model_name", "/workspace/sft_build/runs/rl_phase2_final")
s = setkey(s, "out_dir", "/workspace/sft_build/runs/rl2b")
s = setkey(s, "steps", str(total))
s = setkey(s, "group_size", str(G))
s = s.replace("group_size: 16\n", "group_size: 16\ngreedy_in_group: true                # the LAST member of each group is the greedy completion (15 sampled + 1 greedy)\n", 1)
s = setkey(s, "advantage_norm", "mean                    # Dr.GRPO style: A = r - group mean (no division by the group std)")
s = setkey(s, "learning_rate", "3.0e-6")
s = setkey(s, "hub_repo_id", (os.environ['HF_ORG'] + "/gemma-3-270m-translate-rl2b"))
s = setkey(s, "archive_path", "/workspace/sft_build/runs/rl2b/archive/archive.sqlite")
s = re.sub(r"(curriculum_steps:[^\n]*\n)  b64: \d+\n  b256: \d+\n  b512: \d+\n  b1024: \d+\n",
           lambda m: m.group(1) + "".join(f"  {b}: {STEPS[b]}\n" for b in ("b64", "b256", "b512", "b1024")), s, count=1)
s = re.sub(r"(prompts_per_step_by_bin:[^\n]*\n)  b64: \d+\n  b256: \d+\n  b512: \d+\n  b1024: \d+\n  mixed: \d+\n",
           lambda m: m.group(1) + "".join(f"  {b}: {PPS[b]}\n" for b in ("b64", "b256", "b512", "b1024", "mixed")), s, count=1)
hdr = ("# PHASE 2b of the Gemma RL (owner 2026-10-19: the model does not see enough samples per language). Starts from the FINAL model of phase 2 (branch stage-mixed of the rl2 repo).\n"
       f"# Completions per phase: b64 500k, b256 300k, b512 300k, b1024 100k, mixed 100k = {sum(STEPS[b]*PPS[b]*G for b in STEPS):,} in {total} steps. Changes against phase 2: group 16 (15 sampled + 1 greedy),\n"
       "# advantage = reward - group mean (no std division), lr 3e-6. Same reward, same language draws (alpha 0, x2.5 on the 17 weakest).\n")
open(dst, "w").write(hdr + s)
c = yaml.safe_load(open(dst))
print({k: c[k] for k in ("model_name", "out_dir", "steps", "group_size", "greedy_in_group", "advantage_norm", "learning_rate", "hub_repo_id", "archive_path")})
print(c["curriculum_steps"], c["prompts_per_step_by_bin"])
done = {b: STEPS[b] * PPS[b] * G for b in STEPS}; print("completions", done, "mixed steps", total - sum(c["curriculum_steps"].values()))
