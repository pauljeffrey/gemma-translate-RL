import os
import re
src = "/workspace/sft_build/gemma_mt/configs/rl_rtxpro.yaml"
s = open(src).read()
def setkey(s, key, val, indent=""):
    pat = re.compile(r"^" + indent + re.escape(key) + r":[^\n#]*", re.M)
    assert pat.search(s), key
    return pat.sub(lambda m: f"{indent}{key}: {val}  ", s, count=1)
weak = "tir amh kbp nus kam bam mos umb kmb dyu kea luo kik orm fon yor vmw".split()
w = ", ".join(["efi: 0.0001"] + [f"{l}: 2.5" for l in weak])
s = setkey(s, "model_name", "/workspace/sft_build/runs/rl_phase1_final")
s = setkey(s, "out_dir", "/workspace/sft_build/runs/rl2")
s = setkey(s, "steps", "497")
s = setkey(s, "prompts_per_step", "84")
s = setkey(s, "language_alpha", "0.0")
s = setkey(s, "language_weights", "{" + w + "}")
s = setkey(s, "learning_rate", "2.0e-6")
s = setkey(s, "hub_repo_id", (os.environ['HF_ORG'] + "/gemma-3-270m-translate-rl2"))
s = re.sub(r"(curriculum_steps:[^\n]*\n)  b64: \d+\n  b256: \d+\n  b512: \d+\n  b1024: \d+\n", r"\1  b64: 98\n  b256: 130\n  b512: 75\n  b1024: 45\n", s, count=1)
s = re.sub(r"(prompts_per_step_by_bin:[^\n]*\n)  b64: \d+\n  b256: \d+\n  b512: \d+\n  b1024: \d+\n  mixed: \d+\n", r"\1  b64: 128\n  b256: 96\n  b512: 84\n  b1024: 84\n  mixed: 84\n", s, count=1)
s = re.sub(r"(?<=[0-9A-Za-z_/.\]\}])#(?=\s)", "  #", s)
hdr = ("# PHASE 2 of the Gemma RL (owner 2026-10-19: the first pass saw too few samples per language). Starts from the FINAL model of phase 1 (branch stage-mixed of the rl repo),\n"
       "# ~100k completions per phase for b64 / b256 / mixed, 50k for b512, 30k for b1024 (the long bins are 5-15x slower per completion); uniform language draws (alpha 0) with x2.5 weight\n"
       "# on the 17 weakest languages; lr 2e-6. Same reward as phase 1.\n")
open("/workspace/sft_build/gemma_mt/configs/rl_rtxpro_phase2.yaml", "w").write(hdr + s)
import yaml
c = yaml.safe_load(open("/workspace/sft_build/gemma_mt/configs/rl_rtxpro_phase2.yaml"))
print({k: c[k] for k in ("model_name", "out_dir", "steps", "prompts_per_step", "language_alpha", "learning_rate", "hub_repo_id")})
print(c["curriculum_steps"], c["prompts_per_step_by_bin"], len(c["language_weights"]))
tot = sum(c["curriculum_steps"][b] * c["prompts_per_step_by_bin"][b] for b in c["curriculum_steps"]) + (c["steps"] - sum(c["curriculum_steps"].values())) * c["prompts_per_step_by_bin"]["mixed"]
print("prompts", tot, "completions", tot * c["group_size"])
