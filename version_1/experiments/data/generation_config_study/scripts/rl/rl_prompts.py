"""R12 RL part 1 (venv_gemma): pick the RL prompts (rl_prompts_v3, the 20 languages of the sentence study) -> rl/work/prompts.json with prompt ids."""
import json, os, random, sys
sys.path.insert(0, "/workspace/sft_build/pylibs")
import pandas as pd
from transformers import AutoTokenizer
from gemma_mt import format as fmt
R = "/workspace/jobs/r12/rl"; os.makedirs(R + "/work", exist_ok=True)
LANGS = "swh hau yor ibo zul amh kin som wol sna lug bem tir kbp nus kab fon ach nbl efi".split()
PER = {"b64": 10, "b256": 4, "b1024": 2}
tok = AutoTokenizer.from_pretrained(("/workspace/sft_build/hf/hub/models--" + os.environ['HF_ORG_LEGACY'] + "--gemma-3-270m-translate-rl2c/snapshots/80baefff3b16b120b3e37e70afeb51f1fc248d96"))
out = []
for iso in LANGS:
    p = f"/workspace/sft_build/gemma/rl_prompts_v3/{iso}.parquet"
    if not os.path.exists(p): print("missing", iso); continue
    d = pd.read_parquet(p)
    for b, n in PER.items():
        sub = d[d.length_bin == b]
        sub = sub.sample(n=min(n, len(sub)), random_state=12 + sum(map(ord, iso)))
        for _, r in sub.iterrows():
            out.append({"pid": f"{iso}|{b}|{len(out)}", "iso": iso, "bin": b, "english": r["english"], "ids": fmt.build_prompt(r["english"], iso, tok), "src_afr_tokens": int(r["src_afr_tokens"])})
json.dump(out, open(R + "/work/prompts.json", "w"), ensure_ascii=False)
import collections; print(len(out), collections.Counter(o["bin"] for o in out), max(len(o["ids"]) for o in out))
