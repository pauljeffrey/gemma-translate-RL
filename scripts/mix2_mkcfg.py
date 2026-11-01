import sys, json, re, math
# usage: mkcfg.py VARIANT DATA_DIR OUT_CFG
var, data, out = sys.argv[1:4]
meta = json.load(open(data + "/meta.json"))
tr = meta["splits"]["train"]; mv = meta["mixture_v2"]
steps = math.ceil(tr["n_tokens"] / 65536)
txt = open("/workspace/jobs/r11/code/configs/sft_science.yaml").read()
txt = txt.replace("data_dir: /workspace/jobs/r4/mix_v1", f"data_dir: {data}")
txt = txt.replace("out_dir: /workspace/jobs/r4/runs/sci_sft", f"out_dir: /workspace/jobs/r11/runs/sci_sft_v2{var}")
txt = txt.replace("run_name: sci_sft", f"run_name: sci_sft_v2{var}")
sh = mv["share_by_source"]
import os
if os.path.isdir(data + "/val_extra"):
    txt = txt.replace("extra_val_splits: [val_sci]", "extra_val_splits: [val_sci, val_extra]")
hdr = (f"# MIXTURE v2 variant {var} (agent R11, 2026-10-22): copy of configs/sft_science.yaml with the data paths changed; every hyper-parameter is R4's (lr 3e-6, 65,536 tokens / step, cosine, 6 % warm-up, no weight decay).\n"
       f"# Data {data}: {tr['n_samples']:,} samples, {tr['n_packs']:,} packs of {meta['seq_len']}, {tr['n_tokens']:,} tokens -> {steps} optimizer steps at 65,536 tokens (epochs: 1.0; was 293 for mix_v1). Token shares: "
       + ", ".join(f"{k} {v:.3f}" for k, v in sh.items() if v) + ".\n"
       "# R7's copy of sft_science.yaml differs from R4's in two lines (eval_every 50, gradient_checkpointing false on an empty 96 GB card): apply them on the command line if wanted: eval_every=50 gradient_checkpointing=false.\n")
txt = hdr + txt
open(out, "w").write(txt)
print(steps)
