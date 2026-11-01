#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export the weights of a trainer checkpoint (<run>/last/trainer_state.pt) to a plain Hugging Face directory (model.safetensors + config / tokenizer files).

    python scripts/export_trainer_state.py --state runs/a0u_up6/last/trainer_state.pt --like ckpt/gemma3-270m-up6 --out ckpt/gemma3-270m-up6-a0

`--like` is the HF directory the run was started from (config.json, tokenizer files). The tied output matrix `lm_head.weight` is dropped from the file when it is
equal to the embedding (the checkpoint format of this repo lists it, Hugging Face ties it), so the result has the layout scripts/upcycle_layers.py expects.
"""
import argparse, hashlib, json, os, shutil, sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True); ap.add_argument("--like", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    a = ap.parse_args(argv)
    import torch
    from safetensors.torch import save_file
    ck = torch.load(a.state, map_location="cpu", weights_only=False)
    sd = ck["model"]; step = int(ck["state"]["step"])
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    out = {}
    for k, v in sd.items():
        if k == "lm_head.weight":
            if not torch.equal(v, sd["model.embed_tokens.weight"]):
                raise SystemExit("lm_head.weight differs from the embedding: the checkpoint is not tied, refusing to drop it")
            continue
        out[k] = v.to(torch.float32).to(dt).contiguous()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "model.safetensors")
    save_file(out, path, metadata={"format": "pt"})
    for fn in os.listdir(a.like):
        if fn == "model.safetensors" or fn == "upcycle.json" or os.path.isdir(os.path.join(a.like, fn)):
            continue
        shutil.copyfile(os.path.join(a.like, fn), os.path.join(a.out, fn))
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    info = {"source_state": os.path.abspath(a.state), "step": step, "tensors": len(out), "parameters": int(sum(v.numel() for v in out.values())),
            "dtype": a.dtype, "model_safetensors_sha256": h.hexdigest(), "like": os.path.abspath(a.like)}
    json.dump(info, open(os.path.join(a.out, "export.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    sys.exit(main())
