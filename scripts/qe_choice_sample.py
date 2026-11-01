#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-53 step 2: GRPO-like policy samples on FLORES+ DEV prompts.

    python qe_choice_sample.py --repo <HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c --revision latest --out samples.jsonl \
        --n-prompts 80 --group 8 --temperature 1.0 [--languages yor hau ...]

For every language of `gemma_mt.languages.SUPPORTED_RL` that has a FLORES+ dev file (`openlanguagedata/flores_plus`, dev/<code>.jsonl, read token from
HF_TOKEN) the SAME seeded subset of FLORES sentence ids (English = eng_Latn) is used as prompts; each prompt is sampled `group` times (T=1, top_p=1,
top_k=0 = the RL configuration, HF generate in bf16) with the production prompt format (gemma_mt.format). Row: iso, id, k, src, ref, hyp, raw,
finished, prefix_status. Dev only (D-24: calibration and metric studies never touch devtest). Twi has two FLORES+ varieties; the sentences alternate by
id parity as in build_flores_dev_all.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def flores_dev(code: str):
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("openlanguagedata/flores_plus", f"dev/{code}.jsonl", repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    return read_jsonl(p)


def load_refs(iso: str, code: str):
    if code == "twi_Latn":
        ak, asn = flores_dev("twi_Latn_akua1239"), flores_dev("twi_Latn_asan1239")
        by = {}
        for r in ak:
            by.setdefault(int(r["id"]), {})[0] = r["text"]
        for r in asn:
            by.setdefault(int(r["id"]), {})[1] = r["text"]
        return {i: (d.get(i % 2) or d.get(1 - i % 2)) for i, d in by.items()}
    return {int(r["id"]): r["text"] for r in flores_dev(code)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=(os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"))
    ap.add_argument("--revision", default="latest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=80)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--prompt-batch", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--languages", nargs="*")
    ap.add_argument("--bench-dev", default="/workspace/sft_build/bench/bench_dev.jsonl", help="fallback human references (MAFAND dev rows) for languages without FLORES+ (pcm)")
    a = ap.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    from gemma_mt import format as fmt
    from gemma_mt.eval.runtime import load_tokenizer_and_stops
    from gemma_mt.languages import LANGUAGES, SUPPORTED_RL
    from gemma_mt.rl.text import decode_completion

    path = snapshot_download(a.repo, revision=a.revision, token=os.environ.get("HF_TOKEN"),
                             allow_patterns=["config.json", "generation_config.json", "model.safetensors", "model-*.safetensors", "model.safetensors.index.json",
                                             "tokenizer*", "special_tokens_map.json", "added_tokens.json", "chat_template*"])
    tok, stop, pad = load_tokenizer_and_stops(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to("cuda").eval()
    sha = os.path.basename(os.path.normpath(path))
    print("model", a.repo, a.revision, sha, flush=True)
    en = {int(r["id"]): r["text"] for r in flores_dev("eng_Latn")}
    ids_all = sorted(en)
    chosen = sorted(random.Random(a.seed).sample(ids_all, a.n_prompts))     # the same sentence ids for every language
    isos = a.languages or list(SUPPORTED_RL)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["iso"] for l in open(a.out, encoding="utf-8")}
    t0 = time.time()
    for iso in isos:
        code = LANGUAGES[iso].flores
        if iso in done:
            continue
        refs = None
        if code:
            try:
                refs = load_refs(iso, code)
            except Exception as e:      # no FLORES+ dev file for this language
                print(f"{iso}: no FLORES+ dev ({type(e).__name__})", flush=True)
        if refs is not None:
            prompts = [(i, en[i], refs[i]) for i in chosen if i in refs and refs[i] and refs[i].strip()]
        else:
            mf = [r for r in read_jsonl(a.bench_dev) if r.get("iso") == iso and r.get("benchmark") == "mafand_dev"] if os.path.exists(a.bench_dev) else []
            if not mf:
                continue
            mf = random.Random(a.seed).sample(mf, min(a.n_prompts, len(mf)))
            prompts = [(int(str(r["id"]).split("-")[-1]) if str(r["id"]).split("-")[-1].isdigit() else k, r["english"], r["reference"]) for k, r in enumerate(mf)]
            print(f"{iso}: MAFAND dev fallback, {len(prompts)} prompts", flush=True)
        rows = []
        for s in range(0, len(prompts), a.prompt_batch):
            chunk = prompts[s: s + a.prompt_batch]
            toks = [fmt.build_prompt(e, iso, tok) for _, e, _ in chunk]
            width = max(len(t) for t in toks)
            ids = torch.full((len(toks), width), pad, dtype=torch.long)
            att = torch.zeros_like(ids)
            for j, t in enumerate(toks):
                ids[j, width - len(t):] = torch.tensor(t)
                att[j, width - len(t):] = 1
            with torch.no_grad():
                gen = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), do_sample=True, temperature=a.temperature, top_p=1.0, top_k=0,
                                     num_return_sequences=a.group, max_new_tokens=a.max_new_tokens, eos_token_id=list(stop.ids), pad_token_id=pad)
            for j, (i, e, r) in enumerate(chunk):
                for k in range(a.group):
                    comp = gen[j * a.group + k, width:].tolist()
                    dec = decode_completion(tok, comp, stop.ids, pad_ids=[pad])
                    st = fmt.strip_iso_prefix(dec.text, iso)
                    rows.append({"iso": iso, "id": i, "k": k, "src": e, "ref": r, "hyp": st.text, "raw": dec.text, "finished": dec.finished,
                                 "prefix_status": st.status, "n_tokens": dec.n_tokens})
        with open(a.out, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{iso}: {len(prompts)} prompts x {a.group} -> {len(rows)} samples  [{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
