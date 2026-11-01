#!/usr/bin/env python3
"""tokswap_spm_check.py -- the edited SentencePiece tokenizer.model must give the same ids as the edited tokenizer.json.
For each variant dir (contains tokenizer.json + tokenizer.model) and FLORES+ devtest sets: fraction of sentences with identical id lists.
Usage: tokswap_spm_check.py --orig-json tok_orig/tokenizer.json --orig-spm spm/tokenizer.model --flores flores --variants name=dir ... --out spm_check.json"""
import argparse, json, os
import sentencepiece as spm
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--orig-json", required=True); ap.add_argument("--orig-spm", required=True); ap.add_argument("--flores", required=True)
ap.add_argument("--variants", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--langs", nargs="+", default=["amh_Ethi", "tir_Ethi", "fon_Latn", "eng_Latn", "hau_Latn", "yor_Latn", "swh_Latn", "kin_Latn", "ibo_Latn", "fra_Latn", "ewe_Latn", "bam_Latn"])
a = ap.parse_args()
res = {}
def texts(split, lg):
    p = os.path.join(a.flores, split, lg + ".jsonl")
    return [json.loads(l)["text"] for l in open(p, encoding="utf-8")] if os.path.exists(p) else []
sets = {(sp, lg): texts(sp, lg) for sp in ("dev", "devtest") for lg in a.langs}
sets = {k: v for k, v in sets.items() if v}
def compare(name, js, sp):
    tk = Tokenizer.from_file(js); s = spm.SentencePieceProcessor(model_file=sp)
    out = {}
    for (split, lg), T in sets.items():
        j = [e.ids for e in tk.encode_batch(T, add_special_tokens=False)]
        p = [s.encode(t) for t in T]
        out[f"{split}/{lg}"] = {"n": len(T), "identical": sum(1 for x, y in zip(j, p) if x == y), "tokens_json": sum(map(len, j)), "tokens_spm": sum(map(len, p))}
    res[name] = out
    print(name, {k: f"{v['identical']}/{v['n']}" for k, v in out.items() if split in k and 'devtest' in k}, flush=True)
compare("original", a.orig_json, a.orig_spm)
for v in a.variants:
    name, d = v.split("=", 1)
    compare(name, os.path.join(d, "tokenizer.json"), os.path.join(d, "tokenizer.model"))
json.dump(res, open(a.out, "w"), indent=1)
