#!/usr/bin/env python3
"""tokswap_hf_check.py -- the swapped tokenizer loads through transformers AutoTokenizer (PreTrainedTokenizerFast), keeps the chat template and special-token ids,
and round-trips FLORES+ text (decode(encode(x)) == x).  Usage: tokswap_hf_check.py --orig DIR --swapped DIR [DIR ...] --flores DIR --out JSON"""
import argparse, json, os
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--orig", required=True); ap.add_argument("--swapped", nargs="+", required=True); ap.add_argument("--flores", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
to = AutoTokenizer.from_pretrained(a.orig)
SPECIAL = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<start_of_turn>", "<end_of_turn>", "<unused0>", "<unused98>", "<unused99>", "<unused6241>", "<table>", "<image_soft_token>"]
msgs = [{"role": "user", "content": "Translate the text to Amharic:  Good morning, how are you?"}]
res = {}
def texts(lang, split="dev"):
    return [json.loads(l)["text"] for l in open(os.path.join(a.flores, split, lang + ".jsonl"), encoding="utf-8")]
for d in a.swapped:
    ts = AutoTokenizer.from_pretrained(d)
    r = {"class": type(ts).__name__, "vocab_size_len": len(ts), "special_ids_equal": {s: to.convert_tokens_to_ids(s) == ts.convert_tokens_to_ids(s) for s in SPECIAL},
         "chat_template_equal": to.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) == ts.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
         "chat_template_ids_equal": to.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True) == ts.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True),
         "all_ids_decode_to_string": all(isinstance(ts.convert_ids_to_tokens(i), str) for i in range(0, 262144, 1)),
         "roundtrip": {}}
    for lg in ["amh_Ethi", "tir_Ethi", "fon_Latn", "hau_Latn", "yor_Latn", "eng_Latn", "ewe_Latn"]:
        t = texts(lg)
        ok = sum(1 for x in t if ts.decode(ts.encode(x, add_special_tokens=False), skip_special_tokens=False) == x)
        r["roundtrip"][lg] = f"{ok}/{len(t)}"
    res[os.path.basename(d.rstrip('/'))] = r
    print(os.path.basename(d.rstrip('/')), json.dumps(r)[:600])
json.dump(res, open(a.out, "w"), indent=1)
