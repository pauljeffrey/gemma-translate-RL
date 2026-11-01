#!/usr/bin/env python3
"""tokswap_gguf_check.py -- read back the vocabulary of a GGUF produced by llama.cpp's converter from the swapped tokenizer.model and compare it with
tokenizer.json (pieces at the re-used ids, token types, scores of new pieces below every old piece) and with the ORIGINAL tokenizer.model (all other pieces identical)."""
import argparse, json, sys
import numpy as np
from gguf import GGUFReader
ap = argparse.ArgumentParser(); ap.add_argument("--gguf", required=True); ap.add_argument("--swap", required=True); ap.add_argument("--orig-spm", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
r = GGUFReader(a.gguf)
def field(name):
    f = r.fields[name]
    return f
def strs(f):
    return [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
toks = strs(field("tokenizer.ggml.tokens"))
scores = np.array([float(f) for f in [field("tokenizer.ggml.scores").parts[i][0] for i in field("tokenizer.ggml.scores").data]])
types = np.array([int(field("tokenizer.ggml.token_type").parts[i][0]) for i in field("tokenizer.ggml.token_type").data])
model = bytes(field("tokenizer.ggml.model").parts[field("tokenizer.ggml.model").data[0]]).decode()
smap = json.load(open(a.swap + "/swap_map.json", encoding="utf-8"))
tj = json.load(open(a.swap + "/tokenizer.json", encoding="utf-8"))["model"]["vocab"]
from sentencepiece import sentencepiece_model_pb2 as pb
o = pb.ModelProto(); o.ParseFromString(open(a.orig_spm, "rb").read())
ids = {t["id"]: t["token"] for t in smap["new_tokens"]}
res = {"gguf_vocab_size": len(toks), "tokenizer_model": model,
       "new_pieces_equal_tokenizer_json": sum(1 for i, t in ids.items() if toks[i] == t),
       "n_new": len(ids),
       "new_piece_types": sorted(set(int(types[i]) for i in ids)),
       "min_score_new": float(min(scores[i] for i in ids)), "max_score_new": float(max(scores[i] for i in ids)),
       "min_score_old_pieces": float(min(scores[i] for i in range(len(toks)) if i not in ids)),
       "unchanged_pieces_equal_original_spm": sum(1 for i in range(len(toks)) if i not in ids and toks[i] == o.pieces[i].piece.replace("▁", " ") ) ,
       "n_unchanged": len(toks) - len(ids)}
res["all_tokens_equal_tokenizer_json_ids"] = sum(1 for s, i in tj.items() if i < len(toks) and toks[i].replace(" ", "▁") == s)
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res))
