"""Check that the same N new tokens, written into the SentencePiece model (tokenizer.model, the file llama.cpp's converter reads for Gemma 3) as NORMAL pieces with scores below
every existing piece, give the same ids as the tokenizer.json route. N = 3000 of the joint Ge'ez variant. Output: spm_check.json"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ge_ez_step0_common import *
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb
from tokenizers import Tokenizer
def find(repo, fn): return glob.glob(f"{CACHE}/models--{repo.replace('/','--')}/snapshots/*/{fn}")[0]
def flores(split, code):
    return [json.loads(l)["text"] for l in open(glob.glob(f"{CACHE}/datasets--openlanguagedata--flores_plus/snapshots/*/{split}/{code}.jsonl")[0], encoding="utf-8")]
N = 3000
ent = [l.rstrip("\n").split("\t", 1) for l in open(os.path.join(OUT, "swap_sim_new_tokens_geez_amh_tir_.txt"), encoding="utf-8")][:N]
sp_path = find("google/gemma-3-1b-it", "tokenizer.model")
proto = pb.ModelProto(); proto.ParseFromString(open(sp_path, "rb").read())
minscore = min(p.score for p in proto.pieces)
for k, (i, s) in enumerate(ent):
    pc = proto.pieces[int(i)]
    assert pc.piece.startswith("<unused") and pc.type == 4
    pc.piece = s; pc.type = 1; pc.score = minscore - 1 - k
new_path = os.path.join(CACHE, "tokenizer_geez3000.model")
open(new_path, "wb").write(proto.SerializeToString())
sp0 = spm.SentencePieceProcessor(model_file=sp_path); sp1 = spm.SentencePieceProcessor(model_file=new_path)
# the tokenizer.json route with the same N entries: rebuild from the merges file (chars first, then merges), same ids
M = json.load(open(os.path.join(OUT, "swap_sim_merges_geez_amh_tir_.json"), encoding="utf-8"))
base = json.load(open(find((os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), "latest/tokenizer.json"), encoding="utf-8"))
inv = {i: s for s, i in base["model"]["vocab"].items()}
chars, merges = M["chars"], M["merges"]
nch = min(N, len(chars)); nm = N - nch
v = dict(base["model"]["vocab"])
fr = set()
for k, (i, s) in enumerate(ent):
    i = int(i); v.pop(inv[i]); v[s] = i; fr.add(i)
assert [s for _, s in ent] == chars[:nch] + [a + b for a, b in merges[:nm]]
j = dict(base); m = dict(base["model"]); m["vocab"] = v; m["merges"] = base["model"]["merges"] + merges[:nm]; j["model"] = m
j["added_tokens"] = [t for t in base["added_tokens"] if t["id"] not in fr]
tk = Tokenizer.from_str(json.dumps(j, ensure_ascii=False))
res = {}
for lang, code in (("amh", "amh_Ethi"), ("tir", "tir_Ethi"), ("eng", "eng_Latn"), ("hau", "hau_Latn")):
    T = flores("devtest", code)
    a = [sp1.encode(t) for t in T]; b = [e.ids for e in tk.encode_batch(T, add_special_tokens=False)]; a0 = [sp0.encode(t) for t in T]
    b0 = [e.ids for e in Tokenizer.from_str(json.dumps(base, ensure_ascii=False)).encode_batch(T, add_special_tokens=False)]
    res[lang] = dict(n=len(T), base_spm_equals_base_json=float(np.mean([x == y for x, y in zip(a0, b0)])), modified_spm_equals_modified_json=float(np.mean([x == y for x, y in zip(a, b)])),
                     tokens_per_sent_base=float(np.mean([len(x) for x in b0])), tokens_per_sent_json=float(np.mean([len(x) for x in b])), tokens_per_sent_spm=float(np.mean([len(x) for x in a])))
json.dump(res, open(os.path.join(OUT, "spm_check.json"), "w"), indent=1); print(json.dumps(res, indent=1))
