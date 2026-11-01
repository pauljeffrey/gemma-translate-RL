"""Q1: tokenizer fertility of the Gemma-3 tokenizer vs NLLB's on FLORES+ devtest (human references).
Output: reports/data/ge_ez_step0_2026-10-25/fertility.csv, fertility_vs_nllb.csv, tokenizer_identity.json.
Tokens are counted WITHOUT bos/eos/language-tag specials (content tokens only); NLLB would add 2 (lang tag + </s>), Gemma 1 (<bos>) for the prompt.
Byte-fallback = tokens whose string is <0xNN>. Whitespace word = str.split()."""
import os, sys, json, glob, re, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ge_ez_step0_common import *
from tokenizers import Tokenizer

def find(repo, fn):
    return glob.glob(f"{CACHE}/models--{repo.replace('/','--')}/snapshots/*/{fn}")[0]
def flores(split, code):
    p = glob.glob(f"{CACHE}/datasets--openlanguagedata--flores_plus/snapshots/*/{split}/{code}.jsonl")[0]
    return [json.loads(l)["text"] for l in open(p, encoding="utf-8")]

gem = Tokenizer.from_file(find((os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), "latest/tokenizer.json"))
nllb = Tokenizer.from_file(find("facebook/nllb-200-distilled-600M", "tokenizer.json"))
# identity of the Gemma-3 vocabularies (270M-it, 1B-it, 4B-it, our rl2c)
ident = {}
vocabs = {}
for name, repo, fn in [("rl2c_270m", (os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), "latest/tokenizer.json"), ("270m-it", "google/gemma-3-270m-it", "tokenizer.json"),
                       ("1b-it", "google/gemma-3-1b-it", "tokenizer.json"), ("4b-it", "google/gemma-3-4b-it", "tokenizer.json")]:
    p = find(repo, fn)
    j = json.load(open(p))
    vocabs[name] = j
    ident[name] = {"sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(), "vocab_size": len(j["model"]["vocab"]), "n_merges": len(j["model"]["merges"]),
                   "vocab_sha256": hashlib.sha256(json.dumps(j["model"]["vocab"], sort_keys=True).encode()).hexdigest()[:16],
                   "merges_sha256": hashlib.sha256(json.dumps(j["model"]["merges"]).encode()).hexdigest()[:16]}
for k in ident:
    ident[k]["vocab_and_merges_equal_to_1b"] = (vocabs[k]["model"]["vocab"] == vocabs["1b-it"]["model"]["vocab"] and vocabs[k]["model"]["merges"] == vocabs["1b-it"]["model"]["merges"])
json.dump(ident, open(os.path.join(OUT, "tokenizer_identity.json"), "w"), indent=1)

bytetok = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
gid2tok = {}
for t, i in vocabs["rl2c_270m"]["model"]["vocab"].items():
    gid2tok[i] = t
is_byte = np.zeros(max(gid2tok) + 1, bool)
for i, t in gid2tok.items():
    is_byte[i] = bool(bytetok.match(t))
nllb_unk = nllb.token_to_id("<unk>")

rows, sent = [], {}
for iso, code in FLORES.items():
    texts = flores("devtest", code)
    g = [e.ids for e in gem.encode_batch(texts, add_special_tokens=False)]
    n = [e.ids for e in nllb.encode_batch(texts, add_special_tokens=False)]
    nw = np.array([len(t.split()) for t in texts]); nc = np.array([len(t) for t in texts])
    gl = np.array([len(x) for x in g]); nl = np.array([len(x) for x in n])
    gb = np.array([int(is_byte[x].sum()) for x in g])
    nu = np.array([sum(1 for y in x if y == nllb_unk) for x in n])
    sent[iso] = (gl, nl)
    rows.append(dict(lang=iso, flores=code, n_sent=len(texts), words=int(nw.sum()), chars=int(nc.sum()),
        gemma_tok_per_word=gl.sum() / nw.sum(), gemma_tok_per_char=gl.sum() / nc.sum(), gemma_byte_fallback_share=gb.sum() / gl.sum(),
        gemma_sents_with_byte_fallback=float((gb > 0).mean()), gemma_mean_tok_per_sent=gl.mean(), gemma_p95_tok_per_sent=float(np.percentile(gl, 95)),
        nllb_tok_per_word=nl.sum() / nw.sum(), nllb_tok_per_char=nl.sum() / nc.sum(), nllb_unk_share=nu.sum() / nl.sum(), nllb_mean_tok_per_sent=nl.mean(),
        nllb_p95_tok_per_sent=float(np.percentile(nl, 95)), gemma_over_nllb=gl.sum() / nl.sum()))
import pandas as pd
df = pd.DataFrame(rows)
rng = np.random.default_rng(0)
lo, hi = [], []
for iso in df.lang:
    gl, nl = sent[iso]
    r = []
    for _ in range(2000):
        ix = rng.integers(0, len(gl), len(gl)); r.append(gl[ix].sum() / nl[ix].sum())
    lo.append(np.percentile(r, 2.5)); hi.append(np.percentile(r, 97.5))
df["gemma_over_nllb_ci_lo"], df["gemma_over_nllb_ci_hi"] = lo, hi
# target-side tokens relative to the English source in the same tokenizer
eg, en_ = sent["eng"]
df["gemma_vs_english_tokens"] = [sent[i][0].sum() / eg.sum() for i in df.lang]
df["nllb_vs_english_tokens"] = [sent[i][1].sum() / en_.sum() for i in df.lang]
df.to_csv(os.path.join(OUT, "fertility.csv"), index=False, float_format="%.4f")
tgt = df[df.lang != "eng"]
print(df.round(3).to_string())
print("mean gemma/nllb over non-English 8:", tgt.gemma_over_nllb.mean(), " Latin controls:", tgt[tgt.lang.isin(["hau","yor","swh","kin","ibo"])].gemma_over_nllb.mean())
