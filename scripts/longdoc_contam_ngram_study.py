# SPDX-License-Identifier: AGPL-3.0-or-later
"""Token n-gram length study (rental paths): recovery of 300 known-positive SFT windows and false positives on 80 Gutenberg documents, for n = 6..32. Result: contam/exp_n.json.
Needs the rental data (gemma/packed_fwd, raw/contam_neg); run with PYTHONPATH=<code>:<pylibs>."""
import sys, glob, json, re, time, os
import numpy as np
sys.path.insert(0, "/workspace/sft_build/longdoc_bench/code")
from gemma_mt.eval import longdoc_contam as L
from tokenizers import Tokenizer
tk = Tokenizer.from_file(glob.glob("/workspace/sft_build/hf/hub/models--google--gemma-3-270m-it/snapshots/*/tokenizer.json")[0])
D = "/workspace/sft_build/gemma/packed_fwd/"
rng = np.random.RandomState(0)
split = L.PackedSplit(D + "train")
# positive controls: 300 random samples
idx = np.sort(rng.choice(split.n_samples, 300, replace=False))
pos_tok, pos_txt = [], []
for i in idx:
    o = int(split.sample_offset[i]); l = int(split.sample_len[i]); p = int(split.prompt_len[i])
    slab = np.asarray(split.tokens[o:o + l])
    s, e, v = L.english_bounds(slab, np.array([0]), np.array([p]))
    assert v[0]
    t = slab[s[0]:e[0]].astype(np.int64)
    pos_tok.append(t)
    pos_txt.append(tk.decode(t.tolist(), skip_special_tokens=False))
same = 0; reenc = []
for t, x in zip(pos_tok, pos_txt):
    r = tk.encode(x, add_special_tokens=False).ids
    reenc.append(np.asarray(r))
    same += (list(t) == r)
print("re-encoding identical to in-stream tokens:", same, "/ 300", flush=True)
# how do they differ?
k = 0
for t, r, x in zip(pos_tok, reenc, pos_txt):
    if list(t) != list(r) and k < 3:
        k += 1
        j = next((q for q in range(min(len(t), len(r))) if t[q] != r[q]), min(len(t), len(r)))
        print("diff at", j, len(t), len(r), repr(tk.decode(t[max(0, j - 3):j + 4].tolist())), repr(tk.decode(r[max(0, j - 3):j + 4].tolist())))
# negative controls: gutenberg
neg_docs = []
for f in sorted(glob.glob("/workspace/sft_build/longdoc_bench/raw/contam_neg/pg*.txt")):
    t = open(f, encoding="utf-8", errors="replace").read()
    a = t.find("*** START"); b = t.find("*** END")
    if a >= 0: t = t[t.find("\n", a) + 1:]
    if b >= 0 and b > a: t = t[: t.rfind("*** END")] if "*** END" in t else t
    t = " ".join(t.split())
    w = t.split(" ")
    # 4 docs of 1000 words per book from the middle
    for q in range(4):
        s = len(w) // 5 * (q + 1)
        neg_docs.append(" ".join(w[s:s + 1000]))
neg_tok = [np.asarray(e.ids) for e in tk.encode_batch(neg_docs, add_special_tokens=False)]
print("neg docs", len(neg_tok), "tokens", sum(map(len, neg_tok)), flush=True)
ns = [6, 8, 10, 12, 16, 20, 24, 32]
targets = {}
for n in ns:
    hs = [L.rolling_hashes(L.token_ids64(t), n) for t in pos_tok_re] if False else None
pos_re = reenc
for n in ns:
    hp = np.concatenate([L.rolling_hashes(L.token_ids64(t), n) for t in pos_re])
    hn = np.concatenate([L.rolling_hashes(L.token_ids64(t), n) for t in neg_tok])
    targets[n] = np.unique(np.concatenate([hp, hn]))
t0 = time.time()
hits, stats = L.scan_sft_stream([D + "train", D + "val"], targets, processes=int(sys.argv[1]) if len(sys.argv) > 1 else 12, max_samples=None)
print(stats, flush=True)
res = {}
for n in ns:
    hp = [L.membership(targets[n][hits[n].astype(bool)], L.rolling_hashes(L.token_ids64(t), n)) for t in pos_re]
    hn = [L.membership(targets[n][hits[n].astype(bool)], L.rolling_hashes(L.token_ids64(t), n)) for t in neg_tok]
    pshare = np.array([h.mean() if h.size else np.nan for h in hp]); nshare = np.array([h.mean() if h.size else np.nan for h in hn])
    allp = np.concatenate(hp); alln = np.concatenate(hn)
    res[n] = dict(pos_recovery_ngrams=float(allp.mean()), pos_docs_share_q05=float(np.nanquantile(pshare, 0.05)), pos_min=float(np.nanmin(pshare)), neg_fp_ngrams=float(alln.mean()),
                  neg_share_max=float(np.nanmax(nshare)), neg_share_q99=float(np.nanquantile(nshare, 0.99)), neg_docs_with_any=int((nshare > 0).sum()))
    print(n, res[n], flush=True)
json.dump(res, open("/workspace/sft_build/longdoc_bench/contam/exp_n.json", "w"), indent=1)
