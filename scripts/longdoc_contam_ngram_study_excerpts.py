# SPDX-License-Identifier: AGPL-3.0-or-later
"""Token n-gram length study, part 2: 92 word-boundary excerpts (60-300 words) of SFT windows re-tokenised alone -> share of their n-grams found in the stream. Result: contam/exp_n_excerpts.json.
Needs the rental data (gemma/packed_fwd, raw/contam_neg); run with PYTHONPATH=<code>:<pylibs>."""
import sys, glob, json, time, os
import numpy as np
sys.path.insert(0, "/workspace/sft_build/longdoc_bench/code")
from gemma_mt.eval import longdoc_contam as L
from tokenizers import Tokenizer
tk = Tokenizer.from_file(glob.glob("/workspace/sft_build/hf/hub/models--google--gemma-3-270m-it/snapshots/*/tokenizer.json")[0])
D = "/workspace/sft_build/gemma/packed_fwd/"
rng = np.random.RandomState(1)
split = L.PackedSplit(D + "train")
idx = np.sort(rng.choice(split.n_samples, 600, replace=False))
txts = []
for i in idx:
    o = int(split.sample_offset[i]); l = int(split.sample_len[i]); p = int(split.prompt_len[i])
    slab = np.asarray(split.tokens[o:o + l])
    s, e, v = L.english_bounds(slab, np.array([0]), np.array([p]))
    txts.append(tk.decode(slab[s[0]:e[0]].tolist(), skip_special_tokens=False))
# excerpts: cut at word boundaries, keep a random span of 60-300 words if available (a doc quoting part of a window)
ex = []
for t in txts:
    w = t.split(" ")
    if len(w) < 80: continue
    L_ = int(rng.randint(60, min(300, len(w)) + 1)); a = int(rng.randint(0, len(w) - L_ + 1))
    ex.append(" ".join(w[a:a + L_]))
print("excerpts", len(ex), flush=True)
ex_tok = [np.asarray(e.ids) for e in tk.encode_batch(ex, add_special_tokens=False)]
neg_docs = []
for f in sorted(glob.glob("/workspace/sft_build/longdoc_bench/raw/contam_neg/pg*.txt")):
    t = open(f, encoding="utf-8", errors="replace").read()
    a = t.find("*** START"); 
    if a >= 0: t = t[t.find("\n", a) + 1:]
    if "*** END" in t: t = t[: t.rfind("*** END")]
    w = " ".join(t.split()).split(" ")
    for q in range(4):
        s = len(w) // 5 * (q + 1); neg_docs.append(" ".join(w[s:s + 1000]))
neg_tok = [np.asarray(e.ids) for e in tk.encode_batch(neg_docs, add_special_tokens=False)]
ns = [8, 12, 16, 20]
targets = {n: np.unique(np.concatenate([L.rolling_hashes(L.token_ids64(t), n) for t in ex_tok + neg_tok])) for n in ns}
hits, stats = L.scan_sft_stream([D + "train", D + "val"], targets, processes=12)
res = {}
for n in ns:
    seen = targets[n][hits[n].astype(bool)]
    rec = [L.membership(seen, L.rolling_hashes(L.token_ids64(t), n)) for t in ex_tok]
    miss_per = np.array([ (~r).sum() for r in rec]); tot = np.array([len(r) for r in rec])
    res[n] = dict(excerpt_recovery=float(1 - miss_per.sum() / tot.sum()), mean_missed_windows_per_excerpt=float(miss_per.mean()), excerpts=len(ex_tok), excerpts_missing_gt2=int((miss_per > 2).sum()))
    print("n", n, "excerpt recovery (all ngrams)", 1 - miss_per.sum() / tot.sum(), "mean missed ngrams per excerpt", miss_per.mean(), "excerpts with >2 missed", int((miss_per > 2).sum()), "/", len(ex_tok), "missed==0:", int((miss_per == 0).sum()), flush=True)
    if n in (12, 16):
        negr = [L.membership(seen, L.rolling_hashes(L.token_ids64(t), n)) for t in neg_tok]
        order = np.argsort([-r.mean() for r in negr])[:3]
        for j in order:
            r = negr[j]
            if r.sum() == 0: continue
            pos = np.flatnonzero(r)
            runs = []; st = pos[0]; pv = pos[0]
            for p in pos[1:]:
                if p != pv + 1: runs.append((st, pv)); st = p
                pv = p
            runs.append((st, pv))
            print("  n", n, "neg doc", j, "share", r.mean(), [tk.decode(neg_tok[j][a:b + n].tolist())[:140] for a, b in runs[:4]])

json.dump(res, open("/workspace/sft_build/longdoc_bench/contam/exp_n_excerpts.json", "w"), indent=1)
