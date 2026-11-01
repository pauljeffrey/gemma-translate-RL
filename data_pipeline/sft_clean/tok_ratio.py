#!/usr/bin/env python3
"""Estimate target characters per Gemma-3 token per language from a sample of the final pairs (the old plan counted Gemma target tokens).
    python tok_ratio.py --final out/final --tokenizer tok_google/tokenizer.json --out out/stats/chars_per_token.json [--per-lang 1500]
Output is an ESTIMATE: the token count of a language = target characters / this ratio (the real windows differ by the separator and tag tokens)."""
import argparse, collections, glob, json, random
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--final", required=True); ap.add_argument("--tokenizer", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--per-lang", type=int, default=1500)
a = ap.parse_args()
tk = Tokenizer.from_file(a.tokenizer)
rnd = random.Random(0)
pool = collections.defaultdict(list)       # reservoir per language
seen = collections.Counter()
for f in sorted(glob.glob(a.final + "/*.parquet")):
    t = pq.read_table(f, columns=["iso", "target"]).to_pydict()
    for iso, tg in zip(t["iso"], t["target"]):
        seen[iso] += 1
        if len(pool[iso]) < a.per_lang:
            pool[iso].append(tg)
        else:
            j = rnd.randrange(seen[iso])
            if j < a.per_lang:
                pool[iso][j] = tg
res = {}
for iso, v in pool.items():
    chars = sum(len(x) for x in v)
    toks = sum(len(e.ids) for e in tk.encode_batch(v, add_special_tokens=False))
    res[iso] = round(chars / max(1, toks), 3)
json.dump(res, open(a.out, "w"), indent=1)
print(len(res), "languages;", {k: res[k] for k in list(res)[:8]})
