#!/usr/bin/env python3
"""Add `source_family` to the final shards (so a family such as mt560 can be switched off) and write the per-language token table with / without mt560.
    python add_family.py --final out/final --cpt out/stats/chars_per_token.json --out-csv out/stats/mt560_switch_per_language.csv
Rewrites the shards in place (column appended at the end). Families: mt560, multi-open, finetranslations, opus100, luo-opus, kweku-eng-twi, afrolingu-mt, pontoon, vukuzenzele, wolof-sudoping, wolof-bible, fon-helsinki, luhya, ssa-mte, ibom-mt, other:<repo>."""
import argparse, collections, csv, glob, json, os
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

FAM = [("mt560", "mt560"), ("multi-open", "multi-open"), ("finetranslations", "finetranslations"), ("opus-100", "opus100"), ("english-to-luo-opus", "luo-opus"),
       ("eng_twi", "kweku-eng-twi"), ("AfroLingu", "afrolingu-mt"), ("Pontoon", "pontoon"), ("vukuzenzele", "vukuzenzele"), ("english-wolof-translation", "wolof-sudoping"),
       ("wolof-english-bible", "wolof-bible"), ("Helsinki-en-fon", "fon-helsinki"), ("luhya", "luhya"), ("SSA-MTE", "ssa-mte"), ("ibom-mt", "ibom-mt"), ("flores", "flores-plus-dev-half")]
def fam(s):
    for k, v in FAM:
        if k in s: return v
    return "other:" + s.split(":")[0]
ap = argparse.ArgumentParser(); ap.add_argument("--final", required=True); ap.add_argument("--cpt", required=True); ap.add_argument("--out-csv", required=True)
a = ap.parse_args()
cpt = json.load(open(a.cpt)); cache = {}
chars = collections.defaultdict(lambda: collections.Counter()); pairs = collections.defaultdict(lambda: collections.Counter())
for f in sorted(glob.glob(a.final + "/*.parquet")):
    t = pq.read_table(f)
    if "source_family" in t.column_names:
        t = t.drop(["source_family"])
    srcs = t["source_dataset"].to_pylist()
    for s in set(srcs):
        cache.setdefault(s, fam(s))
    fa = [cache[s] for s in srcs]
    t = t.append_column("source_family", pa.array(fa, pa.string()))
    pq.write_table(t, f + ".tmp", compression="zstd"); os.replace(f + ".tmp", f)
    iso = t["iso"].to_pylist(); ln = pc.utf8_length(t["target"]).to_pylist()
    for i, fm, l in zip(iso, fa, ln):
        chars[i][fm == "mt560"] += l; pairs[i][fm == "mt560"] += 1
with open(a.out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["iso", "pairs_total", "pairs_mt560", "pairs_without_mt560", "est_tokens_total", "est_tokens_mt560", "est_tokens_without_mt560", "mt560_token_share_pct"])
    for i in sorted(chars, key=lambda k: -(chars[k][True] + chars[k][False])):
        r = cpt.get(i, 3.4); tt = (chars[i][True] + chars[i][False]) / r; tm = chars[i][True] / r
        w.writerow([i, pairs[i][True] + pairs[i][False], pairs[i][True], pairs[i][False], int(tt), int(tm), int(tt - tm), round(100 * tm / max(1, tt), 1)])
print("done", len(chars))
