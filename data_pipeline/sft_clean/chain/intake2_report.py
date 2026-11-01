#!/usr/bin/env python3
"""Tables for reports/intake2_pairs_2026-10-31.md from hub/extra2/_stats (run on the instance after intake2_pairs.py finalize + tokens). Prints markdown to stdout."""
import collections, csv, glob, json, os, sys

ROOT = os.environ.get("CLEAN_SFT_ROOT", "/workspace/work")
ST = f"{ROOT}/hub/extra2/_stats"
STAGE = f"{ROOT}/work/_intake2/stage"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intake2_pairs as I                                              # noqa: E402

# source_dataset -> source name
sd2src = {}
for p in glob.glob(f"{STAGE}/*/*.stats.json"):
    s = json.load(open(p))
    for k in s["st"]:
        sd2src[k.split("\t")[1]] = s["source"]
rows = list(csv.DictReader(open(ST + "/per_language_source.csv")))
tok = list(csv.DictReader(open(ST + "/tokens_by_source_language.csv")))
known = set(json.load(open(f"{ROOT}/work/gate/thresholds.json"))["languages"])

agg = collections.defaultdict(collections.Counter); langs = collections.defaultdict(collections.Counter)
for r in rows:
    s = sd2src.get(r["source"], "?")
    ren = {"dropped_by_clean_rules": "dropped_clean", "dropped_dup_within_source": "dropped_dup_within", "dropped_dup_in_chain_inputs": "dropped_dup_existing",
           "removed_by_protected_filter": "removed_protected", "dropped_dup_earlier_source": "dropped_dup_other_source"}
    for k, v in r.items():
        if k not in ("iso", "source"):
            agg[s][ren.get(k, k)] += int(v or 0)
    langs[s][r["iso"]] += int(r["kept"])
tk = collections.defaultdict(int); tkl = collections.Counter(); kept_l = collections.Counter(); raw_l = collections.Counter(); prot_l = collections.Counter(); dup_l = collections.Counter()
for t in tok:
    tk[t["source"]] += int(t["est_total_tokens"]); tkl[t["iso"]] += int(t["est_total_tokens"])
for r in rows:
    kept_l[r["iso"]] += int(r["kept"]); raw_l[r["iso"]] += int(r["raw_rows"]); prot_l[r["iso"]] += int(r["removed_by_protected_filter"])
    dup_l[r["iso"]] += int(r["dropped_dup_in_chain_inputs"])

print("### Per source\n")
print("| # | source | repo(s) | licence class | raw rows read | clean/short/licence drops | dup in chain inputs | dup within / earlier source | removed by protected set | kept | est. tokens (Gemma, en+target) | languages (kept) |")
print("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
for i, (name, spec) in enumerate(I.SOURCES.items(), 1):
    a = agg[name]
    lg = ", ".join(f"{k} {v:,}" for k, v in sorted(langs[name].items(), key=lambda kv: -kv[1]) if v)
    if len(lg) > 260:
        lg = lg[:260].rsplit(",", 1)[0] + f", ... ({sum(1 for v in langs[name].values() if v)} languages)"
    print(f"| {i} | `{name}` | {spec['repo']} | {spec['lic']} | {a['raw_rows']:,} | {a['dropped_clean'] + a['dropped_short_or_placeholder'] + a['dropped_licence']:,} | {a['dropped_dup_existing']:,} | "
          f"{a['dropped_dup_within'] + a['dropped_dup_other_source']:,} | {a['removed_protected']:,} | {a['kept']:,} | {tk[name]:,} | {lg} |")
tot = collections.Counter()
for a in agg.values():
    tot.update(a)
print(f"| | **total** | | | {tot['raw_rows']:,} | {tot['dropped_clean'] + tot['dropped_short_or_placeholder'] + tot['dropped_licence']:,} | {tot['dropped_dup_existing']:,} | {tot['dropped_dup_within'] + tot['dropped_dup_other_source']:,} | {tot['removed_protected']:,} | {tot['kept']:,} | {sum(tk.values()):,} | |")

print("\n### Per language (rows kept, estimated tokens of english+target, protected-set removals)\n")
print("| iso | in thresholds.json | rows kept | est. tokens | raw rows read | dup in chain inputs | removed by protected set |")
print("|---|---|---:|---:|---:|---:|---:|")
for iso, n in sorted(kept_l.items(), key=lambda kv: -kv[1]):
    if n == 0:
        continue
    print(f"| {iso} | {'yes' if iso in known else '**new**'} | {n:,} | {tkl[iso]:,} | {raw_l[iso]:,} | {dup_l[iso]:,} | {prot_l[iso]:,} |")
print(f"| **total** | | {sum(kept_l.values()):,} | {sum(tkl.values()):,} | {sum(raw_l.values()):,} | {sum(dup_l.values()):,} | {sum(prot_l.values()):,} |")
new = sorted(i for i, n in kept_l.items() if n and i not in known)
print("\nNew iso codes (not in work/gate/thresholds.json): " + ", ".join(f"{i} ({kept_l[i]:,})" for i in new))
