#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare mixture variants per language: tokens by component, share of the language's tokens that is NEW relative to mix_v1, human-reference share, self-distilled share,
licence-class shares. Weak = blend + consensus-only languages (work/weak_langs.txt) + the Ge'ez languages amh / tir.

    python scripts/mix2_compare.py --variants A=/workspace/jobs/r11/mix_v2A,A2=... --weak work/weak_langs.txt --out OUT.md
"""
import argparse, json, os

SOURCES = ("science", "replay", "archive", "longdoc", "anchor", "r10_extra", "r14_teacher", "r10_share_alike", "r10_code_switch", "r10_nc")
ap = argparse.ArgumentParser()
ap.add_argument("--variants", required=True)
ap.add_argument("--weak", default="/workspace/jobs/r11/work/weak_langs.txt")
ap.add_argument("--out", required=True)
ap.add_argument("--v1", default="/workspace/jobs/r4/mix_v1")
a = ap.parse_args()
weak = set(open(a.weak).read().strip().split(","))
from gemma_mt.languages import ALL_ISOS, LANGUAGES

v1 = json.load(open(os.path.join(a.v1, "mixture_tables.json")))
v1tok = {i: v1["train_science"].get(i, {}).get("tokens", 0) + v1["train_replay"].get(i, {}).get("tokens", 0) for i in ALL_ISOS}
variants = dict(x.split("=", 1) for x in a.variants.split(","))
T = {}
for name, d in variants.items():
    T[name] = json.load(open(os.path.join(d, "mixture_tables_v2.json")))
    if "weak_extra" in T[name]:
        T[name]["r10_extra"] = T[name].pop("weak_extra")      # mix_v2A was written before the column was renamed (it is empty there)
    for s_ in SOURCES:
        T[name].setdefault(s_, {})
    T[name + "_lic"] = json.load(open(os.path.join(d, "licence_class_tables.json")))
L = ["# Weak / Ge'ez languages: what changes relative to mix_v1 (tokens of the TRAIN mixture)", "",
     "new% = tokens that are not mix_v1 samples (archive + longdoc + anchor + R10 extras) / language total; human% = replay (human references of the original SFT) / total; self% = archive + longdoc + anchor (model outputs) / total; "
     "share of mix_v1 tokens kept = v1 tokens / v2 total.", ""]
for name in variants:
    t = T[name]
    L += [f"## variant {name}", "", "| tag | language | v1 tokens | science | replay | archive | longdoc | anchor | R10 commercial | R14 teacher | R10 share-alike | R10 code-switch | R10 NC | total | new% | human-ref% | self-distilled% |", "|---|---|" + "---:|" * 15]
    tot_new = tot = 0
    for iso in ALL_ISOS:
        if iso not in weak:
            continue
        row = [t[s].get(iso, {"tokens": 0})["tokens"] for s in SOURCES]
        n = sum(row)
        new = sum(row[2:])
        selfd = sum(row[2:5]) + row[6]
        L.append(f"| {iso} | {LANGUAGES[iso].name} | {v1tok[iso]:,} | " + " | ".join(f"{x:,}" for x in row) + f" | {n:,} | {100 * new / max(n, 1):.1f}% | {100 * row[1] / max(n, 1):.1f}% | {100 * selfd / max(n, 1):.1f}% |")
        tot_new += new
        tot += n
    L.append(f"| **weak total** | | | | | | | | | | | | | {tot:,} | {100 * tot_new / max(tot, 1):.1f}% | | |")
    L.append("")
open(a.out, "w").write("\n".join(L) + "\n")
print("wrote", a.out)
