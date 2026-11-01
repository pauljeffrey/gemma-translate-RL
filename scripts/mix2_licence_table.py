#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-language licence-class table of the model-output components (archive, longdoc, anchor) of a mixture and what a licence filter would remove.

    python scripts/mix2_licence_table.py --mix /workspace/jobs/r11/mix_v2A --out /workspace/jobs/r11/mix_v2A/licence_filter.md

Filter 1 (owner's option "A_clean"): exclude rows whose ENGLISH source licence class is nc / unknown (none exist in the RL prompt pools: odc-by, cc-by, cc0, share_alike).
Filter 2 (stricter, share-alike): additionally exclude share_alike (Wikipedia, CC-BY-SA) sources. Writes the exclusion lists as jsonl of {component, key/doc_key, iso, licence_class, n_tokens}
so that a clean variant can be rebuilt with `mix2_build.py --exclude-classes`."""
import argparse, collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mix2_build as B
ap = argparse.ArgumentParser(); ap.add_argument("--mix", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
rows = []
for comp in ("archive", "longdoc", "anchor"):
    p = os.path.join(a.mix, "components", comp + ".jsonl")
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            r = json.loads(l); r["component"] = comp; r["cls"] = B.LIC_CLASSES[B.licence_class(r)]; rows.append(r)
tab = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for r in rows:
    d = tab[r["iso"]][r["cls"]]; d[0] += 1; d[1] += int(r.get("n_tokens") or 0)
L = ["| tag | " + " | ".join(f"{c} rows | {c} tokens" for c in B.LIC_CLASSES[:4]) + " | removed by filter 1 (nc + unknown) rows / tokens | removed by filter 2 (+ share_alike) rows / tokens |", "|---|" + "---:|" * 10]
tot = collections.defaultdict(lambda: [0, 0])
for iso in sorted(tab):
    cells = []
    for c in B.LIC_CLASSES[:4]:
        n, t = tab[iso].get(c, [0, 0]); cells += [f"{n:,}", f"{t:,}"]; tot[c][0] += n; tot[c][1] += t
    f1 = [sum(tab[iso].get(c, [0, 0])[k] for c in ("nc", "unknown")) for k in (0, 1)]
    f2 = [f1[k] + tab[iso].get("share_alike", [0, 0])[k] for k in (0, 1)]
    L.append(f"| {iso} | " + " | ".join(cells) + f" | {f1[0]:,} / {f1[1]:,} | {f2[0]:,} / {f2[1]:,} |")
f1 = [sum(tot[c][k] for c in ("nc", "unknown")) for k in (0, 1)]; f2 = [f1[k] + tot["share_alike"][k] for k in (0, 1)]
L.append("| **all** | " + " | ".join(f"{tot[c][0]:,} | {tot[c][1]:,}" for c in B.LIC_CLASSES[:4]) + f" | {f1[0]:,} / {f1[1]:,} | {f2[0]:,} / {f2[1]:,} |")
open(a.out, "w").write("\n".join(L) + "\n")
with open(a.out.replace(".md", "_exclusions.jsonl"), "w") as f:
    for r in rows:
        if r["cls"] != "commercial_ok":
            f.write(json.dumps({"component": r["component"], "key": r.get("key") or r.get("doc_key"), "iso": r["iso"], "licence_class": r["cls"], "licence": r.get("licence"), "n_tokens": r.get("n_tokens")}) + "\n")
print("languages with a row", len(tab), "filter1", f1, "filter2", f2)
