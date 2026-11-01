#!/usr/bin/env python3
"""Provenance scan of every source repo that contributed rows: licence declared on the card, mentions of NLLB / TranslatePsy / FLORES etc. -> out/stats/provenance_scan.csv
    python prov_scan.py --out out --scan license_scan.json"""
import argparse, csv, glob, json, os, re
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--scan", required=True)
a = ap.parse_args()
scan = json.load(open(a.scan))["datasets"]
srcs = set()
for f in glob.glob(a.out + "/meta/*.parquet"):
    srcs |= {s.split(":")[0] for s in set(pq.read_table(f, columns=["src"])["src"].to_pylist())}
pats = {"nllb": r"nllb|no language left behind", "translatepsy": r"translate\s?psy", "flores": r"flores", "benchmark_words": r"ntrex|mafand|smolsent|\bsalt\b|tico-?19|sib-?200"}
rows = []
for r in sorted(srcs):
    row = {"source_repo": r, "licence_scan_declared": (scan.get(r) or {}).get("declared"), "licence_scan_class": (scan.get(r) or {}).get("class"), "card": "not_read"}
    if re.match(r"^[\w.\-]+/[\w.\-]+$", r):
        try:
            t = open(hf_hub_download(r, "README.md", repo_type="dataset", token=os.environ["HF_TOKEN"], local_dir="/workspace/work/dl/_cards2/" + r.replace("/", "__"))).read()
            row["card"] = "read"
            m = re.search(r"(?im)^license:\s*(.+)$", t)
            row["card_licence"] = m.group(1).strip() if m else ""
            for k, p in pats.items():
                row[k] = len(re.findall(p, t, re.I))
        except Exception as e:
            row["card"] = type(e).__name__
    rows.append(row)
cols = ["source_repo", "card", "card_licence", "licence_scan_declared", "licence_scan_class", *pats]
with open(a.out + "/stats/provenance_scan.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print(len(rows), "sources;", [r["source_repo"] for r in rows if r.get("nllb") or r.get("translatepsy")])
