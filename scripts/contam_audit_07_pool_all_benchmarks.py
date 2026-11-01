#!/usr/bin/env python3
"""Same sample as contam_audit_05 (87,498 SFT windows, private Hub backup r10/pool), for EVERY benchmark of the paper (2026-10-26 audit, agent C1).

English sets and references come from the per-sentence outputs of the paper's evaluation (research_archive_2026-10-23/box/jobs/e1/sys_rl.jsonl.gz: b, iso, id, en, ref).
Per benchmark and window language: windows with at least one benchmark English sentence (sentence equality after normalisation, window split at sentence ends),
and windows whose target also contains the reference of the SAME language for the same row (`pair`). Counts only; scale factor to the full SFT text 130.8.
Output: reports/data/contamination_audit_2026-10-26/sft_pool_benchmark_hits.csv
"""
import csv, gzip, json, os, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from contam_audit_common import ARCHIVE, OUT, hf_token, normalise
from huggingface_hub import HfFileSystem

REPO_ID = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-box-backup-r10-weak-language-data")
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
idx = defaultdict(dict)       # benchmark -> normalised english -> {iso: normalised ref}
n_en = defaultdict(set)
with gzip.open(os.path.join(ARCHIVE, "box", "jobs", "e1", "sys_rl.jsonl.gz"), "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        e = normalise(r["en"])
        if len(e.split()) < 3:
            continue
        idx[r["b"]].setdefault(e, {})[r["iso"]] = normalise(r["ref"])
        n_en[r["b"]].add(e)
fs = HfFileSystem(token=hf_token())
cnt = defaultdict(Counter)
total = Counter()
for p in sorted(x for x in fs.ls(f"datasets/{REPO_ID}/r10/pool", detail=False) if x.endswith(".jsonl")):
    iso = os.path.basename(p)[:-6]
    with fs.open(p, "rb") as f:
        for line in f:
            r = json.loads(line)
            total[iso] += 1
            tgt = None
            seen = set()
            for s in SPLIT.split(r["english"]):
                n = normalise(s)
                if len(n.split()) < 3:
                    continue
                for b, d in idx.items():
                    refs = d.get(n)
                    if refs is None:
                        continue
                    if (b, "en") not in seen:
                        cnt[(b, iso)]["windows_en"] += 1; seen.add((b, "en"))
                    if tgt is None:
                        tgt = normalise(r["target"])
                    if iso in refs and refs[iso] and refs[iso] in tgt and (b, "pair") not in seen:
                        cnt[(b, iso)]["windows_pair"] += 1; seen.add((b, "pair"))
import re
from contam_audit_common import REPO
plan = open(os.path.join(REPO, "reports", "gemma_sft_training_plan_2026-10-14.md"), encoding="utf-8").read()
train = {m.group(1): int(m.group(2).replace(",", "")) for m in re.finditer(r"^\| (\w{3}) \| [^|]+ \| [\d,]+ / [\d,]+ / [\d,]+ / [\d,]+ \| ([\d,]+) \|", plan, re.M)}
rows = []
for (b, iso), c in sorted(cnt.items()):
    rows.append({"benchmark": b, "window_iso": iso, "windows_scanned": total[iso], "windows_with_benchmark_english": c["windows_en"], "windows_with_same_language_reference": c["windows_pair"],
                 "est_windows_full_sft(en)": round(c["windows_en"] * train.get(iso, 0) / total[iso]), "est_windows_full_sft(pair)": round(c["windows_pair"] * train.get(iso, 0) / total[iso])})
with open(os.path.join(OUT, "sft_pool_benchmark_hits.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
tot = Counter(); pr = Counter()
for r in rows:
    tot[r["benchmark"]] += r["windows_with_benchmark_english"]; pr[r["benchmark"]] += r["windows_with_same_language_reference"]
est = Counter(); estp = Counter()
for r in rows:
    est[r["benchmark"]] += r["est_windows_full_sft(en)"]; estp[r["benchmark"]] += r["est_windows_full_sft(pair)"]
print("windows scanned", sum(total.values()))
for b in idx:
    top = sorted(((r["windows_with_benchmark_english"], r["window_iso"]) for r in rows if r["benchmark"] == b), reverse=True)[:6]
    print(b, "distinct english", len(n_en[b]), "windows with english", tot[b], "est full-SFT windows with english", est[b], "with same-language pair", estp[b], "windows with same-language reference", pr[b], "top languages", top)
