#!/usr/bin/env python3
"""Tables from the SFT contamination scan outputs (2026-10-26 contamination audit, agent C1, section 1).

Input (mirrored in the archive; nothing else is read except the public FLORES+ English metadata for the id analysis, fetched read-only):
  research_archive_2026-10-23/box/other/sft_build/bench/test_sets/_contamination/contamination_<benchmark>.json   (+ ../contamination_dev.json)
Flag semantics: gemma_mt/gemma_mt/eval/contamination.py (exact = English sentence equals a sentence of a window OR occurs as a contiguous token run;
sentence = whole-sentence rule only; near = >= 50% of the 8-token n-grams; pair = exact AND the reference occurs in the target side of a same-language window).
Outputs in reports/data/contamination_audit_2026-10-26/:
  flag_summary_by_benchmark.csv, flag_summary_by_language.csv, flores_devtest_exposure_by_id.csv (ids + FLORES metadata, no text), flores_devtest_exposure_summary.json
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from contam_audit_common import OUT, hf_token, load_flags  # noqa: E402

os.makedirs(OUT, exist_ok=True)
BENCH = ["flores_plus", "ntrex", "mafand", "smol_sent", "smol_doc", "ibom_mt", "flores_moz", "glot_story", "salt", "dev"]
by_b, by_l = [], []
for b in BENCH:
    d = load_flags(b)
    ex, se, ne, pa = d["exact"], d["sentence"], d["near"], d["pair"]
    per = defaultdict(Counter)
    for k in ex:
        iso = k.split(":", 1)[0]
        c = per[iso]
        c["n"] += 1
        c["exact"] += bool(ex[k]); c["sentence"] += bool(se[k]); c["near"] += bool(ne[k]); c["pair"] += bool(pa[k])
        c["clean"] += (not ex[k]); c["no_pair"] += (not pa[k]); c["clean_strict"] += (not ex[k] and not ne[k])
    tot = Counter()
    for iso, c in per.items():
        tot.update(c)
        by_l.append({"benchmark": b, "iso": iso, **c})
    n = tot["n"]
    by_b.append({"benchmark": b, "created": d["meta"].get("created"), "training_windows": d["meta"].get("training_windows"), "languages": len(per), **tot,
                 "exact_pct": round(100 * tot["exact"] / n, 1), "sentence_pct": round(100 * tot["sentence"] / n, 1), "pair_pct": round(100 * tot["pair"] / n, 1),
                 "clean_pct": round(100 * tot["clean"] / n, 1), "clean_strict_pct": round(100 * tot["clean_strict"] / n, 1)})
for name, rows in (("flag_summary_by_benchmark.csv", by_b), ("flag_summary_by_language.csv", by_l)):
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("benchmark", "iso"), k))
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
for r in by_b:
    print(r["benchmark"], r["n"], "exact %", r["exact_pct"], "sentence %", r["sentence_pct"], "pair %", r["pair_pct"], "clean %", r["clean_pct"])

# ---- FLORES+ devtest and dev: which ids are exposed, which languages carry the pair
from huggingface_hub import hf_hub_download  # noqa: E402

from contam_audit_common import CACHE
summary = {}
for split, bench in (("devtest", "flores_plus"), ("dev", "dev")):
    meta = {}
    p = hf_hub_download("openlanguagedata/flores_plus", f"{split}/eng_Latn.jsonl", repo_type="dataset", token=hf_token(), cache_dir=CACHE)
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        meta[int(r["id"])] = {"domain": r.get("domain"), "topic": r.get("topic"), "url": r.get("url"), "n_words": len(r["text"].split())}
    d = load_flags(bench)
    ex, pa = d["exact"], d["pair"]
    rows = []
    pair_langs = defaultdict(list)
    exact_by_id = {}
    for k in ex:
        iso, rid = k.split(":", 1)
        if bench == "dev":
            if not rid.startswith(iso + "-") or rid.startswith("mafand"):
                continue
            rid = int(rid.rsplit("-", 1)[1])
        else:
            rid = int(rid)
        exact_by_id[rid] = bool(ex[k])
        if pa[k]:
            pair_langs[rid].append(iso)
    for rid, m in sorted(meta.items()):
        rows.append({"id": rid, "exact": int(exact_by_id.get(rid, False)), "n_pair_langs": len(pair_langs.get(rid, [])), "pair_langs": ";".join(sorted(pair_langs.get(rid, []))), **m})
    with open(os.path.join(OUT, f"flores_{split}_exposure_by_id.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    n = len(rows); ne = sum(r["exact"] for r in rows)
    dom = defaultdict(Counter)
    for r in rows:
        dom[r["domain"]]["n"] += 1; dom[r["domain"]]["exact"] += r["exact"]
    clean = [r for r in rows if not r["exact"]]
    arts = Counter(r["url"] for r in clean)
    all_arts = Counter(r["url"] for r in rows)
    summary[split] = {"n": n, "exact": ne, "exact_pct": round(100 * ne / n, 1), "clean": len(clean),
                      "by_domain": {k: {"n": v["n"], "exact": v["exact"], "exact_pct": round(100 * v["exact"] / v["n"], 1)} for k, v in dom.items()},
                      "clean_articles": len(arts), "articles_total": len(all_arts),
                      "clean_mean_words": round(sum(r["n_words"] for r in clean) / max(1, len(clean)), 1),
                      "all_mean_words": round(sum(r["n_words"] for r in rows) / n, 1),
                      "ids_with_pair_in_any_language": sum(1 for r in rows if r["n_pair_langs"]),
                      "pair_langs_counts": dict(Counter(l for r in rows for l in r["pair_langs"].split(";") if l).most_common())}
json.dump(summary, open(os.path.join(OUT, "flores_exposure_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1)[:3500])
