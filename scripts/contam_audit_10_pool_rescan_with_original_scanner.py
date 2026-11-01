#!/usr/bin/env python3
"""Run the ORIGINAL scanner (gemma_mt/eval/contamination.py: exact = sentence OR contiguous token run, pair = reference in the same-language target) on the
87,498-window SFT sample (private Hub backup r10/pool), window by window, for FLORES+ dev and devtest. (2026-10-26 audit, agent C1, section 1)

Purpose: (a) check that the sample reproduces the scan of 2026-10-18 (95.3% / 98.7% of the FLORES+ English sentences found in the full 11.44 M windows ->
what fraction of the 1,012 / 997 sentences does a 0.76% sample of the windows contain, and does the Poisson picture hold), (b) say which window languages carry
them and how many FLORES sentences a window carries. Output: reports/data/contamination_audit_2026-10-26/sft_pool_rescan.json, sft_pool_rescan_windows.csv.
"""
import csv, json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from contam_audit_common import OUT, hf_token
from gemma_mt.eval.contamination import BenchIndex, scan_windows
from gemma_mt.languages import LANGUAGES
from huggingface_hub import HfFileSystem, hf_hub_download

from contam_audit_common import CACHE
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
rows = []
for split in ("devtest", "dev"):
    en = {int(r["id"]): r["text"] for r in jl(hf_hub_download("openlanguagedata/flores_plus", f"{split}/eng_Latn.jsonl", repo_type="dataset", token=hf_token(), cache_dir=CACHE))}
    for iso, l in LANGUAGES.items():
        if not l.flores:
            continue
        try:
            for r in jl(hf_hub_download("openlanguagedata/flores_plus", f"{split}/{l.flores}.jsonl", repo_type="dataset", token=hf_token(), cache_dir=CACHE)):
                rows.append({"english": en[int(r["id"])], "reference": r["text"], "iso": iso, "split": split, "id": int(r["id"])})
        except Exception:
            pass
index = BenchIndex(rows)
sp = {}
for r in rows:
    from gemma_mt.eval.contamination import normalise
    sp.setdefault(normalise(r["english"]), set()).add((r["split"], r["id"]))
print("index", len(index.texts), "distinct english", flush=True)
fs = HfFileSystem(token=hf_token())
R = ("datasets/" + os.environ['HF_ORG_LEGACY'] + "/sabiyarn-box-backup-r10-weak-language-data/r10/pool")
out = []
found_ids = {"devtest": set(), "dev": set()}
pair_ids = {"devtest": Counter(), "dev": Counter()}
n = 0
for p in sorted(x for x in fs.ls(R, detail=False) if x.endswith(".jsonl")):
    iso = os.path.basename(p)[:-6]
    with fs.open(p, "rb") as f:
        for ln, line in enumerate(f):
            r = json.loads(line); n += 1
            res = scan_windows(index, [(r["english"], r["target"], iso)])
            if not res.exact:
                continue
            ids = set()
            for i in res.exact:
                ids |= sp.get(index.texts[i], set())
            for s, i in ids:
                found_ids[s].add(i)
            pairids = set()
            for (i, piso) in res.pair:
                pairids.add(index.texts[i])
            out.append({"iso": iso, "row": ln, "src_idx": r.get("src_idx"), "n_tok": r.get("n_tok"), "n_exact_sentences": len(res.exact), "n_sentence_rule": len(res.sentence), "n_pair": len(res.pair)})
            for (i, piso) in res.pair:
                for s, ii in sp.get(index.texts[i], set()):
                    pair_ids[s][piso] += 1
with open(os.path.join(OUT, "sft_pool_rescan_windows.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
summ = {"windows": n, "windows_with_flores_english": len(out), "by_iso": dict(Counter(o["iso"] for o in out).most_common()),
        "sentences_per_flores_window": dict(Counter(o["n_exact_sentences"] for o in out)),
        "distinct_devtest_sentences_in_sample": len(found_ids["devtest"]), "distinct_dev_sentences_in_sample": len(found_ids["dev"]),
        "pair_windows": sum(1 for o in out if o["n_pair"]), "pair_by_language_devtest": dict(pair_ids["devtest"]), "pair_by_language_dev": dict(pair_ids["dev"])}
json.dump(summ, open(os.path.join(OUT, "sft_pool_rescan.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
