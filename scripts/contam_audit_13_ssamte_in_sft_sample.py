#!/usr/bin/env python3
"""Is the SSA-MTE train split (converter `ssa_mte` of the SFT build: English source + human reference) in the SFT windows? Test on the 87,498-window sample.
Counts windows (by language) whose English side contains an SSA-MTE English source of the train / dev / test split. Output: ssa_mte_in_sft_sample.csv (counts only)."""
import csv, json, os, re, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from contam_audit_common import OUT, hf_token, normalise
from huggingface_hub import hf_hub_download, HfFileSystem
from contam_audit_common import CACHE as C
LANG = ["Amharic", "Emakhuwa", "Ewe", "Hausa", "Igbo", "Kikuyu", "Kinyarwanda", "Lingala", "Luo", "Nyanja", "Sesotho", "Swahili", "Twi", "Wolof", "Yoruba", "Zulu"]
idx = {}
for sp in ("train", "dev", "test"):
    for L in LANG:
        for l in open(hf_hub_download("McGill-NLP/SSA-MTE", f"{sp}/{L}.json", repo_type="dataset", token=hf_token(), cache_dir=C)):
            r = json.loads(l)
            if r.get("src_sent") and len(r["src_sent"].split()) >= 4:
                idx.setdefault(normalise(r["src_sent"]), set()).add(sp)
fs = HfFileSystem(token=hf_token())
SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
cnt = Counter(); total = Counter()
for p in sorted(x for x in fs.ls(("datasets/" + os.environ['HF_ORG_LEGACY'] + "/sabiyarn-box-backup-r10-weak-language-data/r10/pool"), detail=False) if x.endswith(".jsonl")):
    iso = os.path.basename(p)[:-6]
    for line in fs.open(p, "rb"):
        r = json.loads(line); total[iso] += 1
        sp = set()
        for s in SPLIT.split(r["english"]):
            sp |= idx.get(normalise(s), set())
        for x in sp:
            cnt[(iso, x)] += 1
rows = [{"window_iso": k[0], "ssa_mte_split": k[1], "windows": v} for k, v in sorted(cnt.items())]
with open(os.path.join(OUT, "ssa_mte_in_sft_sample.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["window_iso", "ssa_mte_split", "windows"]); w.writeheader(); w.writerows(rows)
print(Counter({s: sum(v for (i, x), v in cnt.items() if x == s) for s in ("train", "dev", "test")}), "of", sum(total.values()), "windows")
print(sorted(cnt.items(), key=lambda kv: -kv[1])[:12])
