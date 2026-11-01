#!/usr/bin/env python3
"""Is the MAFAND-MT TEST set inside UBC-NLP/AfroLingu-MT (an SFT source: converter `afrolingu_mt`, train split)? (2026-10-26 audit, agent C1)

Fetches the MAFAND-MT test files (public GitHub raw, a few hundred KB) into memory, streams the three AfroLingu-MT files from the Hub (read-only)
and counts, per language and file, MAFAND test pairs whose English and target sentence both occur (normalised) as input/output of one row.
Writes only counts to reports/data/contamination_audit_2026-10-26/mafand_in_afrolingu.csv.
"""
import csv, json, os, sys, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from contam_audit_common import OUT, hf_token, normalise
from huggingface_hub import HfFileSystem

LANGS = ["yor", "ibo", "hau", "pcm", "twi", "swa", "kin", "amh", "lug", "zul", "xho", "sna", "nya", "tsn", "luo"]
BASE = "https://raw.githubusercontent.com/masakhane-io/lafand-mt/main/data/json_files/en-{l}/test.json"
pairs = {}
for l in LANGS:
    try:
        raw = urllib.request.urlopen(BASE.format(l=l), timeout=60).read().decode("utf-8")
    except Exception as e:
        print("skip", l, str(e)[:60]); continue
    rows = [json.loads(x) for x in raw.splitlines() if x.strip()]
    for r in rows:
        t = r.get("translation") or r
        en, tg = t.get("en"), t.get(l) or [v for k, v in t.items() if k != "en"][0]
        pairs[(normalise(en), normalise(tg))] = l
    print(l, len(rows))
by_en = {}
for (e, t), l in pairs.items():
    by_en.setdefault(e, []).append((t, l))
fs = HfFileSystem(token=hf_token())
res = {}
for fn in ("AfroLinfu-MT-dev.jsonl", "AfroLinfu-MT-test.jsonl", "AfroLinfu-MT-train.jsonl"):
    with fs.open(f"datasets/UBC-NLP/AfroLingu-MT/{fn}", "rb") as f:
        for line in f:
            r = json.loads(line)
            a, b = normalise(r["input"]), normalise(r["output"])
            for x, y in ((a, b), (b, a)):
                for t, l in by_en.get(x, ()):
                    if t == y:
                        res.setdefault((fn, l), set()).add(x)
w = csv.writer(open(os.path.join(OUT, "mafand_in_afrolingu.csv"), "w", newline=""))
w.writerow(["file", "mafand_lang", "mafand_test_pairs_found", "mafand_test_pairs_total"])
tot = {l: sum(1 for v in pairs.values() if v == l) for l in LANGS}
for (fn, l), s in sorted(res.items()):
    w.writerow([fn, l, len(s), tot[l]]); print(fn, l, len(s), tot[l])
