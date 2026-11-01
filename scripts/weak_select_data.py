#!/usr/bin/env python3
"""W1 (2026-10-26): build the DEV sample for the weak-language selection. Read-only from the Hub (token HF_TOKEN from the repo-root .env, never printed).

Tier A  FLORES+ dev (openlanguagedata/flores_plus, dev/<code>.jsonl): 100 sentence ids drawn with seed 2026 from the ids present in every Tier-A file.
Tier B  Sunbird/salt text-all dev (ach, nyn only: they have no FLORES+ dev): 100 rows drawn with seed 2026.
Tier C  languages with no dev reference at all: they get the SAME 100 FLORES+ dev English sentences (guards only, no chrF++).
Output: reports/data/weak_selection_2026-10-26/dev_sample.jsonl  (one row per language x sentence: iso, tier, id, english, reference|null) and sample_ids.json.
"""
import json, os, random, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "reports", "data", "weak_selection_2026-10-26")
sys.path.insert(0, os.path.join(ROOT, "gemma_mt"))
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))
from huggingface_hub import hf_hub_download
from gemma_mt.languages import LANGUAGES, SUPPORTED_RL

TOK = os.environ["HF_TOKEN"]
FLORES_OVERRIDE = {"seh": "seh_Latn_MZ", "twi": "twi_Latn_akua1239"}   # Sena has only the Mozambique variety; Twi: Akuapem file (the alternation by id parity of build_flores_dev_all.py is not used)
SALT = {"ach": "ach_text", "nyn": "nyn_text"}
N, SEED = 100, 2026

def rj(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

def flores(code):
    return {int(r["id"]): r["text"] for r in rj(hf_hub_download("openlanguagedata/flores_plus", f"dev/{code}.jsonl", repo_type="dataset", token=TOK))}

def main():
    en = flores("eng_Latn")
    tierA = {}
    for iso in SUPPORTED_RL:
        code = FLORES_OVERRIDE.get(iso) or LANGUAGES[iso].flores
        if code:
            tierA[iso] = flores(code)
    common = set(en)
    for d in tierA.values():
        common &= {i for i, t in d.items() if t.strip()}
    ids = sorted(random.Random(SEED).sample(sorted(common), N))
    # Tier B
    import pyarrow.parquet as pq
    p = hf_hub_download("Sunbird/salt", "text-all/dev-00000-of-00001.parquet", repo_type="dataset", token=TOK)
    salt = [r for r in pq.read_table(p).to_pylist() if (r.get("eng_source_text") or "").strip()]
    saltrows = {}
    for iso, col in SALT.items():
        good = sorted([r for r in salt if (r.get(col) or "").strip()], key=lambda r: r["id"])
        saltrows[iso] = random.Random(SEED).sample(good, N)
    rows = []
    for iso in SUPPORTED_RL:
        if iso in tierA:
            for i in ids:
                rows.append(dict(iso=iso, tier="A", id=f"flores_dev-{i}", english=en[i], reference=tierA[iso][i]))
        elif iso in SALT:
            for r in saltrows[iso]:
                rows.append(dict(iso=iso, tier="B", id=f"salt_dev-{r['id']}", english=r["eng_source_text"].strip(), reference=r[SALT[iso]].strip()))
        else:
            for i in ids:
                rows.append(dict(iso=iso, tier="C", id=f"flores_dev-{i}", english=en[i], reference=None))
    with open(os.path.join(OUT, "dev_sample.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"seed": SEED, "n": N, "flores_dev_ids": ids, "flores_dev_pool_size": len(common),
               "salt_dev_ids": {k: [r["id"] for r in v] for k, v in saltrows.items()},
               "tiers": {t: sorted({r["iso"] for r in rows if r["tier"] == t}) for t in "ABC"}},
              open(os.path.join(OUT, "sample_ids.json"), "w"), indent=1)
    print({t: len({r["iso"] for r in rows if r["tier"] == t}) for t in "ABC"}, "rows", len(rows), "pool", len(common))

main()
