#!/usr/bin/env python3
"""Sample-based attribution of the FLORES+ exposure of the SFT text (2026-10-26 contamination audit, agent C1, section 1).

Data: the replay pool of agent R4 = 87,498 ORIGINAL SFT windows (random sample, 0.76% of the 11,442,172 windows of `gemma/windows_sft`), decoded to text, one
jsonl per language {english, target, n_tok, src_idx}; it sits in the owner's private Hub backup `<HF_ORG_LEGACY>/sabiyarn-box-backup-r10-weak-language-data`
(path r10/pool/*.jsonl; read-only, streamed, nothing stored). For every window it finds FLORES+ dev / devtest English sentences (normalised sentence equality,
the `sentence` rule of gemma_mt/eval/contamination.py, applied to the sentences of the window) and whether the window's target side contains the FLORES+
reference of the same language for the same id (`pair`). Output: reports/data/contamination_audit_2026-10-26/sft_pool_flores_hits.csv (ids and counts only,
no FLORES text) and sft_pool_flores_summary.json.
"""
import csv, json, os, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from contam_audit_common import OUT, hf_token, normalise
from huggingface_hub import HfFileSystem, hf_hub_download
from sacrebleu.metrics import CHRF

REPO_ID = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-box-backup-r10-weak-language-data")
from contam_audit_common import CACHE
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gemma_mt.languages import LANGUAGES  # noqa: E402

SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


eng = {}
for split in ("devtest", "dev"):
    for r in jl(hf_hub_download("openlanguagedata/flores_plus", f"{split}/eng_Latn.jsonl", repo_type="dataset", token=hf_token(), cache_dir=CACHE)):
        eng.setdefault(normalise(r["text"]), []).append((split, int(r["id"])))
ref = {}   # (iso, split, id) -> normalised reference
ref_raw = {}
for iso, l in LANGUAGES.items():
    if not l.flores:
        continue
    for split in ("devtest", "dev"):
        try:
            for r in jl(hf_hub_download("openlanguagedata/flores_plus", f"{split}/{l.flores}.jsonl", repo_type="dataset", token=hf_token(), cache_dir=CACHE)):
                ref[(iso, split, int(r["id"]))] = normalise(r["text"])
                ref_raw[(iso, split, int(r["id"]))] = r["text"]
        except Exception:
            pass
fs = HfFileSystem(token=hf_token())
files = [p for p in fs.ls(f"datasets/{REPO_ID}/r10/pool", detail=False) if p.endswith(".jsonl")]
rows_total = 0
hits = []
per_iso = Counter()
for p in sorted(files):
    iso = os.path.basename(p)[:-6]
    with fs.open(p, "rb") as f:
        for ln, line in enumerate(f):
            r = json.loads(line)
            rows_total += 1
            per_iso[iso] += 1
            sents = [s for s in SPLIT.split(r["english"]) if s.strip()]
            found = []
            for s in sents:
                for (split, i) in eng.get(normalise(s), []):
                    found.append((split, i))
            if not found:
                continue
            tgt = normalise(r["target"])
            pair = [(split, i) for split, i in found if (iso, split, i) in ref and ref[(iso, split, i)] and ref[(iso, split, i)] in tgt]
            sim = ""
            if len(sents) == 1 and len(found) == 1 and (iso, found[0][0], found[0][1]) in ref_raw:
                sim = round(CHRF(word_order=2).sentence_score(r["target"], [ref_raw[(iso, found[0][0], found[0][1])]]).score, 1)
            hits.append({"iso": iso, "chrf_target_vs_flores_ref": sim, "row": ln, "src_idx": r.get("src_idx"), "n_tok": r.get("n_tok"), "n_sent": len(sents),
                         "n_flores_devtest": sum(1 for s, _ in found if s == "devtest"), "n_flores_dev": sum(1 for s, _ in found if s == "dev"),
                         "n_pair": len(pair), "ids": ";".join(f"{s}{i}" for s, i in sorted(set(found)))[:200]})
with open(os.path.join(OUT, "sft_pool_flores_hits.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(hits[0].keys())); w.writeheader(); w.writerows(hits)
n_dt = sum(h["n_flores_devtest"] for h in hits); n_d = sum(h["n_flores_dev"] for h in hits)
bylang = Counter(); 
for h in hits:
    bylang[h["iso"]] += h["n_flores_devtest"] + h["n_flores_dev"]
summ = {"windows_scanned": rows_total, "windows_with_flores_english": len(hits), "devtest_sentence_hits": n_dt, "dev_sentence_hits": n_d,
        "sentence_hits_by_iso": dict(bylang.most_common()), "windows_by_iso": dict(Counter(h["iso"] for h in hits).most_common()),
        "scale_to_full_sft_windows": round(11442172 / rows_total, 1), "pair_windows": sum(1 for h in hits if h["n_pair"]),
        "distinct_devtest_ids": len({i for h in hits for i in h["ids"].split(";") if i.startswith("devtest")}),
        "distinct_dev_ids": len({i for h in hits for i in h["ids"].split(";") if i.startswith("dev") and not i.startswith("devtest")}),
        "single_sentence_windows": sum(1 for h in hits if h["n_sent"] == 1), "lug_chrf_vs_flores_ref": [h["chrf_target_vs_flores_ref"] for h in hits if h["iso"] == "lug"]}
json.dump(summ, open(os.path.join(OUT, "sft_pool_flores_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1))
