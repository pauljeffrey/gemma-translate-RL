"""Vocabulary restriction study: coverage of the FLORES+ devtest reference tokens by (static language set) and (static set + tokens of the SOURCE sentence, per sentence), the 'dynamic' extension.
A name in the English source ("Moldova", "Ubud") is copied into the translation, so the tokens of the source words (as written with and without a leading space) are legitimate per-request additions.
usage: vocab_restrict_cover2.py ALLOWED_DIR LANGS OUTCSV"""
import sys, json, csv, re, collections, numpy as np
from transformers import AutoTokenizer
adir, langs, outcsv = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
tok = AutoTokenizer.from_pretrained("/dev/shm/lc1_models/final2"); V = 262144
refs = collections.defaultdict(list)
for l in open("/workspace/jobs/e2/refs_flores_plus.jsonl"):
    d = json.loads(l)
    if d["iso"] in langs and d["pair_seen"] is False: refs[d["iso"]].append(d)
rows = []
for iso in langs:
    S = np.zeros(V, bool); S[np.load(f"{adir}/allowed_{iso}.npy")] = True; n = hit0 = hit1 = hit2 = 0; sent0 = sent2 = 0; extra = []
    for r in refs[iso]:
        ref = np.array(tok(iso + " " + r["reference"], add_special_tokens=False)["input_ids"] + [106])
        src = set(tok(r["english"], add_special_tokens=False)["input_ids"])
        words = re.findall(r"\w+", r["english"]); var = set()
        for w in set(words): var |= set(tok(w, add_special_tokens=False)["input_ids"]) | set(tok(" " + w, add_special_tokens=False)["input_ids"]) | set(tok(w.capitalize(), add_special_tokens=False)["input_ids"]) | set(tok(" " + w.lower(), add_special_tokens=False)["input_ids"])
        D1 = np.zeros(V, bool); D1[list(src)] = True; D2 = D1.copy(); D2[list(var)] = True
        a = S[ref]; b = (S | D1)[ref]; c = (S | D2)[ref]; n += len(ref); hit0 += a.sum(); hit1 += b.sum(); hit2 += c.sum(); sent0 += a.all(); sent2 += c.all(); extra.append(len(var | src))
    rows.append({"iso": iso, "n_sent": len(refs[iso]), "static_cov": round(hit0 / n, 5), "plus_source_tokens_cov": round(hit1 / n, 5), "plus_source_word_variants_cov": round(hit2 / n, 5),
                 "sent_fully_covered_static": round(sent0 / len(refs[iso]), 4), "sent_fully_covered_dynamic": round(sent2 / len(refs[iso]), 4), "mean_extra_rows_per_sentence": round(float(np.mean(extra)), 1)})
    print(rows[-1], flush=True)
with open(outcsv, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
