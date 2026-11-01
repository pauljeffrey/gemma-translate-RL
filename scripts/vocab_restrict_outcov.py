"""Vocabulary restriction study: share of the tokens the UNRESTRICTED model itself writes (greedy, FLORES+ Q set) that lie inside the static set / the static set plus the source-word tokens.
A sentence whose unrestricted greedy output lies completely inside the allowed set is reproduced EXACTLY by restricted greedy decoding (the argmax is never masked); only the other sentences can change.
usage: vocab_restrict_outcov.py QJSON FULL.jsonl ALLOWED_DIR OUTCSV"""
import sys, json, csv, re, numpy as np
from transformers import AutoTokenizer
qj, fullf, adir, outcsv = sys.argv[1:5]
tok = AutoTokenizer.from_pretrained("/dev/shm/lc1_models/final2"); V = 262144
Q = {(q["iso"], str(q["id"])): q for q in json.load(open(qj))}
H = [json.loads(l) for l in open(fullf)]
def src_ids(text):
    ids = set(tok(text, add_special_tokens=False)["input_ids"])
    for w in set(re.findall(r"\w+", text)): ids |= set(tok(w, add_special_tokens=False)["input_ids"]) | set(tok(" " + w, add_special_tokens=False)["input_ids"]) | set(tok(w.capitalize(), add_special_tokens=False)["input_ids"]) | set(tok(" " + w.lower(), add_special_tokens=False)["input_ids"])
    return ids
rows = []
for iso in sorted({h["iso"] for h in H}):
    S = np.zeros(V, bool); S[np.load(f"{adir}/allowed_{iso}.npy")] = True
    n = a = b = sa = sb = k = 0
    for h in H:
        if h["iso"] != iso: continue
        o = np.array(h["ids"]); D = S.copy(); D[list(src_ids(Q[(iso, str(h["id"]))]["english"]))] = True
        n += len(o); a += S[o].sum(); b += D[o].sum(); sa += S[o].all(); sb += D[o].all(); k += 1
    rows.append({"iso": iso, "sentences": k, "out_tokens": n, "tok_in_static": round(a / n, 5), "tok_in_static_plus_source": round(b / n, 5), "sent_all_in_static": round(sa / k, 4), "sent_all_in_static_plus_source": round(sb / k, 4)}); print(rows[-1])
with open(outcsv, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
