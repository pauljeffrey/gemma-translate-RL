"""Vocabulary restriction study, step 4: the token list of a per-language 'lite' model = allowed output set  U  tokens of English SOURCE text (train English pool) U  prompt template tokens U ids 0..255 and byte tokens.
usage: vocab_restrict_keep.py ALLOWED_DIR OUTDIR LANGS(comma) CSV
Writes OUTDIR/keep_<iso>.npy (+ keep_union.npy over all given languages) and a CSV with set sizes, parameter counts and the share of the FLORES English sources (Q.json, 1,170 sentences) whose tokens are all inside the list (information only: the list is built without them)."""
import sys, json, csv, numpy as np
from transformers import AutoTokenizer
from gemma_mt import format as fmt
adir, out, langs, csvp = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
tok = AutoTokenizer.from_pretrained("/dev/shm/lc1_models/final2"); V = 262144
pool = json.load(open("/dev/shm/vr/pool.json")); texts = pool["b64"] + pool["b256"]
E = np.zeros(V, bool)
for k in range(0, len(texts), 2000):
    for ids in tok(texts[k:k + 2000], add_special_tokens=False)["input_ids"]: E[ids] = True
for iso in langs:  # prompt template (instruction + chat template) for every language, incl. the tag tokens
    for ids in [fmt.build_prompt("Hello, how are you?", iso, tok), tok(iso + " ", add_special_tokens=False)["input_ids"]]: E[ids] = True
E[:256] = True
ASCII = np.array([all(ord(c) < 128 for c in (p or "")) and bool(p) for p in tok.convert_ids_to_tokens(list(range(V)))])  # information: an English-source rule that needs no text sample
for b in range(256): E[tok.convert_tokens_to_ids("<0x%02X>" % b)] = True
Q = json.load(open("/workspace/jobs/c1b/out/Q.json")); qs = [tok(q["english"], add_special_tokens=False)["input_ids"] for q in Q]
BODY = 270_000_000 - 262144 * 640  # transformer body incl. norms (approx.): total 268.1M - embedding 167.8M
rows = []; U = np.zeros(V, bool)
for iso in langs + ["union"]:
    A = np.zeros(V, bool)
    if iso == "union": A = U.copy()
    else: A[np.load(f"{adir}/allowed_{iso}.npy")] = True; U |= A
    K = A | E; keep = np.where(K)[0]; np.save(f"{out}/keep_{iso}.npy", keep.astype(np.int32))
    qi = [K[np.array(x)].all() for x in qs]; qt = np.mean([K[x].mean() for x in qs])
    rows.append({"iso": iso, "allowed_output": int(A.sum()), "english_source_tokens_train": int(E.sum()), "lite_vocab": int(K.sum()), "lite_params_M": round((int(K.sum()) * 640 + BODY) / 1e6, 1), "output_only_params_M": round((int(A.sum()) * 640) / 1e6, 1),
                 "ascii_only_tokens": int(ASCII.sum()), "lite_vocab_with_ascii_rule": int((A | ASCII | E).sum()), "flores_en_sentences_fully_in_vocab": round(float(np.mean(qi)), 4), "flores_en_token_coverage": round(float(qt), 5)})
    print(rows[-1])
with open(csvp, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
