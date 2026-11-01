"""Vocabulary restriction study, step 1: build the allowed output-token set of every language and measure its coverage of the FLORES+ devtest REFERENCE tokens.
usage: vocab_restrict_build_sets.py OUTDIR GENDIR LANGS(comma)
Components of the set of language L (all TRAIN-side, no FLORES devtest / NTREX / eval reference text is ever read to BUILD a set):
  (a) the target tokens of the packed SFT training mix (r4/mix_v1/train: AfriScience-MT TRAIN + the UniMax replay of the original SFT windows) with iso == L
  (b) the tokens the final model itself generates (greedy, vLLM) on English TRAIN prompts (RL prompt pool v3), from vocab_restrict_gen_train.py
  (e) the tokens the model produces when it SAMPLES (T=1.0, 3 samples) on other English TRAIN prompts: the tokens it considers plausible, not only its argmax
  (c) basics: specials/control ids, tag tokens, byte-fallback tokens, digits/punctuation/whitespace pieces, single-character pieces of the script(s) of L (+ASCII letters)
Coverage is measured only afterwards on the reference text (tokenised exactly like a training target: `<iso> <reference>` + <end_of_turn>)."""
import sys, os, json, re, unicodedata, string, collections
import numpy as np
from transformers import AutoTokenizer
out, gendir, langs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
M = "/dev/shm/lc1_models/final2"; MIX = "/workspace/jobs/r4/mix_v1"; REFS = "/workspace/jobs/e2/refs_flores_plus.jsonl"; QJ = "/workspace/jobs/c1b/out/Q.json"
os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(M); V = 262144
pieces = tok.convert_ids_to_tokens(list(range(V)))
text = [(p or "").replace("▁", " ") for p in pieces]

def cp_ranges(s, ranges): return all(any(a <= ord(c) <= b for a, b in ranges) for c in s)
LATIN = [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x250, 0x2AF), (0x2B0, 0x2FF), (0x300, 0x36F), (0x1E00, 0x1EFF), (0xA720, 0xA7FF)]
ETH = [(0x1200, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F)]
ETH_LANGS = {"amh", "tir"}
def is_special(i): return i < 107 or pieces[i] in tok.all_special_tokens
def is_byte(p): return re.fullmatch(r"<0x[0-9A-F]{2}>", p or "") is not None
def is_dpw(s):  # digits, punctuation, symbols, whitespace
    if s == "": return False
    for c in s:
        cat = unicodedata.category(c)
        if c in string.digits or c.isspace() or cat[0] == "P" or cat in ("Zs", "Cc") or (ord(c) < 128 and cat[0] == "S") or c in " ​‌‍…€°×":
            continue
        return False
    return True
SPEC = np.zeros(V, bool); BYTE = np.zeros(V, bool); DPW = np.zeros(V, bool); LAT1 = np.zeros(V, bool); ETH1 = np.zeros(V, bool)
for i in range(V):
    p = pieces[i]
    SPEC[i] = is_special(i); BYTE[i] = is_byte(p)
    if not SPEC[i] and not BYTE[i]:
        DPW[i] = is_dpw(text[i]); s = text[i].lstrip(" ")
        if len(s) == 1 or (len(s) == 2 and s[1] in "̣̀́̂̃̄̌̀"):  # single char (+ a combining accent)
            LAT1[i] = cp_ranges(s, LATIN); ETH1[i] = cp_ranges(s, ETH)
print("class sizes: special", SPEC.sum(), "byte", BYTE.sum(), "digit/punct/space", DPW.sum(), "latin single-char", LAT1.sum(), "ethiopic single-char", ETH1.sum(), flush=True)

# (a) packed training targets
iso_table = json.load(open(MIX + "/meta.json"))["iso_table"]
T = MIX + "/train/"
toks = np.memmap(T + "tokens.u32.bin", dtype=np.uint32, mode="r")
iso_id = np.load(T + "iso_id.npy"); off = np.load(T + "sample_offset.npy").astype(np.int64); pl = np.load(T + "prompt_len.npy").astype(np.int64); tl = np.load(T + "target_len.npy").astype(np.int64); dr = np.load(T + "direction.npy")
def train_counts(iso):
    c = np.zeros(V, np.int64); idx = np.where((iso_id == iso_table.index(iso)) & (dr == 0))[0]; n = 0
    for k in idx:
        t = np.asarray(toks[off[k] + pl[k]: off[k] + pl[k] + tl[k]]).astype(np.int64); n += len(t); c += np.bincount(t, minlength=V)
    return c, len(idx), n

# references
refs = collections.defaultdict(list)
for l in open(REFS):
    d = json.loads(l)
    if d["iso"] in langs: refs[d["iso"]].append(d)
Q = collections.defaultdict(set)
for q in json.load(open(QJ)): Q[q["iso"]].add(str(q["id"]))

def ref_ids(iso, r): return tok(iso + " " + r["reference"], add_special_tokens=False)["input_ids"] + [106]
rows = []
for iso in langs:
    ca, nsamp, ntok = train_counts(iso)
    g = np.load(f"{gendir}/gen_{iso}.npz"); cb = np.bincount(g["ids"], minlength=V)
    gs = f"{gendir}/gens_{iso}.npz"; E = (np.bincount(np.load(gs)["ids"], minlength=V) > 0) if os.path.exists(gs) else np.zeros(V, bool)
    # alphabet closure (information only, a much larger set): every token whose characters all occur in the train/generated text of the language (+ digits, punctuation)
    chars = set("".join(text[i] for i in np.where((ca > 0) | (cb > 0))[0])); AL = np.array([bool(text[i]) and not SPEC[i] and not BYTE[i] and all((c in chars) or c in string.digits or c.isspace() or unicodedata.category(c)[0] == "P" for c in text[i]) for i in range(V)])
    tag = set(tok(iso + " ", add_special_tokens=False)["input_ids"]) | {1, 106, 107, 2}
    cc = SPEC | BYTE | DPW | LAT1 | (ETH1 if iso in ETH_LANGS else False)
    for t in tag: cc[t] = True
    # tokens of the plain ASCII letters/digits are in LAT1/DPW already
    A, B, C = ca > 0, cb > 0, cc
    A2 = ca >= 2
    sets = {"alphabet_closure": AL | SPEC | BYTE, "e_sampled": E, "a+b+c+e": A | B | C | E, "a2+b+c+e": A2 | B | C | E, "a_train": A, "b_gen": B, "c_basics": C, "a2_train_min2": A2, "a+b": A | B, "a+b+c": A | B | C, "a2+b+c": A2 | B | C, "b+c": B | C, "a+c": A | C}
    np.save(f"{out}/allowed_{iso}.npy", np.where(sets["a+b+c+e"])[0].astype(np.int32))
    ev = {"all": refs[iso], "pair_unseen": [r for r in refs[iso] if r["pair_seen"] is False], "Q1170": [r for r in refs[iso] if str(r["id"]) in Q[iso]]}
    rid = {k: [ref_ids(iso, r) for r in v] for k, v in ev.items()}
    for name, S in sets.items():
        row = {"iso": iso, "set": name, "size": int(S.sum()), "train_samples": nsamp, "train_target_tokens": ntok, "gen_tokens": int(len(g["ids"]))}
        for k, lst in rid.items():
            flat = np.array([t for x in lst for t in x]); inside = S[flat] if len(flat) else np.zeros(0, bool)
            row[f"cov_{k}"] = round(float(inside.mean()), 5) if len(flat) else None
            row[f"sent_full_{k}"] = round(float(np.mean([S[np.array(x)].all() for x in lst])), 4) if lst else None
            row[f"n_ref_tok_{k}"] = int(len(flat)); row[f"n_miss_{k}"] = int((~inside).sum()) if len(flat) else 0
            row[f"miss_types_{k}"] = int(len(set(flat[~inside].tolist()))) if len(flat) else 0
        rows.append(row)
    full = [r for r in rows if r["iso"] == iso and r["set"] == "a+b+c+e"][0]
    # unseen reference tokens (misses) of the final set, for the report
    S = sets["a+b+c+e"]; miss = collections.Counter(t for x in rid["pair_unseen"] for t in x if not S[t])
    print(iso, "size a+b+c+e", int(S.sum()), "cov_pair_unseen", full["cov_pair_unseen"], "top misses", [(text[t], c) for t, c in miss.most_common(8)], flush=True)
import csv
keys = list(rows[0].keys())
with open(f"{out}/sets_coverage.csv", "w", newline="") as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
print("wrote", f"{out}/sets_coverage.csv")
