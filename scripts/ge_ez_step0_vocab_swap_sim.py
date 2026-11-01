"""Q2b: simulate adding N new tokens (into freed vocabulary slots) for Ge'ez-script text (amh, tir) or Fon, and measure the fertility on HELD-OUT FLORES+ devtest. No model, no quality claim.

Method = continued BPE on top of the EXISTING Gemma-3 tokenizer (the exact procedure one would apply to tokenizer.json):
  step 0  Ethiopic letters that have no vocabulary entry (they cost 3 byte-fallback tokens) get a single-character token if they occur >= 3 times in the training text;
  step 1  the training text is tokenised with that tokenizer; inside every word (segment that starts at a token beginning with U+2581) the most frequent adjacent token pair is merged,
          repeated greedily. Each merge becomes (a) a new merge rule APPENDED at the END of the merges list (lowest priority: all old merges still apply first, so old text tokenises
          exactly as before unless it contains the new pair after all old merges) and (b) a new vocabulary entry a+b placed in a freed id (unused tail ids 256001-262143 first).
  The result is a real tokenizer.json that is loaded with HF `tokenizers` and used to encode the devtest references (so the numbers are not a re-implementation).
Training text: wikimedia/wikipedia 20231101 (am, ti, fon), HuggingFaceFW/finetranslations original-language text (amh, tir; ODC-By), HuggingFaceFW/fineweb-2 fon_Latn train (ODC-By),
FLORES+ DEV (fon only, 997 sentences). NEVER devtest. A contamination check (exact devtest sentences found in the training text) is written to the output.
Outputs: swap_sim.csv, swap_sim_retokenisation.csv, swap_sim_corpus.json, swap_sim_new_tokens_<variant>.txt
"""
import os, sys, json, glob, re, heapq, copy, time, collections
import numpy as np, pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(__file__))
from ge_ez_step0_common import *
from tokenizers import Tokenizer

NS = [0, 250, 500, 1000, 2000, 3000, 5000, 8000]
def find(repo, fn): return glob.glob(f"{CACHE}/models--{repo.replace('/','--')}/snapshots/*/{fn}")[0]
def flores(split, code):
    p = glob.glob(f"{CACHE}/datasets--openlanguagedata--flores_plus/snapshots/*/{split}/{code}.jsonl")[0]
    return [json.loads(l)["text"] for l in open(p, encoding="utf-8")]
def parq(pat, col, maxdocs=None, maxchars=None):
    out, n = [], 0
    for f in sorted(glob.glob(f"{CACHE}/{pat}")):
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(columns=[col], batch_size=2048):
            for x in b.column(0).to_pylist():
                if x:
                    out.append(x); n += len(x)
                    if maxdocs and len(out) >= maxdocs or maxchars and n >= maxchars: return out
    return out
def to_lines(docs):
    L = []
    for d in docs:
        for l in d.split("\n"):
            l = l.strip()
            if len(l) >= 8: L.append(l)
    return L

base_path = find((os.environ['HF_ORG_LEGACY'] + "/gemma-3-270m-translate-rl2c"), "latest/tokenizer.json")
BASE = json.load(open(base_path, encoding="utf-8"))
BASE_VOCAB = BASE["model"]["vocab"]; BASE_MERGES = BASE["model"]["merges"]; VSTR = set(BASE_VOCAB)
inv_vocab = {v: k for k, v in BASE_VOCAB.items()}
cls = np.load(os.path.join(OUT, "vocab_token_class.npy"), allow_pickle=True)
tail_ids = [i for i in range(256001, 262144) if cls[i] == "unused_tail"]
s1_free = np.load(os.path.join(OUT, "free_ids_S1.npy"))
# Slots: first the 6,143 <unusedN> tail tokens (no merge refers to them, nothing is lost). Beyond that, whole small scripts are freed WITH their merge rules:
# every merge whose part or result lies in the freed set is dropped; a token that loses all its producing merges is dead and is freed too (fixpoint).
EXTRA_SCRIPTS = ["ogham", "runic", "mongolian", "bopomofo", "syriac", "tibetan", "cherokee", "thaana", "oriya", "gurmukhi", "armenian", "lao", "myanmar", "sinhala", "khmer", "georgian"]
F = {inv_vocab_i for inv_vocab_i in range(262144) if cls[inv_vocab_i] in EXTRA_SCRIPTS}
producers = collections.defaultdict(list)
for k, (a, b) in enumerate(BASE_MERGES): producers[a + b].append(k)
DROPPED = set(); Fs = {inv_vocab[i] for i in F}
while True:
    new = False
    for k, (a, b) in enumerate(BASE_MERGES):
        if k not in DROPPED and (a in Fs or b in Fs or a + b in Fs):
            DROPPED.add(k); new = True
    for r, ks in producers.items():
        if r not in Fs and ks and all(k in DROPPED for k in ks): Fs.add(r); new = True
    if not new: break
CLOSURE_IDS = sorted(BASE_VOCAB[t] for t in Fs if BASE_VOCAB[t] >= 107 and t not in ("\u2581",) and cls[BASE_VOCAB[t]] != "byte")
FREE_IDS = tail_ids + CLOSURE_IDS
print("free slots: unused tail", len(tail_ids), "script closure", len(CLOSURE_IDS), "merges dropped if closure used", len(DROPPED), flush=True)

CLOSURE_SET = set(CLOSURE_IDS)
def build(extra_chars, new_merges):
    """-> Tokenizer with single-char tokens + appended merges. Returns (tokenizer, json_dict, list of (id, string))."""
    j = {k: v for k, v in BASE.items() if k != "model"}
    m = dict(BASE["model"]); vocab = dict(BASE_VOCAB)
    freed, new_entries = {}, []
    ids = iter(FREE_IDS)
    def alloc(s):
        i = next(ids); old = inv_vocab[i]; vocab.pop(old); vocab[s] = i; freed[i] = old; new_entries.append((i, s)); return i
    for c in extra_chars: alloc(c)
    merges = list(BASE_MERGES)
    for a, b in new_merges:
        alloc(a + b)
    if any(i in CLOSURE_SET for i in freed):
        merges = [m_ for k, m_ in enumerate(BASE_MERGES) if k not in DROPPED]
    n_old = len(merges)
    merges += [[a, b] for a, b in new_merges]
    m["vocab"] = vocab; m["merges"] = merges
    for a, b in merges[n_old:]:
        if a not in vocab or b not in vocab or a + b not in vocab:
            raise RuntimeError("bad merge %r %r (in vocab: %s %s %s)" % (a, b, a in vocab, b in vocab, a + b in vocab))
    j["model"] = m
    fr = set(freed)
    j["added_tokens"] = [t for t in BASE["added_tokens"] if t["id"] not in fr]
    return Tokenizer.from_str(json.dumps(j, ensure_ascii=False)), new_entries

base_tok = Tokenizer.from_str(json.dumps(BASE, ensure_ascii=False))

def words_from_ids(seqs, tokstr):
    """split token-id sequences into words at tokens starting with U+2581; returns Counter of tuples of token STRINGS"""
    c = collections.Counter()
    for ids in seqs:
        cur = []
        for t in ids:
            s = tokstr[t]
            if s.startswith("▁") and cur:
                c[tuple(cur)] += 1; cur = []
            cur.append(s)
        if cur: c[tuple(cur)] += 1
    return c

def learn_merges(wordcount, nmax, forbid, ok, min_count=5):
    words = [list(w) for w in wordcount]; freq = [wordcount[w] for w in wordcount]
    cnt = collections.Counter(); idx = collections.defaultdict(set)
    for wi, w in enumerate(words):
        for p in zip(w, w[1:]):
            cnt[p] += freq[wi]; idx[p].add(wi)
    heap = [(-c, p) for p, c in cnt.items()]; heapq.heapify(heap)
    out = []
    byte_re = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
    while len(out) < nmax and heap:
        negc, p = heapq.heappop(heap)
        if cnt.get(p, 0) != -negc: 
            if cnt.get(p, 0) > 0: heapq.heappush(heap, (-cnt[p], p))
            continue
        a, b = p; s = a + b
        if -negc < min_count: break
        if not ok(s) or s in VSTR or s in forbid or byte_re.match(a) or byte_re.match(b) or a.endswith("\n") or b.endswith("\n"):
            cnt[p] = 0; continue
        out.append((a, b, -negc)); forbid.add(s)
        for wi in list(idx[p]):
            w = words[wi]; f = freq[wi]; i = 0; nw = []
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    nw.append(s); i += 2
                else:
                    nw.append(w[i]); i += 1
            if nw == w: continue
            for q in zip(w, w[1:]):
                cnt[q] -= f
            for q in zip(nw, nw[1:]):
                cnt[q] += f; idx[q].add(wi); heapq.heappush(heap, (-cnt[q], q))
            words[wi] = nw
        cnt[p] = 0
    return [(a, b) for a, b, _ in out], [c for _, _, c in out]

def stats(tok, texts):
    enc = tok.encode_batch(texts, add_special_tokens=False)
    return enc

# ---------- corpora ----------
corp = {}
wiki = lambda lang: to_lines(parq(f"datasets--wikimedia--wikipedia/snapshots/*/20231101.{lang}/*.parquet", "text"))
corp["amh"] = wiki("am") + to_lines(parq("datasets--HuggingFaceFW--finetranslations/snapshots/*/data/amh_Ethi/*.parquet", "og_full_text", maxdocs=40000))
corp["tir"] = wiki("ti") + to_lines(parq("datasets--HuggingFaceFW--finetranslations/snapshots/*/data/tir_Ethi/*.parquet", "og_full_text", maxdocs=72009))
corp["fon"] = wiki("fon") + to_lines(parq("datasets--HuggingFaceFW--fineweb-2/snapshots/*/data/fon_Latn/train/*.parquet", "text")) + flores("dev", "fon_Latn")
# cap characters per language to ~25M
for k in corp:
    tot = 0; keep = []
    for l in corp[k]:
        tot += len(l); keep.append(l)
        if tot > 25e6: break
    corp[k] = keep
EVAL = {"amh": "amh_Ethi", "tir": "tir_Ethi", "fon": "fon_Latn", "hau": "hau_Latn", "yor": "yor_Latn", "swh": "swh_Latn", "kin": "kin_Latn", "ibo": "ibo_Latn", "eng": "eng_Latn",
        "fra": "fra_Latn", "ewe": "ewe_Latn", "bam": "bam_Latn", "lin": "lin_Latn", "wol": "wol_Latn", "zul": "zul_Latn", "kik": "kik_Latn", "som": "som_Latn"}
test = {k: flores("devtest", v) for k, v in EVAL.items()}
info = {}
for k, L in corp.items():
    big = "\n".join(L)
    hit = sum(1 for s in test[k] if len(s) > 15 and s in big)
    info[k] = dict(lines=len(L), chars=len(big), devtest_sentences_found_verbatim_in_training_text=hit)
json.dump(info, open(os.path.join(OUT, "swap_sim_corpus.json"), "w"), indent=1); print(info, flush=True)

# ---------- simulation ----------
VARIANTS = {"amh": ["amh"], "tir": ["tir"], "geez(amh+tir)": ["amh", "tir"], "fon": ["fon"], "fon_strict": ["fon"]}
rows, retok = [], []
tokstr_base = [None] * 262144
for s, i in BASE_VOCAB.items(): tokstr_base[i] = s
nllb = Tokenizer.from_file(find("facebook/nllb-200-distilled-600M", "tokenizer.json"))
nllb_tok = {k: sum(len(e.ids) for e in nllb.encode_batch(v, add_special_tokens=False)) for k, v in test.items()}
nwords = {k: sum(len(t.split()) for t in v) for k, v in test.items()}
nchars = {k: sum(len(t) for t in v) for k, v in test.items()}
base_enc = {k: [e.ids for e in base_tok.encode_batch(v, add_special_tokens=False)] for k, v in test.items()}
byte_ids = np.zeros(262144, bool)
for s, i in BASE_VOCAB.items(): byte_ids[i] = bool(re.match(r"^<0x[0-9A-Fa-f]{2}>$", s))
for vname, langs in VARIANTS.items():
    t0 = time.time()
    lines = []
    if len(langs) == 1: lines = corp[langs[0]]
    else:   # balance: equal numbers of characters from each language
        capc = min(sum(len(l) for l in corp[l]) for l in langs)
        for l in langs:
            tot = 0
            for x in corp[l]:
                lines.append(x); tot += len(x)
                if tot > capc: break
    # step 0: letters without a token
    ch = collections.Counter(c for l in lines for c in l)
    missing = [(c, n) for c, n in ch.items() if n >= 3 and c not in VSTR and c not in ("▁", " ") and (0x1200 <= ord(c) <= 0x139F or 0x2D80 <= ord(c) <= 0x2DDF or langs == ["fon"] and ord(c) > 127 and c.isalpha())]
    # the single character must also be absent as "▁c" is irrelevant: tokens are looked up as plain characters
    missing = [c for c, n in sorted(missing, key=lambda x: -x[1])]
    t1, _ = build(missing, [])
    enc = t1.encode_batch(lines, add_special_tokens=False)
    tokstr1 = {}
    j1 = json.loads(t1.to_str()); 
    for s, i in j1["model"]["vocab"].items(): tokstr1[i] = s
    wc = words_from_ids([e.ids for e in enc], tokstr1)
    ntok = sum(len(e.ids) for e in enc)
    nmax = max(NS)
    if vname == "fon_strict":      # merged string must contain a letter beyond Latin-1 that Yoruba does not use (e.g. U+0256 ɖ, U+025B ɛ, U+0254 ɔ, U+014B ŋ, U+0263 ɣ, U+01CE ǎ)
        ok = lambda s_: any(ord(c) > 255 and c.isalpha() and c not in "ẹọṣẸỌṢ" for c in s_)
    elif vname == "fon":
        ok = lambda s_: any(ord(c) > 127 and c.isalpha() for c in s_)        # must contain a non-ASCII letter (ɖ ɛ ɔ ǎ ...): no merge a pure-ASCII text could trigger
    else:
        ok = lambda s_: any(0x1200 <= ord(c) <= 0x139F for c in s_) and not any(c.isascii() and c.isalpha() for c in s_)   # must contain an Ethiopic char and no Latin letter
    merges, mc = learn_merges(wc, nmax - len(missing), set(), ok)
    print(vname, "corpus tokens", ntok, "word types", len(wc), "missing chars", len(missing), "merges", len(merges), "t=%.0fs" % (time.time() - t0), flush=True)
    json.dump(dict(chars=missing, merges=merges), open(os.path.join(OUT, "swap_sim_merges_%s.json" % re.sub("[^a-z]", "_", vname)), "w"), ensure_ascii=False)
    if os.environ.get("SIM_STOP_AFTER_LEARN"): continue
    for N in NS:
        nch = min(N, len(missing)); nm = min(max(0, N - nch), len(merges))
        tok, ent = build(missing[:nch], merges[:nm])
        for k in test:
            enc_t = tok.encode_batch(test[k], add_special_tokens=False)
            ids = [e.ids for e in enc_t]
            L = np.array([len(x) for x in ids]); Lb = np.array([len(x) for x in base_enc[k]])
            bf = sum(int(byte_ids[np.array(x, int)].sum()) for x in ids) if True else 0
            changed = float(np.mean([a != b for a, b in zip(ids, base_enc[k])]))
            dec_ok = None
            if N in (1000, 3000, 8000) and k in ("amh", "tir", "fon", "hau", "eng", "fra", "ewe", "yor"):
                dec_ok = float(np.mean([tok.decode(a) == base_tok.decode(b) for a, b in zip(ids, base_enc[k])]))
            rows.append(dict(variant=vname, N=N, n_chars=nch, n_merges=nm, eval_lang=k, tok_per_sent=L.mean(), tok_per_word=L.sum() / nwords[k], tok_per_char=L.sum() / nchars[k],
                             byte_fallback_share=bf / L.sum(), base_tok_per_sent=Lb.mean(), change_vs_base=1 - L.sum() / Lb.sum(), over_nllb=L.sum() / nllb_tok[k],
                             sents_with_changed_tokenisation=changed, decode_roundtrip_equal_to_base=dec_ok))
        if N == max(NS):
            with open(os.path.join(OUT, f"swap_sim_new_tokens_{re.sub('[^a-z]','_',vname)}.txt"), "w", encoding="utf-8") as f:
                for i, s in ent: f.write(f"{i}\t{s}\n")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "swap_sim.csv"), index=False, float_format="%.5f")
df = pd.DataFrame(rows)
print(df[df.eval_lang.isin(["amh", "tir", "fon"]) & df.N.isin([0, 1000, 3000, 8000])].pivot_table(index=["variant", "eval_lang"], columns="N", values="tok_per_sent").round(2).to_string())
