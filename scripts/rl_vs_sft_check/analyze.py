"""Metrics + bootstrap for the RL-vs-SFT weak-signal check.
python analyze.py PRED_DIR OUT.json  [name=file ...]   (default names: sft rl1 b64 c2c + stage branches present)"""
import sys, os, json, re, unicodedata, random, zlib
import numpy as np
sys.path.insert(0, "/path/to/repo/gemma_mt"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sacrebleu.metrics import CHRF
from gemma_mt.languages import LANGUAGES
from gemma_mt.rl.lid import siblings_of
from groups import *

CH = CHRF(word_order=2)
def nfc(s): return unicodedata.normalize("NFC", s or "")
def fold(s): return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).casefold()

def f_from_stats(S):
    eps = 1e-16; factor = CH.beta ** 2; ap = ar = 0.0; eo = 0
    for i in range(CH.order):
        nh, nr, nm = S[3*i:3*i+3]
        if nh > 0 and nr > 0:
            ap += nm / nh; ar += nm / nr; eo += 1
    if eo == 0: return 0.0
    ap /= eo; ar /= eo
    return 100 * (1 + factor) * ap * ar / (factor * ap + ar) if ap + ar else 0.0

def row_stats(hyps, refs):
    return np.array(CH._extract_corpus_statistics([nfc(h) for h in hyps], [[nfc(r) for r in refs]]), dtype=np.float64)
def corpus_chrf(stats): return f_from_stats(stats.sum(0)) if len(stats) else float("nan")
def sent_chrf(h, r): return CH.sentence_score(nfc(h), [nfc(r)]).score

# ---------------------------------------------------------------- anchors
WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.U)
NUM = re.compile(r"\d+(?:[.,]\d+)*")
def numkey(s): return re.sub(r"[.,]", "", s)
def anchors(src):
    nums = [numkey(m.group()) for m in NUM.finditer(src)]
    names = []
    for m in WORD.finditer(src):
        w = m.group()
        if len(w) < 3 or not w[0].isupper(): continue
        pre = src[:m.start()].rstrip(" \t\"'“‘([{«")
        if not pre or pre[-1] in ".!?:;\n…" or src[:m.start()].rstrip(" \t").endswith("\n"): continue   # sentence-initial
        names.append(fold(w))
    return nums, names
def edit_le1(a, b):
    if a == b: return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1: return False
    if la == lb: return sum(x != y for x, y in zip(a, b)) <= 1
    if la > lb: a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]: i += 1
    return a[i:] == b[i+1:]
def name_in(name, ftxt, toks):
    if name in ftxt: return True
    if len(name) < 5: return False
    L = len(name)
    for t in toks:
        if len(t) < L - 1: continue
        for w in (L - 1, L, L + 1):
            for s in range(0, len(t) - w + 1):
                if edit_le1(name, t[s:s+w]): return True
    return False
def anchor_hits(src, tgt):
    nums, names = anchors(src)
    ft = fold(tgt); toks = [t for t in re.findall(r"\w+", ft)]
    tnums = {numkey(m.group()) for m in NUM.finditer(tgt)}
    hn = sum(1 for n in nums if n in tnums)
    hm = sum(1 for n in names if name_in(n, ft, toks))
    return len(nums), hn, len(names), hm

# ---------------------------------------------------------------- degeneration / boilerplate
def words(s): return re.findall(r"\w+", fold(s), re.U)
def ngrams(w, n): return {tuple(w[i:i+n]) for i in range(len(w) - n + 1)}
LOOPRE = re.compile(r"(.{2,40}?)\1{4,}", re.S)
def looping(t):
    w = words(t)
    if len(w) >= 12:
        from collections import Counter
        c = Counter(tuple(w[i:i+3]) for i in range(len(w) - 2))
        if c and max(c.values()) >= 4: return True
    return bool(LOOPRE.search(t))
def boiler_rate(texts):
    """share of texts (>=5 words) that share a word 5-gram with at least one OTHER text"""
    gs = [ngrams(words(t), 5) for t in texts]
    idx = [i for i, t in enumerate(texts) if len(words(t)) >= 5]
    from collections import Counter
    cnt = Counter(g for i in idx for g in gs[i])
    hit = sum(1 for i in idx if any(cnt[g] >= 2 for g in gs[i]))
    return hit / len(idx) if idx else float("nan"), len(idx)

def lid_ok(iso, top1):
    L = LANGUAGES[iso]; own = set(L.glotlid); sib = set(siblings_of(L.glotlid)) if L.glotlid else set()
    return top1 in own, top1 in own or top1 in sib

# ---------------------------------------------------------------- load
pdir, out_p = sys.argv[1], sys.argv[2]
names = [a.split("=") for a in sys.argv[3:]] or [(n, f"lid_{n}.jsonl") for n in ["sft", "rl1", "b64", "c2c"]]
data = {}
for n, fn in names:
    p = os.path.join(pdir, fn)
    if not os.path.exists(p): print("missing", p); continue
    data[n] = [json.loads(l) for l in open(p, encoding="utf-8")]
cks = list(data)
base = data[cks[0]]
key = lambda r: (r["iso"], r["bench"], r["id"])
for n in cks: assert [key(r) for r in data[n]] == [key(r) for r in base], n
langs = [i for i in ALL_LANGS if any(r["iso"] == i for r in base)]
idx_by = {i: [k for k, r in enumerate(base) if r["iso"] == i] for i in langs}

# per language reference-side facts
res = {"checkpoints": cks, "langs": {}}
rng_global = random.Random(99)
cache_stats = {}
for iso in langs:
    ix = idx_by[iso]
    has_ref = base[ix[0]]["reference"] is not None
    L = {"group": GROUP[iso], "n": len(ix), "has_ref": has_ref,
         "benches": {b: sum(1 for k in ix if base[k]["bench"] == b) for b in sorted({base[k]["bench"] for k in ix})}}
    srcs = [base[k]["english"] for k in ix]
    refs = [base[k]["reference"] for k in ix] if has_ref else None
    L["n_no_pair"] = sum(1 for k in ix if base[k]["pair_seen"] is False)
    L["n_clean"] = sum(1 for k in ix if base[k]["contaminated"] is False)
    # human baselines
    hb = {}
    if has_ref:
        c = [anchor_hits(s, r) for s, r in zip(srcs, refs)]
        tn = sum(x[0] for x in c); tm = sum(x[2] for x in c)
        hb["anchor_num_n"] = tn; hb["anchor_name_n"] = tm
        hb["anchor_recall"] = (sum(x[1] for x in c) + sum(x[3] for x in c)) / max(1, tn + tm) if tn + tm else None
        lids = [base[k].get("lid_ref") for k in ix]
        ok = [lid_ok(iso, l[0][0]) for l in lids if l]
        hb["lid_strict"] = float(np.mean([a for a, b in ok])) if ok else None
        hb["lid_sib"] = float(np.mean([b for a, b in ok])) if ok else None
        hb["boiler"], hb["boiler_n"] = boiler_rate(refs[:100])
    L["human"] = hb
    # fixed 40-source sample + derangement
    r40 = random.Random(5 + sum(map(ord, iso)))
    pick = list(range(len(ix))); r40.shuffle(pick); pick = pick[:40]
    n40 = len(pick); shift = r40.randint(1, max(1, n40 - 1)) if n40 > 1 else 0
    sigma = [pick[(j + shift) % n40] for j in range(n40)] if n40 > 1 else pick
    L["_pick"] = pick; L["_sigma"] = sigma
    if has_ref and n40 > 1:
        hb["cross_ref_human"] = float(np.mean([sent_chrf(refs[a], refs[b]) for a, b in zip(pick, sigma)]))
    L["models"] = {}
    for n in cks:
        rows = [data[n][k] for k in ix]
        hyp = [r["hyp"] or "" for r in rows]
        M = {}
        M["finished"] = float(np.mean([bool(r["finished"]) for r in rows]))
        M["empty"] = float(np.mean([not h.strip() for h in hyp]))
        M["prefix_ok"] = float(np.mean([r["prefix_status"] == "ok" for r in rows]))
        M["loop"] = float(np.mean([looping(h) for h in hyp]))
        M["degenerate"] = float(np.mean([(not r["finished"]) or (not h.strip()) or looping(h) for r, h in zip(rows, hyp)]))
        cp = [sent_chrf(h, s) >= 75 if h.strip() else False for h, s in zip(hyp, srcs)]
        M["copy"] = float(np.mean(cp))
        li = [lid_ok(iso, r["lid_hyp"][0][0]) if r.get("lid_hyp") else (False, False) for r in rows]
        M["lid_strict"] = float(np.mean([a for a, b in li])); M["lid_sib"] = float(np.mean([b for a, b in li]))
        M["lid_en"] = float(np.mean([bool(r.get("lid_hyp")) and r["lid_hyp"][0][0] == "eng_Latn" for r in rows]))
        c = [anchor_hits(s, h) for s, h in zip(srcs, hyp)]
        tn = sum(x[0] for x in c); tm = sum(x[2] for x in c)
        M["anchor_n"] = tn + tm
        M["anchor_recall"] = (sum(x[1] for x in c) + sum(x[3] for x in c)) / (tn + tm) if tn + tm else None
        M["anchor_num_recall"] = sum(x[1] for x in c) / tn if tn else None
        M["anchor_name_recall"] = sum(x[3] for x in c) / tm if tm else None
        M["boiler"], M["boiler_n"] = boiler_rate(hyp[:100])
        M["mean_len_words"] = float(np.mean([len(words(h)) for h in hyp]))
        if has_ref:
            st = row_stats(hyp, refs); cache_stats[(n, iso)] = st
            npm = np.array([base[k]["pair_seen"] is False for k in ix]); cl = np.array([base[k]["contaminated"] is False for k in ix])
            M["chrf_all"] = corpus_chrf(st)
            M["chrf_no_pair"] = corpus_chrf(st[npm]) if npm.sum() >= 15 else None
            M["chrf_clean"] = corpus_chrf(st[cl]) if cl.sum() >= 15 else None
            M["chrf_unfinished_free"] = None
            M["len_ratio"] = float(sum(len(nfc(h)) for h in hyp) / max(1, sum(len(nfc(r)) for r in refs)))
            M["own"] = float(np.mean([sent_chrf(hyp[a], refs[a]) for a in pick]))
            M["cross_ref"] = float(np.mean([sent_chrf(hyp[a], refs[b]) for a, b in zip(pick, sigma)])) if n40 > 1 else None
        if n40 > 1:
            M["cross_hyp"] = float(np.mean([sent_chrf(hyp[a], hyp[b]) for a, b in zip(pick, sigma)]))
        # 5-gram attractor among the 40
        M["boiler40"], _ = boiler_rate([hyp[a] for a in pick])
        L["models"][n] = M
    if has_ref:
        L["human"]["boiler40"], _ = boiler_rate([refs[a] for a in pick])
    res["langs"][iso] = L
    print(iso, L["group"], L["n"], {n: (round(L["models"][n].get("chrf_all", float('nan')), 1)) for n in cks}, flush=True)

# ---------------------------------------------------------------- group bootstrap
B = 1000
rng = np.random.default_rng(2026)
def group_langs(g, need_ref=True, subset="all"):
    out = []
    for iso in langs:
        L = res["langs"][iso]
        if L["group"] != g or not L["has_ref"]: continue
        if subset == "no_pair" and L["n_no_pair"] < 15: continue
        if subset == "clean" and L["n_clean"] < 15: continue
        out.append(iso)
    return out
def subset_mask(iso, subset):
    ix = idx_by[iso]
    if subset == "all": return np.ones(len(ix), bool)
    if subset == "no_pair": return np.array([base[k]["pair_seen"] is False for k in ix])
    return np.array([base[k]["contaminated"] is False for k in ix])
res["groups"] = {}
for g in ["consensus", "blend", "control"]:
    G = {}
    for subset in ["all", "no_pair", "clean"]:
        ls = group_langs(g, subset=subset)
        if not ls: continue
        masks = {i: np.where(subset_mask(i, subset))[0] for i in ls}
        pt = {n: float(np.mean([f_from_stats(cache_stats[(n, i)][masks[i]].sum(0)) for i in ls])) for n in cks}
        boots = {n: [] for n in cks}
        for b in range(B):
            sel = rng.integers(0, len(ls), len(ls))
            vals = {n: [] for n in cks}
            for j in sel:
                m = masks[ls[j]]; r_ = m[rng.integers(0, len(m), len(m))]
                for n in cks: vals[n].append(f_from_stats(cache_stats[(n, ls[j])][r_].sum(0)))
            for n in cks: boots[n].append(np.mean(vals[n]))
        boots = {n: np.array(v) for n, v in boots.items()}
        ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        G[subset] = {"n_langs": len(ls), "langs": ls,
                     "mean": pt, "ci": {n: ci(boots[n]) for n in cks},
                     "delta_vs_first": {n: {"mean": pt[n] - pt[cks[0]], "ci": ci(boots[n] - boots[cks[0]])} for n in cks[1:]},
                     "delta_vs_prev": {cks[j]: {"mean": pt[cks[j]] - pt[cks[j-1]], "ci": ci(boots[cks[j]] - boots[cks[j-1]])} for j in range(1, len(cks))}}
        print(g, subset, len(ls), {n: round(pt[n], 2) for n in cks}, flush=True)
    # language-level metrics (bootstrap over languages only)
    for met in ["finished", "degenerate", "copy", "lid_strict", "lid_sib", "lid_en", "anchor_recall", "boiler", "boiler40", "cross_hyp", "own", "cross_ref", "prefix_ok", "len_ratio"]:
        ls = [i for i in langs if res["langs"][i]["group"] == g and all(res["langs"][i]["models"][n].get(met) is not None for n in cks)
              and (met not in ("own", "cross_ref", "len_ratio") or res["langs"][i]["has_ref"])
              and (met != "anchor_recall" or res["langs"][i]["models"][cks[0]]["anchor_n"] >= 10)]
        if not ls: continue
        mat = np.array([[res["langs"][i]["models"][n][met] for n in cks] for i in ls], dtype=float)
        bm = np.array([mat[rng.integers(0, len(ls), len(ls))].mean(0) for _ in range(B)])
        ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
        entry = {"n_langs": len(ls), "mean": {n: float(mat[:, j].mean()) for j, n in enumerate(cks)}, "ci": {n: ci(bm[:, j]) for j, n in enumerate(cks)},
                 "delta_vs_first": {n: {"mean": float(mat[:, j].mean() - mat[:, 0].mean()), "ci": ci(bm[:, j] - bm[:, 0])} for j, n in enumerate(cks) if j}}
        if met in ("anchor_recall", "boiler", "boiler40", "lid_strict", "lid_sib"):
            hk = {"anchor_recall": "anchor_recall", "boiler": "boiler", "boiler40": "boiler40", "lid_strict": "lid_strict", "lid_sib": "lid_sib"}[met]
            hv = [res["langs"][i]["human"].get(hk) for i in ls]
            entry["human_mean"] = float(np.mean([h for h in hv if h is not None])) if any(h is not None for h in hv) else None
        G.setdefault("metrics", {})[met] = entry
    res["groups"][g] = G

# ---------------------------------------------------------------- per-language paired deltas
for iso in langs:
    L = res["langs"][iso]
    if not L["has_ref"]: continue
    for subset in ["all", "no_pair"]:
        m = np.where(subset_mask(iso, subset))[0]
        if len(m) < 15: continue
        for n in cks[1:]:
            ds = []
            a, b = cache_stats[(cks[0], iso)][m], cache_stats[(n, iso)][m]
            for _ in range(400):
                r_ = rng.integers(0, len(m), len(m)); ds.append(f_from_stats(b[r_].sum(0)) - f_from_stats(a[r_].sum(0)))
            L.setdefault("delta", {}).setdefault(subset, {})[n] = {"mean": f_from_stats(b.sum(0)) - f_from_stats(a.sum(0)), "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))], "n": int(len(m))}
for iso in langs:
    res["langs"][iso].pop("_pick"); res["langs"][iso].pop("_sigma")
json.dump(res, open(out_p, "w"), indent=1, ensure_ascii=False)
print("wrote", out_p)
