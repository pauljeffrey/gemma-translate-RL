"""R12 sentence-level scoring library (venv_gemma, CPU; PYTHONPATH=/workspace/sft_build/gemma_mt_2c:/workspace/sft_build/pylibs).
Same sentences, decoding of completions, chrF++/spBLEU, loop / LID / anchor-recall definitions as R5 (reports/data/final_model_decoding_study/study_score.py);
chrF++/BLEU from summed sacrebleu statistics via vectorised formulas (checked against sacrebleu in selfcheck())."""
import glob, gzip, json, os, pickle, re, sys, time, unicodedata
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, "/workspace/sft_build/pylibs")
from sacrebleu.metrics import CHRF, BLEU
R = "/workspace/jobs/r12"
S = json.load(open(R + "/study_set.json"))
rows, STOP = S["rows"], S["stop_ids"]
N = len(rows)
ISOS = list(dict.fromkeys(r["iso"] for r in rows))
IDX = {i: np.array([k for k, r in enumerate(rows) if r["iso"] == i]) for i in ISOS}
GROUPS = json.load(open(R + "/groups.json"))
GM = {"all": ISOS, **{g: [i for i in ISOS if GROUPS[i] == g] for g in ("strong", "medium", "weak")}}
nfc = lambda s: unicodedata.normalize("NFC", s)
REFS = [nfc(r["reference"]) for r in rows]
os.makedirs(R + "/cache", exist_ok=True)

def chrf_vec(S_):
    S_ = np.asarray(S_, dtype=np.float64)
    nh, nr, nm = S_[..., 0::3], S_[..., 1::3], S_[..., 2::3]
    ok = (nh > 0) & (nr > 0)
    prec = np.where(nh > 0, nm / np.maximum(nh, 1), 0.0); rec = np.where(nr > 0, nm / np.maximum(nr, 1), 0.0)
    eff = ok.sum(-1)
    ap = (prec * ok).sum(-1) / np.maximum(eff, 1); ar = (rec * ok).sum(-1) / np.maximum(eff, 1)
    den = 4 * ap + ar
    return np.where(den > 0, 100 * 5 * ap * ar / np.where(den > 0, den, 1), 0.0)

def bleu_vec(S_):
    S_ = np.asarray(S_, dtype=np.float64)
    sl, rl, corr, tot = S_[..., 0], S_[..., 1], S_[..., 2:6], S_[..., 6:10]
    z = (corr == 0) & (tot > 0); k = np.cumsum(z, -1)
    prec = np.where(tot > 0, np.where(corr > 0, 100.0 * corr / np.maximum(tot, 1), 100.0 / (2.0 ** k * np.maximum(tot, 1))), 0.0)
    lg = np.log(np.maximum(prec, 1e-12) / 100.0)
    bp = np.where(sl < rl, np.exp(1 - rl / np.maximum(sl, 1)), 1.0)
    ok = (tot > 0).all(-1) & (corr[..., 0] > 0)
    return np.where(ok, 100.0 * bp * np.exp(lg.mean(-1)), 0.0)

_tok = None
def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(S["model_path"])
    return _tok

def decode_cfg(g):
    from gemma_mt import format as fmt
    from gemma_mt.rl.text import decode_completion
    t = tok(); out = []
    for k, cs in enumerate(g["cands"]):
        row = []
        for ids in cs:
            d = decode_completion(t, ids, STOP)
            st = fmt.strip_iso_prefix(d.text, rows[k]["iso"])
            row.append({"hyp": st.text, "finished": bool(d.finished), "ntok": len(ids)})
        out.append(row)
    return out

def load_gen(name):
    with gzip.open(f"{R}/gens/gen_{name}.json.gz", "rt") as f:
        return json.load(f)

def get_dec(name):
    p = f"{R}/cache/dec_{name}.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    g = load_gen(name); d = {"dec": decode_cfg(g), "seconds": g["seconds"], "tokens": g["tokens"], "kw": g["kw"], "engine": g.get("engine", "vllm")}
    pickle.dump(d, open(p, "wb")); return d

def _stat_chunk(args):
    hs, rs = args
    chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200")
    a = np.array(chrf._extract_corpus_statistics(hs, [rs]), dtype=float)
    b = np.array(bleu._extract_corpus_statistics(hs, [rs]), dtype=float)
    return a, b

NUM = re.compile(r"\d+(?:[.,]\d+)*")
CAPW = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][\w'’-]*\b")
def anchors(src):
    return NUM.findall(src), [m for m in CAPW.findall(src) if m not in ("I",) and len(m) > 1]
def has_num(t, text): return t in text or re.sub(r"[.,]", "", t) in re.sub(r"(?<=\d)[.,](?=\d)", "", text)
def has_name(n, text): return n.casefold() in text.casefold()
RNUM = [anchors(r["english"]) for r in rows]
REFKEEP = [([n for n in a[0] if has_num(n, REFS[k])], [n for n in a[1] if has_name(n, REFS[k])]) for k, a in enumerate(RNUM)]
_clf = None; _acc = None
def lid_tools():
    global _clf, _acc
    if _clf is None:
        from gemma_mt.rl import lid as L
        from gemma_mt.languages import LANGUAGES
        p = glob.glob("/workspace/sft_build/hf/hub/models--cis-lmu--glotlid/snapshots/*/model_v3.bin")[0]
        _clf = L.GlotLIDClassifier(path=p); known = set(_clf.labels()); _acc = {}
        for i in ISOS:
            own = {x for x in LANGUAGES[i].glotlid if x in known}
            _acc[i] = own | {x for x in L.siblings_of(tuple(own)) if x in known} if own else None
    return _clf, _acc

def _rep3(t):
    w = t.split(); g = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return len(g) != len(set(g))
def _dist2(t):
    w = t.split(); g = [tuple(w[i:i + 2]) for i in range(len(w) - 1)]
    return len(set(g)) / len(g) if g else 1.0

def row_stats(hyps, fins, ntoks, pool=None):
    """per-sentence statistics of one output per sentence (list of N str)"""
    from gemma_mt.rl import guards
    H = [nfc(h) for h in hyps]
    cuts = np.array_split(np.arange(N), 12)
    jobs = [([H[k] for k in c], [REFS[k] for k in c]) for c in cuts]
    own = pool is None
    pool = pool or Pool(8)
    res = pool.map(_stat_chunk, jobs)
    if own: pool.close()
    C = np.concatenate([r[0] for r in res]); B = np.concatenate([r[1] for r in res])
    clf, acc = lid_tools(); lab = clf(H)
    fin = np.array(fins, bool)
    loop = np.array([(not fin[k]) or guards.is_repetitive(H[k]) or guards.is_self_repetitive(H[k]) for k in range(N)])
    empty = np.array([not h.strip() for h in H])
    lid = np.array([np.nan if acc[rows[k]["iso"]] is None else float(lab[k] in acc[rows[k]["iso"]]) for k in range(N)])
    nk = np.array([sum(has_num(n, H[k]) for n in REFKEEP[k][0]) for k in range(N)]); nt = np.array([len(REFKEEP[k][0]) for k in range(N)])
    mk = np.array([sum(has_name(n, H[k]) for n in REFKEEP[k][1]) for k in range(N)]); mt = np.array([len(REFKEEP[k][1]) for k in range(N)])
    lr = np.array([len(H[k]) / max(len(REFS[k]), 1) for k in range(N)])
    rep3 = np.array([_rep3(H[k]) and not _rep3(REFS[k]) for k in range(N)]); dist2 = np.array([_dist2(H[k]) for k in range(N)])
    return {"rep3": rep3, "dist2": dist2, "C": C, "B": B, "fin": fin, "loop": loop, "empty": empty, "lid": lid, "nk": nk, "nt": nt, "mk": mk, "mt": mt, "lr": lr, "ntok": np.array(ntoks), "hyp": H}

def get_row(name, k=0, pool=None):
    """row statistics for candidate k of generation `name` (cached)"""
    p = f"{R}/cache/row_{name}.{k}.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    d = get_dec(name)["dec"]
    rs = row_stats([d[i][k]["hyp"] for i in range(N)], [d[i][k]["finished"] for i in range(N)], [d[i][k]["ntok"] for i in range(N)], pool)
    pickle.dump(rs, open(p, "wb")); return rs

def cached_row(key, fn):
    p = f"{R}/cache/row_{key}.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    rs = fn(); pickle.dump(rs, open(p, "wb")); return rs

def per_lang(rs):
    c = {i: float(chrf_vec(rs["C"][IDX[i]].sum(0))) for i in ISOS}
    b = {i: float(bleu_vec(rs["B"][IDX[i]].sum(0))) for i in ISOS}
    return c, b

BOOT = 1000
_samp = None
def samp():
    global _samp
    if _samp is None:
        rng = np.random.default_rng(0)
        _samp = {i: IDX[i][rng.integers(0, len(IDX[i]), size=(BOOT, len(IDX[i])))] for i in ISOS}
    return _samp

def boot(rs):
    """(BOOT, n_isos) chrF++ and spBLEU per resample (paired across configs: same sentence resamples)"""
    sp = samp(); c = np.zeros((BOOT, len(ISOS))); b = np.zeros((BOOT, len(ISOS)))
    for j, i in enumerate(ISOS):
        c[:, j] = chrf_vec(rs["C"][sp[i]].sum(1)); b[:, j] = bleu_vec(rs["B"][sp[i]].sum(1))
    return c, b

def summary(name, rs, bs, base_bs, seconds=None, tokens=None, engine=None, qe=None):
    """one result dict: macro numbers per group, deltas vs base with 95% CI"""
    c, b = per_lang(rs)
    out = {"name": name, "engine": engine, "seconds": seconds, "tokens": tokens, "per_lang_chrf": c, "per_lang_bleu": b}
    for g, ml in GM.items():
        ix = [ISOS.index(i) for i in ml]
        for m, arr, bb in (("chrf", bs[0], base_bs[0]), ("bleu", bs[1], base_bs[1])):
            pt = float(np.mean([(c if m == "chrf" else b)[i] for i in ml]))
            d = arr[:, ix].mean(1) - bb[:, ix].mean(1)
            out[f"{m}_{g}"] = pt; out[f"d_{m}_{g}"] = float(np.mean(d)); out[f"lo_{m}_{g}"] = float(np.percentile(d, 2.5)); out[f"hi_{m}_{g}"] = float(np.percentile(d, 97.5))
    out["finished"] = float(rs["fin"].mean()); out["loop"] = float(rs["loop"].mean()); out["empty"] = float(rs["empty"].mean())
    out["lid"] = float(np.nanmean([np.nanmean(rs["lid"][IDX[i]]) for i in ISOS if not np.all(np.isnan(rs["lid"][IDX[i]]))]))
    out["num_kept"] = float(rs["nk"].sum() / max(rs["nt"].sum(), 1)); out["name_kept"] = float(rs["mk"].sum() / max(rs["mt"].sum(), 1))
    out["rep3"] = float(rs["rep3"].mean()); out["dist2"] = float(rs["dist2"].mean())
    out["len_ratio"] = float(rs["lr"].mean()); out["mean_tokens"] = float(rs["ntok"].mean())
    out["loop_weak"] = float(np.mean([rs["loop"][IDX[i]].mean() for i in GM["weak"]]))
    return out

def selfcheck():
    g = get_row("greedy"); chrf = CHRF(word_order=2)
    for i in ISOS[:3]:
        ref = chrf.corpus_score([g["hyp"][k] for k in IDX[i]], [[REFS[k] for k in IDX[i]]]).score
        print(i, ref, chrf_vec(g["C"][IDX[i]].sum(0)))

def find_row(name):
    for p in (f"{R}/cache/row_{name}.0.pkl", f"{R}/cache/row_{name}.pkl"):
        if os.path.exists(p): return pickle.load(open(p, "rb"))
    raise FileNotFoundError(name)
