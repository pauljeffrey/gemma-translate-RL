"""R12: best-of-n / MBR / QE selection from sampled pools (gens with n candidates per sentence). venv_gemma, CPU (+ AfriCOMET worker for QE).
Utility of MBR = sentence chrF++ (sacrebleu-equal, gemma_mt.rl.consensus), candidate i scored against the OTHER candidates, as in R5."""
import json, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
import sl
from gemma_mt.rl import consensus as cons

def _mat(texts_rows):
    out = []
    for texts in texts_rows:
        P = [cons.Profile(t) for t in texts]; m = len(P); M = np.zeros((m, m), np.float32)
        for a in range(m):
            for b in range(a + 1, m):
                mm = cons.matches(P[a], P[b])
                M[a, b] = cons.f_score(P[a].totals, P[b].totals, mm); M[b, a] = cons.f_score(P[b].totals, P[a].totals, mm)
        out.append(M)
    return out

def pool_data(name, pool=None):
    """(texts[N][n], fins[N][n], ntoks[N][n], M[N,n,n]) ; M[i,a,b] = chrF++ of candidate a (hyp) against candidate b (ref)"""
    p = f"{sl.R}/cache/pool_{name}.pkl"
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    d = sl.get_dec(name)["dec"]
    texts = [[sl.nfc(c["hyp"]) for c in row] for row in d]; fins = np.array([[c["finished"] for c in row] for row in d]); nt = np.array([[c["ntok"] for c in row] for row in d])
    own = pool is None; pool = pool or Pool(10)
    chunks = [texts[i:i + 100] for i in range(0, len(texts), 100)]
    M = np.concatenate([np.stack(x) for x in pool.map(_mat, chunks)])
    if own: pool.close()
    out = {"texts": texts, "fin": fins, "ntok": nt, "M": M, "seconds": sl.get_dec(name)["seconds"], "tokens": sl.get_dec(name)["tokens"]}
    pickle.dump(out, open(p, "wb")); return out

def oracle_scores(name, pool=None):
    """sentence chrF++ of every candidate against the real reference (diagnostic: oracle selection, mean candidate quality)"""
    p = f"{sl.R}/cache/oracle_{name}.npy"
    if os.path.exists(p): return np.load(p)
    pd = pool_data(name, pool); n = pd["fin"].shape[1]
    arr = np.array([[cons.chrf_sentence(pd["texts"][k][a], sl.REFS[k]) for a in range(n)] for k in range(sl.N)], np.float32)
    np.save(p, arr); return arr

def select(pd, n, method, greedy=None, qe=None, w_qe=0.5, n_evid=None, qe_mask=None):
    """index of the picked candidate (among the first n; `-1` = the greedy output when included) for every sentence
    method: first | mbr | mbr_g (greedy output joins candidates and evidence) | qe | mbrqe (mean of z-scored MBR and QE) | oracle"""
    M, fin = pd["M"][:, :n, :n], pd["fin"][:, :n]
    picks = np.zeros(sl.N, int)
    for k in range(sl.N):
        ok = fin[k].copy()
        if not ok.any(): ok[:] = True
        if method == "first": picks[k] = int(np.argmax(ok)); continue
        sc = (M[k].sum(1) / max(n - 1, 1)).astype(float)
        if method in ("qe", "mbrqe") and qe is not None and qe_mask[k]:
            q = qe[k, :n]
            if method == "qe": sc = q.astype(float)
            else: sc = w_qe * (q - q.mean()) / (q.std() + 1e-9) + (1 - w_qe) * (sc - sc.mean()) / (sc.std() + 1e-9)
        sc = np.where(ok, sc, -1e9); picks[k] = int(np.argmax(sc))
    return picks

def mbr_with_greedy(pd, n, greedy_txt, greedy_fin):
    """MBR over n candidates + the greedy output (n + 1 candidates and evidence); returns (hyps, fins, picked_greedy_mask)"""
    from gemma_mt.rl import consensus as C
    hy, fn, pg = [], [], []
    for k in range(sl.N):
        T = list(pd["texts"][k][:n]) + [greedy_txt[k]]; P = [C.Profile(t) for t in T]; m = len(P)
        acc = np.zeros(m)
        for a in range(m):
            for b in range(a + 1, m):
                mm = C.matches(P[a], P[b]); acc[a] += C.f_score(P[a].totals, P[b].totals, mm); acc[b] += C.f_score(P[b].totals, P[a].totals, mm)
        f = np.array(list(pd["fin"][k][:n]) + [greedy_fin[k]]); acc = np.where(f | (~f.any()), acc, -1e9); j = int(np.argmax(acc))
        hy.append(T[j]); fn.append(bool(f[j])); pg.append(j == m - 1)
    return hy, fn, np.array(pg)
