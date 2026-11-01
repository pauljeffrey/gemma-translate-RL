"""best-of-N consensus pick (highest mean chrF++ against the other candidates); same as the R6 harness (a2/assemble.py)."""
import collections
from gemma_mt.eval.longdoc import norm_text

def _grams(text):
    t = norm_text(text)
    chars = t.replace(" ", "")
    cg = [collections.Counter(chars[i:i + n] for i in range(len(chars) - n + 1)) for n in range(1, 7)]
    w = t.split()
    wg = [collections.Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1)) for n in range(1, 3)]
    return cg + wg

def _chrf(h, r, beta=2.0):
    P = R = 0.0
    for a, b in zip(h, r):
        ta, tb = sum(a.values()), sum(b.values())
        if ta == 0 or tb == 0:
            continue
        m = sum((a & b).values())
        P += m / ta; R += m / tb
    P /= 8; R /= 8
    if P + R == 0:
        return 0.0
    return 100 * (1 + beta ** 2) * P * R / (beta ** 2 * P + R)

def consensus_pick(texts):
    g = [_grams(t) for t in texts]
    n = len(texts)
    sc = []
    for i in range(n):
        s = [(_chrf(g[i], g[j]) + _chrf(g[j], g[i])) / 2 for j in range(n) if j != i]
        sc.append(sum(s) / max(len(s), 1))
    return max(range(n), key=lambda i: sc[i]), sc
