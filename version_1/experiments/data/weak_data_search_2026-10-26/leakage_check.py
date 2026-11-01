#!/usr/bin/env python3
"""English-side leakage check of candidate sources against evaluation English text.
Usage: leakage_check.py --pool-dir DIR_OF_EVAL_ENGLISH_FILES --cand name=path[:col] ...
Pool files: *.txt (one sentence per line) or *.jsonl (fields trg / trgs / english / text).
Candidate files: .txt (one English sentence per line), .tsv (column index after ':'), .jsonl (field after ':').
Reports per candidate: n sentences, exact-sentence hits, sentences sharing >=1 word 8-gram with the pool.
On the box, point --pool-dir at FLORES+ dev/devtest English, NTREX, SmolSent/SmolDoc, MAFAND, SALT test, AfriScience English.
"""
import argparse, ast, glob, json, os, re, sys
def norm(s): return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', s.lower())).strip()
def grams(s, n=8):
    w = norm(s).split()
    return {' '.join(w[i:i+n]) for i in range(max(0, len(w)-n+1))}
def load_pool(d):
    sents = []
    for f in sorted(glob.glob(os.path.join(d, '*'))):
        if f.endswith('.txt'):
            sents += [l.rstrip('\n') for l in open(f, encoding='utf-8')]
        elif f.endswith('.jsonl'):
            for l in open(f, encoding='utf-8'):
                try: r = json.loads(l)
                except Exception: continue
                for k in ('trg', 'english', 'text'):
                    if isinstance(r.get(k), str) and r.get('sl', 'x') != 'en': sents.append(r[k])
                v = r.get('trgs')
                if isinstance(v, str):
                    try: v = ast.literal_eval(v)
                    except Exception: v = []
                if isinstance(v, list) and r.get('tl') == 'en': sents += [x for x in v if isinstance(x, str) and len(x.split()) >= 4]
    return sents
def load_cand(spec):
    name, rest = spec.split('=', 1)
    path, _, col = rest.partition(':')
    out = []
    if path.endswith('.jsonl'):
        for l in open(path, encoding='utf-8'):
            try: out.append(str(json.loads(l)[col or 'text']))
            except Exception: pass
    elif path.endswith('.tsv'):
        for i, l in enumerate(open(path, encoding='utf-8')):
            p = l.rstrip('\n').split('\t')
            if len(p) > int(col or 0): out.append(p[int(col or 0)])
    else:
        out = [l.rstrip('\n') for l in open(path, encoding='utf-8')]
    return name, out
ap = argparse.ArgumentParser(); ap.add_argument('--pool-dir', required=True); ap.add_argument('--cand', nargs='+', required=True); ap.add_argument('--max', type=int, default=0)
a = ap.parse_args()
pool = load_pool(a.pool_dir)
pset = {norm(s) for s in pool if s.strip()}
pg = set()
for s in pool: pg |= grams(s)
print(f'pool: {len(pset)} sentences, {len(pg)} 8-grams', file=sys.stderr)
print('candidate,n,exact_hits,ngram8_hits,exact_pct,ngram8_pct')
for spec in a.cand:
    name, sents = load_cand(spec)
    if a.max: sents = sents[:a.max]
    sents = [s for s in sents if s.strip()]
    ex = sum(norm(s) in pset for s in sents)
    ng = sum(bool(grams(s) & pg) for s in sents)
    n = max(1, len(sents))
    print(f'{name},{len(sents)},{ex},{ng},{100*ex/n:.1f},{100*ng/n:.1f}')
