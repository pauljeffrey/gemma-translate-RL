"""Build the evaluation pool: per language up to ~300 English->X sentence pairs with human references (+ flags), and a source-only pool for
languages without any reference. Run on the 3090 box:  python build_pool.py TEST_SETS_DIR OUT.jsonl"""
import sys, json, random, os, unicodedata
sys.path.insert(0, "/workspace/rlcheck"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma_mt.eval import benchmarks as B
from gemma_mt.languages import LANGUAGES
from groups import *

TS, OUT = sys.argv[1], sys.argv[2]
TOTAL = 300
flags = {}
for b in SENT_BENCHES:
    p = os.path.join(TS, "_contamination", f"contamination_{b}.json")
    flags[b] = json.load(open(p)) if os.path.exists(p) else None

def norm(s): return " ".join(unicodedata.normalize("NFKC", s).casefold().split())

pool, nref, report = [], [], {}
flores_en = None
for iso in ALL_LANGS:
    cands = []
    for b in SENT_BENCHES:
        if not B.code_for(b, iso): continue
        try:
            rows = B.load_benchmark(b, path=os.path.join(TS, BENCH_PATH[b]), isos=[iso])
        except Exception as e:
            print("load fail", b, iso, repr(e)[:120]); rows = []
        rows = [r for r in rows if r["iso"] == iso]
        if rows: cands.append((b, rows))
    if not cands:
        report[iso] = {}; continue
    per = max(75, TOTAL // len(cands))
    rng = random.Random(1234 + sum(map(ord, iso)))
    seen = set(); chosen = []
    for b, rows in cands:
        f = flags[b]
        for r in rows:
            k = f"{iso}:{r['id']}"
            r["contaminated"] = None if f is None else bool(f["exact"].get(k)) if k in f["exact"] else None
            r["pair_seen"] = None if f is None else (bool(f["pair"].get(k)) if k in f["pair"] else None)
        rng.shuffle(rows)
        samp = rows[:per]
        np_in = [r for r in samp if r["pair_seen"] is False]
        extra = [r for r in rows[per:] if r["pair_seen"] is False]
        if len(np_in) < min(60, per) and extra:
            samp = samp + extra[: min(60, per) - len(np_in)]
        # also guarantee some clean (English unseen) rows
        cl_in = [r for r in samp if r["contaminated"] is False]
        extra_c = [r for r in rows if r["contaminated"] is False and r not in samp]
        if len(cl_in) < 40 and extra_c:
            samp = samp + extra_c[: 40 - len(cl_in)]
        for r in samp:
            n = norm(r["english"])
            if n in seen: continue
            seen.add(n); chosen.append(r)
    report[iso] = {b: sum(1 for r in chosen if r["benchmark"] == b) for b, _ in cands}
    for r in chosen:
        pool.append({"iso": iso, "bench": r["benchmark"], "id": r["id"], "english": r["english"], "reference": r["reference"],
                     "contaminated": r["contaminated"], "pair_seen": r["pair_seen"]})
# source-only languages: FLORES+ devtest English sentences (same 80 for all), no reference
have = {r["iso"] for r in pool}
en = B._flores_local(os.path.join(TS, "flores_plus"), "eng_Latn", "devtest")
rng = random.Random(77)
sel = rng.sample(en, 80)
for iso in ALL_LANGS:
    if iso in have: continue
    for r in sel:
        pool.append({"iso": iso, "bench": "flores_src_only", "id": str(r["id"]), "english": r["text"].strip(), "reference": None,
                     "contaminated": True, "pair_seen": None})
    report[iso] = {"flores_src_only": len(sel)}
with open(OUT, "w", encoding="utf-8") as f:
    for r in pool: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps(report)); print(len(pool), "rows")
