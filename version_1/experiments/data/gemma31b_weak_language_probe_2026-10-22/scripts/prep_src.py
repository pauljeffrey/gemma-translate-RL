"""R15 stage 4a (CPU): English source chunks for synthetic en->X generation, hold-out filtered, with BM25 retrieval shots. usage: prep_src.py <iso> [<iso> ...]"""
import json, re, random, math, collections, unicodedata, os, sys, glob, hashlib
sys.path.insert(0, "/workspace/jobs/r10"); sys.path.insert(0, "/workspace/jobs/b1/gemma_mt"); sys.path.insert(0, "/workspace/jobs/r4/gemma_mt_sft/scripts"); sys.path.insert(0, "/workspace/jobs/r4/gemma_mt_sft")
import pandas as pd
import clean_stage as C
R = "/workspace/jobs/r15/probe"; OUT = "/workspace/jobs/r15/synth/src"; os.makedirs(OUT, exist_ok=True)
nfc = lambda s: unicodedata.normalize("NFC", s)
W = re.compile(r"\w+", re.U)
words = lambda s: W.findall(s.lower())
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“‘(\[])")
ho, _ = C.build_holdout()
print("holdout ready", flush=True)
for iso in sys.argv[1:]:
    rng = random.Random(int(hashlib.md5(iso.encode()).hexdigest()[:8], 16))
    # probe English (all tests) to exclude
    probe = set()
    for x in open(f"{R}/tests/{iso}.jsonl"): probe.add(" ".join(words(json.loads(x)["english"])))
    pool = [json.loads(x) for x in open(f"/workspace/jobs/r10/pool/{iso}.jsonl")]
    pool = [dict(english=nfc(p["english"]), target=nfc(p["target"])) for p in pool if p["english"].strip() and p["target"].strip()]
    pw = [words(p["english"]) for p in pool]; poolnorm = {" ".join(w) for w in pw}
    frames = []
    for path, tag in ((f"/workspace/sft_build/gemma/rl_prompts_v3/{iso}.parquet", "rl_v3"), (f"/workspace/jobs/r10/rl_prompts_extra_v1/{iso}.parquet", "extra_v1"), (f"/workspace/jobs/r10/rl_prompts_sci_v2/{iso}.parquet", "sci_v2")):
        if os.path.exists(path):
            df = pd.read_parquet(path); df["pool_tag"] = tag
            if "domain" not in df: df["domain"] = "general"
            if "field" not in df: df["field"] = ""
            frames.append(df[["english", "source_dataset", "licence", "domain", "field", "pool_tag", "length_bin"]])
    D = pd.concat(frames, ignore_index=True)
    D = D[D.length_bin.isin(["b64", "b256", "b512", "b1024"])]
    seen = set(); items = []; st = collections.Counter()
    for r in D.itertuples():
        sents = [s.strip() for s in SPLIT.split(nfc(r.english)) if s.strip()]
        i = 0
        while i < len(sents):
            cap = rng.randint(18, 70); cur = []; n = 0
            while i < len(sents) and (not cur or n + len(sents[i].split()) <= cap):
                cur.append(sents[i]); n += len(sents[i].split()); i += 1
            txt = " ".join(cur); nw = len(txt.split())
            if nw < 5 or nw > 80: st["len"] += 1; continue
            if sum(c.isalpha() for c in txt) < 0.6 * len(txt): st["nonalpha"] += 1; continue
            k = " ".join(words(txt))
            if k in seen: st["dup"] += 1; continue
            seen.add(k)
            if k in probe or k in poolnorm: st["probe_or_pool"] += 1; continue
            h = ho.hit(txt)
            if h: st["holdout"] += 1; continue
            items.append(dict(english=txt, source_dataset=r.source_dataset, licence=r.licence, domain=r.domain, field=r.field, pool_tag=r.pool_tag))
    # round-robin over (source_dataset, domain) strata so that any prefix is diverse
    by = collections.defaultdict(list)
    for it in items: by[(it["source_dataset"], it["domain"])].append(it)
    for v in by.values(): rng.shuffle(v)
    order = []; keys = sorted(by, key=lambda k: -len(by[k])); ptr = {k: 0 for k in keys}
    while len(order) < len(items):
        for k in keys:
            if ptr[k] < len(by[k]): order.append(by[k][ptr[k]]); ptr[k] += 1
    # BM25 over the whole pool
    N = len(pool); df_ = collections.Counter()
    for w in pw: df_.update(set(w))
    idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df_.items()}; avg = sum(len(w) for w in pw) / N
    inv = collections.defaultdict(list)
    for i, w in enumerate(pw):
        for t, c in collections.Counter(w).items(): inv[t].append((i, c))
    def shots_for(q):
        qs = set(q); sc = collections.defaultdict(float)
        for t in qs:
            if t in idf:
                for i, c in inv[t]: sc[i] += idf[t] * c * 2.2 / (c + 1.2 * (0.25 + 0.75 * len(pw[i]) / avg))
        out = []
        for i, s in sorted(sc.items(), key=lambda kv: -kv[1])[:60]:
            if len(pw[i]) > 90 or len(pw[i]) < 3: continue
            if len(qs & set(pw[i])) / max(1, len(qs | set(pw[i]))) >= 0.85: continue
            out.append(dict(english=pool[i]["english"], target=pool[i]["target"], bm25=round(s, 2)))
            if len(out) == 8: break
        return out
    n = 0
    with open(f"{OUT}/{iso}.jsonl", "w") as f:
        for j, it in enumerate(order):
            sh = shots_for(words(it["english"]))
            if len(sh) < 3: st["few_shots"] += 1; continue
            it["sid"] = f"{iso}-{j:06d}"; it["shots"] = sh
            f.write(json.dumps(it, ensure_ascii=False) + "\n"); n += 1
    print(iso, "sources", n, "of", len(items), dict(st), "strata", {f"{k[0][:30]}/{k[1]}": len(v) for k, v in by.items()}, flush=True)
