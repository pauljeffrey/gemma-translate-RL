import json, re, random, math, collections, unicodedata, os, pandas as pd
R="/workspace/jobs/r15/probe"; os.makedirs(R+"/tests",exist_ok=True)
import sys; L=sys.argv[1:]
sys.path.insert(0,"/workspace/jobs/b1/gemma_mt")
from gemma_mt.eval.benchmarks import load_benchmark
BP="/workspace/sft_build/bench/test_sets"
nfc=lambda s: unicodedata.normalize("NFC",s)
W=re.compile(r"\w+",re.U)
def words(s): return W.findall(s.lower())
def norm(s): return " ".join(words(s))
def ngr(w,n=6): return {tuple(w[i:i+n]) for i in range(max(0,len(w)-n+1))}
rng=random.Random(15)
stats={}
for l in L:
    pool=[json.loads(x) for x in open(f"/workspace/jobs/r10/pool/{l}.jsonl")]
    pool=[dict(english=nfc(p["english"]),target=nfc(p["target"])) for p in pool if p["english"].strip() and p["target"].strip()]
    pw=[words(p["english"]) for p in pool]
    pnorm={" ".join(w) for w in pw}
    pgr=set().union(*[ngr(w) for w in pw])
    def leak(en):
        w=words(en)
        if " ".join(w) in pnorm: return "exact"
        g=ngr(w)
        if g and len(g&pgr)/len(g)>=0.5: return "6gram"
        return None
    tests=[]; st=collections.Counter()
    # T_val
    for i,x in enumerate(open(f"{R}/raw/val_{l}.jsonl")):
        r=json.loads(x); en=nfc(r["english"]); nw=len(words(en))
        if nw<3 or nw>120: st["val_len_drop"]+=1; continue
        lk=leak(en)
        if lk: st["val_leak_"+lk]+=1; continue
        tests.append(dict(set="val",id=f"val{r['idx']}",english=en,reference=nfc(r["target"])))
    # benchmark rows (human): every available set, capped 100 each
    for bname in ("flores_plus","ntrex","smol_sent","flores_moz","glot_story","ibom_mt","salt","mafand"):
        try: rows=load_benchmark(bname,path=f"{BP}/{bname}",isos=[l])
        except Exception as e: rows=[]
        rows=[r for r in rows if 2<=len(words(r["english"]))<=120 and r["reference"].strip()]
        rows=[r for r in rows if not leak(r["english"])]
        rng.shuffle(rows)
        for r in rows[:100]: tests.append(dict(set=bname,id=f"{bname}{r['id']}",english=nfc(r["english"]),reference=nfc(r["reference"])))
        if rows: st["bench_"+bname]=len(rows)
    wp=f"/workspace/jobs/r10/clean/opus_wikimedia__{l}.jsonl"
    cand=[i for i,p in enumerate(pool) if 5<=len(pw[i])<=80]
    rng.shuffle(cand); seen_idx=set(cand[:60])
    for i in sorted(seen_idx): tests.append(dict(set="seen",id=f"seen{i}",english=pool[i]["english"],reference=pool[i]["target"]))
    # gold (bin)
    if l=="bin":
        df=pd.read_parquet("/workspace/jobs/r10/sft_extra_v1/owner_code_switch/rows.parquet"); g=df[(df.iso==l)&(df.quality_flag=="human_gold")]
        for j,r in enumerate(g.itertuples()): tests.append(dict(set="gold",id=f"gold{j}",english=nfc(r.english),reference=nfc(r.target)))
    # dedupe test english within language
    seen=set(); out=[]
    for t in tests:
        k=norm(t["english"])
        if k in seen: st["dup_drop"]+=1; continue
        seen.add(k); out.append(t)
    tests=out
    ns=[t for t in tests if t["set"]!="seen"]; rs=random.Random(7); rs.shuffle(ns)
    for j,t in enumerate(ns): t["sub"]=1 if j<80 else 0
    for t in tests:
        t.setdefault("sub",0)
    # retrieval pool: all but T_seen; BM25
    P=[i for i in range(len(pool)) if i not in seen_idx]
    df_=collections.Counter(); 
    for i in P: df_.update(set(pw[i]))
    N=len(P); idf={w:math.log(1+(N-n+.5)/(n+.5)) for w,n in df_.items()}
    avg=sum(len(pw[i]) for i in P)/N
    inv=collections.defaultdict(list)
    for i in P:
        tf=collections.Counter(pw[i])
        for w,c in tf.items(): inv[w].append((i,c))
    def bm25(q):
        sc=collections.defaultdict(float)
        for w in set(q):
            if w not in idf: continue
            for i,c in inv[w]:
                sc[i]+=idf[w]*c*2.2/(c+1.2*(0.25+0.75*len(pw[i])/avg))
        return sc
    for t in tests:
        qw=words(t["english"]); qs=set(qw); tn=norm(t["english"]); rt=norm(t["reference"])
        sc=bm25(qw)
        ranked=sorted(sc.items(),key=lambda kv:-kv[1])[:80]
        shots=[]
        for i,s in ranked:
            if len(pw[i])>90 or len(pw[i])<3: continue
            if " ".join(pw[i])==tn: continue
            jac=len(qs&set(pw[i]))/max(1,len(qs|set(pw[i])))
            if jac>=0.85: continue
            if norm(pool[i]["target"])==rt: continue
            shots.append(dict(english=pool[i]["english"],target=pool[i]["target"],bm25=round(s,2)))
            if len(shots)==8: break
        if len(shots)<8:   # pad with random pool items (short ones) so every test has 8
            extra=[i for i in P if 5<=len(pw[i])<=60]; rng.shuffle(extra)
            for i in extra:
                if len(shots)==8: break
                if any(s["english"]==pool[i]["english"] for s in shots): continue
                shots.append(dict(english=pool[i]["english"],target=pool[i]["target"],bm25=0.0)); st["padded_shots"]+=1
        t["shots"]=shots
    json.dump(dict(pool=len(pool),retrieval_pool=len(P)),open(f"{R}/tests/_{l}_poolstat.json","w"))
    with open(f"{R}/tests/{l}.jsonl","w") as f:
        for t in tests: f.write(json.dumps(t,ensure_ascii=False)+"\n")
    c=collections.Counter(t["set"] for t in tests)
    print(l,"pool",len(pool),"retr",len(P),"tests",dict(c),"stats",dict(st),flush=True)
