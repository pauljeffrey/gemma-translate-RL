import pandas as pd, numpy as np, json, re, sys, unicodedata, collections
W=re.compile(r"\w+",re.U); words=lambda s: W.findall(unicodedata.normalize("NFC",s).lower())
D="/workspace/jobs/r15/gemma_synth_v1"; rows=pd.read_parquet(f"{D}/rows.parquet")
out={}
for iso,g in rows.groupby("iso"):
    g=g.sort_values("sid"); tg=g.target.tolist()
    pool=[json.loads(x)["target"] for x in open(f"/workspace/jobs/r10/pool/{iso}.jsonl")]
    pv=set(w for p in pool for w in words(p))
    types=set(); bi=set(); curve=[]; prev_t=0; prev_b=0
    for i,t in enumerate(tg,1):
        w=words(t); types.update(w); bi.update(zip(w,w[1:]))
        if i%250==0 or i==len(tg): curve.append((i,len(types),len(bi),round(100*(len(types)-prev_t)/ (250 if i%250==0 else i%250),1), round(100*(len(bi)-prev_b)/(250 if i%250==0 else i%250),1))); prev_t,prev_b=len(types),len(bi)
    # equal-size comparison: 1000 random pool chunks vs 1000 synthetic pairs, distinct-n on first 20000 words
    rs=np.random.default_rng(0); sp=[tg[i] for i in rs.choice(len(tg),min(1000,len(tg)),replace=False)]; pp=[pool[i] for i in rs.choice(len(pool),min(1000,len(pool)),replace=False)]
    def dn(texts,n,cap=20000):
        ws=[w for t in texts for w in words(t)][:cap]; g_=[tuple(ws[i:i+n]) for i in range(len(ws)-n+1)]; return len(set(g_))/max(1,len(g_))
    sv=set(w for t in tg for w in words(t))
    out[iso]=dict(n_pairs=len(tg),curve=[dict(pairs=a,types=b,bigrams=c,new_types_per_100_pairs=d_,new_bigrams_per_100_pairs=e) for a,b,c,d_,e in curve],
      distinct_equal_size={n:dict(synth=dn(sp,n),pool=dn(pp,n)) for n in (1,2,3)},
      vocab_types_synth=len(sv),vocab_types_pool=len(pv),share_synth_types_not_in_pool=len(sv-pv)/max(1,len(sv)),
      exact_dup_targets=1-len(set(tg))/len(tg),
      src_licence=g.src_licence.value_counts().to_dict(),src_dataset=g.src_source_dataset.value_counts().to_dict(),domain=g.domain.value_counts().to_dict())
json.dump(out,open("/workspace/jobs/r15/gemma_synth_v1/diversity.json","w"),indent=1)
for iso,o in out.items(): print(iso,o["n_pairs"],o["curve"][-3:],o["distinct_equal_size"],round(o["share_synth_types_not_in_pool"],3),o["src_licence"])
