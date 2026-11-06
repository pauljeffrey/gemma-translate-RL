import json, unicodedata, numpy as np, sacrebleu
nfc=lambda x: unicodedata.normalize("NFC",x)
res={}
for m in ("sft","rl"):
    R={k:json.load(open(f"{m}_{k}.json")) for k in ("hf","hf32","vllm","vllm32")}
    items=R["hf"]["items"]; ref=[nfc(i[2]) for i in items]; iso=[i[0] for i in items]
    ch=sacrebleu.metrics.CHRF(word_order=2)
    st={k:np.array(ch._extract_corpus_statistics([nfc(h) for h in R[k]["hyp"]],[ref]),dtype=float) for k in R}
    rng=np.random.default_rng(1); n=len(ref); B=1000
    w=rng.poisson(1.0,size=(B,n)).astype(float)
    langs=sorted(set(iso)); idx={l:np.array([i for i,x in enumerate(iso) if x==l]) for l in langs}
    macro=lambda k,ww: float(np.mean([ch._compute_f_score(ww[idx[l]] @ st[k][idx[l]]) for l in langs]))
    pts={k:macro(k,np.ones(n)) for k in R}
    out={"point":{k:round(v,2) for k,v in pts.items()}}
    for a,b in (("vllm","vllm32"),("hf","hf32"),("vllm32","hf32"),("vllm","hf")):
        d=[macro(b,w[i])-macro(a,w[i]) for i in range(B)]
        out[f"{b} minus {a}"]=[round(pts[b]-pts[a],2),round(float(np.percentile(d,2.5)),2),round(float(np.percentile(d,97.5)),2)]
    res[m]=out; print(m,json.dumps(out))
json.dump(res,open("parity_bootstrap.json","w"),indent=1)
