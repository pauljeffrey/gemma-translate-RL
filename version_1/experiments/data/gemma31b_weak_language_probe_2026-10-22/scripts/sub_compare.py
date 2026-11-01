import json, os, sys, numpy as np, pandas as pd
from sacrebleu.metrics import CHRF
R="/workspace/jobs/r15/probe"; chrf=CHRF(word_order=2)
LANGS="ish bin bbj oke tsc dag ibb urh nbl shi kea ach vmw seh nyn her yao sid ndc kri dua bas wes".split()
nfc=lambda L: __import__("unicodedata").normalize
import unicodedata
N=lambda xs:[unicodedata.normalize("NFC",x) for x in xs]
def L(p): return {json.loads(x)["id"]:json.loads(x) for x in open(p)} if os.path.exists(p) else None
def stats(h,r): return np.array(chrf._extract_corpus_statistics(N(h),[N(r)]),dtype=float)
fs=lambda st: chrf._compute_f_score(st.sum(0).tolist())
def boot(a,b,B=1000):
    rng=np.random.default_rng(0); n=len(a); d=[fs(a[ix])-fs(b[ix]) for ix in (rng.integers(0,n,n) for _ in range(B))]; return np.percentile(d,2.5),np.percentile(d,97.5)
rows=[]
for l in LANGS:
    T=L(f"{R}/tests/{l}.jsonl"); sub=[i for i,t in T.items() if t["sub"]]
    P={s:L(f"{R}/preds/{s}/{l}.jsonl") for s in ["final_1433","gemma_zs","gemma_fs3","gemma_fs8","gemma_fs8note"]}
    ids=[i for i in sub if all(P[s] is None or i in P[s] for s in P)]
    ref=[T[i]["reference"] for i in ids]
    st={s:stats([P[s][i]["hyp"] for i in ids],ref) for s in P if P[s] is not None and len(ids)>=10}
    if not st: continue
    r=dict(lang=l,n_sub=len(ids))
    for s,v in st.items(): r[s+"_chrf"]=fs(v)
    for s in ["gemma_zs","gemma_fs3","gemma_fs8","gemma_fs8note"]:
        if s in st:
            lo,hi=boot(st[s],st["final_1433"]); r[s+"_delta_vs_base"]=fs(st[s])-fs(st["final_1433"]); r[s+"_ci"]=f"[{lo:+.1f},{hi:+.1f}]"
    for s in ["gemma_fs3","gemma_fs8note","gemma_zs"]:
        if s in st and "gemma_fs8" in st:
            lo,hi=boot(st[s],st["gemma_fs8"]); r[s+"_delta_vs_fs8"]=fs(st[s])-fs(st["gemma_fs8"]); r[s+"_vs_fs8_ci"]=f"[{lo:+.1f},{hi:+.1f}]"
    rows.append(r)
A=pd.DataFrame(rows); A.to_csv(f"{R}/summary_sub_ablation.csv",index=False)
pd.set_option("display.width",250,"display.max_columns",40)
print(A[["lang","n_sub","final_1433_chrf","gemma_zs_chrf","gemma_fs3_chrf","gemma_fs8_chrf","gemma_fs8note_chrf","gemma_fs8note_delta_vs_fs8","gemma_fs8note_vs_fs8_ci","gemma_fs3_delta_vs_fs8"]].round(1).to_string())
