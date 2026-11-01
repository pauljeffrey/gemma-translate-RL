import json, sys, numpy as np, glob
from sacrebleu.metrics import CHRF
from scipy.stats import spearmanr
R="/workspace/jobs/r15/probe"; chrf=CHRF(word_order=2)
def L(p): return {json.loads(x)["id"]:json.loads(x) for x in open(p)}
for l in sys.argv[1:]:
    T=L(f"{R}/tests/{l}.jsonl"); G=L(f"{R}/preds/gemma_fs8/{l}.jsonl"); S=L(f"{R}/preds/gemma_fs8t07/{l}.jsonl"); B=L(f"{R}/preds/final_1433/{l}.jsonl")
    ids=[i for i in S if i in G and i in B]
    ref=[T[i]["reference"] for i in ids]
    cs=lambda h,r: chrf.corpus_score(h,[r]).score
    print(l,"n",len(ids),"corpus chrF  base",round(cs([B[i]["hyp"] for i in ids],ref),2),"greedy",round(cs([G[i]["hyp"] for i in ids],ref),2),"t0.7",round(cs([S[i]["hyp"] for i in ids],ref),2))
    for name in ("zs","fs3","fs8note"):
        p=f"{R}/preds/gemma_{name}/{l}.jsonl"
        try: X=L(p)
        except Exception: continue
        j=[i for i in ids if i in X]
        print("  ",name,len(j),round(cs([X[i]["hyp"] for i in j],[T[i]["reference"] for i in j]),2),"(greedy same items",round(cs([G[i]["hyp"] for i in j],[T[i]["reference"] for i in j]),2),")")
    cons=np.array([chrf.sentence_score(S[i]["hyp"],[G[i]["hyp"]]).score for i in ids]); q=np.array([chrf.sentence_score(G[i]["hyp"],[T[i]["reference"]]).score for i in ids])
    print("  spearman(consistency, quality of greedy)=",round(spearmanr(cons,q)[0],3))
    for tau in (0,30,40,50,60,70):
        k=cons>=tau
        if k.sum()>=10: print(f"   tau {tau}: keep {k.mean():.2f}  greedy chrF on kept {cs([G[i]['hyp'] for i,kk in zip(ids,k) if kk],[T[i]['reference'] for i,kk in zip(ids,k) if kk]):.1f}")
