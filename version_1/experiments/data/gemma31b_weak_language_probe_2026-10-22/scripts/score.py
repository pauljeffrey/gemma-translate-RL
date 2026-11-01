import json, os, sys, re, glob, collections, unicodedata, random
import numpy as np, pandas as pd
from sacrebleu.metrics import CHRF
from gemma_mt.rl.lid import GlotLIDClassifier
from gemma_mt.languages import LANGUAGES
from gemma_mt.eval.metrics import spbleu, nfc
R="/workspace/jobs/r15/probe"
LANGS=sys.argv[1].split(); OUTTAG=sys.argv[2] if len(sys.argv)>2 else "all"
SYSTEMS=["final_1433","gemma_zs","gemma_fs3","gemma_fs8","gemma_fs8note"]
W=re.compile(r"\w+",re.U); chrf=CHRF(word_order=2)
lid=GlotLIDClassifier(path=glob.glob("/workspace/sft_build/hf/hub/models--cis-lmu--glotlid/snapshots/*/model_v3.bin")[0])
labels=set(lid.labels) if hasattr(lid,"labels") else set()
def words(s): return W.findall(unicodedata.normalize("NFC",s).lower())
def g4(w): return {tuple(w[i:i+4]) for i in range(len(w)-3)}
def stats(h,r): return np.array(chrf._extract_corpus_statistics(nfc(h),[nfc(r)]),dtype=np.float64)
def fs(st): return chrf._compute_f_score(st.sum(0).tolist())
def boot(sa,sb,B=1000,seed=0):
    n=len(sa); rng=np.random.default_rng(seed); d=[]
    for _ in range(B):
        ix=rng.integers(0,n,n); d.append(fs(sa[ix])-fs(sb[ix]))
    d=np.array(d); return float(np.percentile(d,2.5)),float(np.percentile(d,97.5))
def loadp(sysname,l):
    p=f"{R}/preds/{sysname}/{l}.jsonl"
    return {json.loads(x)["id"]:json.loads(x) for x in open(p)} if os.path.exists(p) else None
rows=[]
for l in LANGS:
    T=[json.loads(x) for x in open(f"{R}/tests/{l}.jsonl")]; byid={t["id"]:t for t in T}
    acc=set(LANGUAGES[l].glotlid); supp=[a for a in acc if a in labels]
    P={s:loadp(s,l) for s in SYSTEMS}
    lidcache={}
    def lids(texts):
        need=[t for t in set(texts) if t not in lidcache]
        if need:
            for t,r in zip(need,lid.top(need,1)): lidcache[t]=r[0][0] if r else ""
        return [lidcache[t] for t in texts]
    base=P["final_1433"]
    ref_lid=np.mean([x in acc for x in lids([t["reference"] for t in T if t["set"]!="seen"])])
    for s in SYSTEMS:
        if P[s] is None: continue
        for pop in ["primary","seen"]+sorted({t["set"] for t in T if t["set"] not in("seen",)}):
            ids=[t["id"] for t in T if t["id"] in P[s] and (base is None or t["id"] in base) and
                 ((pop=="primary" and t["set"]!="seen") or (pop=="seen" and t["set"]=="seen") or t["set"]==pop)]
            if pop=="primary" and s!="final_1433" and s not in("gemma_fs8",): pass
            if len(ids)<5: continue
            h=[P[s][i]["hyp"] for i in ids]; r=[byid[i]["reference"] for i in ids]; src=[byid[i]["english"] for i in ids]
            st=stats(h,r); sc=fs(st)
            row=dict(lang=l,system=s,pop=pop,n=len(ids),chrf=sc,spbleu=spbleu(nfc(h),nfc(r)))
            if s!="final_1433" and base is not None:
                hb=[base[i]["hyp"] for i in ids]; sb=stats(hb,r)
                row["base_chrf"]=fs(sb); row["delta"]=sc-fs(sb); row["ci_lo"],row["ci_hi"]=boot(st,sb)
            lab=lids(h); row["lid_ok"]=float(np.mean([x in acc for x in lab])); row["lid_ref"]=float(np.mean([x in acc for x in lids(r)]))
            row["lid_en"]=float(np.mean([x=="eng_Latn" for x in lab]))
            cp=[]; rep=[]; lr=[]; ov=[]; ovr=[]; fin=[]
            for i,hh,rr,ss in zip(ids,h,r,src):
                hw=words(hh); sw=set(words(ss))
                cp.append(float((len(hw)>0 and np.mean([w in sw for w in hw])>=0.8 and len(hw)>=3) or nfc([hh])[0].strip().lower()==nfc([ss])[0].strip().lower()))
                g=[tuple(hw[j:j+3]) for j in range(len(hw)-2)]; rep.append(1-len(set(g))/len(g) if g else 0.0)
                lr.append(len(hh)/max(1,len(rr)))
                shots=byid[i]["shots"][:P[s][i].get("k_used",8)] if s.startswith("gemma") and s!="gemma_zs" else []
                if shots:
                    sg=set().union(*[g4(words(x["target"])) for x in shots]); hg=g4(words(hh)); rg=g4(words(rr))
                    if hg: ov.append(len(hg&sg)/len(hg))
                    if rg: ovr.append(len(rg&sg)/len(rg))
                fin.append(1.0 if P[s][i].get("finish","stop").lower().find("length")<0 and P[s][i].get("finished",True) else 0.0)
            row.update(copy_en=float(np.mean(cp)),rep3=float(np.mean(rep)),len_ratio_med=float(np.median(lr)),shot_ov_hyp=float(np.mean(ov)) if ov else np.nan,shot_ov_ref=float(np.mean(ovr)) if ovr else np.nan,finished=float(np.mean(fin)),empty=float(np.mean([len(x.strip())==0 for x in h])),glotlid_labels_supported=",".join(supp) or "NONE")
            rows.append(row)
df=pd.DataFrame(rows); df.to_csv(f"{R}/results_{OUTTAG}.csv",index=False)
pd.set_option("display.width",250,"display.max_columns",30)
print(df[df["pop"]=="primary"][["lang","system","n","chrf","delta","ci_lo","ci_hi","lid_ok","lid_ref","copy_en","rep3","shot_ov_hyp","shot_ov_ref","finished"]].round(3).to_string())
