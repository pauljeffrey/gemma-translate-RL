import numpy as np, json, sys, csv
sys.path.insert(0,'/workspace/clean_sft/code')
from gemma_mt.languages import ALL_ISOS
d="/dev/shm/control/tp_packed/train"
iso=np.load(d+"/iso_id.npy"); ps=np.load(d+"/pack_start.npy"); sl=np.load(d+"/sample_len.npy"); tl=np.load(d+"/target_len.npy")
rows={}
for i in range(len(ps)-1):
    a,b=int(ps[i]),int(ps[i+1]); l=int(iso[a]); assert (iso[a:b]==l).all()
    r=rows.setdefault(ALL_ISOS[l],{"n_samples":0,"n_packs":0,"real_tokens":0,"target_tokens":0}); r["n_samples"]+=b-a; r["n_packs"]+=1; r["real_tokens"]+=int(sl[a:b].sum()); r["target_tokens"]+=int(tl[a:b].sum())
P=sum(r["n_packs"] for r in rows.values()); print("TP packs",P,"samples",sum(r["n_samples"] for r in rows.values()))
out=sys.argv[1]
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["iso","n_samples","n_packs","padded_tokens","real_tokens","target_tokens","real_tokens_per_sample"])
    for k in sorted(rows):
        r=rows[k]; w.writerow([k,r["n_samples"],r["n_packs"],r["n_packs"]*4096,r["real_tokens"],r["target_tokens"],round(r["real_tokens"]/r["n_samples"],1)]); print(k,r["n_samples"],r["n_packs"],r["n_packs"]*4096,r["real_tokens"],r["target_tokens"])
    w.writerow(["TOTAL",sum(r["n_samples"] for r in rows.values()),P,P*4096,sum(r["real_tokens"] for r in rows.values()),sum(r["target_tokens"] for r in rows.values()),""])
