import numpy as np, json, sys
sys.path.insert(0,'/workspace/clean_sft/code')
from gemma_mt.languages import ALL_ISOS
d="/dev/shm/control/tp_calib_packed/train"
iso=np.load(d+"/iso_id.npy"); ps=np.load(d+"/pack_start.npy"); sl=np.load(d+"/sample_len.npy")
print("n_samples", len(iso), "n_packs", len(ps), ps[:5], ps[-3:])
# pack_start = first sample index of each pack (assumed)
cnt={}; smp={}; tok={}
for i in range(len(ps)-1):
    a=ps[i]; b=ps[i+1]
    l=int(iso[a]); assert (iso[a:b]==l).all()
    cnt[l]=cnt.get(l,0)+1; smp[l]=smp.get(l,0)+(b-a); tok[l]=tok.get(l,0)+int(sl[a:b].sum())
rows=[]
for l in sorted(cnt):
    rows.append((ALL_ISOS[l],cnt[l],smp[l],tok[l]))
    print(ALL_ISOS[l],cnt[l],smp[l],tok[l], round(smp[l]/cnt[l],1), round(tok[l]/smp[l],1))
rate=sum(r[1]/r[2] for r in rows)  # packs per sample summed over languages
for P in (96300,):
    N=P/rate; print("N for P",P,N)
json.dump({r[0]:{"packs":r[1],"samples":r[2],"tokens":r[3]} for r in rows}, open("/dev/shm/control/calib_rates.json","w"))
