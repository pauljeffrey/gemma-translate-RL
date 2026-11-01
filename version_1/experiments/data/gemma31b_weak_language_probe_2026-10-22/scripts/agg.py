import pandas as pd, numpy as np, sys
R="/workspace/jobs/r15/probe"
d=pd.read_csv(f"{R}/results_all.csv")
rows=[]
for l,g in d.groupby("lang"):
    r={"lang":l}
    b=g[(g.system=="final_1433")&(g["pop"]=="primary")]
    r["base_n"]=int(b.n.iloc[0]) if len(b) else np.nan; r["base_chrf_primary"]=b.chrf.iloc[0] if len(b) else np.nan
    for s in ["gemma_fs8","gemma_fs8note","gemma_fs3","gemma_zs"]:
        x=g[(g.system==s)&(g["pop"]=="primary")]
        k=s.replace("gemma_","")
        if len(x):
            x=x.iloc[0]; r[f"{k}_n"]=int(x.n); r[f"{k}_chrf"]=x.chrf; r[f"{k}_base_chrf_same_items"]=x.base_chrf; r[f"{k}_delta"]=x.delta; r[f"{k}_ci"]=f"[{x.ci_lo:+.1f},{x.ci_hi:+.1f}]"
            r[f"{k}_ci_lo"]=x.ci_lo; r[f"{k}_lid"]=x.lid_ok; r[f"{k}_copy"]=x.copy_en; r[f"{k}_rep3"]=x.rep3; r[f"{k}_finished"]=x.finished; r[f"{k}_shot_ov_excess"]=(x.shot_ov_hyp-x.shot_ov_ref) if pd.notna(x.shot_ov_hyp) else np.nan
    # benchmark-only (out-of-domain) fs8 delta
    ood=g[(g.system=="gemma_fs8")&(~g["pop"].isin(["primary","seen","val"]))]
    if len(ood): r["fs8_ood_sets"]="; ".join(f"{p.pop}:{p.delta:+.1f}[{p.ci_lo:+.1f},{p.ci_hi:+.1f}]n{p.n}" for p in ood.itertuples())
    lb=g.glotlid_labels_supported.iloc[0]; r["glotlid_label_supported"]=lb
    r["lid_ref"]=g[(g.system=="final_1433")&(g["pop"]=="primary")].lid_ref.iloc[0] if len(b) else np.nan
    r["base_lid"]=b.lid_ok.iloc[0] if len(b) else np.nan
    # verdict on registered system fs8 (and fs8note as the second registered few-shot variant)
    best=None
    for k in ["fs8","fs8note"]:      # registered system first: fs8note only counts when fs8 itself does not pass
        if best is None and f"{k}_delta" in r and r[f"{k}_delta"]>=3 and r[f"{k}_ci_lo"]>0: best=k
    clean=lambda k: (r.get(f"{k}_copy",1)<=0.05 and r.get(f"{k}_lid",0)>=max(0.5,(r["base_lid"] if pd.notna(r["base_lid"]) else 0)-0.05) and r.get(f"{k}_finished",0)>=0.9 and r.get(f"{k}_rep3",1)<=0.25 and (r.get(f"{k}_shot_ov_excess",0)<=0.15 or pd.isna(r.get(f"{k}_shot_ov_excess"))))
    if best and clean(best): r["verdict"]=f"PASS ({best})"
    elif best: r["verdict"]=f"FAIL (margin ok, checks not clean: {best})"
    elif "fs8_delta" in r: r["verdict"]="FAIL (no clear margin)" if r["fs8_ci_lo"]<=0 or r["fs8_delta"]<3 else "?"
    else: r["verdict"]="not run"
    rows.append(r)
A=pd.DataFrame(rows); A.to_csv(f"{R}/summary_per_language.csv",index=False)
pd.set_option("display.width",250,"display.max_columns",40)
print(A[["lang","base_n","base_chrf_primary","fs8_n","fs8_chrf","fs8_delta","fs8_ci","fs8note_delta","fs3_delta","zs_chrf","zs_base_chrf_same_items","zs_delta","fs8_lid","base_lid","lid_ref","verdict"]].round(2).to_string())
