import pandas as pd, numpy as np, sys
sys.path.insert(0,"/workspace/jobs/b1/gemma_mt")
from gemma_mt.languages import LANGUAGES
R="/workspace/jobs/r15/probe"
d=pd.read_csv(f"{R}/results_all.csv"); s=pd.read_csv(f"{R}/summary_per_language.csv").set_index("lang"); a=pd.read_csv(f"{R}/summary_sub_ablation.csv").set_index("lang")
order="ish bin bbj oke tsc dag ibb urh nbl shi kea ach vmw seh nyn her yao sid ndc kri dua bas wes".split()
f=lambda x,p=1: "" if pd.isna(x) else f"{x:.{p}f}"
out=[]
out.append("| lang | name | n (primary) | final_1433 chrF++ | Gemma fs8 chrF++ | delta | 95% CI (paired bootstrap) | verdict |\n|---|---|---|---|---|---|---|---|")
for l in order:
    r=s.loc[l]
    out.append(f"| {l} | {LANGUAGES[l].name} | {int(r.fs8_n)} | {f(r.fs8_base_chrf_same_items)} | {f(r.fs8_chrf)} | {r.fs8_delta:+.1f} | {r.fs8_ci} | {r.verdict} |")
T1="\n".join(out)
out=["| lang | n (80-item subset) | final_1433 | zero-shot | k=3 | k=8 (fs8) | k=8 + note | note vs fs8 (95% CI) |\n|---|---|---|---|---|---|---|---|"]
for l in order:
    r=a.loc[l]
    out.append(f"| {l} | {int(r.n_sub)} | {f(r.final_1433_chrf)} | {f(r.gemma_zs_chrf)} | {f(r.gemma_fs3_chrf)} | {f(r.gemma_fs8_chrf)} | {f(r.gemma_fs8note_chrf)} | {'' if pd.isna(r.gemma_fs8note_delta_vs_fs8) else f'{r.gemma_fs8note_delta_vs_fs8:+.1f} {r.gemma_fs8note_vs_fs8_ci}'} |")
T2="\n".join(out)
x=d[(d.system=="gemma_fs8")&(~d["pop"].isin(["primary"]))].copy()
out=["| lang | set | n | final_1433 | Gemma fs8 | delta | 95% CI |\n|---|---|---|---|---|---|---|"]
for l in order:
    for r in x[x.lang==l].itertuples():
        out.append(f"| {l} | {r.pop} | {r.n} | {f(r.base_chrf)} | {f(r.chrf)} | {r.delta:+.1f} | [{r.ci_lo:+.1f}, {r.ci_hi:+.1f}] |")
T3="\n".join(out)
out=["| lang | system | GlotLID accepted-label rate (hyp) | rate on references | copy-of-English | 3-gram repetition | finished (no length cap) | length ratio (chars hyp/ref, median) | 4-gram overlap with shot targets: hyp vs ref |\n|---|---|---|---|---|---|---|---|---|"]
for l in order:
    for sysn in ("final_1433","gemma_fs8"):
        r=d[(d.lang==l)&(d.system==sysn)&(d["pop"]=="primary")]
        if len(r)==0: continue
        r=r.iloc[0]
        ov="" if pd.isna(r.shot_ov_hyp) else f"{r.shot_ov_hyp:.3f} vs {r.shot_ov_ref:.3f}"
        out.append(f"| {l} | {sysn} | {r.lid_ok:.2f} | {r.lid_ref:.2f} | {r.copy_en:.3f} | {r.rep3:.3f} | {r.finished:.2f} | {r.len_ratio_med:.2f} | {ov} |")
T4="\n".join(out)
open(f"{R}/tables.md","w").write("## T1\n"+T1+"\n\n## T2\n"+T2+"\n\n## T3\n"+T3+"\n\n## T4\n"+T4+"\n")
print(T1); print(); print(T2)
