#!/usr/bin/env python3
"""Step 6: within-language evidence. Is the NLLB-mined text different from the rest of the same language's stage-B translation data?

For every language (token_sample.parquet = 1/64 hash sample of the pairs that went into stage B, with text and group):
  * GlotLID v3 (local model_v3.bin) top-1 on the TARGET side: share accepted for the language, share labelled English, share labelled another language;
  * GlotLID top-1 on the ENGLISH side (share 'eng'); copy rate (target == english after casefold/punctuation strip); target/english length ratio outside [0.4, 2.5];
for the full NLLB files (hub/extra2k/nllb): distinct-target share, share of rows whose target occurs >= 3 times in the language, mean AfriCOMET-QE (work/scores_gpu) of the NLLB rows and
the share above the language's gate threshold (thr_cal) and the same for ALL other ingested rows of the language that were scored.
Outputs: nllb_quality_by_language.csv
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd, duckdb
import fasttext

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(H, "..", "..", ".."))
ARC = os.path.join(ROOT, "research_archive_2026-11-03", "databox_backup")
sys.path.insert(0, os.path.join(ROOT, "gemma_mt"))
from gemma_mt.languages import LANGUAGES
GL = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--cis-lmu--glotlid/snapshots/*/model_v3.bin"))[0]
fasttext.FastText.eprint = lambda *a, **k: None
m = fasttext.load_model(GL)
N = 500
norm = lambda s: re.sub(r"[\W_]+", " ", s.casefold()).strip()


def top1(texts):
    out = []
    for t in texts:
        t = " ".join(t.split())[:600]
        if not t:
            out.append(("", 0.0)); continue
        lab, p = m.predict(t, k=1)
        out.append((lab[0].replace("__label__", ""), float(p[0])))
    return out


smp = pd.read_parquet(os.path.join(H, "token_sample.parquet"))
rows = []
for (iso, grp), g in smp.groupby(["iso", "grp"]):
    if len(g) < 60 or iso not in LANGUAGES:
        continue
    g = g.sample(min(len(g), N), random_state=3)
    acc = set(LANGUAGES[iso].glotlid)
    tl = top1(g.target.tolist()); el = top1(g.english.tolist())
    ok = np.mean([t[0] in acc for t in tl]); eng_t = np.mean([t[0].startswith("eng_") for t in tl]); eng_s = np.mean([e[0].startswith("eng_") for e in el])
    cp = np.mean([norm(a) == norm(b) for a, b in zip(g.english, g.target)])
    lr = (g.target.str.len() / g.english.str.len().clip(lower=1))
    rows.append(dict(iso=iso, grp=grp, n=len(g), lid_target_ok=ok, lid_target_eng=eng_t, lid_target_other=1 - ok - eng_t, lid_src_eng=eng_s, copy=cp, len_ratio_extreme=float(((lr < 0.4) | (lr > 2.5)).mean()), len_ratio_med=float(lr.median())))
q = pd.DataFrame(rows)
w = q.pivot(index="iso", columns="grp")
w.columns = [f"{a}_{b}" for a, b in w.columns]
w = w.reset_index()

# ---- full NLLB files: duplicates and QE
con = duckdb.connect()
con.execute("pragma threads=4")
th = json.load(open(os.path.join(ARC, "work", "gate_all", "thresholds.json")))["languages"]
con.execute(f"create view sc as select key, qe from read_parquet('{ARC}/work/scores_gpu/*.parquet')")
out = []
for d in sorted(glob.glob(os.path.join(ARC, "hub", "extra2k", "nllb", "*"))):
    iso = os.path.basename(d)
    fs = glob.glob(d + "/*.parquet")
    r = con.execute(f"select count(*) n, count(distinct target) nt, count(distinct english) ne from read_parquet({fs})").fetchone()
    dup3 = con.execute(f"select coalesce(sum(c),0) from (select count(*) c from read_parquet({fs}) group by target having count(*)>=3)").fetchone()[0]
    qq = con.execute(f"select avg(s.qe), count(s.qe), avg(case when s.qe >= {th.get(iso, {}).get('thr_cal', 0)} then 1.0 else 0.0 end) from read_parquet({fs}) n join sc s using (key)").fetchone()
    out.append(dict(iso=iso, nllb_rows=r[0], nllb_distinct_target_share=r[1] / r[0], nllb_distinct_english_share=r[2] / r[0], nllb_rows_in_targets_repeated_3plus=dup3 / r[0],
                    nllb_qe_mean=qq[0], nllb_qe_scored=qq[1], nllb_keep_rate=qq[2], thr_cal=th.get(iso, {}).get("thr_cal"), lang_median_qe_all_sources=th.get(iso, {}).get("median")))
o = pd.DataFrame(out)
res = o.merge(w, on="iso", how="left")
res.to_csv(os.path.join(H, "nllb_quality_by_language.csv"), index=False)
pd.set_option("display.width", 250)
print(res[["iso", "nllb_rows", "nllb_distinct_target_share", "nllb_rows_in_targets_repeated_3plus", "nllb_qe_mean", "lang_median_qe_all_sources", "nllb_keep_rate", "lid_target_ok_nllb_mined", "lid_target_ok_other", "lid_target_eng_nllb_mined", "copy_nllb_mined", "copy_other"]].round(3).to_string())
