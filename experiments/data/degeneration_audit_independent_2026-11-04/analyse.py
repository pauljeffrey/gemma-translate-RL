"""E-08 step 2: tables from the per-row features (compute_features.py). Writes CSV/JSON/MD next to this script.
Usage: python analyse.py
"""
import json, os, sys, importlib.util
import numpy as np, pandas as pd
from sacrebleu.metrics import CHRF

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ARCH = os.path.join(ROOT, "research_archive_2026-11-03")
BENCH = ["flores_plus", "ntrex", "mafand"]
MODELS = [m for m in ["sft", "stageA", "b1best", "mixedbest", "weakfinal", "a0"] if os.path.exists(os.path.join(HERE, f"features_{m}.parquet"))]
KEY = ["benchmark", "iso", "id"]
rng = np.random.default_rng(0)

sys.path.insert(0, os.path.join(ROOT, "gemma_mt"))
import gemma_mt.languages as GL

R = pd.read_parquet(os.path.join(HERE, "refs_features.parquet"))
F = {m: pd.read_parquet(os.path.join(HERE, f"features_{m}.parquet")) for m in MODELS}

# ------------------------------------------------------------------ language label sets of the NLLB LID
import fasttext
fasttext.FastText.eprint = lambda *a, **k: None
LABELS = set(l.replace("__label__", "") for l in fasttext.load_model(os.environ["LID218"]).get_labels())
EXTRA = {"zgh": ["tzm_Tfng"], "orm": ["gaz_Latn", "orm_Latn"], "mlg": ["plt_Latn"], "swc": ["swh_Latn"], "taq": ["taq_Latn", "taq_Tfng"],
         "knc": ["knc_Arab", "knc_Latn"], "tir": ["tir_Ethi"], "ibb": [], "fuv": ["fuv_Latn"]}
ACCEPT = {}
for iso, L in GL.LANGUAGES.items():
    cand = list(L.glotlid) + EXTRA.get(iso, [])
    # the FLORES+ code of the language is another candidate (NLLB labels are FLORES-200 codes)
    if L.flores:
        cand.append(L.flores)
    ACCEPT[iso] = sorted(set(c for c in cand if c in LABELS))
isos_all = sorted(R["iso"].unique())
covered = [i for i in isos_all if ACCEPT.get(i)]
uncovered = [i for i in isos_all if not ACCEPT.get(i)]

# ------------------------------------------------------------------ calibration on references (control), per benchmark-language cell
R["r_logb"] = np.log(R["r_bytes"].clip(lower=1)); R["r_logc"] = np.log(R["r_lz_bytes"].clip(lower=1))
coef = {}
for (b, i), g in R.groupby(["benchmark", "iso"]):
    g = g[g["r_bytes"] >= 20]
    if len(g) >= 100 and g["r_logb"].std() > 0:
        coef[(b, i)] = np.polyfit(g["r_logb"], g["r_logc"], 1)
    else:
        coef[(b, i)] = None
pooled = np.polyfit(R[R.r_bytes >= 20]["r_logb"], R[R.r_bytes >= 20]["r_logc"], 1)


def zres(df, prefix):
    z = np.full(len(df), np.nan)
    lb = np.log(df[prefix + "bytes"].clip(lower=1).values); lc = np.log(df[prefix + "lz_bytes"].clip(lower=1).values)
    for (b, i), idx in df.groupby(["benchmark", "iso"]).indices.items():
        c = coef.get((b, i)); c = pooled if c is None else c
        z[idx] = lc[idx] - (c[0] * lb[idx] + c[1])
    z[df[prefix + "bytes"].values < 20] = np.nan
    return z


R["r_z"] = zres(R, "r_")
THR = {
    "z": float(np.nanpercentile(R["r_z"], 0.1)),
    "c8": float(np.nanpercentile(R["r_c8_rep"], 99.9)),
    "w3": float(np.nanpercentile(R["r_w3_rep"], 99.9)),
    "tail": float(np.nanpercentile(R["r_tail_rep"], 99.9)),
}
# thresholds are set so that 0.1% of the HUMAN references fire each rule (a false-positive budget fixed by the control, not by the reward's constants)
json.dump({"thresholds_from_references_p0.1_or_p99.9": THR, "n_refs": int(len(R))}, open(os.path.join(HERE, "thresholds.json"), "w"), indent=1)


def add_flags(df, ref):
    df = df.merge(ref[KEY + ["e_chars", "e_words", "e_ends_final", "r_chars", "r_words", "r_ends_final", "r_ends_alpha", "r_en_overlap", "r_rw_loop", "lid_label", "lid_prob"]].rename(
        columns={"lid_label": "r_lid_label", "lid_prob": "r_lid_prob"}), on=KEY, how="left")
    df["h_z"] = zres(df, "h_")
    df["ind_z"] = df["h_z"] < THR["z"]
    df["ind_c8"] = df["h_c8_rep"] > THR["c8"]
    df["ind_w3"] = df["h_w3_rep"] > THR["w3"]
    df["ind_tail"] = df["h_tail_rep"] > THR["tail"]
    df["ind_loop"] = df["ind_z"] | df["ind_c8"] | df["ind_w3"]
    df["ind_loop_any"] = df["ind_loop"] | df["ind_tail"]
    df["unfinished"] = ~df["finished"]
    df["ratio_ref_chars"] = df["h_chars"] / df["r_chars"].clip(lower=1)
    df["ratio_ref_words"] = df["h_words"] / df["r_words"].clip(lower=1)
    # tail: source ends with final punctuation, hypothesis ends in a letter (mid-word / unterminated)
    df["tail_unterminated"] = df["e_ends_final"] & df["h_ends_alpha"]
    df["copy_like"] = (df["h_en_overlap"] >= 0.6) & (df["h_n_long_words"] >= 4)
    df["indep_or_rw"] = df["ind_loop"] | df["rw_loop"]
    return df


def lid_pass(df, labcol="lid_label", probcol=None):
    ok = np.zeros(len(df), dtype=bool)
    lab = df[labcol].values
    for iso, idx in df.groupby("iso").indices.items():
        acc = set(ACCEPT.get(iso, []))
        ok[idx] = np.array([l in acc for l in lab[idx]])
    return ok


# ------------------------------------------------------------------ LID precision / recall on the references (reliable-language rule: recall >= 0.90)
Rl = R.copy()
Rl["pass"] = lid_pass(Rl)
rows = []
pred_acc = {}
for iso in isos_all:
    acc = set(ACCEPT.get(iso, []))
    if not acc:
        rows.append(dict(iso=iso, covered=False, n=int((Rl.iso == iso).sum()))); continue
    truth = Rl.iso == iso
    pred = Rl.lid_label.isin(acc)
    tp = (truth & pred).sum()
    rows.append(dict(iso=iso, covered=True, labels=" ".join(sorted(acc)), n=int(truth.sum()), recall=tp / max(1, truth.sum()), precision=tp / max(1, pred.sum()),
                     top_wrong=" ".join(f"{k}:{v}" for k, v in Rl[truth & ~pred].lid_label.value_counts().head(3).items())))
lidpr = pd.DataFrame(rows)
lidpr["reliable"] = lidpr["covered"] & (lidpr["recall"] >= 0.90)
lidpr.to_csv(os.path.join(HERE, "lid_nllb218_reference_precision_recall.csv"), index=False)
RELIABLE = set(lidpr[lidpr.reliable.fillna(False)].iso)
# English-leak: top-1 English with p >= 0.5, all languages, control on references
R["eng_leak"] = (R.lid_label == "eng_Latn") & (R.lid_prob >= 0.5)
leak_ref = R.groupby("iso")["eng_leak"].mean()
json.dump({"n_languages_in_refs": len(isos_all), "n_covered_by_nllb218": len(covered), "uncovered": uncovered, "n_reliable_recall_ge_0.90": len(RELIABLE),
           "covered_not_reliable": sorted(set(covered) - RELIABLE)}, open(os.path.join(HERE, "lid_coverage.json"), "w"), indent=1)

# ------------------------------------------------------------------ per model tables
out_rows, per_lang, cross, ex_rows = [], [], [], []
chrf = CHRF(word_order=2)
ref_text = {}
D = {}
for m in MODELS:
    d = add_flags(F[m], R)
    d["lid_ok"] = lid_pass(d)
    d["eng_leak"] = (d.lid_label == "eng_Latn") & (d.lid_prob >= 0.5)
    d["lid_checked"] = d["iso"].isin(RELIABLE) & (d["h_chars"] >= 15)
    D[m] = d
# reference "model": the references scored with the same detectors (control)
Rm = R.copy()
Rm["ind_z"] = Rm["r_z"] < THR["z"]; Rm["ind_c8"] = Rm["r_c8_rep"] > THR["c8"]; Rm["ind_w3"] = Rm["r_w3_rep"] > THR["w3"]
Rm["ind_loop"] = Rm.ind_z | Rm.ind_c8 | Rm.ind_w3
Rm["lid_ok"] = lid_pass(Rm); Rm["lid_checked"] = Rm.iso.isin(RELIABLE) & (Rm.r_chars >= 15)

# common cells across the five main models (A0 has fewer languages): macro over languages
cells = None
for m in MODELS:
    if m == "a0":
        continue
    c = set(zip(D[m].benchmark, D[m].iso))
    cells = c if cells is None else cells & c
cells = sorted(cells)


m_common = True
rows = []
for m in MODELS + ["reference"]:
    d = Rm if m == "reference" else D[m]
    for b in BENCH:
        g = d[d.benchmark == b]
        if m != "a0":
            g = g[pd.Series(list(zip(g.benchmark, g.iso))).isin(cells).values]
        if len(g) == 0:
            continue
        r = dict(model=m, benchmark=b, n_rows=len(g), n_lang=g.iso.nunique())
        flags = ["ind_loop", "ind_z", "ind_c8", "ind_w3"] + ([] if m == "reference" else ["rw_loop", "rw_rep", "rw_selfrep", "ind_tail", "unfinished", "tail_unterminated", "copy_like", "rw_copy", "h_eq_src_exact", "rw_lenbad", "rw_roletail", "eng_leak"])
        for c in flags:
            r[c] = g.groupby("iso")[c].mean().mean()
            r[c + "_pooled"] = g[c].mean()
        if m != "reference":
            r["indep_or_rw"] = g.groupby("iso")["indep_or_rw"].mean().mean()
            r["indep_only_given_not_rw"] = g[~g.rw_loop].groupby("iso")["ind_loop"].mean().mean()
            r["ratio_chars_median"] = g.ratio_ref_chars.median(); r["ratio_chars_p90"] = g.ratio_ref_chars.quantile(0.9); r["ratio_chars_p99"] = g.ratio_ref_chars.quantile(0.99)
            r["ratio_chars_gt2"] = (g.ratio_ref_chars > 2).mean(); r["ratio_chars_gt3"] = (g.ratio_ref_chars > 3).mean()
            r["ratio_chars_lt0.5"] = (g.ratio_ref_chars < 0.5).mean(); r["ratio_chars_lt0.2"] = (g.ratio_ref_chars < 0.2).mean()
            r["ratio_words_gt2"] = (g.ratio_ref_words > 2).mean()
            r["prefix_not_ok"] = (g.prefix_status != "ok").mean()
        gl = g[g.lid_checked]
        r["lid_checked_rows"] = len(gl)
        r["lid_pass_reliable"] = gl.groupby("iso")["lid_ok"].mean().mean() if len(gl) else np.nan
        r["lid_pass_reliable_pooled"] = gl["lid_ok"].mean() if len(gl) else np.nan
        rows.append(r)
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(HERE, "summary_by_model_benchmark.csv"), index=False)

# reference length ratios (control): ref chars / english chars share outside [0.5, 2] x cell median
R["ratio_ref_eng"] = R.r_chars / R.e_chars.clip(lower=1)
json.dump({"ref_len_ratio_vs_cell_median_gt2": float((R.ratio_ref_eng / R.groupby(["benchmark", "iso"]).ratio_ref_eng.transform("median") > 2).mean()),
           "ref_len_ratio_vs_cell_median_lt0.5": float((R.ratio_ref_eng / R.groupby(["benchmark", "iso"]).ratio_ref_eng.transform("median") < 0.5).mean())},
          open(os.path.join(HERE, "reference_length_ratio_control.json"), "w"), indent=1)

# ------------------------------------------------------------------ cross-tabulation reward loop flag x independent flag, quality of each cell
xt = []
for m in MODELS:
    d = D[m]
    for b in BENCH + ["all3"]:
        g = d if b == "all3" else d[d.benchmark == b]
        if len(g) == 0:
            continue
        both = (g.rw_loop & g.ind_loop).sum(); rw_only = (g.rw_loop & ~g.ind_loop).sum(); ind_only = (~g.rw_loop & g.ind_loop).sum(); none = (~g.rw_loop & ~g.ind_loop).sum()
        r = dict(model=m, benchmark=b, n=len(g), both=int(both), rw_only=int(rw_only), ind_only=int(ind_only), neither=int(none),
                 rw_rate=(both + rw_only) / len(g), ind_rate=(both + ind_only) / len(g), union_rate=(both + rw_only + ind_only) / len(g),
                 ind_only_rate=ind_only / len(g),
                 ind_only_unfinished=int((~g.rw_loop & g.ind_loop & g.unfinished).sum()),
                 unfinished_not_rw=int((g.unfinished & ~g.rw_loop).sum()), unfinished_not_ind=int((g.unfinished & ~g.ind_loop).sum()), unfinished=int(g.unfinished.sum()))
        xt.append(r)
cross = pd.DataFrame([x for x in xt if x])
cross.to_csv(os.path.join(HERE, "crosstab_reward_vs_independent.csv"), index=False)

# sentence chrF++ of the four cells, mixedbest/stageA/b1best/sft on FLORES+, needs hyp/ref text: re-read the prediction files
sys.path.insert(0, HERE)
import compute_features as CF
qual = []
for m in [x for x in MODELS if x != "a0"]:
    rows_txt = CF.load(CF.FILES[m])
    key2 = {(r["benchmark"], r["iso"], str(r["id"])): r for r in rows_txt}
    d = D[m]
    for name, mask in [("both", d.rw_loop & d.ind_loop), ("rw_only", d.rw_loop & ~d.ind_loop), ("ind_only", ~d.rw_loop & d.ind_loop), ("neither", ~d.rw_loop & ~d.ind_loop)]:
        idx = np.where(mask.values)[0]
        if len(idx) == 0:
            qual.append(dict(model=m, cell=name, n=0)); continue
        pick = idx if len(idx) <= 2000 else rng.choice(idx, 2000, replace=False)
        sc = []
        for i in pick:
            r = key2[(d.benchmark.iat[i], d.iso.iat[i], d.id.iat[i])]
            sc.append(chrf.sentence_score(CF.nfc(r["hyp"]), [CF.nfc(r["reference"])]).score)
        qual.append(dict(model=m, cell=name, n=int(len(idx)), n_scored=int(len(pick)), mean_sentence_chrfpp=float(np.mean(sc)), median=float(np.median(sc)),
                         share_finished=float(d.finished.values[idx].mean())))
    # examples: independent-only flags, mixedbest and stageA
    if m in ("mixedbest", "stageA", "sft"):
        idx = np.where((~d.rw_loop & d.ind_loop).values)[0]
        pick = idx if len(idx) <= 25 else rng.choice(idx, 25, replace=False)
        for i in pick:
            r = key2[(d.benchmark.iat[i], d.iso.iat[i], d.id.iat[i])]
            ex_rows.append(dict(model=m, benchmark=d.benchmark.iat[i], iso=d.iso.iat[i], id=d.id.iat[i], finished=bool(d.finished.iat[i]),
                                z=float(d.h_z.iat[i]), c8=float(d.h_c8_rep.iat[i]), w3=float(d.h_w3_rep.iat[i]), hyp_chars=int(d.h_chars.iat[i]), ref_chars=int(d.r_chars.iat[i]),
                                hyp=r["hyp"][:300], ref=r["reference"][:200]))
pd.DataFrame(qual).to_csv(os.path.join(HERE, "chrf_by_flag_cell.csv"), index=False)
pd.DataFrame(ex_rows).to_csv(os.path.join(HERE, "examples_independent_only.csv"), index=False)

# ------------------------------------------------------------------ per language (FLORES+): worst languages per model
pl = []
for m in MODELS:
    d = D[m]
    for b in BENCH:
        g = d[d.benchmark == b]
        t = g.groupby("iso").agg(n=("id", "size"), rw_loop=("rw_loop", "mean"), ind_loop=("ind_loop", "mean"), unfinished=("unfinished", "mean"),
                                  ratio_med=("ratio_ref_chars", "median"), gt2=("ratio_ref_chars", lambda s: (s > 2).mean()), copy_like=("copy_like", "mean"),
                                  eng_leak=("eng_leak", "mean"), prefix_not_ok=("prefix_status", lambda s: (s != "ok").mean())).reset_index()
        gl = g[g.lid_checked].groupby("iso")["lid_ok"].mean().rename("lid_pass").reset_index()
        t = t.merge(gl, on="iso", how="left"); t["model"] = m; t["benchmark"] = b
        pl.append(t)
pl = pd.concat(pl)
pl["ref_eng_leak"] = pl["iso"].map(leak_ref)
pl.to_csv(os.path.join(HERE, "per_language.csv"), index=False)
print(json.dumps(THR), len(covered), len(uncovered), len(RELIABLE))
print("ok")
