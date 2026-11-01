"""Vocabulary restriction study, step 2 scoring: per-language and macro chrF++ (sacrebleu word_order=2, NFC) of full / mask / slice hypotheses, paired bootstrap CI of the macro difference
(resampling the sentences WITHIN each language, 2000 resamples, same resample for both systems), finished rate, GlotLID wrong-language rate, identical-output rate, share of outputs that changed.
usage: vocab_restrict_score.py QJSON PREFIX OUTCSV [baseline=full] [variants=mask,slice]   (PREFIX.<mode>.jsonl from vocab_restrict_gen.py)"""
import sys, json, collections, numpy as np, unicodedata, os, csv
from sacrebleu.metrics import CHRF
from gemma_mt.languages import LANGUAGES
qj, pre, outcsv = sys.argv[1:4]; base = sys.argv[4] if len(sys.argv) > 4 else "full"; variants = (sys.argv[5] if len(sys.argv) > 5 else "mask,slice").split(",")
rows = json.load(open(qj)); NB = 2000
nfc = lambda s: unicodedata.normalize("NFC", s)
chrf = CHRF(word_order=2)
def load(mode): return {(d["iso"], str(d["id"])): d for d in map(json.loads, open(f"{pre}.{mode}.jsonl"))}
H = {m: load(m) for m in [base] + variants}
isos = sorted({r["iso"] for r in rows}); by = {iso: [r for r in rows if r["iso"] == iso] for iso in isos}
def stats(mode, iso): return np.array([chrf._extract_corpus_statistics([nfc(H[mode][(iso, str(r["id"]))]["hyp"])], [[nfc(r["reference"])]])[0] for r in by[iso]], dtype=np.float64)
S = {m: {iso: stats(m, iso) for iso in isos} for m in H}
def score(st): return chrf._compute_score_from_stats(list(st.sum(0))).score
import fasttext, glob  # lid.f.predict is used because fasttext.predict breaks with numpy 2
fasttext.FastText.eprint = lambda x: None
lid = fasttext.load_model(glob.glob("/workspace/sft_build/hf/hub/models--cis-lmu--glotlid/snapshots/*/model_v3.bin")[0])
def wrong_lang(mode, iso):
    ok = LANGUAGES[iso].glotlid; n = 0; bad = 0
    for r in by[iso]:
        t = H[mode][(iso, str(r["id"]))]["hyp"].replace("\n", " ").strip()
        if not t: bad += 1; n += 1; continue
        lab = lid.f.predict(t, 1, 0.0, "strict")[0][1].replace("__label__", ""); n += 1; bad += lab not in ok
    return bad / n
rng = np.random.default_rng(0); idx = {iso: rng.integers(0, len(by[iso]), size=(NB, len(by[iso]))) for iso in isos}
def boot(modes):
    return {m: np.array([[score(S[m][iso][idx[iso][b]]) for iso in isos] for b in range(NB)]) for m in modes}  # NB x langs
B = boot(list(H))
out = []
def row(name, mode, extra=None):
    per = {iso: score(S[mode][iso]) for iso in isos}; macro = float(np.mean(list(per.values()))); mb = B[mode].mean(1)
    fin = float(np.mean([H[mode][(r["iso"], str(r["id"]))]["finished"] for r in rows])); wl = {iso: wrong_lang(mode, iso) for iso in isos}
    return per, macro, fin, wl
res = {}
for m in H:
    per, macro, fin, wl = row(m, m); res[m] = (per, macro, fin, wl)
    lo, hi = np.percentile(B[m].mean(1), [2.5, 97.5])
    out.append({"system": m, "lang": "macro", "chrf": round(macro, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2), "finished": round(fin, 4), "wrong_lang": round(float(np.mean(list(wl.values()))), 4)})
    for k, iso in enumerate(isos):
        lo, hi = np.percentile(B[m][:, k], [2.5, 97.5]); fl = float(np.mean([H[m][(iso, str(r["id"]))]["finished"] for r in by[iso]]))
        out.append({"system": m, "lang": iso, "chrf": round(per[iso], 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2), "finished": round(fl, 4), "wrong_lang": round(wl[iso], 4)})
for v in variants:
    d = B[v] - B[base]
    lo, hi = np.percentile(d.mean(1), [2.5, 97.5])
    out.append({"system": f"{v}-{base}", "lang": "macro", "chrf": round(res[v][1] - res[base][1], 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3)})
    for k, iso in enumerate(isos):
        lo, hi = np.percentile(d[:, k], [2.5, 97.5]); same = np.mean([H[v][(iso, str(r["id"]))]["ids"] == H[base][(iso, str(r["id"]))]["ids"] for r in by[iso]])
        out.append({"system": f"{v}-{base}", "lang": iso, "chrf": round(res[v][0][iso] - res[base][0][iso], 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "identical_ids": round(float(same), 4)})
    same = np.mean([H[v][(r["iso"], str(r["id"]))]["ids"] == H[base][(r["iso"], str(r["id"]))]["ids"] for r in rows]); out[-len(isos) - 1]["identical_ids"] = round(float(same), 4)
keys = ["system", "lang", "chrf", "ci_lo", "ci_hi", "finished", "wrong_lang", "identical_ids"]
with open(outcsv, "w", newline="") as f:
    w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(out)
for r in out: print(r)
