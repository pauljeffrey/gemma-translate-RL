"""Aggregate sc/*.pkl into CSV tables (per cell, macro with document-cluster bootstrap, pairs).  usage: lc_agg.py [--boot 400]
Outputs RES/cells.csv, macro.csv, pair_mode.csv (sentence minus whole per model), pair_model.csv (every model minus a reference model, same mode), lang_sets.json"""
import sys, os, glob, pickle, json, csv, collections, argparse, zlib
import numpy as np
from lc_common import *

ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=400); ap.add_argument("--only", default=""); ap.add_argument("--langs", default=""); ap.add_argument("--tag", default="")
args = ap.parse_args()
NB = args.boot


def chrf_vec(S):
    S = np.asarray(S, dtype=np.float64)
    nh, nr, nm = S[..., 0::3], S[..., 1::3], S[..., 2::3]
    ok = (nh > 0) & (nr > 0)
    prec = np.where(nh > 0, nm / np.maximum(nh, 1), 0.0); rec = np.where(nr > 0, nm / np.maximum(nr, 1), 0.0)
    eff = ok.sum(-1)
    ap_ = (prec * ok).sum(-1) / np.maximum(eff, 1); ar = (rec * ok).sum(-1) / np.maximum(eff, 1)
    den = 4 * ap_ + ar
    return np.where(den > 0, 100 * 5 * ap_ * ar / np.where(den > 0, den, 1), 0.0)


def bleu_vec(S):
    S = np.asarray(S, dtype=np.float64)
    sl, rl, corr, tot = S[..., 0], S[..., 1], S[..., 2:6], S[..., 6:10]
    z = (corr == 0) & (tot > 0); k = np.cumsum(z, -1)
    prec = np.where(tot > 0, np.where(corr > 0, 100.0 * corr / np.maximum(tot, 1), 100.0 / (2.0 ** k * np.maximum(tot, 1))), 0.0)
    lg = np.log(np.maximum(prec, 1e-12) / 100.0)
    bp = np.where(sl < rl, np.exp(1 - rl / np.maximum(sl, 1)), 1.0)
    ok = (tot > 0).all(-1) & (corr[..., 0] > 0)
    return np.where(ok, 100.0 * bp * np.exp(lg.mean(-1)), 0.0)


# ---------------------------------------------------------------- load
SYS = {}
for p in sorted(glob.glob(SC + "/*.pkl")):
    n = os.path.basename(p)[:-4]
    if args.only and n.split("__")[0] not in args.only.split(","):
        continue
    SYS[n] = pickle.load(open(p, "rb"))
    if args.langs:
        SYS[n] = [r for r in SYS[n] if r["iso"] in args.langs.split(",")]
print("systems:", list(SYS))
SELF = {"nllb3b3": {"nllb"}, "gemini": {"gemini"}}            # a system is never scored against its own pseudo-reference (reported as 'self')
METRICS = ["finished", "loop", "cov_sents", "cov_chars", "length_ratio", "lid_sent", "lid_doc_ok", "tok_ratio"]


def metric_arrays(rows):
    out = {}
    for k in METRICS:
        if k == "tok_ratio":
            out[k] = np.array([(r["new_tokens"] or 0) / max(r["src_tokens"], 1) for r in rows], dtype=float)
        else:
            out[k] = np.array([np.nan if r.get(k) is None else float(r[k]) for r in rows], dtype=float)
    return out


def build(rows):
    """{(set,bin): {iso: {doc_id: row}}}"""
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        d[(r["set"], r["bin"])][r["iso"]][r["doc_id"]] = r
    return d

CELLS = {n: build(rows) for n, rows in SYS.items()}
cells_out, macro_out = [], []


def lang_arrays(cell, ref, langs):
    """per language: (doc ids, C (n,24), B (n,10), metric dict) for documents that have the reference"""
    out = {}
    for iso in langs:
        docs_ = sorted(cell.get(iso, {}))
        rws = [cell[iso][d] for d in docs_ if ref in cell[iso][d]["stats"]] if ref else [cell[iso][d] for d in docs_]
        ids = [r["doc_id"] for r in rws]
        if not rws:
            continue
        out[iso] = {"ids": ids, "rows": rws, "M": metric_arrays(rws)}
        if ref:
            out[iso]["C"] = np.stack([r["stats"][ref][0] for r in rws]); out[iso]["B"] = np.stack([r["stats"][ref][1] for r in rws])
    return out


def macro_boot(arrs, rng_seed=0, nb=NB):
    """arrs: {sys_name: {iso: arrays}} on a COMMON set of languages and documents. Same document draws for every system (paired). returns per system point + CI + draws"""
    names = list(arrs)
    isos = sorted(set.intersection(*[set(a) for a in arrs.values()]))
    if not isos:
        return None, []
    # groups of languages that share the identical document list (resampled together)
    keyed = collections.defaultdict(list)
    for iso in isos:
        keyed[tuple(arrs[names[0]][iso]["ids"])].append(iso)
    rng = np.random.RandomState(rng_seed)
    draws = []
    for _ in range(nb):
        draws.append({k: rng.randint(0, len(k), len(k)) for k in keyed})
    res = {}
    for n in names:
        a = arrs[n]
        pt = {"chrf": [], "bleu": []}
        # point estimates per language
        for iso in isos:
            if "C" in a[iso]:
                pt["chrf"].append(float(chrf_vec(a[iso]["C"].sum(0)))); pt["bleu"].append(float(bleu_vec(a[iso]["B"].sum(0))))
        out = {"isos": isos, "per_lang_chrf": dict(zip(isos, pt["chrf"])) if pt["chrf"] else {}, "per_lang_bleu": dict(zip(isos, pt["bleu"])) if pt["bleu"] else {}}
        out["chrf"] = float(np.mean(pt["chrf"])) if pt["chrf"] else np.nan; out["bleu"] = float(np.mean(pt["bleu"])) if pt["bleu"] else np.nan
        for k in METRICS:
            v = [np.nanmean(a[iso]["M"][k]) if not np.all(np.isnan(a[iso]["M"][k])) else np.nan for iso in isos]
            out[k] = float(np.nanmean(v)) if not all(np.isnan(v)) else np.nan
        cs, bs, ms = [], [], {k: [] for k in METRICS}
        for dr in draws:
            c_l, b_l, m_l = [], [], {k: [] for k in METRICS}
            for key, lg in keyed.items():
                ix = dr[key]
                for iso in lg:
                    if "C" in a[iso]:
                        c_l.append(float(chrf_vec(a[iso]["C"][ix].sum(0)))); b_l.append(float(bleu_vec(a[iso]["B"][ix].sum(0))))
                    for k in METRICS:
                        x = a[iso]["M"][k][ix]
                        m_l[k].append(np.nanmean(x) if not np.all(np.isnan(x)) else np.nan)
            cs.append(np.mean(c_l) if c_l else np.nan); bs.append(np.mean(b_l) if b_l else np.nan)
            for k in METRICS:
                ms[k].append(np.nanmean(m_l[k]) if not all(np.isnan(m_l[k])) else np.nan)
        out["draws"] = {"chrf": np.array(cs), "bleu": np.array(bs), **{k: np.array(v) for k, v in ms.items()}}
        res[n] = out
    return res, isos


def ci(x):
    x = np.asarray(x, dtype=float)
    return (float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))) if np.any(~np.isnan(x)) else (np.nan, np.nan)


# ---------------------------------------------------------------- cells (per model, mode, set, bin, language, reference)
for n, cells in CELLS.items():
    model, mode = n.split("__")
    for (setn, binn), bylang in cells.items():
        for ref in ("human", "nllb", "gemini", None):
            la = lang_arrays(bylang, ref, sorted(bylang))
            for iso, a in la.items():
                row = {"model": model, "mode": mode, "set": setn, "bin": binn, "lang": iso, "ref": ref or "none", "n_docs": len(a["ids"]),
                       "src_tokens_mean": float(np.mean([r["src_tokens"] for r in a["rows"]]))}
                if ref:
                    row["chrf"] = float(chrf_vec(a["C"].sum(0))); row["spbleu"] = float(bleu_vec(a["B"].sum(0)))
                for k in METRICS:
                    row[k] = float(np.nanmean(a["M"][k])) if not np.all(np.isnan(a["M"][k])) else np.nan
                cells_out.append(row)

# ---------------------------------------------------------------- macro tables with bootstrap CIs, per (set, bin, ref): language set = languages present for the system
keys = sorted({(s, b) for cells in CELLS.values() for (s, b) in cells})
pairs_mode, pairs_model = [], []
for (setn, binn) in keys:
    for ref in ("human", "nllb", "gemini", None):
        # each system on its own languages (the table lists n_lang), CIs paired within the same call only when languages coincide -> do per-system calls
        percell = {}
        for n, cells in CELLS.items():
            if (setn, binn) not in cells:
                continue
            la = lang_arrays(cells[(setn, binn)], ref, sorted(cells[(setn, binn)]))
            if la:
                percell[n] = la
        if not percell:
            continue
        for n, la in percell.items():
            model, mode = n.split("__")
            if ref in SELF.get(model, set()):
                continue
            res, isos = macro_boot({n: la}, rng_seed=zlib.crc32(f"{setn}{binn}".encode()) % 1000)
            r = res[n]
            row = {"model": model, "mode": mode, "set": setn, "bin": binn, "ref": ref or "none", "n_lang": len(isos), "langs": " ".join(isos),
                   "n_docs": int(np.mean([len(la[i]["ids"]) for i in isos])), "src_tokens_mean": float(np.mean([x["src_tokens"] for i in isos for x in la[i]["rows"]]))}
            if ref:
                row.update({"chrf": r["chrf"], "chrf_lo": ci(r["draws"]["chrf"])[0], "chrf_hi": ci(r["draws"]["chrf"])[1], "spbleu": r["bleu"], "spbleu_lo": ci(r["draws"]["bleu"])[0], "spbleu_hi": ci(r["draws"]["bleu"])[1]})
            for k in METRICS:
                row[k] = r[k]; row[k + "_lo"], row[k + "_hi"] = ci(r["draws"][k])
            macro_out.append(row)
        # paired: sentence minus whole for each model (same languages / docs)
        models = sorted({n.split("__")[0] for n in percell})
        for m in models:
            if f"{m}__whole" in percell and any(f"{m}__{x}" in percell for x in ("sent", "sentfix")):
                sm_ = f"{m}__sentfix" if f"{m}__sentfix" in percell else f"{m}__sent"
                sub = {k: {i: v for i, v in percell[k].items()} for k in (f"{m}__whole", sm_)}
                common = set(sub[f"{m}__whole"]) & set(sub[sm_])
                sub = {k: {i: v[i] for i in common} for k, v in sub.items()}
                if not common or (ref in SELF.get(m, set())):
                    continue
                res, isos = macro_boot(sub, rng_seed=7)
                a, b = res[sm_], res[f"{m}__whole"]
                for met in (["chrf", "bleu"] if ref else []) + ["finished", "loop", "cov_sents"]:
                    d = a["draws"][met] - b["draws"][met]
                    pairs_mode.append({"model": m, "set": setn, "bin": binn, "ref": ref or "none", "metric": met, "sent_mode": sm_.split("__")[1], "n_lang": len(isos),
                                       "delta": float(a[met] - b[met]), "lo": ci(d)[0], "hi": ci(d)[1]})
print("cells", len(cells_out), "macro", len(macro_out), "pairs_mode", len(pairs_mode))

def dump(name, rows):
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    os.makedirs(RES, exist_ok=True)
    with open(f"{RES}/{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

dump("cells" + args.tag, cells_out); dump("macro" + args.tag, macro_out); dump("pair_mode" + args.tag, pairs_mode)
