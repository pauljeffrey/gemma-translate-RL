#!/usr/bin/env python3
"""Score and compare any number of systems from their `predictions.jsonl` (the file `gemma_mt.eval.sft_benchmarks` / the decoding runs write): per-language and macro tables
with chrF++, spBLEU, finished / empty / copy / repetition / language-ID rates, and a paired bootstrap between named pairs of systems. CPU only.

    python scripts/eval_385m_score.py --system a0=OUT_a0/predictions.jsonl --system m385=OUT_385/predictions.jsonl --out DIR \
        [--pair m385:a0 ...] [--groups groups.json] [--subsets all,clean,no_pair] [--boot 1000] [--processes 8] [--lid-model model_v3.bin | --no-lid] [--min-n 30]

Rows: {"benchmark", "iso", "id", "english", "reference", "hyp", "finished"?, "contaminated"?, "pair_seen"?, "near_dup"?}; a system's rows are matched to another's by (benchmark, iso, id).
Subsets (eval_translation.SUBSETS): all, clean, clean_strict, no_pair; a row without the flag falls out of the flagged subset. gatitos and smol_doc are skipped (word / document level).

Output (DIR): metrics_by_language.csv    one row per system x benchmark x subset x language
              macro.csv                  system x benchmark x subset (+ group): equal weight per language; `common` = only languages with n >= min-n in EVERY listed system
              bootstrap_pairs.csv        per pair x benchmark x subset (+ group): macro delta of chrF++ and spBLEU with a 95% paired bootstrap interval
              bootstrap_by_language.csv  the same per language
The bootstrap resamples the sentences of each language with replacement (the same resample for both systems), takes the corpus score from the summed sufficient statistics of the resample
(sacrebleu CHRF word_order 2 / BLEU flores200, NFC), and averages the language deltas of the same resample, so the interval is for the equal-language-weight macro. It covers test-sentence
variance only (one training seed per system).
"""
import argparse, csv, glob, json, os, sys, unicodedata
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
SUBSETS = {
    "all": lambda r: True,
    "clean": lambda r: r.get("contaminated") is False,
    "clean_strict": lambda r: r.get("contaminated") is False and r.get("near_dup") is False,
    "no_pair": lambda r: r.get("pair_seen") is False,
}
SKIP_BENCHMARKS = ("gatitos", "smol_doc")
nfc = lambda s: unicodedata.normalize("NFC", s)
_G = {}


def _flags(rows):
    """copy / repetition flags of the rows (rl.guards: the same rules as the reward's constraint penalties)."""
    from gemma_mt.rl.guards import is_copy_of_source, is_repetitive, is_self_repetitive
    for r in rows:
        h = r["hyp"]; e = r.get("english", "")
        r["_copy"] = bool(h.strip()) and len(e.split()) >= 4 and is_copy_of_source(e, h, 0.9)
        r["_rep"] = bool(h.strip()) and (is_repetitive(h) or is_self_repetitive(h))


def _lid(rows, model_path):
    from gemma_mt.rl import lid as L
    from gemma_mt.languages import LANGUAGES
    clf = L.GlotLIDClassifier(path=model_path); L.selftest(clf); known = set(clf.labels()); acc = {}
    for iso, lg in LANGUAGES.items():
        own = tuple(x for x in lg.glotlid if x in known)
        acc[iso] = (set(own), set(own) | {x for x in L.siblings_of(own) if x in known}) if own else None
    top = clf([r["hyp"] for r in rows])
    for r, t in zip(rows, top):
        a = acc.get(r["iso"])
        r["_lid"] = None if a is None else ((False, False) if not r["hyp"].strip() else (t in a[0], t in a[1]))


def load(path, lid_model):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["benchmark"] not in SKIP_BENCHMARKS]
    _flags(rows)
    if lid_model: _lid(rows, lid_model)
    else:
        for r in rows: r["_lid"] = None
    return rows


def cell_metrics(args):
    key, rows = args
    from gemma_mt.eval import metrics as M
    n = len(rows); hyps = [r["hyp"] for r in rows]; refs = [r["reference"] for r in rows]
    lv = [r["_lid"] for r in rows if r["_lid"] is not None]
    return dict(zip(("system", "benchmark", "subset", "iso"), key), n=n, chrf=M.chrf_pp(hyps, refs), spbleu=M.spbleu(hyps, refs),
                finished=sum(bool(r.get("finished", True)) for r in rows) / n, empty=sum(not h.strip() for h in hyps) / n,
                copy=sum(r["_copy"] for r in rows) / n, repeat=sum(r["_rep"] for r in rows) / n,
                lid_ok=(sum(v[0] for v in lv) / len(lv)) if lv else "", lid_ok_sib=(sum(v[1] for v in lv) / len(lv)) if lv else "")


def boot_cell(job):
    """job = (pair, benchmark, subset, iso, hyps_a, hyps_b, refs, B, seed) -> point delta and the B resampled deltas of chrF++ and spBLEU (b minus a)."""
    from sacrebleu.metrics import CHRF, BLEU
    pair, bench, sub, iso, ha, hb, refs, B, seed = job
    chrf, bleu = CHRF(word_order=2), BLEU(tokenize="flores200"); refs = [nfc(r) for r in refs]; n = len(refs)
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)
    out = {}
    for name, m in (("chrf", chrf), ("spbleu", bleu)):
        sa = np.array(m._extract_corpus_statistics([nfc(h) for h in ha], [refs]), dtype=np.float64)
        sb = np.array(m._extract_corpus_statistics([nfc(h) for h in hb], [refs]), dtype=np.float64)
        score = lambda s: m._compute_score_from_stats(list(s)).score
        pt = score(sb.sum(0)) - score(sa.sum(0))
        ra, rb = counts @ sa, counts @ sb
        out[name] = (pt, np.array([score(b) - score(a) for a, b in zip(ra, rb)]))
    return (pair, bench, sub, iso, n), out


def ci(v):
    lo, hi = np.percentile(v, [2.5, 97.5]); return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True, metavar="NAME=predictions.jsonl")
    ap.add_argument("--out", required=True); ap.add_argument("--pair", action="append", default=[], metavar="B:A", help="delta = B minus A")
    ap.add_argument("--groups", default=None, help="json {iso: group}; group macros are added (languages without a group fall under 'ungrouped')")
    ap.add_argument("--subsets", default="all,clean,no_pair"); ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lid-model", default=None, help="GlotLID model_v3.bin (default: the Hub cache under HF_HOME if present)"); ap.add_argument("--no-lid", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    subs = [s for s in a.subsets.split(",") if s]; groups = json.load(open(a.groups)) if a.groups else {}
    lid_model = None
    if not a.no_lid:
        lid_model = a.lid_model or next(iter(sorted(glob.glob(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub/models--cis-lmu--glotlid/snapshots/*/model_v3.bin")))), None)
        if lid_model is None: sys.exit("no GlotLID model_v3.bin found: pass --lid-model or --no-lid")
    systems = {}
    for s in a.system:
        name, path = s.split("=", 1); systems[name] = load(path, lid_model); print(name, len(systems[name]), "rows", flush=True)
    cells = defaultdict(list)
    for name, rows in systems.items():
        for r in rows:
            for sub in subs:
                if SUBSETS[sub](r): cells[(name, r["benchmark"], sub, r["iso"])].append(r)
    from gemma_mt.eval import metrics as M
    M.spbleu(["a b c"], ["a b c"]); tokn = M.spbleu_tokenizer()          # flores200 or the offline 13a fallback; the bootstrap refuses the fallback
    open(os.path.join(a.out, "spbleu_tokenizer.txt"), "w").write(tokn + "\n"); print("spBLEU tokenizer:", tokn, flush=True)
    with Pool(a.processes) as p: recs = p.map(cell_metrics, sorted(cells.items()), chunksize=4)
    fields = ["system", "benchmark", "subset", "iso", "n", "chrf", "spbleu", "finished", "empty", "copy", "repeat", "lid_ok", "lid_ok_sib"]
    with open(os.path.join(a.out, "metrics_by_language.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(recs)
    # ---- macro: equal weight per language; "own" = every language of the system with n >= min-n, "common" = intersection over the listed systems
    by = defaultdict(dict)
    for r in recs:
        if r["n"] >= a.min_n: by[(r["benchmark"], r["subset"])].setdefault(r["system"], {})[r["iso"]] = r
    mac = []
    for (bench, sub), per_sys in sorted(by.items()):
        common = set.intersection(*[set(per_sys.get(s, {})) for s in systems]) if len(per_sys) == len(systems) else set()
        for s, d in sorted(per_sys.items()):
            for scope, isos in (("own", set(d)), ("common", common)):
                for g in ["all"] + sorted(set(groups.get(i, "ungrouped") for i in isos) if groups else []):
                    sel = [d[i] for i in sorted(isos) if g == "all" or groups.get(i, "ungrouped") == g]
                    if not sel: continue
                    row = dict(system=s, benchmark=bench, subset=sub, scope=scope, group=g, n_languages=len(sel), n_sentences=sum(x["n"] for x in sel))
                    for k in ("chrf", "spbleu", "finished", "empty", "copy", "repeat"): row[k] = float(np.mean([x[k] for x in sel]))
                    lv = [x["lid_ok"] for x in sel if x["lid_ok"] != ""]; row["lid_ok"] = float(np.mean(lv)) if lv else ""
                    mac.append(row)
    mf = ["system", "benchmark", "subset", "scope", "group", "n_languages", "n_sentences", "chrf", "spbleu", "finished", "empty", "copy", "repeat", "lid_ok"]
    with open(os.path.join(a.out, "macro.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mf); w.writeheader(); w.writerows(mac)
    print("wrote metrics_by_language.csv", len(recs), "macro.csv", len(mac), flush=True)
    if not a.pair: return
    # ---- paired bootstrap
    jobs = []; idx = {name: {(r["benchmark"], r["iso"], str(r["id"])): r for r in rows} for name, rows in systems.items()}
    for pr in a.pair:
        b_, a_ = pr.split(":")
        for (name, bench, sub, iso), rows_b in sorted(cells.items()):
            if name != b_: continue
            ids = [str(r["id"]) for r in rows_b if (bench, iso, str(r["id"])) in idx[a_]]
            if len(ids) < a.min_n: continue
            ra = [idx[a_][(bench, iso, i)] for i in ids]; rb = [idx[b_][(bench, iso, i)] for i in ids]
            jobs.append((pr, bench, sub, iso, [r["hyp"] for r in ra], [r["hyp"] for r in rb], [r["reference"] for r in rb], a.boot, a.seed + len(jobs)))
    with Pool(a.processes) as p: res = p.map(boot_cell, jobs, chunksize=2)
    with open(os.path.join(a.out, "bootstrap_by_language.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pair", "benchmark", "subset", "iso", "n", "d_chrf", "d_chrf_lo", "d_chrf_hi", "d_spbleu", "d_spbleu_lo", "d_spbleu_hi"])
        for (pr, bench, sub, iso, n), o in res: w.writerow([pr, bench, sub, iso, n, o["chrf"][0], *ci(o["chrf"][1]), o["spbleu"][0], *ci(o["spbleu"][1])])
    agg = defaultdict(list)
    for (pr, bench, sub, iso, n), o in res:
        for g in ["all"] + ([groups.get(iso, "ungrouped")] if groups else []): agg[(pr, bench, sub, g)].append(o)
    with open(os.path.join(a.out, "bootstrap_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pair", "benchmark", "subset", "group", "n_languages", "d_chrf", "d_chrf_lo", "d_chrf_hi", "d_spbleu", "d_spbleu_lo", "d_spbleu_hi", "p_b_better_chrf"])
        for (pr, bench, sub, g), os_ in sorted(agg.items()):
            m = {k: np.mean([o[k][1] for o in os_], axis=0) for k in ("chrf", "spbleu")}; pt = {k: float(np.mean([o[k][0] for o in os_])) for k in m}
            w.writerow([pr, bench, sub, g, len(os_), pt["chrf"], *ci(m["chrf"]), pt["spbleu"], *ci(m["spbleu"]), float((m["chrf"] > 0).mean())])
    print("wrote bootstrap_pairs.csv, bootstrap_by_language.csv", len(res), "cells", flush=True)


if __name__ == "__main__":
    main()
