#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AfriCOMET (reference-based STL-1.1 and reference-free QE) and SSA-COMET-MTL of several hypothesis sets on the SAME sentences, per language and macro, with a paired bootstrap of the differences.

    python scripts/post_editor_comet.py --out DIR --system draft=ED/edited_predictions.jsonl:draft --system edited=ED/edited_predictions.jsonl:hyp \
        --system beam4=B/predictions_beam4.jsonl --system mbr16=B/predictions_mbr16.jsonl --pair edited:draft --pair beam4:draft --pair mbr16:draft [--ssa] [--subsets all,clean,no_pair]

`NAME=path[:field]` reads the hypothesis from `field` (default `hyp`). Runs in the AfriCOMET environment (unbabel-comet, transformers 4.x; GPU, fp16). Languages: the 42 whose AfriCOMET validity was measured
(gemma_mt.languages.QE_VALIDATED); other rows get no score. Scores are cached by sha1(english, hypothesis, reference) in DIR/comet_cache_*.jsonl (a rerun scores only new texts).
Writes DIR/comet_by_language.csv, comet_macro.csv (system x benchmark x subset), comet_pairs.csv (paired bootstrap, macro equal weight per language), comet_pairs_by_language.csv.
"""
import argparse, csv, hashlib, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
SUBSETS = {"all": lambda r: True, "clean": lambda r: r.get("contaminated") is False, "no_pair": lambda r: r.get("pair_seen") is False}
key = lambda s, m, r="": hashlib.sha1(f"{s}\x00{m}\x00{r}".encode("utf-8")).hexdigest()


def load_cache(p):
    d = {}
    if os.path.exists(p):
        for l in open(p, encoding="utf-8"):
            try:
                k, v = json.loads(l); d[k] = float(v)
            except Exception:
                pass
    return d


def score_missing(scorer, todo_dict, cache, path, reference, batch=2000):
    todo = list(todo_dict.items())
    with open(path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            sc = scorer([v[0] for _, v in part], [v[1] for _, v in part], [v[2] for _, v in part] if reference else None)
            for (k, _), x in zip(part, sc):
                cache[k] = float(x); f.write(json.dumps([k, float(x)]) + "\n")
            f.flush(); print(f"  scored {min(i + batch, len(todo))}/{len(todo)}", flush=True)


def boot_macro(deltas_by_lang, B=1000, seed=0):
    """deltas_by_lang: {iso: np.array of paired sentence differences}. -> (point macro, lo, hi, per-language list of (iso, n, mean, lo, hi))"""
    import numpy as np
    rng = np.random.default_rng(seed); acc = np.zeros(B); per = []
    for iso, d in sorted(deltas_by_lang.items()):
        n = len(d); idx = rng.integers(0, n, size=(B, n)); bs = d[idx].mean(1); acc += bs
        per.append((iso, n, float(d.mean()), *[float(x) for x in np.percentile(bs, [2.5, 97.5])]))
    acc /= max(1, len(deltas_by_lang))
    return (float(np.mean([p[2] for p in per])) if per else float("nan")), float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5)), per


def parse_system(spec):
    name, rest = spec.split("=", 1)
    path, field = rest, "hyp"
    if ":" in rest and not os.path.exists(rest):
        path, field = rest.rsplit(":", 1)
    return name, path, field


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="append", required=True); ap.add_argument("--pair", action="append", default=[], help="B:A, delta = B - A")
    ap.add_argument("--out", required=True); ap.add_argument("--subsets", default="all,clean"); ap.add_argument("--ssa", action="store_true"); ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--min-n", type=int, default=30); ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--benchmarks", nargs="*", default=None); ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--metrics", default=None, help="comma list of comet_ref,comet_qe,ssa_qe (default comet_ref,comet_qe [+ ssa_qe with --ssa]); never mix covered and uncovered languages in one macro: run ssa_qe with --all-languages as its own invocation")
    ap.add_argument("--all-languages", action="store_true", help="score languages outside the 42 validated ones as well (the checkpoint names more languages than it was validated on)")
    a = ap.parse_args(argv)
    import numpy as np
    from gemma_mt.languages import QE_VALIDATED
    from gemma_mt.rl.comet import QE_MODEL_ID, REF_MODEL_ID, SSA_MTL_MODEL_ID, CometScorer
    os.makedirs(a.out, exist_ok=True); subs = a.subsets.split(","); ok_langs = set(QE_VALIDATED)
    S = {}
    for spec in a.system:
        name, path, field = parse_system(spec); rows = {}
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            if (not a.all_languages and r["iso"] not in ok_langs) or (a.benchmarks and r["benchmark"] not in a.benchmarks): continue
            rows[(r["benchmark"], r["iso"], str(r["id"]))] = dict(r, _h=r.get(field, "") or "")
        S[name] = rows; print(name, len(rows), "rows in covered languages", flush=True)
    metrics = [("comet_ref", REF_MODEL_ID, True), ("comet_qe", QE_MODEL_ID, False)] + ([("ssa_qe", SSA_MTL_MODEL_ID, False)] if a.ssa else [])
    if a.metrics:
        allm = {"comet_ref": ("comet_ref", REF_MODEL_ID, True), "comet_qe": ("comet_qe", QE_MODEL_ID, False), "ssa_qe": ("ssa_qe", SSA_MTL_MODEL_ID, False)}
        metrics = [allm[m] for m in a.metrics.split(",")]
    scores = {}
    for mname, mid, ref in metrics:
        cpath = os.path.join(a.out, f"comet_cache_{mname}.jsonl"); cache = load_cache(cpath)
        need = {}
        for rows in S.values():
            for r in rows.values():
                if not r["_h"].strip(): continue
                k = key(r["english"], r["_h"], r["reference"] if ref else "")
                if k not in cache: need[k] = (r["english"], r["_h"], r["reference"])
        print(f"[{mname}] {len(need)} texts to score ({len(cache)} cached)", flush=True)
        if need:
            sc = CometScorer(mid, batch_size=a.batch_size, device="cpu" if a.cpu else None, half=not a.cpu, reference_based=ref)
            score_missing(sc, need, cache, cpath, ref)
            del sc
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
        scores[mname] = cache
    for rows in S.values():
        for r in rows.values():
            for mname, mid, ref in metrics:
                r["_" + mname] = scores[mname].get(key(r["english"], r["_h"], r["reference"] if ref else "")) if r["_h"].strip() else None
    mnames = [m[0] for m in metrics]
    macro = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(a.out, "comet_by_language.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["system", "benchmark", "subset", "iso", "n"] + mnames)
        for name, rows in S.items():
            cell = defaultdict(list)
            for (b, iso, i), r in rows.items():
                for sub in subs:
                    if SUBSETS[sub](r): cell[(b, sub, iso)].append(r)
            for (b, sub, iso), rs in sorted(cell.items()):
                vals = []
                for m in mnames:
                    v = [r["_" + m] for r in rs if r["_" + m] is not None]
                    vals.append(float(np.mean(v)) if v else float("nan"))
                w.writerow([name, b, sub, iso, len(rs)] + [round(v, 4) for v in vals])
                if len(rs) >= a.min_n:
                    for m, v in zip(mnames, vals): macro[(name, b, sub)][m].append(v)
    with open(os.path.join(a.out, "comet_macro.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["system", "benchmark", "subset", "n_languages"] + mnames)
        for (name, b, sub), d in sorted(macro.items()):
            w.writerow([name, b, sub, len(d[mnames[0]])] + [round(float(np.nanmean(d[m])), 4) for m in mnames])
    fp = open(os.path.join(a.out, "comet_pairs.csv"), "w", newline=""); wp = csv.writer(fp)
    wp.writerow(["pair", "benchmark", "subset", "metric", "n_languages", "macro_b", "macro_a", "delta", "lo", "hi", "langs_better", "langs_ns", "langs_worse"])
    fl = open(os.path.join(a.out, "comet_pairs_by_language.csv"), "w", newline=""); wl = csv.writer(fl); wl.writerow(["pair", "benchmark", "subset", "iso", "metric", "n", "delta", "lo", "hi"])
    for pr in a.pair:
        b_, a_ = pr.split(":")
        for b in sorted({k[0] for k in S[b_]}):
            for sub in subs:
                for m in mnames:
                    dl, vb, va = {}, [], []
                    by = defaultdict(list)
                    for k, rb in S[b_].items():
                        if k[0] != b or k not in S[a_] or not SUBSETS[sub](rb): continue
                        x, y = rb["_" + m], S[a_][k]["_" + m]
                        if x is not None and y is not None: by[k[1]].append((x, y))
                    for iso, v in by.items():
                        if len(v) >= a.min_n:
                            arr = np.array(v); dl[iso] = arr[:, 0] - arr[:, 1]; vb.append(arr[:, 0].mean()); va.append(arr[:, 1].mean())
                    if not dl: continue
                    pt, lo, hi, per = boot_macro(dl, a.boot)
                    for iso, n, d, l, h in per: wl.writerow([pr, b, sub, iso, m, n, round(d, 4), round(l, 4), round(h, 4)])
                    wp.writerow([pr, b, sub, m, len(dl), round(float(np.mean(vb)), 4), round(float(np.mean(va)), 4), round(pt, 4), round(lo, 4), round(hi, 4),
                                 sum(p[3] > 0 for p in per), sum(p[3] <= 0 <= p[4] for p in per), sum(p[4] < 0 for p in per)])
    fp.close(); fl.close()
    print(open(os.path.join(a.out, "comet_pairs.csv")).read())


if __name__ == "__main__":
    main()
