#!/usr/bin/env python3
"""Tables for the 385M contamination audit, from the output of `contam_audit_14_385m_scan.py finalize` (CPU, seconds).

    python contam_audit_14_report.py --work WORK --contam-dir WORK/contamination --bench-root ... --extra-root ... --tree ... --out-dir WORK/analysis

Writes into --out-dir:
  exposure_by_benchmark.csv          per benchmark: rows, distinct English sentences, exposed rows / distinct sentences by the rules, and the same restricted to English of >= 3 and >= 8
                                     normalised tokens (the intake filter ignores shorter protected strings), target-side counts, shares in percent
  exposed_rows.csv                   one line per flagged row of the eight evaluation benchmarks + AfriScience (key columns only, no benchmark text): benchmark, iso, id, token lengths, flags
  hits_by_dataset.csv                for the evaluation benchmarks: the training samples that contain an exposed English (kind 0) or reference (kind 1) string, by dataset (s1..s6, g0), sample kind
                                     (translation wrapper / other), side of the sample (prompt, target), sample language
  min3tok/contamination_<b>.json     the flag files of the evaluation benchmarks with exact / sentence / pair cleared for English of fewer than 3 normalised tokens (Yes, names, headings)
  summary_extra.json                 the FLORES+ dev check (held-out half vs train half), counts by sample kind
"""
import argparse, csv, json, os, pickle, sys
from collections import Counter, defaultdict

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"; os.execv(sys.executable, [sys.executable] + sys.argv)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True); ap.add_argument("--contam-dir", required=True); ap.add_argument("--bench-root", required=True); ap.add_argument("--extra-root", required=True)
    ap.add_argument("--tree", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(); sys.path.insert(0, a.tree)
    import contam_audit_14_385m_scan as M
    from gemma_mt.eval import contamination as C
    os.makedirs(a.out_dir, exist_ok=True)
    sets = M.load_rows(a); N = C.normalise
    ntok = lambda s: len(N(s).split())
    H = pickle.load(open(os.path.join(a.work, "hits.pkl"), "rb")); hits = H["hits"]; fwd_texts = H["fwd_texts"]; rev_texts = H["rev_texts"]
    fid = {t: i for i, t in enumerate(fwd_texts)}
    def load(b):
        f = os.path.join(a.contam_dir, f"contamination_{b}.json" if b in M.EVAL_BENCHMARKS else os.path.join("extra", f"contamination_{b}.json"))
        return json.load(open(f)), json.load(open(os.path.join(a.contam_dir, "extra", f"targetside_{b}.json")))
    rows_out, exposed = [], []
    for b, rows in sets.items():
        fl, ts = load(b)
        agg = Counter(); den = set(); den_exp = set(); den_exp3 = set(); den_exp8 = set(); langs = set(); den_any8 = set()
        for r in rows:
            k = C.row_key(r["iso"], r["id"]); en = N(r["english"]); le = len(en.split()); lr = ntok(r["reference"]); langs.add(r["iso"])
            agg["n"] += 1; den.add(en)
            ex, se, ne, pa = fl["exact"][k], fl["sentence"][k], fl["near"][k], fl["pair"][k]
            ra, rsi, rn = ts["ref_any"][k], ts["ref_same_iso"][k], ts.get("ref_near", {}).get(k, False)
            a8, r8 = ts["any8"][k], ts.get("ref_any8", {}).get(k, False)
            for name, v in (("exact", ex), ("sentence", se), ("near", ne), ("pair", pa), ("ref_any", ra), ("ref_same_iso", rsi), ("ref_near", rn), ("any8", a8), ("ref_any8", r8)): agg[name] += int(v)
            if ex: den_exp.add(en); agg["exact_ge3"] += int(le >= 3); agg["exact_ge8"] += int(le >= 8); den_exp3 |= {en} if le >= 3 else set(); den_exp8 |= {en} if le >= 8 else set()
            if pa: agg["pair_ge3"] += int(le >= 3)
            if ra: agg["ref_any_ge3"] += int(lr >= 3)
            if rsi: agg["ref_same_iso_ge3"] += int(lr >= 3)
            if a8: den_any8.add(en)
            agg["union_strict"] += int(ex or ne or pa or rsi)
            agg["union_strict_ge3"] += int((ex and le >= 3) or ne or (pa and le >= 3) or (rsi and lr >= 3))
            if ex or ne or pa or ra or rn:
                exposed.append(dict(benchmark=b, iso=r["iso"], id=r["id"], en_tokens=le, ref_tokens=lr, exact=int(ex), sentence=int(se), near=int(ne), pair=int(pa), ref_any=int(ra), ref_same_iso=int(rsi), ref_near=int(rn), any8=int(a8)))
        n = agg["n"]; pc = lambda x: round(100.0 * x / max(n, 1), 3)
        rows_out.append(dict(benchmark=b, languages=len(langs), rows=n, distinct_english=len(den), exposed_distinct_english=len(den_exp), exposed_distinct_english_ge3tok=len(den_exp3), exposed_distinct_english_ge8tok=len(den_exp8),
                             distinct_english_sharing_an_8gram=len(den_any8),
                             rows_exact=agg["exact"], rows_sentence_rule=agg["sentence"], rows_near=agg["near"], rows_pair=agg["pair"], rows_exact_ge3tok=agg["exact_ge3"], rows_exact_ge8tok=agg["exact_ge8"], rows_pair_ge3tok=agg["pair_ge3"],
                             rows_ref_any=agg["ref_any"], rows_ref_same_iso=agg["ref_same_iso"], rows_ref_near=agg["ref_near"], rows_ref_any_ge3tok=agg["ref_any_ge3"], rows_ref_same_iso_ge3tok=agg["ref_same_iso_ge3"],
                             rows_en_shares_8gram=agg["any8"], rows_ref_shares_8gram=agg["ref_any8"], rows_strict_union=agg["union_strict"], rows_strict_union_ge3tok=agg["union_strict_ge3"],
                             pct_exact=pc(agg["exact"]), pct_pair=pc(agg["pair"]), pct_ref_same_iso=pc(agg["ref_same_iso"]), pct_strict_union=pc(agg["union_strict"]), pct_strict_union_ge3tok=pc(agg["union_strict_ge3"])))
    with open(os.path.join(a.out_dir, "exposure_by_benchmark.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
    with open(os.path.join(a.out_dir, "exposed_rows.csv"), "w", newline="") as f:
        ex_rows = [e for e in exposed if not e["benchmark"].startswith("flores_plus_dev_trainhalf")]
        w = csv.DictWriter(f, fieldnames=list(exposed[0])); w.writeheader(); w.writerows(ex_rows)
    # min3tok variant of the flag files of the evaluation benchmarks
    d3 = os.path.join(a.out_dir, "min3tok"); os.makedirs(d3, exist_ok=True)
    for b in M.EVAL_BENCHMARKS:
        fl, _ = load(b); fl2 = {k: dict(v) for k, v in fl.items() if k != "meta"}
        for r in sets[b]:
            if len(N(r["english"]).split()) < 3:
                k = C.row_key(r["iso"], r["id"])
                for fn in ("exact", "sentence", "pair"): fl2[fn][k] = False
        C.write_flag_file(os.path.join(d3, f"contamination_{b}.json"), b, sets[b], fl2, {"rule": "as the main files, but exact / sentence / pair are cleared for English of fewer than 3 normalised tokens (the intake filter ignores protected strings below 3 tokens)",
                                                                                           "derived_from": "contamination_%s.json" % b})
    # attribution of the hits (evaluation benchmarks only)
    en_bench = defaultdict(set); rev_bench = defaultdict(set)
    rid = {t: i for i, t in enumerate(rev_texts)}
    for b in M.EVAL_BENCHMARKS + ["afriscience_test", "afriscience_dev"]:
        for r in sets[b]:
            i = fid.get(N(r["english"]))
            if i is not None: en_bench[i].add(b)
            j = rid.get(N(r["reference"]))
            if j is not None: rev_bench[j].add(b)
    arr = {}; tab = {}
    cnt = Counter(); samples = set(); by_kind = Counter()
    for name, gi, sid, kind, rule, side, tr in hits:
        bs = (en_bench if kind == 0 else rev_bench).get(sid)
        by_kind[(name, "translation" if tr else "other")] += 1
        if not bs: continue
        if name not in arr:
            d = os.path.join(a.work, "arrays", name); arr[name] = np.load(f"{d}/iso_id.npy", mmap_mode="r"); tab[name] = json.load(open(f"{d}/meta.json"))["iso_table"]
        iso = tab[name][int(arr[name][gi])]
        for b in bs: cnt[(b, kind, name, "translation" if tr else "other", "prompt" if side == 0 else "target", iso)] += 1
    with open(os.path.join(a.out_dir, "hits_by_dataset.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["benchmark", "kind(0=english,1=reference)", "dataset", "sample_kind", "side", "sample_iso", "hits"])
        for k, v in sorted(cnt.items()): w.writerow(list(k) + [v])
    json.dump({"hits_by_dataset_and_kind_all_strings": {f"{k[0]}|{k[1]}": v for k, v in by_kind.items()}}, open(os.path.join(a.out_dir, "summary_extra.json"), "w"), indent=1)
    for r in rows_out: print(" ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
