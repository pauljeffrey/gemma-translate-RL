#!/usr/bin/env python3
"""Assemble the contamination files of the 385M SFT text from the two scans (plain normalisation = `full`, mark-stripped normalisation = `strip`, R4 of contam_audit_12).

    python contam_audit_14c_assemble.py --base DIR/full/contamination --variant DIR/strip/contamination --out OUT --bench-root ... --extra-root ... --tree ...

Writes into OUT (flag files in the format of gemma_mt.eval.contamination: {"meta", "exact", "sentence", "near", "pair"} keyed "<iso>:<row id>", read by sft_benchmarks --contamination-dir / CONTAM_DIR):
  contamination_<benchmark>.json          MAIN: union of the two scans, English side of the benchmark row found on EITHER side of any training sample (rules: sentence / contiguous run, near >= 50% of the
                                          8-grams, pair = English + reference seen together). `exact` is the flag `contaminated` of the scorer (clean subset), `pair` the `pair_seen` flag (no_pair subset).
  strict/contamination_<benchmark>.json   SENSITIVITY: `exact` also true when the REFERENCE sentence is found in a sample of the same language (target-side exposure), `near` also for reference near duplicates.
  min3tok/ , strict_min3tok/              the same two with exact / sentence / pair cleared when the string is shorter than 3 normalised tokens (the intake filter and contam_audit_12 never match those).
  extra/                                  FLORES+ dev held-out / train half, AfriScience flag files (the train half of the dev ids is exempt by the D2 rule and serves as the positive control).
  results_by_language.csv                 benchmark x language: n, n_contaminated, n_sentence_rule, n_near, n_pair (the columns of results.csv of sft_benchmarks), plus target-side and >= 3 token columns and shares.
  results_by_benchmark.csv                benchmark totals, distinct exposed English, length classes, shares.
"""
import argparse, csv, json, os, sys
from collections import Counter, defaultdict
sys.dont_write_bytecode = True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("base", "variant", "out", "bench_root", "extra_root", "tree"): ap.add_argument("--" + k.replace("_", "-"), required=True)
    ap.add_argument("--scanner-dir", default="/workspace/eval_385m/contam_audit")
    a = ap.parse_args(); sys.path.insert(0, a.tree); sys.path.insert(0, a.scanner_dir)
    import contam_audit_14_385m_scan as M
    from gemma_mt.eval import contamination as C
    sets = M.load_rows(a); N = C.normalise
    FL = ("exact", "sentence", "near", "pair"); TS = ("ref_any", "ref_same_iso", "ref_sentence_any", "any8", "ref_near", "ref_any8")
    os.makedirs(a.out, exist_ok=True)
    per_lang, per_bench = [], []
    for b, rows in sets.items():
        ev = b in M.EVAL_BENCHMARKS
        f1 = os.path.join(a.base, f"contamination_{b}.json" if ev else os.path.join("extra", f"contamination_{b}.json"))
        f2 = os.path.join(a.variant, f"contamination_{b}.json" if ev else os.path.join("extra", f"contamination_{b}.json"))
        d1, d2 = json.load(open(f1)), json.load(open(f2))
        t1 = json.load(open(os.path.join(a.base, "extra", f"targetside_{b}.json"))); t2 = json.load(open(os.path.join(a.variant, "extra", f"targetside_{b}.json")))
        keys = list(d1["exact"])
        assert set(keys) == set(d2["exact"]), (b, "row keys differ")
        union = {fn: {k: bool(d1[fn].get(k) or d2[fn].get(k)) for k in keys} for fn in FL}
        union["near"] = {k: bool(union["near"][k] and not union["exact"][k]) for k in keys}
        tu = {fn: {k: bool(t1.get(fn, {}).get(k) or t2.get(fn, {}).get(k)) for k in keys} for fn in TS}
        only_variant = sum(1 for k in keys if d2["exact"][k] and not d1["exact"][k]); only_base = sum(1 for k in keys if d1["exact"][k] and not d2["exact"][k])
        # strict: target-side exposure counts as contaminated
        strict = {fn: dict(union[fn]) for fn in FL}
        for k in keys:
            strict["exact"][k] = bool(union["exact"][k] or tu["ref_same_iso"][k])
            strict["sentence"][k] = bool(union["sentence"][k] or (tu["ref_sentence_any"][k] and tu["ref_same_iso"][k]))
            strict["near"][k] = bool((union["near"][k] or tu["ref_near"][k]) and not strict["exact"][k])
        def cleared(fl):
            out = {fn: dict(v) for fn, v in fl.items()}
            for r in rows:
                k = C.row_key(r["iso"], r["id"]); le = len(N(r["english"]).split()); lr = len(N(r["reference"]).split())
                if le < 3:
                    for fn in ("sentence", "pair"): out[fn][k] = False
                    out["exact"][k] = bool(tu["ref_same_iso"][k] and lr >= 3) if fl is strict else False
                elif fl is strict and not union["exact"][k] and lr < 3:
                    out["exact"][k] = False
            return out
        meta_x = {"training_text": "385M SFT text: g0 (A0 packs incl. instruction packs, stage A draws) + stage-B shards s1..s6 (incl. tail replay)", "scans": "plain normalisation + mark-stripped normalisation (R4), union",
                  "english_only_in_variant": only_variant, "english_only_in_base": only_base}
        variants = {"": (union, "MAIN: English side of the row in the training text (either side of any sample)"),
                    "strict": (strict, "STRICT: main OR the reference sentence found in a sample of the same language (target-side exposure)"),
                    "min3tok": (cleared(union), "main with exact / sentence / pair cleared for English < 3 normalised tokens"),
                    "strict_min3tok": (cleared(strict), "strict with short strings (< 3 tokens) not counted")}
        for vname, (fl, rule) in variants.items():
            sub = vname if ev else "extra"
            if not ev and vname: continue
            path = os.path.join(a.out, sub, f"contamination_{b}.json") if sub else os.path.join(a.out, f"contamination_{b}.json")
            C.write_flag_file(path, b, rows, fl, {**meta_x, "rule": rule})
        # tables
        agg = defaultdict(Counter); den = set(); denx = set(); den3 = set()
        for r in rows:
            k = C.row_key(r["iso"], r["id"]); en = N(r["english"]); le = len(en.split()); lr = len(N(r["reference"]).split()); d = agg[r["iso"]]
            d["n"] += 1; den.add(en)
            for name, v in (("n_contaminated", union["exact"][k]), ("n_sentence_rule", union["sentence"][k]), ("n_near", union["near"][k]), ("n_pair", union["pair"][k]),
                            ("n_ref_same_iso", tu["ref_same_iso"][k]), ("n_ref_any", tu["ref_any"][k]), ("n_en_shares_8gram", tu["any8"][k]), ("n_strict", strict["exact"][k] or strict["near"][k] or strict["pair"][k]),
                            ("n_contaminated_ge3tok", union["exact"][k] and le >= 3), ("n_pair_ge3tok", union["pair"][k] and le >= 3), ("n_ref_same_iso_ge3tok", tu["ref_same_iso"][k] and lr >= 3)):
                d[name] += int(bool(v))
            if union["exact"][k]: denx.add(en); den3.add(en) if le >= 3 else None
        for iso, d in sorted(agg.items()):
            per_lang.append(dict(benchmark=b, iso=iso, **{k: d[k] for k in ("n", "n_contaminated", "n_sentence_rule", "n_near", "n_pair", "n_ref_same_iso", "n_ref_any", "n_en_shares_8gram", "n_strict", "n_contaminated_ge3tok", "n_pair_ge3tok", "n_ref_same_iso_ge3tok")},
                                 pct_contaminated=round(100 * d["n_contaminated"] / d["n"], 3), pct_strict=round(100 * d["n_strict"] / d["n"], 3)))
        tot = Counter()
        for d in agg.values(): tot.update(d)
        per_bench.append(dict(benchmark=b, protected=int(ev), languages=len(agg), rows=tot["n"], distinct_english=len(den), distinct_english_exposed=len(denx), distinct_english_exposed_ge3tok=len(den3),
                              **{k: tot[k] for k in ("n_contaminated", "n_sentence_rule", "n_near", "n_pair", "n_ref_same_iso", "n_ref_any", "n_en_shares_8gram", "n_strict", "n_contaminated_ge3tok", "n_pair_ge3tok", "n_ref_same_iso_ge3tok")},
                              pct_contaminated=round(100 * tot["n_contaminated"] / tot["n"], 3), pct_pair=round(100 * tot["n_pair"] / tot["n"], 3), pct_strict=round(100 * tot["n_strict"] / tot["n"], 3),
                              pct_contaminated_ge3tok=round(100 * tot["n_contaminated_ge3tok"] / tot["n"], 3), only_in_variant_scan=only_variant, only_in_base_scan=only_base))
        print(f"{b:28s} rows {tot['n']:6d} contaminated {tot['n_contaminated']:5d} (>=3tok {tot['n_contaminated_ge3tok']}) sentence {tot['n_sentence_rule']} near {tot['n_near']} pair {tot['n_pair']} ref_same_iso {tot['n_ref_same_iso']} strict {tot['n_strict']} | only-in-variant {only_variant} only-in-base {only_base}", flush=True)
    for name, data in (("results_by_language.csv", per_lang), ("results_by_benchmark.csv", per_bench)):
        with open(os.path.join(a.out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


if __name__ == "__main__":
    main()
