#!/usr/bin/env python3
"""Attach the contamination flags (contamination_<benchmark>.json of `contam_audit_14_385m_scan.py finalize`) to an existing predictions.jsonl, so that
`eval_385m_score.py` can report the `clean` / `clean_strict` / `no_pair` subsets WITHOUT regenerating anything (the generation run only needs the flag files when it is
started with --contamination-dir / CONTAM_DIR; a run that used --allow-missing-contamination has rows without flags).

    python scripts/contam_audit_14_attach.py --predictions OUT/predictions.jsonl --contam-dir DIR --out OUT/predictions_flagged.jsonl

Sets `contaminated` (exact), `contaminated_sentence`, `near_dup`, `pair_seen` exactly as `gemma_mt.eval.contamination.apply_flags` does inside sft_benchmarks. Rows of a benchmark without
a flag file, or missing from it, get None (never silently clean). Prints the per benchmark counts (the n_contaminated / n_sentence_rule / n_near / n_pair columns of results.csv).
"""
import argparse, json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True); ap.add_argument("--contam-dir", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from gemma_mt.eval import contamination as C
    from gemma_mt.eval import benchmarks as B
    rows = [json.loads(l) for l in open(a.predictions, encoding="utf-8") if l.strip()]
    files = C.default_flag_files(a.contam_dir, sorted({r["benchmark"] for r in rows}))
    missing = C.apply_flags(rows, files)
    tmp = a.out + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, a.out)
    cnt = Counter()
    for r in rows:
        b = r["benchmark"]; cnt[(b, "n")] += 1
        for k, f in (("contaminated", "n_contaminated"), ("contaminated_sentence", "n_sentence_rule"), ("near_dup", "n_near"), ("pair_seen", "n_pair")):
            cnt[(b, f)] += int(bool(r.get(k)))
    print(f"{len(rows)} rows -> {a.out}; flag files: {sorted(files)}; rows without a flag: {missing}")
    for b in sorted({k[0] for k in cnt}):
        print(f"  {b:12s} n={cnt[(b, 'n')]:6d} n_contaminated={cnt[(b, 'n_contaminated')]} n_sentence_rule={cnt[(b, 'n_sentence_rule')]} n_near={cnt[(b, 'n_near')]} n_pair={cnt[(b, 'n_pair')]}")


if __name__ == "__main__":
    main()
