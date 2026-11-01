# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compare the `results.json` of two runs of `scripts/eval_sft_benchmarks.sh` (e.g. the SFT model and an RL checkpoint) language by language.

    python scripts/compare_eval_results.py OLD/results.json NEW/results.json [--system sft] [--old-system sft] [--subset clean] [--metric chrf]

`--system` / `--old-system` say which system of each file is compared (the model under test is always called `sft` by the driver, the untouched
base is `base`). Subsets: all, clean, no_pair, clean_sentence, clean_strict. Metrics: chrf, spbleu, bleu, comet, comet_qe, exact. Prints one table per
benchmark (rows = languages present in both files, with the number of rows of each side) and the macro over those languages; languages with fewer
than --min-rows rows in the subset on either side are listed but left out of the macro."""
import argparse
import json
import sys


def load(path, system):
    with open(path, encoding="utf-8") as f:
        res = json.load(f)
    return res["systems"][system]["benchmarks"]


def pick(rec, subset):
    return rec if subset == "all" else (rec.get(subset) or {})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--system", default="sft", help="system of NEW to compare")
    ap.add_argument("--old-system", default=None, help="system of OLD (default: --system)")
    ap.add_argument("--subset", default="clean")
    ap.add_argument("--metric", default="chrf")
    ap.add_argument("--min-rows", type=int, default=20)
    a = ap.parse_args(argv)
    old, new = load(a.old, a.old_system or a.system), load(a.new, a.system)
    for b in sorted(set(old) & set(new)):
        rows, ds = [], []
        for iso in sorted(set(old[b]["languages"]) & set(new[b]["languages"])):
            o, n = pick(old[b]["languages"][iso], a.subset), pick(new[b]["languages"][iso], a.subset)
            vo, vn = o.get(a.metric), n.get(a.metric)
            if vo is None or vn is None:
                continue
            ok = o.get("n", 0) >= a.min_rows and n.get("n", 0) >= a.min_rows
            rows.append((iso, o.get("n", 0), n.get("n", 0), vo, vn, ok))
            if ok:
                ds.append((vo, vn))
        if not rows:
            continue
        print(f"\n## {b}: {a.metric} on `{a.subset}` (old {a.old_system or a.system} -> new {a.system})\n")
        print("| language | n old | n new | old | new | delta |\n|---|---|---|---|---|---|")
        for iso, no, nn, vo, vn, ok in rows:
            print(f"| {iso}{'' if ok else ' (few rows)'} | {no} | {nn} | {vo:.3f} | {vn:.3f} | {vn - vo:+.3f} |")
        if ds:
            mo, mn = sum(x for x, _ in ds) / len(ds), sum(y for _, y in ds) / len(ds)
            print(f"| **macro ({len(ds)} languages)** | | | {mo:.3f} | {mn:.3f} | {mn - mo:+.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
