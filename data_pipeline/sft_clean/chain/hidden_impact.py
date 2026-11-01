#!/usr/bin/env python3
"""Hidden-NC filter: impact table BEFORE anything is dropped, from the build tables only (laptop, no data download). The exact per-row numbers (rules R1 licence text, R2 NC-origin pairs) come from
`nc_hidden.py report` on the instance (step 22); this script gives (a) what dropping each unsplittable WHOLE source would cost per language (pairs, estimated Gemma target tokens, language left
below 2,000 pairs?) and (b) the per-row overlaps MEASURED on the laptop (AfroLingu-MT vs MAFAND-MT, eng_twi vs pristine-twi) scaled to the output.

    python hidden_impact.py --root <repo> --out reports/data/sft_clean_2026-10-28/bins_commercial/hidden_nc [--afl-overlap /tmp/nchid/afrolingu_mafand_overlap.csv]
Language totals = unique pairs of data/ + data_devhalf/ + specialist/ (qe_cpu/queue_meta.json); AfroLingu pairs are counted once (the specialist copy repeats main rows). Token estimate = pairs x the language's mean Gemma target tokens per pair (bins/per_language_ungated.csv); the real token numbers come from the instance report.
"""
import argparse, collections, csv, json, os, re

WHOLE = [("UBC-NLP/AfroLingu-MT", re.compile(r"UBC-NLP/AfroLingu-MT")), ("African-Languages-Lab/multi-open", re.compile(r"African-Languages-Lab/multi-open")),
         ("KwekuAinoo/eng_twi", re.compile(r"KwekuAinoo/eng_twi")), ("michsethowusu *_mt560 (OPUS mixture)", re.compile(r"michsethowusu/english-.*_mt560"))]
MIN = 2000


def rd(p):
    return list(csv.DictReader(open(p)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/path/to/repo"); ap.add_argument("--out", required=True); ap.add_argument("--afl-overlap", default=None)
    a = ap.parse_args(); D = os.path.join(a.root, "reports/data/sft_clean_2026-10-28"); os.makedirs(a.out, exist_ok=True)
    tpp = {r["iso"]: float(r["target_tokens_gemma"]) / max(float(r["pairs"]), 1) for r in rd(D + "/bins/per_language_ungated.csv")}
    lang_pairs = {k: int(v) for k, v in json.load(open(D + "/qe_cpu/queue_meta.json"))["pairs_by_language"].items()}      # unique pairs of data/ + data_devhalf/ + specialist/ per language
    cell = collections.defaultdict(int)
    for r in rd(D + "/per_language_source.csv"):
        for name, rx in WHOLE:
            if rx.fullmatch(r["source_repo"]): cell[(r["iso"], name)] += int(r["pairs_after"])
    rows = []
    for (iso, name), n in cell.items():
        rem = lang_pairs.get(iso, n) - n
        rows.append([iso, name, n, int(n * tpp.get(iso, 25)), round(100 * n / max(lang_pairs.get(iso, n), 1), 2), lang_pairs.get(iso, n), rem, int(rem < MIN), int(lang_pairs.get(iso, n) >= MIN and rem < MIN)])
    rows.sort(key=lambda r: (r[1], -r[2]))
    with open(a.out + "/hidden_nc_whole_source_by_language_ESTIMATE.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "source", "pairs", "est_gemma_target_tokens", "share_of_language_pairs_pct", "language_pairs_total", "language_pairs_if_source_dropped", "below_2000_if_dropped", "falls_below_2000_only_because_of_this_source"])
        w.writerows(rows)
    tot = collections.defaultdict(lambda: [0, 0, set(), set()])
    for r in rows:
        t = tot[r[1]]; t[0] += r[2]; t[1] += r[3]; t[2].add(r[0])
        if r[8]: t[3].add(r[0])
    with open(a.out + "/hidden_nc_whole_source_totals_ESTIMATE.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["source", "pairs", "est_gemma_target_tokens", "languages_with_rows", "languages_falling_below_2000_if_dropped"])
        for name, t in tot.items(): w.writerow([name, t[0], t[1], len(t[2]), " ".join(sorted(t[3]))])
    allp = sum(r[2] for r in rows); allt = sum(r[3] for r in rows)
    # combined: every whole source dropped at once -> languages below 2,000
    comb = collections.Counter()
    for (iso, name), n in cell.items(): comb[iso] += n
    below_all = sorted((iso, lang_pairs.get(iso, n) - n) for iso, n in comb.items() if lang_pairs.get(iso, n) - n < MIN and lang_pairs.get(iso, n) >= MIN)
    with open(a.out + "/hidden_nc_all_whole_sources_dropped_ESTIMATE.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "pairs_total", "pairs_in_whole_sources", "pairs_left", "below_2000"])
        for iso, n in sorted(comb.items(), key=lambda kv: -kv[1] / max(lang_pairs.get(kv[0], kv[1]), 1)): w.writerow([iso, lang_pairs.get(iso, n), n, lang_pairs.get(iso, n) - n, int(lang_pairs.get(iso, n) - n < MIN)])
    print(f"all four whole sources: {allp:,} pairs, ~{allt:,} Gemma target tokens; languages that would fall below {MIN}: {below_all}")
    for name, t in tot.items(): print(f"  {name}: {t[0]:,} pairs ~{t[1]:,} tokens, {len(t[2])} languages, below {MIN} only because of it: {sorted(t[3])}")
    # per-row overlaps measured on the laptop, scaled to the AfroLingu pairs of the output
    if a.afl_overlap and os.path.exists(a.afl_overlap):
        with open(a.out + "/hidden_nc_R2_measured_afrolingu_vs_mafand.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["iso", "afrolingu_pairs_in_output", "share_pair_equal_to_MAFAND_pair_pct", "est_pairs_R2", "est_gemma_tokens_R2", "raw_rows_checked", "raw_pair_matches"])
            for r in rd(a.afl_overlap):
                n = cell.get((r["iso"], "UBC-NLP/AfroLingu-MT"), 0)
                if not n and not int(r["rows_equal_to_a_MAFAND_pair"]): continue
                sh = float(r["share_pair_pct"]); est = int(n * sh / 100)
                w.writerow([r["iso"], n, sh, est, int(est * tpp.get(r["iso"], 25)), r["afrolingu_en_x_rows_raw"], r["rows_equal_to_a_MAFAND_pair"]])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
