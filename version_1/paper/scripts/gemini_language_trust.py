#!/usr/bin/env python
"""Per-language table: Gemini-2.5-Flash FLORES chrF++ (published, n=10) vs NLLB-3.3B (our re-score, n=1012; and NLLB paper).

Reads reports/data/published_baselines.csv and reports/data/model_comparison_master.csv (laptop repo); writes
reports/data/longctx/gemini_trust_languages.csv and prints a markdown table.  Rule (see plan section 3):
TRUST = Gemini chrF++ >= NLLB-3.3B chrF++ on BOTH NLLB references; MARGINAL = >= on one only / within 3 points below
(n=10 noise); NO_GEMINI_NUMBER = language absent from the Gemini rows.  This is a screening rule, the 50-sentence pilot decides.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gemini_batch_common as C  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO, "reports", "data")


def main():
    langs = C.HUMAN_LANGS + [l for l in C.SCI_LANGS if l not in C.HUMAN_LANGS]
    gem, pub = {}, {}
    for r in csv.DictReader(open(os.path.join(DATA, "published_baselines.csv"), encoding="utf-8")):
        if r["model"] == "Gemini-2.5-Flash" and r["direction"] == "en->xx":
            gem[r["iso"]] = (float(r["chrF++"]), r["benchmark"])
        if r["model"] == "NLLB-200-3.3B" and r["direction"] == "en->xx" and r["iso"] not in pub:
            try:
                pub[r["iso"]] = float(r["chrF++"])
            except ValueError:
                pass
    mine = {}
    for r in csv.DictReader(open(os.path.join(DATA, "model_comparison_master.csv"), encoding="utf-8")):
        if r["model"].startswith("NLLB-200-3.3B (re-run") and r["subset"] == "all" and r["benchmark"].startswith("FLORES+ devtest"):
            mine[r["language"]] = float(r["chrF++"])
    rows = []
    for l in langs:
        g = gem.get(l, (None, ""))[0]
        n_my, n_pub = mine.get(l), pub.get(l)
        d_my = None if g is None or n_my is None else g - n_my
        d_pub = None if g is None or n_pub is None else g - n_pub
        if g is None:
            verdict = "NO_GEMINI_NUMBER"
        else:
            ds = [d for d in (d_my, d_pub) if d is not None]
            verdict = "TRUST" if all(d >= 0 for d in ds) else ("MARGINAL" if min(ds) > -3.0 else "NOT_TRUSTED")
        rows.append(dict(iso=l, name=C.LANG_NAMES[l], in_human=l in C.HUMAN_LANGS, in_sci=l in C.SCI_LANGS,
                         gemini25flash_chrfpp_n10=g, nllb33b_rescored_chrfpp_n1012=n_my, nllb33b_paper_chrfpp=n_pub,
                         diff_vs_rescored=d_my, diff_vs_paper=d_pub, verdict=verdict))
    out = os.path.join(DATA, "longctx")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "gemini_trust_languages.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def f1(x):
        return "n/a" if x is None else "%.1f" % x

    def fd(x):
        return "n/a" if x is None else "%+.1f" % x
    print("| lang | benchmark sets | Gemini-2.5-Flash (n=10) | NLLB-3.3B re-scored (n=1012) | NLLB-3.3B paper | diff vs re-scored | diff vs paper | verdict |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        sets = "+".join(s for s, b in (("human", r["in_human"]), ("sci", r["in_sci"])) if b)
        print("| %s %s | %s | %s | %s | %s | %s | %s | %s |" % (r["iso"], r["name"], sets, f1(r["gemini25flash_chrfpp_n10"]),
              f1(r["nllb33b_rescored_chrfpp_n1012"]), f1(r["nllb33b_paper_chrfpp"]), fd(r["diff_vs_rescored"]), fd(r["diff_vs_paper"]), r["verdict"]))


if __name__ == "__main__":
    main()
