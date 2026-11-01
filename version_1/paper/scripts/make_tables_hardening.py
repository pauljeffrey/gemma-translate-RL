#!/usr/bin/env python3
"""P12 (2026-10-25): tables of Section 7.9 ("What the extra hardening did and did not do") from the H1 evidence in
reports/data/hardening_evidence_2026-10-25/ (reports/hardening_evidence_2026-10-25.md). Writes paper/tables/t_hardening_sentence.md and
t_hardening_decomp.md. Called by make_tables.py; can be run alone:
    python paper/scripts/make_tables_hardening.py
All values are read from the CSVs (percent or percentage points, 2 decimals); no number is typed by hand."""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, ".."))
D = os.path.join(PAPER, "..", "reports", "data", "hardening_evidence_2026-10-25")
TAB = os.path.join(PAPER, "tables")


def rd(name):
    return list(csv.DictReader(open(os.path.join(D, name), encoding="utf-8")))


def pct(x, d=2):
    return f"{100 * float(x):.{d}f}"


def pp(x, d=2):
    return f"{100 * float(x):+.{d}f}"


def ci(lo, hi, d=2):
    return f"[{100 * float(lo):+.{d}f}, {100 * float(hi):+.{d}f}]"


def cic(lo, hi, d=2):
    return f"[{100 * float(lo):+.{d}f},{100 * float(hi):+.{d}f}]"


def write_md(name, header, rows, align):
    sep = "|" + "|".join({"l": ":--", "r": "--:"}[a] for a in align) + "|"
    out = ["| " + " | ".join(header) + " |", sep] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    open(os.path.join(TAB, name + ".md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("table", name, len(rows))


# ----------------------------------------------------------------------------------------------- sentence level
SUM = rd("summary_final2_vs_rl_all.csv")
IND = rd("sentence_indicators_by_system.csv")


def paired(scheme, group, bench, ind):
    for r in SUM:
        if r["scheme"] == scheme and r["group"] == group and r["bench"] == bench and r["indicator"] == ind:
            return r
    raise KeyError((scheme, group, bench, ind))


def human(ind, scheme="all_languages", group="all", bench="ALL"):
    for r in IND:
        if r["system"] == "human reference" and r["scheme"] == scheme and r["group"] == group and r["bench"] == bench:
            return pct(r[ind + "__macro"])
    raise KeyError(ind)


def sentence_rows():
    spec = [("unfinished", "no end token in the budget"), ("degenerate", "degenerate output (v1 hard class)"), ("lid_wrong", "wrong language (GlotLID)"),
            ("jacc_copy", "copy of the source (Jaccard $\\ge$ 0.9)"), ("copy_flag", "substring copy of the source"), ("num_loss", "numbers lost"),
            ("name_loss", "names lost"), ("ent2_pen", "graded entity loss"), ("func_flag", "stop-word profile flag"), ("pad_flag", "padding flag"),
            ("rep_flag", "repetition flag"), ("pen_total", "penalty the final reward would charge (units of 0.01 reward)")]
    rows = []
    for ind, label in spec:
        r = paired("all_languages", "all", "ALL", ind)
        d = float(r["diff_final2_minus_rl"])
        lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
        note = " (level)" if lo <= 0 <= hi or r["verdict"] == "equal" else ""
        rows.append([label, pct(r["rl_final"]), pct(r["final2"]), f"{pp(d)} {ci(lo, hi)}{note}", human(ind)])
    # where the difference sits
    blocks = [("tier", "weak", "weak", "padding / repetition, 17 weak languages"), ("content_mode", "consensus", "consensus-only", "padding / repetition, 11 consensus-only languages"),
              ("tier", "strong", "strong", "padding / repetition, 17 strong languages")]
    for scheme, grp, hgrp, label in blocks:
        a, b = paired(scheme, grp, "ALL", "pad_flag"), paired(scheme, grp, "ALL", "rep_flag")
        rows.append([label, f"{pct(a['rl_final'])} / {pct(b['rl_final'])}", f"{pct(a['final2'])} / {pct(b['final2'])}",
                     f"{pp(a['diff_final2_minus_rl'])} {ci(a['ci_lo'], a['ci_hi'])} / {pp(b['diff_final2_minus_rl'])} {ci(b['ci_lo'], b['ci_hi'])}", human("pad_flag", "tier", hgrp) + " / " + human("rep_flag", "tier", hgrp)])
    a, b = paired("all_languages", "all", "glot_story", "pad_flag"), paired("all_languages", "all", "glot_story", "rep_flag")
    rows.append(["padding / repetition, GlotStoryBook only (8 languages)", f"{pct(a['rl_final'])} / {pct(b['rl_final'])}", f"{pct(a['final2'])} / {pct(b['final2'])}",
                 f"{pp(a['diff_final2_minus_rl'])} {ci(a['ci_lo'], a['ci_hi'])} / {pp(b['diff_final2_minus_rl'])} {ci(b['ci_lo'], b['ci_hi'])}", "n/a"])
    n = paired("all_languages", "all", "ALL", "unfinished")
    return rows, n["n_lang"], n["n_rows"]


# ----------------------------------------------------------------------------------------------- decomposition
DEC = rd("summary_decomposition_rl_to_final2.csv")
LD = rd("longdoc_lc1_paired.csv")
LDS = rd("longdoc_lc1_by_system.csv")


def dec(ind, scheme="all_languages", group="all", bench="flores_plus"):
    for r in DEC:
        if r["bench"] == bench and r["scheme"] == scheme and r["group"] == group and r["indicator"] == ind:
            return r
    raise KeyError(ind)


def ld(a, b, scope, ind, mode="whole"):
    for r in LD:
        if r["a"] == a and r["b"] == b and r["scope"] == scope and r["indicator"] == ind and r["mode"] == mode:
            return r
    raise KeyError((a, b, scope, ind))


def fmt_dec(r, a, b):
    return f"{pp(r[a])} {ci(r[a + '_lo'], r[a + '_hi'])}", f"{pp(r[b])} {ci(r[b + '_lo'], r[b + '_hi'])}"


def decomp_rows():
    rows = []
    for ind, label in [("pad_flag", "padding flag"), ("rep_flag", "repetition flag"), ("name_loss", "names lost"), ("copy_flag", "substring copy of the source"),
                       ("pen_total", "penalty the final reward would charge (0.01 units)")]:
        r = dec(ind)
        x, y = fmt_dec(r, "rl_to_a2w", "a2w_to_final2")
        rows.append([f"{label}, FLORES+ sentences (44 languages)", x, y, f"{pp(r['rl_to_final2'])} {ci(r['rl_to_final2_lo'], r['rl_to_final2_hi'])}"])
    for ind, label in [("pad_flag", "padding flag"), ("rep_flag", "repetition flag")]:
        r = dec(ind, "tier", "weak")
        x, y = fmt_dec(r, "rl_to_a2w", "a2w_to_final2")
        rows.append([f"{label}, FLORES+ sentences, weak languages (13)", x, y, f"{pp(r['rl_to_final2'])} {ci(r['rl_to_final2_lo'], r['rl_to_final2_hi'])}"])
    for scope, sl in [(">= ~2,048 tokens", "about 2,048 tokens"), ("~1,024 tokens", "about 1,024 tokens")]:
        for ind, label in [("unfinished", "unfinished"), ("rep_severe", "severe repetition")]:
            a, b, t = ld("rl", "a2w", scope, ind), ld("a2w", "final2", scope, ind), ld("rl", "final2", scope, ind)
            rows.append([f"whole documents of {sl}, {label} (12 languages)", f"{pp(a['diff_b_minus_a'], 1)} {ci(a['ci_lo'], a['ci_hi'], 1)}",
                         f"{pp(b['diff_b_minus_a'], 1)} {ci(b['ci_lo'], b['ci_hi'], 1)}", f"{pp(t['diff_b_minus_a'], 1)} {ci(t['ci_lo'], t['ci_hi'], 1)}"])
    r = ld("live", "final2", ">= ~2,048 tokens", "rep_severe")
    return rows, r


def main():
    rows, nl, nr = sentence_rows()
    write_md("t_hardening_sentence", ["indicator (macro % of rows)", "RL-final", "final run (step 401)", "difference, pp [95% CI]", "human references"], rows, ["l", "r", "r", "r", "r"])
    drows, live = decomp_rows()
    write_md("t_hardening_decomp", ["indicator (difference in percentage points [95% CI])", "RL-final to A2W (science stage, no RL)", "A2W to final run (hardened RL run)", "RL-final to final run (total)"],
             drows, ["l", "r", "r", "r"])
    print("step 250 -> 401, severe repetition at about 2,048 tokens (whole document):", pp(live["diff_b_minus_a"], 1), ci(live["ci_lo"], live["ci_hi"], 1))
    # the short paper quotes these numbers inline (no table, page limit): copy 0.08%, 40.6 / 29.3 pp, +6.9 pp [2.8, 11.3], names +1.6 pp: see t_hardening_sentence.md / t_hardening_decomp.md


if __name__ == "__main__":
    main()
