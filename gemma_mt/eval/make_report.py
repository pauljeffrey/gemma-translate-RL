# SPDX-License-Identifier: AGPL-3.0-or-later
"""Markdown comparison report from several results.json files (and optional published numbers in a CSV).

    python -m gemma_mt.eval.make_report \\
        --run base=results/base/results.json --run sft=results/sft/results.json --run rl=results/rl/results.json \\
        --baseline-csv published.csv --reference base --out report.md

Baseline CSV (published numbers or other systems, one row per system x language):
    system,iso,benchmark,chrf,spbleu,bleu,comet,comet_qe
    NLLB-200-600M,yor,flores_plus,31.2,,,0.71,
    Google Translate,hau,,,,,0.78,
`benchmark` may be empty (the row then applies to every benchmark); any metric cell may be empty. `iso` is our tag
or a common alias; MACRO rows are ignored (the report computes its own).

For every benchmark the report has (1) a macro table over the languages EVERY system has a value for, with deltas
against the reference system, and (2) one per-language table per metric with a delta column per non-reference
system. Missing values are shown as "-". Higher is better for every metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

from gemma_mt.eval.metrics import ALL_METRICS
from gemma_mt.languages import ALL_ISOS, LANGUAGES, canonical_iso

Table = Dict[Tuple[str, str], Dict[str, float]]      # (benchmark, iso) -> metric -> value
ANY = "*"


def load_results(path: str) -> Tuple[str, Table]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    t: Table = {}
    for b, bd in d["benchmarks"].items():
        for iso, rec in bd["languages"].items():
            t[(b, iso)] = {m: float(rec[m]) for m in ALL_METRICS if m in rec and rec[m] not in (None, "")}
    return d.get("meta", {}).get("name", ""), t


def load_baseline_csv(path: str) -> Dict[str, Table]:
    out: Dict[str, Table] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            system = (row.get("system") or "").strip()
            iso = canonical_iso(row.get("iso") or "")
            if not system or iso is None:
                continue
            bench = (row.get("benchmark") or "").strip() or ANY
            vals = {}
            for m in ALL_METRICS:
                v = (row.get(m) or "").strip()
                if v:
                    vals[m] = float(v)
            if vals:
                out.setdefault(system, {})[(bench, iso)] = vals
    return out


def _get(table: Table, bench: str, iso: str) -> Dict[str, float]:
    return table.get((bench, iso)) or table.get((ANY, iso)) or {}


def _fmt(x: Optional[float], nd: int = 2) -> str:
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def _delta(x: Optional[float], ref: Optional[float], nd: int = 2) -> str:
    if x is None or ref is None or x != x or ref != ref:
        return "-"
    return f"{x - ref:+.{nd}f}"


def _nd(metric: str) -> int:
    return 3 if metric.startswith("comet") else 2


def render_report(systems: "OrderedDict[str, Table]", reference: Optional[str] = None, title: str = "Translation results",
                  metrics: Sequence[str] = ALL_METRICS) -> str:
    names = list(systems)
    if not names:
        raise ValueError("no systems")
    ref = reference or names[0]
    if ref not in systems:
        raise ValueError(f"reference {ref!r} is not among {names}")
    benches = sorted({b for t in systems.values() for (b, _) in t if b != ANY})
    present = [m for m in metrics if any(m in v for t in systems.values() for v in t.values())]
    lines = [f"# {title}", "", f"Reference system for deltas: **{ref}**. Higher is better. chrF++ / spBLEU / BLEU are on a "
             "0-100 scale, AfriCOMET on 0-1.", ""]
    for b in benches:
        isos = [i for i in ALL_ISOS if any(_get(t, b, i) for t in systems.values())]
        if not isos:
            continue
        lines += [f"## {b}", ""]
        # ---- macro over the languages every system has (per metric)
        lines += ["### Macro average (languages common to all systems, per metric)", ""]
        hdr = ["system"] + [f"{m} (n)" for m in present] + [f"Δ{m}" for m in present]
        lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
        macro: Dict[str, Dict[str, Optional[float]]] = {}
        for s in names:
            macro[s] = {}
        counts: Dict[str, int] = {}
        for m in present:
            common = [i for i in isos if all(m in _get(systems[s], b, i) for s in names)]
            counts[m] = len(common)
            for s in names:
                vals = [_get(systems[s], b, i)[m] for i in common]
                macro[s][m] = sum(vals) / len(vals) if vals else None
        for s in names:
            cells = [f"**{s}**" if s == ref else s]
            cells += [f"{_fmt(macro[s][m], _nd(m))} ({counts[m]})" for m in present]
            cells += ["-" if s == ref else _delta(macro[s][m], macro[ref][m], _nd(m)) for m in present]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        # ---- per language, per metric
        others = [s for s in names if s != ref]
        for m in present:
            have = [i for i in isos if any(m in _get(systems[s], b, i) for s in names)]
            if not have:
                continue
            lines += [f"### {b}: {m} per language", ""]
            hdr = ["language"] + names + [f"Δ {s}" for s in others]
            lines += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
            for i in have:
                vals = {s: _get(systems[s], b, i).get(m) for s in names}
                cells = [f"{LANGUAGES[i].name} (`{i}`)"] + [_fmt(vals[s], _nd(m)) for s in names]
                cells += [_delta(vals[s], vals[ref], _nd(m)) for s in others]
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="label=path/to/results.json (repeatable)")
    ap.add_argument("--baseline-csv", action="append", default=[])
    ap.add_argument("--reference", default=None)
    ap.add_argument("--title", default="Translation results")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    systems: "OrderedDict[str, Table]" = OrderedDict()
    for spec in a.run:
        label, _, path = spec.partition("=")
        if not path:
            label, path = "", spec
        name, table = load_results(path)
        systems[label or name or path] = table
    for p in a.baseline_csv:
        for system, table in load_baseline_csv(p).items():
            systems[system] = table
    md = render_report(systems, a.reference, a.title)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {a.out} ({len(systems)} systems)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
