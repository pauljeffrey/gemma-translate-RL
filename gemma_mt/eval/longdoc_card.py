# SPDX-License-Identifier: AGPL-3.0-or-later
"""Composition tables of the benchmark's English side (dataset card, datasheet, report): source x bin x domain with licences and counts, computed from the docs.jsonl rows.

    rows = longdoc.read_jsonl("source_set/core/docs.jsonl"); print(composition_markdown(rows))

Pure functions over rows (no I/O besides what the caller passes), unit tested in tests/test_longdoc_card.py.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

BIN_ORDER = (64, 256, 512, 1024, 2048, 4096)
BIN_LABEL = {64: "b64", 256: "b256", 512: "b512", 1024: "b1024", 2048: "b2048", 4096: "b4096"}


def _table(header: Sequence[str], body: Sequence[Sequence[Any]], align_right_from: int = 1) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" if i < align_right_from else "---:" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in body]
    return "\n".join(out) + "\n"


def source_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One entry per source: counts per bin / domain, licence strings (with counts), licence classes."""
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    out = []
    for s, rs in by.items():
        lic = Counter(r["licence"] for r in rs)
        out.append({"source": s, "n": len(rs), "bins": Counter(r["bin"] for r in rs), "domains": Counter(r["domain"] for r in rs), "licences": lic, "classes": Counter(r["licence_class"] for r in rs)})
    return sorted(out, key=lambda e: (-e["n"], e["source"]))


def composition_markdown(rows: Sequence[Dict[str, Any]], *, licence_max_len: int = 70) -> str:
    """Tables: (1) source x bin with licence and share; (2) source x domain; (3) domain x bin; (4) licence classes. Totals row included."""
    n = len(rows)
    if not n:
        return "(no documents)\n"
    bins = [b for b in BIN_ORDER if any(r["bin"] == b for r in rows)]
    doms = sorted({r["domain"] for r in rows})
    summ = source_summary(rows)

    def lic_text(e: Dict[str, Any]) -> str:
        parts = [f"{k} ({v})" if len(e["licences"]) > 1 else k for k, v in e["licences"].most_common()]
        t = "; ".join(parts)
        return t if len(t) <= licence_max_len else t[: licence_max_len - 1] + "…"

    t1 = _table(["source", "licence"] + [BIN_LABEL[b] for b in bins] + ["total", "share"],
                [[e["source"], lic_text(e)] + [e["bins"].get(b, 0) for b in bins] + [e["n"], f"{100 * e['n'] / n:.1f}%"] for e in summ]
                + [["**all**", ""] + [sum(1 for r in rows if r["bin"] == b) for b in bins] + [n, "100%"]], align_right_from=2)
    t2 = _table(["source"] + doms + ["total"], [[e["source"]] + [e["domains"].get(d, 0) for d in doms] + [e["n"]] for e in summ]
                + [["**all**"] + [sum(1 for r in rows if r["domain"] == d) for d in doms] + [n]])
    dom_bin = Counter((r["domain"], r["bin"]) for r in rows)
    t3 = _table(["domain"] + [BIN_LABEL[b] for b in bins] + ["total"], [[d] + [dom_bin.get((d, b), 0) for b in bins] + [sum(dom_bin.get((d, b), 0) for b in bins)] for d in doms]
                + [["**all**"] + [sum(1 for r in rows if r["bin"] == b) for b in bins] + [n]])
    cls = Counter(r["licence_class"] for r in rows)
    t4 = _table(["licence class", "documents", "share"], [[k, v, f"{100 * v / n:.1f}%"] for k, v in cls.most_common()])
    return ("#### Source x length bin (with licence)\n\n" + t1 + "\n#### Source x domain\n\n" + t2 + "\n#### Domain x length bin\n\n" + t3 + "\n#### Licence classes\n\n" + t4)


def funnel_markdown(funnel: Dict[str, int]) -> str:
    """The selection funnel (pool -> candidates) as a two-column table, drops first."""
    keys = [k for k in funnel if not k.startswith("candidates_bin") and k != "paragraphs_removed"]
    order = ["pool"] + sorted([k for k in keys if k.startswith("drop_")], key=lambda k: -funnel[k]) + ["candidates"]
    rows = [[k, funnel[k]] for k in order if k in funnel]
    rows += [[k, funnel[k]] for k in sorted(funnel) if k.startswith("candidates_bin")]
    rows += [["paragraphs_removed (inside kept documents)", funnel.get("paragraphs_removed", 0)]] if "paragraphs_removed" in funnel else []
    return _table(["step", "documents"], rows)


def split_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter(r.get("split", "public") for r in rows)
    return dict(c)
