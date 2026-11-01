"""Shared helpers for the paper scripts: paths, plotting style, markdown-table parsing.

All figures and tables of the paper are generated from files in this repository (reports/, reports/data/, paper/data/):
    python paper/scripts/make_figures.py      # figures -> paper/figures/*.png|pdf
    python paper/scripts/make_tables.py       # tables  -> paper/tables/*.md
Both are safe to re-run; the final2 curves/tables read paper/data/final2_metrics.jsonl (refresh it with scripts/refresh_final2.sh).
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(PAPER, ".."))
REPORTS = os.path.join(REPO, "reports")
RDATA = os.path.join(REPORTS, "data")
DATA = os.path.join(PAPER, "data")
FIG = os.path.join(PAPER, "figures")
TAB = os.path.join(PAPER, "tables")
for _d in (FIG, TAB, DATA):
    os.makedirs(_d, exist_ok=True)

# Okabe-Ito colour-blind-safe palette
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", verm="#D55E00", pink="#CC79A7", grey="#7F7F7F")
GROUP_COLOR = OrderedDict([("qe_validated", OI["blue"]), ("blend", OI["orange"]), ("consensus_only", OI["verm"]), ("sft_only", OI["grey"])])
GROUP_LABEL = {"qe_validated": "QE-validated", "blend": "QE-blend", "consensus_only": "consensus-only", "sft_only": "SFT-only (no RL)"}


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "lines.linewidth": 1.3,
        "font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    print("figure", name)


# ---------------------------------------------------------------------------------------------- markdown tables
def md_tables(path):
    """Parse every pipe table of a markdown file.  Returns a list of dicts
    {heading, header[list], rows[list[list[str]]], line} in file order; `heading` is the closest preceding '#'-heading
    (or bold/plain line immediately above the table for files that label tables with plain text)."""
    lines = open(path, encoding="utf-8").read().split("\n")
    out, heading, i = [], "", 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("#"):
            heading = ln.lstrip("# ").strip()
        if ln.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|?\s*$", lines[i + 1]):
            hdr = [c.strip() for c in ln.strip().strip("|").split("|")]
            rows, j = [], i + 2
            while j < len(lines) and lines[j].startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            label = lines[i - 1].strip() if i > 0 and lines[i - 1].strip() and not lines[i - 1].startswith("|") else ""
            out.append(dict(heading=heading, label=label, header=hdr, rows=rows, line=i + 1))
            i = j
            continue
        i += 1
    return out


NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def num(s, default=None):
    """First number in a cell like '47.6 [46.2, 49.0]' or '98%' or '+0.8 [+0.3, +1.4]*'."""
    m = NUM.search(s.replace("−", "-"))
    return float(m.group()) if m else default


def nums(s):
    return [float(x) for x in NUM.findall(s.replace("−", "-"))]


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return rows


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


BINS = ["b64", "b256", "b512", "b1024", "b2048", "b4096"]
