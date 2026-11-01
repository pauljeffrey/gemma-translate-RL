# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared helpers of the three E1 pairwise-comparison figure scripts (gemma_mt/scripts/pairwise_fig_*.py).

They read the CSVs of reports/data/pairwise_benchmark_comparison/ (produced on the box by pairwise_stats.py, copied to the repository) and write
paper/figures/figpw1..3_*.png|pdf in the style of paper/scripts/ (Okabe-Ito palette, `setup_matplotlib`, `save`)."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "paper", "scripts"))
from common import OI, GROUP_COLOR, GROUP_LABEL, setup_matplotlib, save  # noqa: E402,F401
import common as _paper_common  # noqa: E402

DATA = os.environ.get("PAIRWISE_DATA", os.path.join(REPO, "reports", "data", "pairwise_benchmark_comparison"))
if os.environ.get("PAIRWISE_FIG_DIR"):          # lets the scripts be tested without touching paper/figures
    _paper_common.FIG = os.environ["PAIRWISE_FIG_DIR"]
    os.makedirs(_paper_common.FIG, exist_ok=True)

METRIC_LABEL = {"chrf_sub": "chrF++", "chrf": "chrF++", "afri_ref": "AfriCOMET\n(reference)", "ssa_ref": "SSA-COMET\n(reference)",
                "ens_ref": "majority of the\n3 reference-based", "afri_qe": "AfriCOMET-QE\n(RL reward)", "ssa_qe": "SSA-COMET\nQE"}
REF_METRICS = ["chrf_sub", "afri_ref", "ssa_ref"]
QE_METRICS = ["afri_qe", "ssa_qe"]
WIN_C, TIE_C, LOSS_C = OI["green"], "#D9D9D9", OI["verm"]


def read(name, sub="main"):
    import pandas as pd
    p = os.path.join(DATA, sub, name)
    return pd.read_csv(p)
