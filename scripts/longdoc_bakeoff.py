#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Translator bake-off on human sentence sets (D-44). Thin wrapper: see gemma_mt/eval/longdoc_bakeoff.py (`--help`)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma_mt.eval.longdoc_bakeoff import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
