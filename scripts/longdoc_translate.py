#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pseudo-reference runner (D-44): English source set -> NLLB-200 / MADLAD-400 sentence translations, resumable, with progress.log. Library:
gemma_mt/eval/longdoc_runner.py (`--help`). The older single-process script is gemma_mt/eval/longdoc_pseudo.py (still importable)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemma_mt.eval.longdoc_runner import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
