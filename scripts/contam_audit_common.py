"""Shared helpers of the 2026-10-26 contamination audit (agent C1). Read-only: no Hub writes, no secrets printed.

The Hub token is read from the repo-root .env (key HF_API_KEY, the READ token) and is never printed or logged.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Dict, Optional

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARCHIVE = os.path.join(REPO, "research_archive_2026-10-23")
FLAGS = os.path.join(ARCHIVE, "box", "other", "sft_build", "bench", "test_sets", "_contamination")
FLAGS_DEV = os.path.join(ARCHIVE, "box", "other", "sft_build", "bench", "contamination_dev.json")
OUT = os.path.join(REPO, "reports", "data", "contamination_audit_2026-10-26")

# Benchmark text fetched by the scripts (FLORES+, SSA-MTE, ...) is cached OUTSIDE the repository (it has its own licences and must never be committed).
CACHE = os.environ.get("CONTAM_AUDIT_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "contam_audit"))

_NON_ALNUM = re.compile(r"[\W_]+", re.UNICODE)


def normalise(text: str) -> str:
    """The normalisation of gemma_mt/gemma_mt/eval/contamination.py: NFKC, case-fold, runs of non-alphanumerics -> one space."""
    return _NON_ALNUM.sub(" ", unicodedata.normalize("NFKC", text or "").casefold()).strip()


def hf_token(key: str = "HF_API_KEY") -> Optional[str]:
    env: Dict[str, str] = {}
    p = os.path.join(REPO, ".env")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k] = v.strip().strip('"').strip("'")
    return env.get(key) or os.environ.get(key)


def load_flags(benchmark: str) -> dict:
    p = FLAGS_DEV if benchmark == "dev" else os.path.join(FLAGS, f"contamination_{benchmark}.json")
    return json.load(open(p, encoding="utf-8"))
