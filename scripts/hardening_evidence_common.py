# SPDX-License-Identifier: AGPL-3.0-or-later
"""H1 (2026-10-25): shared helpers of the "what did the reward hardening protect against?" analysis (scripts hardening_evidence_*.py).

CPU only, no GPU, no API spend, nothing pushed. Inputs are the STORED per-row outputs of SFT-final, RL-final, final2 (step 401), the step-250 checkpoint of the final run and
the science-stage start A2W alpha 0.7 (research archive of 2026-10-23, the extracted `box_bulk/lc1_e2_p5_2026-10-24.tar.gz`, the private Hub dataset
`<HF_ORG_LEGACY>/SabiYarn-MT-benchmark-translations`). The detectors are the project's own (`gemma_mt/rl/anchors*.py`, `textstats.py`, `guards.py`, `segments.py`, `longdoc.py`,
`lid.py`) with the calibration the final run used (`calibration/anchors_v3.json`, `longdoc_v2.json`, `shaping_flores_dev_v8.json`).

Environment variables (all optional; the defaults are the laptop layout):
    HE_REPO      repository root (default: two levels above this file)
    HE_ARCHIVE   research archive (default $HE_REPO/research_archive_2026-10-23)
    HE_WORK      scratch directory for big intermediate files (default /tmp/hardening_evidence_work)
    HE_OUT       output directory of CSVs / figures (default $HE_REPO/reports/data/hardening_evidence_2026-10-25)
    HE_GLOTLID   path of GlotLID model_v3.bin (default: the local Hugging Face cache)
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import sys
import tarfile
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("HE_REPO") or os.path.abspath(os.path.join(HERE, "..", ".."))
ARCHIVE = os.environ.get("HE_ARCHIVE") or os.path.join(REPO, "research_archive_2026-10-23")
WORK = os.environ.get("HE_WORK") or "/tmp/hardening_evidence_work"
OUT = os.environ.get("HE_OUT") or os.path.join(REPO, "reports", "data", "hardening_evidence_2026-10-25")
GM = os.path.join(REPO, "gemma_mt")
CAL = os.path.join(GM, "calibration")
TARBALL = os.path.join(ARCHIVE, "box_bulk", "lc1_e2_p5_2026-10-24.tar.gz")
TAR_DIR = os.path.join(WORK, "tar")
HUB_DIR = os.path.join(WORK, "hubds")
HUB_REPO = (os.environ['HF_ORG_LEGACY'] + "/SabiYarn-MT-benchmark-translations")
JOBS = os.path.join(ARCHIVE, "box", "jobs")

if GM not in sys.path:
    sys.path.insert(0, GM)

os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def nfc(s: Optional[str]) -> str:
    return unicodedata.normalize("NFC", s or "")


# ------------------------------------------------------------------------------------------------ system names
# sft = SFT-final (step 27,638), rl = RL-final (phase 2c, step 1,433, the paper's headline model), final2 = final run step 401, live250 = the final run at step 250,
# a2w = science-stage start model A2W alpha 0.7 (start of the final run).
SYSTEMS = ("sft", "rl", "a2w", "live250", "final2")
LABEL = {"sft": "SFT-final", "rl": "RL-final", "a2w": "A2W (start of final run)", "live250": "final run step 250", "final2": "final2 (step 401)"}
SENT_BENCH = ("flores_plus", "ntrex", "mafand", "smol_sent", "salt", "glot_story", "afs_dev", "afs_test")
BENCH_LABEL = {"flores_plus": "FLORES+", "ntrex": "NTREX-128", "mafand": "MAFAND-MT", "smol_sent": "SmolSent", "salt": "SALT", "glot_story": "GlotStoryBook",
               "afs_dev": "AfriScience dev (sent.)", "afs_test": "AfriScience test (sent.)"}


def ensure_tar_members(prefixes: Sequence[str]) -> str:
    """Extract the members of the lc1/e2/p5 tarball whose path starts with one of `prefixes` into TAR_DIR (once). Returns TAR_DIR."""
    os.makedirs(TAR_DIR, exist_ok=True)
    need = [p for p in prefixes if not os.path.exists(os.path.join(TAR_DIR, p.rstrip("/").split("*")[0]))]
    if not need:
        return TAR_DIR
    with tarfile.open(TARBALL, "r:gz") as tf:
        members = [m for m in tf.getmembers() if any(m.name.startswith(p) for p in prefixes) and m.isfile()]
        tf.extractall(TAR_DIR, members=members, filter="data")
    return TAR_DIR


def ensure_hub(patterns: Sequence[str]) -> str:
    """Download (read-only) the files of the private Hub dataset matching `patterns` into HUB_DIR with HF_TOKEN (environment, or `HE_ENV_FILE` / $HE_REPO/.env)."""
    marker = [p for p in patterns]
    if all(glob.glob(os.path.join(HUB_DIR, p)) for p in marker):
        return HUB_DIR
    env_file = os.environ.get("HE_ENV_FILE") or os.path.join(REPO, ".env")
    if not os.environ.get("HF_TOKEN") and os.path.exists(env_file):
        for line in open(env_file, encoding="utf-8"):
            line = line.strip()
            if line.startswith("HF_TOKEN=") or line.startswith("export HF_TOKEN="):
                os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    from huggingface_hub import snapshot_download

    snapshot_download(HUB_REPO, repo_type="dataset", token=os.environ.get("HF_TOKEN"), local_dir=HUB_DIR, allow_patterns=list(patterns))
    return HUB_DIR


def read_jsonl(path: str) -> List[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def glotlid_path() -> str:
    p = os.environ.get("HE_GLOTLID")
    if p and os.path.exists(p):
        return p
    c = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--cis-lmu--glotlid/snapshots/*/model_v3.bin"))
    if not c:
        raise FileNotFoundError("GlotLID model_v3.bin not found offline (set HE_GLOTLID)")
    return c[0]


# ------------------------------------------------------------------------------------------------ language groups
def content_modes() -> Dict[str, str]:
    """qe | blend | consensus per language, from the shaping calibration of the final run (shaping_flores_dev_v8.json `content`)."""
    c = json.load(open(os.path.join(CAL, "shaping_flores_dev_v8.json"), encoding="utf-8"))
    return {k: (v.get("content") or {}).get("mode", "qe") for k, v in c.items() if isinstance(v, dict) and "content" in v}


def lid_blocks() -> Dict[str, dict]:
    c = json.load(open(os.path.join(CAL, "shaping_flores_dev_v8.json"), encoding="utf-8"))
    return {k: v["lid"] for k, v in c.items() if isinstance(v, dict) and "lid" in v}


# ------------------------------------------------------------------------------------------------ paired bootstrap
def boot_rng(seed: int = 20261025) -> np.random.Generator:
    return np.random.default_rng(seed)


def macro_paired_bootstrap(diffs: Dict[str, np.ndarray], B: int = 1000, seed: int = 20261025, chunk: int = 250) -> Tuple[float, float, float]:
    """`diffs` = {language: per-row paired differences}. Point estimate = mean over languages of the per-language mean (macro). The 95% interval resamples the rows of every
    language independently (the paper's sentence bootstrap: no language-sampling noise). Returns (point, lo, hi)."""
    rng = boot_rng(seed)
    langs = [l for l, d in diffs.items() if len(d)]
    if not langs:
        return float("nan"), float("nan"), float("nan")
    pt = float(np.mean([diffs[l].mean() for l in langs]))
    acc = np.zeros(B)
    for l in langs:
        d = diffs[l]
        n = len(d)
        m = np.empty(B)
        for s in range(0, B, chunk):
            e = min(B, s + chunk)
            idx = rng.integers(0, n, size=(e - s, n))
            m[s:e] = d[idx].mean(1)
        acc += m
    acc /= len(langs)
    lo, hi = np.percentile(acc, [2.5, 97.5])
    return pt, float(lo), float(hi)


def language_bootstrap(per_lang: np.ndarray, B: int = 2000, seed: int = 20261026) -> Tuple[float, float, float]:
    """Resample LANGUAGES (their per-language mean differences) with replacement: the interval that includes language-sampling noise."""
    rng = boot_rng(seed)
    x = np.asarray(per_lang, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def label_effect(lo: float, hi: float, margin: float) -> str:
    """'final2 better' if the whole interval of (final2 - RL-final) lies below zero (all indicators are 'higher = worse'), 'final2 worse' if above, else 'equal';
    '(negligible)' is appended when the point estimate is inside +-margin."""
    if np.isnan(lo):
        return "n/a"
    if hi < 0:
        return "final2 better"
    if lo > 0:
        return "final2 worse"
    return "equal"
