"""Shared helpers for the J1 judge pilot (Gemini-2.5-Pro vs Gemini-3.8-Flash against human DA, SSA-COMET protocol).

Prompt = Figure 8 (MTE, reference-based) of Li et al., "SSA-COMET: Do LLMs Outperform Learned Metrics in Evaluating MT for
Under-Resourced African Languages?", EMNLP 2025 (arXiv 2506.04557v2), verbatim. The only liberties: "[67]Most" is written
"[67] Most" (the PDF text layer has no space), and the layout of one example on one line uses single spaces.
No API key is read or printed here; the key lives in .env and is loaded only by judge_pilot_batch.py.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEED = 2026

# Figure 8, verbatim (header, instruction, closing line).
MTE_HEADER = (
    "Assess the translation adequacy on a continuous scale [0 ~ 100] using the quality levels described below:\n"
    "[0] Nonsense/No meaning preserved: Nearly all information is lost between the translation and source.\n"
    "[34] Some meaning preserved: The translation preserves some of the meaning of the source but misses significant parts.\n"
    "[67] Most meaning preserved: The translation retains most of the meaning of the source.\n"
    "[100] Perfect meaning: The meaning of the translation is completely consistent with the source.\n"
    "Instruction: Please assess the given machine translation based on the source sentence. "
    "Note that you should only output the final score\n"
    "Given examples:\n"
)
MTE_CLOSING = (
    "Based on the examples given, generate the output in exactly the same format, give the score "
    "and do not give any commentary response.\n"
)


def one_line(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def example_block(i: int, src: str, mt: str, ref: str, score: int) -> str:
    return f"Example {i}:\nSource: {one_line(src)} Translation: {one_line(mt)} Reference: {one_line(ref)} Score: {int(score)}\n"


def build_mte_prompt(demos: list[dict], src: str, mt: str, ref: str) -> str:
    """demos: list of dicts with src, mt, ref, score (0-100). Test item ends with 'Score: ' as in the paper."""
    body = "".join(example_block(k + 1, d["src"], d["mt"], d["ref"], d["score"]) for k, d in enumerate(demos))
    return MTE_HEADER + body + "\n" + MTE_CLOSING + f"Source: {one_line(src)} Translation: {one_line(mt)} Reference: {one_line(ref)} Score: "


def pick_five_demos(pool: list[dict], key: str, lo: float, hi: float, rng: random.Random) -> list[dict]:
    """Paper: split the min-max range of the training scores into five equal intervals, draw one example from each.
    The order of the five in the prompt is not specified in the paper: we shuffle with the same seeded rng."""
    width = (hi - lo) / 5.0
    demos = []
    for k in range(5):
        a, b = lo + k * width, lo + (k + 1) * width
        cand = [r for r in pool if (a <= r[key] < b) or (k == 4 and r[key] == hi)]
        if not cand:  # only happens for the small AfriMTE pools: take the unused row closest to the interval centre
            c = (a + b) / 2
            cand = sorted((r for r in pool if r not in demos), key=lambda r: abs(r[key] - c))[:1]
        demos.append(rng.choice(cand))
    rng.shuffle(demos)
    return demos


PARSE_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(text: str | None):
    """Return a float in [0, 100] if the reply is a bare number (optionally with surrounding whitespace / a trailing '%' / a
    leading 'Score:'), else None. Anything with commentary or several numbers is a failure (the paper assigns 0.5 to those)."""
    if text is None:
        return None
    t = text.strip().strip("*`").strip()
    t = re.sub(r"^(score\s*[:=]\s*)", "", t, flags=re.I).strip().rstrip("%").strip()
    if not re.fullmatch(PARSE_RE, t):
        return None
    v = float(t)
    return v if 0.0 <= v <= 100.0 else None


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
