"""Shared helpers for the J2 judge runs (QE validation, FLORES+ SFT vs RL, long documents).

Builds on judge_pilot_common (which holds Figure 8 of Li et al., EMNLP 2025, arXiv 2506.04557, verbatim and is NOT altered).
Figure 9 (QE, score only) is Figure 8 with the "Reference:" field removed from every demonstration and from the test line;
header, instruction, closing line and example layout are identical. No API key is read here.
"""
from __future__ import annotations

from judge_pilot_common import MTE_CLOSING, MTE_HEADER, one_line, parse_score, load_jsonl, SEED, REPO  # noqa: F401

QE_HEADER = MTE_HEADER    # Figure 9 header/instruction is identical to Figure 8's (checked against the PDF text, lines ~1190-1200)
QE_CLOSING = MTE_CLOSING  # same closing line


def example_block_qe(i: int, src: str, mt: str, score: int) -> str:
    return f"Example {i}:\nSource: {one_line(src)} Translation: {one_line(mt)} Score: {int(score)}\n"


def build_qe_prompt(demos: list[dict], src: str, mt: str) -> str:
    """Figure 9. demos: dicts with src, mt, score (a 'ref' key, if present, is ignored). Test item ends with 'Score: '."""
    body = "".join(example_block_qe(k + 1, d["src"], d["mt"], d["score"]) for k, d in enumerate(demos))
    return QE_HEADER + body + "\n" + QE_CLOSING + f"Source: {one_line(src)} Translation: {one_line(mt)} Score: "
