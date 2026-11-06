# SPDX-License-Identifier: AGPL-3.0-or-later
"""385M plan (owner 2026-11-03): news-style English prompts as a second domain next to FineWeb-Edu / Wikipedia, 70 / 30, through the existing D-52 domain mixture.
Rows of files without a `domain` column load as "general" as soon as any row of the pool has a domain."""
import json
import os
from collections import Counter

from gemma_mt.rl.data import PromptSchedule, load_rl_prompts


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _pool(tmp_path):
    root = str(tmp_path / "pool")
    for iso in ("yor", "hau", "ibo"):
        _write(f"{root}/{iso}.jsonl", [{"english": f"General sentence {i} for {iso}.", "iso": iso, "src_afr_tokens": 12, "length_bin": "b64"} for i in range(200)])
        _write(f"{root}/news/{iso}.jsonl", [{"english": f"News sentence {i} for {iso}.", "iso": iso, "src_afr_tokens": 14, "length_bin": "b64", "domain": "news"} for i in range(200)])
    return root


def test_unlabeled_rows_are_general_and_news_is_kept(tmp_path):
    p = load_rl_prompts(_pool(tmp_path))
    assert Counter(p.domain) == {"general": 600, "news": 600}
    assert all(d == "news" for d, e in zip(p.domain, p.english) if e.startswith("News"))


def test_pool_without_any_domain_is_unchanged(tmp_path):
    root = str(tmp_path / "plain")
    _write(f"{root}/yor.jsonl", [{"english": f"A {i}.", "iso": "yor", "src_afr_tokens": 12, "length_bin": "b64"} for i in range(5)])
    assert load_rl_prompts(root).domain == []


def test_schedule_draws_news_thirty_percent(tmp_path):
    p = load_rl_prompts(_pool(tmp_path))
    sch = PromptSchedule(p.iso, 0.0, 7, length_bin=p.length_bin, curriculum={"b64": 50}, domain=p.domain, domain_weights={"general": 0.7, "news": 0.3})
    got = Counter()
    for step in range(50):
        for i in sch.indices(step, 64):
            got[p.domain[i]] += 1
    n = sum(got.values())
    assert abs(got["news"] / n - 0.3) < 0.03 and abs(got["general"] / n - 0.7) < 0.03, got
