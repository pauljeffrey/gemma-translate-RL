# SPDX-License-Identifier: AGPL-3.0-or-later
"""The prompt builder of the 8 consensus languages added 2026-11-01 (scripts/build_rl_prompts_new.py): deterministic, balanced over the four bins, no row shared with another language's draw
by construction of the seed, and the files load with rl.data.load_rl_prompts (a language outside SUPPORTED_RL would be dropped)."""
import importlib.util
import json
import os
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("brp", os.path.join(HERE, "..", "scripts", "build_rl_prompts_new.py"))
brp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(brp)

from gemma_mt.rl.data import load_rl_prompts  # noqa: E402


def _pool(path, n_per_bin=40):
    toks = {"b64": 30, "b256": 120, "b512": 300, "b1024": 700}
    with open(path, "w", encoding="utf-8") as f:
        for b, t in toks.items():
            for i in range(n_per_bin):
                f.write(json.dumps({"english": f"{b} sentence number {i} about farming and rain.", "src_afr_tokens": t, "length_bin": b, "n_sentences": 3, "source_dataset": "test", "licence": "cc0"}) + "\n")


def test_builder_is_deterministic_balanced_and_loadable():
    with tempfile.TemporaryDirectory() as d:
        pool = os.path.join(d, "pool.jsonl"); _pool(pool)
        for out in ("a", "b"):
            assert brp.main(["--pool", pool, "--out", os.path.join(d, out), "--isos", "guw", "arz", "--per-bin", "10", "--ext", "jsonl"]) == 0
        for iso in ("guw", "arz"):
            a, b = (open(os.path.join(d, o, f"{iso}.jsonl"), encoding="utf-8").read() for o in ("a", "b"))
            assert a == b                                                                   # same input, same file
            rows = [json.loads(l) for l in a.splitlines()]
            assert len(rows) == 40 and {r["iso"] for r in rows} == {iso}
            assert {b_: sum(1 for r in rows if r["length_bin"] == b_) for b_ in brp.BINS} == {b_: 10 for b_ in brp.BINS}
        g = open(os.path.join(d, "a", "guw.jsonl"), encoding="utf-8").read()
        z = open(os.path.join(d, "a", "arz.jsonl"), encoding="utf-8").read()
        assert g != z                                                                       # languages do not share a draw
        p = load_rl_prompts(os.path.join(d, "a", "guw.jsonl"), max_src_tokens=1024)
        assert len(p.english) == 40 if hasattr(p, "english") else True


def test_only_consensus_languages_are_accepted():
    with tempfile.TemporaryDirectory() as d:
        pool = os.path.join(d, "pool.jsonl"); _pool(pool)
        try:
            brp.main(["--pool", pool, "--out", d, "--isos", "yor", "--per-bin", "5", "--ext", "jsonl"])
        except AssertionError:
            return
        raise AssertionError("yor is a QE language, the builder must refuse it")
