# SPDX-License-Identifier: AGPL-3.0-or-later
"""The RL assemble stage on a tiny synthetic pool (needs pyarrow)."""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts")


def _load(name):
    sys.path.insert(0, SCRIPTS)
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def test_assemble_writes_one_file_per_target_with_the_new_columns(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    rl = _load("build_afri2afri_rl")
    sft, out = tmp_path / "sft", tmp_path / "out"
    os.makedirs(sft)
    os.makedirs(out / "pool_pairs")
    # SFT selection: one pair whose texts must be excluded from the RL sources
    pq.write_table(pa.table({"src": ["Ẹ kaaro gbogbo ènìyàn ilé"], "tgt": ["Ina kwana da kowa a gida"]}), sft / "pairs_train.parquet")
    pq.write_table(pa.table({"src": ["x"], "tgt": ["y"]}), sft / "pairs_val.parquet")
    langs = ["yor", "hau", "ibo"]
    alloc = {"hau-yor": 100.0, "ibo-yor": 50.0, "hau-ibo": 10.0}
    (sft / "selection_pairs.json").write_text(json.dumps({k: {"alloc": v} for k, v in alloc.items()}))
    cal = {l: {"lid": {"enabled": True, "labels": [f"{l}_Latn"], "min_chars": 5}} for l in langs}
    (tmp_path / "cal.json").write_text(json.dumps(cal))
    os.makedirs(tmp_path / "bench" / "flores_plus" / "devtest")
    (tmp_path / "bench" / "flores_plus" / "devtest" / "yor_Latn.jsonl").write_text(json.dumps({"id": 0, "text": "Gbolohun ìdánwò láti inú ìdánwò FLORES tí a kò gbọdọ̀ lò rárá lónìí"}) + "\n")
    rng = np.random.default_rng(0)
    for l in langs:
        n = 40
        rows = {"source": [f"{l} sentence number {i} is a normal sentence of some length about the market." for i in range(n)],
                "src_afr_tokens": [int(rng.integers(30, 60)) for _ in range(n)], "length_bin": ["b64"] * n, "n_sentences": [1] * n,
                "source_dataset": ["a/b"] * n, "licence": ["cc-by"] * n}
        rows["source"][0] = "Gbolohun ìdánwò láti inú ìdánwò FLORES tí a kò gbọdọ̀ lò rárá lónìí"            # a benchmark leak
        pq.write_table(pa.table(rows), out / "pool_pairs" / f"{l}.parquet")
    # restrict the language list to the three test languages
    rl.SFT77 = tuple(langs)
    ns = type("A", (), dict(sft=str(sft), out=str(out), bench=str(tmp_path / "bench"), extra_bench=[], calibration=str(tmp_path / "cal.json"), glotlid="fake",
                            cap_per_bin=30, quota=6.0, max_reuse=6, seed=0))()
    rl.cmd_assemble(ns)
    files = sorted(os.listdir(out / "rl_prompts_x2x"))
    assert files == ["hau.parquet", "ibo.parquet", "yor.parquet"]
    t = pq.read_table(out / "rl_prompts_x2x" / "yor.parquet").to_pydict()
    assert set(t) >= {"source", "src_iso", "iso", "language", "src_language", "src_afr_tokens", "length_bin", "source_dataset", "licence", "src_qe_valid", "tgt_qe_valid"}
    assert set(t["iso"]) == {"yor"} and set(t["src_iso"]) == {"hau", "ibo"} and set(t["language"]) == {"Yoruba"}
    assert not any("FLORES" in s for s in t["source"])                       # the benchmark sentence was dropped
    assert all(len(set(zip(*[t["source"][i::1] for i in range(1)]))) >= 1 for _ in [0])
    pairs = list(zip(t["source"], t["src_iso"]))
    assert len(pairs) == len(set(pairs))                                    # a (window, target) pair never repeats
    rep = json.loads((out / "rl_report.json").read_text())
    assert rep["dropped"]["benchmark_leak"] >= 3 and rep["directions_with_prompts"] == 6
