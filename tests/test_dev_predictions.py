# SPDX-License-Identifier: AGPL-3.0-or-later
"""Patch 3a (owner 2026-11-03): saving the dev predictions of every dev evaluation (`save_dev_predictions`) changes nothing else in the run."""
import json
import os

import pytest


def _rows(path):
    return [json.loads(l) for l in open(path)]


def test_dev_predictions_saving_alone_changes_nothing_else():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from gemma_mt.rl.train_rl import run_smoke

    a = run_smoke(sft_steps=60, benchmark=True, overrides={"steps": 4, "eval_every": 2})
    b = run_smoke(sft_steps=60, benchmark=True, overrides={"steps": 4, "eval_every": 2, "save_dev_predictions": True})
    ra = [r["reward"] for r in _rows(a["workdir"] + "/rl/metrics.jsonl") if "reward" in r]
    rb = [r["reward"] for r in _rows(b["workdir"] + "/rl/metrics.jsonl") if "reward" in r]
    assert ra == rb and not os.path.exists(a["workdir"] + "/rl/dev_predictions") and os.listdir(b["workdir"] + "/rl/dev_predictions")
