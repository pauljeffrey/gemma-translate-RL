# SPDX-License-Identifier: AGPL-3.0-or-later
"""calibration/shaping_flores_dev_v6.json = v5 unchanged + the 8 consensus languages added 2026-11-01 (scripts/build_calibration_v6.py)."""
import json
import os

from gemma_mt.languages import CONSENSUS_ONLY, SUPPORTED_RL

CAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration")
NEW = ("guw", "lue", "tll", "zne", "lub", "crs", "mfe", "arz")


def test_v6_is_v5_plus_the_new_consensus_languages():
    v5 = json.load(open(os.path.join(CAL, "shaping_flores_dev_v5.json"), encoding="utf-8"))
    v6 = json.load(open(os.path.join(CAL, "shaping_flores_dev_v6.json"), encoding="utf-8"))
    assert all(v6[k] == v5[k] for k in v5)                                                   # v5 untouched inside v6
    langs = {k for k in v6 if not k.startswith("_")}
    assert langs == set(SUPPORTED_RL) and len(langs) == 85
    for k in NEW:
        e = v6[k]
        assert k in CONSENSUS_ONLY and e["content"] == {"mode": "consensus", "beta": 1.0, "why": e["content"]["why"]}
        assert e["reliability_same_source"] == 0.5 and not e.get("anchors")
        lid = e["lid"]
        assert lid["enabled"] and lid["labels"] and lid["recall"] >= 0.90 and lid["min_chars"] in (15, 30, 60, 120)
    assert v6["arz"]["lid"].get("weak") is True and "guw" in v6 and not v6["guw"]["lid"].get("weak")
