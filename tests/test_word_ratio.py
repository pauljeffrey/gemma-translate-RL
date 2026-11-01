import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import build_word_ratio_calibration as b  # noqa: E402
from gemma_mt.rl.reward import RewardConfig  # noqa: E402
from gemma_mt.rl.reward_shaping import ContentShaper  # noqa: E402


def _pairs(ratio, n=120):
    src = "one two three four five six seven eight nine ten"             # 10 words
    return [(src, " ".join(["w"] * max(1, int(round(10 * ratio))))) for _ in range(n)]


def test_block_and_pooled():
    blk = b.word_ratio_block(_pairs(1.2))
    assert blk["n"] == 120 and abs(blk["p50"] - 1.2) < 1e-9 and blk["p99"] == 1.25      # p99 floor
    assert b.word_ratio_block(_pairs(1.2, n=10)) == {}                    # too few sentences -> pooled prior
    hi = b.word_ratio_block(_pairs(5.0))
    assert hi["p99"] == b.P99_CAP                                          # outlier-driven p99 is capped
    p = b.pooled({"a": b.word_ratio_block(_pairs(1.0)), "b": b.word_ratio_block(_pairs(2.0)), "c": b.word_ratio_block(_pairs(1.5))})
    assert p["n_languages"] == 3 and abs(p["p99"] - 1.5) < 1e-9


def _shaper(cal):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cal, f)
    try:
        return ContentShaper.from_file(f.name)
    finally:
        os.unlink(f.name)


def test_shaper_word_excess_is_one_sided_and_per_language():
    sh = _shaper({"zul": {"usable": True, "anchors": [0.5, 0.6, 0.7], "word_ratio": {"p99": 1.25, "p95": 1.0, "n": 200}},
                  "tsn": {"usable": True, "anchors": [0.5, 0.6, 0.7], "word_ratio": {"p99": 2.2, "p95": 1.8, "n": 200}},
                  "_word_ratio_pooled": {"p95": 1.6, "p99": 2.0}})
    src = "a b c d e f g h i j"                                             # 10 words
    long_ = " ".join(["x"] * 20)                                           # ratio 2.0
    assert sh.word_excess("zul", src, long_) > 0.3                         # ln(2.0 / (1.25 x 1.15)) = 0.33
    assert sh.word_excess("tsn", src, long_) == 0.0                        # inside Setswana's own human range
    assert sh.word_excess("xyz", src, long_) == 0.0                        # pooled prior 2.0 x 1.15 = 2.3 > 2.0
    assert sh.word_excess("zul", src, "x") == 0.0                          # too short is not this penalty's job
    assert sh.word_excess("zul", "a b c", long_) == 0.0                    # short sources are ignored
    assert math.isclose(sh.word_excess("zul", src, " ".join(["x"] * 30)), math.log(3.0 / (1.25 * 1.15)))


def test_default_is_off_and_old_calibration_still_loads():
    assert RewardConfig().word_ratio_penalty == 0.0 and RewardConfig().word_ratio_slack == 0.15
    sh = _shaper({"zul": {"usable": True, "anchors": [0.5, 0.6, 0.7]}})
    assert sh.word_excess("zul", "a b c d e", "x " * 40) == 0.0
