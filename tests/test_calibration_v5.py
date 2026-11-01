# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-37 / D-38: calibration/shaping_flores_dev_v5.json (v4 + the 35 added languages + lid + content blocks) and the configs that use it."""
import json
import os

import pytest

pytest.importorskip("yaml")

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.languages import CONSENSUS_ONLY as _CONSENSUS_ALL, LANGUAGES, QE_LANGUAGES, QE_VALIDATED, SUPPORTED_RL as _SUPPORTED_ALL

# calibration v5 covers the original 77 RL languages; the 8 consensus languages added 2026-11-01 are in v6 (tests/test_calibration_v6.py)
SUPPORTED_RL = tuple(_SUPPORTED_ALL[:77])
CONSENSUS_ONLY = tuple(c for c in _CONSENSUS_ALL if c in SUPPORTED_RL)
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.lid import FakeLID
from gemma_mt.rl.reward import CONTENT_MODES, RewardConfig, Sample, TranslationReward
from gemma_mt.rl.reward_shaping import ContentShaper
from gemma_mt.utils import dataclass_from_dict, load_yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V4 = os.path.join(ROOT, "calibration", "shaping_flores_dev_v4.json")
V5 = os.path.join(ROOT, "calibration", "shaping_flores_dev_v5.json")
TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]


def _cal():
    if not os.path.exists(V5):
        pytest.skip("calibration v5 not built")
    return json.load(open(V5, encoding="utf-8"))


def test_v5_extends_v4_without_changing_it():
    cal = _cal()
    v4 = json.load(open(V4, encoding="utf-8"))
    for k, e in v4.items():
        if k.startswith("_"):
            continue
        for field in ("anchors", "auc_vs_mismatched", "reliability_same_source", "len_ratio", "diacritics", "lexical", "omission", "usable", "n"):
            assert cal[k].get(field) == e.get(field), (k, field)
    assert "content" not in v4["yor"] and "lid" not in v4["yor"]                       # v4 itself is untouched


def test_every_rl_language_has_a_content_mode_a_weight_and_a_lid_block():
    cal = _cal()
    langs = {k for k in cal if not k.startswith("_")}
    assert langs == set(SUPPORTED_RL) and len(langs) == 77
    modes = {m: set() for m in CONTENT_MODES}
    for k in SUPPORTED_RL:
        e = cal[k]
        assert e["content"]["mode"] in CONTENT_MODES
        modes[e["content"]["mode"]].add(k)
        assert "lid" in e and "enabled" in e["lid"]
        if e["content"]["mode"] == "consensus":
            assert k in CONSENSUS_ONLY and e["reliability_same_source"] == 0.5 and not e.get("anchors")
        if e["content"]["mode"] != "qe":
            assert 0 < e["content"]["beta"] <= 1
    assert modes["consensus"] == set(CONSENSUS_ONLY)
    assert {"kbp", "nus", "kam", "umb", "kmb", "mos", "bam", "dyu", "sag"} <= modes["blend"]                      # the owner's weak-QE list
    assert modes["blend"] - set(QE_VALIDATED) == set(QE_LANGUAGES) - set(QE_VALIDATED)                                # every listed (unvalidated) language is blended
    assert len(modes["qe"]) + len(modes["blend"]) + len(modes["consensus"]) == 77 and modes["qe"] <= set(QE_VALIDATED)
    for k in ("yor", "hau", "swh", "zul", "amh"):
        assert cal[k]["content"]["mode"] == "qe"


def test_weights_and_anchors_of_the_added_languages():
    cal = _cal()
    pooled = cal["_meta_v5"]["pooled_prior_anchors"]
    for k in ("loz", "ven", "iso", "tiv", "kqn", "wal", "lun", "kua", "kon"):
        e = cal[k]
        assert e["reliability_same_source"] == 0.3 and e["anchors"] == pooled and e["content"]["mode"] == "blend" and "len_ratio" not in e and "lexical" not in e
    for k in ("dyu", "sag"):
        e = cal[k]
        assert e["reliability_same_source"] == 0.4 and e["usable"] is True and e["auc_vs_mismatched"] > 0.6          # the owner's weight, never above the measured one
        assert e["usable_by_auc"] == (e["auc_vs_mismatched"] >= 0.70)
        assert e["n"] >= 900 and len(e["anchors"]) == 3 and e["anchors"][0] < e["anchors"][1] < e["anchors"][2] and "len_ratio" in e
    assert cal["swc"]["anchors"] == cal["swh"]["anchors"] and cal["swc"]["reliability_same_source"] == 0.3
    assert 0.5 < pooled[0] < pooled[1] < pooled[2] < 0.8                                    # the median of the 41 calibrated languages, not the default 0.46 / 0.60 / 0.75
    assert pooled != [0.46, 0.60, 0.75]


def test_the_lid_policy_is_the_owners():
    cal = _cal()
    enabled = cal["_meta_v5"]["lid"]["enabled"]
    assert len(enabled) >= 70 and "pcm" not in enabled and "yor" in enabled
    for k in SUPPORTED_RL:
        b = cal[k]["lid"]
        assert b["enabled"] == (k in enabled)
        if b["enabled"]:
            assert b["recall"] >= 0.90


def test_the_shaper_loads_v5_with_modes_lid_and_reliability():
    cal = _cal()
    sh = ContentShaper.from_file(V5, reliability_from="same_source", fail_floor=0.0)
    assert sh.content["ach"]["mode"] == "consensus" and sh.lid["yor"]["enabled"]
    assert sh.reliability("ach") == 0.5 and sh.reliability("loz") == 0.3 and sh.reliability("dyu") == 0.4 and sh.reliability("efi") == 0.0
    assert sh.reliability("yor") == 1.0 and sh.is_usable("dyu")
    assert "ach" not in sh.cal and "loz" in sh.cal


@pytest.mark.parametrize("name", ["rl.yaml", "rl_rtxpro.yaml"])
def test_the_configs_build_the_reward_with_every_language(name):
    cal = _cal()
    from gemma_mt.rl.train_rl import RLConfig

    d = load_yaml(os.path.join(ROOT, "configs", name))
    rl = dataclass_from_dict(RLConfig, d)
    rc = dataclass_from_dict(RewardConfig, rl.reward)
    assert rc.lid_enabled and rc.consensus and rc.chunking == "sentences" and rc.eos_discipline and rc.nfc and rc.omission_detector
    assert rc.calibration_path.endswith("shaping_flores_dev_v5.json") and rc.selfrep_guard == "consensus" and rc.consensus_min_valid == 3
    rc.calibration_path = V5                                                         # the rental path does not exist here
    lid = FakeLID(fn=lambda t: "und")
    r = TranslationReward(rc, FakeScorer(), TOK, [EOT], lid=lid)
    modes = {l: r.content_spec(l) for l in SUPPORTED_RL}                              # every one of the 77 languages has a mode
    assert sum(1 for m in modes.values() if m[0] == "consensus") == 23 and sum(1 for m in modes.values() if m[0] == "blend") == 21
    assert sum(1 for m in modes.values() if m[0] == "qe") == 33
    assert rl.extra_libs == ([] if name == "rl.yaml" else ["/workspace/sft_build/pylibs"])
    assert rl.language_weights == {"efi": 0.0001}
    # every language runs through the reward with its own accepted label (fake LID) and a plausible translation
    src = "The market opens at eight o'clock in the morning and the farmers from the village bring fresh vegetables to sell."
    good = "Oja n si ni agogo mejo owuro, awon agbe lati abule si mu ewebe tutu wa lati ta."
    import re

    smp = [Sample(TOK.encode(f"{l} {good} @@{l}@@", add_special_tokens=False) + [EOT], l, src) for l in SUPPORTED_RL for _ in range(4)]
    lid.fn = lambda t: cal[re.search(r"@@(\w+)@@", t).group(1)]["lid"]["labels"][0]       # the first accepted label of the completion's own language
    out = r(smp, group_size=4)
    assert len(out.rewards) == 4 * 77 and all(-2.0 <= x <= 2.25 for x in out.rewards)
    assert "wrong_language" not in out.status and out.stats["fail/wrong_language"] == 0.0
    ok = [i for i, s in enumerate(out.status) if s == "ok"]
    assert len(ok) > 4 * 60
    assert all(out.qe[i] is None for i, s in enumerate(smp) if LANGUAGES[s.iso].rl_consensus and out.status[i] == "ok")      # no QE call for the consensus languages
