# SPDX-License-Identifier: AGPL-3.0-or-later
import json

import numpy as np
import pytest

from gemma_mt.rl import reward_shaping as RS


def _grid():
    return np.linspace(-0.2, 1.2, 281)


@pytest.mark.parametrize("preset,lo,hi", [("repo", -2.0, 2.25), ("owner_note", -2.0, 1.0)])
def test_shape_is_bounded_and_monotone(preset, lo, hi):
    for anchors in [(0.46, 0.60, 0.75), (0.50, 0.66, 0.80), (0.3, 0.4, 0.5), (0.7, 0.7, 0.7)]:
        ys = [RS.shape(c, anchors, 2.0, preset) for c in _grid()]
        assert min(ys) >= lo - 1e-9 and max(ys) <= hi + 1e-9
        assert all(b >= a - 1e-12 for a, b in zip(ys, ys[1:])), "shaping must be monotone non-decreasing"
        assert RS.preset_range(preset) == (lo, hi)


def test_repo_preset_knots_are_exact():
    a = (0.50, 0.65, 0.78)
    assert RS.shape(0.0, a) == -2.0
    assert RS.shape(0.50, a) == pytest.approx(0.0)
    assert RS.shape(0.65, a) == pytest.approx(1.0)
    assert RS.shape(0.78, a) == pytest.approx(2.0)
    assert RS.shape(1.0, a) == pytest.approx(2.25)
    assert RS.shape(0.25, a) == pytest.approx(-1.0)          # halfway between 0 and t_fail on the failure ramp
    assert RS.shape(0.575, a) == pytest.approx(0.5)          # halfway t_fail..t_mid
    assert RS.shape(0.715, a) == pytest.approx(1.5)          # halfway t_mid..t_great


def test_owner_note_preset_saturates_at_plus_one():
    a = (0.50, 0.65, 0.78)
    assert RS.shape(0.0, a, 2.0, "owner_note") == -2.0
    assert RS.shape(0.50, a, 2.0, "owner_note") == pytest.approx(0.0)
    assert RS.shape(0.65, a, 2.0, "owner_note") == pytest.approx(1.0)
    assert RS.shape(0.9, a, 2.0, "owner_note") == pytest.approx(1.0)
    assert RS.shape(1.1, a, 2.0, "owner_note") == pytest.approx(1.0)
    # same per-language anchors and same negative side as the repo preset
    for c in (0.0, 0.2, 0.4, 0.5, 0.6, 0.65):
        assert RS.shape(c, a, 2.0, "owner_note") == pytest.approx(RS.shape(c, a, 2.0, "repo"))


def test_fail_floor_is_enforced():
    # a calibration below 0.46 is raised to the floor
    assert RS.shape(0.46, (0.30, 0.55, 0.70)) == pytest.approx(0.0)
    assert RS.shape(0.40, (0.30, 0.55, 0.70)) < 0.0


def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        RS.shape(0.5, preset="nope")


def test_per_language_anchors_change_the_mapping(tmp_path):
    cal = {"yor": {"anchors": [0.52, 0.70, 0.82], "usable": True, "reliability": 1.0},
           "hau": {"anchors": [0.46, 0.55, 0.65], "usable": True, "reliability": 0.5},
           "eng>ibo": {"anchors": [0.48, 0.60, 0.72], "usable": True, "reliability": 0.8},
           "wol": {"anchors": [0.46, 0.5, 0.6], "usable": False, "reliability": 0.0}}
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(cal))
    sh = RS.ContentShaper.from_file(str(p))
    c = 0.60
    ry, rh, ri = sh(c, "yor"), sh(c, "hau"), sh(c, "ibo")
    assert ry < ri < rh, "the same raw QE score must map differently per language"
    assert sh.anchors("yor") == (0.52, 0.70, 0.82)
    assert sh.anchors("zul") == RS.DEFAULT_ANCHORS             # not calibrated -> default
    assert sh.reliability("hau") == 0.5 and sh.reliability("zul") == 1.0
    assert sh.reliability("wol") == 0.0 and not sh.is_usable("wol")
    assert RS.ContentShaper.from_file(str(p), preset="owner_note")(0.95, "yor") == pytest.approx(1.0)


def test_missing_calibration_file_falls_back_to_defaults(tmp_path):
    sh = RS.ContentShaper.from_file(str(tmp_path / "missing.json"))
    assert sh.anchors("yor") == RS.DEFAULT_ANCHORS


def test_adaptive_mode_moves_mid_and_great_but_not_fail():
    sh = RS.ContentShaper(mode="adaptive", calibration={"yor": (0.46, 0.60, 0.75)}, warmup=50, ema=0.0)
    before = sh.anchors("yor")
    sh.observe("yor", [0.8] * 60)
    after = sh.anchors("yor")
    assert after[0] == before[0] == RS.FAIL_FLOOR
    assert after[1] > before[1] and after[2] > before[2]
    cal = RS.ContentShaper(mode="calibrated", calibration={"yor": (0.46, 0.60, 0.75)})
    cal.observe("yor", [0.9] * 500)
    assert cal.anchors("yor") == before


def test_auc_and_reliability_and_pair_anchors():
    assert RS.auc([0.9, 0.8, 0.7], [0.1, 0.2, 0.3]) == 1.0
    assert RS.auc([0.1], [0.9]) == 0.0
    assert RS.auc([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.5)
    assert RS.reliability_from_auc(0.5) == 0.0 and RS.reliability_from_auc(0.95) == 1.0
    assert RS.reliability_from_auc(0.775) == pytest.approx(0.5)
    human = list(np.linspace(0.5, 0.9, 101))
    mism = [0.2] * 50 + [0.55] * 50
    t = RS.pair_anchors(human, mism)
    assert t[0] >= RS.FAIL_FLOOR and t[0] >= 0.55 - 1e-9          # raised to the mismatched P90
    assert t[0] < t[1] < t[2]


def test_calibrate_end_to_end_with_a_stub_scorer(tmp_path):
    import random

    from gemma_mt.rl.calibrate_shaping import calibrate

    pairs = {}
    rows = []
    for iso in ("yor", "hau"):
        for i in range(40):
            en, ref = f"{iso} source sentence number {i} with words", f"{iso} reference translation {i} words words"
            pairs[en] = ref
            rows.append({"english": en, "reference": ref, "iso": iso})
    rng = random.Random(0)

    def scorer(srcs, mts):
        out = []
        for s, m in zip(srcs, mts):
            if s.startswith("hau"):
                out.append(rng.random())                        # QE is blind for hau
            elif m == pairs[s]:
                out.append(0.80 + 0.1 * rng.random())           # human reference
            elif m == s:
                out.append(0.30)                                # copy
            else:
                out.append(0.40 + 0.05 * rng.random())          # mismatched / omitted
        return out

    res = calibrate(rows, scorer, translate=lambda rs: [r["reference"] for r in rs], max_per_language=30, seed=1)
    y, h = res["yor"], res["hau"]
    assert y["usable"] and y["auc_vs_mismatched"] == 1.0 and y["reliability"] == 1.0
    assert y["anchors"][0] == RS.FAIL_FLOOR and y["anchors"][0] < y["anchors"][1] < y["anchors"][2]
    assert 0.8 <= y["anchors"][1] <= 0.9 and y["n"] == 30
    assert not h["usable"] and h["reliability"] == 0.0
    assert y["model"]["below_t_fail"] == 0.0
    # the file the RL stage reads
    p = tmp_path / "shaping.json"
    p.write_text(json.dumps(res))
    sh = RS.ContentShaper.from_file(str(p))
    assert sh.is_usable("yor") and not sh.is_usable("hau") and sh.anchors("yor") == tuple(y["anchors"])


def test_calibration_reports_length_window_and_same_source_accuracy():
    import random

    from gemma_mt.rl.calibrate_shaping import calibrate

    rng = random.Random(1)
    rows = []
    for k in range(40):
        en = " ".join(rng.choice(["market", "farmer", "river", "school", "bread", "house"]) + str(k) for _ in range(8))
        rows.append({"english": en, "reference": en[::-1], "iso": "yor"})

    def scorer(src, mt):
        # reward an exact reversal, punish everything else: gives a clean separation for the test
        return [0.9 if m == s[::-1] else 0.2 for s, m in zip(src, mt)]

    out = calibrate(rows, scorer, None, max_per_language=40)
    e = out["yor"]
    assert {"p05", "p50", "p95", "p01", "p02", "corpus", "s2"} <= set(e["len_ratio"]) and e["len_ratio"]["p05"] <= e["len_ratio"]["p95"]
    assert e["len_ratio"]["corpus"] == 1.0 and e["len_ratio"]["s2"] == 0.0                # reference = reversed source: same length
    assert {"mismatch", "copy", "half_omitted", "append_unrelated"} <= set(e["same_source_accuracy"])
    assert e["same_source_accuracy"]["mismatch"] == 1.0 and 0.0 <= e["reliability_same_source"] <= 1.0


def test_calibration_v4_fields_floor_diacritics_no_data_language_and_loader(tmp_path):
    import random

    from gemma_mt.rl.calibrate_shaping import calibrate

    yor = ["Àwọn àgbẹ̀ láti Ibadan mú ẹ̀fọ́ tútù wá sí ọjà ní òwúrọ̀ ọjọ́ Mọ́ńdè número %d." % i for i in range(40)]
    rows = [{"english": f"The farmers from Ibadan bring fresh vegetables to the market on Monday morning number {i}.", "reference": r,
             "iso": "yor", "benchmark": "flores_plus_dev"} for i, r in enumerate(yor)]
    rows += [{"english": "x y", "reference": "Ọ\u0329mọ náà", "iso": "yor", "benchmark": "mafand_dev"}]        # widens the inventory only
    rng = random.Random(0)
    pairs = {r["english"]: r["reference"] for r in rows}

    def scorer(srcs, mts):
        return [0.8 + 0.05 * rng.random() if pairs.get(s) == m else 0.35 + 0.05 * rng.random() for s, m in zip(srcs, mts)]

    res = calibrate(rows, scorer, None, max_per_language=997, fail_floor=0.0, no_data_languages=["efi"])
    y = res["yor"]
    assert y["n"] == 40 and y["fail_floor"] == 0.0 and y["anchors"][0] < 0.46 <= y["anchors_at_floor_0.46"][0]
    assert y["diacritics"]["applies"] and "0329" in y["diacritics"]["allowed"]["marks"] and y["diacritics"]["threshold"] > 0
    assert res["efi"]["usable"] is False and "no human reference set" in res["efi"]["reason"]
    p = tmp_path / "shaping.json"
    p.write_text(json.dumps({"_meta": {"fail_floor": 0.0}, **res}))
    sh = RS.ContentShaper.from_file(str(p), fail_floor=0.0)
    assert not sh.is_usable("efi") and sh.reliability("efi") == 0.0 and sh.is_usable("yor")
    assert sh.diacritics["yor"]["applies"] and sh.len_stats["yor"]["corpus"] == y["len_ratio"]["corpus"]
    assert sh.length_model("yor").ratio == y["len_ratio"]["corpus"] and sh.length_model("xxx") is None
    # the failure line follows the configured floor
    low = RS.ContentShaper(calibration={"yor": (0.30, 0.40, 0.60)}, fail_floor=0.0)
    old = RS.ContentShaper(calibration={"yor": (0.30, 0.40, 0.60)})
    assert low.anchors("yor")[0] == 0.30 and old.anchors("yor")[0] == RS.FAIL_FLOOR
    assert low(0.40, "yor") == 1.0 and abs(low(0.30, "yor")) < 1e-9
