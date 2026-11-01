# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-47: the reusable parts of the new-language probe scripts (registry sanity, probe-set selection, tag splitting, the log-prob recorder, metric helpers)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

import newlang_probe as P            # noqa: E402
import newlang_probe_sets as S       # noqa: E402
import newlang_registry as R         # noqa: E402
from gemma_mt.languages import ALL_ISOS, LANGUAGES, SUPPORTED_RL  # noqa: E402


def test_registry_controls_are_trained_languages():
    for k, v in R.NEW.items():
        for c in v["controls"]:
            assert c in LANGUAGES and c in SUPPORTED_RL, (k, c)


def test_registry_yardsticks_are_trained_and_have_glot_fallback():
    for k in R.YARDSTICK:
        assert k in SUPPORTED_RL, k
        assert k in R.YARD_GLOT_FALLBACK, k


def test_registry_new_languages_are_not_rl_languages_unless_in_table():
    added_2026_11_01 = {"guw", "lue", "tll", "zne", "lub", "crs", "mfe", "arz"}          # owner 2026-11-01: consensus-only RL languages (calibration v6)
    for k in R.NEW:
        if k in LANGUAGES and k not in added_2026_11_01:
            assert k not in SUPPORTED_RL, f"{k} is a table row without SFT data and must not be an RL language"


def test_registry_set_specs_are_well_formed():
    for k, v in {**R.NEW, **R.YARDSTICK}.items():
        for spec in v["sets"]:
            kind, code = spec.split(":", 1)
            assert kind in ("flores", "floresorig", "smol", "ntrex", "local") and code, (k, spec)


def test_pick_is_deterministic_and_filters_lengths():
    rows = [dict(english=" ".join(["w"] * n), reference="r", id=f"s{i}", orig=(i % 2 == 0)) for i, n in enumerate(list(range(1, 60)) * 3)]
    a, b = S.pick(rows, 20, 7), S.pick(rows, 20, 7)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert len(a) == 20 and all(5 <= len(r["english"].split()) <= 40 for r in a)
    assert [r["id"] for r in S.pick(rows, 20, 8)] != [r["id"] for r in a]
    bigger = S.pick(rows, 40, 7)
    assert [r["id"] for r in bigger[:20]] == [r["id"] for r in a]          # a 150-set starts with the 60-set (the QE of the outputs relies on it)


def test_pick_prefers_original_direction_when_enough():
    rows = [dict(english="a b c d e f g", reference="r", id=f"s{i}", orig=(i < 30)) for i in range(100)]
    assert all(r["orig"] for r in S.pick(rows, 20, 1))
    assert not all(r["orig"] for r in S.pick(rows, 50, 1))


def test_split_tag():
    known = {"yor", "hau", "eng", "fuv"}
    assert P.split_tag("yor E kaaro", known) == ("yor", "E kaaro")
    assert P.split_tag("fuv", known) == ("fuv", "")
    assert P.split_tag("The translation is", known) == (None, "The translation is")
    assert P.split_tag("abc something", known) == (None, "abc something")
    assert P.split_tag("", known) == (None, "")


def test_lp_recorder_gathers_the_chosen_token_one_call_later():
    torch = pytest.importorskip("torch")
    rec = P.LpRecorder()
    s0 = torch.log(torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]))
    ids0 = torch.zeros((2, 1), dtype=torch.long)
    rec(ids0, s0)                                               # first call: nothing to gather yet
    ids1 = torch.tensor([[0, 1], [0, 2]])                       # the sampler chose token 1 for row 0 and token 2 for row 1
    rec(ids1, s0)
    assert len(rec.lps) == 1
    got = rec.lps[0]
    assert float(got[0]) == pytest.approx(float(torch.log(torch.tensor(0.2))), abs=1e-5)
    assert float(got[1]) == pytest.approx(float(torch.log(torch.tensor(0.8))), abs=1e-5)


def test_accepted_labels_for_new_and_trained():
    import newlang_probe_metrics as M

    assert M.accepted_for("fuv", {})[0][0] == "fuv_Latn" and M.accepted_for("fuv", {})[1] == 30
    cal = {"yor": {"lid": {"labels": ["yor_Latn"], "min_chars": 30}}}
    assert M.accepted_for("yor", cal) == (("yor_Latn",), 30)
    assert M.accepted_for("dyu", {})[0] == ("dyu_Latn", "bam_Latn")             # fallback = the production accepted set


def test_all_isos_unchanged_by_the_study():
    assert ALL_ISOS[84] == "ful" and "fuv" not in ALL_ISOS                      # the probe adds no table rows: ids of the packed data stay stable
