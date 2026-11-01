# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-68: the domain-robust calibration (rl/anchors_calibrate_v3.py, scripts/build_anchors_v3_calibration.py, scripts/refit_urh_bench.py)."""
import json
import os
import random
import sys

import pytest

from gemma_mt.rl import anchors_calibrate_v3 as C3
from gemma_mt.rl import anchors_v2 as V

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))


def word(i, pre):
    return pre + "".join(chr(97 + (i // 26 ** k) % 26) for k in range(3))


FW = [word(i, "f") for i in range(30)]
CW = [word(i, "c") for i in range(400)]
EN = [word(i, "e") for i in range(400)]


def text(rng, n, fw_share=0.35):
    return " ".join(rng.choice(FW) if rng.random() < fw_share else rng.choice(CW) for _ in range(n))


def units(rng, n, dom, set_name, fw_share=0.35, lo=12, hi=30):
    out = []
    for _ in range(n):
        L = rng.randint(lo, hi)
        out.append({"src": " ".join(rng.choice(EN) for _ in range(L)), "ref": text(rng, L, fw_share), "set": set_name, "dom": dom})
    return out


def funcs():
    top = [[w, 600 - i, 900] for i, w in enumerate(FW)] + [[w, 120 - i // 5, 150] for i, w in enumerate(CW[:200])]
    return {"top": top, "n_seg": 1000, "n_tok": 20000, "en_tf": {}}


def test_binom_upper_is_what_a_small_set_can_resolve():
    assert C3.binom_upper(0, 100) == pytest.approx(0.0295, abs=0.002)
    assert C3.binom_upper(1, 100) == pytest.approx(0.0466, abs=0.003)
    assert C3.binom_upper(0, 1000) < 0.004 and C3.binom_upper(5, 5) == 1.0


def test_decide_small_heldout_sets_use_the_upper_bound_and_the_gate_needs_an_ood_set():
    lodo = {"bench": {"copy": {"n": 100, "fp": 0.0, "sat": 0.0}, "func": {"n": 100, "fp": 0.01, "sat": 0.0}, "pad": {"n": 100, "fp": 0.0, "sat": 0.0}},
            "jw": {"copy": {"n": 600, "fp": 0.01, "sat": 0.0}, "func": {"n": 600, "fp": 0.01, "sat": 0.0}, "pad": {"n": 600, "fp": 0.0, "sat": 0.0}}}
    d = C3.decide(lodo, {"bench"})
    assert d["copy"]["usable"] and d["pad"]["usable"] and not d["copy"]["gate_ok"]        # 0 of 100: usable (upper bound 2.95 %), but 100 sentences cannot certify a saturated share <= 1 %
    assert not d["func"]["usable"]                                                      # 1 of 100: upper bound 4.7 % > 3 %
    big = {"web": {k: {"n": 600, "fp": 0.005, "sat": 0.0} for k in ("copy", "func", "pad")}, "jw": {k: {"n": 600, "fp": 0.0, "sat": 0.0} for k in ("copy", "func", "pad")}}
    assert C3.decide(big, {"web"})["func"]["gate_ok"]
    d = C3.decide(lodo, set())
    assert d["copy"]["usable"] and not d["copy"]["gate_ok"]                             # no out-of-domain held-out set: never the gate


def test_robust_fw_needs_every_domain_to_use_the_word():
    rng = random.Random(0)
    a = [text(rng, 15) for _ in range(60)]
    b = [" ".join(rng.choice(CW[:50]) for _ in range(15)) for _ in range(60)]          # a domain without function words
    top = [[w, 600, 900] for w in FW[:5]] + [[CW[0], 600, 900]]
    got = C3.robust_fw(top, 1000, {"a": a, "b": b}, 10)
    assert FW[0] not in got and (CW[0] in got or got == [])


def test_single_domain_language_gets_copy_and_pad_graded_but_never_func_or_the_gate():
    rng = random.Random(1)
    u = units(rng, 400, "rel", "pivot_held")
    r = C3.build_language_v3("xx", {"pivot_held": u}, funcs(), lam=1.0)
    assert r["copy"]["usable"] and r["pad"]["usable"] and not r["func"]["usable"]
    assert not r["copy"]["gate_ok"] and not r["pad"]["gate_ok"] and not r["func"]["gate_ok"]
    assert r["pad"]["ttr_mode"] == "seg100" and "pivot_held(400)" in r["pad"]["basis"]


def test_ood_set_with_a_different_profile_switches_the_stop_word_profile_off_and_a_similar_one_on():
    rng = random.Random(2)
    rel = units(rng, 300, "rel", "pivot_held")
    ok = units(rng, 300, "ood", "web", fw_share=0.34)
    r = C3.build_language_v3("xx", {"pivot_held": rel, "web": ok}, funcs(), lam=1.0)
    assert r["func"]["usable"] and r["func"]["lodo"]["has_ood_heldout"] and r["func"]["detect_synthetic"] >= 0.5
    shifted = units(rng, 300, "ood", "web", fw_share=0.08)
    r = C3.build_language_v3("xx", {"pivot_held": rel, "web": shifted}, funcs(), lam=1.0)
    assert not r["func"]["usable"] and not r["func"]["gate_ok"]
    blind = C3.build_language_v3("xx", {"pivot_held": rel}, {**funcs(), "top": funcs()["top"][:40]}, lam=9.0)     # a huge margin pushes every threshold to <= 0: the detector would be silent, not usable
    assert blind["pad"] is None or not blind["pad"]["usable"] or (blind["pad"]["detect_synthetic"] or 0) >= 0.5


def test_english_reference_rows_do_not_pass_the_copy_whitelist_threshold_floor():
    rng = random.Random(3)
    u = units(rng, 300, "rel", "pivot_held")
    c = C3.fit_copy_v3(u, floor=0.2, scale=1.5)
    assert c["t_w"] >= 0.3 - 1e-9 and c["enabled"]


def test_build_replaces_blocks_drops_the_religious_entity_block_and_refit_never_tightens_copy(tmp_path):
    import build_anchors_v3_calibration as B

    rng = random.Random(4)
    cal = {"pooled": {}, "languages": {"xx": {"entity2": {"num": {"enabled": True}}, "copy": {"enabled": True, "t_w": 0.1}}}}
    import v3_data as D

    orig = D.load_sets
    D.load_sets = lambda iso, human=None, **kw: {"pivot_held": units(random.Random(5), 300, "rel", "pivot_held")}
    try:
        new, rep = B.build(cal, {"xx": funcs()}, ["xx"], None, human=None, log=lambda *a: None)
    finally:
        D.load_sets = orig
    e = new["languages"]["xx"]
    assert "entity2" not in e and e["copy"]["usable"] and e["v3"]["lam"] == C3.DEFAULT_LAM and cal["languages"]["xx"]["entity2"]      # the input is not modified
    bench = tmp_path / "b.jsonl"
    bench.write_text("\n".join(json.dumps({"english": " ".join(rng.choice(EN) for _ in range(15)), "reference": text(rng, 15)}) for _ in range(30)))
    assert len(B.load_bench(str(bench))) == 30


def test_heldout_sets_under_100_sentences_are_fitted_but_not_judged():
    lodo = {"tiny": {"copy": {"n": 59, "fp": 0.5, "sat": 0.5}}, "jw": {"copy": {"n": 600, "fp": 0.0, "sat": 0.0}}}
    d = C3.decide(lodo, {"tiny"})
    assert d["copy"]["usable"] and d["copy"]["n_heldout_sets"] == 1 and not d["copy"]["has_ood_heldout"]


def test_a_rebuilt_language_never_reaches_the_gate_even_when_the_numbers_would_allow_it():
    rng = random.Random(8)
    rel = units(rng, 300, "rel", "pivot_held")
    web = units(rng, 400, "ood", "web", fw_share=0.34)
    web2 = units(rng, 400, "ood", "news", fw_share=0.34)
    r = C3.build_language_v3("xx", {"pivot_held": rel, "web": web, "news": web2}, funcs(), lam=1.0)
    assert r["func"]["usable"] and r["func"]["gate_candidate"] and not r["func"]["gate_ok"]
