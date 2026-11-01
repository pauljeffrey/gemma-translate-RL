# SPDX-License-Identifier: AGPL-3.0-or-later
"""R14: the NLLB reward signal (rl/nllb_signal.py + scripts/precompute_nllb_pseudorefs.py): keys, pseudo-reference store, calibration, weight gating, the reward hook,
the forced-decoding wrapper and the precompute script with a fake translator. CPU only (no ctranslate2, no GPU)."""
import importlib.util
import json
import os
import sys
import unicodedata

import pytest

from gemma_mt.rl import archive
from gemma_mt.rl import nllb_signal as N
from gemma_mt.rl.segments import split_sentences

SRC = "The market opens early. The farmers bring fresh vegetables. Prices are low today."
TR = ["Ala ni ogun kan wa si ile oja.", "Agbe omi igi ewe ina oru ojo.", "Osu ọdún ń tí náà."]


def _calib(lang="kea", delta=10.0, auc=0.95, rel=1.0, use=True, anchors=(0.10, 0.30, 0.50)):
    return {lang: {"pseudoref": {"anchors": list(anchors), "auc_vs_mismatched": auc, "reliability": rel},
                   "teacher_vs_student": {"delta": delta, "lo": 1.0, "hi": 20.0}, "use": use}}


def _signal(calib, **cfg_kw):
    cfg = N.NllbSignalConfig(nllb_signal=True, **cfg_kw)
    store = N.PseudoRefStore("/nonexistent")
    store.add("kea", SRC, split_sentences(SRC), TR)
    return N.NllbSignal(cfg, store=store, calibration=calib)


def test_key_is_the_archive_key_and_nfc_invariant():
    assert N.key_of("kea", SRC) == archive.make_key("kea", SRC)
    nfd = unicodedata.normalize("NFD", "Dr. Ọlá came")
    nfc = unicodedata.normalize("NFC", "Dr. Ọlá came")
    assert N.key_of("yor", nfd) == N.key_of("yor", nfc)
    assert len(N.key_of("kea", SRC)) == 16


def test_store_window_chunk_and_sentence_lookup():
    s = N.PseudoRefStore("/nonexistent")
    ss = split_sentences(SRC)
    s.add("kea", SRC, ss, TR)
    assert s.lookup("kea", SRC) == " ".join(TR)                         # the whole window (archive key)
    assert s.lookup("kea", ss[1]) == TR[1]                              # one sentence
    assert s.lookup("kea", ss[0] + " " + ss[1]) == TR[0] + " " + TR[1]  # a merged chunk of the chunk plan
    assert s.lookup("kea", "A sentence nobody translated.") is None
    assert s.lookup("yor", SRC) is None                                 # other language


def test_rejected_sentence_makes_the_chunk_unknown():
    s = N.PseudoRefStore("/nonexistent")
    ss = split_sentences(SRC)
    s.add("kea", SRC, ss, [TR[0], "", TR[2]])                           # '' = rejected by the precompute (copy / loop / empty)
    assert s.lookup("kea", SRC) is None
    assert s.lookup("kea", ss[0]) == TR[0]
    assert s.lookup("kea", ss[1]) is None


def test_chrf01_bounds():
    assert N.chrf01("abc def ghi", "abc def ghi") == pytest.approx(1.0)
    assert N.chrf01("zzz yyy", "abc def ghi") < 0.2


def test_calibrate_scores_anchors_and_reliability():
    human = [0.40, 0.45, 0.50, 0.55, 0.60, 0.42, 0.48, 0.52, 0.58, 0.65]
    mism = [0.05, 0.08, 0.10, 0.12, 0.07, 0.09, 0.11, 0.06, 0.13, 0.10]
    c = N.calibrate_scores(human, mism)
    t_fail, t_mid, t_great = c["anchors"]
    assert 0.0 <= t_fail < t_mid < t_great
    assert t_fail == pytest.approx(0.126, abs=0.01)
    assert c["auc_vs_mismatched"] == pytest.approx(1.0)
    assert c["reliability"] == pytest.approx(1.0)
    noisy = N.calibrate_scores(human, human)
    assert noisy["reliability"] == 0.0                                  # no separation -> no trust


def test_shape01_hits_the_anchor_rewards():
    a = (0.1, 0.3, 0.5)
    assert N.shape01(0.0, a) == pytest.approx(-2.0)
    assert N.shape01(0.1, a) == pytest.approx(0.0)
    assert N.shape01(0.3, a) == pytest.approx(1.0)
    assert N.shape01(0.5, a) == pytest.approx(2.0)
    assert N.shape01(1.0, a) == pytest.approx(2.25)
    assert N.shape01(0.2, a) < N.shape01(0.4, a)


def test_off_by_default_is_identity():
    sig = N.NllbSignal(N.NllbSignalConfig(), calibration=_calib())     # nllb_signal False
    assert sig.weight("kea") == 0.0
    assert sig.blend_chunk("kea", SRC, "anything", 0.7) == 0.7


def test_weight_gating():
    assert _signal(_calib()).weight("kea", "consensus") == pytest.approx(0.35)
    assert _signal(_calib(rel=0.5)).weight("kea", "consensus") == pytest.approx(0.175)
    assert _signal(_calib(delta=1.0)).weight("kea") == 0.0               # teacher not clearly better than the student
    assert _signal(_calib(auc=0.7)).weight("kea") == 0.0                 # does not separate human from mismatched
    assert _signal(_calib(use=False)).weight("kea") == 0.0
    assert _signal(_calib()).weight("kea", "qe") == 0.0                  # qe languages are untouched ...
    assert _signal(_calib(), nllb_languages=["kea"]).weight("kea", "qe") > 0     # ... unless listed
    assert _signal(_calib(), nllb_weights={"kea": 0.2}).weight("kea") == 0.2
    assert _signal(_calib(), nllb_weights={"kea": 0.0}).weight("kea") == 0.0
    assert _signal(_calib()).weight("yor", "consensus") == 0.0           # no calibration block
    assert _signal(_calib(rel=1.0), nllb_weight_default=0.9).weight("kea") == 0.5   # capped by nllb_weight_max


def test_double_pull_rule():
    base = _signal(_calib()).weight("kea")
    assert _signal(_calib(), nllb_double_pull_languages=["kea"]).weight("kea") == pytest.approx(0.5 * base)
    assert _signal(_calib(), nllb_double_pull_languages=["kea"], nllb_double_pull_factor=0.3).weight("kea") == pytest.approx(0.3 * base)
    assert _signal(_calib(), nllb_double_pull_languages=["hau"]).weight("kea") == pytest.approx(base)
    assert _signal(_calib(), nllb_double_pull_languages=["kea"], nllb_weights={"kea": 0.2}).weight("kea") == 0.2      # an explicit weight is final


def test_blend_chunk_moves_toward_the_pseudoreference_value():
    sig = _signal(_calib())
    good = sig.blend_chunk("kea", SRC, " ".join(TR), -1.0, "consensus")   # identical to the pseudo-reference -> top of the curve
    bad = sig.blend_chunk("kea", SRC, "qqq www eee", -1.0, "consensus")
    assert good > bad
    assert good == pytest.approx(0.65 * -1.0 + 0.35 * N.shape01(1.0, (0.10, 0.30, 0.50)))
    assert bad < 0.65 * -1.0 + 0.35 * 0.0                                   # junk is below the t_fail knot
    # no pseudo-reference for this source: the reward value is returned unchanged
    assert sig.blend_chunk("kea", "A different source sentence.", " ".join(TR), 0.4, "consensus") == 0.4
    assert sig.stats["no_pseudoref"] == 1 and sig.stats["applied"] == 2


def test_calibration_file_roundtrip(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"kea": {"nllb": _calib()["kea"], "anchors": [0.5, 0.6, 0.7]}, "_meta": {}}))
    sig = N.NllbSignal(N.NllbSignalConfig(nllb_signal=True, nllb_calibration_path=str(p)), store=N.PseudoRefStore("/nonexistent"))
    assert sig.weight("kea") > 0 and sig.weight("hau") == 0


def test_forced_to_unit():
    assert N.forced_to_unit(0.0) == 1.0 and N.forced_to_unit(-6.0) == 0.0 and N.forced_to_unit(-3.0) == pytest.approx(0.5) and N.forced_to_unit(-30.0) == 0.0


class _FakeModel:
    def __init__(self):
        self.n = 0

    def score(self, iso, srcs, mts, max_batch_tokens=4096):
        self.n += len(srcs)
        return [-1.0 if s.split()[0] == m.split()[0] else -5.0 for s, m in zip(srcs, mts)]


def test_forced_decoding_scorer_counts_and_combination():
    f = N.ForcedDecodingScorer("", "", model=_FakeModel())
    assert f.score("kea", ["a b", "c d"], ["a x", "z z"]) == [-1.0, -5.0]
    assert f.calls == 1 and f.pairs == 2
    calib = _calib()
    calib["kea"]["forced"] = {"anchors": [0.2, 0.5, 0.8], "auc_vs_mismatched": 0.9, "reliability": 1.0}
    store = N.PseudoRefStore("/nonexistent")
    store.add("kea", SRC, split_sentences(SRC), TR)
    src1 = "The market opens early."                                    # one sentence of SRC: chunk-level lookup
    good_mt = TR[0]                                                      # equals the pseudo-reference of src1 -> A at the top of the curve
    # (A) alone
    a_only = N.NllbSignal(N.NllbSignalConfig(nllb_signal=True), store=store, calibration=calib).blend_chunk("kea", src1, good_mt, -1.0, "consensus")
    # (A)+(B), min: the forced score of a mismatching first word is -5 nats/token -> unit 0.17 -> below the fail knot -> pulls the chunk down
    fake = _FakeModel()
    sig = N.NllbSignal(N.NllbSignalConfig(nllb_signal=True, nllb_forced=True), store=store, calibration=calib, forced=N.ForcedDecodingScorer("", "", model=fake))
    sig.prepare([("kea", src1, good_mt, "consensus"), ("kea", src1, good_mt, "consensus"), ("yor", src1, good_mt, "consensus")])
    assert fake.n == 1 and sig.stats["forced_pairs"] == 1                # one batched call, duplicates and unweighted languages skipped
    both = sig.blend_chunk("kea", src1, good_mt, -1.0, "consensus")
    assert fake.n == 1                                                   # served from the cache
    assert both < a_only                                                 # src1 starts with 'The', mt with 'Ala': fake forced score -5 -> min() lowers the value
    low_auc = _calib(); low_auc["kea"]["forced"] = {"anchors": [0.2, 0.5, 0.8], "auc_vs_mismatched": 0.6, "reliability": 0.0}
    sig2 = N.NllbSignal(N.NllbSignalConfig(nllb_signal=True, nllb_forced=True), store=store, calibration=low_auc, forced=N.ForcedDecodingScorer("", "", model=_FakeModel()))
    assert sig2.blend_chunk("kea", src1, good_mt, -1.0, "consensus") == pytest.approx(a_only)     # forced block below nllb_forced_min_auc is ignored


def test_precompute_script_with_a_fake_translator(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    pool = tmp_path / "pool"
    pool.mkdir()
    import pyarrow as pa_

    pq.write_table(pa_.table({"english": [SRC, "Rain fell all night. The river rose."], "source_dataset": ["a", "b"]}), pool / "kea.parquet")
    spec = importlib.util.spec_from_file_location("pre", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "precompute_nllb_pseudorefs.py"))
    pre = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pre)

    class FakeNllb:
        def __init__(self, *a, **k):
            pass

        def translate(self, texts, iso, return_scores=False, **k):
            h = [f"kulu {i} {len(t.split())} ala ni ogun" for i, t in enumerate(texts)]
            return (h, [-0.5] * len(h)) if return_scores else h

    monkeypatch.setattr(pre, "NllbCT2", FakeNllb)
    out = tmp_path / "out"
    rc = pre.main(["--langs", "kea", "--out", str(out), "--ct2-dir", "x", "--hf-dir", "y", "--pools", str(pool / "{iso}.parquet"), "--max-sentences", "0"])
    assert rc == 0
    assert (out / "kea.parquet").exists()
    t = pq.read_table(out / "kea.parquet").to_pydict()
    assert t["key"][0] == archive.make_key("kea", t["source"][0])
    assert set(t["licence_class"]) == {"reward_only_nc"}
    assert [len(a) for a in t["sentences"]] == [len(b) for b in t["translations"]]
    store = N.PseudoRefStore(str(out))
    assert store.lookup("kea", SRC) is not None
    assert json.load(open(out / "LICENCE_REWARD_ONLY_NC.json"))["licence_class"] == "reward_only_nc"


# ------------------------------------------------------------------------------------------------ the reward hook (patches/r14_nllb_reward_hook.patch)
def test_reward_hook_end_to_end():
    """Needs the patched rl/reward.py (RewardConfig.nllb_signal); skipped on an unpatched tree. Off = bit-identical; on = pulled toward the pseudo-reference."""
    from gemma_mt.rl.reward import RewardConfig
    if not hasattr(RewardConfig, "nllb_signal"):
        pytest.skip("rl/reward.py is not patched with r14_nllb_reward_hook.patch")
    from gemma_mt.fake_tokenizer import SPECIALS, FakeGemmaTokenizer
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.rl.reward import Sample, TranslationReward
    from gemma_mt.rl.reward_shaping import ContentShaper

    tok, eot = FakeGemmaTokenizer(), SPECIALS["<end_of_turn>"]
    src = "The market opens at eight o'clock in the morning and the farmers from the village bring fresh vegetables to sell."
    good = "Oja n si ni agogo mejo owuro, awon agbe lati abule si mu ewebe tutu wa lati ta."
    texts = [good, good + " e", good, "Ala ni ogun kan wa si ile ti oja agbe omi igi ewe ina oru ojo osu"] * 2

    def build():
        sh = ContentShaper.from_file(None)
        sh.content = {"yor": {"mode": "consensus"}}
        return TranslationReward(RewardConfig(chunking="sentences", consensus=True), FakeScorer(), tok, [eot], shaper=sh)

    def samples():
        return [Sample(tok.encode(f"yor {t}", add_special_tokens=False) + [eot], "yor", src) for t in texts]

    base = build()(samples(), group_size=8).rewards
    assert build()(samples(), group_size=8).rewards == base                       # deterministic
    rew = build()
    store = N.PseudoRefStore("/nonexistent")
    store.add("yor", src, split_sentences(src), [good])
    cal = _calib("yor", anchors=(0.2, 0.6, 0.9))
    cfg = N.NllbSignalConfig(nllb_signal=True)
    rew.nllb = N.NllbSignal(cfg, store=store, calibration=cal)
    with_nllb = rew(samples(), group_size=8).rewards
    assert with_nllb != base
    assert with_nllb[0] > with_nllb[3]                                             # the sample equal to the pseudo-reference beats the junk one
    rew2 = build()
    rew2.nllb = N.NllbSignal(cfg, store=N.PseudoRefStore("/nonexistent"), calibration=cal)     # no pseudo-reference stored: unchanged
    assert rew2(samples(), group_size=8).rewards == base
