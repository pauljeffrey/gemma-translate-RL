# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-62 (2026-11-01): the compact fluency models (rl/fluency.py) and the default-off fluency penalty of TranslationReward. CPU only, no torch; the models are built in a temp dir from synthetic
text with the same functions as scripts/build_fluency_models.py (the real artifacts are checked only when artifacts/fluency_models exists)."""
import json
import math
import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import build_fluency_models as B          # noqa: E402
import fluency_lm as F                    # noqa: E402

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS    # noqa: E402
from gemma_mt.rl import fluency as FL                                # noqa: E402
from gemma_mt.rl.comet import FakeScorer                             # noqa: E402
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward    # noqa: E402
from gemma_mt.rl.reward_shaping import ContentShaper                 # noqa: E402

TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]
SRC = "The market opens at eight o'clock in the morning and the farmers from the village bring fresh vegetables to sell."


SYL = ["ka", "ru", "mo", "te", "bi", "sa", "lo", "ni", "pe", "du", "wa", "zo", "ha", "ye", "gu", "fi"]
SYL2 = ["ph", "gr", "th", "ch", "sk", "tr", "pl", "dr", "br", "fl", "cr", "sp"]


def synth_language(sent_seed, n=1500, vocab_seed=1, syl=SYL):
    """a tiny Markov 'language': 250 pseudo words, each followed by 12 allowed words -> word order carries information; sentences of about 110 characters (like SRC)"""
    vr = random.Random(vocab_seed)
    vocab = ["".join(vr.choice(syl) for _ in range(vr.randint(1, 3))) for _ in range(250)]
    nxt = {w: vr.sample(vocab, 12) for w in vocab}
    rng = random.Random(sent_seed)
    out = []
    for _ in range(n):
        w = rng.choice(vocab); s = [w]
        while len(" ".join(s)) < 105:
            w = rng.choice(nxt[w]); s.append(w)
        out.append(" ".join(s) + ".")
    return out


def build_dir(tmp_path, iso="ach", usable=True, spelling=None):
    texts = synth_language(1, 1600)
    hold, fit = texts[:300], texts[300:]
    lm = B.prune_char(F.CharLM(5).fit(fit)); wm = B.prune_word(F.WordPMI().fit(fit))
    hc = np.array([lm.score(t) for t in hold]); hw = np.array([wm.score(t) for t in hold])
    meta = {"order": 5, "char": {"mu": float(hc.mean()), "sd": float(hc.std() + 1e-6)}, "word": {"mu": float(hw.mean()), "sd": float(hw.std() + 1e-6)}}
    comp = FL.CompactLanguage(B.to_arrays(lm, wm), meta)
    Hc = np.array([comp.score(t)["comb_z"] for t in hold])
    meta.update({"file": f"{iso}.npz", "usable": usable, "comb": {"mean": float(Hc.mean()), "std": float(Hc.std()), "p05": float(np.percentile(Hc, 5)), "p10": float(np.percentile(Hc, 10))}, "siblings": []})
    np.savez_compressed(os.path.join(tmp_path, f"{iso}.npz"), **B.to_arrays(lm, wm))
    if spelling:
        sp_lm = B.prune_char(F.CharLM(5).fit(spelling["fit"]))
        sh = np.array([sp_lm.score(t) for t in spelling["hold"]]); mu, sd = float(sh.mean()), float(sh.std() + 1e-6)
        np.savez_compressed(os.path.join(tmp_path, f"{iso}_spelling.npz"), **B.char_arrays(sp_lm))
        z = (sh - mu) / sd
        meta["spelling"] = {"enabled": True, "file": f"{iso}_spelling.npz", "mu": mu, "sd": sd, "floor": float(np.percentile(z, 5))}
    json.dump({"version": 1, "order": 5, "languages": {iso: meta}}, open(os.path.join(tmp_path, "manifest.json"), "w"))
    return str(tmp_path), texts, hold


def test_compact_scores_equal_the_validated_dict_models_on_200_sentences(tmp_path):
    texts = synth_language(2, 1600); hold, fit = texts[:300], texts[300:]
    lm = B.prune_char(F.CharLM(5).fit(fit)); wm = B.prune_word(F.WordPMI().fit(fit))
    comp = FL.CompactLanguage(B.to_arrays(lm, wm), {"order": 5, "char": {"mu": -1.0, "sd": 0.5}, "word": {"mu": 0.3, "sd": 0.4}})
    sent = synth_language(3, 130) + [" ".join(reversed(t.split())) for t in hold[:70]]          # natural and word-reversed sentences
    assert len(sent) == 200
    for t in sent:
        assert abs(comp.char_score(t) - lm.score(t)) < 1e-3
        w_ref, w_new = wm.score(t), comp.word_score(t)
        assert (w_ref is None) == (w_new is None) and (w_ref is None or abs(w_ref - w_new) < 1e-3)
    r = comp.score(sent[0])
    assert abs(r["char_z"] - (lm.score(sent[0]) + 1.0) / 0.5) < 1e-3
    assert abs(r["comb_z"] - (r["char_z"] + r["word_z"]) / math.sqrt(2.0)) < 1e-9


def test_scorer_separates_clean_garbled_foreign_and_shuffled_text(tmp_path):
    d, texts, hold = build_dir(tmp_path)
    sc = FL.FluencyScorer(d)
    rng = random.Random(0)
    clean = np.mean([sc.score("ach", t)["comb_z"] for t in synth_language(5, 60)])
    garbled = np.mean([sc.score("ach", "".join(rng.choice("abcdefgh") if rng.random() < 0.2 else c for c in t))["comb_z"] for t in synth_language(5, 60)])
    foreign = np.mean([sc.score("ach", t)["comb_z"] for t in synth_language(9, 60, vocab_seed=9, syl=SYL2)])
    shuf = []
    for t in synth_language(5, 60):
        w = t.split(); rng.shuffle(w); shuf.append(sc.score("ach", " ".join(w))["comb_z"])
    assert clean > garbled + 2 and clean > foreign + 2 and clean > np.mean(shuf) + 0.5
    assert sc.score("xxx", "text")["usable"] is False and sc.score("xxx", "text")["comb_z"] is None
    assert sc.ram_bytes() > 0 and sc.usable("ach")


def reward_for(cfg_kw, d=None):
    cfg = RewardConfig(chunking="sentences", consensus=True, **cfg_kw)
    sh = ContentShaper.from_file(None)
    sh.content = {"ach": {"mode": "consensus"}}
    sh.lid, sh.rel = {}, {}
    return TranslationReward(cfg, FakeScorer(), TOK, [EOT], lid=None, shaper=sh)


def samples(texts):
    return [Sample(TOK.encode(f"ach {t}", add_special_tokens=False) + [EOT], "ach", SRC) for t in texts]


def test_flag_off_is_bit_identical_and_the_scorer_is_not_loaded(tmp_path):
    d, texts, hold = build_dir(tmp_path)
    g = synth_language(11, 6)
    off = reward_for({}).__call__(samples(g))
    assert RewardConfig().fluency is False and reward_for({}).fluency is None and off.fluency == {}
    zero = reward_for(dict(fluency=True, fluency_models_dir=d, fluency_penalty_weight=0.0))(samples(g))       # on, but weight 0: nothing is subtracted
    assert zero.rewards == off.rewards and zero.weights == off.weights and zero.fluency["fluency/checked"] == 6.0


def test_penalty_only_clipped_and_never_below_the_failure_floor(tmp_path):
    d, texts, hold = build_dir(tmp_path)
    clean = synth_language(12, 6)
    rng = random.Random(1)
    junk = ["".join(rng.choice("zqxjvk ") for _ in range(110)) for _ in range(6)]
    base = reward_for({})(samples(clean + junk)); on = reward_for(dict(fluency=True, fluency_models_dir=d, fluency_penalty_weight=0.5, fluency_human_floor=False))(samples(clean + junk))
    for i in range(6):                                   # clean text: unchanged (no bonus, no penalty)
        assert on.rewards[i] == base.rewards[i]
    for i in range(6, 12):                               # junk: lower by at most 1.0, never below -2
        drop = base.rewards[i] - on.rewards[i]
        assert 0.0 < drop <= 1.0 + 1e-9 and on.rewards[i] >= -2.0
    assert on.fluency["fluency/penalty_rate/ach"] == 0.5 and on.fluency["fluency/comb_z/ach"] < 0


def test_unusable_language_gets_no_penalty_unless_all_is_requested(tmp_path):
    d, texts, hold = build_dir(tmp_path, usable=False)
    rng = random.Random(2)
    junk = ["".join(rng.choice("zqxjvk ") for _ in range(110)) for _ in range(4)]
    base = reward_for({})(samples(junk))
    off_u = reward_for(dict(fluency=True, fluency_models_dir=d, fluency_human_floor=False))(samples(junk))
    assert off_u.rewards == base.rewards and off_u.fluency.get("fluency/checked") == 0.0
    all_l = reward_for(dict(fluency=True, fluency_models_dir=d, fluency_human_floor=False, fluency_languages="all"))(samples(junk))
    assert all(a < b for a, b in zip(all_l.rewards, base.rewards))


def test_spelling_term_penalises_stripped_diacritics_of_listed_languages_only(tmp_path):
    rng = random.Random(4)
    accents = {"a": "à", "e": "é", "o": "ọ", "i": "í"}
    def mark(t): return "".join(accents.get(c, c) if rng.random() < 0.7 else c for c in t)
    bible = [mark(t) for t in synth_language(21, 1500)]
    d, texts, hold = build_dir(tmp_path, spelling={"hold": bible[:300], "fit": bible[300:]})
    good = [mark(t) for t in synth_language(22, 5)]
    bad = [t.translate({ord(v): k for k, v in accents.items()}) for t in good]                    # same sentences, diacritics stripped
    base = reward_for({})(samples(bad))
    kw = dict(fluency=True, fluency_models_dir=d, fluency_penalty_weight=0.0, fluency_spelling_weight=0.5)
    on_bad = reward_for(kw)(samples(bad)); on_good = reward_for(kw)(samples(good))
    assert on_bad.fluency["fluency/spelling_penalty_rate/ach"] >= 0.8 and all(a < b for a, b in zip(on_bad.rewards, base.rewards))
    assert on_good.fluency["fluency/spelling_penalty_rate/ach"] <= 0.4
    no_sp = reward_for(dict(kw, fluency_spelling_weight=0.0))(samples(bad))
    assert no_sp.rewards == base.rewards                                                         # spelling term off: nothing changes


def test_the_missing_models_directory_is_an_error_when_the_flag_is_on(tmp_path):
    with pytest.raises(FileNotFoundError):
        reward_for(dict(fluency=True, fluency_models_dir=str(tmp_path / "nothing")))


def test_real_artifacts_if_present():
    d = os.path.join(HERE, "..", "artifacts", "fluency_models")
    if not os.path.isfile(os.path.join(d, "manifest.json")):
        pytest.skip("artifacts/fluency_models not built")
    sc = FL.FluencyScorer(d)
    man = json.load(open(os.path.join(d, "manifest.json")))
    assert sc.has("yor") and sc.usable("yor") and sc.spelling_z("yor", "Ẹ káàárọ̀, báwo ni?") is not None
    good = "Ẹ káàárọ̀, ọjà ń ṣí ní agogo mẹ́jọ òwúrọ̀ àwọn àgbẹ̀ láti abúlé sì mú ewébẹ̀ tútù wá láti tà."
    assert sc.score("yor", good)["comb_z"] > sc.score("yor", "qzv xkj wqz jjk lpq zzx vkq")["comb_z"] + 3
    assert all(m["prune_max_drop"] <= 0.01 for m in man["languages"].values())
