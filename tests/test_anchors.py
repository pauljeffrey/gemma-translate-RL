# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-56: the constraint penalties (rl/anchors.py): entity preservation, repetition, cross-prompt boilerplate, the lexicon / LID hooks, the calibration rules and the reward hooks."""
import json
import random
import threading

import numpy as np
import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import anchors as A
from gemma_mt.rl import anchors_calibrate as K
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
from gemma_mt.rl.reward_shaping import ContentShaper

TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]
STOP = [EOT]

THR = {"3": [0.0, 0.05, 0.05, 0.10, 0.10, 0.10], "4": [0.0, 0.02, 0.03, 0.05, 0.05, 0.05], "5": [0.0, 0.0, 0.01, 0.03, 0.03, 0.03]}
CAP = {"3": 0.3, "4": 0.3, "5": 0.3}


def cal(**over):
    c = {"_meta": {"weak_languages": ["yor", "kin"]},
         "pooled": {"entity": {"num": {"t": 1.0, "min_n": 1, "enabled": True}, "name": {"t": 0.5, "min_n": 3, "enabled": True}},
                    "repeat": {"thr": THR, "cap": CAP, "enabled": True},
                    "boiler": {"thr": 0.03, "cap": 0.4, "enabled": True},
                    "lid_prob": {"pmin": 0.6, "enabled": True}},
         "languages": {"kin": {"entity": {"num": {"t": 1.0, "min_n": 1, "enabled": False}}, "repeat": {"thr": THR, "cap": CAP, "enabled": False}},
                       "swh": {"entity": {"num": {"t": 0.5, "min_n": 1, "enabled": True}}}},
         "proxies": {"swc": "swh"}}
    c.update(over)
    return c


WORDS = ("ala ni ogun kan wa si ile ti oja agbe omi igi ewe ina oru ojo osu odun ata isu eja ewure aso bata ile iwe omo baba iya ana oni ola ilu "
         "ona ese owo ori oju imu enu eti ika ejika orun osan alẹ aaro afẹfẹ ojo egbon yinyin okuta iyanrin odo okun").split()


def rtext(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


# ------------------------------------------------------------------------------------------------ (a) entities
def test_extract_entities_numbers_names_sentence_initial_skipped():
    nums, names = A.extract_entities("In 2019, 1,200 people met Ada Obi and Chinua in Lagos. Then Kano!")
    assert nums == ["2019", "1200"]
    assert "ada" in names and "obi" in names and "chinua" in names and "lagos" in names
    assert "then" not in names and "in" not in names                        # sentence-initial / short words are not names


def test_entity_counts_tolerance_and_separators():
    src = "Yesterday Ada Obi paid 1,200 naira in Maiduguri."
    ec = A.entity_counts(src, "Ada Obi san 1200 naira ni Maidugurí.")      # diacritics, separators
    assert (ec.n_num, ec.hit_num) == (1, 1) and ec.n_name == 3 and ec.hit_name == 3
    ec = A.entity_counts(src, "Adda Obi san 1.200 naira ni Maiduguri")       # one-character edit in a name >= 5 letters (Adda is 4: needs an exact match)
    assert ec.hit_num == 1 and ec.hit_name == 2
    assert A.entity_counts(src, "nothing here").hit_num == 0
    assert A.entity_counts("no entities here", "x").n_num == 0


def test_entity_deficit_is_linear_below_the_human_threshold_and_off_when_disabled():
    an = A.Anchors(cal())
    src = "Ada Obi bought 3 goats and 4 hens for 500 naira."
    full = "Ada Obi ra ewure 3 ati adie 4 fun 500 naira."
    assert an.entity_deficit("yor", src, full) == (0.0, 3)
    d, n = an.entity_deficit("yor", src, "Ada Obi ra ewure ati adie fun naira.")      # all three numbers missing
    assert n == 3 and d == pytest.approx(1.0)
    d, n = an.entity_deficit("yor", src, "ewure 3 adie 4 fun 5 naira")                 # 2 of 3
    assert d == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert an.entity_deficit("kin", src, "x") == (0.0, 0)                               # kin: its own block says the channel is not reliable
    assert an.entity_deficit("yor", "no numbers and no names here", "x") == (0.0, 0)


def test_proxy_language_borrows_the_blocks_and_pooled_prior_covers_the_rest():
    an = A.Anchors(cal())
    assert an.enabled("swc", "entity") and an._blk("swc", "entity")["num"]["t"] == 0.5     # proxy swc -> swh
    assert an.enabled("zzz", "entity") and an._blk("zzz", "entity")["num"]["t"] == 1.0     # no block at all: the pooled prior
    assert not an.enabled("kin", "entity") and not an.enabled("kin", "repeat")
    assert not A.Anchors(cal(), detectors=("boiler",)).enabled("yor", "entity")


# ------------------------------------------------------------------------------------------------ (b) repetition
def test_repeat_shares_and_length_bucket():
    w = "a b c a b c a b c d".split()
    sh = A.repeat_shares(w)
    assert sh[3] == pytest.approx(1 - 4 / 8) and sh[5] == pytest.approx(1 - 4 / 6)      # a b c x3 -> the 3-gram repeats twice
    assert A.repeat_shares(["a"])[3] == 0.0
    assert [A.length_bucket(n) for n in (10, 25, 26, 60, 61, 150, 400, 1000, 1001)] == [0, 0, 1, 1, 2, 2, 3, 4, 5]


def test_repeat_deficit_zero_for_natural_text_one_for_loops_and_source_allowance():
    an = A.Anchors(cal())
    rng = random.Random(1)
    natural = " ".join(rng.sample(WORDS, 40))
    p = A.prepare("yor", "x " * 40, natural, with_grams=False)
    assert an.repeat_deficit("yor", p) == 0.0
    base = " ".join(rng.sample(WORDS, 12))
    looped = base + (" " + base) * 5
    p = A.prepare("yor", "source", looped, with_grams=False)
    assert an.repeat_deficit("yor", p) == pytest.approx(1.0)
    partial = base + " " + base + " " + " ".join(rng.sample(WORDS, 30))                 # one repeated clause: a deficit between 0 and 1
    d = an.repeat_deficit("yor", A.prepare("yor", "source", partial, with_grams=False))
    assert 0.0 < d < 1.0
    assert an.repeat_deficit("yor", A.prepare("yor", "source", "too short", with_grams=False)) is None
    # a source that repeats itself earns the translation the same allowance
    src = base + " " + base + " " + " ".join(rng.sample(WORDS, 30))
    assert an.repeat_deficit("yor", A.prepare("yor", src, partial, with_grams=False)) < d
    assert an.repeat_deficit("kin", A.prepare("kin", "source", looped, with_grams=False)) is None     # kin: detector off for the language


# ------------------------------------------------------------------------------------------------ (c) boilerplate buffer
def _fill(buf, lang, n_prompts, per=1, stock=None, seed=0, words=14):
    rng = random.Random(seed)
    ps = []
    for k in range(n_prompts):
        src = f"English source number {k} about {rtext(rng, 3)} {k}"
        for j in range(per):
            body = rtext(rng, words) + (" " + stock if stock else "")
            p = A.prepare(lang, src, body)
            buf.add(lang, p.key, p.grams, p.src_hashes)
            ps.append(p)
    return ps


def test_buffer_is_silent_until_it_has_150_outputs_from_30_prompts():
    buf = A.BoilerplateBuffer()                                     # defaults 600 / 150 / 30 / 3
    stock = "this stock sentence is in every single output of the model today"
    _fill(buf, "yor", 29, per=4, stock=stock)                       # 116 outputs, 29 prompts
    probe = A.prepare("yor", "a new source", "fresh words here " + stock)
    assert buf.size("yor") == (116, 29) and not buf.ready("yor") and buf.score("yor", probe.key, probe.grams, probe.src_hashes) is None
    _fill(buf, "yor", 40, per=4, stock=stock, seed=5)               # now 276 outputs from 69 prompts (distinct keys: the sources differ by k only for seed 0, so use fresh text)
    assert buf.ready("yor")


def test_buffer_flags_a_stock_sentence_but_not_fresh_text_and_excludes_the_own_prompt():
    buf = A.BoilerplateBuffer(min_outputs=1, min_prompts=1)
    stock = "wọn sọ pé ó dára púpọ̀ fún gbogbo ènìyàn ní ilé yìí lónìí"
    ps = _fill(buf, "yor", 60, stock=stock, seed=2)
    rng = random.Random(99)
    fresh = A.prepare("yor", "another unrelated English source", rtext(rng, 25))
    assert buf.score("yor", fresh.key, fresh.grams, fresh.src_hashes) == pytest.approx(0.0, abs=0.05)
    boiler = A.prepare("yor", "another unrelated English source", rtext(rng, 10) + " " + stock)
    s = buf.score("yor", boiler.key, boiler.grams, boiler.src_hashes)
    assert s > 0.4
    # an output of a prompt already in the buffer: its own stock sentence is not "another prompt"
    own = ps[0]
    solo = A.BoilerplateBuffer(min_outputs=1, min_prompts=1)
    solo.add("yor", own.key, own.grams, own.src_hashes)
    assert solo.score("yor", own.key, own.grams, own.src_hashes) == 0.0
    # only 2 other prompts share the sentence -> below min_other 3
    few = A.BoilerplateBuffer(min_outputs=1, min_prompts=1)
    _fill(few, "yor", 2, stock=stock, seed=3)
    _fill(few, "yor", 30, seed=4)
    assert few.score("yor", boiler.key, boiler.grams, boiler.src_hashes) < 0.05


def test_buffer_ignores_ngrams_copied_from_the_source_and_related_prompts():
    buf = A.BoilerplateBuffer(min_outputs=1, min_prompts=1)
    shared = "the committee approved the new budget for the rural schools programme"
    # the English sources of the buffered prompts all contain the same sentence (boilerplate of the INPUT): their translations legitimately agree
    ps = []
    for k in range(10):
        src = f"{shared} and item {k}"
        body = "ìgbìmọ̀ fọwọ́ sí ìsúná owó tuntun fún àwọn ilé ẹ̀kọ́ ní àwọn abúlé " + f"nkan {k}"
        p = A.prepare("yor", src, body)
        buf.add("yor", p.key, p.grams, p.src_hashes)
        ps.append(p)
    q = A.prepare("yor", f"{shared} and a different item", "ìgbìmọ̀ fọwọ́ sí ìsúná owó tuntun fún àwọn ilé ẹ̀kọ́ ní àwọn abúlé nkan titun")
    assert buf.score("yor", q.key, q.grams, q.src_hashes) == 0.0           # all buffered prompts are related (shared English 6-gram)
    q2 = A.prepare("yor", "a source with nothing in common at all", "ìgbìmọ̀ fọwọ́ sí ìsúná owó tuntun fún àwọn ilé ẹ̀kọ́ ní àwọn abúlé nkan titun")
    assert buf.score("yor", q2.key, q2.grams, q2.src_hashes) > 0.5
    # copied-from-source grams are skipped
    p = A.prepare("yor", "Dangote Cement Plc opened a plant in Obajana Kogi", "Dangote Cement Plc opened a plant in Obajana Kogi")
    assert p.grams.size == 0


def test_buffer_capacity_per_prompt_cap_and_thread_safety():
    buf = A.BoilerplateBuffer(capacity=50, min_outputs=1, min_prompts=1, per_prompt=2)
    rng = random.Random(0)
    p = A.prepare("yor", "same source", rtext(rng, 20))
    for _ in range(16):
        buf.add("yor", p.key, p.grams, p.src_hashes)
    assert buf.size("yor") == (2, 1)                                # a GRPO group adds at most per_prompt outputs

    def work(seed):
        r = random.Random(seed)
        for k in range(200):
            q = A.prepare("yor", f"src {seed} {k}", rtext(r, 15))
            buf.add("yor", q.key, q.grams, q.src_hashes)
            buf.score("yor", q.key, q.grams, q.src_hashes)

    ts = [threading.Thread(target=work, args=(s,)) for s in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert buf.size("yor")[0] <= 50                                 # rolling: the oldest outputs are dropped
    buf.reset()
    assert buf.size("yor") == (0, 0)


def test_boiler_deficit_maps_share_to_zero_one():
    an = A.Anchors(cal(), buffer_min_outputs=1, buffer_min_prompts=1)
    stock = "same closing sentence that the model appends to every translation it writes"
    _fill(an.buffer, "yor", 60, stock=stock, seed=7)
    p = A.prepare("yor", "unrelated source text", "x y z w v " + stock)
    d = an.boiler_deficit("yor", p)
    assert d is not None and 0.0 < d <= 1.0
    clean = A.prepare("yor", "unrelated source text 2", rtext(random.Random(3), 20))
    assert an.boiler_deficit("yor", clean) == 0.0


# ------------------------------------------------------------------------------------------------ (d) lexicon hook, (e) LID probability
def test_lexicon_hook_default_off_and_deficit_with_an_injected_lexicon(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMMA_MT_LEXICON_DIR", str(tmp_path))
    assert A.lexicon_overlap("yor", "The farmers sell vegetables at the market.", "anything") is None    # no lexicon for the language: no signal
    lex = {"meta": {"validated": True}, "e": {"farmer": [3.0, 0, ["agbe", 0.9]], "vegetabl": [3.0, 0, ["ewebe", 0.9]], "market": [3.0, 0, ["oja", 0.9]]}}
    (tmp_path / "yor.json").write_text(json.dumps(lex))
    from gemma_mt.rl import lexicon as L

    L.load_lexicon.cache_clear()
    src = "The farmers sell vegetables at the market."
    good = A.lexicon_overlap("yor", src, "Awon agbe n ta ewebe ni oja.")
    bad = A.lexicon_overlap("yor", src, "Mo feran ounje yii pupo.")
    assert good is not None and good > bad and bad == 0.0
    L.load_lexicon.cache_clear()
    c = cal()
    c["languages"]["yor"] = {"lexicon": {"t": 0.5, "p50": 0.8, "enabled": True}}
    an = A.Anchors(c, detectors=("lexicon",), lexicon_fn=lambda l, s, m: 0.25)
    assert an.lexicon_deficit("yor", src, "x") == pytest.approx(0.5)
    assert A.Anchors(c, detectors=("entity",), lexicon_fn=lambda l, s, m: 0.25).lexicon_deficit("yor", src, "x") is None    # detector not active
    assert A.Anchors(c, detectors=("lexicon",), lexicon_fn=lambda l, s, m: None).lexicon_deficit("yor", src, "x") is None  # lexicon has no opinion


def test_lid_prob_deficit():
    an = A.Anchors(cal(), detectors=("lid_prob",))
    assert an.lid_prob_deficit("yor", True, 0.9) == 0.0
    assert an.lid_prob_deficit("yor", True, 0.3) == pytest.approx(0.5)
    assert an.lid_prob_deficit("yor", False, 0.1) is None and an.lid_prob_deficit("yor", True, None) is None


# ------------------------------------------------------------------------------------------------ calibration rules
def _units(n, rng, num_p=0.5, drop_p=0.0):
    us = []
    for i in range(n):
        has = rng.random() < num_p
        nm = rng.randint(2, 900)
        src = f"Item costs {nm} units today." if has else "The shop opens late."
        ref = ("Nkan ni %d owo loni." % nm if not (has and rng.random() < drop_p) else "Nkan ni owo loni.")
        us.append({"src": src, "ref": ref, "set": "t", "id": str(i)})
    return us


def test_fit_entity_channel_threshold_fp_and_enable_rule():
    rng = random.Random(0)
    us = _units(600, rng, drop_p=0.0)
    tab = K.entity_table(us)
    b = K.fit_entity_channel(tab["own"], tab["other"], "num", 1)
    assert b["t"] == 1.0 and b["fp"] == 0.0 and b["enabled"] and b["detect_other"] > 0.9
    noisy = K.entity_table(_units(600, rng, drop_p=0.4))                         # numbers often written out in words in the references
    b2 = K.fit_entity_channel(noisy["own"], noisy["other"], "num", 1)
    assert b2["t"] == 0.0 and not b2["enabled"]                                    # the 3 % quantile of the recall is 0: no usable threshold
    assert K.fit_entity_channel(tab["own"][:20], tab["other"][:20], "num", 1) is None            # too few judged chunks


def test_fit_repeat_and_pooled_prior():
    rng = random.Random(1)
    units = [{"src": rtext(rng, 30), "ref": " ".join(rng.sample(WORDS, 36)), "set": "t", "id": str(i)} for i in range(300)]
    tab = K.repeat_table(units)["t"]
    r = K.fit_repeat(tab, None)
    assert r["enabled"] and r["fp_cv"] <= K.FP_CV_MAX and set(r["thr"]) == {"3", "4", "5"} and all(len(v) == 6 for v in r["thr"].values())
    assert all(v == sorted(v) for v in r["thr"].values())                           # repetition only grows with length
    p = K.pooled_repeat({"a": tab, "b": tab})
    assert p["enabled"] and p["fp_p90"] <= K.FP_CV_MAX


def test_fit_boiler_threshold_floor_and_pooled():
    sh = np.concatenate([np.zeros(300), np.linspace(0, 0.05, 20)])
    b = K.fit_boiler(sh)
    assert b["thr"] >= K.BOILER_FLOOR and b["enabled"] and b["fp"] <= 0.05
    assert K.fit_boiler(sh[:50]) is None
    bad = K.fit_boiler(np.concatenate([np.zeros(100), np.full(100, 0.5)]))
    assert not bad["enabled"]                                                       # human text that shares a lot with itself: the detector stays off for the language


def test_doc_floor_raises_long_bucket_thresholds_only():
    rows = np.array([[3, 300, 0.2, 0.1, 0.05, 0, 0, 0]] * 150, dtype=float)
    fl = K.doc_floor([rows])
    assert fl["3"][3] == pytest.approx(0.2) and fl["3"][2] is None
    short = np.array([[3, 300, 0.01, 0.0, 0.0, 0, 0, 0]] * 120 + [[1, 40, 0.0, 0.0, 0.0, 0, 0, 0]] * 120, dtype=float)
    t = K._bucket_thr(short, 0.03, None, floor=fl)
    assert t["3"][3] >= 0.2 and t["3"][1] < 0.2


# ------------------------------------------------------------------------------------------------ the reward hooks
def ids(text):
    return TOK.encode(text, add_special_tokens=False) + [EOT]


def mk(tmp_path, calibration=None, content=None, scorer=None, **kw):
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps(calibration or cal()))
    base = dict(anchors_enabled=True, anchors_calibration_path=str(path), consensus=True, chunking="sentences")
    base.update(kw)
    sh = ContentShaper.from_file(None)
    sh.content = dict(content if content is not None else {"yor": {"mode": "consensus"}, "hau": {"mode": "blend", "beta": 0.5}, "swh": {"mode": "qe"}})
    return TranslationReward(RewardConfig(**base), scorer or FakeScorer(), TOK, STOP, shaper=sh)


SRC = "Ada Obi bought 3 goats and 4 hens at the village market for 500 naira on Monday morning."
GOOD = "yor Ada Obi ra ewure 3 ati adie 4 ni oja abule fun 500 naira ni owuro ojo aje."
DROP = "yor Ada Obi ra ewure ati adie ni oja abule fun owo ni owuro ojo aje."


def grp(lang, text, src=SRC, g=4):
    return [Sample(ids(text.replace("yor", lang, 1)), lang, src) for _ in range(g)]


def test_reward_default_off_is_untouched_and_on_only_penalises_deficits(tmp_path):
    off = TranslationReward(RewardConfig(consensus=True, chunking="sentences"), FakeScorer(), TOK, STOP,
                            shaper=_sh({"yor": {"mode": "consensus"}}))
    on = mk(tmp_path)
    assert off.anchors is None and on.anchors is not None
    r_off = off(grp("yor", GOOD) + grp("yor", DROP), group_size=4)
    r_on = on(grp("yor", GOOD) + grp("yor", DROP), group_size=4)
    assert r_off.anchors_penalty == [] and r_off.rewards[:4] == r_on.rewards[:4]            # a good translation loses nothing
    assert all(a > b for a, b in zip(r_off.rewards[4:], r_on.rewards[4:]))                    # the number-dropping one loses weight x deficit
    assert r_on.anchors_penalty[:4] == [0.0] * 4 and all(0.2 < p <= 0.5 + 1e-9 for p in r_on.anchors_penalty[4:])
    assert r_on.stats["anchors/entity/penalised_rate"] == pytest.approx(0.5) and r_on.stats["anchors/applied_rate"] == 1.0
    assert "anchors/penalty/yor" in r_on.stats


def test_reward_modes_off_languages_and_floor(tmp_path):
    rw = mk(tmp_path)
    qe_lang = rw(grp("swh", DROP), group_size=4)                         # swh is a "qe" language: not touched
    assert qe_lang.anchors_penalty == [0.0] * 4 and qe_lang.stats["anchors/applied_rate"] == 0.0
    blend = rw(grp("hau", DROP), group_size=4)
    assert all(p > 0 for p in blend.anchors_penalty)                      # blend is a covered mode
    ctl = mk(tmp_path, anchors_off_languages=["yor"])
    assert ctl(grp("yor", DROP), group_size=4).anchors_penalty == [0.0] * 4          # the CONTROL list
    only_cons = mk(tmp_path, anchors_modes=["consensus"])
    assert only_cons(grp("hau", DROP), group_size=4).anchors_penalty == [0.0] * 4
    # floor: huge weights can never push a reward below the failure penalty
    big = mk(tmp_path, anchors_entity_weight=100.0)
    out = big(grp("yor", DROP), group_size=4)
    assert min(out.rewards) >= big.penalty - 1e-9


def test_reward_repetition_penalty_and_floor_at_the_failure_penalty(tmp_path):
    rng = random.Random(4)
    base = " ".join(rng.sample(WORDS, 12))
    partial = "yor " + base + " " + base + " " + " ".join(rng.sample(WORDS, 30))
    src = " ".join(rng.sample(WORDS, 30))
    rw = mk(tmp_path, selfrep_guard="off")
    off = mk(tmp_path, anchors_repeat_weight=0.0, anchors_entity_weight=0.0, anchors_boiler_weight=0.0, selfrep_guard="off")
    a = rw(grp("yor", partial, src=src), group_size=4)
    b = off(grp("yor", partial, src=src), group_size=4)
    assert all(x < y for x, y in zip(a.rewards, b.rewards)) and 0 < a.anchors_penalty[0] < 1.0
    assert a.stats["anchors/repeat/penalised_rate"] == 1.0


def test_reward_boilerplate_warmup_penalty_and_eval_isolation(tmp_path):
    rw = mk(tmp_path, anchors_buffer_min_outputs=12, anchors_buffer_min_prompts=6, anchors_buffer_per_prompt=2, anchors_entity_weight=0.0)
    stock = "ipari ti o wa ni gbogbo itumo ni ile wa loni ati lola"
    rng = random.Random(8)

    def step(k, with_stock=True):
        src = f"English source about {rtext(rng, 4)} {k} {rng.random()}"
        txt = "yor " + rtext(rng, 14) + (" " + stock if with_stock else "")
        return rw(grp("yor", txt, src=src, g=4), group_size=4)

    first = step(0)
    assert first.anchors_penalty == [0.0] * 4 and first.stats["anchors/boiler/ready_rate"] == 0.0     # buffer empty: no penalty
    for k in range(1, 8):
        out = step(k)
    assert rw.anchors.buffer.ready("yor") and rw.anchors.buffer.size("yor")[0] <= 16 and rw.anchors.buffer.size("yor")[1] >= 6
    assert max(out.anchors_penalty) > 0.3 and out.stats["anchors/boiler/penalised_rate"] == 1.0
    clean = step(100, with_stock=False)
    assert clean.anchors_penalty == [0.0] * 4
    # greedy evaluation (group_size 1 / None): never penalised for boilerplate, never fed to the buffer
    before = rw.anchors.buffer.size("yor")
    ev = rw([Sample(ids("yor " + rtext(rng, 14) + " " + stock), "yor", "an evaluation source")], group_size=1)
    assert ev.anchors_penalty == [0.0] and rw.anchors.buffer.size("yor") == before
    # failed samples (no end token) are never observed and never penalised by the anchors
    bad = rw([Sample(TOK.encode("yor no end token here at all", add_special_tokens=False), "yor", SRC)] * 4, group_size=4)
    assert bad.anchors_penalty == [0.0] * 4 and bad.status[0] == "no_end_of_turn"


def test_reward_config_validation(tmp_path):
    with pytest.raises(ValueError, match="calibration_path"):
        TranslationReward(RewardConfig(anchors_enabled=True), FakeScorer(), TOK, STOP)
    with pytest.raises(ValueError, match=">= 0"):
        mk(tmp_path, anchors_repeat_weight=-0.1)
    with pytest.raises(ValueError, match="lid_enabled"):
        mk(tmp_path, anchors_lid_prob_weight=0.5)
    with pytest.raises(ValueError, match="content modes"):
        mk(tmp_path, anchors_modes=["nope"])
    with pytest.raises(FileNotFoundError):
        TranslationReward(RewardConfig(anchors_enabled=True, anchors_calibration_path=str(tmp_path / "missing.json")), FakeScorer(), TOK, STOP)


def _sh(content):
    sh = ContentShaper.from_file(None)
    sh.content = dict(content)
    return sh


# ------------------------------------------------------------------------------------------------ D-59: guarded hook, per-language weight capped at 0.25, wired into the reward
def _mk_lexdir(tmp_path, weight=0.4, validated=True, density_max=0.5):
    lex = {"meta": {"validated": validated, "weight": weight, "density_max": density_max},
           "e": {"farmer": [3.0, 0, ["agbe", 0.9]], "vegetabl": [3.0, 0, ["ewebe", 0.9]], "market": [3.0, 0, ["oja", 0.9]]}}
    (tmp_path / "yor.json").write_text(json.dumps(lex))
    from gemma_mt.rl import lexicon as L

    L.load_lexicon.cache_clear()


def test_lexicon_hook_stuffing_guard_and_per_language_weight(tmp_path):
    src = "The farmers sell vegetables at the market."
    _mk_lexdir(tmp_path, weight=0.4)
    good = "Awon agbe n ta ewebe ni oja."
    stuffed = "agbe ewebe oja agbe ewebe oja agbe ewebe oja"                 # every token is a lexicon candidate: density 1.0
    d = str(tmp_path)
    assert A.lexicon_overlap("yor", src, good, d) > 0.9
    assert A.lexicon_overlap("yor", src, stuffed, d) == 0.0                   # guarded: density 1.0 -> reward 0
    assert A.lexicon_overlap("yor", src, stuffed, d, guard=False) == 1.0      # raw overlap is gameable
    assert A.lexicon_language_weight("yor", d, 0.25) == 0.25                  # min(cap, 0.4)
    assert A.lexicon_language_weight("yor", d, 0.5) == 0.4
    assert A.lexicon_language_weight("xxx", d, 0.25) == 0.0                   # no lexicon
    an = A.Anchors(cal(), detectors=("lexicon",), lexicon_dir=d, lexicon_per_language=True)
    assert an.lexicon_weight_factor("yor") == 0.25 and an.lexicon_weight_factor("hau") == 0.0
    assert A.Anchors(cal(), detectors=("lexicon",), lexicon_dir=d).lexicon_weight_factor("hau") == 1.0
    _mk_lexdir(tmp_path, weight=0.4, validated=False)                         # validated=false: off whatever the weight says
    assert A.lexicon_overlap("yor", src, good, d) is None and A.lexicon_language_weight("yor", d) == 0.0
    from gemma_mt.rl import lexicon as L

    L.load_lexicon.cache_clear()
