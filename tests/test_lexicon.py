# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dictionary-overlap signal (rl/lexicon.py, scripts/build_lexicons.py, D-57): normalisation, matching, the overlap / coverage numbers, the file contract
(`lexicon_available`, `lexicon_overlap`) and a tiny synthetic corpus in which the builder must recover the planted dictionary."""
import importlib.util
import json
import os
import random

import pytest

from gemma_mt.rl import lexicon as L

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------------------------------------ normalisation
def test_en_stem_maps_inflections_to_one_key():
    assert L.en_stem("walked") == L.en_stem("walking") == L.en_stem("walks") == L.en_stem("walk")
    assert L.en_stem("houses") == L.en_stem("house")
    assert L.en_stem("studies") == L.en_stem("study") or L.en_stem("studies") == "study"
    assert L.en_stem("bus") == "bus" and L.en_stem("class") == "class"       # no stripping of -us / -ss


def test_content_stems_skip_function_words_names_and_numbers():
    s = "On Monday, the farmers of Kaduna sold 250 bags of rice to Mr Smith at the market."
    st = L.english_content_stems(s)
    assert {"farmer", "bag", "market", L.en_stem("rice")} <= set(st)
    assert "the" not in st and "kaduna" not in st and "smith" not in st and "monday" not in st and "250" not in st
    assert st == list(dict.fromkeys(st))                         # unique, order of first occurrence


def test_x_tokens_and_fold_keep_letters_drop_edge_punctuation():
    assert L.x_tokens("“Ẹ kú àárọ̀,” ni ó wí!") == ["ẹ", "kú", "àárọ̀", "ni", "ó", "wí"]
    assert L.x_folded_tokens("Ẹ kú àárọ̀") == ["ẹ", "ku", "aarọ"]        # tone marks folded, the dot below is part of the letter
    assert L.x_folded_tokens("n’ga 12 ...") == ["n'ga"]            # apostrophe folded and kept inside, tokens without a letter dropped
    assert L.fold("Mɛ̃ɖe") == L.fold("MƐ̃ɖe") and L.fold("ọ") != L.fold("o") and L.fold("o\u0329") == L.fold("ọ")


def test_token_matches_modes():
    assert L.token_matches("mntu", "mntu") and not L.token_matches("mntu", "mntuko")           # exact only
    assert L.token_matches("bantu", "abantu", "affix")                                          # prefix agglutination
    assert L.token_matches("bantu", "bantuwo", "affix")                                         # suffix
    assert not L.token_matches("ba", "abantu", "affix")                                         # too short a stem
    assert not L.token_matches("bantu", "abantuxxxxxxxxx", "affix")                             # the stem is a small part of the token


# ------------------------------------------------------------------------------------------------ the overlap number
def make_lex(validated=True, match="exact"):
    ents = {"water": (5.0, 0, (("maji", 0.9), ("mto", 0.3))), "farmer": (4.0, 0, (("mkulima", 0.8),)), L.en_stem("rice"): (4.0, 0, (("mchele", 0.7),)),
            "lord": (6.0, 1, (("bwana", 0.9),)), "market": (3.0, 0, (("soko", 0.8),))}
    return L.Lexicon("zzz", ents, {"validated": validated, "match": match, "weight": 0.2})


def test_overlap_and_coverage_numbers():
    lex = make_lex()
    src = "The farmers sell rice and water at the market in Kaduna."
    full = lex.overlap(src, "Wakulima huuza mchele na maji sokoni.")       # content: farmer, sell, rice, water, market (Kaduna is a name)
    assert full.n_content == 5 and full.n_covered == 4 and full.n_hit == 2          # exact mode: 'mchele' and 'maji' (no 'mkulima', 'soko' only as 'sokoni')
    assert full.overlap == pytest.approx((4 + 5) / (4 + 4 + 5 + 3))
    r = lex.overlap(src, "maji mkulima mchele soko")
    assert r.overlap == 1.0 and r.n_hit == r.n_covered == 4
    half = lex.overlap(src, "maji mkulima")
    assert 0 < half.overlap < 1
    none = lex.overlap(src, "habari za asubuhi")
    assert none.overlap == 0.0
    assert r.coverage == pytest.approx(r.n_covered / r.n_content) and 0 < r.coverage < 1


def test_overlap_none_when_the_lexicon_covers_too_little():
    lex = make_lex()
    r = lex.overlap("Nothing is certain about that decision.", "habari")
    assert r.overlap is None and r.coverage == 0.0
    r = lex.overlap("The water is cold.", "maji")                    # one covered word < MIN_COVERED
    assert r.overlap is None and r.n_covered == 1


def test_overlap_idf_weighting_and_biblical_exclusion():
    lex = make_lex()
    src = "The lord drank water and rice."
    a = lex.overlap(src, "maji")                                      # water (idf 5) hit, lord (6) and rice (4) missed
    assert a.overlap == pytest.approx(5 / 15)
    assert lex.overlap(src, "maji", weighting="uniform").overlap == pytest.approx(1 / 3)
    b = lex.overlap(src, "maji", exclude_biblical=True)
    assert b.overlap == pytest.approx(5 / 9) and b.n_covered == 2


def test_affix_mode_and_topk():
    lex = make_lex()
    src = "The farmers carry water and rice."
    assert lex.overlap(src, "wamakulima mchelewo majimaji", match="exact").overlap == 0.0
    assert lex.overlap(src, "mkulimawa mchelewo", match="affix").n_hit == 2
    assert lex.overlap("The water and rice.", "mto mchele", topk=1).n_hit == 1      # second water candidate ignored with topk=1
    assert lex.overlap("The water and rice.", "mto mchele").n_hit == 2


def test_diacritics_are_folded_on_both_sides():
    ents = {"night": (5.0, 0, (("alé", 0.8),)), L.en_stem("house"): (5.0, 0, (("ilé", 0.8),))}
    lex = L.Lexicon("zzz", {k: (v[0], v[1], tuple((L.fold(c), s) for c, s in v[2])) for k, v in ents.items()}, {})
    assert lex.overlap("The night and the house.", "Ale ati ile").n_hit == 2
    assert lex.overlap("The night and the house.", "alé àti ilé").n_hit == 2


# ------------------------------------------------------------------------------------------------ the file contract (hook for rl/anchors.py)
def write_lexicon(dirpath, iso="zzz", validated=True, weight=0.25):
    obj = {"meta": {"iso": iso, "validated": validated, "weight": weight, "match": "exact"},
           "e": {"water": [5.0, 0, ["maji", 0.9], ["mto", 0.3]], L.en_stem("rice"): [4.0, 0, ["mchele", 0.7]], "farmer": [4.0, 1, ["mkulima", 0.8]]}}
    with open(os.path.join(dirpath, f"{iso}.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f)


def test_files_and_hook_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMMA_MT_LEXICON_DIR", str(tmp_path))
    L.load_lexicon.cache_clear()
    write_lexicon(tmp_path, "zzz", validated=True, weight=0.25)
    write_lexicon(tmp_path, "yyy", validated=False)
    assert L.lexicon_available("zzz") and not L.lexicon_available("yyy") and L.lexicon_available("yyy", validated_only=False)
    assert not L.lexicon_available("nolang")
    assert L.lexicon_weight("zzz") == 0.25 and L.lexicon_weight("yyy") == 0.0 and L.lexicon_weight("nolang") == 0.0
    r = L.lexicon_overlap("zzz", "Water and rice for the farmers.", "maji na mchele")
    assert r.overlap is not None and r.n_covered == 3 and r.n_hit == 2
    r0 = L.lexicon_overlap("nolang", "Water and rice for the farmers.", "maji")
    assert r0.overlap is None and r0.coverage == 0.0 and r0.n_content == 3
    L.load_lexicon.cache_clear()


def test_shipped_lexicons_are_well_formed():
    d = os.path.join(ROOT, "lexicon")
    if not os.path.isdir(d):
        pytest.skip("no lexicon directory")
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        obj = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        lex = L.parse_lexicon(fn[:-5], obj)
        assert len(lex) > 0 and lex.meta["iso"] == fn[:-5]
        for k, (idf, bib, cands) in list(lex.entries.items())[:200]:
            assert k == L.en_stem(k) or k.isalpha()
            assert 1 <= len(cands) <= 3 and all(c == L.fold(c) and 0 < s <= 1 for c, s in cands)
        if lex.validated:
            assert 0 < lex.meta["weight"] <= 1 and lex.meta["match"] in ("exact", "affix")


# ------------------------------------------------------------------------------------------------ the builder on a planted dictionary
def load_builder():
    pytest.importorskip("numpy")
    spec = importlib.util.spec_from_file_location("build_lexicons", os.path.join(ROOT, "scripts", "build_lexicons.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def planted_corpus(n=900, seed=3):
    """English sentences from a 40-word vocabulary; the 'language' translates word for word (planted dictionary), reorders the words, adds a particle
    'ka' for every English function word and an inflection suffix 'wa' to some nouns. Half of the sentences are 'religious'."""
    rng = random.Random(seed)
    nouns = ["house", "river", "farmer", "market", "goat", "mountain", "teacher", "bread", "school", "village", "road", "child"]
    verbs = ["build", "cross", "sell", "buy", "carry", "watch", "teach", "eat"]
    adjs = ["large", "small", "old", "red"]
    rel_words = ["prayer", "blessing", "temple"]
    vocab = nouns + verbs + adjs + rel_words
    dic = {w: "zq" + w[:2] + w[-2:] + str(i % 7) for i, w in enumerate(vocab)}
    en, xs, rel = [], [], []
    for _ in range(n):
        r = rng.random() < 0.5
        w = [rng.choice(adjs), rng.choice(nouns), rng.choice(verbs), rng.choice(nouns)]
        if r:
            w.append(rng.choice(rel_words))
        e = "The %s %s will %s the %s%s." % (w[0], w[1], w[2], w[3], (" with " + w[4]) if r else "")
        x = [dic[v] for v in w]
        if rng.random() < 0.3:
            x[1] += "wa"
        rng.shuffle(x)
        xs.append("ka " + " ".join(x) + " ka")
        en.append(e)
        rel.append(r)
    return en, xs, rel, dic


def test_builder_recovers_the_planted_dictionary():
    B = load_builder()
    en, xs, rel, dic = planted_corpus()
    rw = [L.en_stem(w) for w in ("prayer", "blessing", "temple")]
    stats = {"n_gen": 2000, "n_rel": 2000, "df_gen": {w: 1 for w in rw}, "df_rel": {w: 900 for w in rw}}
    obj = B.train_from_pairs("zzz", en, xs, rel, 0.3, 6, stats, idf_min=1.0)
    ent = obj["e"]
    right = 0
    checked = 0
    for w, f in dic.items():
        k = L.en_stem(w)
        if k not in ent:
            continue
        checked += 1
        right += ent[k][2][0] == L.fold(f) or ent[k][2][0][0] == L.fold(f)
    assert checked >= 20 and right / checked >= 0.9
    assert "the" not in ent and "will" not in ent                          # function words are not content entries
    assert all(ent[L.en_stem(w)][1] == 1 for w in ("prayer", "blessing", "temple") if L.en_stem(w) in ent)    # biblical flag from the religious/general ratio
    assert all(ent[L.en_stem(w)][1] == 0 for w in ("house", "river") if L.en_stem(w) in ent)
    # the lexicon works as a signal: the planted translation scores far above the translation of another sentence
    lex = L.parse_lexicon("zzz", obj)
    good = [lex.overlap(e, x, match="affix").overlap for e, x in zip(en[:100], xs[:100])]
    bad = [lex.overlap(e, xs[(i + 1) % 100], match="affix").overlap for i, e in enumerate(en[:100])]
    assert sum(g is not None for g in good) > 90 and sum(good) / len(good) > 0.9
    assert sum(b or 0 for b in bad) / len(bad) < 0.5


def test_ibm1_without_enough_evidence_gives_no_entries():
    B = load_builder()
    en = ["The cat sat.", "A dog ran."] * 5
    xs = ["paka ameketi", "mbwa alikimbia"] * 5
    obj = B.train_from_pairs("zzz", en, xs, [False] * 10, 0.3, 4, {"n_gen": 100, "n_rel": 0, "df_gen": {}, "df_rel": {}})
    assert obj["e"] == {} or all(v[1] == 0 for v in obj["e"].values())            # MIN_DF_E = 5 sentences: at most the two repeated nouns


# ------------------------------------------------------------------------------------------------ stuffing guard and the validation record
def test_density_and_stuffing_guard():
    ents = {"water": (5.0, 0, (("maji", 0.9),)), "farmer": (4.0, 0, (("mkulima", 0.8),)), L.en_stem("rice"): (4.0, 0, (("mchele", 0.7),))}
    lex = L.Lexicon("zzz", ents, {"validated": True, "match": "exact", "density_max": 0.5})
    src = "The farmer sells rice and water."
    human = lex.overlap(src, "mkulima huuza mchele na maji safi kabisa kwa watu wote")        # 3 of 10 tokens are lexicon candidates
    assert human.overlap == 1.0 and human.density == pytest.approx(0.3)
    assert lex.guarded(human) == pytest.approx(1.0)
    stuffed = lex.overlap(src, "maji mkulima mchele maji mkulima mchele")
    assert stuffed.overlap == 1.0 and stuffed.density == 1.0 and lex.guarded(stuffed) == 0.0
    mid = lex.overlap(src, "maji mkulima mchele a b")                                         # density 0.6 > 0.5: scaled by 1 - 0.1/0.5
    assert lex.guarded(mid) == pytest.approx(0.8)
    assert lex.guarded(lex.overlap("Nothing certain here.", "maji")) is None
    assert lex.overlap(src, "").density is None


def test_lexicon_reward_hook(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMMA_MT_LEXICON_DIR", str(tmp_path))
    L.load_lexicon.cache_clear()
    write_lexicon(tmp_path, "zzz", validated=True)
    assert L.lexicon_reward("nolang", "Water and rice.", "maji") is None
    assert L.lexicon_reward("zzz", "Water and rice for the farmers.", "maji na mchele kwa wakulima wengi sana") == pytest.approx(2 / 3 * 1.0, abs=0.35)
    assert L.lexicon_reward("zzz", "Water and rice for the farmers.", "maji mchele mkulima mto") == 0.0      # density 1.0 -> fully discounted
    L.load_lexicon.cache_clear()


def test_finalize_writes_the_validation_record(tmp_path):
    B = load_builder()
    lexdir = tmp_path / "lexicon"
    lexdir.mkdir()
    for iso in ("aaa", "bbb", "ccc", "ddd"):
        write_lexicon(lexdir, iso, validated=False, weight=0.0)
    flo = {"auc_vs_mismatched": 0.85, "coverage": 0.6, "auc_vs_copy": 0.9, "auc_vs_half_dropped": 0.7, "share_scored": 0.9}
    val = {"aaa": {"flores": {"affix": flo}, "density_max": 0.4, "model": {"affix": {"auc_own_vs_other_source_output": 0.9}}},
           "bbb": {"flores": {"affix": {**flo, "auc_vs_mismatched": 0.66}}},
           "ccc": {"flores": {"affix": {**flo, "coverage": 0.1}}},
           "ddd": {}}
    vp = tmp_path / "val.json"
    vp.write_text(json.dumps(val))
    B.main(["finalize", "--lexicon", str(lexdir), "--validation", str(vp)])
    got = {iso: json.load(open(lexdir / f"{iso}.json"))["meta"] for iso in val}
    assert got["aaa"]["validated"] and got["aaa"]["match"] == "affix" and got["aaa"]["density_max"] == 0.4
    assert got["aaa"]["weight"] == B.proposed_weight(0.85, 0.6) == pytest.approx(0.25 * 0.75, abs=0.01)
    assert not got["bbb"]["validated"] and got["bbb"]["weight"] == 0.0 and "auc" in got["bbb"]["validation"]["reason"]
    assert not got["ccc"]["validated"] and "coverage" in got["ccc"]["validation"]["reason"]
    assert not got["ddd"]["validated"] and got["ddd"]["validation"]["status"] == "not_validated"
