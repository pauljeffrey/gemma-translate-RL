# SPDX-License-Identifier: AGPL-3.0-or-later
import itertools
import json
import os

import numpy as np
import pytest

from gemma_mt.afri2afri import contam, filters, pivot, policy, provenance, text, windows


# ------------------------------------------------------------------------------------------------ text
def test_pivot_key_ignores_case_punctuation_and_unicode_form():
    assert text.pivot_key("Hello,  World!") == text.pivot_key("hello world")
    assert text.pivot_key("Café") == text.pivot_key("Café")              # NFC
    assert text.pivot_key("a_b") == "a b"


def test_pivot_key_keeps_combining_marks_and_unifies_the_yoruba_underdot():
    word = "ọ́"                                                              # o + dot below + acute: no precomposed form
    assert text.pivot_key(word + "kọ") == text.nfc(word + "kọ").casefold()           # nothing was split or dropped
    assert text.pivot_key("ẹ̩") == text.pivot_key("ẹ̣")                    # U+0329 (MAFAND) == U+0323 (FLORES+)


def test_hash64_is_stable_and_shingles_need_n_words():
    assert text.hash64("abc") == text.hash64("abc") != text.hash64("abd")
    assert text.shingles("one two three", 8) == []
    assert len(text.shingles("a b c d e f g h i", 8)) == 2


def test_split_fraction_is_deterministic_and_uniform_enough():
    f = [text.split_key_fraction(str(i)) for i in range(2000)]
    assert f == [text.split_key_fraction(str(i)) for i in range(2000)]
    assert 0.45 < float(np.mean(f)) < 0.55 and 0.0 <= min(f) and max(f) < 1.0
    assert text.split_key_fraction("x", seed=1) != text.split_key_fraction("x", seed=2)


# ------------------------------------------------------------------------------------------------ provenance
def test_provenance_rules():
    c = provenance.classify_repo
    assert c((os.environ['HF_ORG'] + "/tds:train-00000-of-00010")) == "synthetic_mt"
    assert c("oza75/bambara-gec-synthetic") == "synthetic_mt"
    assert c("michsethowusu/english-zulu_sentence-pairs_mt560") == "religious_jw"
    assert c("michsethowusu/african-bible-parallel") == "religious_jw"
    assert c("dsfsi/vukuzenzele-sentence-aligned") == "human_curated"
    assert c("sentence-transformers/parallel-sentences-ccmatrix") == "mixed"
    assert c("some/unknown-repo") == "mixed"
    assert provenance.tier_of("dsfsi/x") == 3 and provenance.tier_of((os.environ['HF_ORG'] + "/tds")) == 0
    assert provenance.pair_tier(3, 1) == 1


def test_licence_combination_takes_the_stricter():
    assert provenance.combine_licences("cc0", "cc-by") == "cc-by"
    assert provenance.combine_licences("cc-by", "share_alike") == "share_alike"
    assert provenance.combine_licences("weird", "cc0") == "other"


# ------------------------------------------------------------------------------------------------ pivot join
def _random_table(seed, n_keys=60, n_lang=7):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_keys):
        for l in rng.permutation(n_lang)[: int(rng.integers(1, n_lang + 1))]:
            for _ in range(int(rng.integers(1, 3))):                              # duplicates of (key, language)
                rows.append((k + 1, int(l), float(rng.random()), int(rng.integers(0, 4))))
    a = np.array(rows)
    return a[:, 0].astype(np.uint64), a[:, 1].astype(np.uint8), a[:, 2], a[:, 3].astype(np.uint8)


def test_pair_count_tensor_matches_brute_force():
    key, iso, score, tier = _random_table(3)
    best = pivot.best_per_key_iso(key, iso, score)
    k, i, t = key[best], iso[best], tier[best]
    assert len(set(zip(k.tolist(), i.tolist()))) == len(k)                          # one row per (key, language)
    starts, sizes = pivot.group_bounds(k)
    got = pivot.pair_count_tensor(i, t, starts, sizes, 7)
    want = np.zeros_like(got)
    for s, z in zip(starts, sizes):
        for p, q in itertools.permutations(range(s, s + z), 2):
            want[min(t[p], t[q]), i[p], i[q]] += 1
    assert (got == want).all()
    assert got.sum() == int((sizes * (sizes - 1)).sum())


def test_best_per_key_iso_keeps_the_highest_score():
    key = np.array([1, 1, 1, 2], np.uint64)
    iso = np.array([5, 5, 6, 5], np.uint8)
    score = np.array([0.1, 0.9, 0.5, 0.2])
    best = pivot.best_per_key_iso(key, iso, score)
    assert sorted(best.tolist()) == [1, 2, 3]


def test_enumerate_pairs_counts_and_uniqueness():
    starts, sizes = np.array([0, 4, 6]), np.array([4, 2, 3])
    g, a, b = pivot.enumerate_pairs(starts, sizes)
    assert len(a) == 6 + 1 + 3
    assert len({(x, y) for x, y in zip(a.tolist(), b.tolist())}) == len(a)
    assert (a < b).all() and (g[a < 4] == 0).all()


def test_pick_by_quota_respects_quota_and_is_random_order_based():
    seg = np.array([0, 0, 0, 1, 1, 2])
    rand = np.array([0.3, 0.1, 0.2, 0.5, 0.4, 0.9])
    keep = pivot.pick_by_quota(seg, rand, np.array([2, 1, 0]))
    assert keep.tolist() == [False, True, True, False, True, False]                 # the smallest `rand` of each segment


# ------------------------------------------------------------------------------------------------ policy
def test_unimax_spends_the_budget_and_caps_epochs():
    sizes = [10, 100, 1000, 100000]
    a = policy.unimax(sizes, 5000, 3.0)
    assert abs(a.sum() - 5000) < 1e-6
    assert a[0] == 30 and a[1] == 300                                               # small items: max_epochs x size
    assert np.allclose(a[2:], 2335)                                                 # the rest shared equally
    big = policy.unimax(sizes, 10 ** 9, 3.0)
    assert np.allclose(big, 3.0 * np.array(sizes))                                  # budget larger than 3 epochs of everything
    assert (policy.unimax([5, 5], 4, 3.0) == [2, 2]).all()


def test_compose_caps_then_relaxes():
    t = policy.compose({"human_curated": 10, "religious_jw": 500, "mixed": 0, "synthetic_mt": 500}, 100)
    assert sum(t.values()) == 100 and t["human_curated"] == 10 and t["synthetic_mt"] == 15 and t["religious_jw"] == 75
    only_syn = policy.compose({"synthetic_mt": 40}, 100)
    assert only_syn["synthetic_mt"] == 40                                           # nothing else exists: the cap is relaxed, shortfall stays
    assert sum(policy.compose({"mixed": 5}, 100).values()) == 5


# ------------------------------------------------------------------------------------------------ filters
def test_lid_verdict():
    cal = {"yor": {"lid": {"enabled": True, "labels": ["yor_Latn"], "min_chars": 30}},
           "pcm": {"lid": {"enabled": False, "labels": ["pcm_Latn"], "min_chars": 30}}}
    pol = filters.lid_policy(cal, fallback={"hau": ("hau_Latn",)})
    assert filters.lid_verdict("yor", 80, "yor_Latn", pol) == filters.LID_OK
    assert filters.lid_verdict("yor", 80, "hau_Latn", pol) == filters.LID_BAD
    assert filters.lid_verdict("yor", 10, "hau_Latn", pol) == filters.LID_SKIP         # too short to judge
    assert filters.lid_verdict("pcm", 80, "eng_Latn", pol) == filters.LID_SKIP         # LID not reliable for this language
    assert filters.lid_verdict("hau", 80, "hau_Latn", pol) == filters.LID_OK           # fallback labels
    assert filters.lid_verdict("zzz", 80, "x", pol) == filters.LID_SKIP


def test_length_ratio_window_uses_the_language_scale():
    enlen = np.array([100] * 400 + [100] * 400)
    tlen = np.array([120] * 400 + [60] * 400)                                         # language 1 writes 0.5x as many characters as language 0
    iso = np.array([0] * 400 + [1] * 400)
    cpe = filters.chars_per_english_char(enlen, tlen, iso, 2)
    assert abs(cpe[0] - 1.2) < 1e-9 and abs(cpe[1] - 0.6) < 1e-9
    e = filters.expected_log_ratio(cpe, 0, 1)
    assert filters.ratio_ok(120, 60, e) and not filters.ratio_ok(120, 400, e)
    assert not filters.ratio_ok(0, 5, 0.0)


def test_copy_script_and_basic_filters():
    assert filters.is_copy("Ndewo, Chioma!", "ndewo chioma")
    assert not filters.is_copy("Ndewo nne", "Sannu uwargida")
    assert filters.basic_ok("Sannu da zuwa") and not filters.basic_ok("see http://x.com now") and not filters.basic_ok("12 34 56 78 90 11")
    assert filters.script_ok("ሰላም ለዓለም እንዴት ነህ", "amh") and not filters.script_ok("Hello world how are you", "amh")
    assert filters.script_ok("Hello world how are you", "yor") and not filters.script_ok("ሰላም ለዓለም እንዴት ነህ", "yor")


# ------------------------------------------------------------------------------------------------ contamination
def _write(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def test_leak_index_exact_and_ngram(tmp_path):
    li = contam.LeakIndex()
    li.add("The quick brown fox jumps over the lazy dog near the river bank.", "flores")
    li.add("Yes", "gatitos")
    assert li.is_leak("the quick brown fox jumps over the lazy dog near the river bank") == "exact"       # case / punctuation insensitive
    long_text = "Prefix words here. The quick brown fox jumps over the lazy dog near the river bank. And a suffix."
    assert li.is_leak(long_text) == "ngram"
    assert li.is_leak("Yes") is None                                                  # one-word benchmark entries never delete training text
    assert li.is_leak("A completely different sentence about markets and rain.") is None


def test_iter_benchmark_texts_reads_every_format(tmp_path):
    r = str(tmp_path)
    _write(f"{r}/flores_plus/devtest/yor_Latn.jsonl", [json.dumps({"id": 0, "text": "ẹ kaaro o"})])
    _write(f"{r}/ntrex/NTREX-128/newstest2019-ref.hau.txt", ["Ina kwana", ""])
    _write(f"{r}/mafand/en-yor/test.json", [json.dumps({"translation": {"en": "Good morning", "yor": "Ẹ kaaro"}})])
    _write(f"{r}/smol_sent/en_yo.jsonl", [json.dumps({"src": "Hello there friend", "trg": "Bawo ni ọrẹ"})])
    _write(f"{r}/gatitos/en_ee.jsonl", [json.dumps({"src": "yes", "trgs": ["Ao", "Eoo"]})])
    _write(f"{r}/smol_doc/en_yo.jsonl", [json.dumps({"srcs": ["Doc one sentence."], "trgs": ["Gbolohun kan."]})])
    _write(f"{r}/extra/bench_dev.jsonl", [json.dumps({"english": "Dev english line", "reference": "Dev ref line", "iso": "yor"})])
    _write(f"{r}/extra/docs.jsonl", [json.dumps({"text": "A whole long document text here.", "sents": ["First document sentence of the benchmark.", "Second one."]})])
    got = list(contam.iter_benchmark_texts(r, [f"{r}/extra/bench_dev.jsonl", f"{r}/extra/docs.jsonl"]))
    texts = {t for _, _, t in got}
    assert {"ẹ kaaro o", "Ina kwana", "Good morning", "Ẹ kaaro", "Hello there friend", "Bawo ni ọrẹ", "yes", "Ao", "Eoo", "Doc one sentence.",
            "Dev english line", "Dev ref line", "First document sentence of the benchmark."} <= texts
    assert {b for b, _, _ in got} >= {"flores_plus", "ntrex", "mafand", "smol_sent", "gatitos", "smol_doc"}


# ------------------------------------------------------------------------------------------------ windows
def test_windows_keep_both_sides_in_the_same_order_and_balance_bins():
    rng = np.random.default_rng(0)
    n = 400
    src = [f"Source sentence number {i}." for i in range(n)]
    tgt = [f"Target sentence number {i}." for i in range(n)]
    ls = [8] * n
    lt = [9] * n
    w = windows.build_windows(src, tgt, ls, lt, rng)
    assert sum(x["n_pairs"] for x in w) == n
    for x in w:                                                                       # sentence i of one side translates sentence i of the other
        a = [s.split()[-1] for s in x["src"].replace(".", ". ").split(". ") if s.strip()]
        b = [t.split()[-1] for t in x["tgt"].replace(".", ". ").split(". ") if t.strip()]
        assert a == b
    assert {x["bin"] for x in w} >= {"b64", "b256"}


def test_windows_one_sided_terminator_stays_standalone_and_missing_stop_is_added():
    rng = np.random.default_rng(1)
    w = windows.build_windows(["Ends here.", "No end", "Both none"], ["Ends here", "No end.", "Both none"], [3, 3, 3], [3, 3, 3], rng)
    assert [x["n_pairs"] for x in w] == [1, 1, 1] or sum(x["n_pairs"] for x in w) == 3
    assert any(x["src"] == "Ends here." and x["tgt"] == "Ends here" for x in w)
    assert any(x["punct_added"] for x in w)


# ------------------------------------------------------------------------------------------------ RL prompts
def test_rl_windows_are_whole_sentences_in_their_bin():
    from gemma_mt.afri2afri import rl_prompts as R

    rng = np.random.default_rng(0)
    sents = [f"Sentence number {i} says something rather ordinary about the market today." for i in range(120)]
    count = lambda texts: [int(len(t) / 4) for t in texts]
    ws = []
    for _ in range(30):
        ws += R.windows_from_doc(sents, rng, count)
    assert ws and all(R.BINS[w["length_bin"]][0] <= w["src_afr_tokens"] < R.BINS[w["length_bin"]][1] for w in ws)
    for w in ws:
        assert w["source"].endswith(".") and w["n_sentences"] == w["source"].count("Sentence number")
    assert {w["length_bin"] for w in ws} >= {"b64", "b256"}


def test_clean_doc_drops_headings_and_markup():
    from gemma_mt.afri2afri import rl_prompts as R

    doc = "Title\n\nÀwọn àkóónú\nÓ se pàtàkì láti mọ díẹ̀ nípa ìtàn ilẹ̀ yìí àti irú ènìyàn tí ń gbé ibẹ̀. Ìyẹn dára gan-an ni.\nSee http://x.org for more words in this line please."
    s = R.clean_doc(doc)
    assert s and all("http" not in x for x in s) and all(len(x) >= 12 for x in s)


def test_assign_targets_is_balanced_distinct_and_capped():
    from gemma_mt.afri2afri import rl_prompts as R

    rng = np.random.default_rng(1)
    targets = [f"t{i}" for i in range(10)]
    out = R.assign_targets(50, targets, {t: 20 for t in targets}, rng)
    assert all(len(set(v.tolist())) == len(v) == 20 for v in out.values())          # distinct rows per target
    use = np.bincount(np.concatenate(list(out.values())), minlength=50)
    assert use.max() - use.min() <= 1                                               # every window is used equally often
    small = R.assign_targets(5, targets, {t: 20 for t in targets}, rng, max_reuse=4)
    assert sum(len(v) for v in small.values()) <= 4 * 5 + len(targets)                # tiny pool: quotas scaled down, no window serves everyone


def test_direction_weights_are_symmetric_clipped_and_mean_one():
    from gemma_mt.afri2afri import rl_prompts as R

    w = R.direction_weights({"a-b": 100.0, "a-c": 10.0, "b-c": 1000.0}, ["a", "b", "c"], lo=0.25, hi=2.0)
    assert w[("a", "b")] == w[("b", "a")] and w[("a", "c")] < w[("a", "b")] < w[("b", "c")] <= 2.0
    assert R.direction_weights({}, ["a", "b"])[("a", "b")] == 0.25
