# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-44: the long-document translation evaluation (gemma_mt/eval/longdoc.py): sentence-by-sentence translation helper, chunk plan on explicit sentence lists
(and its agreement with rl/chunk_plan.plan_chunks), chrF++/BLEU sufficient statistics + document bootstrap, loop / length metrics, chunked AfriCOMET with a
fake scorer, tiers, the whole pipeline and the CLI. CPU only, no network, no model."""
import json
import random

import pytest

from gemma_mt.eval import longdoc as L
from gemma_mt.rl.chunk_plan import plan_chunks
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.segments import LengthModel, split_sentences

sacrebleu = pytest.importorskip("sacrebleu")

WORDS = ("amani ilé ọmọ nkọ̀ wanda abin kasa omu ndi bụ mara ayo ewe adu zaa tiyi baba kitabu mji mtu watoto nyumba maji chakula "
         "shule kazi safari habari nchi").split()


def _sentence(rng, n_words=None):
    n = n_words or rng.randint(6, 16)
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def _english(rng, n):
    return [("The " + " ".join(rng.choice("river market teacher doctor village season road mother farmer ship".split()) + " " + rng.choice(
        "walked saw built carried closed opened lost found sold watched".split()) for _ in range(rng.randint(1, 3))) + " today in the old town again.")
        for _ in range(n)]


def _doc(rng, n_sent=12, iso="yor", doc_id="d1", ref_kind="pseudo", tgt_scale=1.2):
    src = _english(rng, n_sent)
    ref = [_sentence(rng, max(5, int(len(s.split()) * tgt_scale))) for s in src]
    text = " ".join(src)
    sents = L.canonical_sentences(text)
    assert len(sents) == len(src)
    return L.RefDoc(doc_id, iso, text, sents, ref[: len(sents)], ref_kind, "stub", "news", 512)


# ------------------------------------------------------------------------------------------------ text helpers
def test_canonical_sentences_are_stable_and_paragraphs_are_mapped():
    text = "First sentence here. Second one follows!\n\nA new paragraph starts. It has two sentences.\n\nLast one."
    sents = L.canonical_sentences(text)
    assert sents == ["First sentence here.", "Second one follows!", "A new paragraph starts.", "It has two sentences.", "Last one."]
    assert L.sentences_are_stable(sents)
    assert L.paragraph_starts(text, sents) == [0, 2, 4]
    assert L.join_sentences(sents, [0, 2, 4]) == "First sentence here. Second one follows!\n\nA new paragraph starts. It has two sentences.\n\nLast one."
    assert L.join_sentences(sents) == " ".join(sents)
    # a paragraph without a final stop merges with the next sentence: the paragraph map is refused, the sentence list stays valid
    t2 = "A heading without a stop\n\nThen text. More text."
    s2 = L.canonical_sentences(t2)
    assert L.sentences_are_stable(s2)
    assert L.paragraph_starts(t2, s2)[0] == 0


def test_split_long_sentence_keeps_every_word_and_respects_the_limit():
    long = ("alpha beta gamma, " * 80).strip()
    pieces = L.split_long_sentence(long, 300)
    assert len(pieces) >= 4 and all(len(p) <= 300 for p in pieces)
    assert " ".join(pieces).split() == long.split()
    assert L.split_long_sentence("short one.", 300) == ["short one."]
    nospace = "x" * 2000
    assert "".join(L.split_long_sentence(nospace, 300)) == nospace


# ------------------------------------------------------------------------------------------------ translate sentence by sentence, then join
def test_translate_documents_records_alignment_and_dedupes():
    calls = []

    def fn(batch):
        calls.append(list(batch))
        return ["T(" + s + ")" for s in batch]

    docs = [{"doc_id": "a", "text": "One two three. Four five six.\n\nSeven eight nine."}, {"doc_id": "b", "text": "One two three. Brand new sentence here."}]
    out = L.translate_documents(docs, fn, iso="yor", system="stub", batch_size=2)
    assert [d.doc_id for d in out] == ["a", "b"]
    a, b = out
    assert len(a.src_sents) == len(a.tgt_sents) == 3 and a.tgt_sents[0] == "T(One two three.)"
    assert a.para_starts == [0, 2] and a.text == "T(One two three.) T(Four five six.)\n\nT(Seven eight nine.)"
    assert b.tgt_sents[0] == a.tgt_sents[0]
    flat = [s for c in calls for s in c]
    assert sorted(flat) == sorted(set(flat)) and len(flat) == 4          # "One two three." translated once although two documents contain it
    assert all(len(c) <= 2 for c in calls)
    assert a.flagged_share == 0.0 and a.to_ref_row()["ref_sents"] == a.tgt_sents


def test_translate_documents_uses_the_documents_own_alignment_units():
    docs = [{"doc_id": "h", "text": "Dr. A said no. Then left. Fine.", "sents": ["Dr. A said no. Then left.", "Fine."], "para_starts": [0]}]
    out = L.translate_documents(docs, lambda b: ["T:" + s for s in b], iso="yor", system="s")
    assert out[0].src_sents == ["Dr. A said no. Then left.", "Fine."] and out[0].tgt_sents == ["T:Dr. A said no. Then left.", "T:Fine."]


def test_translate_documents_cuts_long_sentences_and_rejoins_them():
    long = ("the old market of the town opened early, " * 12).strip() + "."
    seen = []

    def fn(batch):
        seen.extend(batch)
        return [s.upper() for s in batch]

    out = L.translate_documents([{"doc_id": "x", "text": long + " Short one."}], fn, iso="hau", system="stub", max_sentence_chars=120)
    assert len(out[0].src_sents) == 2 and len(out[0].tgt_sents) == 2        # the alignment unit is still the sentence
    assert max(len(s) for s in seen) <= 120 and len(seen) > 2
    assert out[0].tgt_sents[0] == " ".join(long.upper().split())


def test_translate_documents_flags_and_retries():
    def fn(batch):
        res = []
        for s in batch:
            if "bad1" in s:
                res.append("")                                       # empty
            elif "bad2" in s:
                res.append(s)                                        # copy of the English
            elif "bad3" in s:
                res.append("la " * 30)                               # loop
            else:
                res.append("ok " + s.lower() + " xx")
        return res

    docs = [{"doc_id": "d", "text": "This is bad1 for sure. This is bad2 for sure. This is bad3 for sure. This one is fine and long enough."}]
    d = L.translate_documents(docs, fn, iso="yor", system="s")[0]
    assert d.flags == ["empty", "copy", "loop", ""] and abs(d.flagged_share - 0.75) < 1e-9
    d2 = L.translate_documents(docs, fn, iso="yor", system="s", retry_fn=lambda ss: ["fixed " + s + " zz" for s in ss])[0]
    assert d2.flags == ["", "", "", ""] and d2.tgt_sents[0].startswith("fixed")
    with pytest.raises(RuntimeError):
        L.translate_documents(docs, lambda b: b[:-1], iso="yor", system="s")


# ------------------------------------------------------------------------------------------------ chrF++ / BLEU statistics and bootstrap
def test_stat_rows_sum_to_the_corpus_score():
    rng = random.Random(1)
    refs = [" ".join(_sentence(rng) for _ in range(5)) for _ in range(8)]
    hyps = [r if i % 3 == 0 else " ".join(r.split()[: len(r.split()) - 4]) for i, r in enumerate(refs)]
    cs, bs = L.doc_stat_vectors(hyps, refs)
    chrf, bleu = L.scores_from_stats(cs, bs)
    from sacrebleu.metrics import BLEU, CHRF

    assert abs(chrf - CHRF(word_order=2).corpus_score(hyps, [refs]).score) < 1e-6
    assert abs(bleu - BLEU(tokenize="13a").corpus_score(hyps, [refs]).score) < 1e-6
    sub = [0, 2, 5]
    c2, b2 = L.scores_from_stats(cs, bs, sub)
    assert abs(c2 - CHRF(word_order=2).corpus_score([hyps[i] for i in sub], [[refs[i] for i in sub]]).score) < 1e-6
    # whitespace and NFC differences do not matter
    h2 = [("  " + h.replace(" ", "  ") + "\n") for h in hyps]
    cs2, bs2 = L.doc_stat_vectors(h2, refs)
    assert abs(L.scores_from_stats(cs2, bs2)[0] - chrf) < 1e-9


def test_bootstrap_is_deterministic_and_brackets_the_point():
    rng = random.Random(2)
    refs = [" ".join(_sentence(rng) for _ in range(6)) for _ in range(30)]
    hyps = [r if rng.random() < 0.5 else " ".join(_sentence(rng) for _ in range(6)) for r in refs]
    cs, bs = L.doc_stat_vectors(hyps, refs)
    b1 = L.bootstrap_scores(cs, bs, n_boot=200, seed=3)
    b2 = L.bootstrap_scores(cs, bs, n_boot=200, seed=3)
    assert b1 == b2 and b1["n_docs"] == 30
    assert b1["chrf"]["lo"] <= b1["chrf"]["value"] <= b1["chrf"]["hi"] and b1["chrf"]["se"] > 0
    st = L.bootstrap_scores(cs, bs, n_boot=50, seed=3, strata=[i % 3 for i in range(30)])
    assert st["chrf"]["lo"] <= st["chrf"]["hi"]
    one = L.bootstrap_scores(cs[:1], bs[:1], n_boot=10)
    assert one["chrf"]["lo"] != one["chrf"]["lo"]                          # a single document has no interval (NaN)
    pb = L.paired_bootstrap((cs, bs), (cs, bs), n_boot=50, seed=1)
    assert pb["chrf"]["delta"] == 0.0 and pb["chrf"]["lo"] == 0.0 == pb["chrf"]["hi"]
    worse = L.doc_stat_vectors(["x y z"] * 30, refs)
    pw = L.paired_bootstrap((cs, bs), worse, n_boot=50, seed=1)
    assert pw["chrf"]["delta"] > 10 and pw["chrf"]["p_a_better"] == 1.0
    m = L.bootstrap_mean([1.0, 2.0, 3.0, None, float("nan")], n_boot=100, seed=0)
    assert m["value"] == 2.0 and m["n"] == 3 and m["lo"] <= 2.0 <= m["hi"]
    lo, hi = L.wilson_interval(5, 20)
    assert 0.08 < lo < 0.25 < hi < 0.5 and L.wilson_interval(0, 0)[0] != L.wilson_interval(0, 0)[0]


# ------------------------------------------------------------------------------------------------ failure metrics
def test_repetition_loop_detects_degenerate_output_only():
    ok = "Mo fẹ́ lọ sí ọjà lónìí láti ra ẹja àti ìrẹsì fún ìdílé mi, ṣùgbọ́n òjò bẹ̀rẹ̀ sí rọ̀ ní kété tí mo jáde nílé."
    assert not L.repetition_loop(ok)["loop"]
    loop = ok + " muna muna muna muna muna muna muna muna muna muna"
    r = L.repetition_loop(loop)
    assert r["loop"] and r["period"] == 1 and r["at_tail"] and r["covered_share"] > 0.25
    unit = ok + " " + "ni ọjọ́ kan ṣoṣo " * 6
    r2 = L.repetition_loop(unit)
    assert r2["loop"] and r2["period"] == 4 and r2["repeats"] >= 5
    assert not L.repetition_loop("ha ha ha ha")["loop"]                    # too short to be a loop
    assert not L.repetition_loop("")["loop"]


def test_length_stats_and_clean_end():
    s = L.length_stats([1.0, 1.0, 0.5, 2.0])
    assert s["n"] == 4 and abs(s["share_outside"] - 0.5) < 1e-9 and s["median"] in (1.0,)
    assert L.ends_cleanly("A sentence.") and L.ends_cleanly("He said “no.”") and not L.ends_cleanly("A sentence that stops mid") and not L.ends_cleanly("")


# ------------------------------------------------------------------------------------------------ chunk plan
def test_plan_document_identical_hypothesis_is_fully_covered():
    d = _doc(random.Random(3), 10)
    lm = L.make_length_model(d.src_sents, d.ref_sents)
    p = L.plan_document(d.src_sents, d.ref_sents, " ".join(d.ref_sents), lm)
    assert p.how == "aligned" and p.coverage_chars == 1.0 and p.coverage_sents == 1.0 and p.failed_w == 0.0 and len(p.segments) == 10
    assert [r for _, _, r in p.segments] == d.ref_sents
    p0 = L.plan_document(d.src_sents, d.ref_sents, "", lm)
    assert p0.coverage_chars == 0.0 and p0.omitted_sents == 10 and p0.segments == [] and p0.items == [(None, p0.total_w)]


def test_plan_document_finds_dropped_sentences_and_copies():
    rng = random.Random(4)
    d = _doc(rng, 14, tgt_scale=1.3)
    lm = L.make_length_model(d.src_sents, d.ref_sents)
    keep = [i for i in range(14) if i not in (3, 8)]
    p = L.plan_document(d.src_sents, d.ref_sents, " ".join(d.ref_sents[i] for i in keep), lm)
    assert p.how == "gale_church" and p.omitted_sents == 2 and p.n_hyp == 12
    assert 0.0 < p.coverage_chars < 1.0 and abs(p.coverage_sents - 12 / 14) < 1e-9
    # untranslated English left in the middle is a failed chunk, not a scored one
    hy = list(d.ref_sents)
    hy[5] = d.src_sents[5]
    pc = L.plan_document(d.src_sents, d.ref_sents, " ".join(hy), lm)
    assert pc.copied_sents == 1 and pc.omitted_sents == 0 and pc.coverage_chars < 1.0
    # extra text appended at the end is accounted for on the output side
    pe = L.plan_document(d.src_sents, d.ref_sents, " ".join(d.ref_sents) + " " + " ".join(_sentence(rng) for _ in range(8)), lm)
    assert pe.extra_w > 0 and pe.coverage_chars == 1.0


def test_plan_document_agrees_with_plan_chunks_on_stable_documents():
    """The explicit-list plan makes the SAME decisions as rl/chunk_plan.plan_chunks (D-30 settings: language length model + shortfall 0.7)."""
    rng = random.Random(5)
    n_checked = 0
    for trial in range(40):
        n = rng.randint(2, 18)
        d = _doc(rng, n, tgt_scale=rng.choice([0.9, 1.2, 1.5]))
        lm = L.make_length_model(d.src_sents, d.ref_sents)
        hy = list(d.ref_sents)
        mode = trial % 5
        if mode == 1 and n > 3:
            del hy[rng.randrange(n)]
        elif mode == 2 and n > 4:
            for i in sorted(rng.sample(range(n), 2), reverse=True):
                del hy[i]
        elif mode == 3 and n > 3:
            i = rng.randrange(n - 1)
            hy[i:i + 2] = [hy[i] + " " + hy[i + 1][:12].rstrip(".") + "."]
        elif mode == 4:
            hy = [h.split(".")[0] + "." for h in hy[: max(1, n - 2)]] + [_sentence(rng)]
        text = " ".join(hy)
        if not text.strip():
            continue
        mine = L.plan_document(d.src_sents, d.ref_sents, text, lm, short_frac=0.7, short_min=0.15)
        theirs = plan_chunks(" ".join(d.src_sents), text, length_model=lm, short_frac=0.7, short_min=0.15)
        assert [(a, b) for a, b, _ in mine.segments] == theirs.segments, (trial, mode)
        none_w = sum(w for k, w in theirs.items if k is None)
        assert abs(none_w - (mine.failed_w + mine.extra_w)) < 1e-6, (trial, mode, none_w, mine.failed_w, mine.extra_w)
        assert [(k, round(w, 6)) for k, w in mine.items if k is not None] == [(k, round(w, 6)) for k, w in theirs.items if k is not None]
        assert mine.how == theirs.how and mine.short_beads == theirs.n_short
        n_checked += 1
    assert n_checked >= 30


# ------------------------------------------------------------------------------------------------ AfriCOMET over chunks (fake scorer)
def test_comet_over_plans_one_batched_call_and_penalty():
    rng = random.Random(6)
    d1, d2 = _doc(rng, 6, doc_id="a"), _doc(rng, 6, doc_id="b")
    lm = L.make_length_model(d1.src_sents + d2.src_sents, d1.ref_sents + d2.ref_sents)
    p1 = L.plan_document(d1.src_sents, d1.ref_sents, " ".join(d1.ref_sents), lm)
    p2 = L.plan_document(d2.src_sents, d2.ref_sents, " ".join(d2.ref_sents[:4]), lm)        # last two sentences missing
    calls = []

    def scorer(srcs, mts, refs):
        calls.append((len(srcs), len(mts), len(refs)))
        assert all(r for r in refs)
        return [0.8] * len(srcs)

    out = L.comet_over_plans([p1, p2], scorer, reference_based=True, penalty=0.0)
    assert calls == [(len(p1.segments) + len(p2.segments),) * 3] and out["n_segments"] == len(p1.segments) + len(p2.segments)
    assert abs(out["doc"][0] - 0.8) < 1e-9
    assert out["doc"][1] < 0.8 and abs(out["covered"][1] - 0.8) < 1e-9                    # the omitted chunks count `penalty`, `covered` leaves them out
    out2 = L.comet_over_plans([p2], scorer, reference_based=True, penalty=0.5)
    assert out2["doc"][0] > out["doc"][1]
    qe = L.comet_over_plans([p1], lambda s, m: [0.5] * len(s), reference_based=False)
    assert abs(qe["doc"][0] - 0.5) < 1e-9
    assert L.comet_over_plans([], scorer, reference_based=True)["n_segments"] == 0


# ------------------------------------------------------------------------------------------------ one language, one run
def _set(n_docs=6, iso="yor", seed=7):
    rng = random.Random(seed)
    docs = [_doc(rng, 10 + i, iso=iso, doc_id=f"d{i}") for i in range(n_docs)]
    for i, d in enumerate(docs):
        d.bin = 512 if i % 2 == 0 else 1024
        d.domain = "news" if i % 3 else "health"
    return docs


def test_evaluate_language_perfect_loop_and_missing():
    docs = _set(6)
    hyps = {d.doc_id: L.Hyp(d.doc_id, "yor", d.ref_text, True, "sys") for d in docs}
    res = L.evaluate_language(docs, hyps, iso="yor", scorer_ref=FakeScorer(), scorer_qe=FakeScorer(), n_boot=50)
    assert res["n_docs"] == 6 and res["n_missing"] == 0
    assert res["chrf"]["value"] > 99.9 and res["bleu"]["value"] > 99.9
    assert res["coverage_chars"]["value"] == 1.0 and res["loop_rate"]["value"] == 0.0 and res["finished_rate"]["value"] == 1.0
    assert abs(res["length_ratio"]["mean"] - 1.0) < 1e-6 and res["africomet_ref"]["n"] == 6 and res["africomet_qe"]["n"] == 6
    assert set(res["by_bin"]) == {"512", "1024"} and set(res["by_domain"]) == {"news", "health"} and len(res["rows"]) == 6
    # a loop at the end of two documents, one document not translated at all, one that did not finish
    hyps2 = dict(hyps)
    for i in (0, 1):
        hyps2[docs[i].doc_id] = L.Hyp(docs[i].doc_id, "yor", docs[i].ref_text + " " + "muna " * 40, False, "sys")
    del hyps2[docs[5].doc_id]
    r2 = L.evaluate_language(docs, hyps2, iso="yor", scorer_ref=FakeScorer(), n_boot=50)
    assert r2["n_missing"] == 1 and abs(r2["loop_rate"]["value"] - 2 / 6) < 1e-9 and r2["finished_rate"]["n"] == 5
    assert r2["finished_rate"]["value"] == 2 / 5 * 0 + 3 / 5
    assert r2["chrf"]["value"] < res["chrf"]["value"] and r2["coverage_chars"]["value"] < 1.0
    assert r2["rows"][5]["coverage_chars"] == 0.0 and r2["rows"][5]["doc_chrf"] == 0.0
    assert L.evaluate_language(docs, hyps, iso="hau")["n_docs"] == 0


def test_evaluate_language_omissions_show_up():
    docs = _set(4)
    hyps = {}
    for d in docs:
        keep = [s for i, s in enumerate(d.ref_sents) if i % 4 != 2]            # every 4th sentence dropped
        hyps[d.doc_id] = L.Hyp(d.doc_id, "yor", " ".join(keep), True)
    res = L.evaluate_language(docs, hyps, iso="yor", n_boot=0)
    assert 0.6 < res["coverage_chars"]["value"] < 0.95
    assert res["omitted_sents_total"] >= 3 and res["docs_with_omission"]["value"] == 1.0


def test_tiers_and_macro_and_markdown():
    assert L.assign_tier({"system": "madlad", "chrf": 52.0, "africomet": 0.72, "n": 997})["tier"] == "A"
    assert L.assign_tier({"system": "madlad", "chrf": 52.0, "africomet": 0.55})["tier"] == "B"            # the AfriCOMET side is too low for A
    assert L.assign_tier({"system": "madlad", "chrf": 52.0})["tier"] == "A"                              # language without an AfriCOMET score
    assert L.assign_tier({"system": "nllb", "chrf": 31.0, "africomet": 0.6})["tier"] == "B"
    assert L.assign_tier({"system": "nllb", "chrf": 22.0, "africomet": 0.6})["tier"] == "C"
    assert L.assign_tier(None)["tier"] == "N" and L.assign_tier({"system": "x", "chrf": None})["tier"] == "N"
    lb = L.assign_tier({"system": "nllb", "chrf": 46.0, "chrf_lo": 43.0}, use_lower_bound=True)
    assert lb["tier"] == "B"
    docs_y, docs_h = _set(4, "yor"), _set(4, "hau", seed=8)
    refdocs = docs_y + docs_h
    hyps = {(d.iso, d.doc_id): L.Hyp(d.doc_id, d.iso, d.ref_text, True) for d in refdocs}
    run = L.evaluate_run(refdocs, hyps, tiers={"yor": {"tier": "A"}, "hau": {"tier": "C"}}, scorer_ref=FakeScorer(), n_boot=20)
    assert run["pseudo"]["yor"]["ranked"] is True and run["pseudo"]["hau"]["ranked"] is False and run["pseudo"]["hau"]["tier"] == "C"
    assert run["macro"]["pseudo_ranked"]["chrf"]["n_languages"] == 1 and run["macro"]["pseudo_all"]["chrf"]["n_languages"] == 2
    md = L.report_markdown(run, {"yor": "Yoruba"})
    assert "Yoruba (`yor`)" in md and "(unranked)" in md and "macro averages" in md
    # human references are scored in their own group
    hd = _doc(random.Random(9), 8, iso="yor", doc_id="h1", ref_kind="human")
    hyps[("yor", "h1")] = L.Hyp("h1", "yor", hd.ref_text, True)
    run2 = L.evaluate_run(refdocs + [hd], hyps, n_boot=10)
    assert run2["human"]["yor"]["n_docs"] == 1 and run2["pseudo"]["yor"]["n_docs"] == 4


def test_two_pseudo_reference_systems_are_scored_separately_not_double_counted():
    docs_a = _set(4, "yor")
    docs_b = []
    for d in docs_a:                                                   # a second translator: the same documents with slightly different references
        docs_b.append(L.RefDoc(d.doc_id, "yor", d.source, d.src_sents, [r.replace("a", "e", 1) for r in d.ref_sents], "pseudo", "sysB", d.domain, d.bin))
    for d in docs_a:
        d.ref_system = "sysA"
    hyps = {("yor", d.doc_id): L.Hyp(d.doc_id, "yor", d.ref_text, True) for d in docs_a}
    run = L.evaluate_run(docs_a + docs_b, hyps, tiers={"yor": {"tier": "A", "system": "sysB"}}, n_boot=10)
    assert run["pseudo"]["yor"]["n_docs"] == 4 and run["pseudo"]["yor"]["ref_system"] == "sysB"                  # the tiers file picks the primary
    assert run["pseudo_alt"]["yor"]["sysA"]["n_docs"] == 4 and run["pseudo_alt"]["yor"]["sysA"]["chrf"]["value"] > 99.9
    assert run["pseudo"]["yor"]["chrf"]["value"] < 100.0
    run2 = L.evaluate_run(docs_a + docs_b, hyps, n_boot=10)
    assert run2["pseudo"]["yor"]["ref_system"] == "sysA"                                                          # default: first by name


def test_refdoc_validation_and_loading(tmp_path):
    with pytest.raises(ValueError):
        L.RefDoc("x", "yor", "A b. C d.", ["A b.", "C d."], ["only one"])
    with pytest.raises(ValueError):
        L.RefDoc("x", "yor", "A b.", ["A b."], ["x"], ref_kind="weird")
    rng = random.Random(10)
    d = _doc(rng, 5, doc_id="q")
    docs_p, refs_p, hyp_p = tmp_path / "docs.jsonl", tmp_path / "refs_yor.jsonl", tmp_path / "preds.jsonl"
    L.write_jsonl(str(docs_p), [{"doc_id": "q", "text": d.source, "sents": d.src_sents, "domain": "news", "bin": 512, "para_starts": [0]}])
    L.write_jsonl(str(refs_p), [{"doc_id": "q", "iso": "yor", "ref_sents": d.ref_sents, "ref_system": "stub", "flags": [""] * 4 + ["empty"]},
                                {"doc_id": "missing", "iso": "yor", "ref_sents": ["x"]}])
    L.write_jsonl(str(hyp_p), [{"doc_id": "q", "iso": "yor", "hyp": d.ref_text, "finished": True, "system": "s"}])
    rds = L.load_refdocs(str(docs_p), str(refs_p))
    assert len(rds) == 1 and rds[0].domain == "news" and rds[0].bin == 512 and abs(rds[0].flagged_share - 0.2) < 1e-9
    assert L.load_refdocs(str(docs_p), str(refs_p), only_valid=True) == []
    h = L.load_hyps(str(hyp_p))
    assert h[("yor", "q")].finished is True and h[("yor", "q")].text == d.ref_text


def test_cli_score_with_fake_scorers(tmp_path):
    rng = random.Random(11)
    docs = [_doc(rng, 8, doc_id=f"c{i}") for i in range(3)]
    L.write_jsonl(str(tmp_path / "docs.jsonl"), [{"doc_id": d.doc_id, "text": d.source, "sents": d.src_sents, "domain": "news", "bin": 512} for d in docs])
    L.write_jsonl(str(tmp_path / "refs.jsonl"), [{"doc_id": d.doc_id, "iso": "yor", "ref_sents": d.ref_sents, "ref_system": "stub"} for d in docs])
    L.write_jsonl(str(tmp_path / "preds.jsonl"), [{"doc_id": d.doc_id, "iso": "yor", "hyp": d.ref_text, "finished": True} for d in docs])
    (tmp_path / "tiers.json").write_text(json.dumps({"tiers": {"yor": {"tier": "B"}}}))
    rc = L.main(["score", "--docs", str(tmp_path / "docs.jsonl"), "--refs", str(tmp_path / "refs.jsonl"), "--preds", str(tmp_path / "preds.jsonl"),
                 "--tiers", str(tmp_path / "tiers.json"), "--out", str(tmp_path / "out"), "--comet-ref", "fake", "--comet-qe", "fake", "--n-boot", "10"])
    assert rc == 0
    res = json.loads((tmp_path / "out" / "results.json").read_text())
    assert res["pseudo"]["yor"]["tier"] == "B" and res["pseudo"]["yor"]["chrf"]["value"] > 99.0 and "africomet_ref" in res["pseudo"]["yor"]
    assert "yor" in (tmp_path / "out" / "report.md").read_text()


def test_bins():
    assert L.bin_of(500) == 512 and L.bin_of(1000) == 1024 and L.bin_of(2000) == 2048 and L.bin_of(4000) == 4096
    assert L.bin_of(700) == 0 and L.bin_of(100) == 0 and L.bin_of(9000) == 0
    assert all(lo < hi for lo, hi in L.BIN_RANGES.values())
    rs = sorted(L.BIN_RANGES.values())
    assert all(rs[i][1] < rs[i + 1][0] for i in range(len(rs) - 1))


# ------------------------------------------------------------------------------------------------ the linear-time splitter equals the RL reward's splitter
def test_fast_split_sentences_equals_the_reward_splitter_on_fuzz_texts():
    from gemma_mt.rl.segments import split_sentences as ref_split

    rng = random.Random(7)
    pieces = ["Dr.", "Mr.", "St.", "J.", "U.S.", "e.g.", "i.e.", "etc.", "1.", "12.", "123.", "1234.", "3.5", "www.example.org", "a.b", "It", "was", "the", "river", "and", "Then", "he", "saw",
              "“Hello.”", "\"No!\"", "(yes.)", "[end].", "ok?", "wow!", "so...", "፣", "።", "እንደው", "በተትቢ", "ni", "Mo", "ọ̀", "ṣ", "...", "?!", "A", "b", "17", ":", ";",
              "figs.", "no.", "vol.", "Jan.", "Fig.", "p.m.", "Inc.", "x."]
    for _ in range(1500):
        n = rng.randint(1, 60)
        text = " ".join(rng.choice(pieces) for _ in range(n))
        if rng.random() < 0.2:
            text = text.replace(" ", rng.choice(["  ", "\n", " \n\n ", "\t"]), rng.randint(1, 5))
        assert L.split_sentences(text) == ref_split(text), text
    for text in ["", "   ", "One.", "One. Two.", "Don't stop. Really.", "Version 2.0 is out. See www.example.org. Fine.", "He said \"stop.\" Then left.",
                 "A. B. C. D.", "Chapter 1. Intro. 2. Next item. 3. Last one."]:
        assert L.split_sentences(text) == ref_split(text), text
    # linear time: a 60,000-character text is split in well under a second (the original needs ~0.3 s per 40,000 characters and grows quadratically)
    import time

    big = " ".join(["The river was calm today and the boats left the harbour early."] * 1000)
    t0 = time.time()
    assert len(L.split_sentences(big)) == 1000
    assert time.time() - t0 < 2.0
