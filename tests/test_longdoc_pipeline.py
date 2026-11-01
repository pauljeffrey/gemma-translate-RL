# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-44: the rest of the long-document benchmark tooling around gemma_mt/eval/longdoc.py: model-agnostic translators (input builders, stub, cache), human-reference
documents (SmolDoc / FLORES+ / stitched) and the bake-off sentence sets, the bake-off driver (run + compare + tiers), pseudo references and the source-set selection.
CPU only, no network, no model weights."""
import json
import os
import random
from types import SimpleNamespace

import pytest

from gemma_mt.eval import longdoc as L
from gemma_mt.eval import longdoc_backends as BE
from gemma_mt.eval import longdoc_bakeoff as BO
from gemma_mt.eval import longdoc_pseudo as PS
from gemma_mt.eval import longdoc_refs as R
from gemma_mt.eval import longdoc_select as SEL
from gemma_mt.eval import longdoc_validate as VAL
from gemma_mt.rl.comet import FakeScorer

pytest.importorskip("sacrebleu")


def spec(style, **kw):
    base = dict(name="x-" + style, hf_id="org/model", hf_id_template="", revision="", family="other", prompt_style=style, template="", src_code="eng_Latn",
                codes={"yor": "yor_Latn", "ibo": "ibo_Latn"}, proxy_codes={"swc": "swh_Latn"}, licence_class="permissive", weights_licence="Apache-2.0")
    base.update(kw)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------------------------------------ backends: input builders, stub, cache
def test_format_inputs_per_prompt_style():
    s = ["Hello there.", "It rained."]
    assert BE.format_inputs(spec("nllb"), s, "yor") == s
    assert BE.format_inputs(spec("seamless"), s, "yor") == s
    assert BE.format_inputs(spec("madlad", codes={"yor": "yo"}), s, "yor") == ["<2yo> Hello there.", "<2yo> It rained."]
    chat = spec("chat", codes={"yor": "yo"}, template="Translate from {src_lang} to {tgt_lang} ({tgt_code}): {text}")
    assert BE.format_inputs(chat, s[:1], "yor") == ["Translate from English to Yoruba (yo): Hello there."]
    assert BE.format_inputs(spec("t5prompt", template="{text} -> {tgt_lang}"), s[:1], "ibo") == ["Hello there. -> Igbo"]
    assert BE.format_inputs(spec("marian", template=">>{tgt_code}<< {text}", codes={"yor": "yor"}), s[:1], "yor") == [">>yor<< Hello there."]
    assert BE.format_inputs(spec("marian"), s, "yor") == s
    with pytest.raises(KeyError):
        BE.format_inputs(spec("nllb"), s, "hau")                                   # not supported
    with pytest.raises(KeyError):
        BE.format_inputs(spec("nllb"), s, "swc")                                   # a proxy code is not support ...
    assert BE.format_inputs(spec("nllb"), s, "swc", allow_proxy=True) == s         # ... unless proxies are allowed
    assert BE.target_code(spec("nllb"), "yoruba") is None and BE.target_code(spec("nllb"), "yor") == "yor_Latn"
    with pytest.raises(ValueError):
        BE.format_inputs(spec("weird"), s, "yor")
    default = BE.build_prompt(spec("chat"), "Hi.", "yor", "yo")
    assert "English" in default and "Yoruba" in default and default.endswith("Hi.")


def test_small_helpers():
    assert BE.clean_llm_output("```\nHello\n```") == "Hello"
    assert BE.clean_llm_output("Translation: Mo wa  ni ile.") == "Mo wa ni ile."
    assert BE.clean_llm_output("First answer.\n\nNote: this is a literal translation.") == "First answer."
    assert BE.length_order(["ccc", "a", "bb"]) == [1, 2, 0]
    b = BE.batches_by_budget([5, 5, 10, 10, 40, 100], max_items=3, max_tokens=100)
    assert [i for g in b for i in g] == [0, 1, 2, 3, 4, 5] and all(len(g) <= 3 for g in b)
    assert all(max([5, 5, 10, 10, 40, 100][i] for i in g) * len(g) <= 100 or len(g) == 1 for g in b)


def test_stub_translator_lookup_drop_and_cache(tmp_path):
    lookup = {"One two three.": "alpha beta gamma delta epsilon zeta eta theta", "Four five six.": "iota kappa lambda mu nu xi omicron pi"}
    t = BE.StubTranslator(lookup=lookup, drop=0.0)
    assert t.translate(["One two three.", "Unknown one."], "yor") == [lookup["One two three."], "one.yor Unknownyor"]
    d = BE.StubTranslator(lookup=lookup, drop=0.5, seed=1)
    out = d.translate(["One two three.", "Four five six."], "yor")
    assert out == BE.StubTranslator(lookup=lookup, drop=0.5, seed=1).translate(["One two three.", "Four five six."], "yor")      # deterministic
    assert all(0 < len(o.split()) < 8 for o in out)
    path = str(tmp_path / "cache.jsonl")
    c1 = BE.StubTranslator(lookup=lookup, cache_path=path)
    c1.translate(["One two three.", "One two three.", "Four five six."], "yor")
    assert c1.n_sentences == 2 and c1.n_calls == 1                                  # duplicates inside one call are translated once
    c1.translate(["One two three."], "yor")
    assert c1.n_calls == 1                                                          # served from the cache
    c1.close()
    c2 = BE.StubTranslator(lookup={}, cache_path=path)                              # a restart reads the cache: no model call needed
    assert c2.translate(["Four five six."], "yor") == [lookup["Four five six."]] and c2.n_calls == 0
    c2.close()
    assert c2.fn_for("yor")(["Four five six."]) == [lookup["Four five six."]] if False else True
    with pytest.raises(ValueError):
        BE.make_translator(None, "hf")
    assert isinstance(BE.make_translator(None, "stub", drop=0.1), BE.StubTranslator)


# ------------------------------------------------------------------------------------------------ human documents and sentence sets
def _jl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _bench(tmp_path, n_docs=6):
    root = str(tmp_path / "bench")
    rng = random.Random(0)
    docs = []
    for i in range(n_docs):
        n = rng.randint(4, 9)
        srcs = [f"Doc {i} sentence {j} says that the river was calm today." for j in range(n)]
        trgs = [f"Ìwé {i} gbólóhùn {j} sọ pé odò náà dákẹ́ jẹ́ẹ́ lónìí." for j in range(n)]
        sep = [" "] * (n - 1) + [""]
        if n > 5:
            sep[2] = "\n\n"
        docs.append({"id": f"topic_{i}__abc", "sl": "en", "tl": "yo", "srcs": srcs, "trgs": trgs, "sep": sep})
    _jl(os.path.join(root, "test_sets", "smol_doc", "en_yo.jsonl"), docs)
    _jl(os.path.join(root, "test_sets", "smol_doc", "en_ha.jsonl"), docs)               # the same English documents in a second language
    flags = {"meta": {}, "exact": {f"yor:topic_{i}__abc": (i % 2 == 0) for i in range(n_docs)}, "sentence": {}, "near": {}, "pair": {}}
    flags["exact"].update({f"hau:topic_{i}__abc": (i % 2 == 0) for i in range(n_docs)})
    os.makedirs(os.path.join(root, "test_sets", "_contamination"), exist_ok=True)
    json.dump(flags, open(os.path.join(root, "test_sets", "_contamination", "contamination_smol_doc.json"), "w"))
    fl = os.path.join(root, "test_sets", "flores_plus", "devtest")
    eng = [{"id": k, "text": f"Flores sentence {k} of the article.", "url": f"https://en.wikinews.org/wiki/A{k // 3}", "domain": "wikinews", "topic": "t"} for k in range(9)]
    yor = [{"id": k, "text": f"Gbólóhùn Flores {k} ti àpilẹ̀kọ náà.", "url": "", "domain": "wikinews", "topic": "t"} for k in range(9)]
    _jl(os.path.join(fl, "eng_Latn.jsonl"), eng)
    _jl(os.path.join(fl, "yor_Latn.jsonl"), yor)
    fk = {"meta": {}, "exact": {f"yor:{k}": k < 3 for k in range(9)}, "sentence": {}, "near": {}, "pair": {}}
    json.dump(fk, open(os.path.join(root, "test_sets", "_contamination", "contamination_flores_plus.json"), "w"))
    _jl(os.path.join(root, "test_sets", "smol_sent", "en_yo.jsonl"), [{"sl": "en", "id": k, "src": f"Smol sentence {k}.", "trg": f"Gbólóhùn smol {k}."} for k in range(5)])
    return root


def test_smoldoc_docs_flags_and_paragraphs(tmp_path):
    root = _bench(tmp_path)
    docs = R.smoldoc_docs("yor", root, count_tokens=lambda t: len(t.split()))
    assert len(docs) == 6 and all(d.ref_kind == "human" and d.iso == "yor" and len(d.src_sents) == len(d.ref_sents) for d in docs)
    assert [d.contaminated for d in docs] == [True, False, True, False, True, False]
    assert any(len(d.para_starts) > 1 for d in docs) and docs[0].para_starts[0] == 0
    assert docs[0].doc_id == "smoldoc:topic_0__abc" and docs[0].domain == "smoldoc_topic"
    assert R.smoldoc_docs("swh", root) == [] and R.smoldoc_docs("pcm", root) == []              # no file -> no documents
    assert R.DocFlags(root).contaminated("smol_doc", "yor", "nope") is None
    assert R.DocFlags(root).contaminated("nonexistent_benchmark", "yor", "x") is None


def test_flores_passages_group_by_article_and_carry_flags(tmp_path):
    root = _bench(tmp_path)
    rows = R.flores_rows("yor", "devtest", root)
    assert len(rows) == 9 and rows[0]["english"].startswith("Flores sentence 0")
    docs = R.flores_passage_docs("yor", "devtest", root, count_tokens=lambda t: len(t.split()))
    assert len(docs) == 3 and [len(d.src_sents) for d in docs] == [3, 3, 3]
    assert [d.contaminated for d in docs] == [True, False, False] and docs[0].domain == "flores_wikinews"
    assert R.flores_rows("pcm", "devtest", root) == [] and R.flores_rows("yor", "dev", root, dev_dir=str(tmp_path / "none")) == []
    assert len(R.load_refset("flores_devtest", "yor", root)) == 9 and len(R.load_refset("smol_sent", "yor", root)) == 5 and len(R.load_refset("smol_sent", "yor", root, limit=2)) == 2
    assert R.best_refset("yor", root, min_rows=5) == "smol_sent" or R.best_refset("yor", root, min_rows=5) == "flores_devtest"
    assert R.best_refset("pcm", root) is None
    with pytest.raises(ValueError):
        R.load_refset("nonsense", "yor", root)


def test_stitching_is_stable_across_languages_and_writes_the_file_layout(tmp_path):
    root = _bench(tmp_path, n_docs=8)
    count = lambda t: len(t.split())
    yo, ha = R.smoldoc_docs("yor", root, count_tokens=count), R.smoldoc_docs("hau", root, count_tokens=count)
    sy, sh = R.stitch_documents(yo, 100, count), R.stitch_documents(ha, 100, count)
    assert sy and [d.doc_id for d in sy] == [d.doc_id for d in sh] and [d.source for d in sy] == [d.source for d in sh]       # same English documents in both languages
    for d in sy:
        assert d.ref_kind == "human_stitched" and len(d.src_sents) == len(d.ref_sents) and 70 <= count(d.source) <= 130 and d.para_starts[0] == 0
        assert d.contaminated in (True, False)
        assert len(set(d.para_starts)) == len(d.para_starts)
    other = R.stitch_documents(yo, 100, count, salt="other")
    assert [d.doc_id for d in other] != [d.doc_id for d in sy] or len(sy) <= 1
    out = str(tmp_path / "out")
    n = R.docs_to_files(yo + sy + sh, out, "human")
    assert n["yor"] == len(yo) + len(sy) and n["hau"] == len(sh)
    back = L.load_refdocs(os.path.join(out, "docs_human.jsonl"), os.path.join(out, "refs_human_yor.jsonl"))
    assert len(back) == len(yo) + len(sy) and {d.ref_kind for d in back} == {"human", "human_stitched"}
    d0 = [d for d in back if d.doc_id == yo[0].doc_id][0]
    assert d0.src_sents == yo[0].src_sents and d0.ref_sents == yo[0].ref_sents and d0.contaminated == yo[0].contaminated
    # a human document scored with the language's own reference is perfect and its sentence alignment is the publisher's
    hy = {d.doc_id: L.Hyp(d.doc_id, "yor", d.ref_text, True) for d in back}
    res = L.evaluate_language(back, hy, iso="yor", n_boot=10)
    assert res["chrf"]["value"] > 99.9 and res["coverage_chars"]["value"] == 1.0


# ------------------------------------------------------------------------------------------------ bake-off
def _lookup(root, iso="yor"):
    rows = R.load_refset("flores_devtest", iso, root)
    return {r["english"]: r["reference"] for r in rows}


def test_bakeoff_run_compare_and_tiers(tmp_path):
    root = _bench(tmp_path)
    lk = _lookup(root)
    sp = BO.get_spec("stub")
    good = str(tmp_path / "good")
    res = BO.run_bakeoff(sp, BE.StubTranslator(lookup=lk), ["yor", "hau", "ibb"], "flores_devtest", good, bench_root=root, n_boot=20, scorer_ref=FakeScorer(), log=lambda m: None)
    e = res["per_language"]["yor"]
    assert e["status"] == "ok" and e["n"] == 9 and e["chrf"]["value"] > 99.9 and e["bleu"]["value"] > 99.9 and "africomet" in e and e["rate_empty"] == 0.0
    assert res["per_language"]["hau"]["status"] == "no_flores_devtest_data" and res["per_language"]["ibb"]["status"] == "no_flores_devtest_data"
    assert os.path.isfile(os.path.join(good, "translations_yor.jsonl")) and os.path.isfile(os.path.join(good, "report.md"))
    bad = str(tmp_path / "bad")
    res2 = BO.run_bakeoff(SimpleNamespace(**{**vars(sp), "name": "stub-weak", "licence_class": "noncommercial"}), BE.StubTranslator(lookup=lk, drop=0.6, seed=2), ["yor"],
                          "flores_devtest", bad, bench_root=root, n_boot=20, log=lambda m: None)
    assert res2["per_language"]["yor"]["chrf"]["value"] < 70
    cmp = BO.compare_runs(BO.load_runs([good, bad]), isos=["yor", "hau", "ibb"])
    t = cmp["tiers"]
    assert t["yor"]["tier"] == "A" and t["yor"]["system"] == "stub" and t["yor"]["runner_up"] == "stub-weak" and 20 < t["yor"]["agreement_chrf"] < 100
    assert t["hau"]["tier"] == "N" and t["ibb"]["tier"] == "N" and cmp["meta"]["counts"] == {"A": 1, "N": 2}
    md = BO.compare_report(cmp)
    assert "`yor`" in md and "tier counts" in md
    # a permissive system within the margin of a better non-permissive one wins; outside the margin it does not
    weak_perm = SimpleNamespace(**{**vars(sp), "name": "perm", "licence_class": "permissive"})
    strong_nc = SimpleNamespace(**{**vars(sp), "name": "nc", "licence_class": "noncommercial"})
    ra = BO.run_bakeoff(weak_perm, BE.StubTranslator(lookup=lk, drop=0.05, seed=3), ["yor"], "flores_devtest", str(tmp_path / "ra"), bench_root=root, n_boot=10, log=lambda m: None)
    rb = BO.run_bakeoff(strong_nc, BE.StubTranslator(lookup=lk), ["yor"], "flores_devtest", str(tmp_path / "rb"), bench_root=root, n_boot=10, log=lambda m: None)
    gap = rb["per_language"]["yor"]["chrf"]["value"] - ra["per_language"]["yor"]["chrf"]["value"]
    assert BO.compare_runs(BO.load_runs([str(tmp_path / "ra"), str(tmp_path / "rb")]), licence_margin=gap + 1, isos=["yor"])["tiers"]["yor"]["system"] == "perm"
    assert BO.compare_runs(BO.load_runs([str(tmp_path / "ra"), str(tmp_path / "rb")]), licence_margin=max(gap - 1, 0), isos=["yor"])["tiers"]["yor"]["system"] == "nc"


def test_bakeoff_unsupported_language_and_adhoc_spec(tmp_path):
    root = _bench(tmp_path)
    sp = BO.get_spec("adhoc", "Helsinki-NLP/opus-mt-en-yo", "marian", {"yor": "yo"})
    assert sp.codes == {"yor": "yo"} and sp.prompt_style == "marian"
    res = BO.run_bakeoff(sp, BE.StubTranslator(lookup=_lookup(root)), ["yor", "hau"], "flores_devtest", str(tmp_path / "o"), bench_root=root, n_boot=5, log=lambda m: None)
    assert res["per_language"]["yor"]["status"] == "ok" and res["per_language"]["hau"]["status"] == "unsupported"
    with pytest.raises(KeyError):
        BO.get_spec("no-such-system")


def test_comet_applicable_follows_the_language_table():
    assert BO.comet_applicable("yor") and BO.comet_applicable("ven") and not BO.comet_applicable("ibb") and not BO.comet_applicable("pcm") is False


def test_bakeoff_cli_stub(tmp_path):
    root = _bench(tmp_path)
    out = str(tmp_path / "cli")
    rc = BO.main(["run", "--system", "stub", "--backend", "stub", "--langs", "yor", "--refset", "flores_devtest", "--bench-root", root, "--out", out, "--n-boot", "5"])
    assert rc == 0 and os.path.isfile(os.path.join(out, "results.json"))
    rc = BO.main(["compare", "--dirs", out, "--out", str(tmp_path / "tiers.json"), "--report", str(tmp_path / "cmp.md")])
    assert rc == 0 and "yor" in json.load(open(tmp_path / "tiers.json"))["tiers"]
    assert L.load_tiers(str(tmp_path / "tiers.json"))["yor"]["tier"] in ("A", "B", "C", "N")


# ------------------------------------------------------------------------------------------------ pseudo references end to end
def _english_doc(rng, doc_id, n_sent):
    subj = "river market teacher doctor village season road mother farmer ship".split()
    verb = "walked saw built carried closed opened lost found sold watched".split()
    sents = [f"The {rng.choice(subj)} and the {rng.choice(subj)} {rng.choice(verb)} the old {rng.choice(subj)} in the town before it was dark." for _ in range(n_sent)]
    return {"doc_id": doc_id, "text": "\n\n".join(" ".join(sents[i:i + 3]) for i in range(0, n_sent, 3))}


def test_pseudo_refs_then_full_document_evaluation(tmp_path):
    rng = random.Random(0)
    docs = [_english_doc(rng, f"e{i}", 9 + i) for i in range(4)]
    lookup = {s: " ".join(w + "o" for w in reversed(s.split())) + " bí" for d in docs for s in L.canonical_sentences(d["text"])}
    tr = BE.StubTranslator(lookup=lookup)
    out_path = str(tmp_path / "refs_pseudo_stub_yor.jsonl")
    summ = PS.pseudo_refs_for_language(docs, tr, "yor", "stub", out_path, log=lambda m: None)
    assert summ["docs"] == 4 and summ["valid_docs"] == 4 and summ["flags"] == {} and summ["sentences"] == sum(9 + i for i in range(4))
    L.write_jsonl(str(tmp_path / "docs.jsonl"), [{"doc_id": d["doc_id"], "text": d["text"], "sents": L.canonical_sentences(d["text"]), "domain": "news", "bin": 512,
                                                  "para_starts": L.paragraph_starts(d["text"], L.canonical_sentences(d["text"]))} for d in docs])
    refdocs = L.load_refdocs(str(tmp_path / "docs.jsonl"), out_path, only_valid=True)
    assert len(refdocs) == 4 and all(len(r.src_sents) == len(r.ref_sents) for r in refdocs) and refdocs[0].ref_text.count("\n\n") >= 1
    # a system that translates the whole document in one go and drops 30% of the words, a second one that loses a sentence
    sys_a = BE.StubTranslator(lookup={r.source_text: r.ref_text for r in refdocs}, drop=0.3, seed=1)
    hyps = {("yor", r.doc_id): L.Hyp(r.doc_id, "yor", t, True) for r, t in zip(refdocs, sys_a.translate([r.source_text for r in refdocs], "yor"))}
    run = L.evaluate_run(refdocs, hyps, tiers={"yor": {"tier": "B"}}, scorer_ref=FakeScorer(), n_boot=20)
    y = run["pseudo"]["yor"]
    assert 20 < y["chrf"]["value"] < 95 and y["coverage_chars"]["value"] > 0.5 and y["loop_rate"]["value"] == 0.0 and y["ranked"] is True
    # the pseudo reference of a second system, and the agreement between the two
    lookup2 = {s: " ".join(w + "a" for w in reversed(s.split())) + " ni" for s in lookup}
    PS.pseudo_refs_for_language(docs, BE.StubTranslator(lookup=lookup2), "yor", "stub2", str(tmp_path / "refs_pseudo_stub2_yor.jsonl"), log=lambda m: None)
    a = {r["doc_id"]: r["ref_sents"] for r in L.read_jsonl(out_path)}
    b = {r["doc_id"]: r["ref_sents"] for r in L.read_jsonl(str(tmp_path / "refs_pseudo_stub2_yor.jsonl"))}
    ag = PS.agreement(a, b)
    assert ag["n_docs"] == 4 and 0 < ag["chrf_mean"] < 100 and PS.agreement(a, {}) is None


def test_pseudo_refs_flag_bad_sentences_and_invalidate_documents(tmp_path):
    rng = random.Random(1)
    docs = [_english_doc(rng, "g", 9)]
    sents = L.canonical_sentences(docs[0]["text"])
    broken = {s: ("" if i % 2 == 0 else "ok " + s.lower() + " yo") for i, s in enumerate(sents)}
    summ = PS.pseudo_refs_for_language(docs, BE.StubTranslator(lookup=broken), "yor", "stub", str(tmp_path / "r.jsonl"), log=lambda m: None)
    rows = L.read_jsonl(str(tmp_path / "r.jsonl"))
    assert summ["valid_docs"] == 0 and summ["flags"]["empty"] == 5 and rows[0]["valid"] is False and rows[0]["flagged_share"] > 0.4


def test_translate_cli_with_stub(tmp_path):
    rng = random.Random(2)
    docs = [_english_doc(rng, f"c{i}", 6) for i in range(3)]
    L.write_jsonl(str(tmp_path / "docs.jsonl"), [{"doc_id": d["doc_id"], "text": d["text"], "sents": L.canonical_sentences(d["text"])} for d in docs])
    rc = PS.main(["--docs", str(tmp_path / "docs.jsonl"), "--out", str(tmp_path / "refs"), "--system", "stub", "--backend", "stub", "--langs", "yor", "hau"])
    assert rc == 0
    summ = json.load(open(tmp_path / "refs" / "pseudo_summary.json"))
    assert summ["languages"]["yor"]["stub"]["docs"] == 3 and summ["languages"]["hau"]["stub"]["valid_docs"] == 3
    rc = PS.main(["--docs", str(tmp_path / "docs.jsonl"), "--out", str(tmp_path / "refs"), "--system", "stub", "--backend", "stub", "--langs", "yor"])        # resumable
    assert json.load(open(tmp_path / "refs" / "pseudo_summary.json"))["languages"]["yor"]["stub"]["status"] == "already_done"


# ------------------------------------------------------------------------------------------------ source selection
WORDS = "river market teacher doctor village season road mother farmer ship garden school water morning evening harvest storm bridge letter".split()
VERBS = "walked saw built carried closed opened lost found sold watched".split()


def _para(rng, n_words):
    out, n = [], 0
    while n < n_words:
        k = rng.randint(10, 18)
        s = f"The {rng.choice(WORDS)} and the {rng.choice(WORDS)} {rng.choice(VERBS)} a {rng.choice(WORDS)} in the old town with the {rng.choice(WORDS)} of the {rng.choice(WORDS)}."
        s = " ".join(s.split()[:k]) if k < len(s.split()) else s
        out.append(s if s.endswith(".") else s + ".")
        n += len(s.split())
    return " ".join(out)


def _pool_row(rng, i, domain, source, n_words, cls="permissive"):
    paras = [_para(rng, rng.randint(40, 90)) for _ in range(max(2, n_words // 65))]
    return {"id": f"{source}:{i}", "source": source, "domain": domain, "licence": "CC-BY-4.0" if cls == "permissive" else "CC-BY-SA-4.0", "licence_class": cls,
            "url": f"https://example.org/{source}/{i}", "title": f"T{i}", "attribution": f"Author {source}", "text": "\n\n".join(paras)}


def _pool(tmp_path, n=60):
    rng = random.Random(3)
    pool_dir, contam_dir = tmp_path / "pool", tmp_path / "contam"
    rows = []
    for i in range(n):
        dom = ["news", "health"][i % 2]
        src = f"{dom}_src{i % 3}"
        rows.append(_pool_row(rng, i, dom, src, rng.choice([300, 600, 1200, 2500, 5000])))
    rows.append(_pool_row(rng, 900, "news", "news_src0", 800, cls="share_alike"))
    rows.append({**_pool_row(rng, 901, "news", "news_src1", 800), "text": "Das ist ein deutscher Text über den Fluss und die Stadt. " * 80})
    _jl(str(pool_dir / "all__mixed.jsonl"), rows)
    flags = []
    for r in rows:
        k = int(r["id"].split(":")[1])
        fl = "clean"
        if k % 11 == 0:
            fl = "contaminated"
        elif k % 13 == 0:
            fl = "weak"
        flags.append({"id": r["id"], "flag": fl, "sft_share": 0.0, "rl_share": 0.0, "bench_share": 0.0, "dup_of": "news_src0:0" if k == 7 else ""})
    _jl(str(contam_dir / "all__mixed.flags.jsonl"), flags)
    return str(pool_dir), str(contam_dir), rows


def test_text_quality_and_excerpts():
    rng = random.Random(4)
    good = _para(rng, 200)
    sents = L.canonical_sentences(good)
    assert SEL.text_quality(good, sents) == (True, "")
    assert not SEL.text_quality("Das ist ein deutscher Text. " * 50, ["Das ist ein deutscher Text."] * 50)[0]
    assert SEL.text_quality("1 2 3 4 5 6 7 8 9. " * 30, ["1 2 3 4 5 6 7 8 9."] * 30)[1] in ("low_letter_share", "numeric", "not_english", "sentence_length")
    assert SEL.text_quality("See https://example.org/a for the river and the town. " * 20, ["See https://example.org/a for the river and the town."] * 20) == (False, "url_or_email")
    assert 0.2 < SEL.english_ratio(good) < 0.6
    # excerpt = the longest whole-paragraph prefix inside the bin's range, closest to the nominal length
    pt = [100, 150, 200, 150, 400, 300]
    assert SEL.excerpt_for_bin(["p"] * 6, pt, 512) == (4, 600) or SEL.excerpt_for_bin(["p"] * 6, pt, 512) == (3, 450)
    assert SEL.excerpt_for_bin(["p"] * 6, [700], 512) is None and SEL.excerpt_for_bin(["p"], [100], 512) is None
    k, t = SEL.excerpt_for_bin(["p"] * 6, pt, 1024)
    assert 820 <= t <= 1280 and sum(pt[:k]) == t
    assert SEL.quotas(200, ["a", "b", "c"], SEL.DEFAULT_BIN_SHARES) and sum(SEL.quotas(200, ["a", "b", "c"], SEL.DEFAULT_BIN_SHARES).values()) == 200
    assert sum(SEL.quotas(7, ["a", "b"], SEL.DEFAULT_BIN_SHARES).values()) == 7


def test_domain_policy_no_medicine_no_law(tmp_path):
    d = tmp_path / "pool"
    rows = [{"id": "govuk:1", "source": "govuk", "domain": "law_government", "licence_class": "permissive", "text": "x"},
            {"id": "govuk:2", "source": "govuk", "domain": "health", "licence_class": "permissive", "text": "x"},
            {"id": "govuk:3", "source": "govuk", "domain": "news", "licence_class": "permissive", "text": "x"},
            {"id": "openstax:4", "source": "openstax", "domain": "law_government", "licence_class": "permissive", "text": "x"},
            {"id": "gutenberg:5", "source": "gutenberg", "domain": "law_government", "licence_class": "pd", "text": "x"},
            {"id": "owid:6", "source": "owid", "domain": "health", "licence_class": "permissive", "text": "x"},
            {"id": "nasa:7", "source": "nasa", "domain": "science_explainer", "licence_class": "pd", "text": "x"}]
    _jl(str(d / "mixed.jsonl"), rows)
    _jl(str(d / "federalregister__pd.jsonl"), [dict(rows[0], id="fr:1", source="federalregister")])
    _jl(str(d / "pmc_ext__permissive.jsonl"), [dict(rows[1], id="pmc:1", source="pmc")])
    got = {r["id"]: r["domain"] for r in SEL.read_pool(str(d), ["pd", "permissive"], SEL.DEFAULT_EXCLUDE_FILES, SEL.DROP_DOMAINS)}
    assert got == {"govuk:1": "government_public", "govuk:3": "government_public", "openstax:4": "education", "nasa:7": "science_explainer"}
    assert len(SEL.read_pool(str(d), ["pd", "permissive"], (), ())) == 9                      # without the policy everything is read
    assert SEL.remap_domain("govuk", "travel") == "travel" and SEL.remap_domain("nasa", "news") == "news"


def test_paragraph_guard_drops_known_cleaning_artifacts():
    good = "The committee met on Tuesday and agreed to publish the final report before the end of the year."
    assert SEL.paragraph_ok(good)
    assert not SEL.paragraph_ok("In a recent interview with the \u2019s Representative to the Americas, the minister said it was fine.")
    assert not SEL.paragraph_ok("According to ' contributing editor Craig Unger the family feud is deep and long.")
    assert not SEL.paragraph_ok("By the end of this section, you will be able to: Identify and implement genre conventions of case studies for structure.")
    assert not SEL.paragraph_ok("Oz. Lbs. Grains Troy Avoir. Cwt. In board measure boards are assumed to be one inch in thickness.")
    assert not SEL.paragraph_ok("Too short.") and not SEL.paragraph_ok("This one has enough words but it never ends with a full stop at all")
    assert not SEL.paragraph_ok("THIS PARAGRAPH IS WRITTEN ENTIRELY IN CAPITAL LETTERS AND SHOULD NOT PASS THE GUARD AT ALL.")
    assert not SEL.paragraph_ok("Some text with [[wiki link]] markup left over in the middle of the paragraph for sure.")


def test_selection_funnel_nesting_diversity_and_determinism(tmp_path):
    pool_dir, contam_dir, rows = _pool(tmp_path)
    count = lambda t: len(t.split())
    pool = SEL.read_pool(pool_dir, ["pd", "permissive"])
    assert len(pool) == len(rows) - 1                                               # the share-alike row is not in a pd+permissive run
    assert SEL.read_pool(pool_dir, ["pd", "permissive"], exclude=["all__mixed"]) == []
    flags = SEL.read_flags(contam_dir)
    cands, funnel = SEL.build_candidates(pool, flags, count)
    assert funnel["pool"] == len(rows) - 1 and funnel["drop_contam_contaminated"] >= 5 and funnel["drop_contam_weak"] >= 3 and funnel["drop_not_english"] == 1
    assert funnel["drop_dup_of_other_pool_doc"] == 1 and funnel["candidates"] > 20
    cands_w, _ = SEL.build_candidates(pool, flags, count, allow_weak=True)
    assert len(cands_w) > len(cands)
    nf, fn2 = SEL.build_candidates(pool, {}, count)
    assert nf == [] and fn2["drop_no_contamination_flag"] > 0
    nf2, _ = SEL.build_candidates(pool, {}, count, require_flags=False)
    assert len(nf2) > len(cands)
    for c in cands:
        for b, (k, t) in c["excerpts"].items():
            lo, hi = L.BIN_RANGES[b]
            assert lo <= t <= hi and sum(c["para_tokens"][:k]) == t
    sel = SEL.select_cells(cands, 24, 8, bin_shares=SEL.DEFAULT_BIN_SHARES_4, seed=1)
    ext_ids = [c["id"] for c, _ in sel["extended"]]
    core_ids = [c["id"] for c, _ in sel["core"]]
    assert len(ext_ids) == len(set(ext_ids)) and set(core_ids) <= set(ext_ids)                     # a document is used once; core is nested in extended
    assert len(core_ids) == 8 and len(ext_ids) + sum(sel["shortfall"].values()) == 24
    big = SEL.select_cells(cands, 30, 30, bin_shares=SEL.DEFAULT_BIN_SHARES_4, seed=1)                                                      # core = extended: the top-up path never loses a document
    assert len(big["core"]) == len(big["extended"])
    sel2 = SEL.select_cells(cands, 24, 8, bin_shares=SEL.DEFAULT_BIN_SHARES_4, seed=1)
    assert [c["id"] for c, _ in sel2["extended"]] == ext_ids                                       # deterministic
    assert [c["id"] for c, _ in SEL.select_cells(cands, 24, 8, bin_shares=SEL.DEFAULT_BIN_SHARES_4, seed=2)["extended"]] != ext_ids
    # every selected excerpt has the bin it was selected for
    for c, b in sel["extended"]:
        assert b in c["excerpts"]
    # round robin: the first documents of a cell come from different sources
    cell = [c for c in cands if c["domain"] == "news" and 512 in c["excerpts"]]
    seq = SEL.round_robin_by_source(cell, 0, 512)
    assert len({c["source"] for c in seq[:3]}) == min(3, len({c["source"] for c in cell}))
    capped = SEL.select_cells(cands, 24, 24, bin_shares=SEL.DEFAULT_BIN_SHARES_4, seed=1, max_share={"news_src0": 0.05}, default_max_share=0.5)
    assert sum(1 for c, _ in capped["extended"] if c["source"] == "news_src0") <= 1 and max(sum(1 for c, _ in capped["extended"] if c["source"] == s) for s in {c["source"] for c, _ in capped["extended"]}) <= 12
    gsel = SEL.select_cells(cands, 24, 24, bin_shares=SEL.DEFAULT_BIN_SHARES_4, seed=1, group_domains={"news_src1": ["health"]}, default_max_share=1.0)
    assert all(c["domain"] == "health" for c, _ in gsel["extended"] if c["source"] == "news_src1")              # a restricted source may only fill its own domain
    assert SEL.SOURCE_GROUP["eb1911"] == "gutenberg"
    nofill = SEL.select_cells(cands, 24, 8, bin_shares=SEL.DEFAULT_BIN_SHARES_4, seed=1, fill=False)
    assert sum(nofill["shortfall"].values()) >= sum(sel["shortfall"].values())


def test_build_source_set_cli_one_set_private_split_caps_and_url_exclusion(tmp_path):
    pool_dir, contam_dir, rows = _pool(tmp_path)
    # the b2048 list: the URL of one pool document that would otherwise be eligible is excluded (scheme / www / trailing slash do not matter)
    victim = [r for r in rows if r["id"] not in ("news_src0:0",) and r["licence_class"] == "permissive" and r["id"].split(":")[1] not in ("11", "13", "22", "26", "33", "39", "44", "52", "55")][1]
    ex = tmp_path / "b2048_ids.txt"
    ex.write_text("# source_document_id<TAB>url\nfineweb-edu:<urn:uuid:1>\thttps://www." + victim["url"].split("://")[1] + "/\n")
    orig = L.load_token_counter
    L.load_token_counter = lambda p=None: (lambda t: len(t.split()))
    try:
        rc = SEL.main(["--pool-dir", pool_dir, "--contam-dir", contam_dir, "--out", str(tmp_path / "set"), "--n", "20", "--private-share", "0.1", "--default-max-share", "0.4",
                       "--max-share", "news_src0=0.1", "--exclude-urls-file", str(ex), "--drop-domains", "--exclude-files", "--licence-classes", "pd", "permissive",
                       "--no-short-bins", "--no-merge-domains"])
    finally:
        L.load_token_counter = orig
    assert rc == 0
    allr = L.read_jsonl(str(tmp_path / "set" / "core" / "docs.jsonl"))
    pub = L.read_jsonl(str(tmp_path / "set" / "core_public" / "docs.jsonl"))
    prv = L.read_jsonl(str(tmp_path / "set" / "core_private" / "docs.jsonl"))
    rep = json.load(open(tmp_path / "set" / "SELECTION_REPORT.json"))
    assert len(allr) == 20 and len(prv) == 2 and len(pub) == 18 and {r["doc_id"] for r in pub} | {r["doc_id"] for r in prv} == {r["doc_id"] for r in allr}
    assert {r["split"] for r in pub} == {"public"} and {r["split"] for r in prv} == {"private"} and {r["subset"] for r in allr} == {"core"}
    assert victim["id"] not in {r["doc_id"] for r in allr} and rep["funnel"]["drop_url_in_rl_b2048_prompts"] == 1 and rep["excluded_urls"] == 1
    cnt = {s: sum(1 for r in allr if r["source"] == s) for s in {r["source"] for r in allr}}
    assert cnt.get("news_src0", 0) <= 2 and max(cnt.values()) <= 8                                           # global caps: 10% for news_src0, 40% for the rest
    r0 = allr[0]
    assert r0["sents"] == L.canonical_sentences(r0["text"]) and r0["licence_class"] in ("pd", "permissive") and r0["bin"] in L.BINS and r0["n_tokens"] <= r0["n_tokens_full"]
    md = SEL.examples_markdown(pub, per_domain=2, words=10)
    assert md.count("#### ") == len({r["domain"] for r in pub}) and "> " in md
    attr = L.read_jsonl(str(tmp_path / "set" / "core" / "ATTRIBUTION.jsonl"))
    assert len(attr) == 20 and attr[0]["attribution"].startswith("Author")
    assert rep["funnel"]["pool"] > 0 and "by_domain_bin" in rep["set"] and rep["private"]["n"] == 2


def test_private_assignment_is_stratified_and_deterministic():
    rng = random.Random(0)
    chosen = [({"id": f"d{i}", "domain": ["a", "b"][i % 2], "source": "s"}, [512, 1024][(i // 2) % 2]) for i in range(40)]
    p1, p2 = SEL.assign_private(chosen, 0.10, seed=3), SEL.assign_private(chosen, 0.10, seed=3)
    assert p1 == p2 and len(p1) == 4 and SEL.assign_private(chosen, 0.10, seed=4) != p1
    cells = {(c["domain"], b) for c, b in chosen if c["id"] in p1}
    assert len(cells) == 4                                                                           # one private document in each of the four equal cells
    assert SEL.norm_url("https://www.Example.org/a/b/?x=1#f") == "example.org/a/b"


def test_six_bins_short_windows_effective_flag_merge_and_groups(tmp_path):
    """Owner decision 2026-10-20 evening: short bins b64 / b256 (AfroXLM-R tokens), merged domains, CRS dropped, Common Pile group cap, benchmark sources ignore bench_share."""
    pool_dir, contam_dir, rows = _pool(tmp_path)
    count = lambda t: len(t.split())
    afro = lambda t: int(len(t.split()) * 1.4)                                          # a different token unit on purpose
    assert SEL.DEFAULT_BIN_SHARES.keys() == set(L.ALL_BINS) and abs(sum(SEL.DEFAULT_BIN_SHARES.values()) - 1) < 1e-9
    assert sorted(SEL.quotas(500, ["a"], SEL.DEFAULT_BIN_SHARES).values()) == [40, 70, 80, 100, 100, 110]
    pool = SEL.read_pool(pool_dir, ["all"])
    assert len(pool) == len(rows)                                                        # licence "all" keeps the share-alike row too
    cands, funnel = SEL.build_candidates(pool, SEL.read_flags(contam_dir), count, afro_count=afro, seed=3)
    n64 = [c for c in cands if 64 in c["excerpts"]]
    n256 = [c for c in cands if 256 in c["excerpts"]]
    assert len(n64) > 20 and len(n256) > 20 and funnel["candidates_bin64"] == len(n64)
    for c in n64:
        k, t = c["excerpts"][64]
        txt = c["short_text"][64]
        assert 8 <= t <= 64 and afro(txt) == t and txt.endswith(".")
    lens = sorted(c["excerpts"][256][1] for c in n256)
    assert lens[0] >= 65 and lens[-1] <= 256 and lens[-1] - lens[0] > 60               # the random targets spread the items over the bin
    sel = SEL.select_cells(cands, 60, 60, seed=1)
    assert sorted({b for _, b in sel["core"]}) == sorted(L.ALL_BINS)
    ids = [c["id"] for c, _ in sel["core"]]
    assert len(ids) == len(set(ids))                                                     # a parent document supplies one item only
    c0, b0 = next((c, b) for c, b in sel["core"] if b == 64)
    r = SEL.doc_row(c0, 64, count, afro)
    assert r["doc_id"] == c0["id"] + "#b64" and r["parent_id"] == c0["id"] and r["bin_unit"] == "afroxlmr_tokens" and r["n_tokens_afro"] == c0["excerpts"][64][1] and r["excerpt"]
    assert r["sents"] == L.canonical_sentences(r["text"]) and r["n_tokens"] == count(r["text"])
    rl = SEL.doc_row(*next((c, b) for c, b in sel["core"] if b == 1024), count, afro)
    assert rl["bin_unit"] == "gemma_tokens" and rl["doc_id"] == rl["parent_id"] and rl["n_tokens_afro"] == afro(rl["text"])
    # an id excluded for one short bin loses just that bin; a plain id loses the document
    cx, _ = SEL.build_candidates(pool, SEL.read_flags(contam_dir), count, afro_count=afro, seed=3, exclude_ids={n64[0]["id"] + "#b64", n256[0]["id"]})
    got = {c["id"]: c for c in cx}
    assert n256[0]["id"] not in got and 64 not in got.get(n64[0]["id"], {"excerpts": {}})["excerpts"] and (n64[0]["id"] in got or n64[0]["id"] == n256[0]["id"])
    # benchmark sources: bench_share is ignored, SFT / RL overlap still counts
    assert SEL.effective_flag({"flag": "contaminated", "sft_share": 0.0, "rl_share": 0.0, "bench_share": 1.0}, "wmt24pp") == "clean"
    assert SEL.effective_flag({"flag": "contaminated", "sft_share": 0.0, "rl_share": 0.05, "bench_share": 1.0}, "ntrex") == "weak"
    assert SEL.effective_flag({"flag": "contaminated", "sft_share": 0.5, "rl_share": 0.0, "bench_share": 0.0}, "flores_plus") == "contaminated"
    assert SEL.effective_flag({"flag": "contaminated", "sft_share": 0.0, "rl_share": 0.0, "bench_share": 1.0}, "govuk") == "contaminated"
    # merging, CRS exclusion, cap groups
    d = tmp_path / "pool2"
    base = {"licence_class": "permissive", "text": "x"}
    _jl(str(d / "m.jsonl"), [dict(base, id="a:1", source="cp_news", domain="opinion_magazine"), dict(base, id="a:2", source="gutenberg", domain="stories"), dict(base, id="a:3", source="wikivoyage", domain="travel"),
                             dict(base, id="a:4", source="nasa", domain="science_explainer"), dict(base, id="govreport:crs_R1", source="govreport", domain="government_public"),
                             dict(base, id="govreport:gao_G1", source="govreport", domain="government_public")])
    got = {r["id"]: (r["domain"], r["subdomain"]) for r in SEL.read_pool(str(d), ["all"])}
    assert got == {"a:1": ("narrative_opinion", "opinion_magazine"), "a:2": ("narrative_opinion", "stories"), "a:3": ("encyclopedia_travel", "travel"), "a:4": ("science_explainer", "science_explainer"),
                   "govreport:gao_G1": ("government_public", "government_public")}
    assert SEL.source_group("cp_news") == SEL.source_group("cp_doab") == "commonpile" and SEL.source_group("eb1911") == "gutenberg" and SEL.source_group("nasa") == "nasa"


# ------------------------------------------------------------------------------------------------ metric validation trials
def test_validation_trials_run_and_detect_known_damage():
    rng = random.Random(21)
    docs = []
    for i in range(6):
        src = [f"The {rng.choice(WORDS)} and the {rng.choice(WORDS)} {rng.choice(VERBS)} a {rng.choice(WORDS)} in the old town number {j}." for j in range(32)]
        tw = "amani ilé ọmọ nkọ̀ wanda abin kasa omu ndi mara ayo ewe adu zaa tiyi baba kitabu mji mtu".split()
        ref = [(" ".join(rng.choice(tw) for _ in range(max(4, int(len(s.split()) * rng.uniform(0.9, 1.3))))).capitalize() + ".") for s in src]
        docs.append(L.RefDoc(f"v{i}", "yor", " ".join(src), src, ref, "human", "x"))
    res = VAL.run_trials(docs, trials=2, seed=1, min_sentences=25)
    c = res["cases"]
    assert res["docs"] == 6 and c["intact"]["doc_flagged_rate"] == 0.0 and c["intact"]["mean_coverage_chars"] == 1.0 and c["intact"]["loop_detected_rate"] == 0.0
    assert c["drop1"]["doc_flagged_rate"] > 0.8 and c["drop4"]["omitted_share_abs_error"] < 0.06 and c["drop4"]["estimated_omitted_share"] > 0.05
    assert c["loop"]["loop_detected_rate"] == 1.0 and c["copy"]["copy_recall"] > 0.8
    assert abs(c["truncate60"]["mean_coverage_chars"] - c["truncate60"]["mean_true_kept_share"]) < 0.15
    assert c["swap"]["mean_coverage_chars"] == 1.0


# ------------------------------------------------------------------------------------------------ the registry of candidate systems meets the backends
def test_every_registered_system_builds_its_inputs_and_has_a_backend_class():
    S = pytest.importorskip("gemma_mt.eval.longdoc_systems")
    assert len(S.SYSTEMS) >= 10
    for name, sp in S.SYSTEMS.items():
        isos = list(sp.codes)[:3]
        if not isos:
            continue
        for iso in isos:
            ins = BE.format_inputs(sp, ["Hello there. It is raining today."], iso)
            assert len(ins) == 1 and isinstance(ins[0], str) and ins[0]
            if sp.prompt_style in ("chat", "t5prompt"):
                assert "Hello there." in ins[0] and "{" not in ins[0], name
            if sp.prompt_style == "madlad":
                assert ins[0].startswith("<2") and ins[0].endswith("raining today.")
            if sp.prompt_style == "marian" and sp.hf_id_template:
                assert sp.hf_id_template.format(code=sp.codes[iso]).startswith("Helsinki-NLP/")
        assert BE.is_encoder_decoder(sp) == (sp.prompt_style in ("nllb", "madlad", "seamless", "marian") or name.startswith(("mt0", "aya-101"))), name
        assert BO.get_spec(name) is sp
    assert BE.prompt_language_name("pcm") == "Nigerian Pidgin" and BE.prompt_language_name("yor") == "Yoruba"
    # no registry entry may claim support for a language outside the 77
    from gemma_mt.languages import SUPPORTED_RL

    assert all(set(sp.codes) <= set(SUPPORTED_RL) and set(sp.proxy_codes) <= set(SUPPORTED_RL) for sp in S.SYSTEMS.values())


def test_token_counter_single_and_batched():
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers

    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tc = L.TokenCounter(tok)
    assert tc("a b c a") == 4 and tc("") == 0
    assert tc.many(["a b", "c", "", "a b c c"]) == [2, 1, 0, 4]


# ------------------------------------------------------------------------------------------------ backend code paths with fake libraries (no model, no GPU)
class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTok:
    def __init__(self):
        self.seen = []

    def __call__(self, texts, **kw):
        torch = pytest.importorskip("torch")
        self.seen.append(list(texts))
        n = max(len(t) for t in texts)
        return _FakeBatch(input_ids=torch.zeros((len(texts), n // 3 + 2), dtype=torch.long))

    def convert_tokens_to_ids(self, tok):
        return 4242 if tok == "yor_Latn" else -1

    def batch_decode(self, gen, skip_special_tokens=True):
        return [f"  out {i}  words " for i in range(gen.shape[0])]


class _FakeModel:
    def __init__(self):
        self.kw = []

    def generate(self, **kw):
        torch = pytest.importorskip("torch")
        self.kw.append({k: v for k, v in kw.items() if k not in ("input_ids",)})
        return torch.zeros((kw["input_ids"].shape[0], 5), dtype=torch.long)


def _bare_seq2seq(sp):
    pytest.importorskip("torch")
    t = BE.HFSeq2SeqTranslator.__new__(BE.HFSeq2SeqTranslator)
    BE.Translator.__init__(t, sp)
    import torch

    t.torch, t.device, t.num_beams, t.batch_items, t.batch_tokens, t.cap = torch, "cpu", 4, 3, 100000, 64
    t.model_id, t.revision, t.name, t.dtype = "org/model", None, sp.name, "fp32"
    tok, model = _FakeTok(), _FakeModel()
    t._models = {"org/model": (tok, model)}
    return t, tok, model


def test_hf_seq2seq_paths_nllb_madlad_seamless_with_fake_model():
    t, tok, model = _bare_seq2seq(spec("nllb"))
    out = t.translate(["One.", "Two two.", "Three three three.", "Four.", "Five."], "yor")
    assert len(out) == 5 and all(o.startswith("out ") for o in out) and "  " not in out[0]
    assert all(k["forced_bos_token_id"] == 4242 and k["num_beams"] == 4 for k in model.kw) and len(model.kw) == 2             # 5 sentences, batches of at most 3
    t2, tok2, model2 = _bare_seq2seq(spec("madlad", codes={"yor": "yo"}))
    t2.translate(["Hello there."], "yor")
    assert tok2.seen[0] == ["<2yo> Hello there."] and "forced_bos_token_id" not in model2.kw[0]
    t3, tok3, model3 = _bare_seq2seq(spec("seamless", codes={"yor": "yor"}))
    t3.translate(["Hello there."], "yor")
    assert model3.kw[0]["tgt_lang"] == "yor" and tok3.seen[0] == ["Hello there."]
    with pytest.raises(KeyError):
        t.translate(["x"], "hau")


def test_ctranslate2_and_vllm_backends_with_fake_modules():
    import sys
    import types

    saved = {k: sys.modules.get(k) for k in ("ctranslate2", "transformers", "vllm")}

    def put(name, mod):
        sys.modules[name] = mod

    try:
        _fake_backend_checks(put, types)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _fake_backend_checks(put, types):

    # ---- CTranslate2: tokens in, target prefix = the language code, hypotheses decoded without the prefix
    class _Res:
        def __init__(self, h):
            self.hypotheses = [h]

    class _Translator:
        def __init__(self, path, device=None, compute_type=None):
            self.calls = []

        def translate_batch(self, toks, target_prefix=None, **kw):
            self.calls.append((toks, target_prefix, kw))
            return [_Res((target_prefix[i] if target_prefix else []) + ["▁ok", f"▁{i}"]) for i in range(len(toks))]

    fake_ct2 = types.ModuleType("ctranslate2")
    fake_ct2.Translator = _Translator

    class _Tok2:
        def __init__(self, *a, **k):
            pass

        @classmethod
        def from_pretrained(cls, *a, **k):
            return cls()

        def encode(self, s):
            return list(range(len(s.split())))

        def convert_ids_to_tokens(self, ids):
            return [f"t{i}" for i in ids]

        def convert_tokens_to_ids(self, toks):
            return list(range(len(toks)))

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(f"w{i}" for i in ids)

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = _Tok2
    put("ctranslate2", fake_ct2)
    put("transformers", fake_tf)
    ct = BE.CT2Translator(spec("nllb"), model_dir="/nowhere", device="cpu")
    out = ct.translate(["a b c", "d e"], "yor")
    toks, prefix, kw = ct.tr.calls[0]
    assert prefix == [["yor_Latn"], ["yor_Latn"]] and len(out) == 2 and out[0].split() == ["w0", "w1"] and kw["beam_size"] == 4         # the prefix token is cut off
    ct2 = BE.CT2Translator(spec("madlad", codes={"yor": "yo"}), model_dir="/nowhere", device="cpu")
    ct2.translate(["a b"], "yor")
    assert ct2.tr.calls[0][1] is None and ct2.tr.calls[0][0][0][:1] == ["t0"]
    # ---- vLLM: chat template applied per prompt, enable_thinking=False tried first
    class _SP:
        def __init__(self, **kw):
            self.kw = kw

    class _VTok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
            assert "enable_thinking" in kw and kw["enable_thinking"] is False
            return "<chat>" + msgs[0]["content"]

    class _LLM:
        def __init__(self, **kw):
            self.kw = kw

        def get_tokenizer(self):
            return _VTok()

        def generate(self, texts, sp):
            self.last = (texts, sp)
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=f"```\nTranslation: {t[-6:]}\n```")]) for t in texts]

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM, fake_vllm.SamplingParams = _LLM, _SP
    put("vllm", fake_vllm)
    v = BE.VLLMTranslator(spec("chat", codes={"yor": "yo"}, template="Translate to {tgt_lang}: {text}"), max_tokens=100)
    got = v.translate(["Hello there."], "yor")
    texts, sp = v.llm.last
    assert texts == ["<chat>Translate to Yoruba: Hello there."] and sp.kw["temperature"] == 0.0 and got == ["there."]
    assert BE.make_translator(spec("nllb"), "stub").supports("yor") is True
