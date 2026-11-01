# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-42: the SMOL / GATITOS benchmarks (smol_sent, smol_doc, gatitos): language table, loaders, download layout, metrics, contamination rule for
documents, the opt-in switch of the driver and the whole pipeline with fakes. CPU only, no network."""
import json
import os

import pytest

from gemma_mt.eval import benchmarks as B
from gemma_mt.eval import contamination as C
from gemma_mt.eval import fetch_data as FD
from gemma_mt.eval import metrics as M
from gemma_mt.languages import ALL_ISOS, LANGUAGES


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _jl(*rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


# ------------------------------------------------------------------------------------------------ language table
def test_default_benchmarks_are_unchanged_and_the_extras_are_opt_in():
    assert B.BENCHMARKS == ("flores_plus", "ntrex", "mafand")
    assert B.SMOL_BENCHMARKS == ("smol_sent", "smol_doc", "gatitos") and B.OTHER_BENCHMARKS == ("ibom_mt", "flores_moz", "glot_story", "salt")
    assert B.EXTRA_BENCHMARKS == B.SMOL_BENCHMARKS + B.OTHER_BENCHMARKS and B.ALL_BENCHMARKS == B.BENCHMARKS + B.EXTRA_BENCHMARKS
    from gemma_mt.eval import sft_benchmarks as S

    assert S.build_parser().parse_args([]).benchmarks == list(B.BENCHMARKS)
    assert B.coverage_table().splitlines()[0] == "| tag | name | flores_plus | ntrex | mafand |"
    assert "smol_sent" in B.coverage_table(B.ALL_BENCHMARKS).splitlines()[0]


def test_smol_codes_are_the_verified_ones_and_only_our_languages_have_them():
    L = LANGUAGES
    # codes that differ from the obvious ISO-639-1 / our tag, each verified on sentences (GlotLID + token coverage in our training targets)
    assert (L["kab"].smol_sent, L["kab"].gatitos) == ("ber-Latn", "ber-Latn")        # Kabyle in Latin script, not `ber` (Tifinagh Central Atlas Tamazight)
    assert L["kon"].smol_sent == "kg" and L["nbl"].smol_sent == "nr" and L["ssw"].smol_sent == "ss" and L["run"].smol_sent == "rn"
    assert L["tso"].smol_sent == "ts" and L["twi"].smol_sent == "ak" and L["orm"].smol_sent == "om" and L["mlg"].smol_sent == "mg"
    assert L["sag"].gatitos == "sg" and L["sag"].smol_sent is None                   # Sango only in GATITOS
    assert L["iso"].gatitos == "iso" and L["iso"].smol_sent is None and L["iso"].smol_doc is None
    assert L["lua"].gatitos == "lua" and L["lua"].smol_sent is None                  # SmolSent `lu` is Luba-Katanga: NOT mapped to our Luba-Kasai
    assert L["swh"].gatitos is None and L["yor"].gatitos is None                     # high-resource languages are not in GATITOS
    for iso in ("ach", "efi", "kri", "tiv", "ndc", "nyn", "nbl", "kon"):             # the languages that had NO test set before
        assert not (L[iso].flores or L[iso].ntrex or L[iso].mafand) and L[iso].smol_sent
    assert "nde" not in B.languages_with_data("smol_sent")                           # North Ndebele has no SFT data
    assert len(B.languages_with_data("smol_sent")) == len(B.languages_with_data("smol_doc")) == 45 and len(B.languages_with_data("gatitos")) == 35
    assert set(B.languages_with_data("smol_sent")) <= set(ALL_ISOS)


def test_parse_benchmark_names_accepts_commas_groups_and_rejects_typos():
    p = B.parse_benchmark_names
    assert p(["smol_sent,smol_doc,gatitos"]) == ["smol_sent", "smol_doc", "gatitos"]
    assert p(["flores_plus", "gatitos,smol_sent", "flores_plus"]) == ["flores_plus", "gatitos", "smol_sent"]
    assert p(["smol"]) == list(B.SMOL_BENCHMARKS) and p(["other"]) == list(B.OTHER_BENCHMARKS) and p(["extra"]) == list(B.EXTRA_BENCHMARKS)
    assert p(["all"]) == list(B.ALL_BENCHMARKS)
    with pytest.raises(ValueError):
        p(["smol_sentt"])


# ------------------------------------------------------------------------------------------------ loaders
def _smol_dir(tmp_path):
    root = tmp_path / "test_sets"
    _write(str(root / "smol_sent" / "en_ach.jsonl"), _jl(
        {"sl": "en", "tl": "ach", "id": 0, "src": "Hello there.", "trg": "Icwiny.", "is_src_orig": True},
        {"sl": "en", "tl": "ach", "id": 1, "src": "", "trg": "empty source", "is_src_orig": True},
        {"sl": "en", "tl": "ach", "id": 2, "src": "Good night.", "trg": "Nino maber.", "is_src_orig": True}))
    _write(str(root / "smol_doc" / "en_ach.jsonl"), _jl(
        {"id": "topic_1__a", "sl": "en", "tl": "ach", "srcs": ["One.", "Two.", "Three."], "trgs": ["Acel.", "Aryo.", "Adek."], "sep": [" ", "\n\n", ""],
         "factuality": "ok", "is_src_orig": True},
        {"id": "topic_2__b", "sl": "en", "tl": "ach", "srcs": ["Only."], "trgs": ["Kekene.", "extra"], "sep": [""], "is_src_orig": True},          # misaligned: dropped
        {"id": "topic_3__c", "sl": "en", "tl": "ach", "srcs": ["A.", "B."], "trgs": ["X.", "Y."], "is_src_orig": True}))                          # no `sep` (older revision)
    _write(str(root / "gatitos" / "en_ach.jsonl"), _jl(
        {"src": "I", "trgs": ["An", "An", "Mana"], "sl": "en", "tl": "ach", "is_source_orig": True},
        {"src": "dog", "trgs": ["gweno"], "sl": "en", "tl": "ach"},
        {"src": "dog", "trgs": ["gwok", "gweno"], "sl": "en", "tl": "ach"},                                                                          # same English twice: merged
        {"src": "", "trgs": ["x"], "sl": "en", "tl": "ach"}, {"src": "cat", "trgs": [], "sl": "en", "tl": "ach"}))                                  # dropped
    return root


def test_smol_loaders_read_the_documented_files(tmp_path):
    root = _smol_dir(tmp_path)
    s = B.load_benchmark("smol_sent", path=str(root / "smol_sent"), isos=["ach"])
    assert [(r["english"], r["reference"], r["id"], r["iso"], r["benchmark"]) for r in s] == [
        ("Hello there.", "Icwiny.", "0", "ach", "smol_sent"), ("Good night.", "Nino maber.", "2", "ach", "smol_sent")]
    d = B.load_benchmark("smol_doc", path=str(root / "smol_doc"), isos=["ach"])
    assert [r["id"] for r in d] == ["topic_1__a", "topic_3__c"]
    assert d[0]["english"] == "One. Two.\n\nThree." and d[0]["reference"] == "Acel. Aryo.\n\nAdek." and d[0]["n_sentences"] == 3 and d[0]["factuality"] == "ok"
    assert d[1]["english"] == "A. B." and d[1]["reference"] == "X. Y."
    g = B.load_benchmark("gatitos", path=str(root / "gatitos"), isos=["ach"])
    assert [(r["english"], r["reference"], r["references"]) for r in g] == [("I", "An", ["An", "Mana"]), ("dog", "gweno", ["gweno", "gwok"])]
    assert B.load_benchmark("smol_sent", path=str(root / "smol_sent"), isos=["ach"], limit=1)[0]["id"] == "0"
    assert B.load_benchmark("smol_sent", path=str(root / "smol_sent"), isos=["yor"]) == []                                      # no file -> no rows, never invented
    with pytest.raises(ValueError):
        B.load_benchmark("smol_sent", isos=["ach"])                                                                             # needs a local path


def test_join_with_sep():
    assert B.join_with_sep(["a", "b"], [" ", ""]) == "a b" and B.join_with_sep(["a", "b"], None) == "a b"
    assert B.join_with_sep(["a", "b"], ["\n", "\n"]) == "a\nb" and B.join_with_sep(["a", "b"], [" "]) == "a b"        # wrong length -> single spaces


# ------------------------------------------------------------------------------------------------ download layout
def test_fetch_smol_raw_layout_pinned_revision_and_source_file(tmp_path):
    calls = []

    def fake_get(url, dest, retries=3, timeout=120.0):
        calls.append(url)
        if "en_ach.jsonl" in url and "gatitos" in url:
            return False
        _write(dest, _jl({"sl": "en", "tl": "x", "id": 0, "src": "a", "trg": "b"}))
        return True

    real = FD._http_get
    FD._http_get = fake_get
    try:
        r = FD.fetch_smol_raw(str(tmp_path / "smol_sent"), "smol_sent", ["ach", "kab", "yor"])
        assert r == {"fetched": ["ach", "ber-Latn", "yo"], "missing": []}
        assert all(f"/resolve/{B.SMOL_REVISION}/smolsent/en_" in u for u in calls) and "ber-Latn" in calls[1]
        assert os.path.exists(tmp_path / "smol_sent" / "en_ber-Latn.jsonl")
        src = json.load(open(tmp_path / "smol_sent" / "SOURCE.json"))
        assert src["licence"] == "cc-by-4.0" and src["revision"] == B.SMOL_REVISION and "2502.12301" in src["citation"]
        n = len(calls)
        FD.fetch_smol_raw(str(tmp_path / "smol_sent"), "smol_sent", ["ach", "kab", "yor"])                                      # idempotent
        assert len(calls) == n
        g = FD.fetch_smol_raw(str(tmp_path / "gatitos"), "gatitos", ["ach", "sag", "yor"])                                      # yor: no GATITOS code, not requested
        assert g == {"fetched": ["sg"], "missing": ["ach"]}
        info = FD.ensure_benchmarks(str(tmp_path / "all"), ["smol_doc"], ["efi"])
        assert info["smol_doc"]["fetched"] == ["efi"]
    finally:
        FD._http_get = real


# ------------------------------------------------------------------------------------------------ metrics
def test_exact_match_and_multi_reference_chrf():
    assert M.normalise_word(" “Gweno.” ") == "gweno" and M.normalise_word("Ẹhẹ") == M.normalise_word("Ẹhẹ".lower())
    assert M.normalise_word("é") == M.normalise_word("é")                                         # NFC
    hyps = ["Gweno.", "xx", "", "mana"]
    refs = [["gweno", "gwok"], ["a"], ["a"], ["An", "Mana"]]
    assert M.exact_match(hyps, refs) == 0.5
    pytest.importorskip("sacrebleu")
    best = M.chrf_pp_multi(["gwok"], [["gweno", "gwok"]])
    assert best > 99 and M.chrf_pp_multi(["gwok"], [["gweno"]]) < best
    assert M.chrf_pp_multi(["ab"], [["ab"]]) == M.chrf_pp(["ab"], ["ab"])


def test_evaluate_rows_scores_word_rows_with_exact_match_and_skips_bleu():
    pytest.importorskip("sacrebleu")
    from gemma_mt.eval import eval_translation as E

    rows = [{"english": "I", "reference": "An", "references": ["An", "Mana"], "iso": "ach", "benchmark": "gatitos", "id": "0"},
            {"english": "dog", "reference": "gweno", "references": ["gweno"], "iso": "ach", "benchmark": "gatitos", "id": "1"}]
    outs = [{"hyp": "Mana.", "prefix_status": "ok", "finished": True}, {"hyp": "kal", "prefix_status": "ok", "finished": True}]
    res = E.evaluate_rows(rows, outs, min_subset_rows=1)
    rec = res["benchmarks"]["gatitos"]["languages"]["ach"]
    assert rec["exact"] == 0.5 and "bleu" not in rec and "spbleu" not in rec and rec["chrf"] > 0
    assert res["benchmarks"]["gatitos"]["macro"]["exact"] == 0.5
    csv_rows = E.csv_rows(res)
    assert csv_rows[0]["exact"] == 0.5 and "exact" in E.CSV_FIELDS


# ------------------------------------------------------------------------------------------------ contamination of documents and words
def test_document_rule_is_exact_or_near_and_words_use_the_sentence_rule_only():
    doc_seen = " ".join(f"word{i}" for i in range(40))
    doc_half = " ".join(f"alpha{i}" for i in range(40))
    doc_new = " ".join(f"gamma{i}" for i in range(40))
    docs = [{"english": e, "reference": "r", "iso": "ach", "benchmark": "smol_doc", "id": f"d{k}"} for k, e in enumerate((doc_seen, doc_half, doc_new))]
    words = [{"english": "dog", "reference": "gweno", "references": ["gweno", "gwok"], "iso": "ach", "benchmark": "gatitos", "id": "w0"},
             {"english": "cat", "reference": "kal", "references": ["kal"], "iso": "ach", "benchmark": "gatitos", "id": "w1"}]
    rows = docs + words
    idx = C.BenchIndex(rows)
    windows = [("Intro. " + doc_seen + " Outro.", "t", "ach"),                                      # the whole document inside a window
               (" ".join(f"alpha{i}" for i in range(25)) + " then unrelated tail text", "t", "ach"),   # 18 of the 33 8-grams of doc_half (> 50%)
               ("The dog barked at the cat.", "t", "ach"),                                           # "dog" occurs, but not as a whole sentence
               ("dog", "gwok", "ach")]                                                              # the entry is a whole window sentence, translation in the target
    scan = C.scan_windows(idx, windows)
    flags = C.flags_for_rows(rows, idx, scan)
    docflags = C.doc_level_flags(flags)
    assert [docflags["exact"][f"ach:d{k}"] for k in range(3)] == [True, True, False]
    assert [flags["exact"][f"ach:d{k}"] for k in range(3)] == [True, False, False] and flags["near"]["ach:d1"] is True
    assert docflags["near"] == flags["near"]
    assert flags["exact"]["ach:w0"] is True and flags["pair"]["ach:w0"] is True        # whole-sentence "dog" + a translation in the target
    assert flags["exact"]["ach:w1"] is False                                           # "cat" only occurs inside a sentence: not contaminated for a word


def test_gatitos_pair_uses_every_accepted_translation():
    rows = [{"english": "dog", "reference": "gweno", "references": ["gweno", "gwok"], "iso": "ach", "benchmark": "gatitos", "id": "0"}]
    idx = C.BenchIndex(rows)
    scan = C.scan_windows(idx, [("dog", "kwo gwok kwo", "ach")])                         # only the SECOND accepted translation is in the target
    assert C.flags_for_rows(rows, idx, scan)["pair"]["ach:0"] is True


# ------------------------------------------------------------------------------------------------ the driver
def test_driver_per_benchmark_budgets_limits_and_comet_policy():
    from gemma_mt.eval import sft_benchmarks as S

    a = S.build_parser().parse_args(["--benchmarks", "smol_sent,smol_doc", "gatitos", "--bench-limit", "smol_doc=5", "gatitos=7"])
    assert S.bench_max_new_tokens(a, "smol_doc") == 1024 and S.bench_max_new_tokens(a, "gatitos") == 48
    assert S.bench_max_new_tokens(a, "flores_plus") == S.bench_max_new_tokens(a, "smol_sent") == 768
    assert S.bench_limits(a) == {"smol_doc": 5, "gatitos": 7}
    assert S.bench_batch_size(a, "smol_doc") == S.bench_batch_size(a, "gatitos") == S.bench_batch_size(a, "smol_sent") == 64          # 0 = --hf-batch-size
    b = S.build_parser().parse_args(["--hf-batch-size", "128", "--doc-batch-size", "32", "--word-batch-size", "512"])
    assert (S.bench_batch_size(b, "smol_doc"), S.bench_batch_size(b, "gatitos"), S.bench_batch_size(b, "ntrex")) == (32, 512, 128)
    with pytest.raises(ValueError):
        S.bench_limits(S.build_parser().parse_args(["--bench-limit", "flores=3"]))


def test_limit_rows_keeps_clean_rows_first_then_the_earliest_others_in_original_order():
    from gemma_mt.eval import sft_benchmarks as S

    rows = [{"iso": "ach", "id": str(i), "contaminated": i not in (3, 5, 7)} for i in range(8)]       # clean rows: ids 3, 5, 7
    rows += [{"iso": "efi", "id": str(i), "contaminated": True} for i in range(3)]
    out = S.limit_rows(rows, 5)
    assert [r["id"] for r in out if r["iso"] == "ach"] == ["0", "1", "3", "5", "7"]         # 3 clean + the 2 earliest others, original order
    assert [r["id"] for r in out if r["iso"] == "efi"] == ["0", "1", "2"]                  # fewer than n: untouched
    assert [r["id"] for r in S.limit_rows(rows, 2) if r["iso"] == "ach"] == ["3", "5"]


def test_pipeline_runs_the_smol_benchmarks_with_their_own_budgets(tmp_path):
    pytest.importorskip("sacrebleu")
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from test_eval_sft import _hooks

    from gemma_mt.eval import sft_benchmarks as S

    root = _smol_dir(tmp_path)
    bench = tmp_path / "bench"
    os.makedirs(bench / "test_sets")
    flagdir = tmp_path / "elsewhere"                                            # --contamination-dir: the flag files need not live in --bench-dir
    for name in B.SMOL_BENCHMARKS:
        os.replace(root / name, bench / "test_sets" / name)
        rows = B.load_benchmark(name, path=str(bench / "test_sets" / name))
        idx = C.BenchIndex(rows)
        flags = C.flags_for_rows(rows, idx, C.scan_windows(idx, [(rows[0]["english"], "t", rows[0]["iso"])]))
        C.write_flag_file(str(flagdir / f"contamination_{name}.json"), name, rows, C.doc_level_flags(flags) if name == "smol_doc" else flags)
    a = S.build_parser().parse_args(["--out", str(tmp_path / "out"), "--bench-dir", str(bench), "--vllm-python", "x", "--max-new-tokens", "6",
                                     "--doc-max-new-tokens", "9", "--word-max-new-tokens", "3", "--benchmarks", "smol_sent,smol_doc,gatitos",
                                     "--languages", "ach", "--min-subset-rows", "1", "--unit-rows", "50",
                                     "--contamination-dir", str(flagdir)])
    h = _hooks(tmp_path)
    assert S.run(a, h) == 0
    res = json.load(open(tmp_path / "out" / "results.json"))
    assert res["failures"] == [] and set(res["systems"]["sft"]["benchmarks"]) == set(B.SMOL_BENCHMARKS)
    assert res["systems"]["sft"]["meta"]["max_new_tokens_by_benchmark"] == {"smol_sent": 6, "smol_doc": 9, "gatitos": 3}
    g = res["systems"]["sft"]["benchmarks"]["gatitos"]["languages"]["ach"]
    assert "exact" in g and g["n"] == 2 and "comet" not in g                                      # words: no AfriCOMET
    assert "comet" not in res["systems"]["sft"]["benchmarks"]["smol_doc"]["languages"]["ach"]       # documents: no AfriCOMET
    md = open(tmp_path / "out" / "report.md").read()
    assert "word level" in md and "exact match" in md and "### smol_doc" in md
    assert len(B.load_benchmark("smol_doc", path=str(bench / "test_sets" / "smol_doc"))) == 2


# ------------------------------------------------------------------------------------------------ the four "other" public sets
def test_other_sets_language_codes():
    L = LANGUAGES
    assert L["ibb"].ibom_mt == "ibb" and L["efi"].ibom_mt == "efi" and L["seh"].flores_moz == "seh_Latn_MZ"
    assert (L["kua"].glot_story, L["her"].glot_story, L["nbl"].glot_story) == ("kua", "her", "nbl")           # the ISO639-3 column (the Language column says kj / hz / nr)
    assert (L["nyn"].salt, L["ach"].salt, L["swh"].salt) == ("nyn", "ach", "swa")
    for iso in ("ibb", "seh", "kqn", "lun", "loz", "kua", "her", "dag", "tsc"):                              # had NO test set before
        assert not (L[iso].flores or L[iso].ntrex or L[iso].mafand or L[iso].smol_sent or L[iso].gatitos)
    assert set(B.languages_with_data("glot_story")) == {"kqn", "lun", "loz", "kua", "her", "dag", "tsc", "nbl"}


def test_pair_file_loaders_and_flores_moz_layout(tmp_path):
    root = tmp_path / "t"
    _write(str(root / "ibom_mt" / "en_ibb.jsonl"), _jl({"id": "0", "src": "Hello", "trg": "Ndi"}, {"id": "1", "src": "", "trg": "x"}))
    _write(str(root / "glot_story" / "en_kua.jsonl"), _jl({"id": "asp_1#0", "src": "A dog.", "trg": "Embwa.", "license": "CC-BY"}))
    rows = B.load_benchmark("ibom_mt", path=str(root / "ibom_mt"), isos=["ibb", "efi"])
    assert [(r["iso"], r["english"], r["reference"], r["benchmark"]) for r in rows] == [("ibb", "Hello", "Ndi", "ibom_mt")]
    g = B.load_benchmark("glot_story", path=str(root / "glot_story"))
    assert g[0]["iso"] == "kua" and g[0]["license"] == "CC-BY" and g[0]["id"] == "asp_1#0"
    _write(str(root / "flores_moz" / "devtest" / "eng_Latn.jsonl"), _jl({"id": "1", "text": "One."}, {"id": "2", "text": "Two."}))
    _write(str(root / "flores_moz" / "devtest" / "seh_Latn_MZ.jsonl"), _jl({"id": "1", "text": "Cimodzi."}, {"id": "2", "text": "Ziwiri."}))
    m = B.load_benchmark("flores_moz", path=str(root / "flores_moz"))
    assert [(r["iso"], r["english"], r["reference"], r["benchmark"], r["id"]) for r in m] == [
        ("seh", "One.", "Cimodzi.", "flores_moz", "1"), ("seh", "Two.", "Ziwiri.", "flores_moz", "2")]


def test_fetch_other_sets_normalise_to_the_pair_layout(tmp_path):
    csv_ibom = 'id,eng,ibb\n0,"Hello, world",Ndi\n1,Bye,Ke\n'
    csv_glot = ("Text,Text Number,License,Text By,Translation By,Language,File Name,Source,ISO639-3,Script,Parallel ID,Story Category\n"
                "A dog.,0,CC-BY,x,y,eng,f,s,eng,Latn,asp_1,asp\n"
                "A cat.,1,CC-BY,x,y,eng,f,s,eng,Latn,asp_1,asp\n"
                "Embwa.,0,CC-BY,x,y,kj,f,s,kua,Latn,asp_1,asp\n"
                "Lozi one.,0,CC-BY,x,y,loz-na,f,s,loz,Latn,asp_1,asp\n"
                "Lozi two.,0,CC-BY-NC,x,y,loz-zm,f,s,loz,Latn,asp_1,asp\n"
                "No english.,9,CC-BY,x,y,kj,f,s,kua,Latn,asp_2,asp\n")
    calls = []

    def fake_get(url, dest, retries=3, timeout=120.0):
        calls.append(url)
        if "ibom-mt" in url:
            body = csv_ibom
        elif "GlotStoryBook" in url:
            body = csv_glot
        else:
            body = _jl({"id": "1", "text": "x"})
        _write(dest, body)
        return True

    real = FD._http_get
    FD._http_get = fake_get
    try:
        r = FD.fetch_ibom_mt(str(tmp_path / "ibom_mt"), ["ibb"])
        assert r == {"fetched": ["ibb"], "missing": []} and f"/resolve/{B.IBOM_REVISION}/eng-ibb/test.csv" in calls[0]
        rows = B.load_benchmark("ibom_mt", path=str(tmp_path / "ibom_mt"), isos=["ibb"])
        assert [(x["english"], x["reference"]) for x in rows] == [("Hello, world", "Ndi"), ("Bye", "Ke")]
        assert not os.path.exists(tmp_path / "ibom_mt" / "_raw" / "eng-ibb.test.csv") and os.path.exists(tmp_path / "ibom_mt" / "SOURCE.json")
        g = FD.fetch_glot_story(str(tmp_path / "glot_story"), ["kua", "loz", "her"])
        assert g == {"fetched": ["kua", "loz"], "missing": ["her"]}
        gl = B.load_benchmark("glot_story", path=str(tmp_path / "glot_story"), isos=["kua", "loz"])
        assert [(x["iso"], x["id"], x["english"]) for x in gl] == [("kua", "asp_1#0", "A dog."), ("loz", "asp_1#0", "A dog."), ("loz", "asp_1#0@loz-zm", "A dog.")]
        assert {x["license"] for x in gl if x["iso"] == "loz"} == {"CC-BY", "CC-BY-NC"}
        assert len({(x["iso"], x["id"]) for x in gl}) == len(gl)                          # dialect variants of one sentence do not share a key
        n = len(calls)
        FD.fetch_glot_story(str(tmp_path / "glot_story"), ["kua", "loz"])                  # idempotent
        assert len(calls) == n
        m = FD.fetch_flores_moz(str(tmp_path / "flores_moz"), ["seh"])
        assert m == {"fetched": ["seh_Latn_MZ"], "missing": []} and os.path.exists(tmp_path / "flores_moz" / "devtest" / "eng_Latn.jsonl")
    finally:
        FD._http_get = real


def test_comet_pair_release_frees_the_workers_and_keeps_the_scores(tmp_path):
    import sys

    from gemma_mt.eval import runtime as RT

    pair = RT.CometPair(python=sys.executable, pythonpath=None, batch_size=4, cache_dir=str(tmp_path / "cc"), fake=True)
    try:
        a = pair.qe(["src one", "src two"], ["mt one", "mt two"])
        assert len(pair._workers) == 1 and pair.stats()["qe"] == {"cache_hits": 0, "scored": 2}
        proc = next(iter(pair._workers.values())).proc
        pair.release()
        assert pair._workers == {} and proc.poll() is not None                      # the child process (and its GPU memory) is gone
        assert pair.stats()["qe"] == {"cache_hits": 0, "scored": 2}                  # counters survive the release
        b = pair.qe(["src one", "src two"], ["mt one", "mt two"])                   # a new worker is started lazily, the disk cache answers
        assert b == a and pair.stats()["qe"] == {"cache_hits": 2, "scored": 2}
    finally:
        pair.close()


def test_compare_eval_results_script(tmp_path):
    import contextlib
    import importlib.util
    import io

    spec = importlib.util.spec_from_file_location("compare_eval_results", os.path.join(os.path.dirname(__file__), "..", "scripts", "compare_eval_results.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def res(chrf_all, chrf_clean, n_clean=30):
        lang = {"ach": {"n": 100, "chrf": chrf_all, "clean": {"n": n_clean, "chrf": chrf_clean}}, "efi": {"n": 100, "chrf": 5.0, "clean": {"n": 3, "chrf": 4.0}}}
        return {"systems": {"sft": {"benchmarks": {"smol_sent": {"languages": lang}}}}}

    json.dump(res(10.0, 8.0), open(tmp_path / "a.json", "w"))
    json.dump(res(14.0, 12.5), open(tmp_path / "b.json", "w"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert mod.main([str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--subset", "clean"]) == 0
    out = buf.getvalue()
    assert "| ach | 30 | 30 | 8.000 | 12.500 | +4.500 |" in out and "efi (few rows)" in out and "macro (1 languages)" in out


def test_skip_base_run_with_the_extra_benchmarks_writes_a_complete_report(tmp_path):
    """The RL-checkpoint recipe of the report uses --skip-base: EVAL_DONE, one system, tables without the base column."""
    pytest.importorskip("sacrebleu")
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from test_eval_sft import _hooks

    from gemma_mt.eval import sft_benchmarks as S

    root = _smol_dir(tmp_path)
    bench = tmp_path / "bench"
    os.makedirs(bench / "test_sets")
    for name in B.SMOL_BENCHMARKS:
        os.replace(root / name, bench / "test_sets" / name)
        rows = B.load_benchmark(name, path=str(bench / "test_sets" / name))
        idx = C.BenchIndex(rows)
        flags = C.flags_for_rows(rows, idx, C.scan_windows(idx, [(rows[0]["english"], "t", rows[0]["iso"])]))
        C.write_flag_file(str(bench / f"contamination_{name}.json"), name, rows, C.doc_level_flags(flags) if name == "smol_doc" else flags)
    a = S.build_parser().parse_args(["--out", str(tmp_path / "out"), "--bench-dir", str(bench), "--vllm-python", "x", "--max-new-tokens", "6", "--skip-base",
                                     "--benchmarks", "smol", "--languages", "ach", "--min-subset-rows", "1", "--unit-rows", "50", "--bench-limit", "smol_sent=1"])
    assert S.run(a, _hooks(tmp_path)) == 0
    res = json.load(open(tmp_path / "out" / "results.json"))
    assert set(res["systems"]) == {"sft"} and res["meta"]["rows"]["smol_sent"] == 1 and (tmp_path / "out" / "EVAL_DONE").exists()
    assert "gatitos" in open(tmp_path / "out" / "report.md").read()
