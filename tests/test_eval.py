# SPDX-License-Identifier: AGPL-3.0-or-later
import csv
import json
import os

import pytest

from gemma_mt.eval import benchmarks as B
from gemma_mt.eval.make_report import load_baseline_csv, load_results, render_report
from gemma_mt.languages import SUPPORTED_RL


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_flores_plus_local_layouts(tmp_path):
    d = tmp_path / "flores"
    _write_jsonl(str(d / "eng_Latn" / "devtest.jsonl"), [{"id": i, "text": f"english {i}"} for i in range(5)])
    _write_jsonl(str(d / "yor_Latn" / "devtest.jsonl"), [{"id": i, "text": f"yoruba {i}"} for i in range(5)])
    _write_jsonl(str(d / "devtest" / "hau_Latn.jsonl"), [{"id": i, "text": f"hausa {i}"} for i in range(3)])
    rows = B.load_flores_plus(["yor", "hau", "ibb"], "devtest", str(d))     # ibb has no FLORES+ code -> no rows
    assert {r["iso"] for r in rows} == {"yor", "hau"}
    assert len(rows) == 8
    r = [x for x in rows if x["iso"] == "yor"][2]
    assert r == {"english": "english 2", "reference": "yoruba 2", "iso": "yor", "benchmark": "flores_plus", "id": "2"}
    assert len(B.load_flores_plus(["yor"], "devtest", str(d), limit=2)) == 2


def test_ntrex_local_files_and_length_check(tmp_path):
    d = tmp_path / "NTREX-128"
    d.mkdir()
    (d / "newstest2019-src.eng.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (d / "newstest2019-ref.yor.txt").write_text("okan\nmeji\nmeta\n", encoding="utf-8")
    (d / "newstest2019-ref.swa.txt").write_text("moja\nmbili\ntatu\n", encoding="utf-8")   # swh -> "swa"
    rows = B.load_ntrex(["yor", "swh", "hau"], str(tmp_path))
    assert {r["iso"] for r in rows} == {"yor", "swh"} and len(rows) == 6
    (d / "newstest2019-ref.hau.txt").write_text("daya\n", encoding="utf-8")
    with pytest.raises(ValueError):
        B.load_ntrex(["hau"], str(d))


def test_mafand_local_translation_dicts_and_code_mapping(tmp_path):
    d = tmp_path / "mafand"
    _write_jsonl(str(d / "en-yor" / "test.jsonl"),
                 [{"translation": {"en": "Hello", "yor": "Bawo"}}, {"translation": {"en": "Bye", "yor": "O dabo"}}])
    _write_jsonl(str(d / "en-swa" / "test" / "0000.jsonl"), [{"translation": {"en": "Water", "swa": "Maji"}}])
    rows = B.load_mafand(["yor", "swh", "ewe"], "test", str(d))      # ewe has no en-ewe pair in MAFAND (fr-ewe)
    assert sorted((r["iso"], r["reference"]) for r in rows) == [("swh", "Maji"), ("yor", "Bawo"), ("yor", "O dabo")]


def test_front_door_and_normalised_file(tmp_path):
    p = tmp_path / "bench.jsonl"
    _write_jsonl(str(p), [{"english": "a", "reference": "b", "iso": "SWA"}, {"english": "c", "reference": "", "iso": "yor"}])
    rows = B.load_normalised_file(str(p), "custom")
    assert len(rows) == 1 and rows[0]["iso"] == "swh" and rows[0]["benchmark"] == "custom"
    with pytest.raises(ValueError):
        B.load_benchmark("nope")
    with pytest.raises(ValueError):
        B.load_benchmark("ntrex")                                    # needs a local path
    assert "yor_Latn" in B.coverage_table()


def test_metrics_and_macro_average():
    pytest.importorskip("sacrebleu")
    from gemma_mt.eval import metrics as M

    hyp = ["the cat sat on the mat", "good morning"]
    assert M.chrf_pp(hyp, hyp) == pytest.approx(100.0)
    assert M.bleu(hyp, hyp) == pytest.approx(100.0)
    assert M.chrf_pp(["xyz"], ["the cat sat"]) < 10
    macro = M.macro_average({"yor": {"chrf": 40.0, "comet": 0.7}, "hau": {"chrf": 20.0}})
    assert macro["chrf"] == 30.0 and macro["chrf_n_languages"] == 2 and macro["comet"] == 0.7


def _fake_results(tmp_path, name, shift):
    pytest.importorskip("sacrebleu")
    from gemma_mt.eval.eval_translation import evaluate_rows, write_results
    from gemma_mt.rl.comet import FakeScorer

    rows, outs = [], []
    for iso, n in (("yor", 6), ("hau", 6), ("ibb", 4)):
        for i in range(n):
            ref = f"the reference sentence number {i} for {iso}"
            hyp = ref if i % 2 == 0 else ref.replace("reference", "x" * shift)
            rows.append({"english": f"source {i}", "reference": ref, "iso": iso, "benchmark": "flores_plus", "id": str(i)})
            outs.append({"hyp": hyp, "raw": "yor " + hyp, "prefix_status": "ok", "finished": True})
    sc = FakeScorer()
    res = evaluate_rows(rows, outs, comet_ref=lambda s, m, r: sc(s, m), comet_qe=lambda s, m: sc(s, m))
    out = tmp_path / name
    write_results(str(out), name, {"model": name}, res, rows, outs)
    return res, str(out / "results.json")


def test_evaluate_rows_structure_and_comet_only_for_supported(tmp_path):
    res, path = _fake_results(tmp_path, "sft", 1)
    fl = res["benchmarks"]["flores_plus"]
    assert set(fl["languages"]) == {"yor", "hau", "ibb"}
    assert "comet" in fl["languages"]["yor"] and "comet" not in fl["languages"]["ibb"], "AfriCOMET only where supported"
    assert fl["languages"]["yor"]["n"] == 6 and fl["macro"]["n_languages"] == 3
    assert fl["languages"]["yor"]["prefix_ok_rate"] == 1.0
    assert fl["macro"]["chrf_n_languages"] == 3 and fl["macro"]["comet_n_languages"] == 2
    d = os.path.dirname(path)
    with open(os.path.join(d, "results.csv")) as f:
        rows = list(csv.DictReader(f))
    assert [r["iso"] for r in rows] == ["yor", "hau", "ibb", "MACRO"] or {r["iso"] for r in rows} >= {"yor", "MACRO"}
    assert len(open(os.path.join(d, "predictions.jsonl")).read().splitlines()) == 16
    assert json.load(open(path))["meta"]["name"] == "sft"


def test_report_has_deltas_per_language_and_baseline_csv(tmp_path):
    _, p_base = _fake_results(tmp_path, "base", 5)
    _, p_rl = _fake_results(tmp_path, "rl", 1)
    bl = tmp_path / "published.csv"
    bl.write_text("system,iso,benchmark,chrf,comet\nNLLB,yor,flores_plus,50.5,0.8\nNLLB,hau,,45.0,\nNLLB,MACRO,,1,1\n")
    from collections import OrderedDict

    systems = OrderedDict()
    systems["base"] = load_results(p_base)[1]
    systems["rl"] = load_results(p_rl)[1]
    systems.update(load_baseline_csv(str(bl)))
    assert ("flores_plus", "yor") in systems["NLLB"] and ("*", "hau") in systems["NLLB"] and len(systems["NLLB"]) == 2
    md = render_report(systems, reference="base")
    assert "## flores_plus" in md and "Δ rl" in md and "Δ NLLB" in md
    assert "Yoruba (`yor`)" in md and "50.50" in md
    # the macro table is restricted to languages common to every system for that metric: chrF -> yor, hau
    macro_rows = [l for l in md.splitlines() if l.startswith("| **base**")]
    assert macro_rows and "(2)" in macro_rows[0]
    # rl is better than base on chrF -> positive delta
    line = [l for l in md.splitlines() if l.startswith("| **base**") or l.startswith("| rl |")]
    assert any("+" in l for l in line)
    with pytest.raises(ValueError):
        render_report(systems, reference="missing")


def test_generate_translations_greedy_alignment_and_prefix_status():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from gemma_mt.eval.eval_translation import generate_translations
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.modeling import build_tiny_model

    tok = FakeGemmaTokenizer()
    model = build_tiny_model(seed=3).eval()
    rows = [{"english": e, "reference": "r", "iso": i, "benchmark": "b", "id": str(k)}
            for k, (e, i) in enumerate([("Good morning", "yor"), ("A much longer English sentence than the others", "hau"),
                                        ("Hi", "ibo")])]
    a = generate_translations(model, tok, rows, stop_ids=[106, 1], pad_id=0, device="cpu", batch_size=2, max_new_tokens=8)
    b = generate_translations(model, tok, rows, stop_ids=[106, 1], pad_id=0, device="cpu", batch_size=3, max_new_tokens=8)
    assert [x["raw"] for x in a] == [x["raw"] for x in b], "batching / length sorting must not change greedy output"
    assert len(a) == 3 and all(set(x) == {"raw", "hyp", "prefix_status", "finished"} for x in a)
    assert all(x["prefix_status"] in ("ok", "wrong", "missing", "empty") for x in a)


def test_reverse_rows_swap_source_and_reference():
    from gemma_mt.eval.eval_translation import reverse_rows

    rows = [{"english": "Good morning", "reference": "E kaaro", "iso": "yor", "benchmark": "flores_plus", "id": "3"}]
    r = reverse_rows(rows)[0]
    assert (r["english"], r["reference"], r["iso"], r["prompt_iso"], r["id"]) == ("E kaaro", "Good morning", "yor", "eng", "3")
    assert rows[0]["english"] == "Good morning"            # input not mutated


def test_text_metrics_are_invariant_to_nfc_vs_nfd_references():
    import unicodedata

    from gemma_mt.eval import metrics as M

    hyp = ["Ọ̀rọ̀ ẹ̀kọ́ ṣe pàtàkì fún àwọn ọmọ Yorùbá gbogbo."]
    ref_nfc = [unicodedata.normalize("NFC", hyp[0])]
    ref_nfd = [unicodedata.normalize("NFD", hyp[0])]
    assert ref_nfc != ref_nfd
    assert M.chrf_pp(hyp, ref_nfc) == M.chrf_pp(hyp, ref_nfd) == 100.0
    assert M.bleu(hyp, ref_nfd) == M.bleu(hyp, ref_nfc)
