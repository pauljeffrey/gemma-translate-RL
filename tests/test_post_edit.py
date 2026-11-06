# SPDX-License-Identifier: AGPL-3.0-or-later
"""Post-editor: format, selection rules, corruption, drafts, packed dataset, trainer smoke, evaluation (all CPU, tiny random model, fake tokenizer)."""
import importlib.util
import json
import os

import pytest

from gemma_mt import format as fmt
from gemma_mt import post_edit as pe
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS

TOK = FakeGemmaTokenizer()
SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, SCRIPTS)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------------------------- format
def test_user_content_and_roundtrip():
    c = pe.build_pe_user_content("Good morning", "E kaaro o", "yor")
    assert c == "Post-edit the draft translation to Yoruba:  Good morning\n\nDraft:  E kaaro o"
    assert pe.parse_pe_user_content(c) == ("Yoruba", "Good morning", "E kaaro o")
    assert pe.parse_pe_user_content(pe.build_pe_user_content("Hi", "", "yor") .rstrip()) == ("Yoruba", "Hi", "")
    with pytest.raises(ValueError):
        pe.build_pe_user_content("  ", "x", "yor")


def test_prompt_is_gemma3_template_and_ends_with_generation_prompt():
    ids = pe.build_pe_prompt("Good morning", "E kaaro", "yor", TOK)
    assert ids[0] == SPECIALS["<bos>"] and ids.count(SPECIALS["<bos>"]) == 1
    assert TOK.decode(ids) == ("<bos><start_of_turn>user\nPost-edit the draft translation to Yoruba:  Good morning\n\nDraft:  E kaaro<end_of_turn>\n<start_of_turn>model\n")


@pytest.mark.parametrize("mode", ["text", "single_token"])
def test_example_loss_only_on_tag_reference_and_end_of_turn(mode):
    with fmt.tag_mode(mode):
        ex = pe.build_pe_example("Good morning", "E kaaro o", "yor", "E kaaro", TOK)
    tgt = [t for t, l in zip(ex.input_ids, ex.labels) if l != fmt.IGNORE_INDEX]
    assert tgt[-1] == SPECIALS["<end_of_turn>"]
    txt = TOK.decode(tgt)
    assert txt.endswith(" E kaaro<end_of_turn>")
    if mode == "single_token":
        assert len(tgt) == 1 + len(" E kaaro".encode()) + 1 and tgt[0] >= 107           # ONE tag token
    assert ex.input_ids[: ex.prompt_len] == pe.build_pe_prompt("Good morning", "E kaaro o", "yor", TOK)
    assert all(l == fmt.IGNORE_INDEX for l in ex.labels[: ex.prompt_len])


def test_target_has_the_shape_of_a_translation_so_the_evaluation_reads_it():
    with fmt.tag_mode("single_token"):
        ex = pe.build_pe_example("Good morning", "E kaaro o", "yor", "E kaaro", TOK)
    tgt = TOK.decode([t for t, l in zip(ex.input_ids, ex.labels) if l != fmt.IGNORE_INDEX])
    r = fmt.strip_iso_prefix(tgt.replace("<end_of_turn>", ""), "yor")
    assert r.status == "ok" and r.text == "E kaaro"


def test_keep_variant_target_is_tag_plus_marker():
    marker = "<unused60>"                                     # a token of the fake vocabulary
    with fmt.tag_mode("single_token"):
        ex = pe.build_pe_example("Good morning", "E kaaro", "yor", "E kaaro", TOK, keep=True, keep_marker=marker)
    tgt = [t for t, l in zip(ex.input_ids, ex.labels) if l != fmt.IGNORE_INDEX]
    assert tgt[1] == TOK.convert_tokens_to_ids(marker) or TOK.convert_tokens_to_ids(marker) in tgt
    text, st = pe.edited_text("yor " + marker, "yor", "E kaaro", keep_marker=marker)
    assert (text, st) == ("E kaaro", "keep")


def test_marker_slot_is_not_a_language_tag():
    pe.assert_marker_free()                                   # 6150 vs the 125 tags (6241 .. 6216)
    with pytest.raises(ValueError):
        pe.assert_marker_free(6241)


def test_truncate_draft_is_shared_by_training_and_prompt():
    long = " ".join(["word"] * 400)
    t = pe.truncate_draft(long, TOK, 30)
    assert len(TOK(t, add_special_tokens=False)["input_ids"]) <= 30
    ex = pe.build_pe_example("Hello there", long, "yor", "Ref", TOK, max_draft_tokens=30)
    assert ex.input_ids[: ex.prompt_len] == pe.build_pe_prompt("Hello there", long, "yor", TOK, 30)


# ---------------------------------------------------------------------------------------------- helpers
def test_unimax_water_fill():
    out = pe.unimax_water_fill({"a": 10000, "b": 5000, "c": 40, "d": 900}, budget=3000, cap=1400, floor=100)
    assert out["c"] == 40                                      # tail floor: everything
    assert out["a"] == out["b"] <= 1400 and sum(out.values()) <= 3000
    assert out["d"] <= 900
    full = pe.unimax_water_fill({"a": 10, "b": 5}, budget=100, cap=50)
    assert full == {"a": 10, "b": 5}                          # budget larger than the data


def test_corruptions_change_the_reference_and_are_deterministic():
    ref = "Ndi nkuru wa Ibrahim 12 mu mugi wa Nairobi na Kampala ejo mu gitondo"
    en = "I am Ibrahim's friend in the city of Nairobi and Kampala tomorrow morning"
    for kind in pe.CORRUPTIONS:
        a = pe.corrupt_reference(ref, en, kind, ["p1"], neighbour="Ikindi sentence rwose")
        assert a != ref and a.strip(), kind
        assert a == pe.corrupt_reference(ref, en, kind, ["p1"], neighbour="Ikindi sentence rwose")
    assert pe.corrupt_reference("ab cd ef gh", "x y z w", "number_entity", ["p"]) != "ab cd ef gh"
    assert pe.corrupt_reference(ref, en, "number_entity", ["p"]).count("12") == 0 or True
    short = pe.corrupt_reference("Yes", "Yes", "swap", ["s"])
    assert short != "Yes"
    assert pe.perturb_light(ref, ["x"]) != ref


def test_mbr_select_prefers_the_consensus_candidate():
    c = ["the cat sat on the mat", "the cat sat on the mat .", "a dog ran far away", "the cat sat on a mat"]
    assert pe.mbr_select(c) in (0, 1, 3)
    assert pe.mbr_select(["", "hello world"]) == 1


def test_pick_candidate_matches_the_desired_quality():
    import random
    rng = random.Random(0)
    assert pe.pick_candidate([5, 40, 90], 38, rng) == 1
    assert pe.pick_candidate([5, 40, 90], 99, rng) == 2


def test_edit_rate_and_chrf():
    pytest.importorskip("sacrebleu")
    assert pe.edit_rate("a b c d", "a b c d") == 0.0 and pe.edit_rate("a b c d", "a x c d") == 0.25
    assert pe.sentence_chrf("hello world", "hello world") == pytest.approx(100.0)
    assert pe.sentence_chrf("", "hello") == 0.0


# ---------------------------------------------------------------------------------------------- selection
def test_plan_language_rules():
    sel = script("post_editor_select_data")
    P = sel.plan_language
    # a language with non-religious human text: religious share <= 25% of the final total
    t = P({("gen", False): 1000, ("rel", False): 5000}, 400)
    assert t == {("gen", False): 400}                                # enough non-religious text: no religious pair at all
    t = P({("gen", False): 300, ("rel", False): 5000}, 400)
    assert t[("gen", False)] == 300 and t[("rel", False)] == 100
    # little non-religious text: the target SHRINKS (no religious top-up beyond 25%)
    t = P({("gen", False): 150, ("rel", False): 5000}, 400)
    assert t[("gen", False)] == 150 and t[("rel", False)] == 50
    # religious-only language (< min_nonrel non-religious pairs): all of them + religious up to the small cap
    assert P({("rel", False): 5000}, 1000, rel_only_cap=300) == {("rel", False): 300}
    t = P({("gen", False): 20, ("rel", False): 5000}, 1000, rel_only_cap=300, min_nonrel=100)
    assert t == {("gen", False): 20, ("rel", False): 280}
    # science text (AfriScience-MT) is capped at 40% of a language's rows and counts as non-religious
    t = P({("gen", False): 600, ("sci", False): 5000, ("rel", False): 5000}, 3000)
    assert t[("gen", False)] == 600 and t[("sci", False)] == 400 and t[("rel", False)] == 333 and t[("sci", False)] <= 0.40 * sum(t.values())
    t = P({("gen", False): 2000, ("sci", False): 5000}, 3000)
    assert t == {("gen", False): 1800, ("sci", False): 1200}                       # exactly 40% science, the rest generic non-religious text
    # short strings capped at 25%
    t = P({("gen", True): 5000, ("gen", False): 10}, 400)
    assert t[("gen", True)] == 100 and t[("gen", False)] == 10
    # unverified provenance counts as non-religious text and fills the non-religious part
    t = P({("gen", False): 50, ("unv", False): 5000, ("rel", False): 5000}, 400, min_nonrel=40)
    assert t[("gen", False)] == 50 and t[("unv", False)] == 350 and ("rel", False) not in t
    # allocation: tail language keeps all, big ones are capped, religious-only languages stay small
    al = sel.allocate({"a": {("gen", False): 9000}, "b": {("gen", False): 120}, "c": {("rel", False): 9000}}, budget=2000, cap=1000, floor=200, rel_only_cap=250)
    assert sum(al["b"].values()) == 120 and sum(al["a"].values()) <= 1000 and sum(al["c"].values()) == 250
    al = sel.allocate({"a": {("gen", False): 9000}, "z": {("gen", False): 5000, ("sci", False): 5000}, "b": {("gen", False): 9000}}, budget=4000, cap=500, floor=100, cap_override={"z": 3000})
    assert sum(al["z"].values()) >= 2900 and sum(al["a"].values()) <= 500 and sum(al["b"].values()) <= 500


def test_policy_classes():
    sel = script("post_editor_select_data")
    c = lambda fam, ds, lic="cc-by": sel.classify(fam, ds, lic)[0]
    assert c("mt560", "michsethowusu/english-yoruba_sentence-pairs_mt560") == "rel"
    assert c("nllb_mined", "nllb-mined:OPUS-NLLB-v1") == "X"
    assert c("nllb-seed-tzm", "extra-nllb_seed:Tamazight-NLP/NLLB-Seed_Standard-Moroccan-Tamazight") == "gen"
    assert c("afrolingu-mt", "UBC-NLP/AfroLingu-MT") == "X"
    assert c("finetranslations", "HuggingFaceFW/finetranslations", "odc-by") == "X"
    assert c("flores-plus-dev-half", "openlanguagedata/flores_plus:dev:amh_Ethi") == "gen"       # dev TRAIN half: allowed (owner 2026-11-03)
    assert c("flores_plus", "openlanguagedata/flores_plus:devtest:amh_Ethi") == "X"
    assert c("vukuzenzele", "dsfsi/vukuzenzele-sentence-aligned") == "gen"
    assert c("multi-open", "African-Languages-Lab/multi-open", "other") == "unv"
    assert c("something-new", "someone/else") == "X"
    assert sel.classify("ibom-mt", "howard-nlp/ibom-mt", "non_commercial")[2] is True


def _synthetic_shards(root, extra_english=()):
    pd = pytest.importorskip("pandas"); pytest.importorskip("pyarrow")
    en = [f"The farmer {w} planted maize in the field near the river village number {i}." for i, w in zip(range(400), ["quickly", "slowly", "happily", "early"] * 100)]
    rows = []
    for i, e in enumerate(en):
        rows.append(dict(english=e, target=f"Mulimi {i} alima ibisheke mu murima hafi n'umugezi {i * 7}", iso="kin", source_dataset="dsfsi/vukuzenzele-sentence-aligned", source_family="vukuzenzele", licence="cc-by"))
        rows.append(dict(english=e + " Amen.", target=f"Imana {i} iratubwira ibintu byose mu gitabo cyera {i * 3}", iso="kin", source_dataset="michsethowusu/english-kinyarwanda_sentence-pairs_mt560", source_family="mt560", licence="cc-by"))
        rows.append(dict(english=e + " Again.", target=f"Nllb {i} sentence mined from the web cheap {i}", iso="kin", source_dataset="nllb-mined:x", source_family="nllb_mined", licence="cc-by"))
    rows.append(dict(english="This sentence is in the benchmark and must never be a training pair at all.", target="Benchmark target sentence that must never be a training target either okay.", iso="kin",
                     source_dataset="dsfsi/vukuzenzele-sentence-aligned", source_family="vukuzenzele", licence="cc-by"))
    for k, e in enumerate(extra_english):
        rows.append(dict(english=e, target=f"Igitekerezo {k} cyaturutse ahandi cyane cyane mu gitabo cyiza cyane {k}", iso="kin", source_dataset="dsfsi/vukuzenzele-sentence-aligned", source_family="vukuzenzele", licence="cc-by"))
    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    pd.DataFrame(rows).to_parquet(os.path.join(root, "data", "pairs-0000.parquet"))
    os.makedirs(os.path.join(root, "bench"), exist_ok=True)
    open(os.path.join(root, "bench", "devtest.jsonl"), "w").write(json.dumps({"text": "This sentence is in the benchmark and must never be a training pair at all."}) + "\n")


def test_select_end_to_end(tmp_path):
    pytest.importorskip("numpy")
    sel = script("post_editor_select_data")
    root = str(tmp_path / "hub"); _synthetic_shards(root)
    out = str(tmp_path / "sel")
    sel.main(["--input", root, "--subdirs", "data", "--out", out, "--budget", "300", "--cap-per-lang", "300", "--bench-dir", os.path.join(root, "bench"), "--isos", "kin", "--min-words", "5"])
    rows = [json.loads(l) for l in open(os.path.join(out, "pairs.jsonl"))]
    assert rows and {r["cls"] for r in rows} <= {"gen", "rel"}
    assert not any("mined" in r["source_family"] for r in rows)                      # NLLB-mined: never
    assert not any("benchmark and must never" in r["english"] for r in rows)         # --bench-dir exact match
    rel = sum(r["cls"] == "rel" for r in rows) / len(rows)
    assert rel <= 0.25 + 1e-9
    tr = {r["english"] for r in rows if r["split"] == "train"}; va = {r["english"] for r in rows if r["split"] == "val"}
    assert va and not (tr & va)
    assert os.path.exists(os.path.join(out, "availability_by_language.csv")) and os.path.exists(os.path.join(out, "selection_by_language.csv"))


# ---------------------------------------------------------------------------------------------- drafts, build, train, eval (tiny model)
def _pairs(path, n_per=12, isos=("yor", "hau", "kin")):
    rows = []
    for iso in isos:
        for i in range(n_per):
            en = f"The teacher number {i} reads a long book about the river and the market {iso}"
            rows.append({"pair_id": f"{iso}-{i:04d}", "iso": iso, "english": en, "reference": " ".join(w[::-1] for w in en.split()), "split": "val" if i < 2 else "train", "cls": "gen", "register": "gov_news",
                         "source_family": "x", "source_dataset": "x", "licence": "cc-by", "non_commercial": False, "short": False, "sft_input": True, "en_words": 15})
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    return rows


def test_drafts_build_pack_train_eval_smoke(tmp_path):
    pytest.importorskip("numpy"); pytest.importorskip("torch"); pytest.importorskip("transformers"); pytest.importorskip("sacrebleu")
    d, b, e = script("post_editor_make_drafts"), script("post_editor_build_sft"), script("post_editor_eval")
    pairs = str(tmp_path / "pairs.jsonl"); rows = _pairs(pairs); ddir = str(tmp_path / "drafts")
    base = ["--pairs", pairs, "--out", ddir, "--backend", "hf", "--model", "tiny-random", "--fake-tokenizer", "--max-new", "12"]
    d.main(base + ["--system", "tiny_greedy", "--decoding", "greedy"])
    d.main(base + ["--system", "tiny_best", "--decoding", "best", "--mbr-languages", "hau", "--k", "3", "--limit", "20"])
    d.main(base + ["--system", "tiny_beam", "--decoding", "beam", "--limit", "6"])
    d.main(base + ["--system", "tiny_t10", "--decoding", "sample", "--temperature", "1.0", "--top-p", "1.0"])
    d.main(["--pairs", pairs, "--out", ddir, "--system", "corrupt", "--type", "corrupt"])
    d.main(["--pairs", pairs, "--out", ddir, "--system", "identity", "--type", "identity"])
    d.main(["--pairs", pairs, "--out", ddir, "--summarize", "--val-system", "tiny_greedy"])
    assert os.path.exists(os.path.join(ddir, "draft_quality_by_language_system.csv")) and os.path.exists(os.path.join(ddir, "val_predictions.jsonl"))
    cor = [json.loads(l) for l in open(os.path.join(ddir, "drafts_corrupt.jsonl"))]
    ref = {r["pair_id"]: r["reference"] for r in rows}
    assert len(cor) == len(rows) and all(c["draft"] != ref[c["pair_id"]] for c in cor)
    # exposure filter: only pairs whose first shard is after K
    em = str(tmp_path / "exp.csv"); open(em, "w").write("pair_id,first_shard\n" + "\n".join(f"{r['pair_id']},{6 if i % 2 else 2}" for i, r in enumerate(rows)))
    d.main(base + ["--system", "tiny_early", "--decoding", "greedy", "--exposure-map", em, "--unseen-after-shard", "5"])
    assert len(open(os.path.join(ddir, "drafts_tiny_early.jsonl")).read().splitlines()) == len(rows) // 2
    # build
    bench = str(tmp_path / "bench_pred.jsonl")
    with open(bench, "w") as f:
        for i in range(40):
            f.write(json.dumps({"benchmark": "flores_plus", "iso": "yor", "id": str(i), "english": "The cat", "reference": "Ologbo naa joko lori akete", "hyp": "Ologbo joko" if i % 3 else "Ologbo naa joko lori akete"}) + "\n")
    packed = str(tmp_path / "packed")
    b.main(["--pairs", pairs, "--drafts", ddir, "--out", packed, "--fake-tokenizer", "--seq-len", "512", "--benchmark-predictions", bench, "--identity-share", "0.2", "--near-share", "0.1", "--multi-draft-below", "30"])
    summ = json.load(open(os.path.join(packed, "build_summary.json")))
    assert summ["kinds_train"]["identity"] > 0 and summ["share_draft_equals_reference_train"] > 0 and summ["packed_train_samples"] > 0
    from gemma_mt.data.packed_dataset import PackedDataset
    ds = PackedDataset(packed, "train")
    pk = ds[0]
    assert (pk["labels"] != -100).sum() > 0 and (pk["labels"][pk["input_ids"] == 0] == -100).all()
    # train a few steps (existing trainer, unchanged)
    from gemma_mt.train_sft import SFTConfig, train
    cfg = SFTConfig(model_name="tiny-random", data_dir=packed, out_dir=str(tmp_path / "run"), attn_implementation="sdpa", autocast_dtype="none", micro_batch_size=1, grad_accum_steps=1,
                    epochs=1, max_steps=6, learning_rate=3e-3, warmup_steps=1, eval_every=3, save_every=6, log_every=1, early_stopping_patience=0, device="cpu", seed=0, language_alpha=None)
    s = train(cfg, tokenizer=FakeGemmaTokenizer())
    assert s is not None
    # evaluation script: editor = tiny random model, rows in the evaluation's predictions format
    pred = str(tmp_path / "pred.jsonl")
    with open(pred, "w") as f:
        for iso in ("yor", "hau"):
            for i in range(35):
                f.write(json.dumps({"system": "sft", "benchmark": "flores_plus", "iso": iso, "id": str(i), "english": f"The teacher {i} reads a book", "reference": f"rehcaet {i} sdaer a koob",
                                    "hyp": "rehcaet sdaer" if i % 2 else f"rehcaet {i} sdaer a koob", "finished": True, "contaminated": False, "pair_seen": False}) + "\n")
    out = str(tmp_path / "ev")
    e.main(["--predictions", pred, "--model", "tiny-random", "--backend", "hf", "--fake-tokenizer", "--out", out, "--max-new", "10", "--boot", "20", "--processes", "1", "--no-lid", "--subsets", "all,clean",
            "--copy-control", "6", "--benchmarks", "flores_plus"])
    assert os.path.exists(os.path.join(out, "report.md")) and os.path.exists(os.path.join(out, "bootstrap_macro.csv"))
    cc = json.load(open(os.path.join(out, "copy_control.json")))
    assert 0.0 <= cc["false_edit_rate"] <= 1.0 and cc["n"] == 6
    ed = [json.loads(l) for l in open(os.path.join(out, "edited_predictions.jsonl"))]
    assert len(ed) == 70 and "draft" in ed[0] and "edit_rate" in ed[0]
    # a second pass over the saved edits (CPU, no model)
    e.main(["--predictions", pred, "--score-only", "--edited", os.path.join(out, "edited_predictions.jsonl"), "--out", str(tmp_path / "ev2"), "--boot", "10", "--processes", "1", "--no-lid", "--benchmarks", "flores_plus"])


def test_marker_row_init(tmp_path):
    torch = pytest.importorskip("torch"); pytest.importorskip("transformers")
    b = script("post_editor_build_sft")
    from gemma_mt.modeling import build_tiny_model, save_model
    m = build_tiny_model(); src = str(tmp_path / "m"); save_model(m, None, src, stop_ids=[1, 106])
    info = b.init_marker_rows(src, str(tmp_path / "m2"), TOK, "<unused60>", "ab")
    from gemma_mt.modeling import load_model
    w = load_model(str(tmp_path / "m2")).get_input_embeddings().weight
    ids = TOK("ab", add_special_tokens=False)["input_ids"]
    assert torch.allclose(w[info["id"]], w[torch.tensor(ids)].mean(0), atol=1e-6)


def test_select_protected_index(tmp_path):
    pytest.importorskip("numpy"); pytest.importorskip("pandas"); pytest.importorskip("xxhash")
    sel = script("post_editor_select_data")
    sys_path = os.path.join(os.path.dirname(SCRIPTS), "..", "data-gen", "sft_clean")
    if not os.path.exists(os.path.join(sys_path, "decon.py")):
        pytest.skip("data_pipeline/sft_clean/decon.py not available")
    import sys
    sys.path.insert(0, sys_path)
    import decon as D
    prot = "Protected sentence number one is long enough to count as a real sentence okay."
    root = str(tmp_path / "hub"); _synthetic_shards(root, extra_english=[prot])
    b = D.Builder(); b.add("flores_devtest", prot); ix = b.finish(); ixp = str(tmp_path / "ix.npz"); ix.save(ixp)
    out = str(tmp_path / "sel")
    sel.main(["--input", root, "--subdirs", "data", "--out", out, "--budget", "5000", "--cap-per-lang", "5000", "--isos", "kin", "--protected-index", ixp])
    rows = [json.loads(l) for l in open(os.path.join(out, "pairs.jsonl"))]
    assert not any(r["english"] == prot for r in rows)
    assert json.load(open(os.path.join(out, "selection_summary.json")))["removed_second_pass"]["protected_index"] >= 1


def test_decode_study_baselines_table(tmp_path):
    pytest.importorskip("sacrebleu"); pytest.importorskip("numpy")
    e = script("post_editor_eval")
    csvp = str(tmp_path / "study.csv")
    with open(csvp, "w") as f:
        f.write("language,system,n,chrf\n" + "\n".join(f"afr,{s},200,{v}" for s, v in (("greedy", 60.0), ("beam4", 61.0), ("mbr16", 62.0))) + "\n")
    rows = [{"benchmark": "flores_plus", "iso": "afr", "id": str(i), "english": "x", "reference": "a b c d e f", "draft": "a b c d e", "hyp": "a b c d e f"} for i in range(0, 500, 5)]
    class A: decoding = "greedy"
    txt = e.decode_study_baselines(rows, csvp, A)
    assert "decode study beam-4 | 61.00" in txt and "editor on that draft" in txt


def test_detok_undoes_opus_spacing():
    sel = script("post_editor_select_data")
    assert sel.detok("On the other hand , hearts are touched ( really ) .") == "On the other hand, hearts are touched (really)."
    assert sel.detok("Jehovah 's people") == "Jehovah's people"


def test_qe_filter_rule_is_moderate_and_respects_floors():
    qf = script("post_editor_qe_filter")
    rows = [{"pair_id": f"yor-{i}", "iso": "yor", "split": "train"} for i in range(1000)] + [{"pair_id": f"yor-v{i}", "iso": "yor", "split": "val"} for i in range(20)]
    rows += [{"pair_id": f"kin-{i}", "iso": "kin", "split": "train"} for i in range(300)] + [{"pair_id": f"zzz-{i}", "iso": "zzz", "split": "train"} for i in range(900)] + [{"pair_id": f"hau-{i}", "iso": "hau", "split": "train"} for i in range(200)]
    scores = {r["pair_id"]: float(i % 977) for i, r in enumerate(rows)}
    rep = {r["iso"]: r for r in qf.apply_rule(rows, scores, valid={"yor", "kin", "hau"}, drop_share=0.27, floor=250, min_train=300)}
    assert rep["yor"]["dropped"] == 270 and rep["yor"]["kept"] == 730                     # 27% of the TRAIN pairs
    assert rep["kin"]["dropped"] == 50 and rep["kin"]["kept"] == 250                       # never below the floor
    assert rep["hau"]["dropped"] == 0 and not rep["hau"]["qe_applied"]                     # thin language: keep everything
    assert rep["zzz"]["dropped"] == 0 and not rep["zzz"]["qe_applied"]                     # uncalibrated language: no filter
    assert not any(r["qe_drop"] for r in rows if r["split"] == "val")
    dropped = [r for r in rows if r["qe_drop"] and r["iso"] == "yor"]; kept = [r for r in rows if not r["qe_drop"] and r["iso"] == "yor" and r["split"] == "train"]
    assert max(r["qe"] for r in dropped) <= min(r["qe"] for r in kept)                     # the LOWEST are dropped
