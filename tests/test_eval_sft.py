# SPDX-License-Identifier: AGPL-3.0-or-later
"""The post-SFT benchmark evaluation: data download layout, contamination flags, subsets, vLLM-path generation, caches, markers, report.
All CPU, no network, no GPU: fakes for tokenizer / engine / scorers."""
import json
import os
import sys
import time

import pytest

from gemma_mt.eval import benchmarks as B
from gemma_mt.eval import contamination as C
from gemma_mt.eval import fetch_data as FD
from gemma_mt.eval import runtime as RT


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ------------------------------------------------------------------------------------------------ language table / loaders
def test_benchmark_codes_checked_against_the_real_listings():
    from gemma_mt.languages import LANGUAGES

    assert LANGUAGES["sna"].ntrex == "sna-Latn" and LANGUAGES["twi"].flores == "twi_Latn_akua1239"
    assert LANGUAGES["luo"].mafand == "luo" and LANGUAGES["bem"].ntrex == "bem" and LANGUAGES["kea"].flores == "kea_Latn"
    assert LANGUAGES["orm"].flores == "gaz_Latn" and LANGUAGES["efi"].flores is None and LANGUAGES["efi"].mafand is None
    assert len(B.languages_with_data("flores_plus")) >= 44 and len(B.languages_with_data("ntrex")) >= 25
    assert set(B.languages_with_data("mafand")) >= {"yor", "hau", "ibo", "pcm", "twi", "swh", "kin", "amh", "lug", "zul", "xho", "sna", "nya", "tsn", "luo"}


def test_mafand_json_lines_in_a_dot_json_file_are_read(tmp_path):
    d = tmp_path / "mafand"
    _write(str(d / "en-yor" / "test.json"), '{"translation": {"en": "Hello", "yor": "Bawo"}}\n{"translation": {"en": "Bye", "yor": "O dabo"}}\n')
    rows = B.load_mafand(["yor"], "test", str(d))
    assert [(r["english"], r["reference"], r["id"]) for r in rows] == [("Hello", "Bawo", "0"), ("Bye", "O dabo", "1")]


def test_fetch_data_layout_is_what_the_loaders_read(tmp_path):
    calls = []

    def fake_get(url, dest, retries=3, timeout=120.0):
        calls.append(url)
        if "ref.lug" in url or "en-xho" in url:
            return False                                                       # not upstream
        if "newstest2019-src.eng" in url:
            body = "one\ntwo\n"
        elif "newstest2019-ref" in url:
            body = "uno\ndos\n"
        else:
            body = json.dumps({"translation": {"en": "a", "xx": "b"}}) + "\n"
        _write(dest, body)
        return True

    real = FD._http_get
    FD._http_get = fake_get
    try:
        _fetch_checks(tmp_path, calls)
    finally:
        FD._http_get = real


def _fetch_checks(tmp_path, calls):
    r1 = FD.fetch_ntrex_raw(str(tmp_path / "ntrex"), ["yor", "sna", "lug"])
    assert r1["fetched"] == ["yor", "sna-Latn"] and r1["missing"] == ["lug"]
    rows = B.load_benchmark("ntrex", path=str(tmp_path / "ntrex"), isos=["yor", "sna"])
    assert {r["iso"] for r in rows} == {"yor", "sna"} and len(rows) == 4
    n = len(calls)
    FD.fetch_ntrex_raw(str(tmp_path / "ntrex"), ["yor", "sna"])               # idempotent: files exist
    assert len(calls) == n
    r2 = FD.fetch_mafand_raw(str(tmp_path / "mafand"), ["yor", "xho"], "test")
    assert r2 == {"fetched": ["yor"], "missing": ["xho"]}
    assert os.path.exists(tmp_path / "mafand" / "en-yor" / "test.json")
    with pytest.raises(ValueError):
        FD.fetch_flores_plus_files(str(tmp_path / "f"), ["yor"], "dev")        # dev is calibration data: never downloaded here


# ------------------------------------------------------------------------------------------------ contamination
def test_normalise_is_nfkc_casefold_punctuation_free():
    assert C.normalise("  Hello,   WORLD!! (it’s) ") == "hello world it s"
    assert C.normalise("ﬁne print") == C.normalise("fine print")          # NFKC: ligature and no-break space
    assert C.normalise("") == "" and C.normalise(None) == ""


def _bench_rows():
    return [
        {"english": "The cat sat on the mat near the old red barn today.", "reference": "yor cat ref", "iso": "yor", "benchmark": "t", "id": "1"},
        {"english": "The cat sat on the mat near the old red barn today.", "reference": "hau cat ref", "iso": "hau", "benchmark": "t", "id": "1"},
        {"english": "Completely unrelated sentence about quantum chromodynamics lattices.", "reference": "r2", "iso": "yor", "benchmark": "t", "id": "2"},
        {"english": "First part of a long text. Second part is here too.", "reference": "r3", "iso": "yor", "benchmark": "t", "id": "3"},
        {"english": "One two three four five six seven eight nine ten eleven twelve.", "reference": "r4", "iso": "yor", "benchmark": "t", "id": "4"},
        {"english": "Short one.", "reference": "r5", "iso": "yor", "benchmark": "t", "id": "5"},
    ]


def test_scan_windows_sentence_run_near_and_pair_rules():
    rows = _bench_rows()
    idx = C.BenchIndex(rows)
    assert len(idx.texts) == 5                                                  # the cat sentence is shared by two languages
    windows = [
        # whole sentence of a window (sentence rule) -> also the pair for yor (reference occurs in the target)
        ("Intro text. The cat sat on the mat near the old red barn today. Outro.", "x. yor cat ref. y", "yor"),
        # a benchmark row made of two sentences, contained in a window as a contiguous run (the sentence rule alone cannot see it)
        ("Before. First part of a long text. Second part is here too. After.", "t", "yor"),
        # near duplicate: 8 of the 12 tokens' 8-grams? the window shares the first 11 tokens, so most 8-grams match
        ("One two three four five six seven eight nine ten eleven and then something else.", "t", "yor"),
        # unrelated
        ("Nothing to see here at all in this window of text, really.", "t", "hau"),
    ]
    res = C.scan_windows(idx, windows)
    ids = idx.ids
    cat = ids[C.normalise(rows[0]["english"])]
    two = ids[C.normalise(rows[3]["english"])]
    ten = ids[C.normalise(rows[4]["english"])]
    shortid = ids[C.normalise(rows[5]["english"])]
    assert cat in res.exact and cat in res.sentence
    assert two in res.exact and two not in res.sentence                         # contiguous run, not a window sentence
    assert ten not in res.exact and shortid not in res.exact
    assert (cat, "yor") in res.pair and (cat, "hau") not in res.pair
    flags = C.flags_for_rows(rows, idx, res)
    assert flags["exact"] == {"yor:1": True, "hau:1": True, "yor:2": False, "yor:3": True, "yor:4": False, "yor:5": False}
    assert flags["sentence"]["yor:3"] is False and flags["sentence"]["yor:1"] is True
    assert flags["near"]["yor:4"] is True and flags["near"]["yor:2"] is False
    assert flags["pair"]["yor:1"] is True and flags["pair"]["hau:1"] is False
    summ = C.summarise(rows, flags)
    assert summ["total"] == {"n": 6, "exact": 3, "sentence": 2, "near": 1, "pair": 1}


def test_scan_parquet_dir_and_flag_files_roundtrip(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    rows = _bench_rows()
    wdir = tmp_path / "windows"
    wdir.mkdir()
    for k in range(2):
        t = pa.table({"english": ["Intro. The cat sat on the mat near the old red barn today. Outro." if k == 0 else "zzz"],
                      "target": ["yor cat ref"], "iso": ["yor"]})
        pq.write_table(t, str(wdir / f"windows-{k}.parquet"))
    idx = C.BenchIndex(rows)
    res = C.scan_parquet_dir(idx, str(wdir), processes=2, log=lambda m: None)
    assert res.windows == 2
    flags = C.flags_for_rows(rows, idx, res)
    path = str(tmp_path / "bench" / "contamination_t.json")
    meta = C.write_flag_file(path, "t", rows, flags, {"training_text": "x"})
    assert meta["total"]["exact"] == 2
    ev = [dict(r) for r in rows]
    assert C.apply_flags(ev, {"t": path}) == {}
    assert [r["contaminated"] for r in ev] == [True, True, False, False, False, False]
    assert ev[0]["contaminated_sentence"] is True and ev[0]["pair_seen"] is True and ev[1]["pair_seen"] is False
    other = [dict(rows[0], benchmark="u"), dict(rows[0], id="999")]
    miss = C.apply_flags(other, {"t": path})
    assert miss == {"u": 1} or miss == {"u": 1, "t": 1}
    assert other[0]["contaminated"] is None and other[1]["contaminated"] is None, "unknown rows are never silently clean"
    assert C.default_flag_files(str(tmp_path / "bench"), ["t", "zz"]) == {"t": path}


# ------------------------------------------------------------------------------------------------ evaluate_rows with subsets
def _flagged_rows_and_outputs():
    rows, outs = [], []
    for iso in ("yor", "hau", "ibb"):
        for i in range(40):
            ref = f"the reference sentence number {i} for {iso}"
            hyp = ref if i % 2 == 0 else ref.replace("reference", "x")
            cont = i < 10                                                       # first 10 rows are contaminated
            rows.append({"english": f"source {i}", "reference": ref, "iso": iso, "benchmark": "flores_plus", "id": str(i),
                         "contaminated": cont, "contaminated_sentence": cont and i < 5, "near_dup": i == 12, "pair_seen": i < 3})
            outs.append({"hyp": hyp, "raw": "yor " + hyp, "prefix_status": "ok" if i % 4 else "missing", "finished": i != 7})
    return rows, outs


def test_evaluate_rows_clean_subset_and_macros_and_csv(tmp_path):
    pytest.importorskip("sacrebleu")
    from gemma_mt.eval import eval_translation as E
    from gemma_mt.rl.comet import FakeScorer

    rows, outs = _flagged_rows_and_outputs()
    sc = FakeScorer()
    res = E.evaluate_rows(rows, outs, comet_ref=lambda s, m, r: sc(s, m), comet_qe=lambda s, m: sc(s, m), min_subset_rows=30)
    yor = res["benchmarks"]["flores_plus"]["languages"]["yor"]
    assert yor["n"] == 40 and yor["clean"]["n"] == 30 and yor["n_contaminated"] == 10 and yor["n_sentence_rule"] == 5 and yor["n_pair"] == 3
    assert yor["clean_sentence"]["n"] == 35 and yor["clean_strict"]["n"] == 29 and yor["no_pair"]["n"] == 37
    assert "spbleu" in yor["clean"] and "spbleu" not in yor["clean_strict"] and "chrf" in yor["clean_strict"], "slow metrics only for all / clean"
    assert "comet" in yor["clean"] and "comet" not in res["benchmarks"]["flores_plus"]["languages"]["ibb"]["clean"], "AfriCOMET only where supported"
    mac = res["benchmarks"]["flores_plus"]
    assert mac["macro"]["n_languages"] == 3 and mac["macro_clean"]["n_languages"] == 3 and mac["macro_clean"]["n_rows"] == 90
    assert mac["macro_clean"]["chrf"] != mac["macro"]["chrf"]
    # a language below the minimum is left out of the subset macro
    res2 = E.evaluate_rows(rows, outs, min_subset_rows=31)
    assert res2["benchmarks"]["flores_plus"]["macro_clean"]["n_languages"] == 0
    assert res["macro_all_clean"]["chrf_n_languages"] == 3
    # prefix / finished / empty on the clean subset use the same rows
    assert yor["clean"]["finished_rate"] == pytest.approx(1.0) and yor["finished_rate"] == pytest.approx(39 / 40)
    flat = E.csv_rows(res)
    assert {(r["iso"], r["subset"]) for r in flat if r["iso"] in ("yor", "MACRO")} >= {("yor", "all"), ("yor", "clean"), ("MACRO", "clean")}
    E.write_csv(str(tmp_path / "r.csv"), {}, systems={"sft": res, "base": res})
    head = open(tmp_path / "r.csv").readline().strip().split(",")
    assert head[0] == "system" and "subset" in head and "n_contaminated" in head


def test_subsets_exist_only_when_every_row_of_the_benchmark_is_flagged():
    pytest.importorskip("sacrebleu")
    from gemma_mt.eval import eval_translation as E

    rows, outs = _flagged_rows_and_outputs()
    rows[5]["contaminated"] = None
    res = E.evaluate_rows(rows, outs)
    assert "clean" not in res["benchmarks"]["flores_plus"]["languages"]["yor"] and "macro_clean" not in res["benchmarks"]["flores_plus"]
    assert "clean_sentence" in res["benchmarks"]["flores_plus"]["languages"]["yor"]


def test_scoring_is_chunked_cached_and_respects_the_deadline():
    pytest.importorskip("sacrebleu")
    from gemma_mt.eval import eval_translation as E

    rows, outs = _flagged_rows_and_outputs()
    calls = []

    def ref(s, m, r):
        calls.append(len(s))
        return [0.5] * len(s)

    E.evaluate_rows(rows, outs, comet_ref=ref, comet_chunk=50)
    assert sum(calls) == 80 and max(calls) <= 50, "only yor and hau are AfriCOMET languages: 80 rows in chunks of <= 50"
    with pytest.raises(E.TimeLimitExceeded):
        E.evaluate_rows(rows, outs, comet_ref=ref, comet_chunk=10, deadline=time.time() - 1)


# ------------------------------------------------------------------------------------------------ runtime helpers
def test_cached_scorer_scores_each_pair_once_and_survives_a_restart(tmp_path):
    seen = []

    def fn(s, m, r=None):
        seen.append(len(s))
        return [float(len(x) + len(y)) for x, y in zip(s, m)]

    path = str(tmp_path / "c.jsonl")
    c = RT.CachedScorer(fn, path, reference_based=False)
    assert c(["a", "bb"], ["x", "yy"]) == [2.0, 4.0]
    assert c(["a", "ccc"], ["x", "z"]) == [2.0, 4.0] and seen == [2, 1]
    c.close()
    with open(path, "a") as f:
        f.write('["half-written')                                                # a torn last line must not break the restart
    c2 = RT.CachedScorer(fn, path, reference_based=False)
    assert c2(["a", "bb", "ccc"], ["x", "yy", "z"]) == [2.0, 4.0, 4.0] and seen == [2, 1] and c2.hits == 3
    r = RT.CachedScorer(lambda s, m, rr: [float(len(x)) for x in rr], None, reference_based=True)
    assert r(["a"], ["b"], ["refref"]) != r(["a"], ["b"], ["r"]) and r.key("a", "b", "r1") != r.key("a", "b", "r2")


def test_wait_for_gpu_bounded_and_watchdog_fires():
    free = iter([3.0, 3.0, 30.0])
    t = [0.0]
    out = RT.wait_for_gpu(24, 100, poll_s=1, probe=lambda: next(free), sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])
    assert out == 30.0 and t[0] == 2.0
    t[0] = 0.0
    with pytest.raises(TimeoutError):
        RT.wait_for_gpu(24, 5, poll_s=1, probe=lambda: 1.0, sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])
    assert RT.wait_for_gpu(24, 5, probe=lambda: None) != RT.wait_for_gpu(24, 5, probe=lambda: None)         # nan without nvidia-smi
    fired = []
    dog = RT.Watchdog(0.05, lambda: fired.append("cb"), exit_fn=lambda code: fired.append(code)).start()
    time.sleep(0.4)
    assert fired == ["cb", 3]
    dog2 = RT.Watchdog(0.05, lambda: fired.append("no"), exit_fn=lambda code: fired.append("no")).start()
    dog2.cancel()
    time.sleep(0.2)
    assert "no" not in fired


def test_worker_scorer_reference_based_and_timeout(tmp_path):
    from gemma_mt.rl.comet import WorkerScorer

    w = WorkerScorer(fake=True, reference_based=True, startup_timeout=60, request_timeout=60)
    assert len(w(["a b"], ["c d"], ["e"])) == 1
    w.close()
    # a worker that never answers is killed and reported, not waited for
    hang = tmp_path / "hang.py"
    hang.write_text("import sys, time\ntime.sleep(60)\n")
    import subprocess

    ws = WorkerScorer.__new__(WorkerScorer)
    import queue

    ws._lines = queue.Queue()
    ws.proc = subprocess.Popen([sys.executable, str(hang)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    t0 = time.time()
    with pytest.raises(RuntimeError, match="did not answer"):
        ws._readline(0.5, "scoring")
    assert time.time() - t0 < 5 and ws.proc.poll() is not None


# ------------------------------------------------------------------------------------------------ vLLM-path generation (HF-backed stand-in engine)
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_generate_translations_through_the_sampler_chunks_and_matches_hf():
    from gemma_mt.eval.eval_translation import TimeLimitExceeded, generate_translations
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.modeling import build_tiny_model
    from gemma_mt.rl.vllm_sampler import HFBackedEngine, VLLMSampler

    tok = FakeGemmaTokenizer()
    rows = [{"english": f"Good morning number {k}", "reference": "r", "iso": i, "benchmark": "b", "id": str(k)}
            for k, i in enumerate(["yor", "hau", "ibo", "yor", "hau"])]
    eng = HFBackedEngine("tiny-random")
    sampler = VLLMSampler(eng)
    with pytest.raises(Exception):
        generate_translations(None, tok, rows, stop_ids=[106, 1], pad_id=0, device=None, sampler=sampler)    # never synced -> refuses
    sampler.assume_loaded(0)
    a = generate_translations(None, tok, rows, stop_ids=[106, 1], pad_id=0, device=None, max_new_tokens=6, sampler=sampler, chunk=2)
    assert eng.generate_calls == 3 and len(a) == 5
    b = generate_translations(eng.model, tok, rows, stop_ids=[106, 1], pad_id=0, device="cpu", batch_size=2, max_new_tokens=6)
    assert [x["raw"] for x in a] == [x["raw"] for x in b], "the engine path and the HF path decode the same greedy text"
    assert set(a[0]) == {"raw", "hyp", "prefix_status", "finished"}
    with pytest.raises(TimeLimitExceeded):
        generate_translations(None, tok, rows, stop_ids=[106, 1], pad_id=0, device=None, sampler=sampler, deadline=time.time() - 1)


# ------------------------------------------------------------------------------------------------ the whole pipeline with fakes
def _local_benchmarks(root):
    en = [f"Sentence number {i} about the weather in the capital city of the country." for i in range(6)]
    for iso, code in (("yor", "yor_Latn"), ("hau", "hau_Latn")):
        _write(os.path.join(root, "flores_plus", "devtest", f"{code}.jsonl"),
               "".join(json.dumps({"id": i, "text": f"{iso} translation {i}"}) + "\n" for i in range(6)))
    _write(os.path.join(root, "flores_plus", "devtest", "eng_Latn.jsonl"), "".join(json.dumps({"id": i, "text": e}) + "\n" for i, e in enumerate(en)))
    for iso, code in (("yor", "yor"), ("hau", "hau")):
        _write(os.path.join(root, "mafand", f"en-{code}", "test.json"),
               "".join(json.dumps({"translation": {"en": f"News {iso} {i} told the reporters.", code: f"{iso} labaru {i}"}}) + "\n" for i in range(4)))


def _hooks(tmp_path, comet=True):
    from gemma_mt.eval import sft_benchmarks as S
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.format import StopTokens
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.rl.vllm_sampler import HFBackedEngine, VLLMSampler

    class FakeComet:
        def __init__(self):
            self.sc = FakeScorer()
            self.closed = False

        def reference(self, s, m, r):
            return self.sc(s, m)

        def qe(self, s, m):
            return self.sc(s, m)

        def stats(self):
            return {"fake": True}

        def close(self):
            self.closed = True

    made = []

    def make_sampler(path, **kw):
        s = VLLMSampler(HFBackedEngine("tiny-random"))
        s.assume_loaded(0)
        made.append(kw)
        return s

    h = S.Hooks(resolve_model=lambda spec, token=None, revision=None: RT.ModelRef(spec, "tiny-random", spec, revision, "a" * 40),
                load_tokenizer_and_stops=lambda path: (FakeGemmaTokenizer(), StopTokens(106, 1, [106, 1]), 0),
                make_sampler=make_sampler, comet_factory=(lambda args, out: FakeComet()) if comet else None,
                ensure_data=lambda root, names, isos, token: None, wait_for_gpu=lambda need, wait: float("nan"))
    h.made = made
    return h


def _args(tmp_path, *extra):
    from gemma_mt.eval import sft_benchmarks as S

    bench = tmp_path / "bench"
    _local_benchmarks(str(bench / "test_sets"))
    a = S.build_parser().parse_args(["--out", str(tmp_path / "out"), "--bench-dir", str(bench), "--vllm-python", "x", "--max-new-tokens", "6",
                                     "--benchmarks", "flores_plus", "mafand", "--min-subset-rows", "2", "--unit-rows", "5", *extra])
    return a


def _make_flags(tmp_path):
    bench = tmp_path / "bench"
    for name in ("flores_plus", "mafand"):
        rows = B.load_benchmark(name, path=str(bench / "test_sets" / name))
        idx = C.BenchIndex(rows)
        wins = [(rows[0]["english"], "t", rows[0]["iso"]), (rows[3]["english"], "t", rows[3]["iso"])]       # two sentences are "in the training text"
        C.write_flag_file(str(bench / f"contamination_{name}.json"), name, rows, C.flags_for_rows(rows, idx, C.scan_windows(idx, wins)))


def test_pipeline_end_to_end_markers_outputs_resume_and_report(tmp_path):
    pytest.importorskip("sacrebleu")
    from gemma_mt.eval import sft_benchmarks as S

    a = _args(tmp_path)
    _make_flags(tmp_path)
    h = _hooks(tmp_path)
    rc = S.run(a, h)
    out = tmp_path / "out"
    assert rc == 0 and (out / "EVAL_DONE").exists() and not (out / "EVAL_FAILED").exists() and not (out / "EVAL_RUNNING").exists()
    res = json.load(open(out / "results.json"))
    assert set(res["systems"]) == {"sft", "base"} and res["failures"] == []
    fl = res["systems"]["sft"]["benchmarks"]["flores_plus"]
    assert fl["languages"]["yor"]["n"] == 6 and fl["languages"]["yor"]["clean"]["n"] < 6 and "macro_clean" in fl
    assert res["systems"]["sft"]["meta"]["sha"] == "a" * 40 and res["systems"]["sft"]["meta"]["sampler"] == "vllm"
    preds = [json.loads(l) for l in open(out / "predictions.jsonl")]
    assert {p["system"] for p in preds} == {"sft", "base"} and "contaminated" in preds[0] and len(preds) == 2 * (12 + 8)
    assert "system" in open(out / "results.csv").readline()
    md = open(out / "report.md").read()
    assert "Macro averages, clean" in md and "Contamination" in md and "Output format failures" in md and "Runtime" in md
    assert len(h.made) == 2                                                    # one engine per system
    # a second run resumes from the caches: no engine is started
    h2 = _hooks(tmp_path)
    assert S.run(_args_again(a), h2) == 0 and h2.made == []
    # EVAL_DONE is rewritten, results identical
    assert json.load(open(out / "results.json"))["systems"]["sft"]["benchmarks"]["flores_plus"]["macro"]["chrf"] == fl["macro"]["chrf"]


def _args_again(a):
    return a


def test_pipeline_failure_modes_write_eval_failed_and_never_claim_success(tmp_path):
    pytest.importorskip("sacrebleu")
    from gemma_mt.eval import sft_benchmarks as S

    # (1) missing contamination files abort before any GPU work
    a = _args(tmp_path)
    h = _hooks(tmp_path)
    assert S.run(a, h) == 1
    marker = json.load(open(tmp_path / "out" / "EVAL_FAILED"))
    assert "contamination" in marker["reason"] and h.made == [] and not (tmp_path / "out" / "EVAL_DONE").exists()
    # (2) the engine cannot start: a failure for the system, partial results of the other one still written, EVAL_FAILED
    _make_flags(tmp_path)
    h2 = _hooks(tmp_path)
    good = h2.make_sampler
    calls = []

    def flaky(path, **kw):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError("vLLM worker exited with code 1 before connecting")
        return good(path, **kw)

    h2.make_sampler = flaky
    a2 = _args(tmp_path, "--no-resume")
    assert S.run(a2, h2) == 1
    marker = json.load(open(tmp_path / "out" / "EVAL_FAILED"))
    assert "before connecting" in marker["reason"] and marker["partial_results"] is True
    res = json.load(open(tmp_path / "out" / "results.json"))
    assert set(res["systems"]) == {"sft"} and res["failures"]
    # (3) fallback to HF is explicit and recorded, never silent
    h3 = _hooks(tmp_path)
    h3.make_sampler = lambda path, **kw: (_ for _ in ()).throw(RuntimeError("no vllm"))
    h3.load_hf_model = lambda path, dtype: __import__("gemma_mt.modeling", fromlist=["x"]).build_tiny_model()
    a3 = _args(tmp_path, "--no-resume", "--sampler-fallback-hf")
    assert S.run(a3, h3) == 0                                                  # complete results: EVAL_DONE, with the degradation recorded as a warning
    res3 = json.load(open(tmp_path / "out" / "results.json"))
    assert res3["systems"]["sft"]["meta"]["sampler"] == "hf" and any("HF fallback" in w for w in res3["warnings"]) and res3["failures"] == []
    assert "no vllm" in json.load(open(tmp_path / "out" / "EVAL_DONE"))["warnings"][0]


def test_budget_exits_with_75_and_continues_from_the_caches(tmp_path):
    pytest.importorskip("sacrebleu")
    from gemma_mt.eval import sft_benchmarks as S

    a = _args(tmp_path, "--budget-seconds", "0.001")
    _make_flags(tmp_path)
    h = _hooks(tmp_path)
    time.sleep(0.01)
    assert S.run(a, h) == 75
    out = tmp_path / "out"
    assert (out / "EVAL_RUNNING").exists() and not (out / "EVAL_DONE").exists() and not (out / "EVAL_FAILED").exists()
    a.budget_seconds = 0
    assert S.run(a, _hooks(tmp_path)) == 0 and (out / "EVAL_DONE").exists()


def test_generation_units_never_split_a_language():
    from gemma_mt.eval.sft_benchmarks import make_units

    rows = [{"iso": i} for i in ["a"] * 5 + ["b"] * 3 + ["c"] * 6 + ["d"] * 1]
    units = make_units(rows, 8)
    assert [len(u) for u in units] == [8, 7] and sum(units, []) == list(range(15))
    assert [len(u) for u in make_units(rows, 1)] == [5, 3, 6, 1] and make_units([], 5) == []


def test_report_renders_without_comet_or_clean_subset():
    from gemma_mt.eval.sft_report import render_report

    res = {"meta": {"seconds": 1.0, "rows": {"ntrex": 2}}, "failures": [],
           "systems": {"sft": {"meta": {"model": "m", "sha": "abc", "timings": {"ntrex": {"sampler": "hf", "generation_seconds": 1.0}}},
                               "benchmarks": {"ntrex": {"languages": {"yor": {"n": 2, "chrf": 10.0, "prefix_ok_rate": 1.0, "finished_rate": 1.0,
                                                                              "empty_rate": 0.0}},
                                                        "macro": {"chrf": 10.0, "n_languages": 1, "chrf_n_languages": 1}}}}}}
    md = render_report(res)
    assert "ntrex" in md and "10.00" in md
