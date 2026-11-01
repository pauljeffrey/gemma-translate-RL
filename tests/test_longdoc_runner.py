# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-44: the pseudo-reference runner (gemma_mt/eval/longdoc_runner.py, scripts/longdoc_translate.py): stub-engine end to end (paragraphs, a very long sentence), resume after
a crash / a graceful stop, progress.log format, progress.json schema, unsupported-language skipping, OOM halving, failure isolation, scatter correctness, torn files,
and - when the libraries are there - a real (tiny, random-weight) HF Marian model on the CPU, the real opus-mt-en-ig if it is in the local HF cache, and a real
CTranslate2 CPU run of a converted tiny model.  No network, no big weights."""
import contextlib
import io
import json
import os
import re
from types import SimpleNamespace

import pytest

from gemma_mt.eval import longdoc as L
from gemma_mt.eval import longdoc_runner as R


class SimulatedCrash(BaseException):
    """Like a kill -9 / power loss: not an Exception, so the runner's failure isolation must NOT swallow it."""


LONG = "Word0 " + " ".join(f"word{i}" for i in range(1, 260))             # one 1.9k-character 'sentence' without punctuation
DOCS = [
    {"doc_id": "d1", "text": "The cat sat. It was happy.\n\nA second paragraph here. And another sentence!", "bin": 512},
    {"doc_id": "d2", "text": "Single sentence item.", "bin": 64},
    {"doc_id": "d3", "text": "The cat sat. " + LONG + ". Final words.", "bin": 1024},
    {"doc_id": "d4", "text": "The cat sat. Tiny.\n\nPara two is longer than the first by quite a lot of words indeed.", "bin": 512},
    {"doc_id": "d5", "text": "Zebras run fast. Lions sleep all day long in the sun. Birds sing. The river flows to the sea. Rain fell.", "bin": 256},
]


def stub_text(s, iso):
    return " ".join(f"{w}{iso}" for w in reversed(s.split()))


def make_spec(name="stubsys", codes=("yor", "hau", "ibo"), **kw):
    base = dict(name=name, hf_id="", revision="rev123", family="stub", prompt_style="nllb", template="", hf_id_template="", backends=("stub",), params_b=1.0, max_input_tokens=512,
                src_code="eng_Latn", codes={c: c for c in codes}, proxy_codes={"swc": "swh"}, weights_licence="n/a", licence_class="permissive")
    base.update(kw)
    return SimpleNamespace(**base)


class Clock:
    def __init__(self):
        self.t = 1_700_000_000.0

    def __call__(self):
        self.t += 0.7
        return self.t


def write_docs(tmp_path, docs=DOCS):
    p = str(tmp_path / "docs.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return p


def make_runner(tmp_path, out, spec=None, langs=("yor",), docs=DOCS, hook=None, engines=None, **kw):
    spec = spec or make_spec()
    dp = write_docs(tmp_path, docs)
    loaded = R.load_source_docs([dp])
    limit = kw.pop("engine_limit", None)

    def factory(sp):
        e = R.StubEngine(sp, on_batch=hook, max_input_tokens=limit)
        if engines is not None:
            engines.append(e)
        return e

    args = dict(engine_factory=factory, langs=list(langs), window_docs=2, max_batch_sents=2, flush_sentences=1, progress_every=0.0, echo=None, install_signals=False, clock=Clock())
    args.update(kw)
    return R.Runner(loaded, str(out), [spec], **args), loaded, dp


def rows_of(out, system="stubsys", iso="yor"):
    return L.read_jsonl(os.path.join(str(out), system, f"{iso}.jsonl"))


def strip(rows):
    return [{k: v for k, v in r.items() if k != "created"} for r in rows]


# ------------------------------------------------------------------------------------------------ end to end
def test_stub_end_to_end_matches_translate_documents(tmp_path):
    out = tmp_path / "out"
    runner, docs, dp = make_runner(tmp_path, out)
    assert runner.run() == 0
    rows = rows_of(out)
    assert [r["doc_id"] for r in rows] == [d["doc_id"] for d in DOCS]                       # document order
    ref = L.translate_documents(DOCS, lambda ss: [stub_text(s, "yor") for s in ss], iso="yor", system="stubsys")
    for r, t in zip(rows, ref):
        assert r["src_sents"] == t.src_sents and r["tgt_sents"] == t.tgt_sents and r["para_starts"] == t.para_starts and r["flags"] == t.flags
        assert len(r["tgt_sents"]) == len(r["src_sents"]) == len(r["flags"])
        for k in ("doc_id", "iso", "system", "flagged_share", "valid", "n_src_tokens", "n_tgt_tokens", "backend", "revision", "created", "bin", "code"):
            assert k in r
        assert r["backend"] == "stub" and r["revision"] == "rev123" and r["system"] == "stubsys" and r["iso"] == "yor"
    d1 = next(r for r in rows if r["doc_id"] == "d1")
    assert d1["para_starts"] == [0, 2]                                                      # paragraphs survive
    d3 = next(r for r in rows if r["doc_id"] == "d3")
    assert len(d3["src_sents"]) == 3 and len(d3["src_sents"][1]) > 1500 and len(d3["tgt_sents"][1].split()) == 260    # the long sentence is cut for the call, re-joined whole
    # refs file = the format of longdoc.load_refdocs
    refs_path = os.path.join(str(out), "stubsys", "yor.refs.jsonl")
    for r in L.read_jsonl(refs_path):
        assert r["ref_kind"] == "pseudo" and r["ref_system"] == "stubsys" and "ref_sents" in r and "valid" in r
    # load_refdocs needs the document's `sents` (docs.jsonl of the source set carries them): write a source set the way the selection step does
    sp = str(tmp_path / "docs_with_sents.jsonl")
    L.write_jsonl(sp, [{**d, "sents": x["sents"], "para_starts": x["para_starts"]} for d, x in zip(DOCS, docs)])
    rd = L.load_refdocs(sp, refs_path)
    assert len(rd) == len(DOCS) and all(x.ref_kind == "pseudo" and x.ref_system == "stubsys" for x in rd)
    summ = json.load(open(os.path.join(str(out), "stubsys", "yor.summary.json")))
    assert summ["docs"] == len(DOCS) and summ["sentences"] == sum(len(d["sents"]) for d in docs) and summ["errors"] == 0


def test_token_limit_splits_overlong_pieces(tmp_path):
    out = tmp_path / "out"
    engines = []
    runner, docs, _ = make_runner(tmp_path, out, engines=engines, engine_limit=40)
    runner.run()
    sent = [x for e in engines for call in e.calls for x in call]
    assert sent and all(len(x.split()) + 1 <= 40 for x in sent)                             # every unit fits the model's input limit
    d3 = next(r for r in rows_of(out) if r["doc_id"] == "d3")
    assert len(d3["tgt_sents"][1].split()) == 260       # all words kept


def test_scatter_correct_with_length_sorted_batches(tmp_path):
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split()
    docs = []
    for i in range(6):                                       # sentences of very different lengths in shuffled order, with duplicates across documents
        sents = [" ".join(words[: (j * 7 + i * 3) % 11 + 1]).capitalize() + "." for j in range(5)]
        docs.append({"doc_id": f"x{i}", "text": " ".join(sents)})
    out = tmp_path / "out"
    runner, loaded, _ = make_runner(tmp_path, out, docs=docs, langs=("yor", "hau"), window_docs=3, max_batch_sents=3)
    runner.run()
    for iso in ("yor", "hau"):
        for r, d in zip(rows_of(out, iso=iso), loaded):
            assert r["src_sents"] == d["sents"]
            assert r["tgt_sents"] == [stub_text(s, iso) for s in d["sents"]]


def test_batches_are_longest_first_and_within_budget(tmp_path):
    out = tmp_path / "out"
    engines = []
    runner, docs, _ = make_runner(tmp_path, out, engines=engines, window_docs=0, max_batch_sents=64, max_batch_tokens=40)
    runner.run()
    calls = engines[0].calls
    lens = [[len(x.split()) + 1 for x in c] for c in calls]
    assert all(max(l) * len(l) <= 40 or len(l) == 1 for l in lens)                          # padded tokens within the budget (a single longer item aside)
    assert lens[0][0] == max(max(l) for l in lens)                                          # the longest unit goes first (an OOM would show at once)
    flat = [x for c in calls for x in c]
    assert len(flat) == len(set(flat))                                                      # identical sentences are translated once


# ------------------------------------------------------------------------------------------------ resume
def test_resume_after_crash_is_identical_and_translates_nothing_twice(tmp_path):
    full = tmp_path / "full"
    r0, _, _ = make_runner(tmp_path, full)
    r0.run()
    part = tmp_path / "part"
    seen = []
    n = {"calls": 0}

    def crash(inputs, iso):
        n["calls"] += 1
        if n["calls"] == 4:
            raise SimulatedCrash()
        seen.extend(inputs)

    r1, _, _ = make_runner(tmp_path, part, hook=crash)
    try:
        r1.run()
        raise AssertionError("the crash must propagate")
    except SimulatedCrash:
        pass
    assert not os.path.isfile(os.path.join(str(part), "stubsys", "yor.summary.json"))     # not finished -> no summary
    done_before = len(seen)
    assert 0 < done_before
    seen2 = []
    r2, _, _ = make_runner(tmp_path, part, hook=lambda inputs, iso: seen2.extend(inputs))
    r2.run()
    assert not (set(seen) & set(seen2))                                                     # nothing translated in the first run is translated again
    assert strip(rows_of(part)) == strip(rows_of(full))                                    # identical to the uninterrupted run (document order, same flags / tokens)
    assert json.load(open(os.path.join(str(part), "stubsys", "yor.summary.json")))["docs"] == len(DOCS)
    # third start: everything done, no engine call at all
    seen3 = []
    r3, _, _ = make_runner(tmp_path, part, hook=lambda inputs, iso: seen3.extend(inputs))
    r3.run()
    assert seen3 == []
    assert any("already done" in l for l in r3.prog.lines)


def test_graceful_stop_flushes_and_resumes(tmp_path):
    full = tmp_path / "full"
    make_runner(tmp_path, full)[0].run()
    part = tmp_path / "part"
    holder = {}
    n = {"c": 0}

    def hook(inputs, iso):
        n["c"] += 1
        if n["c"] == 3:
            holder["r"]._on_signal(15, None)                  # what SIGTERM does: ask to stop after this batch

    r1, _, _ = make_runner(tmp_path, part, hook=hook)
    holder["r"] = r1
    assert r1.run() == 0
    pj = json.load(open(os.path.join(str(part), "progress.json")))
    assert pj["state"] == "stopped"
    assert any("stop" in l for l in r1.prog.lines)
    seen = []
    r2, _, _ = make_runner(tmp_path, part, hook=lambda inputs, iso: seen.extend(inputs))
    r2.run()
    assert strip(rows_of(part)) == strip(rows_of(full))
    assert json.load(open(os.path.join(str(part), "progress.json")))["state"] == "done"


def test_torn_lines_are_repaired_on_resume(tmp_path):
    out = tmp_path / "out"
    n = {"c": 0}

    def crash(inputs, iso):
        n["c"] += 1
        if n["c"] == 5:
            raise SimulatedCrash()

    try:
        make_runner(tmp_path, out, hook=crash)[0].run()
    except SimulatedCrash:
        pass
    sdir = os.path.join(str(out), "stubsys")
    with open(os.path.join(sdir, "sentence_cache.jsonl"), "a", encoding="utf-8") as f:
        f.write('{"i": "yor", "k": "abc", "t": "half writ')                                  # a torn final line, no newline
    if os.path.isfile(os.path.join(sdir, "yor.jsonl")):
        with open(os.path.join(sdir, "yor.jsonl"), "a", encoding="utf-8") as f:
            f.write('{"doc_id": "d9", "iso"')
    full = tmp_path / "full"
    make_runner(tmp_path, full)[0].run()
    make_runner(tmp_path, out)[0].run()
    assert strip(rows_of(out)) == strip(rows_of(full))


def test_changed_source_sentences_are_not_reused(tmp_path):
    out = tmp_path / "out"
    make_runner(tmp_path, out)[0].run()
    docs2 = [dict(DOCS[0], text="A different text now. With two sentences."), *DOCS[1:]]
    seen = []
    r, loaded, _ = make_runner(tmp_path, out, docs=docs2, hook=lambda inputs, iso: seen.extend(inputs))
    r.run()
    got = rows_of(out)
    assert got[0]["src_sents"] == loaded[0]["sents"] and got[0]["tgt_sents"] == [stub_text(s, "yor") for s in loaded[0]["sents"]]
    assert set(seen) == {"A different text now.", "With two sentences."}                   # only the changed document is translated


# ------------------------------------------------------------------------------------------------ progress
LINE = re.compile(r"^\d\d:\d\d:\d\d \[stub\] yor  doc +\d+/\d+  sent \d+/\d+  \d+ tok/s  lang \d+%  all \d+%  ETA (\d\d:\d\d|--:--)$")
DONE = re.compile(r"^\d\d:\d\d:\d\d \[stub\] yor  DONE  \d+ docs  \d+ sent  flagged \d+\.\d%  invalid docs \d+  \d+ min  avg \d+ tok/s$")


def test_progress_log_and_json(tmp_path):
    out = tmp_path / "out"
    runner, docs, _ = make_runner(tmp_path, out, spec=make_spec(name="stub", short=None))
    runner.run()
    lines = open(os.path.join(str(out), "progress.log"), encoding="utf-8").read().splitlines()
    prog = [l for l in lines if "tok/s  lang" in l]
    done = [l for l in lines if "  DONE  " in l]
    assert len(prog) >= 3 and all(LINE.match(l) for l in prog), prog[:3]
    assert len(done) == 1 and DONE.match(done[0]), done
    assert re.match(r"^\d\d:\d\d:\d\d ALL DONE", lines[-1])
    first = [int(m.group(1)) for l in prog for m in [re.search(r"all (\d+)%", l)]]
    assert first == sorted(first) and first[-1] <= 100                                      # monotone percent
    pj = json.load(open(os.path.join(str(out), "progress.json")))
    assert set(pj) >= {"updated", "started", "elapsed_s", "state", "current", "totals", "systems"}
    assert set(pj["totals"]) >= {"docs_done", "docs_total", "sent_done", "sent_total", "pct", "eta_s", "eta"}
    st = pj["systems"]["stub"]["yor"]
    assert set(st) >= {"status", "docs_done", "docs_total", "sent_done", "sent_total", "tok_s", "started", "finished", "flagged_share"}
    assert st["status"] == "done" and st["docs_done"] == st["docs_total"] == len(DOCS) and st["sent_done"] == st["sent_total"]
    assert pj["state"] == "done" and pj["totals"]["pct"] == 100.0
    assert not os.path.exists(os.path.join(str(out), "progress.json.tmp"))


def test_eta_helpers():
    assert R.fmt_eta(None) == "--:--" and R.fmt_eta(0) == "00:00" and R.fmt_eta(2 * 3600 + 41 * 60) == "02:41" and R.fmt_eta(150 * 3600) == "150:00"
    e = R.Eta(alpha=0.5, min_samples=2)
    assert e.eta(100) is None                                                               # nothing measured: no invented ETA
    e.update(100, 1.0)
    assert e.eta(100) is None                                                               # one sample is not enough
    e.update(100, 1.0)
    assert e.eta(1000) == 10.0
    e.update(300, 1.0)                                                                      # smoothed: 0.5*300 + 0.5*100 = 200
    assert abs(e.rate - 200.0) < 1e-9


def test_unsupported_languages_are_skipped_with_a_log_line(tmp_path):
    out = tmp_path / "out"
    runner, docs, _ = make_runner(tmp_path, out, spec=make_spec(codes=("yor",)), langs=("yor", "hau", "swc"))
    runner.run()
    assert "skip  stubsys hau  not supported (no code)" in [l.split(" ", 1)[1] for l in runner.prog.lines]
    assert any(l.endswith("skip  stubsys swc  not supported (no code)") for l in runner.prog.lines)     # a proxy code is not support
    assert not os.path.exists(os.path.join(str(out), "stubsys", "hau.jsonl"))
    pj = json.load(open(os.path.join(str(out), "progress.json")))
    assert pj["systems"]["stubsys"]["hau"]["status"] == "skipped" and pj["systems"]["stubsys"]["yor"]["status"] == "done"
    assert pj["totals"]["docs_total"] == len(DOCS)                                          # skipped languages are not in the totals
    # --allow-proxy: swc is translated with its proxy code
    out2 = tmp_path / "out2"
    r2, _, _ = make_runner(tmp_path, out2, spec=make_spec(codes=("yor",)), langs=("yor", "swc"), allow_proxy=True)
    r2.run()
    assert os.path.isfile(os.path.join(str(out2), "stubsys", "swc.jsonl"))


# ------------------------------------------------------------------------------------------------ robustness
def test_oom_halves_the_budget_and_retries(tmp_path):
    out = tmp_path / "out"
    seen = {"ooms": 0}

    def hook(inputs, iso):
        padded = max(len(x.split()) + 1 for x in inputs) * len(inputs)
        if len(inputs) > 1 and padded > 60:
            seen["ooms"] += 1
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 23.69 GiB total capacity)")

    runner, loaded, _ = make_runner(tmp_path, out, hook=hook, max_batch_tokens=400, max_batch_sents=64, window_docs=0)
    runner.run()
    assert seen["ooms"] >= 1 and runner.budget <= 200
    assert any("batch budget halved" in l for l in runner.prog.lines)
    ref = L.translate_documents(DOCS, lambda ss: [stub_text(s, "yor") for s in ss], iso="yor", system="stubsys")
    for r, t in zip(rows_of(out), ref):                                                     # every sentence is translated correctly in spite of the OOMs
        assert r["tgt_sents"] == t.tgt_sents and "error" not in r["flags"]
    assert json.load(open(os.path.join(str(out), "stubsys", "yor.summary.json")))["oom_halvings"] == seen["ooms"]


def test_failing_sentence_is_isolated_and_flagged(tmp_path):
    out = tmp_path / "out"
    docs = [{"doc_id": "a", "text": "Good one. This has POISON inside. Another good one here."}, {"doc_id": "b", "text": "Fine text. Also fine."}]

    def hook(inputs, iso):
        if any("POISON" in x for x in inputs):
            raise ValueError("kernel launch failed")

    runner, loaded, _ = make_runner(tmp_path, out, docs=docs, hook=hook, max_batch_sents=8, window_docs=0)
    runner.run()
    rows = rows_of(out)
    a = rows[0]
    assert a["flags"][1] == "error" and a["tgt_sents"][1] == "" and a["flags"][0] == "" and a["tgt_sents"][0] == stub_text("Good one.", "yor")
    assert rows[1]["flags"] == ["", ""] and not rows[0]["valid"] and rows[1]["valid"]
    assert any(l.split(" ", 1)[1].startswith("retry") for l in runner.prog.lines) and any(" error  " in l for l in runner.prog.lines)
    assert json.load(open(os.path.join(str(out), "stubsys", "yor.summary.json")))["errors"] == 1
    # a later run with --retry-errors translates the sentence again (the failure was transient: no hook now)
    seen = []
    r2, _, _ = make_runner(tmp_path, out, docs=docs, hook=lambda inputs, iso: seen.extend(inputs), retry_errors=True, max_batch_sents=8, window_docs=0)
    r2.run()
    assert seen == ["This has POISON inside."] and rows_of(out)[0]["flags"] == ["", "", ""]


def test_helpers_is_oom_parse_resolve():
    assert R.is_oom(RuntimeError("CUDA out of memory. Tried to allocate")) and R.is_oom(RuntimeError("CUDA failed with error out of memory")) and R.is_oom(MemoryError("std::bad_alloc"))
    assert not R.is_oom(ValueError("shape mismatch"))
    assert R.parse_map_arg(["a=1", "b=2", "3"], "x") == {"a": "1", "b": "2", "": "3"}
    assert R.split_langs(["yor,hau", "ibo"]) == ["yor", "hau", "ibo"] and R.split_langs(None) is None
    n = R.resolve_system("nllb200-3.3b")
    m3, m7 = R.resolve_system("madlad400-3b-mt"), R.resolve_system("madlad400-7b-mt-bt")
    assert n.name == "nllb-200-3.3b" and n.prompt_style == "nllb" and n.src_code == "eng_Latn" and len(n.codes) == 44
    assert m3.prompt_style == "madlad" and len(m3.codes) == 44 and m7.revision and R.short_name(m3) == "madlad3b" and R.short_name(n) == "nllb3b"
    assert n.max_input_tokens == 512 and m3.max_input_tokens == 256
    assert R.resolve_system("stub").codes["yor"] == "yor" and len(R.ISO77) == 77
    ad = R.resolve_system("", hf_id="Helsinki-NLP/opus-mt-en-ig", prompt_style="marian", codes={"ibo": "ibo"})
    assert ad.name == "opus-mt-en-ig" and ad.codes == {"ibo": "ibo"}


def test_dry_run_prints_support_table_and_writes_nothing(tmp_path):
    dp = write_docs(tmp_path)
    out = tmp_path / "outdry"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = R.main(["--docs", dp, "--out", str(out), "--system", "nllb200-3.3b", "--system", "madlad400-3b-mt", "--dry-run", "--langs", "yor", "hau", "kab", "bbj"])
    text = buf.getvalue()
    assert rc == 0 and not os.path.exists(str(out))
    assert "nllb-200-3.3b" in text and "madlad400-3b-mt" in text and "TOTAL" in text and "ASSUMED" in text
    assert re.search(r"nllb-200-3.3b: skipped, no code \(1\): bbj", text) and re.search(r"madlad400-3b-mt: skipped, no code \(2\): kab bbj", text)       # kab is NLLB-only


def test_main_stub_cli_two_runs(tmp_path):
    dp = write_docs(tmp_path)
    out = tmp_path / "cli"
    lines = []
    rc = R.main(["--docs", dp, "--out", str(out), "--system", "stub", "--langs", "yor", "hau", "--window-docs", "2", "--progress-every", "0"], echo=lines.append)
    assert rc == 0 and os.path.isfile(os.path.join(str(out), "stub", "hau.jsonl")) and any("ALL DONE" in l for l in lines)
    lines2 = []
    R.main(["--docs", dp, "--out", str(out), "--system", "stub", "--langs", "yor", "hau"], echo=lines2.append)
    assert sum("already done" in l for l in lines2) == 2
    rc = R.main(["--docs", dp, "--out", str(tmp_path / "lim"), "--system", "stub", "--langs", "yor", "--limit-docs", "2"], echo=lambda s: None)
    assert len(rows_of(tmp_path / "lim", "stub")) == 2
    R.main(["--docs", dp, "--out", str(tmp_path / "lim2"), "--system", "stub", "--langs", "yor", "--limit-sentences", "3"], echo=lambda s: None)
    assert len(rows_of(tmp_path / "lim2", "stub")) == 1                                      # whole documents until >= 3 sentences: d1 alone has 4


# ------------------------------------------------------------------------------------------------ CTranslate2 call shape (fake translator, no library)
def test_ct2_engine_call_shape_without_the_library():
    spec = R.resolve_system("nllb200-3.3b")
    calls = {}

    class Tok:
        def __call__(self, inputs, truncation=True, max_length=None, add_special_tokens=True, **kw):
            calls["max_length"] = max_length
            return {"input_ids": [[7] + [i + 10 for i in range(len(s.split()))] + [2] for s in inputs]}

        def convert_ids_to_tokens(self, ids):
            return [f"t{i}" for i in ids]

        def convert_tokens_to_ids(self, toks):
            return [int(t[1:]) if t[1:].isdigit() else 0 for t in toks]

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(f"w{i}" for i in ids if i > 3)

    class Tr:
        def translate_batch(self, source, **kw):
            calls.update(kw)
            calls["source"] = source
            return [SimpleNamespace(hypotheses=[[kw["target_prefix"][0][0], "t11", "t12", "t2"]]) for _ in source]

    e = object.__new__(R.CT2Engine)
    e.spec, e.allow_proxy, e.name, e.max_input_tokens, e.revision = spec, False, spec.name, 512, ""
    e.tok, e.translator, e.beams, e.cap, e.factor, e.rep, e.nrep = Tok(), Tr(), 4, 512, 3.0, 1.0, 0
    out = e.translate_inputs(["one two three", "a b"], "yor", 1000)
    assert calls["target_prefix"] == [["yor_Latn"], ["yor_Latn"]] and calls["beam_size"] == 4 and calls["batch_type"] == "tokens" and calls["num_hypotheses"] == 1
    assert calls["max_batch_size"] == 5 * 2 and calls["max_decoding_length"] == min(512, 3 * 5 + 16) + 1 and calls["max_length"] == 512
    assert out == [("w11 w12", 3), ("w11 w12", 3)]                                         # the forced language token is dropped; the </s> counts as a generated token
    # MADLAD: no target prefix, the code is in the text
    m = R.resolve_system("madlad400-3b-mt")
    e.spec, e.name, e.max_input_tokens = m, m.name, 256
    assert e.format(["Hello."], "yor") == ["<2yo> Hello."]
    calls.clear()

    class Tr2(Tr):
        def translate_batch(self, source, **kw):
            calls.update(kw)
            return [SimpleNamespace(hypotheses=[["t11", "t2"]]) for _ in source]

    e.translator = Tr2()
    assert e.translate_inputs(["<2yo> Hello."], "yor", 100) == [("w11", 2)] and calls["target_prefix"] is None and calls["max_length"] == 256


# ------------------------------------------------------------------------------------------------ real models on the CPU (tiny, random weights)
def _tiny_marian(path):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import MarianConfig, MarianMTModel, PreTrainedTokenizerFast

    words = sorted({w.lower().strip(".,!") for d in DOCS for w in d["text"].split()} | {w for i in range(300) for w in (f"word{i}",)})
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    from tokenizers.processors import TemplateProcessing

    tk.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    fast.save_pretrained(path)
    torch.manual_seed(0)
    cfg = MarianConfig(vocab_size=len(vocab) + 1, d_model=16, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
                       decoder_ffn_dim=32, max_position_embeddings=64, pad_token_id=0, eos_token_id=1, decoder_start_token_id=0, forced_eos_token_id=1, bos_token_id=0)
    MarianMTModel(cfg).eval().save_pretrained(path)
    return path


def test_hf_engine_real_tiny_marian_cpu(tmp_path):
    mdir = _tiny_marian(str(tmp_path / "tiny"))
    spec = R.resolve_system("", hf_id="tiny-marian", prompt_style="marian", codes={"yor": "yor"}, max_input_tokens=64)
    out = tmp_path / "out"
    dp = write_docs(tmp_path, [d for d in DOCS if d["doc_id"] != "d3"])
    docs = R.load_source_docs([dp])
    runner = R.Runner(docs, str(out), [spec], engine_factory=lambda sp: R.HFEngine(sp, device="cpu", dtype="fp32", num_beams=2, model_path=mdir, max_new_tokens=12), langs=["yor"],
                      window_docs=2, max_batch_sents=3, progress_every=0.0, echo=None, install_signals=False)
    runner.run()
    rows = L.read_jsonl(os.path.join(str(out), spec.name, "yor.jsonl"))
    assert len(rows) == len(docs)
    for r, d in zip(rows, docs):
        assert r["src_sents"] == d["sents"] and len(r["tgt_sents"]) == len(d["sents"]) and r["backend"] == "hf"
        assert r["n_src_tokens"] > 0 and r["n_tgt_tokens"] >= 0
    # batch composition must not change a beam-search result of a real model: translate the sentences one at a time and compare
    eng = R.HFEngine(spec, device="cpu", dtype="fp32", num_beams=2, model_path=mdir, max_new_tokens=12)
    uniq = list(dict.fromkeys(s for d in docs for s in d["sents"]))
    single = {s: eng.translate_inputs([s], "yor", 100)[0][0] for s in uniq}
    for r in rows:
        assert r["tgt_sents"] == [L.norm_text(single[s]) for s in r["src_sents"]]


def test_hf_engine_real_opus_mt_en_ig_if_cached(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    pytest.importorskip("sentencepiece")
    try:
        transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ig", local_files_only=True)
        transformers.AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ig", local_files_only=True)
    except Exception:
        pytest.skip("Helsinki-NLP/opus-mt-en-ig is not in the local HF cache")
    flores = "/workspace/sft_build/bench/test_sets/flores_plus/devtest/eng_Latn.jsonl"
    if os.path.isfile(flores):
        rows = [json.loads(l) for l in open(flores, encoding="utf-8")][:10]
        sents = [r.get("text") or r.get("sentence") or r.get("english") for r in rows]
    else:
        sents = ["The weather is nice today.", "She goes to the market every morning.", "We will meet tomorrow at noon."] * 3
    docs = [{"doc_id": f"f{i}", "text": s} for i, s in enumerate(sents)]
    dp = write_docs(tmp_path, docs)
    spec = R.resolve_system("", hf_id="Helsinki-NLP/opus-mt-en-ig", prompt_style="marian", codes={"ibo": "ibo"})
    out = tmp_path / "out"
    runner = R.Runner(R.load_source_docs([dp]), str(out), [spec], engine_factory=lambda sp: R.HFEngine(sp, device="cpu", dtype="fp32", num_beams=4), langs=["ibo"], window_docs=0,
                      max_batch_sents=4, progress_every=0.0, echo=None, install_signals=False)
    runner.run()
    rows = L.read_jsonl(os.path.join(str(out), spec.name, "ibo.jsonl"))
    assert len(rows) == len(docs) and all(r["tgt_sents"] and r["tgt_sents"][0].strip() for r in rows)
    assert all(r["tgt_sents"][0] != r["src_sents"][0] for r in rows)


def test_ct2_engine_real_cpu_run_of_a_converted_tiny_model(tmp_path):
    pytest.importorskip("ctranslate2")
    import ctranslate2

    mdir = _tiny_marian(str(tmp_path / "tiny"))
    cdir = str(tmp_path / "tiny_ct2")
    try:
        keep = [f for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json") if os.path.isfile(os.path.join(mdir, f))]
        ctranslate2.converters.TransformersConverter(mdir, copy_files=keep).convert(cdir, quantization="int8", force=True)
    except Exception as e:                                   # the converter may not support the installed transformers version
        pytest.skip(f"conversion not possible here: {type(e).__name__}: {str(e)[:100]}")
    spec = R.resolve_system("", hf_id="tiny-marian", prompt_style="marian", codes={"yor": "yor"}, max_input_tokens=64)
    dp = write_docs(tmp_path, [d for d in DOCS if d["doc_id"] != "d3"])
    docs = R.load_source_docs([dp])
    out = tmp_path / "out"
    runner = R.Runner(docs, str(out), [spec], engine_factory=lambda sp: R.CT2Engine(sp, cdir, device="cpu", compute_type="int8", num_beams=2, max_new_tokens=12), langs=["yor"],
                      window_docs=2, max_batch_sents=3, progress_every=0.0, echo=None, install_signals=False)
    runner.run()
    rows = L.read_jsonl(os.path.join(str(out), spec.name, "yor.jsonl"))
    assert [r["doc_id"] for r in rows] == [d["doc_id"] for d in docs]
    for r, d in zip(rows, docs):
        assert r["src_sents"] == d["sents"] and len(r["tgt_sents"]) == len(d["sents"]) and r["backend"] == "ct2" and r["n_src_tokens"] > 0
    eng = R.CT2Engine(spec, cdir, device="cpu", compute_type="int8", num_beams=2, max_new_tokens=12)
    uniq = list(dict.fromkeys(s for d in docs for s in d["sents"]))
    single = {s: eng.translate_inputs([s], "yor", 100)[0][0] for s in uniq}
    for r in rows:
        assert r["tgt_sents"] == [L.norm_text(single[s]) for s in r["src_sents"]]


def test_ensure_ct2_model_reuses_a_finished_conversion(tmp_path):
    spec = R.resolve_system("madlad400-3b-mt")
    d = R.ct2_dir_for(spec, "abcdef1234567", "int8_float16", str(tmp_path))
    os.makedirs(d)
    with open(os.path.join(d, "convert_info.json"), "w") as f:
        f.write("{}")
    msgs = []
    assert R.ensure_ct2_model(spec, "abcdef1234567", "int8_float16", str(tmp_path), log=msgs.append) == d and "using the converted model" in msgs[0]


# ------------------------------------------------------------------------------------------------ NLLB-style and MADLAD-style tiny models (random weights): HF vs real CTranslate2
def _tiny_vocab_tokenizer(specials, extra_words=()):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from tokenizers.processors import TemplateProcessing

    words = sorted({w.lower().strip(".,!") for d in DOCS for w in d["text"].split()} | {f"word{i}" for i in range(300)} | set(extra_words))
    vocab = {s: i for i, s in enumerate(specials)}
    for w in words:
        vocab.setdefault(w, len(vocab))
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tk.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", vocab["</s>"])])
    return tk, vocab


def _tiny_madlad(path):
    """T5 with the special-token layout of MADLAD-400: <unk>=0, <s>=1 is the PAD token, </s>=2, decoder_start_token_id=0, gated-gelu, untied embeddings."""
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

    tk, vocab = _tiny_vocab_tokenizer(["<unk>", "<s>", "</s>"], extra_words=["<2yo>", "<2ha>"])
    PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<s>", eos_token="</s>", unk_token="<unk>").save_pretrained(path)
    torch.manual_seed(1)
    cfg = T5Config(vocab_size=len(vocab), d_model=16, d_kv=8, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, feed_forward_proj="gated-gelu", tie_word_embeddings=False,
                   decoder_start_token_id=0, pad_token_id=1, eos_token_id=2, is_encoder_decoder=True)
    T5ForConditionalGeneration(cfg).eval().save_pretrained(path)
    return path


def _tiny_nllb(path):
    """M2M100 (the NLLB architecture) with language-code tokens and the NllbTokenizerFast input layout [eng_Latn] ... </s>."""
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from transformers import M2M100Config, M2M100ForConditionalGeneration, NllbTokenizerFast

    tk, vocab = _tiny_vocab_tokenizer(["<s>", "<pad>", "</s>", "<unk>", "eng_Latn", "yor_Latn", "hau_Latn"])
    NllbTokenizerFast(tokenizer_object=tk, src_lang="eng_Latn", tgt_lang="yor_Latn", bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
                      additional_special_tokens=["eng_Latn", "yor_Latn", "hau_Latn"]).save_pretrained(path)
    torch.manual_seed(2)
    cfg = M2M100Config(vocab_size=len(vocab), d_model=16, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2, encoder_ffn_dim=32,
                       decoder_ffn_dim=32, max_position_embeddings=64, pad_token_id=1, eos_token_id=2, bos_token_id=0, decoder_start_token_id=2, activation_function="relu",
                       scale_embedding=True)
    M2M100ForConditionalGeneration(cfg).eval().save_pretrained(path)
    return path


def _ct2_convert(mdir, cdir):
    import ctranslate2

    keep = [f for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json") if os.path.isfile(os.path.join(mdir, f))]
    ctranslate2.converters.TransformersConverter(mdir, copy_files=keep).convert(cdir, quantization="float32", force=True)


def _hf_vs_ct2(tmp_path, spec, mdir, cdir, iso, expect_prefix_dropped):
    docs = R.load_source_docs([write_docs(tmp_path, [d for d in DOCS if d["doc_id"] != "d3"])])
    hf = R.HFEngine(spec, device="cpu", dtype="fp32", num_beams=1, model_path=mdir, max_new_tokens=10)
    ct = R.CT2Engine(spec, cdir, device="cpu", compute_type="float32", num_beams=1, max_new_tokens=10)
    uniq = list(dict.fromkeys(s for d in docs for s in d["sents"]))
    xin = hf.format(uniq, iso)
    assert xin == ct.format(uniq, iso) and hf.count_tokens(xin) == ct.count_tokens(xin)
    a = [t for t, n in hf.translate_inputs(xin, iso, 1000)]
    b = [t for t, n in ct.translate_inputs(xin, iso, 1000)]
    assert len(a) == len(b) == len(uniq)
    assert a == b, list(zip(a, b))[:3]                                                      # greedy float32: the same tokens from both libraries (decoder start, prefix, specials)
    # in batches of different composition the result of each sentence is the same
    b2 = []
    for i in range(0, len(xin), 3):
        b2 += [t for t, n in ct.translate_inputs(xin[i: i + 3], iso, 1000)]
    assert b2 == b
    return a


def test_ct2_vs_hf_tiny_madlad_style_t5(tmp_path):
    pytest.importorskip("ctranslate2")
    mdir = _tiny_madlad(str(tmp_path / "t5"))
    cdir = str(tmp_path / "t5_ct2")
    _ct2_convert(mdir, cdir)
    spec = R.resolve_system("", hf_id="tiny-madlad", prompt_style="madlad", codes={"yor": "yo", "hau": "ha"}, max_input_tokens=64)
    out = _hf_vs_ct2(tmp_path, spec, mdir, cdir, "yor", False)
    assert any(x.strip() for x in out)


def test_ct2_vs_hf_tiny_nllb_style_m2m100(tmp_path):
    pytest.importorskip("ctranslate2")
    mdir = _tiny_nllb(str(tmp_path / "nllb"))
    cdir = str(tmp_path / "nllb_ct2")
    _ct2_convert(mdir, cdir)
    spec = R.resolve_system("", hf_id="tiny-nllb", prompt_style="nllb", codes={"yor": "yor_Latn", "hau": "hau_Latn"}, max_input_tokens=64)
    out = _hf_vs_ct2(tmp_path, spec, mdir, cdir, "yor", True)
    assert any(x.strip() for x in out)


# ------------------------------------------------------------------------------------------------ a real SIGTERM
def test_sigterm_subprocess_exits_zero_and_resumes(tmp_path):
    import signal
    import subprocess
    import sys
    import time

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = tmp_path / "out"
    dp = write_docs(tmp_path)
    code = (
        "import sys, time\n"
        f"sys.path.insert(0, {root!r})\n"
        "from gemma_mt.eval import longdoc_runner as R\n"
        "spec = R.resolve_system('stub')\n"
        f"docs = R.load_source_docs([{dp!r}])\n"
        "r = R.Runner(docs, sys.argv[1], [spec], engine_factory=lambda sp: R.StubEngine(sp, on_batch=lambda i, iso: time.sleep(0.15)), langs=['yor', 'hau'], window_docs=2,\n"
        "             max_batch_sents=2, flush_sentences=1, progress_every=0.0)\n"
        "sys.exit(r.run())\n")
    p = subprocess.Popen([sys.executable, "-c", code, str(out)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    log = os.path.join(str(out), "progress.log")
    while time.time() - t0 < 30 and not (os.path.isfile(log) and "tok/s" in open(log).read()):
        time.sleep(0.05)
    time.sleep(0.3)
    p.send_signal(signal.SIGTERM)                                                           # the child is OUR process: nothing else is signalled
    outtxt = p.communicate(timeout=30)[0]
    assert p.returncode == 0, outtxt
    assert json.load(open(os.path.join(str(out), "progress.json")))["state"] == "stopped" and "stop  signal" in open(log).read()
    r2, _, _ = make_runner(tmp_path, out, spec=R.resolve_system("stub"), langs=("yor", "hau"))
    r2.run()
    full = tmp_path / "full"
    make_runner(tmp_path, full, spec=R.resolve_system("stub"), langs=("yor", "hau"))[0].run()
    for iso in ("yor", "hau"):
        assert strip(rows_of(out, "stub", iso)) == strip(rows_of(full, "stub", iso))
