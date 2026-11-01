# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-41: the best-translation archive (rl/archive.py): key, eligibility, UPSERT rule, background writer, crash-restart, queue overflow, CLI, trainer hook."""
import contextlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import unicodedata

import pytest

from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl import archive as A
from gemma_mt.rl.archive import TranslationArchive, archive_step, eligibility, make_key
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]
STOP = [EOT, SPECIALS["<eos>"]]
THR = dict(A.DEFAULT_MIN_REWARD_BY_MODE)


def row(iso="yor", source="The market opens.", mode="qe", tr="Oja n si.", reward=1.5, status="ok", qe=0.7, chrf=None, dia=None, lid="yor_Latn",
        n_chunks=1, failed=0, lbin="b64"):
    return (iso, source, mode, tr, reward, status, qe, chrf, dia, lid, n_chunks, failed, lbin)


def db_rows(path):
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    try:
        return {r["key"]: dict(r) for r in c.execute("SELECT * FROM best")}
    finally:
        c.close()


def with_arch(fn):
    """Injects `arch` (a factory of archives under tmp_path, all closed afterwards). Written as a decorator, not a pytest fixture, so that tests/run_tests.py (no pytest) runs it too."""
    import functools
    import inspect

    wants_tmp = "tmp_path" in inspect.signature(fn).parameters

    def wrapper(tmp_path):
        made = []

        def make(**kw):
            a = TranslationArchive(str(tmp_path / "a" / "archive.sqlite"), register_atexit=False, **kw)
            made.append(a)
            return a

        try:
            return fn(make, tmp_path) if wants_tmp else fn(make)
        finally:
            for a in made:
                a.close()

    functools.update_wrapper(wrapper, fn, assigned=("__name__", "__doc__", "__module__"), updated=())
    del wrapper.__wrapped__                                  # inspect.signature must see (tmp_path)
    return wrapper


# ------------------------------------------------------------------------------------------------ key
def test_key_is_16_hex_of_sha1_iso_newline_nfc_source_and_stable_across_nfc_nfd():
    import hashlib

    nfc = unicodedata.normalize("NFC", "Ọ̀kan àti méjì")
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd
    assert make_key("yor", nfc) == make_key("yor", nfd)
    assert make_key("yor", nfc) == hashlib.sha1(("yor\n" + nfc).encode("utf-8")).hexdigest()[:16]
    assert len(make_key("yor", "x")) == 16 and all(c in "0123456789abcdef" for c in make_key("yor", "x"))
    assert make_key("yor", "x") != make_key("ibo", "x") and make_key("yor", "x") != make_key("yor", "x ")


# ------------------------------------------------------------------------------------------------ eligibility
@pytest.mark.parametrize("kw,why", [
    ({}, None),
    ({"status": "wrong_language", "reward": -2.0}, "status:wrong_language"),
    ({"status": "no_end_of_turn", "reward": -2.0}, "status:no_end_of_turn"),
    ({"tr": "  "}, "empty_text"),
    ({"reward": float("nan")}, "nonfinite_reward"),
    ({"failed": 1}, "failed_chunk"),
    ({"reward": 0.99}, "below_threshold"),
    ({"reward": 1.0}, None),                                   # the threshold itself qualifies (+1.0 = t_mid)
    ({"dia": 0.0}, None),
    ({"dia": 0.2}, "diacritics"),
    ({"mode": "consensus", "reward": 0.6, "chrf": 60.0}, None),
    ({"mode": "consensus", "reward": 0.4, "chrf": 55.0}, "below_threshold"),
    ({"mode": "consensus", "reward": 0.0, "chrf": None}, "no_consensus"),
    ({"mode": "blend", "reward": 1.2, "chrf": None}, "no_consensus"),
    ({"mode": "blend", "reward": 1.2, "chrf": 30.0}, "low_consensus"),
    ({"mode": "blend", "reward": 1.2, "chrf": 62.0}, None),
    ({"mode": "blend", "reward": 0.9, "chrf": 62.0}, "below_threshold"),
])
def test_eligibility(kw, why):
    assert eligibility(row(**kw), THR, A.DEFAULT_MIN_CONSENSUS_CHRF) == why


@with_arch
def test_threshold_override_and_unknown_mode(arch):
    assert eligibility(row(reward=0.8), {**THR, "qe": 0.5}, 50.0) is None
    with pytest.raises(ValueError):
        arch(min_reward_by_mode={"nonsense": 1.0})
    a = arch(min_reward_by_mode={"qe": 0.25})                  # partial dicts are merged over the defaults
    assert a.min_reward_by_mode == {"qe": 0.25, "blend": 1.0, "consensus": 0.5}


# ------------------------------------------------------------------------------------------------ UPSERT rule
@with_arch
def test_insert_improve_ineligible_dedupe(arch):
    a = arch()
    p = a.path
    # step 1: 4 samples of one source (one GRPO group): two ineligible, two eligible -> ONE row, the better one; a second source never eligible -> no row
    a.submit(1, "b64", [row(tr="bad", reward=-2.0, status="degenerate"), row(tr="ok1", reward=1.2), row(tr="ok2", reward=1.6), row(tr="low", reward=0.4),
                        row(source="never good", tr="meh", reward=0.5)])
    assert a.flush()
    rows = db_rows(p)
    assert len(rows) == 1
    r = rows[make_key("yor", "The market opens.")]
    assert (r["translation"], r["reward"], r["step_first"], r["step_best"], r["times_seen"], r["n_improved"]) == ("ok2", 1.6, 1, 1, 4, 0)
    assert (r["iso"], r["mode"], r["stage"], r["length_bin"], r["n_chunks"], r["lid_ok"], r["lid_label"]) == ("yor", "qe", "b64", "b64", 1, 1, "yor_Latn")
    assert (a.inserted, a.improved, a.size) == (1, 0, 1)
    # step 2: not better (equal / lower / better by less than epsilon) -> unchanged, times_seen grows
    a.submit(2, "b64", [row(tr="ok3", reward=1.6), row(tr="ok4", reward=1.5), row(tr="ok5", reward=1.6 + 1e-5), row(tr="inelig", reward=9.0, status="copy_source")])
    a.flush()
    r = db_rows(p)[make_key("yor", "The market opens.")]
    assert (r["translation"], r["reward"], r["step_best"], r["times_seen"], r["n_improved"]) == ("ok2", 1.6, 1, 8, 0)
    # step 3: a higher reward but a failed chunk / diacritic deficit / wrong status never replaces it
    a.submit(3, "b64", [row(tr="x1", reward=2.0, failed=1), row(tr="x2", reward=2.0, dia=0.3), row(tr="x3", reward=2.0, status="wrong_language")])
    a.flush()
    r = db_rows(p)[make_key("yor", "The market opens.")]
    assert (r["translation"], r["times_seen"], r["n_improved"]) == ("ok2", 11, 0)
    # step 4: strictly better and eligible -> replaced, the best-step columns follow, step_first stays
    a.submit(4, "b256", [row(tr="best", reward=2.1, qe=0.8, lbin="b256")])
    a.flush()
    r = db_rows(p)[make_key("yor", "The market opens.")]
    assert (r["translation"], r["reward"], r["qe"], r["step_first"], r["step_best"], r["stage"], r["length_bin"], r["times_seen"], r["n_improved"]) == \
        ("best", 2.1, 0.8, 1, 4, "b256", "b256", 12, 1)
    # the SAME text scored higher again (fp16 batch noise) is not an improvement
    a.submit(5, "b256", [row(tr="best", reward=2.1 + 0.01)])
    a.flush()
    r = db_rows(p)[make_key("yor", "The market opens.")]
    assert (r["reward"], r["n_improved"], r["step_best"], r["times_seen"]) == (2.1, 1, 4, 13)
    assert (a.inserted, a.improved, a.size) == (1, 1, 1)
    c = a.counters()
    assert c["archive/size"] == 1 and c["archive/inserted"] == 1 and c["archive/improved"] == 1 and c["archive/dropped_queue"] == 0
    assert abs(c["archive/mean_best_reward"] - 2.1) < 1e-9


@with_arch
def test_nfd_and_nfc_sources_are_the_same_row_and_the_stored_source_is_nfc(arch):
    a = arch()
    nfc = unicodedata.normalize("NFC", "Ọ̀kan àti méjì")
    nfd = unicodedata.normalize("NFD", nfc)
    a.submit(1, "", [row(source=nfd, tr="a", reward=1.1)])
    a.submit(2, "", [row(source=nfc, tr="b", reward=1.4)])
    a.flush()
    rows = db_rows(a.path)
    assert len(rows) == 1
    (r,) = rows.values()
    assert r["source"] == nfc and r["translation"] == "b" and r["n_improved"] == 1


@with_arch
def test_iso_separates_identical_sources(arch):
    a = arch()
    a.submit(1, "", [row(iso="yor", tr="y", reward=1.2), row(iso="ibo", tr="i", reward=1.3)])
    a.flush()
    assert len(db_rows(a.path)) == 2


@with_arch
def test_counters_table_has_rejection_reasons_and_per_language_eligibility(arch):
    a = arch()
    a.submit(1, "", [row(), row(status="degenerate", reward=-2.0), row(iso="ibo", source="s2", reward=0.1)])
    a.flush()
    st = A.compute_stats(a.path)
    assert st["samples_seen"] == 3 and st["samples_eligible"] == 1
    assert st["reject_reasons"] == {"below_threshold": 1, "status:degenerate": 1}
    assert st["eligible_rate_by_language"] == {"ibo": 0.0, "yor": 0.5}


# ------------------------------------------------------------------------------------------------ the writer thread
def test_thread_flushes_on_close_without_flush_call_and_close_is_idempotent(tmp_path):
    p = str(tmp_path / "x.sqlite")
    a = TranslationArchive(p, flush_seconds=3600.0, register_atexit=False)
    for s in range(5):
        a.submit(s + 1, "", [row(source=f"src {s}", reward=1.2)])
    stats = a.close()
    assert stats["archive/size"] == 5 and len(db_rows(p)) == 5
    assert not a._thread.is_alive()
    a.close()
    assert a.submit(9, "", [row()]) is False                  # a closed archive refuses quietly


def test_periodic_commit_makes_rows_visible_without_flush(tmp_path):
    p = str(tmp_path / "x.sqlite")
    a = TranslationArchive(p, flush_seconds=0.2, register_atexit=False)
    try:
        a.submit(1, "", [row(reward=1.3)])
        deadline = time.time() + 10
        while time.time() < deadline and not db_rows(p):
            time.sleep(0.05)
        assert len(db_rows(p)) == 1, "the writer must commit every flush_seconds even when nothing else happens"
    finally:
        a.close()


def test_submit_does_not_block_and_queue_overflow_drops_and_counts(tmp_path):
    p = str(tmp_path / "x.sqlite")
    a = TranslationArchive(p, queue_size=2, register_atexit=False)
    entered, release = threading.Event(), threading.Event()
    real = a._apply

    def slow(conn, step, stage, rows):
        entered.set()
        release.wait(30)
        return real(conn, step, stage, rows)

    a._apply = slow
    try:
        batch = [row(source=f"s{i}", reward=1.2) for i in range(10)]
        assert a.submit(1, "", batch) is True                 # taken by the writer, which now blocks inside _apply
        assert entered.wait(10)
        assert a.submit(2, "", batch) and a.submit(3, "", batch)       # the two queue slots
        t0 = time.perf_counter()
        results = [a.submit(4 + k, "", batch) for k in range(5)]       # queue full: dropped, never blocked
        assert time.perf_counter() - t0 < 0.5
        assert results == [False] * 5
        assert a.dropped_queue == 50 and a.dropped_batches == 5
        release.set()
        assert a.flush()
        assert a.counters()["archive/dropped_queue"] == 50
        assert len(db_rows(p)) == 10                           # the three accepted batches wrote the same 10 sources
        assert db_rows(p)[make_key("yor", "s0")]["times_seen"] == 3
    finally:
        release.set()
        a.close()


def test_a_failing_batch_is_rolled_back_counted_and_does_not_stop_the_writer(tmp_path):
    p = str(tmp_path / "x.sqlite")
    a = TranslationArchive(p, register_atexit=False)
    err = io.StringIO()
    real = a._apply
    calls = {"n": 0}

    def flaky(conn, step, stage, rows):
        calls["n"] += 1
        if calls["n"] == 1:
            real(conn, step, stage, rows)                      # writes, then fails: the savepoint must undo it
            raise OSError("disk full (test)")
        return real(conn, step, stage, rows)

    a._apply = flaky
    with contextlib.redirect_stderr(err):
        a.submit(1, "", [row(source="first", reward=1.2)])
        a.submit(2, "", [row(source="second", reward=1.2)])
        assert a.flush()
        rows = db_rows(p)
        assert [r["source"] for r in rows.values()] == ["second"]
        assert a.errors == 1 and "disk full" in a.last_error and a.size == 1
        a.close()
    assert "translation archive write failed" in err.getvalue()


# ------------------------------------------------------------------------------------------------ crash / restart
def test_crash_restart_reopens_the_database_and_continues(tmp_path):
    """A child process writes, flushes and is then killed without close() (os._exit); the parent reopens the file, finds the rows, improves one and adds one."""
    p = str(tmp_path / "crash" / "archive.sqlite")
    child = (
        "import os, sys\n"
        f"sys.path.insert(0, {ROOT!r})\n"
        "from gemma_mt.rl.archive import TranslationArchive\n"
        f"a = TranslationArchive({p!r}, flush_seconds=3600.0)\n"
        "r = lambda s, tr, rw: ('yor', s, 'qe', tr, rw, 'ok', 0.7, None, None, 'yor_Latn', 1, 0, 'b64')\n"
        "a.submit(1, 'b64', [r('one', 't1', 1.2), r('two', 't2', 1.3)])\n"
        "assert a.flush()\n"
        "a.submit(2, 'b64', [r('three', 'LOST', 1.9)])\n"           # queued after the last commit: may be lost on a hard kill, but must never corrupt the file
        "import time; time.sleep(0.3)\n"
        "os._exit(3)\n")
    rc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True, timeout=120)
    assert rc.returncode == 3, rc.stderr
    rows = db_rows(p)                                          # the committed rows survive; the file opens (WAL recovery)
    assert {r["source"] for r in rows.values()} >= {"one", "two"}
    assert sqlite3.connect(p).execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    a = TranslationArchive(p, register_atexit=False)
    try:
        assert a.size >= 2 and a.size_at_open == a.size
        a.submit(10, "mixed", [row(source="one", tr="t1-better", reward=1.8, lbin="b256"), row(source="four", tr="t4", reward=1.1)])
        assert a.flush()
        rows = db_rows(p)
        r1 = rows[make_key("yor", "one")]
        assert (r1["translation"], r1["step_first"], r1["step_best"], r1["stage"], r1["n_improved"]) == ("t1-better", 1, 10, "mixed", 1)
        assert make_key("yor", "four") in rows
    finally:
        a.close()


# ------------------------------------------------------------------------------------------------ the trainer's hook
def _ids(text):
    return TOK.encode(text, add_special_tokens=False) + [EOT]


class _Const:
    def __init__(self, v):
        self.v = v

    def __call__(self, s, m, r=None):
        return [self.v] * len(s)


SRC = "The market opens at eight o'clock and the farmers bring fresh vegetables."
GOOD = "yor Oja n si ni agogo mejo, awon agbe si mu ewebe tutu wa."


def test_hook_reads_the_reward_output_and_archives_the_text_the_scorer_saw(tmp_path):
    rw = TranslationReward(RewardConfig(consensus=True, content_modes={"kam": "blend", "ach": "consensus"}), _Const(0.80), TOK, STOP)
    G = 8
    samples = []
    for k, (iso, tag) in enumerate([("yor", "yor"), ("kam", "kam"), ("ach", "ach")]):
        for g in range(G):
            body = f"{tag} Oja n si ni agogo mejo, awon agbe si mu ewebe tutu wa." if g < 6 else f"{tag}"
            samples.append(Sample(_ids(body), iso, SRC + f" {k}"))
    out = rw(samples, group_size=G)
    assert len(out.n_chunks) == len(out.failed_chunks) == len(samples)
    assert out.n_chunks[0] >= 1 and out.failed_chunks[0] == 0
    a = TranslationArchive(str(tmp_path / "h" / "archive.sqlite"), register_atexit=False)
    try:
        assert archive_step(a, rw, samples, out, 7, "b64", ["b64", "b256", "b512"], G)
        assert a.flush()
        rows = db_rows(a.path)
    finally:
        a.close()
    by = {r["iso"]: r for r in rows.values()}
    assert set(by) == {"yor", "kam", "ach"}                    # 3 sources, one row each (8 samples per source deduped)
    assert by["yor"]["mode"] == "qe" and by["kam"]["mode"] == "blend" and by["ach"]["mode"] == "consensus"
    assert by["yor"]["translation"] == "Oja n si ni agogo mejo, awon agbe si mu ewebe tutu wa."          # the tag is stripped, exactly out.texts
    assert by["yor"]["translation"] in out.texts
    assert by["kam"]["consensus_chrf"] is not None and by["kam"]["consensus_chrf"] >= 50
    assert by["yor"]["length_bin"] == "b64" and by["kam"]["length_bin"] == "b256" and by["ach"]["length_bin"] == "b512"
    assert all(r["step_first"] == 7 and r["stage"] == "b64" and r["times_seen"] == G for r in rows.values())
    # the two garbage samples per group (tag only) never got in; the stored reward is one the reward function really returned
    for r in rows.values():
        assert any(abs(r["reward"] - x) < 1e-9 for x in out.rewards)


def test_hook_requires_consensus_for_blend_languages_even_when_the_reward_fell_back_to_qe(tmp_path):
    rw = TranslationReward(RewardConfig(consensus=True, content_modes={"kam": "blend"}), _Const(0.80), TOK, STOP)
    samples = [Sample(_ids(GOOD.replace("yor", "kam")), "kam", SRC)] + [Sample(_ids("kam"), "kam", SRC)] * 7     # only 1 valid sample: no consensus (min 3)
    out = rw(samples, group_size=8)
    assert out.content_mode[0] == "qe" and out.consensus_chrf[0] is None and out.rewards[0] > 1.0     # scored by QE alone ...
    a = TranslationArchive(str(tmp_path / "b.sqlite"), register_atexit=False)
    try:
        archive_step(a, rw, samples, out, 1, "", None, 8)
        a.flush()
        assert db_rows(a.path) == {}                           # ... but the language is a blend language: no row
        assert A.compute_stats(a.path)["reject_reasons"].get("no_consensus") == 1
    finally:
        a.close()


def test_hook_overhead_for_1000_samples_is_far_below_50_ms(tmp_path):
    rw = TranslationReward(RewardConfig(consensus=True, content_modes={"kam": "blend"}), _Const(0.80), TOK, STOP)
    langs = ["yor", "kam", "ibo", "hau", "swh"]
    samples = [Sample(_ids(f"{langs[(i // 8) % 5]} Oja n si ni agogo mejo {i % 8}"), langs[(i // 8) % 5], SRC + f" {i // 8}") for i in range(1000)]
    out = rw(samples, group_size=8)
    a = TranslationArchive(str(tmp_path / "o.sqlite"), register_atexit=False, queue_size=1000)
    try:
        times = []
        for step in range(1, 8):
            t0 = time.perf_counter()
            archive_step(a, rw, samples, out, step, "b64", ["b64"] * 125, 8)
            times.append(time.perf_counter() - t0)
        a.flush()
        assert sorted(times)[len(times) // 2] < 0.05, f"hook median {1000 * sorted(times)[len(times) // 2]:.1f} ms"
        assert a.dropped_queue == 0 and a.errors == 0
    finally:
        a.close()


# ------------------------------------------------------------------------------------------------ CLI
def _filled(arch, per_lang=5, langs=("yor", "ibo", "hau")):
    a = arch()
    rows = []
    for l in langs:
        for i in range(per_lang):
            rows.append(row(iso=l, source=f"{l} source {i} | with pipe\nand newline", tr=f"{l} translation {i}", reward=1.0 + 0.1 * i))
    a.submit(1, "b64", rows)
    a.submit(2, "b256", [row(iso="yor", source="yor source 0 | with pipe\nand newline", tr="better", reward=2.2, lbin="b256")])
    a.flush()
    return a


@with_arch
def test_cli_stats_export_review(arch, tmp_path):
    a = _filled(arch)
    st = A.compute_stats(a.path)
    assert st["total"]["rows"] == 15 and st["total"]["languages"] == 3 and st["total"]["improved_rows"] == 1 and st["total"]["improvements"] == 1
    assert {r["iso"]: r["rows"] for r in st["by_language"]} == {"hau": 5, "ibo": 5, "yor": 5}
    assert {r["stage"]: r["rows"] for r in st["by_stage"]} == {"b64": 14, "b256": 1}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert A.main(["stats", a.path]) == 0
    text = buf.getvalue()
    assert "15 rows" in text and "rows per language" in text and "yor" in text and "rows per stage" in text
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert A.main(["stats", a.path, "--json"]) == 0
    assert json.loads(buf.getvalue())["total"]["rows"] == 15
    out_dir = str(tmp_path / "exp")
    with contextlib.redirect_stdout(io.StringIO()):
        assert A.main(["export", a.path, "--out", out_dir, "--format", "jsonl"]) == 0
    lines = [json.loads(l) for l in open(os.path.join(out_dir, "archive.jsonl"), encoding="utf-8")]
    assert len(lines) == 15 and set(lines[0]) == set(A.COLUMNS)
    # review: default 3 per language, deterministic, pipes and newlines escaped, explicit cap respected
    md = A.review_markdown(a.path, seed=1)
    body = [l for l in md.splitlines() if l.startswith("| ") and not l.startswith("| lang") and not l.startswith("|---")]
    assert len(body) == 9 and all(l.count(" | ") >= 5 for l in body)
    assert {l.split(" | ")[0] for l in body} == {"| yor", "| ibo", "| hau"}
    assert md == A.review_markdown(a.path, seed=1) and md != A.review_markdown(a.path, seed=2)
    assert "\\|" in md and "\n" not in "".join(l for l in body if "with pipe" in l)
    assert len([l for l in A.review_markdown(a.path, per_language=1).splitlines() if l.startswith("| yor")]) == 1
    only = A.review_markdown(a.path, languages=["ibo"], min_reward=1.25)
    assert "| yor" not in only and "| ibo" in only
    with contextlib.redirect_stdout(io.StringIO()):
        assert A.main(["review", a.path, "--out", str(tmp_path / "r.md")]) == 0
    assert open(str(tmp_path / "r.md"), encoding="utf-8").read().startswith("# Archive review")


@with_arch
def test_cli_export_parquet_has_all_columns(arch, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    a = _filled(arch)
    paths = A.export_archive(a.path, str(tmp_path / "exp"), "both")
    assert [os.path.basename(p) for p in paths] == ["archive.jsonl", "archive.parquet"]
    t = pq.read_table(paths[1])
    assert t.num_rows == 15 and list(t.column_names) == list(A.COLUMNS)


@with_arch
def test_cli_runs_as_a_module_without_the_training_stack(arch):
    a = _filled(arch)
    r = subprocess.run([sys.executable, "-m", "gemma_mt.rl.archive", "stats", a.path], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0 and "15 rows" in r.stdout, r.stderr


# ------------------------------------------------------------------------------------------------ config + trainer smoke
def test_config_defaults_validation_and_override_merge():
    from gemma_mt.rl.train_rl import RLConfig, apply_overrides, validate_config

    c = RLConfig()
    assert c.archive is False and c.archive_path is None and c.archive_min_reward_by_mode == {"qe": 1.0, "blend": 1.0, "consensus": 0.5}
    validate_config(RLConfig(archive=True))
    with pytest.raises(ValueError):
        validate_config(RLConfig(archive=True, archive_min_reward_by_mode={"nope": 1.0}))
    with pytest.raises(ValueError):
        validate_config(RLConfig(archive=True, archive_flush_seconds=0))
    d = apply_overrides({"archive_min_reward_by_mode": {"qe": 1.0, "blend": 1.0, "consensus": 0.5}, "archive": False}, {"archive_min_reward_by_mode": {"consensus": 0.7}, "archive": True})
    assert d == {"archive_min_reward_by_mode": {"qe": 1.0, "blend": 1.0, "consensus": 0.7}, "archive": True}


def test_yaml_configs_archive_off_in_rl_and_on_in_phase2():
    pytest.importorskip("yaml")
    from gemma_mt.rl.train_rl import RLConfig
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    assert dataclass_from_dict(RLConfig, load_yaml(os.path.join(ROOT, "configs", "rl.yaml"))).archive is False
    p2 = os.path.join(ROOT, "configs", "rl_rtxpro_phase2.yaml")
    if os.path.exists(p2):
        c = dataclass_from_dict(RLConfig, load_yaml(p2))
        assert c.archive is True and c.archive_path == "/workspace/sft_build/runs/rl2/archive/archive.sqlite"
        assert c.archive_min_reward_by_mode == {"qe": 1.0, "blend": 1.0, "consensus": 0.5}


def test_train_rl_smoke_with_the_archive_on(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from gemma_mt.rl.train_rl import run_smoke

    path = str(tmp_path / "ar" / "archive.sqlite")
    s = run_smoke(str(tmp_path / "w"), sft_steps=60, overrides={"archive": True, "archive_path": path, "archive_min_reward_by_mode": {"qe": -5.0, "blend": -5.0, "consensus": -5.0},
                                                               "archive_flush_seconds": 0.2})
    assert s["steps"] == 4 and s["archive"]["archive/dropped_queue"] == 0 and s["archive"]["archive/errors"] == 0
    rows = [json.loads(l) for l in open(s["workdir"] + "/rl/metrics.jsonl")]
    train = [r for r in rows if "reward" in r]
    assert len(train) == 4 and all("archive/size" in r and "archive/inserted" in r and "archive/improved" in r and "archive/dropped_queue" in r for r in train)
    st = A.compute_stats(path)
    assert st["total"]["rows"] > 0 and st["total"]["rows"] == s["archive"]["archive/size"]
    assert st["samples_seen"] == 4 * 3 * 4                      # 4 steps x 3 prompts x group 4
    got = db_rows(path)
    assert all(r["step_first"] >= 1 and r["step_best"] <= 4 and r["times_seen"] >= 1 for r in got.values())
    # reopening the same file continues: a second run on the same path keeps the first run's rows
    n0 = len(got)
    s2 = run_smoke(str(tmp_path / "w2"), sft_steps=60, overrides={"archive": True, "archive_path": path, "archive_min_reward_by_mode": {"qe": -5.0, "blend": -5.0, "consensus": -5.0}})
    assert s2["archive"]["archive/size"] >= n0


def test_train_rl_smoke_default_has_no_archive(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from gemma_mt.rl.train_rl import run_smoke

    s = run_smoke(str(tmp_path / "w"), sft_steps=60)
    assert "archive" not in s and not os.path.exists(os.path.join(s["workdir"], "rl", "archive"))
