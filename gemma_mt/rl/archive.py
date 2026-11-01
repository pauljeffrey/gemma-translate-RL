# SPDX-License-Identifier: AGPL-3.0-or-later
"""Best-translation archive for the Gemma RL (D-41, owner 2026-10-19).

    "when rewards start getting good, close to +1, it might make sense to start saving the translations for later review."

Every sampled completion of a training step is looked at once, right after its reward was computed. The archive keeps, per SOURCE (key =
first 16 hex digits of sha1(iso + "\\n" + NFC(source))), the best translation that was ever found for it, and replaces it whenever a better
one turns up later in training. Nothing here changes the reward, the advantages or the update; the archive is read-only with respect to training.

    reward of the step ──► hook (train_rl.py) ──► TranslationArchive.submit(step, stage, rows)     main thread: one tuple per sample + one queue.put_nowait
                                                          │ bounded queue (full: drop the step's rows and count them)
                                                          ▼
                                       ONE writer thread ──► SQLite (WAL, one file) ── committed every `flush_seconds`, on flush() and on close()

A sample enters / replaces the stored entry only when it is ELIGIBLE (all of):
    * status ok (none of the hard failure classes: no_end_of_turn, post_eos_rubbish, dangling_marker, bad_prefix, empty, copy_source, degenerate,
      length_ratio, wrong_language, special_tokens) and a non-empty translation;
    * no failed chunk (an omitted / extra / copied sentence of the chunked reward);
    * GlotLID passed wherever the guard is on (status ok already means the language check passed; `lid_ok` = 1 when the text was long enough to be checked, NULL otherwise);
    * diacritic_deficit == 0 (None = the language does not use diacritics in the calibration);
    * reward >= the threshold of the language's content mode (`min_reward_by_mode`: qe 1.0, blend 1.0, consensus 0.5; +1.0 is the language's median-human QE anchor t_mid, the
      shaped chunk reward after the length / lexical / diacritic penalties; consensus 0.5 = group chrF++ >= 58.8);
    * blend and consensus languages: the group consensus exists and (blend) is >= `min_consensus_chrf`, so a blend reward that rests on an
      unreliable QE alone (the reason these languages are blended) does not qualify.
It REPLACES the stored translation only if its reward is higher by more than `epsilon` and its text differs (the same text re-scored with a
slightly different fp16 batch is not an improvement). `times_seen` counts every sampled completion of the source, eligible or not, while the row exists.

Caveat for the owner: a `qe` reward is a function of the text alone, so "higher reward" is comparable across steps. A `consensus` / `blend` reward
also depends on the sibling samples of the group it was drawn in (mean chrF++ against them) and so is noisier as a ranking; the stored value is
the best reward SEEN, not an absolute quality.

stdlib only (the CLI must run anywhere):
    python -m gemma_mt.rl.archive stats  PATH [--json]
    python -m gemma_mt.rl.archive export PATH [--out DIR] [--format both|parquet|jsonl]
    python -m gemma_mt.rl.archive review PATH [--per-language 3] [--seed 0] [--languages yor,hau] [--min-reward R] [--out FILE.md]
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import queue
import random
import sqlite3
import sys
import threading
import time
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
MODES = ("qe", "blend", "consensus")
DEFAULT_MIN_REWARD_BY_MODE: Dict[str, float] = {"qe": 1.0, "blend": 1.0, "consensus": 0.5}
DEFAULT_MIN_CONSENSUS_CHRF = 50.0          # blend languages: mean chrF++ against the valid siblings of the group (light noise between samples of a good group = 66-71, heavy noise 46-51)
DEFAULT_EPSILON = 1e-3                     # a replacement needs a reward higher by more than this
COLUMNS = ("key", "iso", "mode", "source", "translation", "reward", "qe", "consensus_chrf", "diacritic_deficit", "lid_ok", "lid_label", "n_chunks",
           "step_first", "step_best", "length_bin", "stage", "times_seen", "n_improved", "updated")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS best (
  key TEXT PRIMARY KEY, iso TEXT NOT NULL, mode TEXT NOT NULL, source TEXT NOT NULL, translation TEXT NOT NULL,
  reward REAL NOT NULL, qe REAL, consensus_chrf REAL, diacritic_deficit REAL, lid_ok INTEGER, lid_label TEXT,
  n_chunks INTEGER NOT NULL, step_first INTEGER NOT NULL, step_best INTEGER NOT NULL, length_bin TEXT, stage TEXT,
  times_seen INTEGER NOT NULL, n_improved INTEGER NOT NULL DEFAULT 0, updated REAL NOT NULL);
CREATE INDEX IF NOT EXISTS best_iso ON best(iso);
CREATE TABLE IF NOT EXISTS counters (name TEXT PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT);
"""

# one observation = one sampled completion, in this order (the hook builds them; tests build them by hand)
Row = Tuple[str, str, str, str, float, str, Optional[float], Optional[float], Optional[float], Optional[str], int, int, str]
ROW_FIELDS = ("iso", "source", "mode", "translation", "reward", "status", "qe", "consensus_chrf", "diacritic_deficit", "lid_label", "n_chunks", "failed_chunks", "length_bin")


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def make_key(iso: str, source: str) -> str:
    """Stable index of a source: first 16 hex digits of sha1(iso + '\\n' + NFC(source)). NFC and NFD spellings of the same source give the same key."""
    return hashlib.sha1((iso + "\n" + nfc(source)).encode("utf-8")).hexdigest()[:16]


def eligibility(row: Sequence[Any], min_reward_by_mode: Dict[str, float], min_consensus_chrf: float) -> Optional[str]:
    """None when the observation may enter / replace the archive entry, otherwise the first reason it may not (`status:<class>`, `empty_text`, `nonfinite_reward`, `failed_chunk`,
    `no_consensus`, `low_consensus`, `diacritics`, `below_threshold`)."""
    iso, source, mode, translation, reward, status, qe, chrf, dia, lid_label, n_chunks, failed, length_bin = row
    if status != "ok":
        return "status:" + str(status)
    if not translation or not translation.strip():
        return "empty_text"
    if reward is None or not math.isfinite(reward):
        return "nonfinite_reward"
    if failed:
        return "failed_chunk"
    if mode != "qe":
        if chrf is None:
            return "no_consensus"
        if mode == "blend" and chrf < min_consensus_chrf:
            return "low_consensus"
    if dia is not None and dia > 1e-9:
        return "diacritics"
    if reward < min_reward_by_mode.get(mode, 1.0) - 1e-9:
        return "below_threshold"
    return None


class TranslationArchive:
    """SQLite (WAL) archive of the best translation per source, written by one background thread. `submit` never blocks the caller."""

    def __init__(self, path: str, min_reward_by_mode: Optional[Dict[str, float]] = None, min_consensus_chrf: float = DEFAULT_MIN_CONSENSUS_CHRF,
                 flush_seconds: float = 20.0, queue_size: int = 64, epsilon: float = DEFAULT_EPSILON, register_atexit: bool = True):
        if flush_seconds <= 0 or queue_size < 1 or epsilon < 0:
            raise ValueError("archive: flush_seconds > 0, queue_size >= 1 and epsilon >= 0 are required")
        thr = dict(DEFAULT_MIN_REWARD_BY_MODE)
        for k, v in (min_reward_by_mode or {}).items():
            if k not in MODES:
                raise ValueError(f"archive_min_reward_by_mode: unknown mode {k!r}, known: {MODES}")
            thr[k] = float(v)
        self.path = path
        self.min_reward_by_mode, self.min_consensus_chrf = thr, float(min_consensus_chrf)
        self.flush_seconds, self.epsilon = float(flush_seconds), float(epsilon)
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        # opened HERE (a bad path / read-only disk stops the run before step 1), then used by the writer thread only
        self._conn = sqlite3.connect(path, timeout=30.0, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._conn.execute("INSERT OR REPLACE INTO meta(name, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        self._conn.execute("INSERT OR REPLACE INTO meta(name, value) VALUES ('config', ?)",
                           (json.dumps({"min_reward_by_mode": thr, "min_consensus_chrf": self.min_consensus_chrf, "epsilon": self.epsilon, "opened": time.time()}),))
        n, s = self._conn.execute("SELECT COUNT(*), COALESCE(SUM(reward), 0.0) FROM best").fetchone()
        self.size, self._sum_best = int(n), float(s)
        self.size_at_open = self.size
        # counters of THIS process (read by the trainer for its log line; the writer thread updates them after each committed batch)
        self.inserted = self.improved = self.dropped_queue = self.dropped_batches = self.errors = 0
        self.samples = self.eligible = self.batches = 0
        self.submit_seconds = 0.0
        self.last_error: Optional[str] = None
        self._q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._closed = False
        self._close_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="translation-archive-writer", daemon=True)
        self._thread.start()
        if register_atexit:
            atexit.register(self.close)             # a crashing trainer still flushes what it has (SIGKILL loses at most `flush_seconds` of rows)

    # ------------------------------------------------------------------------------------------------ producer side
    def submit(self, step: int, stage: str, rows: List[Row]) -> bool:
        """Queue one step's observations. Never blocks: a full queue drops the whole step (counted in `dropped_queue` as samples). Returns True when queued."""
        if self._closed or not rows:
            return False
        t0 = time.perf_counter()
        try:
            self._q.put_nowait(("batch", int(step), stage or "", rows))
            ok = True
        except queue.Full:
            self.dropped_queue += len(rows)
            self.dropped_batches += 1
            ok = False
        self.submit_seconds += time.perf_counter() - t0
        return ok

    def flush(self, timeout: float = 60.0) -> bool:
        """Block until everything submitted so far is written AND committed. False on timeout / closed archive."""
        if self._closed or not self._thread.is_alive():
            return False
        ev = threading.Event()
        try:
            self._q.put(("flush", ev), timeout=timeout)
        except queue.Full:
            return False
        return ev.wait(timeout)

    def close(self, timeout: float = 120.0) -> Dict[str, Any]:
        """Drain the queue, commit, close the database and stop the thread. Idempotent. Returns the final counters."""
        with self._close_lock:
            if not self._closed:
                self._closed = True
                try:
                    self._q.put(("stop",), timeout=timeout)
                except queue.Full:
                    pass
                self._thread.join(timeout)
        return self.counters()

    def counters(self) -> Dict[str, float]:
        """The numbers the trainer logs (`archive/...`)."""
        return {"archive/size": float(self.size), "archive/inserted": float(self.inserted), "archive/improved": float(self.improved),
                "archive/dropped_queue": float(self.dropped_queue), "archive/errors": float(self.errors),
                "archive/eligible_rate": self.eligible / max(self.samples, 1),
                "archive/mean_best_reward": self._sum_best / self.size if self.size else float("nan"),
                "archive/submit_ms": 1000.0 * self.submit_seconds / max(self.batches + self.dropped_batches, 1)}

    # ------------------------------------------------------------------------------------------------ writer thread
    def _run(self) -> None:
        conn = self._conn
        last_commit = time.monotonic()
        in_tx = False

        def commit() -> None:
            nonlocal in_tx, last_commit
            if in_tx:
                conn.execute("COMMIT")
                in_tx = False
            last_commit = time.monotonic()

        try:
            while True:
                try:
                    item = self._q.get(timeout=min(self.flush_seconds, 1.0))
                except queue.Empty:
                    item = None
                if item is None:
                    if in_tx and time.monotonic() - last_commit >= self.flush_seconds:
                        commit()
                    continue
                kind = item[0]
                if kind == "batch":
                    try:
                        if not in_tx:
                            conn.execute("BEGIN")
                            in_tx = True
                        conn.execute("SAVEPOINT b")
                        try:
                            delta = self._apply(conn, item[1], item[2], item[3])
                            conn.execute("RELEASE b")
                        except BaseException:
                            conn.execute("ROLLBACK TO b")
                            conn.execute("RELEASE b")
                            raise
                        self._account(delta)
                    except Exception as e:        # a failing write must never stop the training run: counted, first message kept
                        self.errors += 1
                        self.last_error = f"{type(e).__name__}: {e}"
                        if self.errors == 1:
                            print(f"[gemma_mt] WARNING: translation archive write failed ({self.last_error}); further failures are only counted", file=sys.stderr, flush=True)
                    if in_tx and time.monotonic() - last_commit >= self.flush_seconds:
                        commit()
                elif kind == "flush":
                    try:
                        commit()
                    except Exception as e:
                        self.errors += 1
                        self.last_error = f"{type(e).__name__}: {e}"
                    item[1].set()
                else:                              # stop
                    break
        finally:
            try:
                commit()
            except Exception as e:
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"
            try:
                conn.close()
            except Exception:
                pass

    def _apply(self, conn: sqlite3.Connection, step: int, stage: str, rows: List[Row]) -> Dict[str, Any]:
        """One step's observations -> UPSERT rule. Pure database work; returns the deltas of the in-memory counters (applied only after the savepoint was released)."""
        thr, min_chrf, eps = self.min_reward_by_mode, self.min_consensus_chrf, self.epsilon
        keys: Dict[Tuple[str, str], str] = {}
        per: Dict[str, List[Any]] = {}              # key -> [n_seen, best eligible row or None]
        seen_lang: Counter = Counter()
        elig_lang: Counter = Counter()
        reasons: Counter = Counter()
        n_eligible = 0
        for row in rows:
            iso, source = row[0], row[1]
            kk = (iso, source)
            key = keys.get(kk)
            if key is None:
                key = keys[kk] = make_key(iso, source)
            e = per.get(key)
            if e is None:
                e = per[key] = [0, None]
            e[0] += 1
            seen_lang[iso] += 1
            why = eligibility(row, thr, min_chrf)
            if why is not None:
                reasons[why] += 1
                continue
            n_eligible += 1
            elig_lang[iso] += 1
            if e[1] is None or row[4] > e[1][4]:
                e[1] = row
        now = time.time()
        inserted = improved = 0
        d_sum = 0.0
        for key, (n_seen, best) in per.items():
            cur = conn.execute("SELECT reward, translation FROM best WHERE key = ?", (key,)).fetchone()
            if cur is None:
                if best is None:
                    continue                        # never eligible: no row (the counters keep the rejection reasons)
                iso, source, mode, tr, reward, _st, qe, chrf, dia, lid_label, n_chunks, _f, lbin = best
                conn.execute("INSERT INTO best (key, iso, mode, source, translation, reward, qe, consensus_chrf, diacritic_deficit, lid_ok, lid_label, n_chunks, step_first, step_best,"
                             " length_bin, stage, times_seen, n_improved, updated) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)",
                             (key, iso, mode, nfc(source), tr, reward, qe, chrf, dia, None if lid_label is None else 1, lid_label, n_chunks, step, step, lbin, stage, n_seen, now))
                inserted += 1
                d_sum += reward
                continue
            old_reward, old_tr = cur
            if best is not None and best[4] > old_reward + eps and best[3] != old_tr:
                _iso, _src, _mode, tr, reward, _st, qe, chrf, dia, lid_label, n_chunks, _f, lbin = best
                conn.execute("UPDATE best SET translation=?, reward=?, qe=?, consensus_chrf=?, diacritic_deficit=?, lid_ok=?, lid_label=?, n_chunks=?, step_best=?, length_bin=?, stage=?,"
                             " times_seen = times_seen + ?, n_improved = n_improved + 1, updated=? WHERE key=?",
                             (tr, reward, qe, chrf, dia, None if lid_label is None else 1, lid_label, n_chunks, step, lbin, stage, n_seen, now, key))
                improved += 1
                d_sum += reward - old_reward
            else:
                conn.execute("UPDATE best SET times_seen = times_seen + ? WHERE key = ?", (n_seen, key))
        # persistent counters (eligibility per language and the rejection reasons are what the owner needs to tune the thresholds)
        cnt: Counter = Counter({"samples": len(rows), "eligible": n_eligible, "inserted": inserted, "improved": improved})
        for r, k in reasons.items():
            cnt["reject:" + r] = k
        for l, k in seen_lang.items():
            cnt["seen:" + l] = k
        for l, k in elig_lang.items():
            cnt["eligible:" + l] = k
        conn.executemany("INSERT INTO counters(name, value) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET value = value + excluded.value", list(cnt.items()))
        return {"inserted": inserted, "improved": improved, "sum": d_sum, "samples": len(rows), "eligible": n_eligible}

    def _account(self, d: Dict[str, Any]) -> None:
        self.inserted += d["inserted"]
        self.improved += d["improved"]
        self.size += d["inserted"]
        self._sum_best += d["sum"]
        self.samples += d["samples"]
        self.eligible += d["eligible"]
        self.batches += 1


# ---------------------------------------------------------------------------------------------------- the trainer's hook
def archive_step(archive: TranslationArchive, reward: Any, samples: Sequence[Any], out: Any, step: int, stage: Optional[str] = None,
                 prompt_bins: Optional[Sequence[str]] = None, group_size: int = 1) -> bool:
    """Hand one reward call's results to the archive (no recomputation: every field is read off `RewardOutput`). `reward` = the TranslationReward (for the
    language's NOMINAL content mode: a blend language scored by QE alone because its group had no consensus is still a blend language here),
    `prompt_bins` = the length bin of each PROMPT (rows r, r+1, .. of one group share it), `step` = the logged step number (1-based)."""
    from gemma_mt.languages import canonical_iso

    n = len(samples)
    chrf = out.consensus_chrf if len(out.consensus_chrf) == n else [None] * n
    dia = out.diacritic_deficit if len(out.diacritic_deficit) == n else [None] * n
    lid = out.lid_label if len(out.lid_label) == n else [None] * n
    nch = out.n_chunks if len(getattr(out, "n_chunks", ())) == n else [0] * n
    fch = out.failed_chunks if len(getattr(out, "failed_chunks", ())) == n else [0] * n
    isos: Dict[str, Tuple[str, str]] = {}
    srcs: Dict[str, str] = {}
    rows: List[Row] = []
    G = max(int(group_size), 1)
    for r in range(n):
        s = samples[r]
        e = isos.get(s.iso)
        if e is None:
            iso = canonical_iso(s.iso) or s.iso
            e = isos[s.iso] = (iso, reward.content_spec(iso)[0])
        src = srcs.get(s.source)
        if src is None:
            src = srcs[s.source] = nfc(s.source)
        rows.append((e[0], src, e[1], out.texts[r], out.rewards[r], out.status[r], out.qe[r], chrf[r], dia[r], lid[r], nch[r], fch[r],
                     prompt_bins[r // G] if prompt_bins else ""))
    return archive.submit(step, stage or "", rows)


# ---------------------------------------------------------------------------------------------------- reading / CLI
def _connect_ro(path: str) -> sqlite3.Connection:
    if not os.path.exists(path):
        raise SystemExit(f"no such archive: {path}")
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=1")
    return conn


def read_rows(path: str, iso: Optional[Sequence[str]] = None, min_reward: Optional[float] = None) -> List[Dict[str, Any]]:
    conn = _connect_ro(path)
    try:
        q, args = "SELECT " + ", ".join(COLUMNS) + " FROM best", []
        cond = []
        if iso:
            cond.append("iso IN (%s)" % ",".join("?" * len(iso)))
            args += list(iso)
        if min_reward is not None:
            cond.append("reward >= ?")
            args.append(min_reward)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        return [dict(r) for r in conn.execute(q + " ORDER BY iso, key", args)]
    finally:
        conn.close()


def compute_stats(path: str) -> Dict[str, Any]:
    conn = _connect_ro(path)
    try:
        by_lang = [dict(r) for r in conn.execute(
            "SELECT iso, mode, COUNT(*) AS rows, AVG(reward) AS mean_reward, MIN(reward) AS min_reward, MAX(reward) AS max_reward, "
            "SUM(n_improved > 0) AS improved_rows, SUM(n_improved) AS improvements, AVG(times_seen) AS mean_times_seen FROM best GROUP BY iso, mode ORDER BY iso")]
        by_mode = [dict(r) for r in conn.execute(
            "SELECT mode, COUNT(*) AS rows, AVG(reward) AS mean_reward, SUM(n_improved > 0) AS improved_rows, SUM(n_improved) AS improvements FROM best GROUP BY mode ORDER BY mode")]
        by_stage = [dict(r) for r in conn.execute(
            "SELECT stage, COUNT(*) AS rows, AVG(reward) AS mean_reward FROM best GROUP BY stage ORDER BY stage")]
        by_bin = [dict(r) for r in conn.execute(
            "SELECT length_bin, COUNT(*) AS rows, AVG(reward) AS mean_reward FROM best GROUP BY length_bin ORDER BY length_bin")]
        tot = dict(conn.execute("SELECT COUNT(*) AS rows, AVG(reward) AS mean_reward, SUM(n_improved > 0) AS improved_rows, SUM(n_improved) AS improvements, "
                                "MIN(step_first) AS first_step, MAX(step_best) AS last_step, COUNT(DISTINCT iso) AS languages FROM best").fetchone())
        counters = {r["name"]: r["value"] for r in conn.execute("SELECT name, value FROM counters")}
    finally:
        conn.close()
    seen = {k[5:]: v for k, v in counters.items() if k.startswith("seen:")}
    elig = {k[9:]: v for k, v in counters.items() if k.startswith("eligible:")}
    for r in by_lang:
        r["eligible_rate"] = elig.get(r["iso"], 0.0) / seen[r["iso"]] if seen.get(r["iso"]) else None
    return {"total": tot, "by_mode": by_mode, "by_language": by_lang, "by_stage": by_stage, "by_length_bin": by_bin,
            "samples_seen": counters.get("samples", 0.0), "samples_eligible": counters.get("eligible", 0.0),
            "reject_reasons": {k[7:]: v for k, v in sorted(counters.items()) if k.startswith("reject:")},
            "eligible_rate_by_language": {l: (elig.get(l, 0.0) / v if v else None) for l, v in sorted(seen.items())}}


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def format_stats(st: Dict[str, Any]) -> str:
    t = st["total"]
    lines = [f"archive: {t['rows']} rows, {t['languages']} languages, mean best reward {_fmt(t['mean_reward'])}, "
             f"{_fmt(t['improved_rows'] or 0)} rows improved after their first insert ({_fmt(t['improvements'] or 0)} improvements), steps {t['first_step']}..{t['last_step']}",
             f"samples looked at {int(st['samples_seen'])}, eligible {int(st['samples_eligible'])} "
             f"({_fmt(st['samples_eligible'] / st['samples_seen'] if st['samples_seen'] else None)})", ""]
    if st["reject_reasons"]:
        lines.append("rejected by: " + ", ".join(f"{k} {int(v)}" for k, v in sorted(st["reject_reasons"].items(), key=lambda kv: -kv[1])))
        lines.append("")
    lines += ["rows per mode", "mode        rows   mean_reward  improved_rows  improvements"]
    for r in st["by_mode"]:
        lines.append(f"{r['mode']:<10} {r['rows']:>6}  {_fmt(r['mean_reward']):>11}  {_fmt(r['improved_rows'] or 0):>13}  {_fmt(r['improvements'] or 0):>12}")
    lines += ["", "rows per stage (stage of the step that produced the stored best; '' = not staged)", "stage       rows   mean_reward"]
    for r in st["by_stage"]:
        lines.append(f"{(r['stage'] or '-'):<10} {r['rows']:>6}  {_fmt(r['mean_reward']):>11}")
    lines += ["", "rows per prompt length bin", "bin         rows   mean_reward"]
    for r in st["by_length_bin"]:
        lines.append(f"{(r['length_bin'] or '-'):<10} {r['rows']:>6}  {_fmt(r['mean_reward']):>11}")
    lines += ["", "rows per language", "iso  mode        rows  mean_reward  min    max    improved_rows  improvements  mean_seen  eligible_rate"]
    for r in st["by_language"]:
        lines.append(f"{r['iso']:<4} {r['mode']:<9} {r['rows']:>6}  {_fmt(r['mean_reward']):>11}  {_fmt(r['min_reward'], 2):>5}  {_fmt(r['max_reward'], 2):>5}  "
                     f"{_fmt(r['improved_rows'] or 0):>13}  {_fmt(r['improvements'] or 0):>12}  {_fmt(r['mean_times_seen'], 1):>9}  {_fmt(r['eligible_rate']):>13}")
    return "\n".join(lines)


def export_archive(path: str, out_dir: str, fmt: str = "both") -> List[str]:
    rows = read_rows(path)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    written: List[str] = []
    if fmt in ("both", "jsonl"):
        p = os.path.join(out_dir, stem + ".jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        written.append(p)
    if fmt in ("both", "parquet"):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            if fmt == "parquet":
                raise SystemExit("parquet export needs pyarrow (not importable here); use --format jsonl, or run on the rental")
            print("pyarrow is not importable: wrote the jsonl only", file=sys.stderr)
        else:
            p = os.path.join(out_dir, stem + ".parquet")
            pq.write_table(pa.table({c: [r[c] for r in rows] for c in COLUMNS}), p)
            written.append(p)
    return written


def _cell(text: Any, limit: int) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\r", " ").replace("\n", " ⏎ ").replace("|", "\\|")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def review_markdown(path: str, per_language: int = 3, seed: int = 0, languages: Optional[Sequence[str]] = None, min_reward: Optional[float] = None,
                    max_chars: int = 700) -> str:
    """A random sample of up to `per_language` archived entries per language (deterministic in `seed`) as a markdown table for human review."""
    rows = read_rows(path, iso=languages, min_reward=min_reward)
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["iso"], []).append(r)
    rng = random.Random(seed)
    out = [f"# Archive review: {len(rows)} entries, {len(by)} languages, {per_language} per language at most (seed {seed})", "",
           "| lang | mode | reward | source (English) | best translation | key |", "|---|---|---|---|---|---|"]
    for iso in sorted(by):
        pool = by[iso]
        pick = rng.sample(pool, min(per_language, len(pool)))
        for r in sorted(pick, key=lambda x: x["key"]):
            out.append(f"| {iso} | {r['mode']} | {r['reward']:.3f} | {_cell(r['source'], max_chars)} | {_cell(r['translation'], max_chars)} | {r['key']} |")
    return "\n".join(out) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats", help="rows per language / mode / stage, mean best reward, improvements, rejection reasons")
    s.add_argument("path")
    s.add_argument("--json", action="store_true")
    e = sub.add_parser("export", help="parquet + jsonl with all columns")
    e.add_argument("path")
    e.add_argument("--out", default=None, help="directory (default: <archive dir>/export)")
    e.add_argument("--format", default="both", choices=("both", "parquet", "jsonl"))
    r = sub.add_parser("review", help="random sample per language as a markdown table for human review")
    r.add_argument("path")
    r.add_argument("--per-language", type=int, default=3)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--languages", default=None, help="comma separated iso codes (default: all)")
    r.add_argument("--min-reward", type=float, default=None)
    r.add_argument("--max-chars", type=int, default=700)
    r.add_argument("--out", default=None, help="write the markdown here instead of stdout")
    a = ap.parse_args(argv)
    if a.cmd == "stats":
        st = compute_stats(a.path)
        print(json.dumps(st, indent=2, ensure_ascii=False) if a.json else format_stats(st))
    elif a.cmd == "export":
        for p in export_archive(a.path, a.out or os.path.join(os.path.dirname(os.path.abspath(a.path)), "export"), a.format):
            print("wrote", p)
    else:
        md = review_markdown(a.path, a.per_language, a.seed, a.languages.split(",") if a.languages else None, a.min_reward, a.max_chars)
        if a.out:
            with open(a.out, "w", encoding="utf-8") as f:
                f.write(md)
            print("wrote", a.out)
        else:
            sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
