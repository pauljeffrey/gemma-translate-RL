# SPDX-License-Identifier: AGPL-3.0-or-later
"""The post-SFT benchmark evaluation (owner 2026-10-18): the SFT model and the untouched base model, English -> African, on the TEST sets
of FLORES+ (devtest), NTREX-128 and MAFAND-MT, with a contamination filter, AfriCOMET and a marker file for the supervisor.

    bash scripts/eval_sft_benchmarks.sh [model-id]        # the entry point on the rental (sets the environment)

What it does, in order (every step has a wall-clock limit; the process never waits forever):
  1. preflight: tokens, interpreters, contamination flag files, GPU memory (waits at most --gpu-wait seconds for it to be free)
  2. benchmark data: downloaded when missing (fetch_data), rows normalised, `--max-per-language` applied, contamination flags attached
  3. per system (SFT first, then base): download the snapshot, start the vLLM worker (or load the HF model), greedy generation benchmark by
     benchmark (cached on disk: a restart does not regenerate), close the engine, AfriCOMET-STL / QE in child processes (per-pair cache)
  4. results.json / results.csv / predictions.jsonl / report.md in --out, then EVAL_DONE (all fine) or EVAL_FAILED (reasons, partial
     results are still written)

Exit code 0 only when EVAL_DONE was written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from gemma_mt import format as fmt
from gemma_mt.eval import benchmarks as B
from gemma_mt.eval import contamination as C
from gemma_mt.eval import eval_translation as E
from gemma_mt.eval import runtime as RT
from gemma_mt.eval.runtime import log
from gemma_mt.languages import SUPPORTED_RL, canonical_iso

DEFAULT_SFT = (os.environ['HF_ORG'] + "/gemma3-270m-translate")
DEFAULT_BASE = "google/gemma-3-270m-it"
MARKERS = ("EVAL_DONE", "EVAL_FAILED", "EVAL_RUNNING")


@dataclass
class Failure:
    system: str
    benchmark: str
    stage: str
    reason: str

    def __str__(self) -> str:
        return f"[{self.system}/{self.benchmark}/{self.stage}] {self.reason}"


@dataclass
class Hooks:
    """Everything that needs a GPU, the network or a subprocess; tests replace these."""
    resolve_model: Callable[..., Any] = RT.resolve_model
    load_tokenizer_and_stops: Callable[..., Any] = RT.load_tokenizer_and_stops
    make_sampler: Callable[..., Any] = RT.make_vllm_sampler
    load_hf_model: Optional[Callable[..., Any]] = None
    comet_factory: Optional[Callable[..., Any]] = None          # (args, out) -> object with .reference .qe .close() .stats()
    ensure_data: Callable[..., Any] = None                       # (root, names, isos, token) -> info
    wait_for_gpu: Callable[..., Any] = RT.wait_for_gpu


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-model", default=DEFAULT_SFT, help="Hub id or local dir of the SFT model")
    ap.add_argument("--sft-revision", default=None, help="branch / tag / commit of the SFT model (default: main = best validation checkpoint)")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--out", default="/workspace/sft_build/eval/sft_best")
    ap.add_argument("--bench-dir", default="/workspace/sft_build/bench",
                    help="contamination_<benchmark>.json live here; the test sets are downloaded to <bench-dir>/test_sets")
    ap.add_argument("--contamination-dir", default=None,
                    help="directory with contamination_<benchmark>.json (default: --bench-dir); lets the flag files of the extra benchmarks live next to their test sets")
    ap.add_argument("--benchmarks", nargs="*", default=list(B.BENCHMARKS),
                    help="benchmarks to run, space or comma separated (default: flores_plus ntrex mafand). Opt-in extras (D-42): smol_sent smol_doc gatitos, "
                         "or the group names smol / extra (= those three) and all, e.g. --benchmarks smol_sent,smol_doc,gatitos")
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--max-per-language", type=int, default=0)
    ap.add_argument("--bench-limit", nargs="*", default=[], metavar="BENCHMARK=N",
                    help="per-benchmark cap on the rows per language, applied AFTER the contamination flags are attached: the CLEAN rows of a language are kept "
                         "first, the rest is filled with its earliest rows (so a small sample still has a usable clean subset). Overrides --max-per-language, "
                         "e.g. --bench-limit smol_doc=40 gatitos=1000")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--doc-max-new-tokens", type=int, default=1024, help="generation budget of the document-level benchmarks (smol_doc)")
    ap.add_argument("--word-max-new-tokens", type=int, default=48, help="generation budget of the word-level benchmarks (gatitos)")
    ap.add_argument("--sampler", default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--sampler-fallback-hf", action="store_true", help="if the vLLM worker cannot start, generate with HF (slow); recorded in the results")
    ap.add_argument("--vllm-python", default=None)
    ap.add_argument("--vllm-gpu-util", type=float, default=0.2)
    ap.add_argument("--vllm-max-model-len", type=int, default=4096)
    ap.add_argument("--vllm-max-num-seqs", type=int, default=512)
    ap.add_argument("--hf-batch-size", type=int, default=64)
    ap.add_argument("--doc-batch-size", type=int, default=0, help="HF batch size of the document benchmark smol_doc (0 = --hf-batch-size); documents need 4-5k tokens of KV per row")
    ap.add_argument("--word-batch-size", type=int, default=0, help="HF batch size of the word-level benchmark gatitos (0 = --hf-batch-size)")
    ap.add_argument("--unit-rows", type=int, default=8192,
                    help="generation unit: consecutive languages are packed into blocks of about this many rows; a block is cached and resumed as a whole")
    ap.add_argument("--vllm-chunk", type=int, default=8192, help="prompts per engine call (progress / deadline granularity)")
    ap.add_argument("--budget-seconds", type=float, default=0,
                    help="stop gracefully (exit code 75, progress kept) when this invocation has run this long; for shared GPUs, rerun to continue")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--comet-python", default=None)
    ap.add_argument("--comet-pythonpath", default=None)
    ap.add_argument("--comet-batch-size", type=int, default=64)
    ap.add_argument("--comet-sequential", action="store_true", help="keep one AfriCOMET model on the GPU at a time (nearly full GPU)")
    ap.add_argument("--min-subset-rows", type=int, default=30)
    ap.add_argument("--gen-timeout", type=float, default=5400, help="seconds of generation per system and benchmark")
    ap.add_argument("--score-timeout", type=float, default=5400, help="seconds of AfriCOMET scoring per system and benchmark")
    ap.add_argument("--total-timeout", type=float, default=0, help="seconds for the whole run; the watchdog kills the process (0 = none)")
    ap.add_argument("--gpu-min-free-gib", type=float, default=24.0)
    ap.add_argument("--gpu-wait", type=float, default=1800.0)
    ap.add_argument("--extra-libs", default=None, help="directories appended to sys.path (e.g. pylibs for sacrebleu)")
    ap.add_argument("--allow-missing-contamination", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="ignore cached generations / scores")
    return ap


# ------------------------------------------------------------------------------------------------ small helpers
def write_marker(out: str, name: str, payload: Dict[str, Any]) -> None:
    os.makedirs(out, exist_ok=True)
    for m in MARKERS:
        if m != name and os.path.exists(os.path.join(out, m)) and name in ("EVAL_DONE", "EVAL_FAILED"):
            try:
                os.remove(os.path.join(out, m))
            except OSError:
                pass
    tmp = os.path.join(out, name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    os.replace(tmp, os.path.join(out, name))


def rows_digest(rows: Sequence[Dict[str, Any]]) -> str:
    h = hashlib.sha1()
    for r in rows:
        h.update(f"{r['iso']}\x00{r['id']}\x00{r['english']}\x01".encode("utf-8"))
    return h.hexdigest()


def bench_max_new_tokens(args, bench: str) -> int:
    """Generation budget of one benchmark: the global --max-new-tokens, except documents (smol_doc) and words (gatitos)."""
    if bench in B.DOC_LEVEL_BENCHMARKS:
        return int(args.doc_max_new_tokens)
    if bench in B.WORD_LEVEL_BENCHMARKS:
        return int(args.word_max_new_tokens)
    return int(args.max_new_tokens)


def bench_batch_size(args, bench: str) -> int:
    """HF batch size of one benchmark (the vLLM path ignores it)."""
    if bench in B.DOC_LEVEL_BENCHMARKS and getattr(args, "doc_batch_size", 0):
        return int(args.doc_batch_size)
    if bench in B.WORD_LEVEL_BENCHMARKS and getattr(args, "word_batch_size", 0):
        return int(args.word_batch_size)
    return int(args.hf_batch_size)


def bench_limits(args) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for spec in getattr(args, "bench_limit", None) or []:
        name, _, n = str(spec).partition("=")
        if name not in B.ALL_BENCHMARKS or not n.strip().isdigit():
            raise ValueError(f"--bench-limit expects BENCHMARK=N with a benchmark in {B.ALL_BENCHMARKS}, got {spec!r}")
        out[name] = int(n)
    return out


def gen_cache_path(out: str, system: str, bench: str, unit: int) -> str:
    return os.path.join(out, "work", system, f"gen_{bench}_{unit:03d}.jsonl")


def make_units(rows: Sequence[Dict[str, Any]], unit_rows: int) -> List[List[int]]:
    """Row indices of one benchmark cut into generation units: consecutive languages are packed until `unit_rows` rows are reached (a
    language is never split, so a unit is a self-contained block that is cached, resumed and budgeted as a whole)."""
    units: List[List[int]] = []
    cur: List[int] = []
    last_iso = None
    for i, r in enumerate(rows):
        if r["iso"] != last_iso and cur and len(cur) >= unit_rows:
            units.append(cur)
            cur = []
        cur.append(i)
        last_iso = r["iso"]
    if cur:
        units.append(cur)
    return units


def save_generations(path: str, header: Dict[str, Any], outputs: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for o in outputs:
            f.write(json.dumps({k: o.get(k) for k in ("raw", "hyp", "prefix_status", "finished")}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_generations(path: str, header: Dict[str, Any]):
    """(outputs, stored header) when the file exists and its header equals `header` on every key of `header`; else None."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            head = json.loads(f.readline())
            outs = [json.loads(line) for line in f if line.strip()]
    except Exception:
        return None
    if any(head.get(k) != v for k, v in header.items()) or len(outs) != header["n_rows"]:
        return None
    return outs, head


def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(), "host": platform.node()}
    for m in ("torch", "transformers", "sacrebleu", "huggingface_hub"):
        try:
            info[m] = __import__(m).__version__
        except Exception:
            pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
        if out:
            info["gpu"] = out.splitlines()[0]
    except Exception:
        pass
    return info


def training_state_of(path: str) -> Dict[str, Any]:
    p = os.path.join(path, "training_state.json")
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return {k: d.get(k) for k in ("step", "val_loss_macro", "best_step", "pushed_at_utc", "tokens_seen", "epoch_equivalents")}
    except Exception:
        return {}


# ------------------------------------------------------------------------------------------------ data
def load_benchmark_rows(args, hooks: Hooks, failures: List[Failure]) -> Dict[str, List[Dict[str, Any]]]:
    root = os.path.join(args.bench_dir, "test_sets")
    isos = [canonical_iso(l) for l in args.languages] if args.languages else None
    token = os.environ.get("HF_TOKEN")
    ensure = hooks.ensure_data
    if ensure is None:
        from gemma_mt.eval.fetch_data import ensure_benchmarks as ensure
    out: Dict[str, List[Dict[str, Any]]] = {}
    args.benchmarks = B.parse_benchmark_names(args.benchmarks)
    limits = bench_limits(args)
    for b in args.benchmarks:
        try:
            ensure(root, [b], isos, token)
            rows = B.load_benchmark(b, path=os.path.join(root, b), isos=isos, limit=0 if b in limits else args.max_per_language)
            if not rows:
                raise RuntimeError("no rows loaded")
            out[b] = rows
            log(f"benchmark {b}: {len(rows)} rows, {len({r['iso'] for r in rows})} languages")
        except Exception as e:  # noqa: BLE001 - one broken download must not stop the other benchmarks
            failures.append(Failure("-", b, "data", f"{type(e).__name__}: {e}"))
            log(f"benchmark {b} FAILED: {e}")
    return out


def limit_rows(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """At most `n` rows per language (original order kept): its clean rows (`contaminated` is False) first, then its earliest other rows."""
    by: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        by.setdefault(r["iso"], []).append(i)
    keep = set()
    for idx in by.values():
        if len(idx) <= n:
            keep.update(idx)
            continue
        clean = [i for i in idx if rows[i].get("contaminated") is False][:n]
        taken = set(clean)
        keep.update(clean)
        keep.update([i for i in idx if i not in taken][: n - len(clean)])
    return [r for i, r in enumerate(rows) if i in keep]


def attach_flags(rows_by_bench: Dict[str, List[Dict[str, Any]]], args, failures: List[Failure]) -> Dict[str, str]:
    cdir = getattr(args, "contamination_dir", None) or args.bench_dir
    files = C.default_flag_files(cdir, list(rows_by_bench))
    for b, rows in rows_by_bench.items():
        if b not in files:
            msg = f"no contamination file {os.path.join(cdir, f'contamination_{b}.json')}"
            if not args.allow_missing_contamination:
                raise RuntimeError(msg + " (run gemma_mt.eval.contamination on the box that has gemma/windows_sft, or pass "
                                         "--allow-missing-contamination)")
            log("warning: " + msg)
    flat = [r for rows in rows_by_bench.values() for r in rows]
    missing = C.apply_flags(flat, files)
    for b, n in missing.items():
        if n and b in files:
            failures.append(Failure("-", b, "contamination", f"{n} rows have no flag in {files[b]} (wrong benchmark version?)"))
    return files


# ------------------------------------------------------------------------------------------------ one system
WARNINGS: List[str] = []          # things that did not stop the run but that the reader must know (recorded in results.json and EVAL_DONE)


class BudgetExhausted(RuntimeError):
    """`--budget-seconds` of this invocation are used up: progress is on disk, run the script again (exit code 75)."""


def run_system(name: str, spec: str, revision: Optional[str], rows_by_bench: Dict[str, List[Dict[str, Any]]], args, hooks: Hooks,
               comet: Any, failures: List[Failure], out: str, budget_deadline: Optional[float] = None) -> Dict[str, Any]:
    """Generate + score one system on every benchmark. Returns {"meta": {...}, "benchmarks": ..., "macro_all": ...} (+ outputs for
    predictions under key "_outputs")."""
    t_sys = time.time()
    token = os.environ.get("HF_TOKEN")
    pin = os.path.join(out, "work", name, "revision.json")
    if revision is None and not args.no_resume and os.path.isfile(pin):          # the Hub branch may move between two invocations of one run: stay on the first commit
        try:
            with open(pin, encoding="utf-8") as f:
                p = json.load(f)
            if p.get("model") == spec and p.get("sha"):
                revision = p["sha"]
                log(f"{name}: pinned to the commit {revision} resolved by the first invocation of this run")
        except Exception:
            pass
    ref = hooks.resolve_model(spec, token=token, revision=revision)
    if ref.sha and not os.path.isdir(spec):
        os.makedirs(os.path.dirname(pin), exist_ok=True)
        with open(pin, "w", encoding="utf-8") as f:
            json.dump({"model": spec, "sha": ref.sha}, f)
    tok, stop, pad = hooks.load_tokenizer_and_stops(ref.path)
    meta: Dict[str, Any] = {"name": name, "model": spec, "revision": revision, "sha": ref.sha, "path": ref.path, "decoding": "greedy",
                            "max_new_tokens": args.max_new_tokens, "prompt_template": fmt.INSTRUCTION_TEMPLATE, "direction": fmt.EN2X,
                            "stop_ids": stop.ids, "training_state": training_state_of(ref.path), "timings": {}}
    outputs: Dict[str, List[Optional[Dict[str, Any]]]] = {b: [None] * len(rows) for b, rows in rows_by_bench.items()}
    todo: List[tuple] = []                                         # (benchmark, unit index, row indices, header)
    for b, rows in rows_by_bench.items():
        meta["timings"][b] = {"generation_seconds": 0.0, "units": 0, "units_cached": 0, "sampler": None}
        for k, idx in enumerate(make_units(rows, args.unit_rows)):
            hdr = {"model": spec, "sha": ref.sha, "max_new_tokens": bench_max_new_tokens(args, b), "n_rows": len(idx), "decoding": "greedy",
                   "rows_digest": rows_digest([rows[i] for i in idx])}
            cached = None if args.no_resume else load_generations(gen_cache_path(out, name, b, k), hdr)
            meta["timings"][b]["units"] += 1
            if cached is not None:
                for i, o in zip(idx, cached[0]):
                    outputs[b][i] = o
                meta["timings"][b]["units_cached"] += 1
                meta["timings"][b]["generation_seconds"] += float(cached[1].get("seconds") or 0)
                meta["timings"][b]["sampler"] = cached[1].get("sampler")
            else:
                todo.append((b, k, idx, hdr))
    if todo:
        log(f"{name}: {len(todo)} generation unit(s) to run")
        free = hooks.wait_for_gpu(args.gpu_min_free_gib, args.gpu_wait)            # raises TimeoutError: the card is still held by someone else
        meta["gpu_free_gib_before_start"] = None if free != free else round(free, 1)
        sampler = model = None
        sampler_used = args.sampler
        if args.sampler == "vllm":
            try:
                sampler = hooks.make_sampler(ref.path, python=args.vllm_python, gpu_util=args.vllm_gpu_util,
                                             max_model_len=args.vllm_max_model_len, max_num_seqs=args.vllm_max_num_seqs,
                                             log_path=os.path.join(out, "work", name, "vllm_worker.log"),
                                             request_timeout=args.gen_timeout + 120)
            except Exception as e:  # noqa: BLE001
                if not args.sampler_fallback_hf:
                    raise
                log(f"WARNING: vLLM worker failed to start ({type(e).__name__}: {str(e)[:300]}); falling back to HF generate")
                WARNINGS.append(f"[{name}] vLLM unavailable, HF fallback used: {str(e)[:300]}")
                sampler_used = "hf"
        if sampler is None:
            model = hooks.load_hf_model(ref.path, args.dtype) if hooks.load_hf_model else _load_hf(ref.path, args.dtype)
            sampler_used = "hf"
        meta["sampler"] = sampler_used
        bench_start: Dict[str, float] = {}
        broken = False
        try:
            for b, k, idx, hdr in todo:
                rows = rows_by_bench[b]
                bench_start.setdefault(b, time.time())
                if budget_deadline is not None and time.time() >= budget_deadline:
                    raise BudgetExhausted(f"{name}: invocation budget used up before {b} unit {k}")
                if broken or time.time() > bench_start[b] + args.gen_timeout:
                    if not broken:
                        failures.append(Failure(name, b, "generation", f"wall-clock limit of {args.gen_timeout:.0f}s per benchmark reached"))
                    continue
                t0 = time.time()
                sub = [rows[i] for i in idx]
                log(f"{name}/{b} unit {k}: generating {len(sub)} rows ({sampler_used})")
                dl = bench_start[b] + args.gen_timeout
                if budget_deadline is not None:
                    dl = min(dl, budget_deadline)
                try:
                    outs = E.generate_translations(model, tok, sub, stop_ids=stop.ids, pad_id=pad, device=getattr(model, "device", None),
                                                   batch_size=bench_batch_size(args, b), max_new_tokens=bench_max_new_tokens(args, b), sampler=sampler,
                                                   deadline=dl, chunk=args.vllm_chunk, log=log)
                except E.TimeLimitExceeded as e:
                    if budget_deadline is not None and time.time() >= budget_deadline:
                        raise BudgetExhausted(str(e)) from e
                    failures.append(Failure(name, b, "generation", f"{e}"))
                    continue
                except Exception as e:  # noqa: BLE001
                    failures.append(Failure(name, b, "generation", f"{type(e).__name__}: {str(e)[:500]}"))
                    log(f"{name}/{b} unit {k}: generation FAILED: {e}")
                    broken = True                           # engine / model broken: the remaining units would fail the same way
                    continue
                dt = time.time() - t0
                for i, o in zip(idx, outs):
                    outputs[b][i] = o
                tm = meta["timings"][b]
                tm["generation_seconds"] = round(tm["generation_seconds"] + dt, 1)
                tm["sampler"] = sampler_used
                save_generations(gen_cache_path(out, name, b, k), {**hdr, "seconds": round(dt, 1), "sampler": sampler_used}, outs)
                log(f"{name}/{b} unit {k}: {len(sub)} rows in {dt:.0f}s ({len(sub) / max(dt, 1e-9):.0f} rows/s)")
        finally:
            if sampler is not None:
                try:
                    sampler.close()
                except Exception:
                    pass
            model = None
            release_cuda()
    meta["sampler"] = "+".join(sorted({t["sampler"] for t in meta["timings"].values() if t.get("sampler")})) or meta.get("sampler", "none")
    for b in list(outputs):
        if any(o is None for o in outputs[b]):                    # a benchmark with a failed unit is not evaluated (partial numbers would mislead)
            if not any(f.benchmark == b and f.system == name for f in failures):
                failures.append(Failure(name, b, "generation", "incomplete generations"))
            del outputs[b]

    # ---- scoring
    parts: Dict[str, Dict[str, Any]] = {}
    for b, outs in outputs.items():
        rows = rows_by_bench[b]
        t0 = time.time()
        try:
            dl = time.time() + args.score_timeout
            if budget_deadline is not None:
                dl = min(dl, budget_deadline)
            use_comet = comet if b not in B.WORD_LEVEL_BENCHMARKS + B.DOC_LEVEL_BENCHMARKS else None     # words / documents: AfriCOMET is not meaningful
            res = E.evaluate_rows(rows, outs, comet_ref=use_comet.reference if use_comet else None, comet_qe=use_comet.qe if use_comet else None,
                                  min_subset_rows=args.min_subset_rows, deadline=dl, log=log)
        except E.TimeLimitExceeded as e:
            if budget_deadline is not None and time.time() >= budget_deadline:
                raise BudgetExhausted(str(e)) from e
            failures.append(Failure(name, b, "scoring", f"{type(e).__name__}: {str(e)[:500]}"))
            log(f"{name}/{b}: scoring FAILED: {e}")
            res = E.evaluate_rows(rows, outs, min_subset_rows=args.min_subset_rows)
            res["comet_failed"] = True
        except Exception as e:  # noqa: BLE001
            failures.append(Failure(name, b, "scoring", f"{type(e).__name__}: {str(e)[:500]}"))
            log(f"{name}/{b}: scoring FAILED: {e}")
            res = E.evaluate_rows(rows, outs, min_subset_rows=args.min_subset_rows)          # text metrics only, so the numbers are not lost
            res["comet_failed"] = True
        meta["timings"].setdefault(b, {})["scoring_seconds"] = round(meta["timings"].get(b, {}).get("scoring_seconds", 0) + time.time() - t0, 1)
        parts[b] = res["benchmarks"][b]
    merged = E.overall(parts, args.min_subset_rows)
    if comet is not None and hasattr(comet, "release"):
        comet.release()                                          # the AfriCOMET workers' GPU memory must be free for the next system's generation
    meta["seconds_total"] = round(time.time() - t_sys, 1)
    meta["spbleu_tokenizer"] = E.M.spbleu_tokenizer()
    if any(b in B.EXTRA_BENCHMARKS for b in rows_by_bench):
        meta["max_new_tokens_by_benchmark"] = {b: bench_max_new_tokens(args, b) for b in rows_by_bench}
    meta["comet"] = None if comet is None else comet.stats()
    return {"meta": meta, **merged, "_outputs": outputs}


def release_cuda() -> None:
    """Give the CUDA caching allocator's idle blocks back to the card. Without it the second system of a run (base) would see the first one's cache as
    used memory and wait for `--gpu-min-free-gib` on a shared card (found 2026-10-19: the base system waited for the free memory its own SFT run held)."""
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort, never an error of the evaluation
        pass


def _load_hf(path: str, dtype_name: str):
    import torch

    from gemma_mt.modeling import load_model

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    return load_model(path, dtype=dtype, device="cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------------------------ whole run
def write_outputs(out: str, run_meta: Dict[str, Any], systems: Dict[str, Dict[str, Any]], rows_by_bench, failures: List[Failure]) -> None:
    os.makedirs(out, exist_ok=True)
    clean = {n: {k: v for k, v in s.items() if k != "_outputs"} for n, s in systems.items()}
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": run_meta, "failures": [str(x) for x in failures], "warnings": list(WARNINGS), "systems": clean}, f, indent=1,
                  ensure_ascii=False)
    E.write_csv(os.path.join(out, "results.csv"), {}, systems={n: s for n, s in clean.items()})
    with open(os.path.join(out, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for sname, s in systems.items():
            for b, outs in s.get("_outputs", {}).items():
                for r, o in zip(rows_by_bench[b], outs):
                    f.write(json.dumps({"system": sname, **r, "hyp": o["hyp"], "raw": o.get("raw", ""),
                                        "prefix_status": o.get("prefix_status"), "finished": o.get("finished", True)},
                                       ensure_ascii=False) + "\n")
    from gemma_mt.eval.sft_report import render_report

    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_report({"meta": run_meta, "failures": [str(x) for x in failures], "warnings": list(WARNINGS), "systems": clean}))


def run(args, hooks: Optional[Hooks] = None) -> int:
    hooks = hooks or Hooks()
    out = args.out
    os.makedirs(out, exist_ok=True)
    for m in MARKERS:
        try:
            os.remove(os.path.join(out, m))
        except OSError:
            pass
    t0 = time.time()
    WARNINGS.clear()
    budget_deadline = t0 + args.budget_seconds if args.budget_seconds else None
    failures: List[Failure] = []
    systems: Dict[str, Dict[str, Any]] = {}
    rows_by_bench: Dict[str, List[Dict[str, Any]]] = {}
    run_meta: Dict[str, Any] = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args), "environment": environment_info()}
    write_marker(out, "EVAL_RUNNING", {"pid": os.getpid(), "started": run_meta["started"]})

    def _on_timeout():
        write_marker(out, "EVAL_FAILED", {"reason": f"total wall-clock limit of {args.total_timeout:.0f}s reached; process killed",
                                            "failures": [str(x) for x in failures]})
    dog = RT.Watchdog(args.total_timeout, _on_timeout).start() if args.total_timeout else None
    comet = None
    incomplete: Optional[str] = None
    try:
        RT.add_extra_libs(args.extra_libs)
        if args.sampler == "vllm" and not args.vllm_python and hooks.make_sampler is RT.make_vllm_sampler:
            raise RuntimeError("--sampler vllm needs --vllm-python")
        if not os.environ.get("HF_TOKEN") and not args.allow_missing_contamination and hooks.resolve_model is RT.resolve_model:
            log("warning: HF_TOKEN is not set (the SFT repo is private, FLORES+ and gemma-3 are gated)")
        rows_by_bench = load_benchmark_rows(args, hooks, failures)
        if not rows_by_bench:
            raise RuntimeError("no benchmark could be loaded")
        run_meta["contamination_files"] = attach_flags(rows_by_bench, args, failures)
        for b, n in bench_limits(args).items():
            if b in rows_by_bench:
                rows_by_bench[b] = limit_rows(rows_by_bench[b], n)
        run_meta["rows"] = {b: len(r) for b, r in rows_by_bench.items()}
        if not args.no_comet:
            if hooks.comet_factory:
                comet = hooks.comet_factory(args, out)
            elif args.comet_python:
                comet = RT.CometPair(python=args.comet_python, pythonpath=args.comet_pythonpath, batch_size=args.comet_batch_size,
                                     cache_dir=None if args.no_resume else os.path.join(out, "work", "comet_cache"),
                                     sequential=args.comet_sequential)
            else:
                raise RuntimeError("AfriCOMET needs --comet-python (child process with transformers 4.x, D-33) or --no-comet")
        plan = [("sft", args.sft_model, args.sft_revision)] + ([] if args.skip_base else [("base", args.base_model, None)])
        for name, spec, rev in plan:
            try:
                systems[name] = run_system(name, spec, rev, rows_by_bench, args, hooks, comet, failures, out, budget_deadline)
            except BudgetExhausted:
                raise
            except Exception as e:  # noqa: BLE001
                failures.append(Failure(name, "-", "system", f"{type(e).__name__}: {str(e)[:800]}"))
                log(f"system {name} FAILED:\n{traceback.format_exc()}")
    except BudgetExhausted as e:
        incomplete = str(e)
    except BaseException as e:  # noqa: BLE001
        failures.append(Failure("-", "-", "pipeline", f"{type(e).__name__}: {str(e)[:800]}"))
        log(f"pipeline FAILED:\n{traceback.format_exc()}")
    finally:
        if comet is not None:
            try:
                comet.close()
            except Exception:
                pass
    run_meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_meta["seconds"] = round(time.time() - t0, 1)
    if incomplete is not None:
        if dog:
            dog.cancel()
        write_marker(out, "EVAL_RUNNING", {"pid": os.getpid(), "incomplete": incomplete, "at": run_meta["finished"]})
        log(f"budget of {args.budget_seconds:.0f}s used up ({incomplete}); progress is cached, run again (exit code 75)")
        return 75
    if systems:
        try:
            write_outputs(out, run_meta, systems, rows_by_bench, failures)
        except Exception as e:  # noqa: BLE001
            failures.append(Failure("-", "-", "write", f"{type(e).__name__}: {e}"))
            log(f"writing the outputs FAILED:\n{traceback.format_exc()}")
    if dog:
        dog.cancel()
    expected = {"sft"} | (set() if args.skip_base else {"base"})
    ok = not failures and expected <= set(systems) and all(
        set(systems[n].get("benchmarks", {})) == set(rows_by_bench) for n in expected if n in systems)
    if ok:
        write_marker(out, "EVAL_DONE", {"finished": run_meta["finished"], "seconds": run_meta["seconds"], "out": out,
                                         "sft_model": args.sft_model, "sft_sha": systems["sft"]["meta"].get("sha"), "warnings": list(WARNINGS),
                                         "files": ["results.json", "results.csv", "predictions.jsonl", "report.md"]})
        log(f"EVAL_DONE in {run_meta['seconds']:.0f}s -> {out}")
        return 0
    write_marker(out, "EVAL_FAILED", {"finished": run_meta["finished"], "seconds": run_meta["seconds"],
                                       "reason": "; ".join(str(x) for x in failures) or "a system or benchmark is missing",
                                       "failures": [str(x) for x in failures], "partial_results": bool(systems)})
    log("EVAL_FAILED: " + "; ".join(str(x) for x in failures))
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.benchmarks = B.parse_benchmark_names(args.benchmarks)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
