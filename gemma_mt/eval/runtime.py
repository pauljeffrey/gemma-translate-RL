# SPDX-License-Identifier: AGPL-3.0-or-later
"""Runtime plumbing of the evaluation scripts: model download, vLLM sampler for greedy evaluation, AfriCOMET child-process scorers with a
per-row score cache, GPU memory wait, a global watchdog. Everything that touches a GPU, the network or a subprocess lives here so the
pure logic (eval_translation, contamination, reports) stays testable on a CPU laptop.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ models
@dataclass
class ModelRef:
    spec: str                    # what was asked for (Hub id or local dir)
    path: str                    # local directory with config + weights + tokenizer
    repo: Optional[str]          # Hub id, None for a local directory
    revision: Optional[str]      # branch / tag asked for
    sha: Optional[str]           # commit hash of the downloaded snapshot (Hub only)


_SHA = re.compile(r"^[0-9a-f]{40}$")


def resolve_model(spec: str, token: Optional[str] = None, revision: Optional[str] = None) -> ModelRef:
    """A local directory is used as it is; anything else is a Hub id downloaded with `snapshot_download` (config, weights, tokenizer files;
    HF_HOME decides where) and the commit sha is read from the snapshot directory name."""
    if os.path.isdir(spec):
        return ModelRef(spec, os.path.abspath(spec), None, None, None)
    from huggingface_hub import snapshot_download

    # top-level files only (the training repo may also hold best/ or optimizer folders that evaluation does not need)
    path = snapshot_download(spec, revision=revision, token=token,
                             allow_patterns=["config.json", "generation_config.json", "model.safetensors", "model-*.safetensors",
                                             "model.safetensors.index.json", "tokenizer*", "special_tokens_map.json",
                                             "added_tokens.json", "chat_template*", "training_state.json"])
    sha = os.path.basename(os.path.normpath(path))
    return ModelRef(spec, path, spec, revision, sha if _SHA.match(sha) else None)


def load_tokenizer_and_stops(path: str):
    """(tokenizer, StopTokens, pad_id): tokenizer from the model dir, template checked against the Gemma 3 reference, stop ids resolved
    from the tokenizer (+ generation_config warnings printed)."""
    from transformers import AutoTokenizer, GenerationConfig

    from gemma_mt import format as fmt

    tok = AutoTokenizer.from_pretrained(path)
    fmt.assert_template_matches_reference(tok)
    try:
        gcfg = GenerationConfig.from_pretrained(path)
    except Exception:
        gcfg = None
    stop = fmt.resolve_stop_tokens(tok, gcfg)
    for w in stop.warnings:
        log(f"warning: {w}")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return tok, stop, pad


# ------------------------------------------------------------------------------------------------ vLLM
def make_vllm_sampler(model_dir: str, *, python: str, gpu_util: float = 0.2, max_model_len: int = 4096, max_num_seqs: int = 512,
                      log_path: Optional[str] = None, sync_dir: Optional[str] = None, startup_timeout: float = 900.0,
                      request_timeout: float = 3600.0, seed: int = 0, extra: Optional[Dict[str, Any]] = None,
                      engine_factory: Optional[Callable[..., Any]] = None):
    """A VLLMSampler on a vLLM worker process (`python` = interpreter of the environment that has vllm) that loads `model_dir` itself;
    nothing is synced. `engine_factory` replaces the real WorkerEngine in tests."""
    from gemma_mt.rl.vllm_engine import EngineConfig, WorkerEngine
    from gemma_mt.rl.vllm_sampler import VLLMSampler

    ec = EngineConfig(model=model_dir, gpu_memory_utilization=gpu_util, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                      seed=seed, extra=dict(extra or {}))
    if engine_factory is not None:
        engine = engine_factory(ec)
    else:
        engine = WorkerEngine(ec, python=python, log_path=log_path, sync_dir=sync_dir or ("/dev/shm" if os.path.isdir("/dev/shm") else "/tmp"),
                              startup_timeout=startup_timeout, request_timeout=request_timeout)
    sampler = VLLMSampler(engine, seed=seed)
    sampler.assume_loaded(0)
    return sampler


# ------------------------------------------------------------------------------------------------ AfriCOMET
class CachedScorer:
    """Wraps a scorer `fn(srcs, mts[, refs]) -> scores` with an append-only jsonl cache keyed by sha1(src, mt, ref): a restarted evaluation
    does not score a pair twice. `fn` is called only for the pairs not in the cache, in the order given."""

    def __init__(self, fn: Callable, path: Optional[str], reference_based: bool):
        self.fn, self.path, self.reference_based = fn, path, reference_based
        self.cache: Dict[str, float] = {}
        self.hits = self.misses = 0
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        k, v = json.loads(line)
                        self.cache[k] = float(v)
                    except Exception:
                        continue                              # a half-written last line
        self._fh = open(path, "a", encoding="utf-8") if path else None

    @staticmethod
    def key(src: str, mt: str, ref: str = "") -> str:
        return hashlib.sha1(f"{src}\x00{mt}\x00{ref}".encode("utf-8")).hexdigest()

    def __call__(self, srcs: Sequence[str], mts: Sequence[str], refs: Optional[Sequence[str]] = None) -> List[float]:
        keys = [self.key(s, m, refs[i] if (self.reference_based and refs is not None) else "") for i, (s, m) in enumerate(zip(srcs, mts))]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        self.hits += len(keys) - len(todo)
        self.misses += len(todo)
        if todo:
            if self.reference_based:
                sc = self.fn([srcs[i] for i in todo], [mts[i] for i in todo], [refs[i] for i in todo])
            else:
                sc = self.fn([srcs[i] for i in todo], [mts[i] for i in todo])
            for i, v in zip(todo, sc):
                self.cache[keys[i]] = float(v)
                if self._fh:
                    self._fh.write(json.dumps([keys[i], float(v)]) + "\n")
            if self._fh:
                self._fh.flush()
        return [self.cache[k] for k in keys]

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


class CometPair:
    """AfriCOMET-STL-1.1 (with reference) and AfriCOMET-QE-STL-1.1 as two child-process scorers (python + pythonpath of an environment with
    transformers 4.x, D-33), created lazily and closed together."""

    def __init__(self, *, python: str, pythonpath: Optional[str], batch_size: int = 32, cache_dir: Optional[str] = None,
                 startup_timeout: float = 1800.0, request_timeout: float = 1800.0, ref_model: Optional[str] = None,
                 qe_model: Optional[str] = None, fake: bool = False, sequential: bool = False):
        from gemma_mt.rl.comet import QE_MODEL_ID, REF_MODEL_ID

        self.python, self.pythonpath, self.batch_size = python, pythonpath, batch_size
        self.cache_dir, self.startup_timeout, self.request_timeout = cache_dir, startup_timeout, request_timeout
        self.ref_model, self.qe_model, self.fake = ref_model or REF_MODEL_ID, qe_model or QE_MODEL_ID, fake
        self.sequential = sequential                  # at most ONE worker (model) alive: for GPUs that are nearly full
        self._workers: Dict[str, Any] = {}
        self._cached: Dict[str, CachedScorer] = {}
        self._released: Dict[str, Dict[str, int]] = {}       # counters of scorers that were released (stats() keeps adding them up)

    def _drop(self, kind: str) -> None:
        c, w = self._cached.pop(kind, None), self._workers.pop(kind, None)
        if c is not None:
            c.close()
        if w is not None:
            w.close()

    def _scorer(self, kind: str) -> CachedScorer:
        if self.sequential:
            self._drop("qe" if kind == "ref" else "ref")
        if kind not in self._cached:
            from gemma_mt.rl.comet import WorkerScorer

            ref = kind == "ref"
            w = WorkerScorer(python=self.python, model_id=self.ref_model if ref else self.qe_model, batch_size=self.batch_size,
                             fake=self.fake, pythonpath=self.pythonpath, reference_based=ref,
                             startup_timeout=self.startup_timeout, request_timeout=self.request_timeout)
            self._workers[kind] = w
            path = os.path.join(self.cache_dir, f"comet_{kind}.jsonl") if self.cache_dir else None
            if path:
                os.makedirs(self.cache_dir, exist_ok=True)
            self._cached[kind] = CachedScorer(w, path, reference_based=ref)
        return self._cached[kind]

    def reference(self, srcs, mts, refs):
        return self._scorer("ref")(srcs, mts, refs)

    def qe(self, srcs, mts):
        return self._scorer("qe")(srcs, mts)

    def stats(self) -> Dict[str, Any]:
        out = {k: dict(v) for k, v in self._released.items()}
        for k, c in self._cached.items():
            d = out.setdefault(k, {"cache_hits": 0, "scored": 0})
            d["cache_hits"] += c.hits
            d["scored"] += c.misses
        return out

    def release(self) -> None:
        """Close the worker processes (their GPU memory goes back to the card) but keep the scores: the next call starts a worker again and reads the
        on-disk cache. Used between two systems on a shared GPU, where an idle AfriCOMET worker would keep the next system's generation waiting."""
        for k, c in self._cached.items():
            d = self._released.setdefault(k, {"cache_hits": 0, "scored": 0})
            d["cache_hits"] += c.hits
            d["scored"] += c.misses
        self.close()

    def close(self) -> None:
        for c in self._cached.values():
            c.close()
        for w in self._workers.values():
            try:
                w.close()
            except Exception:
                pass
        self._workers.clear()
        self._cached.clear()


# ------------------------------------------------------------------------------------------------ GPU / watchdog
def gpu_free_gib() -> Optional[float]:
    """Smallest free memory (GiB) over the visible GPUs according to nvidia-smi; None when there is no nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=30).stdout
        vals = [float(x) / 1024.0 for x in out.split() if x.strip()]
        return min(vals) if vals else None
    except Exception:
        return None


def wait_for_gpu(min_free_gib: float, max_wait_s: float, poll_s: float = 15.0, probe: Callable[[], Optional[float]] = gpu_free_gib,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time) -> float:
    """Block until `min_free_gib` GiB are free on the GPU (another job may hold it), at most `max_wait_s` seconds. Returns the free GiB seen
    (nan without a GPU probe). Raises TimeoutError when the memory does not become free: the pipeline must not wait forever."""
    t_end = clock() + max_wait_s
    while True:
        free = probe()
        if free is None:
            return float("nan")
        if free >= min_free_gib:
            return free
        if clock() >= t_end:
            raise TimeoutError(f"only {free:.1f} GiB of GPU memory free after {max_wait_s:.0f}s, need {min_free_gib:.1f}")
        sleep(poll_s)


class Watchdog:
    """After `seconds` the `on_timeout` callback runs and the process is killed with os._exit(3): nothing in the evaluation may hang the
    machine that is about to start the RL run. Children (vLLM worker, COMET workers) see EOF on their pipes and exit."""

    def __init__(self, seconds: float, on_timeout: Callable[[], None], exit_fn: Callable[[int], None] = os._exit):
        self.seconds, self.on_timeout, self.exit_fn = seconds, on_timeout, exit_fn
        self._timer = threading.Timer(seconds, self._fire)
        self._timer.daemon = True

    def _fire(self) -> None:
        try:
            self.on_timeout()
        finally:
            self.exit_fn(3)

    def start(self) -> "Watchdog":
        self._timer.start()
        return self

    def cancel(self) -> None:
        self._timer.cancel()


def add_extra_libs(paths: Optional[str]) -> None:
    """Append directories to sys.path (AFTER the interpreter's own site-packages, so they only fill gaps such as sacrebleu)."""
    for p in (paths or "").split(os.pathsep):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)
