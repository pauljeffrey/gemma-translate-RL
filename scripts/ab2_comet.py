# SPDX-License-Identifier: AGPL-3.0-or-later
# AB2: eval-only copy of /workspace/jobs/e2/lib/gemma_mt/rl/comet.py (D-55: UnifiedMetric = SSA-COMET-MTL, max_batch_chars) so that the RL tree of this ablation (AB1 snapshot) stays untouched; imports renamed.
"""AfriCOMET scorers. Three interchangeable implementations of the same call:

    scorer(srcs: list[str], mts: list[str], refs: list[str] | None = None) -> list[float]

  CometScorer       in-process: loads the checkpoint with `unbabel-comet`, fp16 on GPU, batches by length. This is
                    what the RL stage uses (reference-free QE: masakhane/africomet-qe-stl-1.1, inputs (src, mt)).
  WorkerScorer      the same model behind a subprocess (gemma_mt.rl.comet_worker) for the case where unbabel-comet
                    must live in another virtualenv (it pins transformers < 5).
  FakeScorer        deterministic, model-free, for tests and --smoke.

  UnifiedMetric     (D-55) CometScorer also loads UnifiedMetric checkpoints (McGill-NLP/ssa-comet-mtl, run reference-free: src + mt only). Those encode ONE joint
                    sequence <s> mt </s></s> src </s> of <= 512 positions; a pair over the joint budget is cut into pieces that fit (rl/joint_budget.py) and the
                    pieces are averaged, so a long translation never silently truncates the source.

Segment limit: AfriCOMET's encoder (Davlan/afro-xlmr-large-76L, max 512 positions) tokenizes the source and the
translation SEPARATELY and truncates each at 510 tokens (comet/encoders/base.py: max_length = max_positions - 2).
`CometScorer.token_lengths` reports the lengths in those tokens so the reward can flag truncated segments.
Results are returned in INPUT order (the batching sorts by length internally and un-sorts).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Callable, List, Optional, Sequence

from ab2_joint_budget import JOINT_CONTENT_BUDGET

COMET_MAX_SEGMENT_TOKENS = 510   # encoder positions (512) minus <s> and </s>
SSA_MTL_MODEL_ID = "McGill-NLP/ssa-comet-mtl"
QE_MODEL_ID = "masakhane/africomet-qe-stl-1.1"
REF_MODEL_ID = "masakhane/africomet-stl-1.1"
ENCODER_ID = "Davlan/afro-xlmr-large-76L"


def batched_in_length_order(items: Sequence, key: Callable, run_batch: Callable, batch_size: int, max_cost: Optional[float] = None) -> list:
    """Apply `run_batch(list_of_items) -> list_of_scores` over items sorted by `key` and return scores in the
    ORIGINAL order. Factored out so the un-sorting can be unit-tested without a model.
    `max_cost` (D-55, the two-model worker): a batch also ends when len(batch) * key(longest item of the batch) would exceed it, so a batch of LONG chunks never has the
    memory of batch_size of them (None = fixed-size batches, the behaviour of the single-model path)."""
    order = sorted(range(len(items)), key=lambda i: key(items[i]))
    out = [None] * len(items)
    pos = 0
    while pos < len(order):
        end = min(pos + batch_size, len(order))
        if max_cost is not None:
            end = pos + 1
            while end < min(pos + batch_size, len(order)) and (end + 1 - pos) * key(items[order[end]]) <= max_cost:   # sorted ascending: the newest item is the longest
                end += 1
        idx = order[pos:end]
        pos = end
        scores = run_batch([items[i] for i in idx])
        if len(scores) != len(idx):
            raise RuntimeError(f"scorer returned {len(scores)} scores for {len(idx)} inputs")
        for i, sc in zip(idx, scores):
            out[i] = float(sc)
    return out


class CometScorer:
    def __init__(self, model_id: str = QE_MODEL_ID, batch_size: int = 32, device: Optional[str] = None,
                 half: bool = True, reference_based: bool = False, max_batch_chars: Optional[int] = None):
        import torch
        from comet import download_model, load_from_checkpoint

        path = model_id if os.path.isfile(model_id) else download_model(model_id)
        self.model = load_from_checkpoint(path)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if half and self.device.startswith("cuda"):
            self.model = self.model.half()
        self.model.to(self.device)
        self.batch_size = batch_size
        self.reference_based = reference_based
        self.model_id = model_id
        self.unified = type(self.model).__name__ == "UnifiedMetric"      # D-55: SSA-COMET-MTL; prepare_sample returns a TUPLE of inputs, the score comes from predict_step
        self.joint_budget = JOINT_CONTENT_BUDGET
        self.max_batch_chars = max_batch_chars                           # D-55: cap on len(batch) x (chars of the longest src + mt in it); None = fixed-size batches
        self.last_stats: dict = {}                                       # statistics of the last call (unified: pairs cut to fit the joint budget)

    def _tokenizer(self):
        return self.model.encoder.tokenizer

    def token_lengths(self, texts: Sequence[str]) -> List[int]:
        """Lengths in AfriCOMET-encoder tokens INCLUDING <s> and </s>, before truncation."""
        tok = self._tokenizer()
        return [len(x) for x in tok(list(texts), add_special_tokens=True, truncation=False)["input_ids"]]

    def content_lengths(self, texts: Sequence[str]) -> List[int]:
        """Content tokens WITHOUT <s> / </s> (what the joint budget counts)."""
        if not len(texts):
            return []
        tok = self._tokenizer()
        return [len(x) for x in tok(list(texts), add_special_tokens=False, truncation=False)["input_ids"]]

    def _call_unified(self, srcs, mts, refs) -> List[float]:
        """UnifiedMetric: src + mt only (one forward over the joint [mt, src] sequence), or with a reference the mean of the three views (the model card's predict).
        Reference-free pairs over the joint budget are cut into pieces (joint_budget.split_pair) and the piece scores averaged by source characters."""
        import torch
        from ab2_joint_budget import combine, piece_weights, plan_pieces

        def run_rows(rows):
            def run(batch):
                with torch.no_grad():
                    inputs = self.model.prepare_sample(batch, stage="predict")
                    inputs = tuple({k: v.to(self.device) for k, v in d.items()} for d in inputs)
                    return [float(x) for x in self.model.predict_step(inputs).scores.flatten().float().tolist()]

            return batched_in_length_order(rows, lambda r: len(r["src"]) + len(r["mt"]), run, self.batch_size, self.max_batch_chars)

        if self.reference_based:
            self.last_stats = {"joint_split": 0, "joint_pieces": len(srcs), "n": len(srcs)}
            return run_rows([{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs)])
        plans, n_split = plan_pieces(srcs, mts, self.content_lengths, self.joint_budget)
        flat = [{"src": s, "mt": m} for pieces in plans for s, m in pieces]
        flat_scores = run_rows(flat)
        out, k = [], 0
        for pieces in plans:
            sc = flat_scores[k: k + len(pieces)]
            k += len(pieces)
            out.append(sc[0] if len(pieces) == 1 else combine(sc, piece_weights(pieces)))
        self.last_stats = {"joint_split": n_split, "joint_pieces": len(flat), "n": len(srcs)}
        return out

    def __call__(self, srcs: Sequence[str], mts: Sequence[str], refs: Optional[Sequence[str]] = None) -> List[float]:
        import torch

        if len(srcs) != len(mts):
            raise ValueError("srcs and mts must be the same length")
        if self.reference_based and refs is None:
            raise ValueError("this scorer needs references")
        if self.unified:
            return self._call_unified(srcs, mts, refs)
        rows = [{"src": s, "mt": m, **({"ref": r} if self.reference_based else {})}
                for s, m, r in zip(srcs, mts, refs if refs is not None else [None] * len(srcs))]

        def run(batch):
            with torch.no_grad():
                inputs = self.model.prepare_sample(batch, stage="predict")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                return [float(x) for x in self.model(**inputs).score.flatten().float().tolist()]

        return batched_in_length_order(rows, lambda r: len(r["src"]) + len(r["mt"]), run, self.batch_size, self.max_batch_chars)


class WorkerScorer:
    """Same interface, model served by `python -m gemma_mt.rl.comet_worker` (JSON lines over stdin/stdout).

    `pythonpath` is prepended to the CHILD's PYTHONPATH only (D-33: a transformers 4.x directory next to the comet libraries).
    `reference_based=True` starts the worker with `--reference-based` (AfriCOMET-STL with references; the call then needs `refs`).
    `startup_timeout` / `request_timeout` (seconds, None = wait forever): a worker that does not answer in time is killed and the call
    raises RuntimeError (the evaluation pipeline must never hang).
    `extra_model_ids` (D-55): further checkpoints loaded into the SAME worker process (the QE ensemble: AfriCOMET-QE + SSA-COMET-MTL); `view(k)` is a scorer for model k
    (0 = `model_id`) that shares the process; the models run one after the other (one process, one CUDA context)."""

    def __init__(self, python: str = sys.executable, model_id: str = QE_MODEL_ID, batch_size: int = 32,
                 fake: bool = False, gpu: bool = True, cwd: Optional[str] = None, pythonpath: Optional[str] = None,
                 reference_based: bool = False, startup_timeout: Optional[float] = None, request_timeout: Optional[float] = None,
                 extra_model_ids: Sequence[str] = (), max_batch_chars: Optional[int] = None):
        import queue
        import threading

        env = {k: v for k, v in os.environ.items()}
        if not gpu:
            env["CUDA_VISIBLE_DEVICES"] = ""
        self.model_ids = [model_id, *extra_model_ids]
        cmd = [python, "-m", "gemma_mt.rl.comet_worker"] + [a for m in self.model_ids for a in ("--model", m)] + (["--fake"] if fake else []) + \
              (["--reference-based"] if reference_based else []) + \
              (["--max-batch-chars", str(int(max_batch_chars))] if max_batch_chars else [])
        root = cwd or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
        if pythonpath:    # extra entries for the CHILD only (unbabel-comet 2.2.7 needs transformers < 5: put a transformers 4.x directory here, D-33)
            env["PYTHONPATH"] = pythonpath + os.pathsep + env["PYTHONPATH"]
        self.batch_size = batch_size
        self.request_timeout = request_timeout
        self.last_stats: dict = {}
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env, cwd=root)
        self._lines: "queue.Queue" = queue.Queue()

        def _reader(stream, q):
            try:
                for line in stream:
                    q.put(line)
            finally:
                q.put(None)                               # EOF marker

        threading.Thread(target=_reader, args=(self.proc.stdout, self._lines), daemon=True).start()
        ready = self._readline(startup_timeout, "start-up")
        if not ready or not json.loads(ready).get("ready"):
            self.close()
            raise RuntimeError(f"COMET worker failed to start: {ready!r}")

    def _readline(self, timeout: Optional[float], what: str):
        import queue

        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            self.close(kill=True)
            raise RuntimeError(f"COMET worker did not answer within {timeout:.0f}s during {what}") from None
        if line is None:
            self._lines.put(None)
        return line

    def view(self, model: int, batch_size: Optional[int] = None) -> "WorkerView":
        """A scorer for model `model` of this worker (0 = the main model); shares the process, so it must be used from one thread at a time."""
        return WorkerView(self, model, batch_size or self.batch_size)

    def __call__(self, srcs, mts, refs=None, model: int = 0, batch_size: Optional[int] = None):
        rows = [{"src": s, "mt": m, "ref": (refs[i] if refs is not None else "")} for i, (s, m) in enumerate(zip(srcs, mts))]
        req = {"data": rows, "batch_size": batch_size or self.batch_size}
        if model:
            req["model"] = int(model)                      # absent = model 0: the protocol of the single-model worker is unchanged
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self._readline(self.request_timeout, "scoring")
        if not line:
            raise RuntimeError("COMET worker exited")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"COMET worker error: {reply['error']}")
        self.last_stats = reply.get("stats") or {}
        return [float(x) for x in reply["scores"]]

    def close(self, kill: bool = False) -> None:
        if self.proc.poll() is None:
            try:
                if kill:
                    raise RuntimeError("kill")
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()
                try:
                    self.proc.wait(timeout=10)
                except Exception:
                    pass


class WorkerView:
    """Scorer for one model of a multi-model WorkerScorer (same call signature as every scorer)."""

    def __init__(self, worker: WorkerScorer, model: int, batch_size: int):
        self.worker, self.model, self.batch_size = worker, model, batch_size
        self.last_stats: dict = {}

    def __call__(self, srcs, mts, refs=None):
        out = self.worker(srcs, mts, refs, model=self.model, batch_size=self.batch_size)
        self.last_stats = self.worker.last_stats
        return out


class FakeScorer:
    """Model-free QE stand-in: higher when the translation has a length similar to the source and uses a rich
    character set. Deterministic, in [0, 1]. Records every call for tests."""

    def __init__(self):
        self.calls: List[tuple] = []

    @staticmethod
    def score_one(src: str, mt: str) -> float:
        if not mt.strip():
            return 0.0
        ratio = min(len(mt), len(src)) / max(len(mt), len(src), 1)
        richness = min(1.0, len(set(mt.lower())) / 20.0)
        return 0.5 * ratio + 0.5 * richness

    def token_lengths(self, texts):
        return [len(t.split()) + 2 for t in texts]

    def __call__(self, srcs, mts, refs=None):
        self.calls.append((list(srcs), list(mts)))
        return [self.score_one(s, m) for s, m in zip(srcs, mts)]
