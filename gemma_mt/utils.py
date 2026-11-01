# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small shared helpers: config loading, seeds, RNG state, jsonl/wandb logging, distributed context, LR schedule."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import random
import time
from typing import Any, Dict, Optional

import numpy as np


# ------------------------------------------------------------------------------------------------ config
def load_env_file(path: str = ".env") -> None:
    """Load KEY=VALUE lines of a .env file into os.environ (existing variables win). Private identifiers (organisation, buckets, hosts, tokens) live there, never in the code."""
    import os
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _expand(obj):
    import os
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    return obj


def load_yaml(path: str) -> Dict[str, Any]:
    """YAML config with ${VAR} expanded from the environment (the repository .env is loaded first)."""
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError("PyYAML is required to read config files (pip install pyyaml)") from e
    load_env_file(os.environ.get("ENV_FILE", ".env"))
    with open(path, encoding="utf-8") as f:
        return _expand(yaml.safe_load(f) or {})


def dataclass_from_dict(cls, d: Dict[str, Any], strict: bool = True):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown and strict:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: v for k, v in d.items() if k in names})


def parse_overrides(pairs) -> Dict[str, Any]:
    """`key=value` strings (value parsed as JSON when possible) -> dict."""
    out: Dict[str, Any] = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"override {p!r} must look like key=value")
        k, v = p.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


# ------------------------------------------------------------------------------------------------ schedule
def lr_at(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Linear warmup to `peak_lr`, then cosine decay to `peak_lr * min_lr_ratio` at `total_steps`.
    `step` is the 0-based index of the optimizer step about to be taken."""
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return peak_lr
    t = min(1.0, max(0.0, (step - warmup_steps) / (total_steps - warmup_steps)))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t)))


# ------------------------------------------------------------------------------------------------ seeds / rng
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def get_rng_state() -> Dict[str, Any]:
    import torch

    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: Dict[str, Any]) -> None:
    import torch

    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    # a checkpoint loaded with map_location=cuda holds the RNG states on the GPU; torch wants CPU ByteTensors
    torch.set_rng_state(torch.as_tensor(st["torch"]).cpu().to(torch.uint8))
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.as_tensor(x).cpu().to(torch.uint8) for x in st["cuda"]])


# ------------------------------------------------------------------------------------------------ logging
class MetricsLogger:
    """Append-only jsonl log (+ optional wandb). Only the main rank should construct a real one."""

    def __init__(self, path: Optional[str], wandb_project: Optional[str] = None, run_name: Optional[str] = None,
                 config: Optional[dict] = None, enabled: bool = True):
        self.enabled = enabled
        self._f = None
        self._wandb = None
        if not enabled:
            return
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._f = open(path, "a", encoding="utf-8")
        if wandb_project:
            try:
                import wandb

                wandb.init(project=wandb_project, name=run_name, config=config or {})
                self._wandb = wandb
            except Exception as e:  # logging must never kill a run
                print(f"[gemma_mt] wandb disabled: {e}", flush=True)

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        rec = {"step": step, "time": round(time.time(), 1), **metrics}
        if self._f:
            self._f.write(json.dumps(rec, default=float) + "\n")
            self._f.flush()
        if self._wandb:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._f:
            self._f.close()
        if self._wandb:
            self._wandb.finish()


# ------------------------------------------------------------------------------------------------ distributed
class Dist:
    """torchrun-aware context. Single process when WORLD_SIZE is unset / 1."""

    def __init__(self, device: Optional[str] = None):
        import torch

        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if device is None:
            device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.world > 1:
            import torch.distributed as dist

            if not dist.is_initialized():
                dist.init_process_group(backend="nccl" if self.device.type == "cuda" else "gloo")
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def all_reduce_sum(self, x):
        import torch

        if self.world == 1:
            return x
        import torch.distributed as dist

        t = torch.as_tensor(x, dtype=torch.float64, device=self.device)
        dist.all_reduce(t)
        return t

    def barrier(self) -> None:
        if self.world > 1:
            import torch.distributed as dist

            dist.barrier()
