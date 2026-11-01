# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run-time controls of a long RL run, read from files in the output directory (D-56): nothing to restart, nothing to cancel.

    <out_dir>/WINDDOWN    "50" (or an empty file = 50): from the next step the learning rate decays LINEARLY to 0 over the next N steps, then the run finishes normally (the trainer's last
                          iteration saves `last/`, pushes latest / best, writes summary.json). The step at which the file was first seen is remembered in <out_dir>/WINDDOWN.state, so a
                          restart in the middle of the wind-down continues it (it does not start a new one). Deleting the file CANCELS a wind-down that has not finished.
    <out_dir>/LR_SCALE    one float (e.g. "0.5"): a multiplier on the learning rate, read again at every step (the KL guard's own scale, the warm-up and the wind-down multiply with it).
                          Missing / unreadable file = 1.0; clamped to [0, 4].

`RunControls.update(step)` is called once at the start of every training step with the number of completed steps; it returns the multiplier and the (possibly shortened) final step.

Also here (2026-10-28, clean rerun, agent T2; pure python + numpy, tests/test_kl_guard_plateau.py):
  KLGuard        the KL guard of the trainer as an object: cut (x factor) when the window median KL is above the threshold, OPTIONAL recovery (x recover_factor, never above 1.0)
                 when the median of a longer window stays below a fraction of the threshold, full state_dict()/load_state_dict() so a restart does not reset it.
  PlateauStop    saturation detector: 95 % bootstrap CI (over block means) of the OLS slope of the training reward over the last `window` steps contains 0, the finished rate is high,
                 and the window is free of guard events / learning-rate changes; the trainer then asks RunControls.request_winddown() so the run ends normally.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

DEFAULT_WINDDOWN = 50
LR_SCALE_MAX = 4.0


@dataclass
class ControlState:
    lr_mult: float                 # file scale x wind-down factor (the trainer multiplies it into the learning rate)
    end_step: int                  # the run ends when `step` reaches this
    file_scale: float
    winddown: bool
    winddown_start: Optional[int] = None
    winddown_steps: Optional[int] = None
    winddown_factor: float = 1.0


def parse_count(text: str, default: int = DEFAULT_WINDDOWN) -> int:
    """The first integer in the file (>= 1); anything else (empty, text, 0, negative) = the default."""
    for tok in text.replace(",", " ").split():
        try:
            v = int(float(tok))
        except ValueError:
            continue
        return v if v >= 1 else default
    return default


class RunControls:
    def __init__(self, out_dir: str, total_steps: int, log: Callable[[str], None] = print):
        self.dir, self.total, self.log = out_dir, int(total_steps), log
        self.p_wind = os.path.join(out_dir, "WINDDOWN")
        self.p_state = os.path.join(out_dir, "WINDDOWN.state")
        self.p_lr = os.path.join(out_dir, "LR_SCALE")
        self._start: Optional[int] = None
        self._n: Optional[int] = None
        self._bad_lr: Optional[str] = None
        self._last_scale = 1.0
        try:                                              # a restart in the middle of a wind-down continues it
            if os.path.exists(self.p_state) and os.path.exists(self.p_wind):
                st = json.load(open(self.p_state))
                self._start, self._n = int(st["start"]), int(st["steps"])
        except Exception:
            self._start = self._n = None

    # ------------------------------------------------------------------------------------------------
    def _read_scale(self) -> float:
        try:
            with open(self.p_lr) as f:
                txt = f.read().strip()
        except OSError:
            return 1.0
        try:
            v = float(txt.split()[0])
            if v != v or v in (float("inf"), float("-inf")):
                raise ValueError
        except (ValueError, IndexError):
            if txt != self._bad_lr:
                self._bad_lr = txt
                self.log(f"[gemma_mt] LR_SCALE file unreadable ({txt[:40]!r}): ignored (scale 1.0)")
            return 1.0
        return min(max(v, 0.0), LR_SCALE_MAX)

    def update(self, step: int) -> ControlState:
        scale = self._read_scale()
        if scale != self._last_scale:
            self.log(f"[gemma_mt] LR_SCALE file: learning-rate multiplier {self._last_scale:g} -> {scale:g} (step {step})")
            self._last_scale = scale
        wd = os.path.exists(self.p_wind)
        if wd and self._start is None:
            try:
                n = parse_count(open(self.p_wind).read())
            except OSError:
                n = DEFAULT_WINDDOWN
            self._start, self._n = int(step), n
            try:
                with open(self.p_state, "w") as f:
                    json.dump({"start": self._start, "steps": self._n}, f)
            except OSError:
                pass
            self.log(f"[gemma_mt] WINDDOWN file: learning rate decays linearly to 0 over {n} steps from step {step}; the run ends at step {min(self.total, step + n)}")
        elif not wd and self._start is not None:
            self.log(f"[gemma_mt] WINDDOWN file removed: wind-down cancelled (it had started at step {self._start})")
            self._start = self._n = None
            try:
                os.remove(self.p_state)
            except OSError:
                pass
        if self._start is None:
            return ControlState(scale, self.total, scale, False)
        k = step - self._start                            # 0 on the first wind-down step
        end = min(self.total, self._start + self._n)
        fac = max(0.0, 1.0 - k / float(self._n))
        return ControlState(scale * fac, end, scale, True, self._start, self._n, fac)

    def finish(self, step: int) -> Optional[dict]:
        """Called when the run ends. A wind-down that reached its end is archived (WINDDOWN -> WINDDOWN.done, state removed), so restarting the run in the same directory does not end it
        again at once. Returns the wind-down record for summary.json."""
        if self._start is None:
            return None
        rec = {"started_at_step": self._start, "steps": self._n, "finished_at_step": int(step), "completed": step >= min(self.total, self._start + self._n)}
        if rec["completed"]:
            for src, dst in ((self.p_wind, self.p_wind + ".done"), (self.p_state, None)):
                try:
                    if dst:
                        os.replace(src, dst)
                    else:
                        os.remove(src)
                except OSError:
                    pass
        return rec


    # ------------------------------------------------------------------------------------------------ requests made by the trainer itself
    def request_winddown(self, n: int, reason: str = "", evidence: Optional[dict] = None) -> bool:
        """Ask for a wind-down of `n` steps exactly as an operator would (write <out_dir>/WINDDOWN); it starts at the next `update()`. False (nothing written) when a wind-down is already
        requested or running. The reason / evidence go to <out_dir>/WINDDOWN.reason.json so the stop is auditable (the file survives a restart like WINDDOWN.state does)."""
        if os.path.exists(self.p_wind) or self._start is not None:
            return False
        n = max(int(n), 1)
        with open(self.p_wind, "w") as f:
            f.write(str(n))
        try:
            with open(os.path.join(self.dir, "WINDDOWN.reason.json"), "w") as f:
                json.dump({"n": n, "reason": reason, "evidence": evidence or {}}, f, indent=2, default=str)
        except OSError:
            pass
        self.log(f"[gemma_mt] wind-down of {n} steps requested by the trainer ({reason})")
        return True

    def winddown_active(self) -> bool:
        """A wind-down was requested (file present) or is running."""
        return os.path.exists(self.p_wind) or self._start is not None

    def restore_lr_scale(self, value: float) -> bool:
        """After a restart: when the LR_SCALE file is gone but the checkpoint recorded an operator multiplier != 1, write it again (the operator's cut must not be undone silently).
        An operator who wants 1.0 writes "1". Returns True when the file was written."""
        value = float(value)
        if os.path.exists(self.p_lr) or abs(value - 1.0) < 1e-12:
            return False
        with open(self.p_lr, "w") as f:
            f.write(f"{min(max(value, 0.0), LR_SCALE_MAX):g}\n")
        self._last_scale = 1.0
        return True


# ====================================================================================================================================================================
# KL guard
# ====================================================================================================================================================================
def _median_upper(xs: List[float]) -> float:
    """The element at index len//2 of the sorted list (what the trainer has always used)."""
    return sorted(xs)[len(xs) // 2]


class KLGuard:
    """KL guard with a persistent state.

    cut       `window` full, median(KL of the last `window` steps) > threshold, >= `cooldown` steps since the last cut, scale above the floor
              -> scale = max(scale * factor, min_scale)                                                        (this is the behaviour of 2c and final2, unchanged)
    recover   (only with recover=True) scale < max_scale, the last `recover_window` steps are ALL after the last guard event (cut or recovery) and at least `cooldown` steps after it,
              and median(KL of those steps) < recover_below * threshold -> scale = min(scale * recover_factor, max_scale)
    The guard never raises the learning rate above the planned one (max_scale = 1.0) and never touches the operator's LR_SCALE file.
    `step` is the number of COMPLETED steps including the one that produced `kl`; a new scale applies from the next step on.
    """

    def __init__(self, threshold: float, factor: float = 0.5, min_scale: float = 0.125, window: int = 5, cooldown: int = 10, recover: bool = False,
                 recover_factor: float = 1.25, recover_below: float = 0.5, recover_window: int = 20, max_scale: float = 1.0):
        self.threshold, self.factor, self.min_scale = float(threshold), float(factor), float(min_scale)
        self.window, self.cooldown = max(int(window), 1), int(cooldown)
        self.recover, self.recover_factor, self.recover_below = bool(recover), float(recover_factor), float(recover_below)
        self.recover_window, self.max_scale = max(int(recover_window), 1), float(max_scale)
        self.lr_scale = 1.0
        self.last_cut = -10 ** 9
        self.last_event = -10 ** 9
        self.kl_hist: List[float] = []
        self.kl_long: List[float] = []
        self.n_cuts = 0
        self.n_recoveries = 0
        self.last_median = float("nan")

    @property
    def enabled(self) -> bool:
        return self.threshold > 0

    def update(self, step: int, kl: float) -> Optional[Dict[str, Any]]:
        """Feed the KL of the step just finished. Returns an event dict ({"step", "event": "cut"|"recover", "kl_median", "threshold", "scale_old", "scale_new", "window"}) or None."""
        if not self.enabled:
            return None
        self.kl_hist = (self.kl_hist + [float(kl)])[-self.window:]
        if self.recover:
            self.kl_long = (self.kl_long + [float(kl)])[-self.recover_window:]
        med = _median_upper(self.kl_hist)       # median: a single-token outlier step (2c steps 607, 747) must not cut the learning rate
        self.last_median = med
        if (len(self.kl_hist) >= self.window and med > self.threshold and step - self.last_cut >= self.cooldown and self.lr_scale > self.min_scale + 1e-12):
            old = self.lr_scale
            self.lr_scale = max(old * self.factor, self.min_scale)
            self.last_cut = self.last_event = int(step)
            self.n_cuts += 1
            return {"step": int(step), "event": "cut", "kl_median": med, "threshold": self.threshold, "scale_old": old, "scale_new": self.lr_scale, "window": len(self.kl_hist)}
        if (self.recover and self.lr_scale < self.max_scale - 1e-12 and len(self.kl_long) >= self.recover_window
                and step - self.last_event >= max(self.cooldown, self.recover_window)):
            med_l = _median_upper(self.kl_long)
            self.last_median = med_l
            if med_l < self.recover_below * self.threshold:
                old = self.lr_scale
                self.lr_scale = min(old * self.recover_factor, self.max_scale)
                if self.max_scale - self.lr_scale < 1e-9:
                    self.lr_scale = self.max_scale
                self.last_event = int(step)
                self.n_recoveries += 1
                return {"step": int(step), "event": "recover", "kl_median": med_l, "threshold": self.threshold, "scale_old": old, "scale_new": self.lr_scale, "window": len(self.kl_long)}
        return None

    # ---------------------------------------------------------------------------------------------- persistence
    def state_dict(self) -> Dict[str, Any]:
        return {"lr_scale": self.lr_scale, "last_cut": self.last_cut, "last_event": self.last_event, "kl_hist": list(self.kl_hist), "kl_long": list(self.kl_long),
                "n_cuts": self.n_cuts, "n_recoveries": self.n_recoveries}

    def load_state_dict(self, d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Restore from a checkpoint's `state["kl_guard"]` (None / {} = a checkpoint written before the guard was persisted: nothing changes, returns None). The scale is clamped into the
        CURRENT [min_scale, max_scale], the histories into the current windows (so a changed config is respected). Returns the restored values for logging."""
        if not d:
            return None
        self.lr_scale = min(max(float(d.get("lr_scale", 1.0)), self.min_scale), self.max_scale)
        self.last_cut = int(d.get("last_cut", -10 ** 9))
        self.last_event = int(d.get("last_event", self.last_cut))
        self.kl_hist = [float(x) for x in d.get("kl_hist", [])][-self.window:]
        self.kl_long = [float(x) for x in d.get("kl_long", [])][-self.recover_window:]
        self.n_cuts, self.n_recoveries = int(d.get("n_cuts", 0)), int(d.get("n_recoveries", 0))
        return {"lr_scale": self.lr_scale, "last_cut": self.last_cut, "last_event": self.last_event, "kl_hist_len": len(self.kl_hist), "n_cuts": self.n_cuts, "n_recoveries": self.n_recoveries}


# ====================================================================================================================================================================
# Plateau / saturation stop
# ====================================================================================================================================================================
def plateau_slope_test(values: List[float], block: int = 20, n_boot: int = 2000, seed: int = 0, conf: float = 0.95, min_blocks: int = 4) -> Dict[str, Any]:
    """The M1 plateau test (reports/lean_schedule_<S3_PREFIX>.md B.3): the last len(values) // block * block values are cut into non-overlapping blocks of `block` steps (aligned
    to the END, the oldest leftover values are dropped), the OLS slope of the block means against the block index (per step = slope / block) is bootstrapped over the BLOCKS
    (`n_boot` resamples with replacement, resamples with a single distinct block have no slope and are skipped); `contains_zero` = the `conf` percentile interval contains 0.
    Returns {"slope", "ci_low", "ci_high" (all per STEP), "contains_zero", "n_blocks", "block_means", "n_valid_boot"}; fewer than `min_blocks` blocks -> contains_zero False and "insufficient": True
    (a test that cannot be run never stops a run)."""
    import numpy as np

    v = [float(x) for x in values]
    nb = len(v) // max(int(block), 1)
    if nb < min_blocks:
        return {"slope": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "contains_zero": False, "n_blocks": nb, "block_means": [], "n_valid_boot": 0, "insufficient": True}
    use = np.asarray(v[len(v) - nb * block:], dtype=np.float64).reshape(nb, block)
    means = use.mean(axis=1)
    x = np.arange(nb, dtype=np.float64)
    slope = float(((x - x.mean()) * (means - means.mean())).sum() / ((x - x.mean()) ** 2).sum()) / block
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, nb, size=(int(n_boot), nb))
    X, Y = idx.astype(np.float64), means[idx]
    xm, ym = X.mean(axis=1, keepdims=True), Y.mean(axis=1, keepdims=True)
    sxx = ((X - xm) ** 2).sum(axis=1)
    sxy = ((X - xm) * (Y - ym)).sum(axis=1)
    ok = sxx > 0
    slopes = sxy[ok] / sxx[ok] / block
    alpha = (1.0 - conf) / 2.0
    lo, hi = (float(q) for q in np.quantile(slopes, [alpha, 1.0 - alpha])) if slopes.size else (float("nan"), float("nan"))
    eps = 1e-12
    return {"slope": slope, "ci_low": lo, "ci_high": hi, "contains_zero": bool(slopes.size and lo <= eps and hi >= -eps), "n_blocks": nb,
            "block_means": [float(m) for m in means], "n_valid_boot": int(slopes.size), "insufficient": False}


class PlateauStop:
    """Saturation detector (config `plateau_stop`). `observe()` after every step, `check()` at evaluation boundaries. It can fire only when ALL of these hold (the first failing one is
    returned as `blocked`):
      last_stage      the current curriculum stage is the last one of the run (stopping an earlier stage would skip the stages after it; the stop ends the RUN)
      min_steps       steps done in the stage >= min_steps
      history         the last `window` steps are all observed, consecutive, in this stage
      guard_event     no KL-guard cut / recovery in the window (an event at step e changes the learning rate from step e+1, so e >= first_step - 1 counts)
      lr_changed      the learning-rate multiplier (guard scale x operator file scale) is not constant over the window
      lr_low          that multiplier is below `min_lr_scale` (a flat reward at 12 % of the planned lr is not saturation: the final2 confound)
      finished        the mean TRAINING finished rate of the window >= min_finished, and the greedy evaluation finished rate of the stage's bin (if given) >= min_finished
      rising          the bootstrap CI of the slope excludes 0 (the reward is still moving)
      ci_too_wide     (only with ci_high_max > 0) the upper end of the CI, per 100 steps, is above ci_high_max: the window is too noisy to say that the gain is small. "CI contains 0" alone is also true
                      when the data cannot tell a slope of +0.05 per 100 steps from 0 (replay on the 2c b256 bin: it fires at step 125 with CI [-0.057, +0.093]); this bound turns "no evidence of a trend"
                      into "evidence that any trend is smaller than ci_high_max"
    """

    def __init__(self, window: int = 100, block: int = 20, min_finished: float = 0.99, min_lr_scale: float = 0.5, n_boot: int = 2000, seed: int = 0, ci_high_max: float = 0.0):
        self.window, self.block, self.min_finished, self.min_lr_scale = int(window), int(block), float(min_finished), float(min_lr_scale)
        self.ci_high_max = float(ci_high_max)
        self.n_boot, self.seed = int(n_boot), int(seed)
        self.hist: List[List[Any]] = []             # [step, reward, finished, lr_mult, stage]
        self.guard_events: List[int] = []
        self.triggered: Optional[Dict[str, Any]] = None
        self.last_check: Optional[Dict[str, Any]] = None

    def observe(self, step: int, reward: float, finished: float, lr_mult: float, stage: str) -> None:
        self.hist = (self.hist + [[int(step), float(reward), float(finished), float(lr_mult), str(stage)]])[-self.window:]

    def note_guard_event(self, step: int) -> None:
        self.guard_events = (self.guard_events + [int(step)])[-64:]

    def check(self, step: int, stage: str, stage_start: int, min_steps: int, is_last_stage: bool, eval_finished: Optional[float] = None) -> Dict[str, Any]:
        """Decision at the boundary after `step` completed steps. Returns {"fire": bool, "blocked": str | None, "steps_in_stage", ...evidence}."""
        out: Dict[str, Any] = {"fire": False, "blocked": None, "step": int(step), "stage": stage, "steps_in_stage": int(step) - int(stage_start), "min_steps": int(min_steps), "window": self.window}

        def blocked(why: str) -> Dict[str, Any]:
            out["blocked"] = why
            self.last_check = out
            return out

        if self.triggered is not None:
            return blocked("already_triggered")
        if not is_last_stage:
            return blocked("not_last_stage")
        if out["steps_in_stage"] < int(min_steps):
            return blocked("min_steps")
        h = self.hist
        if (len(h) < self.window or h[-1][0] != int(step) or h[-1][0] - h[0][0] != self.window - 1 or any(r[4] != stage for r in h)
                or any(b[0] - a[0] != 1 for a, b in zip(h, h[1:]))):
            return blocked("history")
        first = h[0][0]
        if any(e >= first - 1 for e in self.guard_events):
            return blocked("guard_event")
        lrs = [r[3] for r in h]
        if max(lrs) - min(lrs) > 1e-9 * max(max(lrs), 1e-12):
            return blocked("lr_changed")
        out["lr_mult"] = lrs[-1]
        if lrs[-1] < self.min_lr_scale - 1e-12:
            return blocked("lr_low")
        fin = sum(r[2] for r in h) / len(h)
        out["finished_train_window"], out["finished_eval_greedy"] = fin, eval_finished
        if fin < self.min_finished or (eval_finished is not None and eval_finished < self.min_finished):
            return blocked("finished")
        t = plateau_slope_test([r[1] for r in h], self.block, self.n_boot, self.seed + int(step))
        out.update({"slope_per_100_steps": 100.0 * t["slope"], "ci_low_per_100_steps": 100.0 * t["ci_low"], "ci_high_per_100_steps": 100.0 * t["ci_high"], "n_blocks": t["n_blocks"],
                    "block_means": t["block_means"], "reward_window_mean": sum(r[1] for r in h) / len(h)})
        if t.get("insufficient") or not t["contains_zero"]:
            return blocked("rising")
        if self.ci_high_max > 0 and out["ci_high_per_100_steps"] > self.ci_high_max:
            return blocked("ci_too_wide")
        out["fire"] = True
        self.last_check = out
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {"hist": [list(r) for r in self.hist], "guard_events": list(self.guard_events), "triggered": self.triggered}

    def load_state_dict(self, d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not d:
            return None
        self.hist = [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), str(r[4])] for r in d.get("hist", [])][-self.window:]
        self.guard_events = [int(e) for e in d.get("guard_events", [])]
        self.triggered = d.get("triggered")
        return {"hist_len": len(self.hist), "guard_events": len(self.guard_events), "triggered": self.triggered is not None}


# ====================================================================================================================================================================
# Patience-based early stop (2026-10-29, Amendment 4) and surgical extension of the cap
# ====================================================================================================================================================================
class PatienceStop:
    """Stop a stage when nothing has improved for `patience` steps (config `early_stop`).

    Three smoothed signals (mean of the last `block` steps of the current stage) are tracked against their best value so far:
      reward          higher is better, an improvement is a new best by more than eps_reward
      finished rate   training finished rate, higher is better, eps_finished
      greedy loop     loop rate of the greedy group member, LOWER is better, eps_loop (only with greedy_in_group)
    `last_improve_step` = the latest step at which ANY of them set a new best by more than its epsilon (the first smoothed value counts as an improvement). The rule fires when
    steps_in_stage >= min_steps AND step - last_improve_step >= patience AND the last `patience` steps are all observed, consecutive and in this stage AND contain no KL-guard event
    (an event at e changes the lr from e+1: e >= first_step - 1 counts) AND the lr multiplier is constant and >= min_lr_scale AND the mean training finished rate (and the mean greedy finished
    rate when observed) of that window is >= min_finished AND the stage is the last of the run. The best values are compared against, not the previous block, so a slow creep below eps per
    block does not keep resetting the clock, and a single noisy block cannot set a new best by less than eps.
    `steps_since_improve()` is also what the extension rule reads ("still improving at the cap")."""

    def __init__(self, patience: int = 60, block: int = 10, eps_reward: float = 0.02, eps_finished: float = 0.002, eps_loop: float = 0.005, min_finished: float = 0.99,
                 min_lr_scale: float = 0.5):
        self.patience, self.block = int(patience), int(block)
        self.eps_reward, self.eps_finished, self.eps_loop = float(eps_reward), float(eps_finished), float(eps_loop)
        self.min_finished, self.min_lr_scale = float(min_finished), float(min_lr_scale)
        self.stage: Optional[str] = None
        self.hist: List[List[Any]] = []          # [step, reward, finished, greedy_finished|None, greedy_loop|None, lr_mult]
        self.guard_events: List[int] = []
        self.best: Dict[str, Optional[float]] = {"reward": None, "finished": None, "loop": None}
        self.best_step: Dict[str, int] = {"reward": -1, "finished": -1, "loop": -1}
        self.last_improve_step: Optional[int] = None
        self.last_improve_by: str = ""
        self.triggered: Optional[Dict[str, Any]] = None
        self.smoothed: Dict[str, Optional[float]] = {}

    # ------------------------------------------------------------------------------------------------ feeding
    def observe(self, step: int, reward: float, finished: float, lr_mult: float, stage: str, greedy_loop: Optional[float] = None, greedy_finished: Optional[float] = None) -> None:
        if stage != self.stage:                  # a new stage inside one run: the clocks start again
            self.stage, self.hist, self.last_improve_step, self.last_improve_by = stage, [], None, ""
            self.best, self.best_step = {"reward": None, "finished": None, "loop": None}, {"reward": -1, "finished": -1, "loop": -1}
        self.hist = (self.hist + [[int(step), float(reward), float(finished), None if greedy_finished is None else float(greedy_finished),
                                   None if greedy_loop is None else float(greedy_loop), float(lr_mult)]])[-max(self.patience, self.block):]
        h = self.hist[-self.block:]
        if len(h) < self.block or h[-1][0] - h[0][0] != self.block - 1:
            return
        sm = {"reward": sum(r[1] for r in h) / len(h), "finished": sum(r[2] for r in h) / len(h),
              "loop": (sum(r[4] for r in h) / len(h)) if all(r[4] is not None for r in h) else None}
        better = {"reward": lambda new, old: new > old + self.eps_reward, "finished": lambda new, old: new > old + self.eps_finished, "loop": lambda new, old: new < old - self.eps_loop}
        for k, v in sm.items():
            if v is None:
                continue
            if self.best[k] is None or better[k](v, self.best[k]):
                self.best[k], self.best_step[k] = v, int(step)
                self.last_improve_step, self.last_improve_by = int(step), k
        self.smoothed = sm

    def note_guard_event(self, step: int) -> None:
        self.guard_events = (self.guard_events + [int(step)])[-64:]

    def steps_since_improve(self, step: int) -> Optional[int]:
        return None if self.last_improve_step is None else int(step) - self.last_improve_step

    # ------------------------------------------------------------------------------------------------ decision
    def check(self, step: int, stage: str, stage_start: int, min_steps: int, is_last_stage: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {"fire": False, "blocked": None, "step": int(step), "stage": stage, "steps_in_stage": int(step) - int(stage_start), "min_steps": int(min_steps), "patience": self.patience,
                               "steps_since_improve": self.steps_since_improve(step), "last_improve_by": self.last_improve_by, "best": dict(self.best), "best_step": dict(self.best_step)}

        def blocked(why: str) -> Dict[str, Any]:
            out["blocked"] = why
            return out

        if self.triggered is not None:
            return blocked("already_triggered")
        if not is_last_stage:
            return blocked("not_last_stage")
        if out["steps_in_stage"] < int(min_steps):
            return blocked("min_steps")
        if self.last_improve_step is None or self.stage != stage:
            return blocked("no_signal_yet")
        if out["steps_since_improve"] < self.patience:
            return blocked("improving")
        h = self.hist[-self.patience:]
        if len(h) < self.patience or h[-1][0] != int(step) or h[-1][0] - h[0][0] != self.patience - 1:
            return blocked("history")
        if any(e >= h[0][0] - 1 for e in self.guard_events):
            return blocked("guard_event")
        lrs = [r[5] for r in h]
        if max(lrs) - min(lrs) > 1e-9 * max(max(lrs), 1e-12):
            return blocked("lr_changed")
        out["lr_mult"] = lrs[-1]
        if lrs[-1] < self.min_lr_scale - 1e-12:
            return blocked("lr_low")
        fin = sum(r[2] for r in h) / len(h)
        gf = [r[3] for r in h if r[3] is not None]
        out["finished_train_window"], out["finished_greedy_window"] = fin, (sum(gf) / len(gf) if gf else None)
        if fin < self.min_finished or (gf and sum(gf) / len(gf) < self.min_finished):
            return blocked("finished")
        out["fire"] = True
        return out

    # ------------------------------------------------------------------------------------------------ persistence
    def state_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "hist": [list(r) for r in self.hist], "guard_events": list(self.guard_events), "best": dict(self.best), "best_step": dict(self.best_step),
                "last_improve_step": self.last_improve_step, "last_improve_by": self.last_improve_by, "triggered": self.triggered}

    def load_state_dict(self, d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not d:
            return None
        self.stage = d.get("stage")
        self.hist = [list(r) for r in d.get("hist", [])][-max(self.patience, self.block):]
        self.guard_events = [int(e) for e in d.get("guard_events", [])]
        self.best = {k: d.get("best", {}).get(k) for k in ("reward", "finished", "loop")}
        self.best_step = {k: int(d.get("best_step", {}).get(k, -1)) for k in ("reward", "finished", "loop")}
        lis = d.get("last_improve_step")
        self.last_improve_step, self.last_improve_by = (None if lis is None else int(lis)), str(d.get("last_improve_by", ""))
        self.triggered = d.get("triggered")
        return {"hist_len": len(self.hist), "last_improve_step": self.last_improve_step, "triggered": self.triggered is not None}


def extension_decision(step: int, base_cap: int, cap_now: int, extend_steps: int, max_frac: float, since_improve: Optional[int], recent: int) -> Dict[str, Any]:
    """`extend_if_improving`: at the decision point of a run that has not been stopped, extend the cap by `extend_steps` when the improvement criterion fired within the last `recent` steps,
    in repeated blocks until the total extension reaches `max_frac` x the BASE cap (the last block is shortened to fit). Returns {"extend": bool, "reason", "extend_steps", "new_cap", ...}."""
    used = int(cap_now) - int(base_cap)
    room = int(base_cap * max_frac + 1e-9) - used
    d: Dict[str, Any] = {"step": int(step), "base_cap": int(base_cap), "cap": int(cap_now), "extension_used": used, "extension_room": max(room, 0), "steps_since_improve": since_improve,
                         "recent": int(recent), "extend": False, "extend_steps": 0, "new_cap": int(cap_now)}
    if room <= 0:
        d["reason"] = "max_extension_reached"
    elif since_improve is None:
        d["reason"] = "no_signal"
    elif since_improve >= recent:
        d["reason"] = "not_improving"
    else:
        n = min(int(extend_steps), room)
        d.update(extend=True, extend_steps=n, new_cap=int(cap_now) + n, reason="improving")
    return d
