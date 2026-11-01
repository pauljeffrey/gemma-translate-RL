#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helper commands of the clean-rerun RL launcher (agent C1, 2026-10-28). CPU only; nothing here trains.

    validate   load every yaml of configs/clean_rerun[_specialist] with the REAL config dataclasses (RLConfig strict, RewardConfig strict, validate_config) and replay the prompt
               schedule of the whole stage on a synthetic pool; also checks the plan invariants (greedy on, guard thresholds, caps) and that the yaml equals what the generator writes
    manifest   pin the launch manifest (rule file / amendment sha256, protected index version + manifest sha256, config hashes, trainer code hashes, git commit, GPU type, start model sha)
    gate       read <run>/metrics.jsonl and decide whether a stage (or the 49-step preflight) is healthy enough to continue
    winddown   background loop: writes <run>/WINDDOWN (N) when the step counter reaches steps - N (the planned wind-down of the final run)
    handoff    print the model directory that the next stage starts from (final or best)
    disclose   replay the prompt schedule of a finished stage and count the prompts drawn per source_dataset / domain (LibreTexts disclosure), and count archive rows whose source is a LibreTexts window

On the laptop (no torch / transformers): add --stub-heavy. On the rental: run with the trainer's interpreter.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
TREE_DEFAULT = os.path.abspath(os.path.join(HERE, ".."))                 # .../gemma_mt (contains the gemma_mt package)
STAGE_FILES = {"preflight": "rl_00_preflight.yaml", "A": "rl_A_b64.yaml", "B1": "rl_B1_b256.yaml", "B2": "rl_B2_b512.yaml", "B3": "rl_B3_b1024.yaml", "mixed": "rl_C_mixed.yaml", "final": "rl_F_final.yaml"}
PIN = {   # frozen before any training (reports/preregistration_clean_rerun_2026-10-28.md)
    "rule": ("7b86fc11de28cf952bbc8f7a5f634b5fceda6adbeb76996548360c8ed923c0f5", "DECISION_RULE.txt"),
    "amendment_1": ("a5885cf2b6218411af3807b763b5b8aeb7003e8c9c759cf3425ae05cf533c4c2", "AMENDMENT_1.txt"),
    "amendment_2": ("955941c5f6699511cc76a73cf77e2b3ca715c7cd4521de0635a4556cd3a92caa", "AMENDMENT_2.txt"),
    "weak_rule": ("4472d7115aaf21497dbd32bf7531b3bbd8b67e8b4243ffbccc7850d6c54c6ab7", "weak_selection_2026-10-26/DECISION_RULE.txt"),
}
INDEX_VERSION = "protected-2026-10-28.3"
INDEX_MANIFEST_SHA = "22172f30b1c31d585e1f000411e0267b54f701ae173b30beab5cb84cab1f1b7e"   # reports/data/sft_clean_2026-10-28/protected_manifest_v3.json (laptop copy)
INDEX_BASE_SHA_PREFIX = "f094fd2b24d6a0e4"                                                   # the (unchanged) base index file, D2 report section 1
HSEL_SHA = "036295b9d2c5d21a04eb2da93be320137f8d2244b8de7f2b1d8b9835894df37d"
HCAL_SHA = "e7c0605ddc36ee3e886a6e447c678e9291e86b3825fba6969f71057f9c6b0e93"


# ------------------------------------------------------------------------------------------------ helpers
def sha256_file(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def install_stubs() -> None:
    """Laptop mode: replace torch / transformers / ... by inert modules so that `gemma_mt.rl.train_rl` imports (config classes and the pure-python schedule only)."""
    import importlib.abc
    import importlib.machinery
    from unittest.mock import MagicMock

    stub = ("torch", "transformers", "vllm", "huggingface_hub", "sacrebleu", "comet", "fasttext", "safetensors", "accelerate", "datasets", "tqdm", "scipy", "sklearn")

    class M(MagicMock):
        __path__: list = []

    class F(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in stub:
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
            return None

        def create_module(self, spec):
            m = M()
            m.__name__, m.__spec__, m.__path__ = spec.name, spec, []
            return m

        def exec_module(self, m):
            pass

    sys.meta_path.insert(0, F())


def setup_imports(tree: str, stub: bool) -> None:
    if stub:
        install_stubs()
    sys.path.insert(0, tree)


# ------------------------------------------------------------------------------------------------ validate
def _expect(errors: List[str], cond: bool, msg: str) -> None:
    if not cond:
        errors.append(msg)


def cmd_validate(a) -> int:
    setup_imports(a.tree, a.stub_heavy)
    import numpy as np
    from gemma_mt.languages import SUPPORTED_RL
    from gemma_mt.rl.data import PromptSchedule
    from gemma_mt.rl.reward import RewardConfig
    from gemma_mt.rl.train_rl import RLConfig, validate_config
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    sys.path.insert(0, HERE)
    import clean_rerun_make_configs as gen

    spec = a.arm == "specialist"
    cfg_dir = os.path.abspath(a.configs or os.path.join(a.tree, "configs", "clean_rerun_specialist" if spec else "clean_rerun"))
    P = gen.paths(a.root)
    langs_all = sorted(SUPPORTED_RL)
    langs = [l for l in langs_all if (l in gen.SPECIALIST_LANGS)] if spec else langs_all
    bad_lang = [l for l in gen.WEAK_LIST + gen.SPECIALIST_LANGS if l not in langs_all]
    errors: List[str] = []
    _expect(errors, not bad_lang, f"weak/specialist languages that are not RL languages: {bad_lang}")
    rows = []
    for stage, fname in STAGE_FILES.items():
        path = os.path.join(cfg_dir, fname)
        errs: List[str] = []
        try:
            d = load_yaml(path)
            cfg = dataclass_from_dict(RLConfig, d)                       # strict: an unknown key raises
            validate_config(cfg)
            rc = dataclass_from_dict(RewardConfig, cfg.reward)           # strict
        except Exception as e:                                           # noqa: BLE001
            errors.append(f"{fname}: {type(e).__name__}: {e}")
            continue
        S = gen.STAGES[stage]
        fin, pre = stage == "final", stage == "preflight"
        # ---- plan invariants
        _expect(errs, cfg.greedy_in_group and cfg.greedy_stats == "sampled" and cfg.greedy_adv_scale == 0.3 and cfg.greedy_adv_cap == 3.0, "greedy scoring is not in the D-67 form")
        _expect(errs, cfg.group_size == 16, "group_size != 16")
        _expect(errs, abs(cfg.kl_guard_threshold - (0.25 if fin else 0.15)) < 1e-12, f"kl_guard_threshold {cfg.kl_guard_threshold}")
        _expect(errs, cfg.kl_guard_recover == fin and cfg.kl_guard_recover_factor == 1.25 and cfg.kl_guard_recover_below == 0.5 and cfg.kl_guard_recover_window == 20, "kl_guard_recover fields")
        _expect(errs, cfg.kl_estimator == "k3c" and cfg.advantage_norm == "mean_std" and cfg.grad_clip == 1.0, "k3c / mean_std / grad_clip")
        _expect(errs, cfg.max_src_tokens == 1024, "max_src_tokens must be 1024 (source-side cap, Amendment 2)")
        _expect(errs, "b2048" not in cfg.prompts_per_step_by_bin and "b2048" not in cfg.max_new_tokens_by_bin and "b2048" not in cfg.curriculum_steps, "b2048 must not appear")
        _expect(errs, cfg.reward.get("consensus") is True and cfg.reward.get("qe_ensemble") is True and cfg.reward.get("longdoc_v2") is True and cfg.reward.get("anchors_enabled") is True, "a reward component is off")
        _expect(errs, bool(cfg.reward.get("nllb_signal")) == gen.nllb_on(stage), f"nllb_signal must be on exactly in {gen.NLLB_STAGES} (+ the preflight when A has it)")
        if gen.nllb_on(stage):
            _expect(errs, list(cfg.reward.get("nllb_languages") or []) == gen.load_nllb_languages(spec)["languages"], "reward.nllb_languages differs from nllb_languages.json")
            if gen.load_nllb_languages(spec)["provisional"] and stage == "A":
                print("NOTE: the NLLB language list in nllb_languages.json is PROVISIONAL (placeholder until the H_sel language rule has run after A0)")
        _expect(errs, cfg.adaptive_lang_weights is False and cfg.lang_weights_file is True, "adaptive weights must stay off, LANG_WEIGHTS file on")
        _expect(errs, cfg.length_bins and cfg.run_controls and cfg.eval_min_per_bin >= 16, "length_bins / run_controls / eval_min_per_bin")
        _expect(errs, cfg.steps == S["steps"] or spec or a.allow_other_steps, f"steps {cfg.steps} != plan {S['steps']}")
        _expect(errs, cfg.keep_single_checkpoint is False, "keep_single_checkpoint must be false (the hand-off needs final/)")
        if not spec:
            _expect(errs, set(cfg.language_weights) == set(gen.WEIGHTED_LANGS) and all(v == 2.5 for v in cfg.language_weights.values()), "language_weights != the 25-language dev-defined weak list + the 4 owner-priority languages (efi tiv bin ish) at 2.5")
        if stage in ("A", "B1", "B2", "B3", "preflight"):
            _expect(errs, cfg.curriculum_steps == {S["bin"]: cfg.steps}, f"curriculum {cfg.curriculum_steps}")
        else:
            _expect(errs, cfg.curriculum_steps == {"b64": 0}, f"curriculum {cfg.curriculum_steps} (mixed from step 0 needs {{b64: 0}})")
        if fin:
            _expect(errs, cfg.domain_weights == {"science": 0.7, "general": 0.3}, "domain_weights")
            _expect(errs, cfg.language_floor_steps == 20 and cfg.language_floor_prompts == 2, "language floor 2 prompts / 20 steps")
            _expect(errs, abs(sum(cfg.domain_bin_weights["science"].values()) - 1.0) < 1e-9 and abs(cfg.domain_bin_weights["science"]["b1024"] - 0.35) < 1e-12, "science bin weights")
            _expect(errs, cfg.learning_rate == 1.0e-6 and cfg.warmup_steps == 20 and cfg.kl_coef == 0.1, "final run lr / warm-up / kl_coef")
        else:
            _expect(errs, cfg.learning_rate == 2.0e-6 and cfg.warmup_steps == 10 and cfg.kl_coef == 0.05, "stage lr / warm-up / kl_coef")
        for b, n in cfg.prompts_per_step_by_bin.items():
            _expect(errs, n <= gen.BIN_PPS[b], f"prompts_per_step_by_bin[{b}]={n} above the 2c maximum {gen.BIN_PPS[b]}")
        _expect(errs, cfg.kl_guard_restore == (stage in ("preflight", "A")) and cfg.greedy_share_warn == 0.35 and cfg.plateau_stop_ci_high_max == 0.06, "kl_guard_restore / greedy_share_warn / plateau_stop_ci_high_max")
        if cfg.plateau_stop:
            _expect(errs, cfg.steps > cfg.plateau_stop_min_steps, "plateau_stop on but inert (steps <= min_steps)")
        # ---- replay the schedule of the whole stage on a synthetic pool (every language x bin x domain, 200 rows per cell)
        try:
            iso, lb, dm = [], [], []
            doms = ["science", "general"] if fin else [""]
            for l in langs:
                for b in ("b64", "b256", "b512", "b1024"):
                    for dname in doms:
                        iso += [l] * 200
                        lb += [b] * 200
                        dm += [dname] * 200
            sch = PromptSchedule(np.asarray(iso), cfg.language_alpha, cfg.seed, length_bin=lb, curriculum=cfg.curriculum_steps or None, language_weights=cfg.language_weights or None,
                                 domain=dm if fin else None, domain_weights=cfg.domain_weights or None, domain_bin_weights=cfg.domain_bin_weights or None,
                                 floor_steps=cfg.language_floor_steps, floor_prompts=cfg.language_floor_prompts)
            n_prompts = 0
            bins = collections.Counter()
            per_lang = collections.Counter()
            per_dom = collections.Counter()
            win = collections.defaultdict(collections.Counter)         # 20-step block -> language -> prompts (floor check)
            for s in range(cfg.steps):
                cur = sch.bin_of_step(s)
                n = cfg.prompts_per_step_by_bin.get(cur, cfg.prompts_per_step)
                idx = sch.indices(s, n)
                _expect(errs, len(idx) == n, f"step {s}: {len(idx)} prompts, expected {n}")
                n_prompts += len(idx)
                bins[cur] += 1
                for k in idx:
                    per_lang[iso[k]] += 1
                    per_dom[dm[k] or "general"] += 1
                    win[s // 20][iso[k]] += 1
            if fin:
                full = [blk for blk in win if (blk + 1) * 20 <= cfg.steps]
                worst = min((win[blk][l] for blk in full for l in langs), default=None)
                _expect(errs, worst is None or worst >= 2, f"language floor violated: a language got {worst} prompts in a 20-step block")
            weak = [per_lang[l] for l in gen.WEAK_LIST if l in per_lang]
            other = [per_lang[l] for l in langs if l not in gen.WEAK_LIST]
            rows.append(dict(stage=stage, steps=cfg.steps, completions=n_prompts * cfg.group_size, prompts=n_prompts, stage_bins=dict(bins),
                             min_prompts_lang=min(per_lang[l] for l in langs), weak_mean=round(statistics.mean(weak), 1) if weak else None,
                             other_mean=round(statistics.mean(other), 1) if other else None, domains=dict(per_dom), plateau=cfg.plateau_stop, thr=cfg.kl_guard_threshold))
        except Exception as e:                                                      # noqa: BLE001
            errs.append(f"schedule replay: {type(e).__name__}: {e}")
        # ---- the yaml is what the generator writes (no hand edits)
        if not a.no_generated_check:
            gtxt = gen.render(stage, "specialist" if spec else "a0", P)
            if gtxt != open(path, encoding="utf-8").read():
                errs.append("file differs from the generator output (hand edit or stale: rerun scripts/clean_rerun_make_configs.py)")
        # ---- files named in the yaml exist (rental only)
        if a.check_files:
            for key in ("model_name", "prompts_path", "benchmark_eval_path"):
                v = d.get(key)
                if v and not os.path.exists(v) and not (key == "model_name" and stage not in ("preflight", "A")):
                    errs.append(f"missing {key}: {v}")
            for k, v in cfg.reward.items():
                if isinstance(v, str) and v.startswith("/") and k.endswith(("_path", "_dir")) and not os.path.exists(v):
                    if k in ("nllb_pseudoref_dir", "nllb_calibration_path") and not cfg.reward.get("nllb_signal"):
                        continue
                    if k.startswith("science_anchor") and not fin:
                        continue
                    errs.append(f"missing reward.{k}: {v}")
            for ax in cfg.benchmark_aux:
                if not os.path.exists(ax["path"]):
                    errs.append(f"missing benchmark_aux {ax['name']}: {ax['path']}")
        if errs:
            errors += [f"{fname}: {m}" for m in errs]
    print(f"{'stage':9s} {'steps':>5s} {'completions':>11s} {'prompts':>7s} {'min/lang':>8s} {'weak':>6s} {'other':>6s} {'plateau':>7s} {'kl_thr':>6s}  bins / domains")
    for r in rows:
        print(f"{r['stage']:9s} {r['steps']:5d} {r['completions']:11d} {r['prompts']:7d} {r['min_prompts_lang']:8d} {str(r['weak_mean']):>6s} {str(r['other_mean']):>6s} {str(r['plateau']):>7s} {r['thr']:6.2f}  {r['stage_bins']} {r['domains']}")
    tot = sum(r["completions"] for r in rows if r["stage"] != "preflight")
    print(f"completions A..final (preflight excluded): {tot:,}")
    if errors:
        print("\nPROBLEMS:")
        for e in errors:
            print("  -", e)
        return 1
    print("\nALL STAGE YAMLS LOAD STRICTLY (RLConfig + RewardConfig + validate_config), THE SCHEDULES REPLAY, THE INVARIANTS HOLD")
    return 0


# ------------------------------------------------------------------------------------------------ manifest
def _run(cmd: List[str], cwd: Optional[str] = None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:                                                         # noqa: BLE001
        return f"ERROR {type(e).__name__}"


def cmd_manifest(a) -> int:
    man: Dict[str, Any] = {"written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(), "python": sys.version.split()[0], "arm": a.arm, "stage": a.stage}
    fatal: List[str] = []
    # rule file, amendments, weak-list rule
    base = a.prereg_dir
    for key, (want, rel) in PIN.items():
        p = os.path.join(base, rel) if key != "weak_rule" else os.path.join(a.weak_dir, os.path.basename(rel)) if a.weak_dir else os.path.join(base, "..", "weak_selection_2026-10-26", "DECISION_RULE.txt")
        if os.path.exists(p):
            got = sha256_file(p)
            man[f"{key}_sha256"] = got
            if got != want:
                fatal.append(f"{key}: sha256 {got} != pinned {want}")
        else:
            fatal.append(f"{key} file missing: {p}")
    # protected index version and manifest
    man["protected_index_version"] = a.index_version
    if a.index_version != INDEX_VERSION:
        fatal.append(f"protected index version {a.index_version} != {INDEX_VERSION}")
    if a.index_manifest and os.path.exists(a.index_manifest):
        got = sha256_file(a.index_manifest)
        man["protected_index_manifest_sha256"] = got
        if got != a.expect_index_manifest_sha:
            fatal.append(f"index manifest sha256 {got} != expected {a.expect_index_manifest_sha}")
    else:
        fatal.append(f"protected index manifest missing: {a.index_manifest}")
    if a.heldout_split and os.path.exists(a.heldout_split):
        hs = json.load(open(a.heldout_split))
        man["h_split_file_sha256"] = sha256_file(a.heldout_split)
        man["h_sel_sha256"], man["h_cal_sha256"] = hs.get("h_sel_sha256"), hs.get("h_cal_sha256")
        if hs.get("h_sel_sha256") != HSEL_SHA or hs.get("h_cal_sha256") != HCAL_SHA:
            fatal.append("H_sel / H_cal id-list hashes differ from the pinned ones")
    # configs
    cfgs = {}
    for p in sorted(glob.glob(os.path.join(a.configs, "*.yaml"))):
        cfgs[os.path.basename(p)] = sha256_file(p)
    man["config_dir"] = a.configs
    man["config_sha256"] = cfgs
    if len(cfgs) < 7:
        fatal.append(f"only {len(cfgs)} yaml files in {a.configs}")
    # trainer code (the repo working tree is not clean: hash the files that decide the run, too)
    code = {}
    for rel in ("gemma_mt/rl/train_rl.py", "gemma_mt/rl/grpo.py", "gemma_mt/rl/reward.py", "gemma_mt/rl/data.py", "gemma_mt/rl/run_controls.py", "gemma_mt/rl/lang_weights.py", "gemma_mt/rl/sampling.py", "gemma_mt/rl/vllm_sampler.py"):
        p = os.path.join(a.tree, rel)
        code[rel] = sha256_file(p) if os.path.exists(p) else None
    man["trainer_code_sha256"] = code
    man["launcher_sha256"] = {os.path.basename(p): sha256_file(p) for p in (os.path.join(HERE, "launch_clean_rerun.sh"), os.path.abspath(__file__), os.path.join(HERE, "clean_rerun_make_configs.py")) if os.path.exists(p)}
    top = _run(["git", "rev-parse", "--show-toplevel"], cwd=a.tree)
    if top and not top.startswith("ERROR") and os.path.isdir(top):
        man["git_commit"] = _run(["git", "rev-parse", "HEAD"], cwd=top)
        man["git_branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=top)
        man["git_dirty_entries"] = len([l for l in _run(["git", "status", "--porcelain"], cwd=top).splitlines() if l.strip()])
    else:
        man["git_commit"] = os.environ.get("GIT_COMMIT", "UNKNOWN (not a git tree; set GIT_COMMIT)")
        if man["git_commit"].startswith("UNKNOWN"):
            fatal.append("no git commit available (copy the tree with its .git or export GIT_COMMIT)")
    # GPU
    gpu = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    man["gpu"] = gpu
    if not a.allow_gpu and "RTX PRO 6000" not in gpu:
        fatal.append(f"GPU is {gpu!r}, not an RTX PRO 6000 (set ALLOW_GPU=1 to override; the cost model assumes it)")
    for pkg in ("torch", "vllm", "transformers"):
        try:
            import importlib.metadata as md
            man[f"{pkg}_version"] = md.version(pkg)
        except Exception:                                                          # noqa: BLE001
            man[f"{pkg}_version"] = None
    # start model + pools
    if a.start_model:
        mp = os.path.join(a.start_model, "model.safetensors")
        man["start_model"] = a.start_model
        man["start_model_sha256"] = sha256_file(mp) if os.path.exists(mp) else None
        if man["start_model_sha256"] is None:
            fatal.append(f"{mp} missing")
    for name, d in (("prompts_clean_v3", a.pool_v3), ("prompts_clean_sci_v2", a.pool_sci)):
        if d and os.path.isdir(d) and a.hash_pools:
            files = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
            h = hashlib.sha256()
            for p in files:
                h.update(f"{os.path.relpath(p, d)}\t{sha256_file(p)}\n".encode())
            man[name] = {"dir": d, "files": len(files), "tree_sha256": h.hexdigest(), "hub_revision": os.environ.get("PROMPTS_HUB_REVISION", ("UNSET (" + os.environ['HF_ORG_LEGACY'] + "/sabiyarn-clean-rl-prompts-2026-10-28)"))}
        elif d:
            man[name] = {"dir": d, "hashed": False}
    man["sft_arm"] = a.arm
    man["notes"] = ["libretexts: sci_v2 prompts with source_dataset common-pile/libretexts are INCLUDED as English prompts only (licence label cc-by vs README CC BY-NC-SA unresolved); disclose in the paper",
                    "nllb_signal is on in the final run only (NC licence, reward-only, disclosed)"]
    man["fatal"] = fatal
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    with open(a.out + ".sha256", "w") as f:
        f.write(sha256_file(a.out) + "  " + os.path.basename(a.out) + "\n")
    print(f"manifest written: {a.out} (sha256 {sha256_file(a.out)[:16]}...)")
    if fatal:
        print("MANIFEST PROBLEMS (launch refused):")
        for m in fatal:
            print("  -", m)
        return 1
    return 0


# ------------------------------------------------------------------------------------------------ gate
def read_metrics(run: str) -> List[Dict[str, Any]]:
    out = []
    p = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _pct(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def cmd_gate(a) -> int:
    rows = read_metrics(a.run)
    # a restart appends again from the resume step: keep the LAST record of every step
    train: Dict[int, Dict[str, Any]] = {}
    evals: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        s = int(r.get("step", -1))
        if "kl" in r and "grad_norm" in r:
            train[s] = r
        if any(k.startswith(("eval/", "bench/")) for k in r):
            evals.setdefault(s, {}).update(r)
    steps = sorted(train)
    rep: Dict[str, Any] = {"stage": a.stage, "run": a.run, "steps_logged": len(steps), "last_step": steps[-1] if steps else None}
    hard: List[str] = []
    warn: List[str] = []
    if not steps:
        hard.append("no training steps in metrics.jsonl")
    else:
        kl = [train[s]["kl"] for s in steps]
        gn = [train[s]["grad_norm"] for s in steps]
        fin = [train[s]["finished_rate"] for s in steps]
        lr_scale = [train[s].get("kl_guard_lr_scale", 1.0) for s in steps]
        ts = [train[s]["t_step"] for s in steps if "t_step" in train[s]]
        share = [train[s]["greedy_adv_token_share"] for s in steps if "greedy_adv_token_share" in train[s]]
        tail = steps[len(steps) // 2:]
        klt = [train[s]["kl"] for s in tail]
        rep.update(kl_median=statistics.median(kl), kl_max=max(kl), kl_tail_median=statistics.median(klt), grad_norm_median=statistics.median(gn), grad_norm_p95=_pct(gn, 0.95), grad_norm_max=max(gn),
                   finished_mean=statistics.mean(fin), finished_min=min(fin), guard_min_scale=min(lr_scale), t_step_median=statistics.median(ts) if ts else None, greedy_adv_share_median=statistics.median(share) if share else None)
        if ts:
            rep["gpu_hours_per_100_steps"] = round(100 * statistics.median(ts) / 3600, 3)
        cuts = 0
        ev_path = os.path.join(a.run, "kl_guard_events.jsonl")
        if os.path.exists(ev_path):
            cuts = sum(1 for l in open(ev_path) if '"cut"' in l)
        rep["guard_cuts"] = cuts
        if not all(math.isfinite(x) for x in kl + gn):
            hard.append("non-finite KL or grad norm")
        thr = 0.25 if a.stage == "final" else 0.15
        if a.stage == "preflight":
            if cuts or min(lr_scale) < 1.0:
                hard.append(f"KL guard fired in the preflight ({cuts} cuts, lowest scale {min(lr_scale)})")
            if statistics.median(klt) > 0.08:
                hard.append(f"median KL over the second half {statistics.median(klt):.3f} > 0.08 (2b-b64 healthy 0.045; AB2 4-language arm reached 0.12-0.14)")
            if max(kl) > 0.15:
                hard.append(f"max KL {max(kl):.3f} > 0.15")
            if statistics.median(gn) > 3.0:
                hard.append(f"median grad norm {statistics.median(gn):.2f} > 3.0 (AB2 1.9; D-67 greedy-stats recipe 0.60)")
            if min(fin) < 0.98:
                warn.append(f"a step had finished rate {min(fin):.3f} < 0.98")
        else:
            if cuts >= 4:
                hard.append(f"{cuts} KL-guard cuts in this stage (>= 4)")
            elif cuts:
                warn.append(f"{cuts} KL-guard cut(s); lowest lr scale {min(lr_scale):.3f}: record the effective lr with the result")
            if max(kl) > thr:
                warn.append(f"max KL {max(kl):.3f} > threshold {thr}")
            if statistics.median(gn) > 3.0:
                hard.append(f"median grad norm {statistics.median(gn):.2f} > 3.0")
        if share and statistics.median(share) > 0.35:
            warn.append(f"greedy members carry a median {statistics.median(share):.2f} of the token-weighted |advantage| (> 0.35)")
        # evaluations: greedy loop / finished, dev chrF++
        es = sorted(evals)
        if es:
            last = evals[es[-1]]
            loop = last.get("eval/greedy_loop_rate")
            gfin = last.get("eval/greedy_finished_rate")
            rep["eval_last_step"] = es[-1]
            rep["greedy_loop_rate_last"] = loop
            rep["greedy_finished_rate_last"] = gfin
            rep["greedy_loop_rate_by_bin_last"] = {k.split("/")[-1]: v for k, v in last.items() if k.startswith("eval/greedy_loop_rate_bin/")}
            rep["eval_reward_last"] = last.get("eval/reward")
            if loop is not None and loop > (0.02 if a.stage == "preflight" else 0.03):
                hard.append(f"greedy loop rate {loop:.3f} at step {es[-1]} (limit {0.02 if a.stage == 'preflight' else 0.03})")
            elif loop is not None and loop > 0.01:
                warn.append(f"greedy loop rate {loop:.3f} at step {es[-1]} (> 0.01)")
            if gfin is not None and gfin < 0.97:
                hard.append(f"greedy finished rate {gfin:.3f} < 0.97")
        bench = [(s, evals[s]["bench/chrf_macro"]) for s in es if "bench/chrf_macro" in evals[s]]
        if bench:
            rep["dev_chrf_series"] = bench
            first, lastv = bench[0][1], bench[-1][1]
            best = max(v for _, v in bench)
            rep["dev_chrf_first_last_best"] = [first, lastv, best]
            bfin = evals[bench[-1][0]].get("bench/finished_rate")
            if bfin is not None and bfin < 0.95:
                hard.append(f"dev greedy finished rate {bfin:.3f} < 0.95")
            if len(bench) >= 2:
                drop = first - lastv
                if drop >= (0.5 if a.stage == "preflight" else 1.0):
                    hard.append(f"dev chrF++ fell {drop:.2f} from the first to the last evaluation")
                elif best - lastv >= 0.3:
                    warn.append(f"dev chrF++ is {best - lastv:.2f} below its best of this stage (AB2 pattern): consider HANDOFF=best")
        else:
            warn.append("no dev chrF++ evaluation in the stage")
        summ = os.path.join(a.run, "summary.json")
        if os.path.exists(summ):
            sj = json.load(open(summ))
            rep["stop_reason"] = sj.get("stop_reason")
            rep["steps_done"] = sj.get("steps")
    rep["hard_failures"], rep["warnings"] = hard, warn
    rep["verdict"] = "FAIL" if hard else ("PASS_WITH_WARNINGS" if warn else "PASS")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("dev_chrf_series",)}, indent=2, default=str))
    return 1 if hard else 0


# ------------------------------------------------------------------------------------------------ winddown watcher / handoff
def cmd_winddown(a) -> int:
    """Loop until summary.json exists. When the last logged step >= steps - n and no WINDDOWN file exists: write it (n steps: the lr decays linearly to 0, the run ends at `steps`)."""
    p_w = os.path.join(a.run, "WINDDOWN")
    p_s = os.path.join(a.run, "summary.json")
    p_d = os.path.join(a.run, "WINDDOWN.done")
    target = a.steps - a.n
    while not os.path.exists(p_s):
        if not os.path.exists(p_w) and not os.path.exists(p_d):
            try:
                with open(os.path.join(a.run, "metrics.jsonl"), "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(0, size - 200000))
                    tail = f.read().decode("utf-8", "ignore").splitlines()
                last = 0
                for line in reversed(tail):
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "kl" in r and "grad_norm" in r:
                        last = int(r["step"])
                        break
                if last >= target:
                    with open(p_w, "w") as f:
                        f.write(str(a.n))
                    print(f"[winddown-watch] step {last} >= {target}: wrote WINDDOWN {a.n} (the run ends at step {a.steps})", flush=True)
            except OSError:
                pass
        time.sleep(a.poll)
    return 0


def cmd_handoff(a) -> int:
    d = os.path.join(a.run, "best" if a.mode == "best" else "final")
    if not os.path.exists(os.path.join(d, "model.safetensors")) and not glob.glob(os.path.join(d, "*.safetensors")):
        print(f"no model in {d}", file=sys.stderr)
        return 1
    print(d)
    return 0


# ------------------------------------------------------------------------------------------------ disclose
def cmd_disclose(a) -> int:
    setup_imports(a.tree, a.stub_heavy)
    from gemma_mt.rl.data import PromptSchedule, load_rl_prompts
    from gemma_mt.rl.train_rl import RLConfig, _subset, split_eval_prompts
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    cfg = dataclass_from_dict(RLConfig, load_yaml(a.config))
    last_step = a.steps or cfg.steps
    pr = load_rl_prompts(cfg.prompts_path, max_src_tokens=cfg.max_src_tokens)
    tr_idx, ev_idx = split_eval_prompts(pr, min(cfg.eval_prompts, len(pr) // 10), cfg.seed, cfg.eval_min_per_bin)
    pr = _subset(pr, tr_idx)
    sch = PromptSchedule(pr.iso, cfg.language_alpha, cfg.seed, length_bin=pr.length_bin if cfg.length_bins else None, curriculum=cfg.curriculum_steps or None,
                         language_weights=cfg.language_weights or None, domain=pr.domain or None, domain_weights=cfg.domain_weights or None,
                         domain_bin_weights=cfg.domain_bin_weights or None, floor_steps=cfg.language_floor_steps, floor_prompts=cfg.language_floor_prompts)
    by_src: collections.Counter = collections.Counter()
    drawn: set = set()
    for s in range(last_step):
        cur = sch.bin_of_step(s)
        n = cfg.prompts_per_step_by_bin.get(cur, cfg.prompts_per_step)
        for k in sch.indices(s, n):
            by_src[pr.source_dataset[k]] += 1
            drawn.add(k)
    out = {"config": a.config, "steps_replayed": last_step, "prompts_drawn": sum(by_src.values()), "distinct_prompts": len(drawn), "by_source_dataset": dict(by_src.most_common())}
    lib = {pr.english[k] for k in range(len(pr)) if "libretexts" in (pr.source_dataset[k] or "").lower()}
    out["libretexts_prompts_in_pool"] = len(lib)
    if a.archive and os.path.exists(a.archive):
        import sqlite3
        con = sqlite3.connect(a.archive)
        n_lib = n_all = 0
        for (src,) in con.execute("SELECT source FROM best"):
            n_all += 1
            n_lib += 1 if src.strip() in lib else 0
        out["archive_rows"], out["archive_rows_from_libretexts_windows"] = n_all, n_lib
    print(json.dumps(out, indent=2))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--tree", default=TREE_DEFAULT)
    v.add_argument("--configs", default=None)
    v.add_argument("--arm", choices=["a0", "specialist"], default="a0")
    v.add_argument("--root", default="/workspace/clean_rerun")
    v.add_argument("--stub-heavy", action="store_true")
    v.add_argument("--check-files", action="store_true")
    v.add_argument("--no-generated-check", action="store_true")
    v.add_argument("--allow-other-steps", action="store_true")
    m = sub.add_parser("manifest")
    m.add_argument("--tree", default=TREE_DEFAULT)
    m.add_argument("--configs", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--prereg-dir", required=True, help="reports/data/prereg_clean_2026-10-28 (DECISION_RULE.txt, AMENDMENT_1.txt, AMENDMENT_2.txt)")
    m.add_argument("--weak-dir", default=None, help="reports/data/weak_selection_2026-10-26")
    m.add_argument("--index-version", default=INDEX_VERSION)
    m.add_argument("--index-manifest", default=None, help="protected_manifest.json of protected-2026-10-28.3")
    m.add_argument("--expect-index-manifest-sha", default=INDEX_MANIFEST_SHA)
    m.add_argument("--heldout-split", default=None, help="heldout_split_cal_sel_v2.json")
    m.add_argument("--start-model", default=None)
    m.add_argument("--pool-v3", default=None)
    m.add_argument("--pool-sci", default=None)
    m.add_argument("--hash-pools", action="store_true")
    m.add_argument("--arm", default="A0")
    m.add_argument("--stage", default="all")
    m.add_argument("--allow-gpu", action="store_true")
    g = sub.add_parser("gate")
    g.add_argument("--stage", required=True)
    g.add_argument("--run", required=True)
    g.add_argument("--out", default=None)
    w = sub.add_parser("winddown")
    w.add_argument("--run", required=True)
    w.add_argument("--steps", type=int, required=True)
    w.add_argument("--n", type=int, default=30)
    w.add_argument("--poll", type=float, default=20.0)
    h = sub.add_parser("handoff")
    h.add_argument("--run", required=True)
    h.add_argument("--mode", choices=["final", "best"], default="final")
    d = sub.add_parser("disclose")
    d.add_argument("--tree", default=TREE_DEFAULT)
    d.add_argument("--config", required=True)
    d.add_argument("--steps", type=int, default=0)
    d.add_argument("--archive", default=None)
    d.add_argument("--out", default=None)
    d.add_argument("--stub-heavy", action="store_true")
    a = ap.parse_args(argv)
    return {"validate": cmd_validate, "manifest": cmd_manifest, "gate": cmd_gate, "winddown": cmd_winddown, "handoff": cmd_handoff, "disclose": cmd_disclose}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
