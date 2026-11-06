#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helper commands of the clean-rerun RL launcher (agent C1, 2026-10-28). CPU only; nothing here trains.

    validate   load every yaml of configs/clean_rerun[_specialist] with the REAL config dataclasses (RLConfig strict, RewardConfig strict, validate_config) and replay the prompt
               schedule of the whole stage on a synthetic pool; also checks the plan invariants (greedy on, guard thresholds, caps) and that the yaml equals what the generator writes
    manifest   pin the launch manifest (rule file / amendment sha256, protected index version + manifest sha256, config hashes, trainer code hashes, git commit, GPU type, start model sha)
    gate       read <run>/metrics.jsonl and decide whether a stage (or the 49-step preflight) is healthy enough to continue
    winddown   background loop: writes <run>/WINDDOWN (N) when the step counter reaches steps - N (the planned wind-down of the final run)
    f2-precheck  OPTIONAL stage F2 (385M plan): from the final run's one-pass dev-paper predictions decide RUN / SKIP (finished >= 97 % and low loop rate -> SKIP)
    f2-decide    OPTIONAL stage F2: from the dev-paper predictions of F and F2 decide KEEP_F2 / DISCARD_F2 (paired bootstrap on paper-level chrF, or finished rate +2 points with chrF not lower)
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
    b256 = a.plan == "b256_385m"
    if b256:                                       # 385M plan (owner 2026-11-02): the RL context is capped at b256 (stages preflight A B1 mixed final)
        import clean_rerun_385m_make_configs as w385
        w385.apply_b256_cap(gen)
    max_src, bins_present = (w385.B256_MAX_SRC, ("b64", "b256")) if b256 else (1024, ("b64", "b256", "b512", "b1024"))

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
        if stage not in gen.STAGES:                # B2 / B3 are not part of the b256 plan
            continue
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
        _expect(errs, cfg.max_src_tokens == max_src, f"max_src_tokens must be {max_src} (source-side cap{'; b256 plan' if b256 else ', Amendment 2'})")
        if b256:
            _expect(errs, not ({"b512", "b1024", "b2048"} & (set(cfg.prompts_per_step_by_bin) | set(cfg.max_new_tokens_by_bin) | set(cfg.curriculum_steps) | set(cfg.domain_bin_weights.get("science", {})) | set(cfg.domain_bin_weights.get("general", {})))), "a bin above b256 appears in a b256-plan yaml")
            _expect(errs, cfg.vllm_max_model_len <= 4096 and cfg.vllm_max_model_len >= 2048, f"vllm_max_model_len {cfg.vllm_max_model_len} outside the b256 plan range 2048..4096")
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
        if b256:                                    # owner 2026-11-03: dynamic sampling in every training stage, the reference term in every stage (rows with a reference only), the reference domain in the translation stages
            _expect(errs, cfg.dynamic_sampling is (stage != "preflight") and (stage == "preflight" or (cfg.dynamic_sampling_oversample >= 1.0 and cfg.dynamic_sampling_max_refills >= 1)), "dynamic_sampling must be on in A / B1 / mixed / final and off in the preflight")
            _expect(errs, (cfg.reward.get("reference_weight", 0.0) == w385.REF_WEIGHT) is (stage != "preflight"), "reward.reference_weight must be set (w385.REF_WEIGHT) in every stage but the preflight")
            _expect(errs, (cfg.adv_std_floor > 0) is (stage != "preflight") and cfg.log_group_stats is (stage != "preflight"), "adv_std_floor / log_group_stats must be on in every stage but the preflight")
            if stage in ("B1", "mixed", "final"):
                _expect(errs, cfg.save_dev_predictions is True, "save_dev_predictions must be on in B1 / mixed / final")
            if stage in ("mixed", "final"):
                _expect(errs, cfg.heldout_comet and cfg.hacking_alarm and cfg.early_stop_dev_require_comet and cfg.lang_stall_log and cfg.heldout_comet_model == "masakhane/africomet-stl-1.1" and cfg.scorer.get("python"),
                        "the held-out AfriCOMET monitor, hacking alarm, combined stop and language history must be on in mixed / final")
            if stage in ("B1", "mixed", "final"):
                _expect(errs, set(cfg.guide_mode_by_language.values()) == {"mbr"} and {"orm", "lug", "ewe", "tir"} <= set(cfg.guide_mode_by_language) and cfg.guide_mbr_extra_samples > 0, "MBR guide for the weak languages (+ orm lug) must be on in B1 / mixed / final")
                _expect(errs, cfg.early_stop_dev and cfg.benchmark_eval_every == 25 and cfg.early_stop_dev_k == 3, "early_stop_dev (k 3, benchmark_eval_every 25) must be on in B1 / mixed / final")
            _expect(errs, (cfg.guide_fraction > 0) is (stage != "preflight") and (stage == "preflight" or (cfg.guide_beams >= 2 and 0 < cfg.guide_adv_share_cap < 1 and cfg.hard_fail_mask == "off")), "beam guide must be on (capped, hard_fail_mask off) in every stage but the preflight")
            if stage in ("A", "B1", "mixed"):
                _expect(errs, abs(cfg.domain_weights.get("ref", 0.0) - w385.REF_SHARE) < 1e-9 and abs(sum(cfg.domain_weights.values()) - 1.0) < 1e-9, f"domain_weights {cfg.domain_weights}: ref share must be {w385.REF_SHARE} and the shares must sum to 1")
        if not spec:
            _expect(errs, set(cfg.language_weights) == set(gen.WEIGHTED_LANGS) and all(v == 2.5 for v in cfg.language_weights.values()), "language_weights != the 25-language dev-defined weak list + the 4 owner-priority languages (efi tiv bin ish) at 2.5")
        if stage in ("A", "B1", "B2", "B3", "preflight"):
            _expect(errs, cfg.curriculum_steps == {S["bin"]: cfg.steps}, f"curriculum {cfg.curriculum_steps}")
        else:
            _expect(errs, cfg.curriculum_steps == {"b64": 0}, f"curriculum {cfg.curriculum_steps} (mixed from step 0 needs {{b64: 0}})")
        if fin:
            _expect(errs, cfg.domain_weights == {"science": 0.7, "general": 0.3}, "domain_weights")
            _expect(errs, cfg.language_floor_steps == 20 and cfg.language_floor_prompts == 2, "language floor 2 prompts / 20 steps")
            if b256:
                _expect(errs, abs(sum(cfg.domain_bin_weights["science"].values()) - 1.0) < 1e-9 and abs(cfg.domain_bin_weights["science"]["b256"] - 2 / 3) < 1e-6 and abs(cfg.domain_bin_weights["science"]["b64"] - 1 / 3) < 1e-6, "science bin weights (b256 plan: b64 1/3, b256 2/3)")
                _expect(errs, cfg.domain_bin_weights["general"] == {"b64": 0.5, "b256": 0.5}, "general bin weights (b256 plan: 1/2, 1/2)")
            else:
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
            doms = list(cfg.domain_weights) if cfg.domain_weights else [""]          # 385M plan: general / news / ref (translation stages), science / general (final)
            for l in langs:
                for b in bins_present:
                    for dname in doms:
                        iso += [l] * 200
                        lb += [b] * 200
                        dm += [dname] * 200
            sch = PromptSchedule(np.asarray(iso), cfg.language_alpha, cfg.seed, length_bin=lb, curriculum=cfg.curriculum_steps or None, language_weights=cfg.language_weights or None,
                                 domain=dm if cfg.domain_weights else None, domain_weights=cfg.domain_weights or None, domain_bin_weights=cfg.domain_bin_weights or None,
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
    if "F2" in (a.extra_stages or []):
        if not b256:
            errors.append("--extra-stages F2 needs --plan b256_385m")
        else:
            errs = []
            fname = w385.F2_FILE
            try:
                d = load_yaml(os.path.join(cfg_dir, fname)); cfg = dataclass_from_dict(RLConfig, d); validate_config(cfg); dataclass_from_dict(RewardConfig, cfg.reward)
                fin_cfg = dataclass_from_dict(RLConfig, load_yaml(os.path.join(cfg_dir, STAGE_FILES["final"])))
            except Exception as e:                                           # noqa: BLE001
                errors.append(f"{fname}: {type(e).__name__}: {e}")
                cfg = None
            if cfg is not None:
                _expect(errs, cfg.steps == w385.F2_STEPS, f"steps {cfg.steps} != {w385.F2_STEPS}")
                _expect(errs, cfg.model_name.endswith("runs_385m/final/final") and cfg.out_dir.endswith("runs_385m/F2") and (cfg.hub_repo_id, cfg.hub_branch) == w385.F2_HUB, "model_name / out_dir / hub_repo_id / hub_branch (w385.F2_HUB)")
                _expect(errs, cfg.max_src_tokens == w385.F2_MAX_SRC and cfg.vllm_max_model_len == w385.F2_MAX_MODEL_LEN, "max_src_tokens / vllm_max_model_len")
                _expect(errs, cfg.vllm_max_model_len >= 2400 + w385.F2_MAX_NEW["b2048"], "vllm_max_model_len below the b2048 prompt (<= ~2,300 Gemma tokens) + max_new_tokens")
                _expect(errs, cfg.prompts_per_step_by_bin == w385.F2_PPS and cfg.max_new_tokens_by_bin == w385.F2_MAX_NEW, "prompts_per_step_by_bin / max_new_tokens_by_bin")
                _expect(errs, cfg.domain_weights == {"science": 1.0} and cfg.domain_bin_weights == {"science": w385.F2_BIN_WEIGHTS}, "F2 is science only, b1024 : b2048 = 60 : 40")
                _expect(errs, cfg.prompts_path.endswith("clean_sci_v2_long") and cfg.curriculum_steps == {"b64": 0}, "prompts_path / curriculum_steps (mixed from step 0)")
                _expect(errs, cfg.gradient_checkpointing is True and cfg.vllm_gpu_memory_utilization == 0.3 and cfg.archive is False, "gradient_checkpointing / vllm 0.30 / no archive")
                _expect(errs, cfg.learning_rate == 1.0e-6 and cfg.kl_coef == 0.1 and abs(cfg.kl_guard_threshold - 0.25) < 1e-12 and cfg.kl_guard_recover and cfg.group_size == 16 and cfg.greedy_in_group, "F2 lr / kl_coef / guard / group")
                _expect(errs, cfg.plateau_stop is False and cfg.early_stop and cfg.extend_steps > cfg.planned_winddown_last > 0, "plateau_stop off, early stop on, extend_steps > planned_winddown_last")
                same = {k for k in set(cfg.reward) | set(fin_cfg.reward) if cfg.reward.get(k) != fin_cfg.reward.get(k)}
                _expect(errs, same == {"nllb_signal", "nllb_languages"}, f"the reward must equal the final run's except the NLLB signal (differences: {sorted(same)})")
                _expect(errs, cfg.reward.get("longdoc_v2") is True and cfg.reward.get("longdoc_min_src_words") == 1 and cfg.reward.get("nllb_signal") is False, "longdoc_v2 on / NLLB off")
                _expect(errs, any(x["name"] == "afrisci_dev_doc" for x in cfg.benchmark_aux), "the dev-paper monitor afrisci_dev_doc is missing")
                try:
                    iso, lb, dm = [], [], []
                    for l in langs:
                        for b in ("b64", "b256", "b512", "b1024", "b2048"):
                            for dname in ("science", "general"):
                                iso += [l] * 200; lb += [b] * 200; dm += [dname] * 200
                    sch = PromptSchedule(np.asarray(iso), cfg.language_alpha, cfg.seed, length_bin=lb, curriculum=cfg.curriculum_steps or None, language_weights=cfg.language_weights or None,
                                         domain=dm, domain_weights=cfg.domain_weights or None, domain_bin_weights=cfg.domain_bin_weights or None,
                                         floor_steps=cfg.language_floor_steps, floor_prompts=cfg.language_floor_prompts)
                    bins, per_dom, per_lang, win, n_prompts = collections.Counter(), collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter), 0
                    for st_ in range(cfg.steps):
                        n = cfg.prompts_per_step_by_bin.get(sch.bin_of_step(st_), cfg.prompts_per_step)
                        idx = sch.indices(st_, n)
                        _expect(errs, len(idx) == n, f"step {st_}: {len(idx)} prompts, expected {n}")
                        n_prompts += len(idx)
                        for k in idx:
                            bins[lb[k]] += 1; per_dom[dm[k]] += 1; per_lang[iso[k]] += 1; win[st_ // 20][iso[k]] += 1
                    _expect(errs, set(bins) == {"b1024", "b2048"} and set(per_dom) == {"science"}, f"F2 draws bins {dict(bins)} / domains {dict(per_dom)}")
                    share = bins["b2048"] / max(sum(bins.values()), 1)
                    _expect(errs, abs(share - 0.4) < 0.05, f"b2048 share {share:.3f} != 0.40 +- 0.05")
                    full = [blk for blk in win if (blk + 1) * 20 <= cfg.steps]
                    worst = min((win[blk][l] for blk in full for l in langs), default=None)
                    _expect(errs, worst is None or worst >= 2, f"language floor violated: a language got {worst} prompts in a 20-step block")
                    rows.append(dict(stage="F2", steps=cfg.steps, completions=n_prompts * cfg.group_size, prompts=n_prompts, stage_bins=dict(bins), min_prompts_lang=min(per_lang[l] for l in langs),
                                     weak_mean=round(statistics.mean([per_lang[l] for l in gen.WEAK_LIST if l in per_lang]), 1), other_mean=round(statistics.mean([per_lang[l] for l in langs if l not in gen.WEAK_LIST]), 1),
                                     domains=dict(per_dom), plateau=cfg.plateau_stop, thr=cfg.kl_guard_threshold))
                except Exception as e:                                                  # noqa: BLE001
                    errs.append(f"schedule replay: {type(e).__name__}: {e}")
                if a.check_files:
                    for key in ("prompts_path", "benchmark_eval_path"):
                        if d.get(key) and not os.path.exists(d[key]):
                            errs.append(f"missing {key}: {d[key]}")
                    for ax in cfg.benchmark_aux:
                        if not os.path.exists(ax["path"]):
                            errs.append(f"missing benchmark_aux {ax['name']}: {ax['path']}")
            errors += [f"{fname}: {m}" for m in errs]
    print(f"{'stage':9s} {'steps':>5s} {'completions':>11s} {'prompts':>7s} {'min/lang':>8s} {'weak':>6s} {'other':>6s} {'plateau':>7s} {'kl_thr':>6s}  bins / domains")
    for r in rows:
        print(f"{r['stage']:9s} {r['steps']:5d} {r['completions']:11d} {r['prompts']:7d} {r['min_prompts_lang']:8d} {str(r['weak_mean']):>6s} {str(r['other_mean']):>6s} {str(r['plateau']):>7s} {r['thr']:6.2f}  {r['stage_bins']} {r['domains']}")
    tot = sum(r["completions"] for r in rows if r["stage"] not in ("preflight", "F2"))
    print(f"completions A..final (preflight and the optional F2 excluded): {tot:,}")
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
    if len(cfgs) < 5:                                  # 7 yamls in the full plan, 5 (+ the optional rl_F2_long.yaml) in the b256 plan of the 385M model
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
        thr = 0.25 if a.stage in ("final", "F2") else 0.15
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


# ------------------------------------------------------------------------------------------------ optional stage F2 (385M plan): pre-check and keep / discard decision
F2_DEV_PAPERS = 150                 # AfriScience-MT dev: 25 papers x amh hau lug nso yor zul, one greedy pass, whole paper in ONE prompt (scripts/eval_afriscience.py --splits dev --levels document)


def _paper_table(path: str, level: str = "document", chrf=None) -> Dict[Any, Dict[str, Any]]:
    """{(iso, paper): {chrf, finished, loop, empty}} from the predictions_dev.json that scripts/eval_afriscience.py writes ({level: {iso: [{paper, hyp, finished, ref}]}}).
    chrf = sacrebleu CHRF(word_order=0) per paper (the paper's chrF; NFC-normalised), loop = rl.guards.is_repetitive (the trainer's loop definition)."""
    import unicodedata
    if chrf is None:
        from sacrebleu.metrics import CHRF
        scorer = CHRF(word_order=0)
        chrf = lambda h, r: float(scorer.sentence_score(h, [r]).score)       # noqa: E731
    from gemma_mt.rl.guards import is_repetitive
    nf = lambda x: unicodedata.normalize("NFC", x)                           # noqa: E731
    data = json.load(open(path, encoding="utf-8"))
    if level not in data:
        raise SystemExit(f"{path}: no level {level!r} (levels: {sorted(data)}); run eval_afriscience.py with --levels {level}")
    out: Dict[Any, Dict[str, Any]] = {}
    for iso, rows in data[level].items():
        for r in rows:
            hyp = nf(r["hyp"])
            out[(iso, str(r["paper"]))] = dict(chrf=chrf(hyp, nf(r["ref"])), finished=bool(r.get("finished", True)) and bool(hyp.strip()), loop=bool(is_repetitive(hyp)), empty=not hyp.strip())
    return out


def _summ(t: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    n = len(t)
    per = collections.defaultdict(list)
    for (iso, _), v in t.items():
        per[iso].append(v)
    return dict(papers=n, chrf_mean=statistics.mean(v["chrf"] for v in t.values()) if n else float("nan"), finished_rate=sum(v["finished"] for v in t.values()) / max(n, 1),
                loop_rate=sum(v["loop"] for v in t.values()) / max(n, 1),
                per_language={iso: dict(papers=len(vs), chrf=round(statistics.mean(x["chrf"] for x in vs), 2), finished=round(sum(x["finished"] for x in vs) / len(vs), 3), loop=round(sum(x["loop"] for x in vs) / len(vs), 3)) for iso, vs in sorted(per.items())})


def cmd_f2_precheck(a) -> int:
    """Run F2 at all? After the final run (F): one greedy pass over the dev papers, whole paper in one prompt. SKIP when the finished rate is >= --min-finished (97 %) AND the loop rate is <= --max-loop (2 %,
    the preflight limit of the greedy loop rate); otherwise RUN. Exit code 0 = RUN, 10 = SKIP, 1 = invalid input."""
    setup_imports(a.tree, a.stub_heavy)
    t = _paper_table(a.preds)
    r = _summ(t)
    rep: Dict[str, Any] = dict(stage="F2-precheck", preds=a.preds, thresholds=dict(min_finished=a.min_finished, max_loop=a.max_loop), **r)
    if r["papers"] < a.min_papers:
        rep["verdict"] = "INVALID"
        rep["reason"] = f"{r['papers']} papers < {a.min_papers} (the dev set is 25 papers x 6 languages)"
    else:
        skip = r["finished_rate"] >= a.min_finished and r["loop_rate"] <= a.max_loop
        rep["verdict"] = "SKIP" if skip else "RUN"
        rep["reason"] = (f"finished {r['finished_rate']:.3f} >= {a.min_finished} and loop {r['loop_rate']:.3f} <= {a.max_loop}: whole-paper decoding already works, F2 would only cost GPU hours" if skip
                         else f"finished {r['finished_rate']:.3f} (need >= {a.min_finished}) or loop {r['loop_rate']:.3f} (need <= {a.max_loop}) misses the threshold")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_language"}, indent=2, default=str))
    return {"RUN": 0, "SKIP": 10}.get(rep["verdict"], 1)


def cmd_f2_decide(a) -> int:
    """Keep F2 or discard it (owner rule 2026-11-02). On the same dev papers (25 per language, one greedy pass, whole paper in one prompt): KEEP_F2 if
      (A) the paper-level chrF improves over F by more than the paired-bootstrap 95 % interval: the lower end of the percentile interval of the mean paired difference (F2 - F, resampling papers) is > 0, OR
      (B) the finished rate rises by at least 2 points (--min-finished-gain 0.02) with the mean paper chrF not lower (point estimate F2 >= F);
    otherwise DISCARD_F2 (keep F). Optional sentence-level predictions of both models (--sentence-base / --sentence-new, eval_afriscience.py --levels sentence) are REPORTED with a warning when the
    sentence chrF++ falls by more than --sentence-warn (0.5): production translates sentence by sentence, the rule above does not look at it. Exit code 0 = KEEP_F2, 10 = DISCARD_F2, 1 = invalid."""
    import numpy as np
    setup_imports(a.tree, a.stub_heavy)
    base, new = _paper_table(a.base), _paper_table(a.new)
    rep: Dict[str, Any] = dict(stage="F2-decide", base=a.base, new=a.new, rule=dict(ci=0.95, min_finished_gain=a.min_finished_gain))
    if set(base) != set(new) or len(base) < a.min_papers:
        rep.update(verdict="INVALID", reason=f"paper sets differ or are too small (base {len(base)}, new {len(new)}, need the same >= {a.min_papers})")
    else:
        keys = sorted(base)
        d = np.asarray([new[k]["chrf"] - base[k]["chrf"] for k in keys], dtype=np.float64)
        rng = np.random.default_rng(a.seed)
        idx = rng.integers(0, len(d), size=(a.bootstrap, len(d)))
        means = d[idx].mean(axis=1)
        lo, hi = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
        sb, sn = _summ(base), _summ(new)
        gain = sn["finished_rate"] - sb["finished_rate"]
        crit_a = lo > 0.0
        crit_b = gain >= a.min_finished_gain - 1e-12 and sn["chrf_mean"] >= sb["chrf_mean"]
        rep.update(base_summary=sb, new_summary=sn, chrf_diff_mean=float(d.mean()), chrf_diff_ci95=[lo, hi], finished_gain=gain, criterion_A_chrf_ci_above_0=bool(crit_a),
                   criterion_B_finished_plus_2_points_chrf_not_lower=bool(crit_b), verdict="KEEP_F2" if (crit_a or crit_b) else "DISCARD_F2")
        rep["reason"] = ("A" if crit_a else "") + ("B" if crit_b else "") or "neither criterion holds: keep F"
    if a.sentence_base and a.sentence_new:
        from sacrebleu.metrics import CHRF
        def sent(path):
            data = json.load(open(path, encoding="utf-8"))["sentence"]
            return {iso: float(CHRF(word_order=2).corpus_score([x["hyp"] for x in rows], [[x["ref"] for x in rows]]).score) for iso, rows in data.items()}
        sb, sn = sent(a.sentence_base), sent(a.sentence_new)
        delta = statistics.mean(sn[i] - sb[i] for i in sb if i in sn)
        rep["sentence_chrf_pp_macro_delta"] = delta
        rep["sentence_warning"] = (f"sentence-level dev chrF++ fell {-delta:.2f} (> {a.sentence_warn}): production translates sentence by sentence, think before keeping F2" if delta < -a.sentence_warn else None)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("base_summary", "new_summary")}, indent=2, default=str))
    return {"KEEP_F2": 0, "DISCARD_F2": 10}.get(rep["verdict"], 1)


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
    v.add_argument("--extra-stages", nargs="*", default=[], choices=["F2"], help="also validate the OPTIONAL stage F2 (385M plan: rl_F2_long.yaml)")
    v.add_argument("--plan", choices=["full", "b256_385m"], default="full", help="b256_385m: the 385M plan with the RL context capped at b256 (stages preflight A B1 mixed final, max_src_tokens 256)")
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
    f = sub.add_parser("f2-precheck")
    f.add_argument("--preds", required=True, help="predictions_dev.json of the FINAL run (scripts/eval_afriscience.py --splits dev --levels document --max-new-tokens-document 7168)")
    f.add_argument("--min-finished", type=float, default=0.97)
    f.add_argument("--max-loop", type=float, default=0.02)
    f.add_argument("--min-papers", type=int, default=F2_DEV_PAPERS)
    f.add_argument("--out", default=None)
    f.add_argument("--tree", default=TREE_DEFAULT)
    f.add_argument("--stub-heavy", action="store_true")
    g2 = sub.add_parser("f2-decide")
    g2.add_argument("--base", required=True, help="predictions_dev.json of F (the final run's model)")
    g2.add_argument("--new", required=True, help="predictions_dev.json of F2")
    g2.add_argument("--min-finished-gain", type=float, default=0.02)
    g2.add_argument("--bootstrap", type=int, default=10000)
    g2.add_argument("--seed", type=int, default=1234)
    g2.add_argument("--min-papers", type=int, default=F2_DEV_PAPERS)
    g2.add_argument("--sentence-base", default=None)
    g2.add_argument("--sentence-new", default=None)
    g2.add_argument("--sentence-warn", type=float, default=0.5)
    g2.add_argument("--out", default=None)
    g2.add_argument("--tree", default=TREE_DEFAULT)
    g2.add_argument("--stub-heavy", action="store_true")
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
    return {"validate": cmd_validate, "manifest": cmd_manifest, "gate": cmd_gate, "winddown": cmd_winddown, "handoff": cmd_handoff, "disclose": cmd_disclose, "f2-precheck": cmd_f2_precheck, "f2-decide": cmd_f2_decide}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
