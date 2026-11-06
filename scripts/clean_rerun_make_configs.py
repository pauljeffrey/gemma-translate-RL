#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Writes the RL stage configs of the CLEAN RERUN (agent C1, 2026-10-28) under gemma_mt/configs/clean_rerun/ (arm A0, 77 languages) or
gemma_mt/configs/clean_rerun_specialist/ (arm S1/S2, the 20 specialist languages, dose-matched steps).

    python scripts/clean_rerun_make_configs.py                      # writes configs/clean_rerun/*.yaml
    python scripts/clean_rerun_make_configs.py --arm specialist     # writes configs/clean_rerun_specialist/*.yaml
    python scripts/clean_rerun_make_configs.py --root /workspace/clean_rerun --print-table

One yaml per stage: rl_00_preflight, rl_A_b64, rl_B1_b256, rl_B2_b512, rl_B3_b1024, rl_C_mixed, rl_F_final. The field names are those of the old yamls
(rl_rtxpro_phase2c*.yaml, rl_science_final2.yaml); the new trainer fields (kl_guard_recover*, plateau_stop*, greedy_share_warn, kl_guard_restore) are the ones of
gemma_mt/rl/train_rl.py of 2026-10-28 (agent T2). The numbers and the evidence for them: reports/clean_rl_stage_plan_2026-10-28.md.
Edit THIS file (not the yamls) and regenerate; `scripts/clean_rerun_tools.py validate` loads every yaml with the real config dataclasses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------------------------------------------------------------------------ the weak list (reports/weak_language_selection_2026-10-26.md, rule sha256 4472d711...6ab7)
WEAK_LIST = ["tir", "dyu", "amh", "kbp", "fon", "nus", "mos", "kam", "bam", "umb", "kea", "kmb", "vmw", "kab", "seh", "yor", "wol", "ewe", "luo", "tum",   # S1: 20 lowest dev chrF++ of 45
             "nyn", "ach",                                                                                                                                  # S2: below the cut on SALT dev
             "kon", "pcm", "shi"]                                                                                                                           # S3: guard trips (kon: LID label artefact, lower confidence)
SPECIALIST_LANGS = ["amh", "tir", "fon", "dyu", "kbp", "nus", "mos", "kam", "bam", "umb", "kea", "kmb", "vmw", "kab", "seh", "wol", "ewe", "luo", "tum", "yor"]   # prereg G + W (+ yor, Amendment 1)
WEAK_WEIGHT = 2.5
# OWNER-PRIORITY languages (owner 2026-10-29: efik, tiv, ibibio (bin = Edo/Bini in the tag set, ish = Esan) are NEEDED languages). They are NOT on the dev-defined weak list: a decision made on need, NOT on test or dev data.
# They join the x2.5 draw class in every A0 yaml (old values: efi 2.5, bin 3.5, ish 4.0; tiv 1.0), so the weak class has 29 members. The specialist yamls carry the same entries; the 20-language specialist pool has no rows of these
# four, so there they have no effect.
OWNER_PRIORITY = ["efi", "tiv", "bin", "ish"]
WEIGHTED_LANGS = WEAK_LIST + OWNER_PRIORITY

# ------------------------------------------------------------------------------------------------ stage table (evidence: reports/clean_rl_stage_plan_2026-10-28.md)
# steps = HARD CAP; plateau_min = plateau_stop_min_steps (None = plateau stop off: the cap is shorter than the 100-step window or already the evidence-based plateau length)
# es = (patience, min_steps) of the patience-based early stop (Amendment 4; owner 2026-10-29: early stopping on EVERY bin and the final run). Stage A cap 400 (owner). Evening of 2026-10-29 (coordinator, owner asked for
# more patience; GPU USD 0.58 per hour, a false stop costs more than extra steps on the cheap bins): A 150/150, B1 80/80, B2 50/30, B3 40/30, mixed 40/30, final 80/60 (the first values were 100/150, 60/80, 40/30, 30/30, 30/30, 60/60).
# extend = steps of one extension block (about 25 % of the base cap, rounded to 5; extend_if_improving, hard maximum +50 %). The slope-CI rule (plateau_stop) is kept where its 100-step window fits.
STAGES: Dict[str, Dict[str, Any]] = {
    "preflight": dict(file="rl_00_preflight.yaml", bin="b64", steps=49, pps=64, plateau_min=None, es=None, extend=0),
    "A": dict(file="rl_A_b64.yaml", bin="b64", steps=400, pps=64, plateau_min=150, es=(150, 150), extend=100),
    "B1": dict(file="rl_B1_b256.yaml", bin="b256", steps=150, pps=48, plateau_min=100, es=(80, 80), extend=40),
    "B2": dict(file="rl_B2_b512.yaml", bin="b512", steps=80, pps=42, plateau_min=None, es=(50, 30), extend=20),
    "B3": dict(file="rl_B3_b1024.yaml", bin="b1024", steps=60, pps=42, plateau_min=None, es=(40, 30), extend=15),
    "mixed": dict(file="rl_C_mixed.yaml", bin="mixed", steps=60, pps=42, plateau_min=None, es=(40, 30), extend=15),
    "final": dict(file="rl_F_final.yaml", bin="mixed", steps=150, pps=42, plateau_min=100, es=(80, 60), extend=40),
}
# NLLB-3.3B pseudo-reference chrF++ signal (reward.nllb_signal; reward use only, CC-BY-NC output, disclosed): owner 2026-10-29 evening: wanted in the RL stages (not SFT) for the very poorly resourced languages. On in
# NLLB_STAGES (default A, B1, mixed, final; the preflight mirrors stage A because it is A's first 49 steps), OFF in B2 / B3 (b512 / b1024 windows are 14-28 sentences long: a pseudo-reference per sentence of a long window
# has not been validated, and the NLLB sentence-level decoding of long windows is where it fails). The language list is read from `nllb_languages.json` (default configs/clean_rerun/nllb_languages.json), written by the
# language-rule step on H_sel after A0 exists; the committed file is a PROVISIONAL placeholder. Weights: nllb_weight_default / nllb_weights below, per-language values from the calibration file.
NLLB_STAGES_DEFAULT = ["A", "B1", "mixed", "final"]
NLLB_STAGES: List[str] = list(NLLB_STAGES_DEFAULT)
NLLB_LANG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "clean_rerun", "nllb_languages.json")


def nllb_on(stage: str) -> bool:
    return stage in NLLB_STAGES or (stage == "preflight" and "A" in NLLB_STAGES)


def load_nllb_languages(spec: bool = False, path: Optional[str] = None) -> Dict[str, Any]:
    """{"languages": [...], "provisional": bool, "weights": {...}, "source": str} from nllb_languages.json; the specialist arm keeps only its own 20 languages."""
    d = json.load(open(path or NLLB_LANG_FILE, encoding="utf-8"))
    langs = [str(x) for x in d.get("languages", [])]
    if len(set(langs)) != len(langs):
        raise ValueError(f"{path or NLLB_LANG_FILE}: duplicate languages")
    if spec:
        langs = [l for l in langs if l in SPECIALIST_LANGS]
    return {"languages": langs, "provisional": bool(d.get("provisional", True)), "weights": {str(k): float(v) for k, v in (d.get("weights") or {}).items()}, "source": str(d.get("source", ""))}


PLANNED_WINDDOWN_LAST = 30        # final run: the trainer starts the planned lr decay itself, N steps before the (possibly extended) cap
OPTIMIZER = "muon"                # Amendment 4: Muon (2-D decoder matrices) + AdamW (embeddings / norms), update RMS matched to AdamW so the RL learning rates below stay as they are; "adamw" = the old optimizer
HUB_ORG = os.environ['HF_ORG']          # all models are pushed here, PRIVATE (owner 2026-10-29); the old <HF_ORG_LEGACY> repos are not used
HUB_NAME = "gemma-mt-clean-rl"    # repo = <org>/<name>[-spec]-<stage lower case>: gemma-mt-clean-rl-a, -b1, -b2, -b3, -mixed, -final (specialist copies: gemma-mt-clean-rl-spec-...)


def r5(x: float) -> int:
    return int(round(x / 5.0)) * 5


def stage_es(stage: str, spec: bool, steps: int) -> Dict[str, Any]:
    """Early-stop / extension numbers of a stage. Specialist arm: the caps are scaled by SPECIALIST_SCALE, so patience, min_steps, the extension block and the planned wind-down are scaled the same
    way (floors: patience and min_steps >= 30 = 3 smoothing blocks; extension block >= 5; extension > planned wind-down); an early stop whose min_steps is not below the cap is inert but the tracker still
    feeds the extension rule."""
    S = STAGES[stage]
    if S["es"] is None:
        return dict(on=False, patience=60, min_steps=80, extend_on=False, extend_steps=0, planned=0)
    pat, mn = S["es"]
    ext = S["extend"]
    planned = PLANNED_WINDDOWN_LAST if stage == "final" else 0
    if spec:
        sc = SPECIALIST_SCALE
        pat, mn = max(30, r5(pat * sc)), max(30, r5(mn * sc))
        ext = max(5, r5(ext * sc))
        planned = max(10, r5(planned * sc)) if planned else 0
    if planned:
        ext = max(ext, planned + 5)
    return dict(on=True, patience=pat, min_steps=mn, extend_on=True, extend_steps=ext, planned=planned)
SPECIALIST_SCALE = 0.4367   # = (2.5 / 114.5) / (1 / 20): per-language prompts of a weak language of the 77-language run reproduced with 20 equally weighted languages
BIN_PPS = {"b64": 64, "b256": 48, "b512": 42, "b1024": 42, "mixed": 42}
BIN_MAX_NEW = {"b64": 384, "b256": 1280, "b512": 2304, "b1024": 3584, "mixed": 3584}
SCI_BIN_WEIGHTS = {"b64": 0.11, "b256": 0.22, "b512": 0.32, "b1024": 0.35}   # b1024 keeps its OLD absolute share 0.35; the b2048 mass (0.35) goes to the shorter bins in the old proportions 1:2:3
GEN_BIN_WEIGHTS = {"b64": 0.25, "b256": 0.25, "b512": 0.25, "b1024": 0.25}


# ------------------------------------------------------------------------------------------------ yaml emission (no PyYAML needed to WRITE; PyYAML reads the result back in the validator)
def fv(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        s = repr(v)
        if "e" in s and "." not in s.split("e")[0]:
            m, e = s.split("e")
            s = f"{m}.0e{e}"
        return s
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, str):
        if v == "" or any(c in v for c in ":#{}[],&*!|>'\"%@`") or v.strip() != v or v.lower() in ("null", "true", "false", "yes", "no", "on", "off", "~"):
            return json.dumps(v)
        return v
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fv(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {fv(x)}" for k, x in v.items()) + "}"
    raise TypeError(type(v))


class Y:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def raw(self, text: str = "") -> None:
        self.lines.append(text)

    def c(self, text: str, indent: int = 0) -> None:
        self.lines.append(" " * indent + "# " + text)

    def kv(self, key: str, value: Any, comment: str = "", indent: int = 0) -> None:
        line = " " * indent + f"{key}: {fv(value)}"
        self.lines.append(line + (f"  # {comment}" if comment else ""))

    def block(self, key: str, d: Dict[str, Any], comment: str = "", indent: int = 0) -> None:
        self.lines.append(" " * indent + f"{key}:" + (f"  # {comment}" if comment else ""))
        for k, v in d.items():
            self.lines.append(" " * (indent + 2) + f"{k}: {fv(v)}")

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


# ------------------------------------------------------------------------------------------------ paths
def paths(root: str) -> Dict[str, str]:
    return dict(
        root=root,
        prompts_v3=f"{root}/prompts/clean_v3",
        prompts_sci=f"{root}/prompts/clean_sci_v2",
        sft_model=f"{root}/models/sft_final",          # the SFT-final of the arm that wins the pre-registered rule (launcher: SFT_MODEL)
        sci_model=f"{root}/models/sci_sft_start",       # the science-stage model chosen by the S-set guardrails (launcher: SCI_MODEL)
        runs=f"{root}/runs",
        cal=f"{root}/assets/calibration",
        lex=f"{root}/assets/lexicon",
        dev=f"{root}/assets/dev",
        nllb=f"{root}/assets/pseudorefs_reward_only_nc_v1",
        venv_py="/workspace/venv_gemma/bin/python",
        venv_vllm="/workspace/venv_vllm/bin/python",
        pylibs="/workspace/sft_build/pylibs",
        pylibs_tf4="/workspace/sft_build/pylibs_tf4",
        lid="/workspace/sft_build/hf/hub/models--cis-lmu--glotlid/snapshots/85cd6716494360367b75f642b5bc78667605d0b4/model_v3.bin",
    )


def stage_model(stage: str, P: Dict[str, str]) -> str:
    return {"preflight": P["sft_model"], "A": P["sft_model"], "B1": f"{P['runs']}/A/final", "B2": f"{P['runs']}/B1/final",
            "B3": f"{P['runs']}/B2/final", "mixed": f"{P['runs']}/B3/final", "final": P["sci_model"]}[stage]


def reward_block(stage: str, P: Dict[str, str], spec: bool = False) -> List[Tuple[str, Any, str]]:
    fin = stage == "final"
    nl = load_nllb_languages(spec)
    nllb = nllb_on(stage)
    cal = P["cal"]
    r: List[Tuple[str, Any, str]] = [
        ("preset", "repo", "repo: [-2, +2.25]"),
        ("mode", "calibrated", ""),
        ("calibration_path", f"{cal}/shaping_clean_v1.json", "REFIT on T2 only (H_cal, SSA-MTE train, IbomMT train); produced by the calibration agents; replaces shaping_flores_dev_v8.json"),
        ("fail_floor", 0.0, "D-32"),
        ("neg_scale", 2.0, ""),
        ("penalty", None, "null = floor of the range"),
        ("prefix_policy", "require", ""),
        ("lid_enabled", True, "D-37 GlotLID guard (thresholds re-measured on H_cal inside shaping_clean_v1.json `lid`)"),
        ("lid_min_chars", 15, ""),
        ("lid_model_path", P["lid"], ""),
        ("lid_disable", [], ""),
        ("consensus", True, "D-38 group consensus, ALL stages (owner 2026-10-28: every reward component stays, including consensus)"),
        ("consensus_min_valid", 3, ""),
        ("consensus_skip_value", 0.0, ""),
        ("content_mode_default", "qe", ""),
        ("content_modes", {}, ""),
        ("consensus_beta", {}, ""),
        ("consensus_beta_default", 0.5, ""),
        ("selfrep_guard", "consensus", ""),
        ("min_len_ratio_uncalibrated", 0.4, ""),
        ("max_comet_tokens", 510, ""),
        ("nfc", True, ""),
        ("chunking", "sentences", "Gale-Church sentence alignment, as in 2c"),
        ("omission_detector", True, ""),
        ("omission_cost", None, "language's own value from the calibration file"),
        ("omission_short_frac", 0.7, ""),
        ("omission_short_min", 0.15, ""),
        ("diacritic_penalty", 1.0, ""),
        ("chunk_agg", "source_length", ""),
        ("extra_output_penalty", True, ""),
        ("length_penalty", 1.0, ""),
        ("word_ratio_penalty", 0.5, "2026-10-21; calibration `word_ratio`"),
        ("word_ratio_slack", 0.15, ""),
        ("science_anchor_penalty", 0.5 if fin else 0.0, "D-52, science prompts only; 0 in the general stages (they carry no science prompts)"),
    ]
    if fin:
        r += [("science_anchor_path", f"{cal}/science_anchor_clean.json", "REFIT on the clean AfriScience-MT train set (data S3); the old science_anchor_v1.json was fitted on test splits"),
              ("science_anchor_domains", ["science"], "")]
    r += [
        ("anchors_enabled", True, "D-56 constraint penalties"),
        ("anchors_calibration_path", f"{cal}/anchors_clean.json", "REFIT on T2 (anchors v1-v3 design, lexicon blocks); replaces anchors_v2.json"),
        ("anchors_modes", ["consensus", "blend"], ""),
        ("anchors_off_languages", [], ""),
        ("anchors_entity_weight", 0.5, ""),
        ("anchors_repeat_weight", 1.0, ""),
        ("anchors_boiler_weight", 1.0, ""),
        ("anchors_lexicon_weight", 1.0, ""),
        ("anchors_lexicon_per_language", True, ""),
        ("anchors_lexicon_weight_cap", 0.25, ""),
        ("anchors_lexicon_guard", True, ""),
        ("anchors_lexicon_dir", P["lex"], "rebuilt from clean SFT pairs, validated on H_cal"),
        ("anchors_v2_enabled", True, "D-61"),
        ("anchors_v2_modes", ["qe", "blend", "consensus"], ""),
        ("anchors_v2_off_languages", [], ""),
        ("anchors_entity_mode", "soft", ""),
        ("anchors_copy_scale", 2.0, ""),
        ("anchors_copy_mode", "cap", ""),
        ("anchors_func_weight", 0.3, ""),
        ("anchors_pad_weight", 1.0, ""),
        ("anchors_stuff_gate", 0.9, ""),
        ("nllb_signal", nllb, "D-66 NLLB-3.3B pseudo-reference chrF++ signal: ON in " + "/".join(NLLB_STAGES) + " (owner 2026-10-29; off in B2 / B3 = long bins), reward only, CC-BY-NC output, disclose in the paper"),
        ("nllb_pseudoref_dir", P["nllb"], "pseudo-references of the clean prompt pools (prompts that match a protected set are dropped); NC licence"),
        ("nllb_calibration_path", f"{cal}/nllb_reward_calibration_clean.json", "language rule evaluated on H_sel, NOT devtest"),
        ("nllb_modes", ["consensus", "blend"], ""),
        ("nllb_languages", nl["languages"] if nllb else [], ("PROVISIONAL placeholder list (configs/clean_rerun/nllb_languages.json): replace it with the output of the H_sel language rule after A0 exists" if nl["provisional"] else "from nllb_languages.json (H_sel language rule)") + "; weights from the calibration file" if nllb else "off in this stage"),
        ("nllb_weight_default", 0.35, ""),
        ("nllb_weight_max", 0.5, ""),
        ("nllb_weights", {**{"kon": 0.0}, **nl["weights"]}, "per-language overrides (nllb_languages.json `weights`); everything else from the calibration file / nllb_weight_default"),
        ("nllb_min_teacher_delta", 2.0, ""),
        ("nllb_min_auc", 0.8, ""),
        ("nllb_forced", False, ""),
        ("longdoc_v2", True, "D-60 long-document limits (only sources with >= longdoc_min_src_words words)"),
        ("longdoc_calibration_path", f"{cal}/longdoc_clean.json", "REFIT without AfriLongDoc / SmolDoc; the old longdoc_v2.json was fitted on forbidden NLLB references"),
        ("longdoc_min_src_words", 150, ""),
        ("longdoc_repetition", True, ""),
        ("longdoc_align", "anchored", ""),
        ("longdoc_lexicon", False, ""),
        ("longdoc_agg", "cvar_excess", ""),
        ("longdoc_agg_alpha", 1.0, ""),
        ("longdoc_agg_q", 0.25, ""),
        ("longdoc_coverage_kappa", 0.5, ""),
        ("longdoc_pool_severity", 0.5, ""),
        ("anchors_bigram_path", f"{cal}/anchors_clean_bigrams.npz", ""),
        ("anchors_lid_prob_weight", 0.0, ""),
        ("anchors_buffer_capacity", 600, ""),
        ("anchors_buffer_per_prompt", 4, ""),
        ("anchors_buffer_min_outputs", 150, ""),
        ("anchors_buffer_min_prompts", 30, ""),
        ("qe_ensemble", True, "D-55 AfriCOMET-QE + SSA-COMET-MTL"),
        ("qe_ensemble_delta", 1.0, ""),
        ("qe_ensemble_reliability", "ensemble", ""),
        ("lexical_penalty", 1.0, ""),
        ("reliability_from", "same_source", ""),
        ("eos_discipline", True, "D-26"),
        ("eos_truncated_penalty", None, ""),
        ("eos_post_end_penalty", None, ""),
        ("eos_dangling_penalty", None, ""),
    ]
    return r


# ------------------------------------------------------------------------------------------------ render one stage
def hub_repo(stage: str, spec: bool) -> Optional[str]:
    return None if stage == "preflight" else f"{HUB_ORG}/{HUB_NAME}{'-spec' if spec else ''}-{stage.lower()}"


def render(stage: str, arm: str, P: Dict[str, str], steps_override: Optional[Dict[str, int]] = None) -> str:
    S = STAGES[stage]
    fin = stage == "final"
    pre = stage == "preflight"
    spec = arm == "specialist"
    steps = int(S["steps"])
    if spec and stage != "preflight":
        steps = max(10, int(round(steps * SPECIALIST_SCALE / 5.0)) * 5) if steps >= 50 else max(10, int(round(steps * SPECIALIST_SCALE)))
    if steps_override and stage in steps_override:
        steps = int(steps_override[stage])
    b = S["bin"]
    langs = SPECIALIST_LANGS if spec else None
    lw = {k: WEAK_WEIGHT for k in (SPECIALIST_LANGS + OWNER_PRIORITY if spec else WEIGHTED_LANGS)}
    prompts_path = (P["prompts_sci"] if fin else P["prompts_v3"]) + ("_specialist" if spec else "")
    out_dir = f"{P['runs']}/{'preflight' if pre else stage}"
    lr, warm = (1.0e-6, 20) if fin else (2.0e-6, 10)
    kl_thr = 0.25 if fin else 0.15
    kl_coef = 0.1 if fin else 0.05
    y = Y()
    y.c(f"CLEAN RERUN, RL stage {'F (final run)' if fin else ('0 (preflight shakedown)' if pre else stage)}, arm {'S1/S2 specialist (20 languages, dose-matched steps)' if spec else 'A0 (77 languages)'}. GENERATED by scripts/clean_rerun_make_configs.py (agent C1, 2026-10-28): edit the generator, not this file.")
    y.c(("Evidence and costs for every number: reports/clean_rl_stage_plan_2026-10-28.md. Pre-registration: reports/preregistration_clean_rerun_2026-10-28.md (rule sha256 7b86fc11...c0f5, Amendments 1 and 2: the RL bins are capped at 1,024 SOURCE tokens, b2048 is dropped)."))
    if pre:
        y.c("PREFLIGHT SHAKEDOWN: the first 49 steps of stage A (same recipe, greedy member on), to check KL, grad norm, the greedy loop rate and the step time BEFORE the long run (AB2 arm F: 49 b64 steps took about 16 min). Nothing is pushed.")
    elif stage == "A":
        y.c("Stage A: b64 only, from the SFT-final of the arm that won the pre-registered rule. HARD CAP 400 steps (409,600 completions = the old dev-saturation point of ~300 k completions x 1.26 for the 25-language x2.5 list, +8 %); the plateau stop ends earlier.")
    elif stage in ("B1", "B2", "B3", "mixed"):
        y.c(f"Stage {stage}: {'mixed bins b64..b1024 (uniform per prompt)' if stage == 'mixed' else b + ' only'}; RESTART from the previous stage's final/ model with a FRESH reference (= model_name), fresh optimizer, warm-up 10, exactly as phase 2c restarted from the 2b stage-b64 model.")
    else:
        y.c("Final run: science 70 % / general 30 % mixed bins from step 0 (b2048 dropped), NLLB signal on, KL reference = the START model (the science-stage model), planned WINDDOWN of the last 30 steps written by the launcher.")
    y.c("SPDX-License-Identifier: AGPL-3.0-or-later")
    y.kv("model_name", stage_model(stage, P), "KL reference AND policy start; the launcher overrides it with key=value (HANDOFF=final|best)")
    y.kv("tokenizer_name", None)
    y.kv("prompts_path", prompts_path, ("ONE pool directory (parquet per language) of the clean RL prompt dataset (a private " + os.environ['HF_ORG'] + (" dataset repo; no " + os.environ['HF_ORG_LEGACY'] + " repo is read); never the repo root")) + ("; sci_v2 carries the `domain` column (science / general)" if fin else "; v3 has no domain column (general only)"))
    y.kv("eval_prompts_path", None, "null: a hash-selected hold-out of `eval_prompts` prompts (the same hold-out in every stage: same pool, same seed)")
    y.kv("out_dir", out_dir)
    y.raw()
    y.kv("attn_implementation", "sdpa")
    y.kv("param_dtype", "float32")
    y.kv("ref_dtype", "float32")
    y.kv("autocast_dtype", "bfloat16")
    y.raw()
    y.kv("steps", steps, "HARD CAP. The plateau stop / WINDDOWN can only end the stage earlier" + ("; specialist: 0.4367 x the 77-language cap, rounded to 5" if spec and not pre else ""))
    y.kv("prompts_per_step", S["pps"], "fallback; the per-bin table below decides")
    y.kv("group_size", 16, "15 sampled + 1 greedy (D-67 form); 16 is the 2c group size")
    y.kv("greedy_stats", "sampled", "D-67: baseline / std over the SAMPLED members only")
    y.kv("greedy_adv_cap", 3.0, "|advantage of the greedy member| <= 3 before the scale")
    y.kv("greedy_adv_scale", 0.3, "D-59/D-67 measured: 1.0 -> grad norm 10-17 (clip crushes everything else), 0.3 -> 0.60")
    y.kv("greedy_in_group", True, "owner 2026-10-28: the GREEDY completion is scored in EVERY stage (it is the one that loops at inference). NOTE: the only divergence of the record (2b) had a greedy member, but also lr 3e-6, no std division and no guard")
    y.kv("greedy_share_warn", 0.35, "warn when the greedy members carry more than 35 % of the token-weighted |advantage| (uniform: 6 %)")
    y.kv("micro_batch_seqs", 320, "the launcher's OOM ladder lowers this (and micro_batch_tokens)")
    y.kv("num_updates_per_batch", 1)
    y.kv("temperature", 1.0)
    y.kv("top_p", 1.0)
    y.kv("top_k", 0)
    y.kv("allow_truncated_sampling", False)
    y.c("---- rollout sampler (D-28): vLLM, weights pushed after every optimizer step")
    y.kv("sampler", "vllm")
    y.kv("vllm_mode", "worker")
    y.kv("vllm_python", P["venv_vllm"])
    y.kv("vllm_gpu_memory_utilization", 0.2)
    y.kv("vllm_kv_cache_gib", None)
    y.kv("vllm_max_model_len", 10240 if fin else 8192, "prompt + max_new_tokens; the final run also decodes whole AfriScience dev papers (7,168 new tokens) in the document monitor" if fin else "b1024: ~1,300 prompt + 3,584 new (b2048 is gone: 8,192 is enough)")
    y.kv("vllm_max_num_seqs", 512)
    y.kv("vllm_max_num_batched_tokens", None)
    y.kv("vllm_enforce_eager", False)
    y.kv("vllm_sleep", False)
    y.kv("vllm_attention_backend", None)
    y.kv("vllm_sync_dir", "/dev/shm")
    y.kv("vllm_startup_timeout", 1800)
    y.kv("vllm_request_timeout", 3600)
    y.kv("vllm_check_every", 25)
    y.kv("vllm_logp_gap_tol", 0.25)
    y.kv("max_new_tokens", 1024, "fallback; max_new_tokens_by_bin decides")
    y.kv("max_src_tokens", 1024, "SOURCE-side cap (Amendment 2): AfriCOMET-encoder tokens of the English window; b2048 rows are dropped by this")
    y.kv("length_bins", True)
    if fin:
        y.block("curriculum_steps", {"b64": 0}, "a staged bin with 0 steps keeps the schedule in curriculum mode, i.e. MIXED from step 0 (domain -> bin -> language -> prompt)")
    elif stage == "mixed":
        y.block("curriculum_steps", {"b64": 0}, "no single-bin stage: MIXED from step 0 (every prompt draws its bin uniformly over b64 b256 b512 b1024)")
    else:
        y.block("curriculum_steps", {b: steps}, "one single-bin stage of `steps` steps (no mixed steps)")
    y.block("prompts_per_step_by_bin", BIN_PPS, "2c values = the MAXIMUM: nothing in the record shows a gain from larger batches (phase 2 used 2x these with group 8: same dev gain per completion)")
    y.block("max_new_tokens_by_bin", BIN_MAX_NEW, "Gemma tokens; unchanged from 2c (b2048 and its 7,168 cap are gone)")
    y.kv("micro_batch_tokens", 49152)
    y.kv("language_alpha", 0.0, "draw weight ~ pool_size ** 0 = uniform over languages x language_weights")
    if spec:
        y.kv("language_weights", lw, "specialist arm: its 20 languages (all weak-list members) at the same x2.5, i.e. equal relative weights as before; the four owner-priority languages efi tiv bin ish are listed as in the A0 yamls but have no rows in the specialist pool (no effect)")
    else:
        y.kv("language_weights", lw, "the 25 languages of the DEV-DEFINED weak list x2.5 (reports/weak_language_selection_2026-10-26.md; rule sha256 4472d711...6ab7) PLUS the 4 OWNER-PRIORITY languages efi tiv bin ish x2.5 (owner 2026-10-29, chosen on need, not on test data). Everything else 1.0, including the 27 unmeasured languages (the floor protects them)")
    if fin:
        y.kv("language_floor_steps", 20, "D-59 floor: every language >= 2 prompts in every aligned block of 20 steps (needs the domain mixture, i.e. the sci_v2 pool)")
        y.kv("language_floor_prompts", 2)
    else:
        y.c("language floor: NOT available here (it needs domain_weights, i.e. a `domain` column, which the v3 pool lacks). Not needed: with 42-64 prompts per step a weight-1 language gets 7-11 prompts per 20 steps (P[< 2] < 0.1 %).")
    y.kv("run_controls", True, "<out_dir>/WINDDOWN and <out_dir>/LR_SCALE")
    if fin:
        y.kv("domain_weights", {"science": 0.7, "general": 0.3}, "owner / D-52")
        y.block("domain_bin_weights", {"science": SCI_BIN_WEIGHTS, "general": GEN_BIN_WEIGHTS},
                "science: b1024 keeps its old ABSOLUTE share 0.35 (old b64 .05 b256 .10 b512 .15 b1024 .35 b2048 .35); the b2048 mass goes to the shorter bins in the old proportions 1:2:3. general: uniform")
    y.raw()
    y.kv("learning_rate", lr, "2c / AB2 value" if not fin else "D-59 lower end of the owner's range; fresh Adam state here (the old final run carried the 2c moments, which made the same lr ~3x faster)")
    y.kv("kl_guard_threshold", kl_thr, "median KL of the last 5 steps above this halves the lr (floor 1/8, once per 10 steps)" + ("; 0.25 in the final run: reference = start model, kl_coef 0.1 puts the healthy KL at 0.10-0.14 (old final: 250 of 401 steps ran at 25-75 % lr with 0.15)" if fin else "; 3-7x the healthy KL of b64..mixed (0.02-0.06)"))
    y.kv("kl_guard_factor", 0.5)
    y.kv("kl_guard_min_scale", 0.125)
    y.kv("kl_guard_window", 5)
    y.kv("kl_guard_cooldown", 10)
    restore = stage in ("preflight", "A")
    y.kv("kl_guard_restore", restore, ("true: a crash-restart of the same run (or A continuing the preflight) keeps the guard state" if restore else "FALSE at the deliberate fresh-reference restart (coordinator/T2); a crash-restart of this stage then re-arms scale 1.0 - the launcher logs it"))
    y.kv("kl_guard_recover", fin, "T2: after a cut, when the median KL of the last 20 steps is below 0.5 x threshold the scale is multiplied by 1.25 (never above 1.0); on in the FINAL run only" if fin else "off in the stages (T2 recommendation); on in the final run")
    y.kv("kl_guard_recover_factor", 1.25)
    y.kv("kl_guard_recover_below", 0.5)
    y.kv("kl_guard_recover_window", 20)
    pm = S["plateau_min"]
    if pm is not None and spec and stage != "preflight":
        pm = max(100, pm)
    y.kv("plateau_stop", bool(pm) and steps > int(pm), ("T2: training-reward slope CI over the last 100 steps contains 0, finished >= 0.99, no guard event in the window -> WINDDOWN 10, then a normal finish" if pm else "off: the hard cap (%d steps) is already shorter than / equal to the 100-step window or the evidence-based plateau length" % steps))
    y.kv("plateau_stop_min_steps", int(pm) if pm else 150, "validator needs >= window; inert when it is >= steps")
    y.kv("plateau_stop_window", 100)
    y.kv("plateau_stop_block", 20)
    y.kv("plateau_stop_min_finished", 0.99)
    y.kv("plateau_stop_winddown", 10)
    y.kv("plateau_stop_min_lr_scale", 0.5)
    y.kv("plateau_stop_bootstrap", 2000)
    y.kv("plateau_stop_min_steps_by_bin", {})
    y.kv("plateau_stop_ci_high_max", 0.06, "power guard (T2 recommendation 0.06): the window's slope CI must also exclude a gain above 0.06 reward per 100 steps (2c b256 averaged 0.022 per 100 steps)")
    es = stage_es(stage, spec, steps)
    y.c("---- patience-based early stop on EVERY stage (Amendment 4): no new best of the smoothed (10-step) training reward, finished rate or greedy loop rate for `patience` steps after `min_steps` -> WINDDOWN 10 (same mechanism as the plateau stop; whichever fires first). Never inside a window with a guard cut, a changed lr, or a finished rate < 0.99")
    y.kv("early_stop", es["on"], "off in the preflight (49 steps)" if not es["on"] else "values checked by replay on the old 2b / 2c / final2 series and a Monte-Carlo false-stop table (T2 hand-off 2026-10-29)")
    y.kv("early_stop_patience", es["patience"])
    y.kv("early_stop_min_steps", es["min_steps"], "inert when >= steps (the tracker still feeds the extension rule)")
    y.kv("early_stop_block", 10, "smoothing block; the 10-step block mean of the 2c training reward has sd 0.02-0.03")
    y.kv("early_stop_eps_reward", 0.02, "a new best must beat the old by this much (about one block sd)")
    y.kv("early_stop_eps_finished", 0.002)
    y.kv("early_stop_eps_loop", 0.005, "greedy loop rate: lower is better")
    y.kv("early_stop_min_finished", 0.99)
    y.kv("early_stop_min_lr_scale", 0.5)
    y.kv("early_stop_winddown", 10)
    y.c("---- surgical extension: at the cap (final run: N steps before it, where its planned wind-down would start), if the improvement criterion above fired within the last `extend_recent` steps, the cap is raised by `extend_steps` (about 25 % of the base cap), repeated until +50 % at most. Every decision is logged (extensions.jsonl, summary.json)")
    y.kv("extend_if_improving", es["extend_on"])
    y.kv("extend_steps", es["extend_steps"], "extend_steps > planned_winddown_last is required" if es["planned"] else "")
    y.kv("extend_max_frac", 0.5, "hard maximum +50 % of the base cap")
    y.kv("extend_recent", 25)
    y.kv("planned_winddown_last", es["planned"], "final run: the trainer starts the planned lr decay itself (the launcher's external watcher is then not started), so an extension moves it" if es["planned"] else "")
    y.c("---- optimizer (Amendment 4): fresh at the start of every stage (every stage has its own out_dir; the final run carries no Adam moments)")
    y.kv("optimizer", OPTIMIZER, "muon: Muon on the 2-D decoder matrices + AdamW on embeddings / norms / 1-D, update RMS matched to AdamW (match_rms_adamw), so the learning rate below is the AdamW-equivalent one and was NOT re-tuned" if OPTIMIZER == "muon" else "adamw")
    y.kv("muon_momentum", 0.95)
    y.kv("muon_ns_steps", 5)
    y.kv("warmup_steps", warm)
    y.kv("weight_decay", 0.0)
    y.kv("grad_clip", 1.0)
    y.kv("advantage_norm", "mean_std", "std-normalised, as 2c (2b dropped the division and diverged)")
    y.c("---- D-65 language draw weights as a run control; the adaptive updater stays OFF (it never changed a weight in the record, and the dose-response is not identifiable: R13/D-65)")
    y.kv("lang_weights_file", True, "<out_dir>/LANG_WEIGHTS.json (operator override, read every step)")
    y.kv("lang_stats_log", True, "lang/{n,reward,radj,fail}/<iso> per step: free per-language training statistics")
    y.kv("lang_weights_log_every", 50)
    y.kv("adaptive_lang_weights", False)
    y.kv("hard_fail_mask", "off")
    y.kv("std_eps", 1.0e-4)
    y.kv("kl_coef", kl_coef, "final run: reference = start model, 0.1 as executed in the old final run (its yaml said 0.05, the CLI override 0.1)" if fin else "")
    y.kv("kl_estimator", "k3c", "k3 with the log-ratio clamped at 5 (the 2c outlier fix)")
    y.kv("clip_eps", None)
    y.kv("loss_agg", "token")
    y.kv("loss_chunk_tokens", 2048)
    y.kv("gradient_checkpointing", False)
    y.raw()
    y.lines.append("reward:")
    for k, v, c in reward_block(stage, P, spec):
        y.kv(k, v, c, indent=2)
    y.raw()
    y.lines.append("scorer:")
    y.kv("type", "worker", "QE models run in a CHILD process with transformers 4.x (D-33)", indent=2)
    y.kv("python", P["venv_py"], "", indent=2)
    y.kv("pythonpath", f"{P['pylibs_tf4']}:{P['pylibs']}", "", indent=2)
    y.kv("model_id", "masakhane/africomet-qe-stl-1.1", "", indent=2)
    y.kv("ensemble_model_id", "McGill-NLP/ssa-comet-mtl", "D-55: one worker with both checkpoints", indent=2)
    y.kv("batch_size", 128, "the pair peaks at 6.7 GB at 128", indent=2)
    y.raw()
    y.kv("eval_every", 25, "hold-out reward + greedy finished / loop rate every 25 steps (also at the last step)")
    y.kv("eval_min_per_bin", 16, "D-45: >= 16 hold-out prompts of every bin, so the per-bin greedy loop rate exists in every stage")
    y.kv("eval_prompts", 192 if fin else 128)
    if fin:
        y.kv("benchmark_eval_path", f"{P['dev']}/bench_afrisci_dev_sent.jsonl", "AfriScience-MT DEV sentences (843 per language x 6 languages: amh hau lug nso yor zul); an S set; the test split is never used during training")
        y.kv("benchmark_eval_per_language", 200)
        y.kv("benchmark_max_new_tokens", 512)
    else:
        y.kv("benchmark_eval_path", f"{P['dev']}/bench_dev_hsel.jsonl", "S set = FLORES+ H_sel articles (296 ids, sha256 036295b9...df37d) + MAFAND/SSA-MTE dev where licensed; REBUILT on H_sel (the old 760-sentence set was cut from contaminated dev)")
        y.kv("benchmark_eval_per_language", 40)
        y.kv("benchmark_max_new_tokens", 256)
    y.kv("benchmark_eval_every", 25 if pre else 50, "dev chrF++ every 50 steps (sd of the 19-language macro 0.09-0.10); the preflight evaluates at 25 and 49")
    y.kv("select_by", "benchmark_chrf")
    y.kv("select_min_gain", 0.2 if fin else 0.1)
    y.kv("select_min_finished_rate", 0.95)
    if fin:
        y.kv("resume_reset_selection", True)
        y.lines.append("benchmark_aux:  # monitor-only dev sets (never used for selection)")
        y.lines.append(f"  - {{name: afrisci_dev_doc, path: {P['dev']}/bench_afrisci_dev_doc.jsonl, per_language: 25, max_new_tokens: 7168, every: 100}}")
        y.lines.append(f"  - {{name: general_dev, path: {P['dev']}/bench_dev_hsel.jsonl, per_language: 40, max_new_tokens: 256}}")
        y.lines.append(f"  - {{name: weak_dev, path: {P['dev']}/bench_weak_hsel.jsonl, per_language: 60, max_new_tokens: 256, batch_size: 64}}")
    else:
        y.lines.append("benchmark_aux:  # monitor-only")
        y.lines.append(f"  - {{name: weak_dev, path: {P['dev']}/bench_weak_hsel.jsonl, per_language: 60, max_new_tokens: 256, batch_size: 64}}")
    y.kv("save_every", 50)
    y.kv("log_every", 1)
    y.kv("check_sampler_logprobs", False)
    y.kv("seed", 1234)
    y.kv("extra_libs", [P["pylibs"]], "sacrebleu for the dev chrF++")
    y.kv("hub_repo_id", hub_repo(stage, spec), (os.environ['HF_ORG'] + (", PRIVATE (new names, not the old " + os.environ['HF_ORG_LEGACY'] + " ones); the launcher passes hub_repo_id=$HUB_PREFIX-<stage> (HUB_PREFIX defaults to ")) + f"{HUB_ORG}/{HUB_NAME}{'-spec' if spec else ''}" + "; HUB_PREFIX= empty disables the push)" if hub_repo(stage, spec) else "no push in the preflight")
    y.kv("hub_branch", None)
    y.kv("hub_latest_branch", "latest", "the LATEST checkpoint = the end-of-stage model (the one handed to the next stage)")
    y.kv("hub_latest_every", 0 if pre else 25)
    y.kv("hub_stage_branches", False)
    y.kv("hub_private", True, "explicit: the pusher also re-asserts private on an existing repo, and the trainer refuses hub_private: false unless hub_allow_public: true")
    y.kv("hub_allow_public", False)
    y.kv("eval_vllm_only", True, "Amendment 4: every evaluation generation (hold-out reward, dev benchmark, aux sets) goes through vLLM; no HF generate fallback")
    y.kv("keep_single_checkpoint", False, "False so that <out_dir>/final/ (the last-step HF model, the hand-off to the next stage) and best/ are written")
    y.kv("archive", not pre, "D-41 best-translation archive (feeds the science-stage mixture component A); tag: rows carry the iso and source, LibreTexts prompts are identified by their source text only" if not pre else "")
    y.kv("archive_path", None if pre else f"{out_dir}/archive/archive.sqlite")
    y.kv("archive_min_reward_by_mode", {"qe": 1.0, "blend": 1.0, "consensus": 0.5})
    y.kv("archive_min_consensus_chrf", 50.0)
    y.kv("archive_flush_seconds", 20)
    y.kv("archive_queue_size", 64)
    y.kv("resume", True, "a crash-restart of the same stage resumes from <out_dir>/last (a NEW stage has a new out_dir: fresh optimizer)")
    y.kv("wandb_project", None)
    y.kv("run_name", None)
    return y.text()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["a0", "specialist"], default="a0")
    ap.add_argument("--root", default="/workspace/clean_rerun")
    ap.add_argument("--out", default=None, help="output directory (default configs/clean_rerun or configs/clean_rerun_specialist next to scripts/)")
    ap.add_argument("--print-table", action="store_true")
    ap.add_argument("--optimizer", choices=["muon", "adamw"], default=None, help="override the optimizer written into the yamls (default: OPTIMIZER = muon; the validator expects the default)")
    ap.add_argument("--nllb-stages", default=",".join(NLLB_STAGES_DEFAULT), help="comma list of stages with the NLLB pseudo-reference signal (default A,B1,mixed,final; B2 / B3 are not allowed without --nllb-allow-long)")
    ap.add_argument("--nllb-allow-long", action="store_true", help="allow B2 / B3 in --nllb-stages (long bins: pseudo-reference quality not validated)")
    ap.add_argument("--nllb-languages-file", default=None, help="nllb_languages.json (default configs/clean_rerun/nllb_languages.json)")
    ap.add_argument("--steps", default=None, help='JSON {"A": 560} cap overrides (regenerates with other caps)')
    a = ap.parse_args(argv)
    if a.optimizer:
        globals()["OPTIMIZER"] = a.optimizer
    ns = [x.strip() for x in a.nllb_stages.split(",") if x.strip()]
    bad = [x for x in ns if x not in STAGES or x == "preflight"]
    if bad or (not a.nllb_allow_long and any(x in ("B2", "B3") for x in ns)):
        ap.error(f"--nllb-stages: unknown stage(s) or a long bin without --nllb-allow-long: {bad or [x for x in ns if x in ('B2', 'B3')]}")
    NLLB_STAGES[:] = ns
    if a.nllb_languages_file:
        globals()["NLLB_LANG_FILE"] = a.nllb_languages_file
    nl = load_nllb_languages()
    if ns and nl["provisional"]:
        print(f"WARNING: the NLLB language list {nl['languages']} is PROVISIONAL (placeholder until the H_sel language rule has run after A0); stages with the signal: {ns}")
    here = os.path.dirname(os.path.abspath(__file__))
    out = a.out or os.path.join(here, "..", "configs", "clean_rerun" if a.arm == "a0" else "clean_rerun_specialist")
    out = os.path.abspath(out)
    os.makedirs(out, exist_ok=True)
    P = paths(a.root)
    so = json.loads(a.steps) if a.steps else None
    for st, S in STAGES.items():
        txt = render(st, "specialist" if a.arm == "specialist" else "a0", P, so)
        with open(os.path.join(out, S["file"]), "w") as f:
            f.write(txt)
        if a.print_table:
            import re
            m = re.search(r"^steps: (\d+)", txt, re.M)
            print(f"{S['file']:22s} steps {m.group(1)}")
    print("wrote", len(STAGES), "yaml files to", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
