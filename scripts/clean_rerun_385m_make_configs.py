#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""RL stage configs for the 385M grown model (owner 2026-11-01): configs/clean_rerun_385m/*.yaml.

    python scripts/clean_rerun_385m_make_configs.py [--out configs/clean_rerun_385m] [--root /workspace/clean_rerun]

The stage recipe, caps, reward components and every number of the clean-rerun plan (reports/clean_rl_stage_plan_2026-10-28.md) are UNCHANGED: this script runs the real generator
(scripts/clean_rerun_make_configs.py) into a temp directory and then applies a small, explicit set of overrides line by line, so the yamls stay comparable with configs/clean_rerun
(diff them). Overrides:
  model_name   preflight / A: $ROOT/models/g15_385m_sft_final = the SFT final of the growth-1 run (runs/g1b_s6/last, 39 layers, 385,144,448 parameters, exported to HF format together
               with the RENAMED tokenizer files); final: $ROOT/models/sci_sft_start_385m; B1..mixed: the previous stage's final/ under runs_385m
  tokenizer_name  $ROOT/assets/tokenizer_gemma3_tags (the renamed tokenizer: the 125 language tags read <yor> <hau> ...; same ids as <unusedK>; artifacts/tokenizer_gemma3_tags)
  out_dir, archive_path, runs  $ROOT/runs_385m/... (never collides with a run of the 270M plan)
  reward.calibration_path  calibration/clean_2026-10-28/shaping_clean_v2.json (v1 + the 8 new consensus languages; scripts/build_calibration_clean_v2.py)
  prompts_path  preflight..mixed: $ROOT/prompts/clean_v3_385m (clean_v3 + artifacts/rl_prompts_new/*.parquet for guw lue tll zne lub crs mfe arz; scripts/clean_rerun_385m_prepare_assets.sh);
               final: unchanged (the science pool has a `domain` column the 8 new files lack)
  hub_repo_id  <HF_ORG>/gemma-mt-385m-rl-<stage> (private)
  save_every  20 (owner rule; the current generator writes 50)
  the new reward flags (confidence / fluency / spelling) stay OFF; commented lines show how to switch them on.
OPTIONAL stage F2 ("final_long", science only, b1024 + b2048; owner 2026-11-02): configs/clean_rerun_385m/rl_F2_long.yaml, derived from the rewritten FINAL yaml by make_f2() below. It is written
  next to the five stage yamls but is NOT a stage of the plan: the launcher runs it only with STAGE_PLAN=b256+long (or EXTRA_STAGES=F2); reports/rl_stage_plan_385m_b256_2026-11-02.md section 6.
The SINGLE-TOKEN TAG MODE has no yaml key (RLConfig is strict): it is the environment variable GEMMA_MT_TAG_MODE=single_token, exported by scripts/launch_clean_rerun_385m.sh.
"""
import argparse, os, re, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import clean_rerun_make_configs as gen

B256_MAX_SRC = 256                     # upper edge of the b256 bin (rl/data.py BIN_EDGES b256 = (65, 257)): source-side cap in AfriCOMET-encoder tokens
B256_MAX_MODEL_LEN = 4096              # prompt (<= ~400 Gemma tokens for 256 AfriCOMET tokens + the template) + 1,280 new tokens = ~1,700; 4,096 = 2.4x margin
B256_SCI_BIN_WEIGHTS = {"b64": round(1 / 3, 10), "b256": round(2 / 3, 10)}   # the old science weights b64 .11 : b256 .22 renormalised to sum 1
B256_GEN_BIN_WEIGHTS = {"b64": 0.5, "b256": 0.5}
B256_STAGE_MODEL = {"preflight": "sft", "A": "sft", "B1": "A", "mixed": "B1", "final": "sci"}

# ---- optional stage F2 (final_long): the numbers of the old B3 / b2048 / final configs (configs/clean_rerun/rl_B3_b1024.yaml, configs/rl_rtxpro_phase2c_b2048.yaml, D-45)
F2_FILE = "rl_F2_long.yaml"
F2_STEPS = 70
F2_MAX_SRC = 2048                      # b2048 = AfriCOMET-encoder tokens 1,025..2,048 (rl/data.py BIN_EDGES)
F2_MAX_MODEL_LEN = 10240               # b2048 prompt <= ~2,300 Gemma tokens + 7,168 new tokens = ~9,500 (the old final run's value, also used by eval_afriscience.py)
F2_BIN_WEIGHTS = {"b1024": 0.6, "b2048": 0.4}       # science mass; the AfriScience-MT dev papers are 16 b1024 : 9 b2048 (64 : 36), train 97 : 77
F2_PPS = {"b1024": 42, "b2048": 21, "mixed": 28}    # 2c per-bin maxima for the single bins; mixed 28 x 16 = 448 completions (28 x (0.6 x 1 + 0.4 x 2) = 39 b1024-units, the b1024 stage had 42)
F2_MAX_NEW = {"b64": 384, "b256": 1280, "b512": 2304, "b1024": 3584, "b2048": 7168, "mixed": 7168}      # the old generator's table + D-45 (3.5 x 2,048; ONE cap per mixed step)


def apply_b256_cap(g) -> None:
    """Patch the clean-rerun generator module `g` IN PLACE to the b256-capped stage list (owner 2026-11-02). Idempotent."""
    for st in ("B2", "B3"):
        g.STAGES.pop(st, None)
    for d, new in ((g.BIN_PPS, {"b64": 64, "b256": 48, "mixed": 42}), (g.BIN_MAX_NEW, {"b64": 384, "b256": 1280, "mixed": 1280}),
                   (g.SCI_BIN_WEIGHTS, B256_SCI_BIN_WEIGHTS), (g.GEN_BIN_WEIGHTS, B256_GEN_BIN_WEIGHTS)):
        d.clear(); d.update(new)

    def stage_model(stage, P):
        return {"preflight": P["sft_model"], "A": P["sft_model"], "B1": f"{P['runs']}/A/final", "mixed": f"{P['runs']}/B1/final", "final": P["sci_model"]}[stage]
    g.stage_model = stage_model

FLAGS = """\
  # ---- NEW, OFF (2026-11-01): switch on one at a time AFTER the preflight has been examined (reports/rl_reward_explained_2026-10-19_v3.md sections 32-34)
  # confidence: true          # D-47 gate / trust / cap (confidence_gate, confidence_trust, confidence_cap default true once the master switch is on; confidence_relative true)
  # fluency: true             # char-5gram + word-order penalty (penalty only): fluency_models_dir: /workspace/clean_rerun/assets/fluency_models, fluency_penalty_weight: 0.5, fluency_z_floor: -2.0
  # fluency_spelling_weight: 0.5   # Bible-trained diacritics term, only yor and kik
"""


NEWS_SHARE = 0.3
REF_SHARE = 0.15       # owner 2026-11-03: share of the draws that are REFERENCE prompts (human en->X pairs, domain "ref", files in <prompts_path>/ref/); the rest is split 70 / 30 general / news
REF_WEIGHT = 0.35      # reward.reference_weight: weight of the chrF++-against-the-reference term on those prompts
NEWS_LINE = (f"domain_weights: {{general: {(1 - REF_SHARE) * (1 - NEWS_SHARE):.3f}, news: {(1 - REF_SHARE) * NEWS_SHARE:.3f}, ref: {REF_SHARE:.3f}}}  # owner 2026-11-03: 15 % reference prompts (human en->X pairs with a chrF++ term, "
             f"files in <prompts_path>/ref/), the other 85 % split 70 % FineWeb-Edu / Wikipedia (general) and 30 % news-style English (BBC News, <prompts_path>/news/)")
NEWS_LINE_PREFLIGHT = f"domain_weights: {{general: {1 - NEWS_SHARE:.1f}, news: {NEWS_SHARE:.1f}}}  # owner 2026-11-03: 70 % FineWeb-Edu / Wikipedia (general), 30 % news-style English (BBC News); the news files live in <prompts_path>/news/"   # the preflight has no reference pool
ALL_FEATURES = ("ds", "adv", "log", "guide", "ref", "mbr", "devstop", "devpred", "monitor")      # owner 2026-11-03 additions; `--features` picks a subset (a run that is resumed from a checkpoint can switch them on one by one); default = all
FEATURES = set(ALL_FEATURES)
ADV_LINES = """\
# advantage std floor (owner 2026-11-03): advantage = (r - mean) / max(std, adv_std_floor); only groups with std below the floor (near-flat: jitter in the 3rd decimal of a QE score) are damped, |A| ~ std / floor.
# 0.02 reward units = ~0.005 AfriCOMET-QE units (the shaping slope is ~4 reward units per QE unit) = noise level. The preflight did not log per-group stds: <out_dir>/group_stats.jsonl (log_group_stats) now does,
# scripts/group_std_percentiles.py reads it; re-set the floor at ~the 10th percentile of the group stds at a checkpoint boundary if it differs.
adv_std_floor: 0.02
"""
LOG_LINES = """\
log_group_stats: true  # per-prompt group statistics (prompt, language, bin, domain, std, mean, hard failures, guide) -> <out_dir>/group_stats.jsonl, group_std/p05..p75 in the step record (input of a later prompt replay by learnability)
"""
GUIDE_LINES = """\
# beam-guided group member (owner 2026-11-03): for guide_fraction of the prompts the beam-guide_beams translation of the CURRENT policy (HF generate on the training model) replaces the last sampled member of the group
# (group size unchanged: samples ..., guide, greedy). Scored like any member (guards, penalties, reference term), part of the group mean / std and of the KL term like the greedy member; a hard-failed guide is dropped
# from its group; guide_adv_share_cap limits the guides' share of the token-weighted |advantage| of the step. Ratio fixed at 1 (one update per batch), no importance weight. Logged: guide/... and guide_by_language.json.
guide_fraction: 0.25
guide_beams: 4
guide_adv_share_cap: 0.25
"""
LATER_STAGES = ("B1", "mixed", "final")                   # stage A had run when these two additions were made
MBR_EXTRA_LANGS = ("orm", "lug")                           # added to the weak list (x2.5 language weights) for the MBR guide: the decode study gains (orm +7.3, ewe +4.6, lug +4.3 over beam-4); ewe and tir are in the weak list already
DEV_LINES = """\
# early stop on the DEV metrics (owner 2026-11-03): the dev macro chrF++ (select_by benchmark_chrf) and weak_dev have both not improved by >= early_stop_dev_eps for early_stop_dev_k consecutive DEV evaluations
# (benchmark_eval_every 25: 3 x 25 = 75 steps) while the stage has >= 75 steps and the finished rates are >= 0.98 -> WINDDOWN 10, normal finish, stop_reason dev_plateau. Stage A's flat dev (40.14 @100 .. 40.50 @400)
# never stopped the training-reward trackers. A KL-guard event blocks it only inside the stale window; there is no learning-rate condition.
early_stop_dev: true
early_stop_dev_k: 3
early_stop_dev_eps: 0.15
early_stop_dev_min_steps: 75
early_stop_dev_min_finished: 0.98
early_stop_dev_winddown: 10
early_stop_dev_aux: weak_dev
"""
DEVPRED_LINES = """\
save_dev_predictions: true  # the greedy dev translations of every dev evaluation -> <out_dir>/dev_predictions/step_<N>.jsonl (small; scripts/dev_comet_retro.py scores them offline)
"""
MONITOR_LINES = """\
# held-out AfriCOMET monitor, reward-hacking alarm, combined dev stop, per-language dev history (patch 3, owner 2026-11-03). masakhane/africomet-stl-1.1 is REFERENCE-BASED and not one of the QE checkpoints the policy trains on
# (africomet-qe-stl-1.1, ssa-comet-mtl); it scores the dev translations against the human dev references for the AfriCOMET languages in a separate comet worker that is released after each dev evaluation.
heldout_comet: true
heldout_comet_model: masakhane/africomet-stl-1.1
heldout_comet_batch_size: 64
heldout_comet_keep_loaded: false
heldout_comet_bootstrap: 200
hacking_alarm: true  # hold-out QE slope > 0 over the last k + 1 dev evaluations while BOTH the dev chrF++ macro fell > 0.3 and the held-out AfriCOMET macro fell > 0.005 -> WINDDOWN 10, stop_reason hacking_alarm (continue from best/)
hacking_alarm_k: 3
hacking_alarm_eps_chrf: 0.3
hacking_alarm_eps_comet: 0.005
hacking_alarm_winddown: 10
early_stop_dev_require_comet: true  # a dev_plateau stop also needs the held-out AfriCOMET macro to be flat (< early_stop_dev_eps_comet) over the same k dev evaluations
early_stop_dev_eps_comet: 0.004
lang_stall_log: true  # <out_dir>/dev_by_language.jsonl + dev_stalled/<iso> flags (logging only)
lang_stall_k: 3
lang_stall_eps_chrf: 0.5
lang_stall_eps_comet: 0.01
"""
DS_LINES = """\
# dynamic sampling (DAPO-style, owner 2026-11-03; rl/dynamic_sampling.py): groups whose rewards have no spread (std < dynamic_sampling_min_std) give zero advantage: draw ceil(prompts x oversample) prompts, drop the flat
# groups, refill the deficit with replacement prompts of the same language / domain / bin (at most max_refills extra rounds) and update on the first `prompts` informative groups. reward / finished rate / greedy_*
# / per-language statistics are logged on ALL generated groups; ds/... counters and <out_dir>/dynamic_sampling_by_language.json show what the filter did.
dynamic_sampling: true
dynamic_sampling_oversample: 1.1
dynamic_sampling_max_refills: 2
dynamic_sampling_min_std: 0.01
"""


HUB_BY_STAGE = {   # stage -> (repo, branch); the preflight pushes nothing
    "A": ((os.environ['HF_ORG'] + "/gemma-0.4B-translate-rl"), "stage-a-b64"),
    "B1": ((os.environ['HF_ORG'] + "/gemma-0.4B-translate-rl"), "stage-b1-b256"),
    "mixed": ((os.environ['HF_ORG'] + "/gemma-0.4B-translate-rl"), None),
    "final": ((os.environ['HF_ORG'] + "/gemma-0.4B-science-rl"), None),
}
F2_HUB = ((os.environ['HF_ORG'] + "/gemma-0.4B-science-rl"), "stage-f2-long")


def rewrite(txt: str, stage: str, root: str, P) -> str:
    g, runs, runs385 = f"{root}/prompts/clean_v3", f"{root}/runs/", f"{root}/runs_385m/"
    txt = txt.replace(runs, runs385)
    txt = txt.replace(f"{root}/models/sci_sft_start", f"{root}/models/sci_sft_start_385m")
    txt = txt.replace(f"{root}/models/sft_final", f"{root}/models/g15_385m_sft_final")
    txt = re.sub(r"^tokenizer_name: null.*$", f"tokenizer_name: {root}/assets/tokenizer_gemma3_tags  # RENAMED tokenizer (<yor> ... instead of <unusedK>, same ids); it must also be the tokenizer of the exported model dirs", txt, flags=re.M)
    txt = txt.replace("shaping_clean_v1.json", "shaping_clean_v2.json")
    if stage != "final":
        txt = txt.replace(g, g + "_385m")
    txt = txt.replace("gemma-mt-clean-rl", "gemma-mt-385m-rl")
    # owner 2026-11-03: Hub names. The b64 and b256 translation stages go to ONE repo (branches stage-a-b64 / stage-b1-b256, the mixed stage = main), the science stage to its own repo
    repo, branch = HUB_BY_STAGE.get(stage, (None, None))
    if repo:
        txt = re.sub(r"^hub_repo_id: \S+", f"hub_repo_id: {repo}", txt, flags=re.M)
        txt = re.sub(r"^hub_branch: .*$", f"hub_branch: {branch if branch else 'null'}", txt, flags=re.M)
    txt = re.sub(r"^save_every: \d+", "save_every: 20", txt, flags=re.M)      # owner rule: a resumable checkpoint every 20 steps (the generator now writes 50; the committed 270M yamls say 20)
    # owner 2026-11-01: keep the NLLB pseudo-reference signal ON but with a LOW weight (the human-DA study of 2026-10-23: w = 0.25 helps, 0.35 is flat, 0.5 hurts; the SFT data now contains NLLB-translated text, double pull)
    txt = re.sub(r"^(\s*)nllb_weight_default: [0-9.]+", r"\1nllb_weight_default: 0.25", txt, flags=re.M)
    txt = re.sub(r"^(\s*)nllb_weight_max: [0-9.]+", r"\1nllb_weight_max: 0.35", txt, flags=re.M)
    # ---- RL context capped at b256 (owner 2026-11-02)
    txt = re.sub(r"^max_src_tokens: \d+.*$", f"max_src_tokens: {B256_MAX_SRC}  # b256 CAP (owner 2026-11-02): SOURCE-side cap in AfriCOMET-encoder tokens = the upper edge of the b256 bin; rows of b512 and longer are dropped", txt, flags=re.M)
    txt = re.sub(r"^vllm_max_model_len: \d+.*$", f"vllm_max_model_len: {B256_MAX_MODEL_LEN}  # b256 CAP: prompt (<= ~400 Gemma tokens) + <= 1,280 new tokens = ~1,700, 2.4x margin", txt, flags=re.M)
    txt = re.sub(r"^[ \t]*- \{name: afrisci_dev_doc,.*\n", "", txt, flags=re.M)      # whole-paper document monitor (7,168 new tokens): not needed when production decodes sentence by sentence
    txt = txt.replace("mixed bins b64..b1024 (uniform per prompt)", "mixed bins b64 + b256 (uniform per prompt, equal mass; b256 CAP)")
    txt = txt.replace("(every prompt draws its bin uniformly over b64 b256 b512 b1024)", "(every prompt draws its bin uniformly over the bins present: b64 and b256; b256 CAP)")
    txt = txt.replace("Gemma tokens; unchanged from 2c (b2048 and its 7,168 cap are gone)", "Gemma tokens; b64 / b256 as in 2c, mixed = the b256 value (b256 CAP: b512, b1024, b2048 are gone)")
    txt = txt.replace("science: b1024 keeps its old ABSOLUTE share 0.35 (old b64 .05 b256 .10 b512 .15 b1024 .35 b2048 .35); the b2048 mass goes to the shorter bins in the old proportions 1:2:3. general: uniform",
                      "b256 CAP (owner 2026-11-02): science b64 : b256 = 1 : 2 (the old .11 : .22 renormalised); general: uniform over b64 and b256")
    txt = txt.replace("the RL bins are capped at 1,024 SOURCE tokens, b2048 is dropped", "the 270M plan capped the RL bins at 1,024 SOURCE tokens; THIS 385M plan caps them at 256 (b256 CAP, owner 2026-11-02)")
    txt = txt.replace("(owner 2026-10-29; off in B2 / B3 = long bins)", "(owner 2026-10-29; the long-bin stages B2 / B3 are not run in the b256-capped plan)")
    txt = txt.replace("Final run: science 70 % / general 30 % mixed bins from step 0 (b2048 dropped)", "Final run: science 70 % / general 30 % mixed bins b64 + b256 from step 0 (b512 and longer dropped: b256 CAP)")
    # owner 2026-11-02: the long-document reward (longdoc_v2) is applied to the SHORT bins as well: every source of >= 1 word goes through the longdoc path (repetition severity instead of the v1 whole-text guards,
    # failed chunk on a stuttering sentence, coverage severity, anchored alignment; tail aggregation = plain mean below 8 chunks, i.e. identical to the normal path for one to seven chunks).
    # longdoc_compare_v1 logs, next to the reward, what the v1 guards would have done (longdoc/delta_reward_*, longdoc/v1_*_rate): read it in the preflight.
    txt = re.sub(r"^(\s*)longdoc_min_src_words: \d+.*$", r"\1longdoc_min_src_words: 1  # 385M plan (owner 2026-11-02): longdoc_v2 on EVERY source, also b64 / b256 (was 150 words); see scripts/clean_rerun_385m_make_configs.py", txt, flags=re.M)
    txt = re.sub(r"^(\s*)longdoc_v2: true(.*)$", r"\1longdoc_v2: true\2\n\1longdoc_compare_v1: true  # diagnostic only: logs the longdoc path against the v1 guards / plain mean for the preflight", txt, flags=re.M)
    # owner 2026-11-03: dynamic sampling in every training stage (not the preflight), reference-anchored reward term (rows with a human reference only; no effect where the pool has none)
    if stage != "preflight":
        blocks = [("ds", DS_LINES), ("adv", ADV_LINES), ("log", LOG_LINES), ("guide", GUIDE_LINES)]
        if "guide" in FEATURES and "mbr" in FEATURES and stage in LATER_STAGES:
            weak = list(dict.fromkeys(list(gen.WEIGHTED_LANGS) + list(MBR_EXTRA_LANGS)))
            blocks.append(("mbr", "# MBR guide for the weak languages (owner 2026-11-03): the weak list (the x2.5 language weights) + orm lug; MBR over the group's members + the beam output + guide_mbr_extra_samples extra samples\n"
                                  "# (T 0.7, top-p 0.95, drawn for those prompts only), utility chrF++ (the consensus utility); a winner that is a group member keeps the beam output as the guide; the extra samples are skipped for a step\n"
                                  "# when the running guide overhead exceeds guide_mbr_max_overhead_frac. All other languages use the beam guide.\n"
                                  f"guide_mode_by_language: {{{', '.join(f'{l}: mbr' for l in weak)}}}\nguide_mbr_extra_samples: 16\nguide_mbr_temperature: 0.7\nguide_mbr_top_p: 0.95\nguide_mbr_max_overhead_frac: 0.20\n"))
        if "devstop" in FEATURES and stage in LATER_STAGES:
            blocks.append(("devstop", DEV_LINES))
        if "devpred" in FEATURES and stage in LATER_STAGES:
            blocks.append(("devpred", DEVPRED_LINES))
        if "monitor" in FEATURES and stage in ("mixed", "final"):
            blocks.append(("monitor", MONITOR_LINES))
        extra = "".join(blk for key, blk in blocks if key in FEATURES)
        if "devstop" in FEATURES and stage in LATER_STAGES:
            txt = re.sub(r"^benchmark_eval_every: 50(.*)$", r"benchmark_eval_every: 25  # dev chrF++ every 25 steps (early_stop_dev counts dev evaluations; the sd of the 19-language macro is 0.09-0.10)", txt, count=1, flags=re.M)
            assert "benchmark_eval_every: 25" in txt
        if extra:
            txt = re.sub(r"^(greedy_share_warn: .*\n)", lambda m: m.group(1) + extra, txt, count=1, flags=re.M)
        if "ref" in FEATURES:
            txt = re.sub(r"^(\s*)longdoc_compare_v1: true(.*)$", lambda m: m.group(0) + f"\n{m.group(1)}reference_weight: {REF_WEIGHT}  # owner 2026-11-03: r = (1 - w) r_existing + w consensus_value(chrF++ vs the human reference), on prompts that carry one (domain ref); failures / guards / penalties untouched", txt, count=1, flags=re.M)
            assert "reference_weight:" in txt
    # owner 2026-11-03: NEWS-style English prompts (BBC News from the Hub, pool artifacts/rl_prompts_news -> <pool>/news/<iso>.parquet, `domain` column "news") are mixed into the translation stages:
    # 30 % of the draws, 70 % FineWeb-Edu / Wikipedia (unlabeled rows load as "general", see rl/data.py). The final (science) stage keeps its own science 0.7 / general 0.3 mixture.
    if stage != "final":
        txt = re.sub(r"^(language_weights: .*)$", lambda m: m.group(1) + "\n" + (NEWS_LINE_PREFLIGHT if stage == "preflight" or "ref" not in FEATURES else NEWS_LINE), txt, count=1, flags=re.M)
        assert "domain_weights" in txt
    txt = txt.replace("\nscorer:", "\n" + FLAGS + "\nscorer:", 1)
    hdr = (f"# 385M GROWN MODEL (owner 2026-11-01): GENERATED by scripts/clean_rerun_385m_make_configs.py from the clean-rerun generator (same recipe, caps and reward as configs/clean_rerun; see that script for the overrides).\n"
           f"# Policy and KL reference start: the SFT final of the growth-1 run (runs/g1b_s6/last, exported to HF format) = 39 layers (36 sliding + 3 global, sliding_window_pattern 13), 385,144,448 parameters.\n"
           f"# SINGLE-TOKEN TAG MODE: run with GEMMA_MT_TAG_MODE=single_token (scripts/launch_clean_rerun_385m.sh exports it); there is no yaml key for it.\n")
    return hdr + txt


def _put(txt: str, key: str, value, comment: str = "", indent: str = "") -> str:
    pat = re.compile(rf"^{indent}{re.escape(key)}:.*$", re.M)
    assert len(pat.findall(txt)) == 1, f"{key}: expected exactly one line"
    return pat.sub(lambda m: f"{indent}{key}: {value}" + (f"  # {comment}" if comment else ""), txt)


def _block(txt: str, key: str, body: str) -> str:
    pat = re.compile(rf"^{re.escape(key)}:.*\n(?:  .*\n)+", re.M)
    assert len(pat.findall(txt)) == 1, f"{key}: expected exactly one block"
    return pat.sub(lambda m: body, txt)


def make_f2(final_txt: str, root: str) -> str:
    """The OPTIONAL stage F2 = the rewritten FINAL yaml with the long-document settings (owner 2026-11-02). Everything not listed here is the final run's: reward rules, language weights and floor, KL
    coefficient / guard (0.1 / 0.25, recover on), learning rate 1e-6, Muon, group 16 with the greedy member, anchors, longdoc_v2 (calibration longdoc_clean.json), the science anchor."""
    t = final_txt
    t = re.sub(r"^# CLEAN RERUN, RL stage F \(final run\), arm A0", "# CLEAN RERUN, OPTIONAL RL stage F2 (final_long), arm A0", t, flags=re.M)
    t = re.sub(r"^# Final run: .*$", "# F2 (owner 2026-11-02, OPTIONAL, science only): b1024 60 % + b2048 40 % of the SCIENCE prompts, 70 steps, from the FINAL run's model (policy start AND KL reference, fresh optimizer). Kept only if it passes the\n"
               "# paper-level gate of scripts/clean_rerun_tools.py f2-decide (reports/rl_stage_plan_385m_b256_2026-11-02.md section 6); run only when f2-precheck says RUN. NLLB signal OFF (long windows: D-66, as B2 / B3).", t, count=1, flags=re.M)
    t = _put(t, "model_name", f"{root}/runs_385m/final/final", "the FINAL run's end model (F) = policy start AND KL reference; the launcher sets it to the hand-off of the final run (HANDOFF=final|best)")
    t = _put(t, "prompts_path", f"{root}/prompts/clean_sci_v2_long", "clean_sci_v2 (b64..b1024, decontaminated) + zz_b2048/ (the b2048 science rows of the ORIGINAL sci_v2 AFTER the decon.py scan: the clean pool dropped b2048 unscanned); scripts/clean_rerun_385m_prepare_f2_pool.sh")
    t = _put(t, "out_dir", f"{root}/runs_385m/F2")
    t = _put(t, "steps", F2_STEPS, "HARD CAP. F2 is short: it only has to teach the model to finish and not loop on whole papers")
    t = _put(t, "vllm_gpu_memory_utilization", 0.3, "0.30 (final: 0.20): sequences of up to ~9.5k tokens need a larger KV pool (reports/rl_gpu_ram_385m_2026-11-01.md)")
    t = _put(t, "vllm_max_model_len", F2_MAX_MODEL_LEN, "b2048 prompt (<= ~2,300) + 7,168 new tokens; the old final run's value")
    t = _put(t, "max_src_tokens", F2_MAX_SRC, "F2: SOURCE-side cap = the upper edge of b2048; the science weights below leave only b1024 and b2048 in use")
    t = _put(t, "gradient_checkpointing", "true", "F2: a 9.5k-token sequence x 39 layers does not fit an 80 GB card without it next to vLLM 0.30 (~15 GB trainer with it, reports/rl_gpu_ram_385m_2026-11-01.md); compute +25-35 % on the update part only")
    t = _put(t, "prompts_per_step", F2_PPS["mixed"], "fallback; the per-bin table below decides")
    t = _block(t, "prompts_per_step_by_bin", "prompts_per_step_by_bin:  # F2: 2c maxima for the single bins; mixed 28 x group 16 = 448 completions per step\n" + "".join(f"  {b}: {n}\n" for b, n in F2_PPS.items()))
    t = _block(t, "max_new_tokens_by_bin", "max_new_tokens_by_bin:  # Gemma tokens; the old generator's table + D-45 (b2048 7,168 = 3.5 x 2,048); ONE cap per mixed step, so b1024 prompts of F2 also get 7,168\n" + "".join(f"  {b}: {n}\n" for b, n in F2_MAX_NEW.items()))
    t = _put(t, "domain_weights", "{science: 1.0}", "F2: science prompts only (the pool also carries general rows; they are never drawn)")
    t = _block(t, "domain_bin_weights", f"domain_bin_weights:  # F2: b1024 : b2048 = 60 : 40 of the science mass (the AfriScience-MT dev papers are 16 b1024 : 9 b2048)\n  science: {{b1024: {F2_BIN_WEIGHTS['b1024']}, b2048: {F2_BIN_WEIGHTS['b2048']}}}\n")
    t = _put(t, "warmup_steps", 10, "F2: 70-step stage, warm-up as the B stages (final: 20)")
    t = _put(t, "plateau_stop", "false", "off: the 70-step cap is shorter than the 100-step window")
    t = _put(t, "plateau_stop_min_steps", 150)
    t = _put(t, "early_stop_patience", 40, "as B3 / mixed")
    t = _put(t, "early_stop_min_steps", 30, "as B3 / mixed")
    t = _put(t, "extend_steps", 20, "extend_steps > planned_winddown_last is required")
    t = _put(t, "planned_winddown_last", 15, "F2: the trainer starts the planned lr decay itself 15 steps before the (possibly extended) cap")
    t = _put(t, "eval_every", 35, "hold-out reward + greedy finished / loop rate at steps 35 and 70")
    t = _put(t, "eval_prompts", 96, "F2: long greedy decodes (up to 7,168 tokens) make every evaluation prompt expensive")
    t = _put(t, "benchmark_eval_every", 35, "sentence-level dev chrF++ at steps 35 and 70: the PRODUCTION mode must not regress")
    if "ds" in FEATURES:
        t = _put(t, "dynamic_sampling", "false", "F2: long documents (b1024 / b2048) cost far more per prompt; flat groups are rare there and an over-sampled draw would not pay (the translation stages and the final run use it)")
    if "devstop" in FEATURES:
        t = _put(t, "early_stop_dev", "false", "F2: 70 steps, early_stop_dev_min_steps is 75 anyway")
    if "devpred" in FEATURES:
        t = _put(t, "save_dev_predictions", "false", "F2: off")
    if "monitor" in FEATURES:
        for key in ("heldout_comet", "hacking_alarm", "early_stop_dev_require_comet", "lang_stall_log"):
            t = _put(t, key, "false", "F2: off" if key == "heldout_comet" else "")
    if "guide" in FEATURES:
        t = _put(t, "guide_fraction", "0.0", "F2: a beam search over a 7k-token document costs minutes; no guide in the long-document stage")
    t = _put(t, "archive", "false", "F2 feeds nothing")
    t = _put(t, "archive_path", "null")
    t = re.sub(r"^hub_repo_id: \S+", f"hub_repo_id: {F2_HUB[0]}", t, count=1, flags=re.M)
    t = re.sub(r"^hub_branch: .*$", f"hub_branch: {F2_HUB[1]}", t, count=1, flags=re.M)
    t = _put(t, "nllb_signal", "false", "F2 = long windows: NLLB pseudo-references exist per sentence of the clean pools only (D-66 is off in the long-bin stages B2 / B3 for the same reason)", indent="  ")
    t = _put(t, "nllb_languages", "[]", "off in this stage", indent="  ")
    t = t.replace("benchmark_aux:  # monitor-only dev sets (never used for selection)\n",
                  "benchmark_aux:  # monitor-only dev sets (never used for selection)\n"
                  f"  - {{name: afrisci_dev_doc, path: {root}/assets/dev/bench_afrisci_dev_doc.jsonl, per_language: 25, max_new_tokens: 7168, every: 35}}\n", 1)
    assert "afrisci_dev_doc" in t
    return t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(HERE, "..", "configs", "clean_rerun_385m")); ap.add_argument("--root", default="/workspace/clean_rerun")
    ap.add_argument("--features", default=",".join(ALL_FEATURES), help=f"owner 2026-11-03 additions to switch ON in the stage yamls, subset of {ALL_FEATURES} (ds = dynamic sampling, adv = advantage std floor, log = group stats, guide = beam guide, ref = reference term + reference domain, mbr = MBR guide for the weak languages (needs guide; stages B1 / mixed / final), devstop = early stop on the dev metrics (stages B1 / mixed / final; also benchmark_eval_every 25), devpred = save the dev predictions of every dev evaluation (B1 / mixed / final), monitor = held-out AfriCOMET monitor + reward-hacking alarm + combined dev stop + per-language dev history (mixed / final))")
    a = ap.parse_args()
    global FEATURES
    FEATURES = {f for f in a.features.split(",") if f}
    bad = FEATURES - set(ALL_FEATURES)
    if bad:
        ap.error(f"unknown feature(s) {sorted(bad)}")
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    P = gen.paths(a.root)
    apply_b256_cap(gen)
    for stale in ("rl_B2_b512.yaml", "rl_B3_b1024.yaml"):                  # generated files of the removed stages
        if os.path.exists(os.path.join(out, stale)):
            os.remove(os.path.join(out, stale))
    with tempfile.TemporaryDirectory() as td:
        rc = gen.main(["--root", a.root, "--out", td])
        assert rc == 0
        for stage, S in gen.STAGES.items():
            txt = open(os.path.join(td, S["file"]), encoding="utf-8").read()
            new = rewrite(txt, stage, a.root, P)
            open(os.path.join(out, S["file"]), "w", encoding="utf-8").write(new)
            if stage == "final":
                open(os.path.join(out, F2_FILE), "w", encoding="utf-8").write(make_f2(new, a.root))      # OPTIONAL stage, not in gen.STAGES
    nl = os.path.join(HERE, "..", "configs", "clean_rerun", "nllb_languages.json")
    import shutil; shutil.copyfile(nl, os.path.join(out, "nllb_languages.json"))
    print("wrote", len(gen.STAGES), "stage yamls +", F2_FILE, "(optional) to", out)


if __name__ == "__main__":
    main()
