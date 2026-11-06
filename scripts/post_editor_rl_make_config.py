#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Editor-RL config generator: configs/clean_rerun_385m/rl_A_b64.yaml (the recipe of the translation RL, stage A) -> configs/post_editor/pe_rl.yaml with the editor settings.

    python scripts/post_editor_rl_make_config.py --out configs/post_editor/pe_rl.yaml [--model DIR] [--tokenizer DIR] [--prompts FILE] [--run-dir DIR] [--glot model_v3.bin] [--off-terms churn,...] [--tau 0.02] [--refcomet-tau 0.01]

What changes against the translation RL (every value is the owner's 2026-11-03 design, docs/POST_EDITOR_RL.md):
  policy / KL reference  E1 final (editor SFT); tokenizer = the renamed RL tokenizer (same ids as the original one)
  prompts                the editor pool (pool_train.parquet: English + human reference or none + draft), no length bins, no domain weights
  group                  16 = 13 sampled + 1 IDENTITY (the unchanged draft) + 1 beam-4 EDITOR output (guide) + 1 greedy; guide_fraction 1.0
  reward                 consensus OFF (it rewards copying), NLLB signal OFF, reference_weight 0 (the reference enters through the editor reward instead), lexicon terms OFF (no certified-clean lexicon on this box)
  optimiser              lr 1e-6, KL 0.05 (k3c = k3 with the log-ratio clamped at 5), KL guard, dynamic sampling, save_every 20
  off                    Hub push, archive, benchmark dev evaluation inside the run (the evaluation is done after the run, on the benchmarks, not on training prompts)
"""
import argparse, os, sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(ROOT, "configs/clean_rerun_385m/rl_A_b64.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="/workspace/pe/run_e1/out/final")
    ap.add_argument("--tokenizer", default="/workspace/models/rl_stage_a")
    ap.add_argument("--prompts", default="/workspace/pe/rlpool/pool_train.parquet")
    ap.add_argument("--run-dir", default="/workspace/pe/rl_run")
    ap.add_argument("--cal", default="/workspace/pe/tree/gemma_mt/calibration/clean_2026-10-28")
    ap.add_argument("--glot", default="")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--prompts-per-step", type=int, default=64)
    ap.add_argument("--off-terms", default="")
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--refcomet-tau", type=float, default=0.01)
    ap.add_argument("--comet-batch", type=int, default=16)
    ap.add_argument("--micro-batch-seqs", type=int, default=32)
    a = ap.parse_args(argv)

    raw = open(a.base, encoding="utf-8").read()
    c = yaml.safe_load(raw)
    r = c["reward"]
    cal = a.cal
    c.update(
        model_name=a.model, tokenizer_name=a.tokenizer, prompts_path=a.prompts, out_dir=a.run_dir, eval_prompts_path=None,
        steps=a.steps, prompts_per_step=a.prompts_per_step, group_size=16, greedy_in_group=True,
        guide_fraction=1.0, guide_beams=4, guide_beam_chunk=12, guide_adv_share_cap=0.30,
        length_bins=False, curriculum_steps={}, prompts_per_step_by_bin={}, max_new_tokens_by_bin={}, domain_weights=None, language_weights={}, language_alpha=0.0,
        max_new_tokens=384, max_src_tokens=256, micro_batch_seqs=a.micro_batch_seqs, micro_batch_tokens=8192, loss_chunk_tokens=512,
        gradient_checkpointing=True, ref_dtype="float32",
        vllm_gpu_memory_utilization=0.14, vllm_max_model_len=2048, vllm_max_num_seqs=256,
        learning_rate=1.0e-6, kl_coef=0.05, kl_estimator="k3c", save_every=20, log_every=1,
        plateau_stop=False, early_stop=False, extend_if_improving=False,
        select_by="reward", benchmark_eval_path=None, benchmark_aux=[], benchmark_eval_every=0,
        eval_every=20, eval_prompts=96, eval_min_per_bin=0,
        archive=False, archive_path=None, hub_repo_id=None, hub_latest_every=0, keep_single_checkpoint=False, editor_mode=True,
        run_name="pe_rl", lang_weights_file=True, adaptive_lang_weights=False,
    )
    for k in [k for k in c if k.startswith("hub_") and k != "hub_repo_id"]:
        c.pop(k)
    sc = dict(c["scorer"], python="/workspace/venv_comet/bin/python", pythonpath=None, batch_size=a.comet_batch)
    sc.pop("ensemble_model_id", None)                      # owner 2026-11-03: AfriCOMET only (SSA-COMET-MTL out of the reward; it is an independent evaluation metric)
    c["scorer"] = sc
    r.update(
        qe_ensemble=False,
        calibration_path=f"{cal}/shaping_clean_final.json", anchors_calibration_path=f"{cal}/anchors_clean_final.json", longdoc_calibration_path=f"{cal}/longdoc_clean_final.json",
        anchors_bigram_path=f"{cal}/bigrams_clean_final.npz", anchors_lexicon_dir=None, anchors_lexicon_weight=0.0, anchors_lexicon_guard=False,
        consensus=False, nllb_signal=False, nllb_pseudoref_dir="", nllb_calibration_path="", reference_weight=0.0, longdoc_compare_v1=False,
    )
    if a.glot:
        r["lid_model_path"] = a.glot
    c["reward"] = r
    c["editor"] = dict(
        identity_member=True,
        weights=dict(tau=a.tau),
        ref_weights=dict(gain=0.35, ref=1.0, tau=a.tau),
        ref_weights_refcomet_langs=dict(ref=0.6, refcomet=1.0, refcomet_tau=a.refcomet_tau),
        off_terms=[t for t in a.off_terms.split(",") if t],
        qe_gain_languages="qe_mode",
        ref_comet=dict(python="/workspace/venv_comet/bin/python", batch_size=a.comet_batch),
        weight_floor_unscored=0.5,
    )
    header = (
        "# EDITOR RL (post-editor E1 -> RL), GENERATED by scripts/post_editor_rl_make_config.py from configs/clean_rerun_385m/rl_A_b64.yaml: edit the generator, not this file.\n"
        "# Design and reward: docs/POST_EDITOR_RL.md. Run with GEMMA_MT_TAG_MODE=single_token. Report: reports/post_editor_rl_2026-11-03.md.\n"
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(c, f, sort_keys=False, width=200, allow_unicode=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
