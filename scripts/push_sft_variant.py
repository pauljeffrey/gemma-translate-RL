#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-59: push one SFT-stage model (a trained variant, or an interpolated alpha of it) as a model folder `sft/<folder>/` of a Hub repo (owner-authorised 2026-10-22).

    python scripts/push_sft_variant.py --variant B_v1 --mix mix_v2B_v1 --repo <HF_ORG_LEGACY>/gemma-science-translate --folder B_v1 --alpha 1.0 \
        --run /workspace/jobs/r7/runs/sft_B_v1 --log /workspace/jobs/r7/logs/sft_B_v1.log --config /workspace/jobs/r11/configs/sft_science_v2_B_v1.yaml [--notes notes.md] [--dry-run]

* The folder is built from `final_1433` + the run's weights-only checkpoint with scripts/interpolate_weights.py (alpha 1.0 = the trained model), written to a staging dir
  under /workspace/jobs/r7/push_stage (deleted after the upload; free disk is checked first), then uploaded with `upload_folder(path_in_repo="sft/<folder>", allow_patterns=model files)`;
  NO delete_patterns, so the repo root and the other folders (the RL run's main/latest files) are untouched. No optimizer state, no trainer_state.pt.
* Repo (private) is created if it does not exist. Token: gemma_mt.hub.push_checkpoint.write_token() (re-read from /workspace/sft_build/.hf_write_token at every call).
* A README.md model card is generated (variant, mixture component table from the mix's meta.json, hyper-parameters from the config, steps, start model, validation numbers from the training
  log, optional --notes file with the guardrail results, licence note, date). Re-running with another --notes replaces only that folder's README.
* Appends repo / folder / commit sha to /workspace/jobs/r7/SFT_STATUS.txt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

ALLOW = ["*.safetensors", "config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json", "added_tokens.json", "chat_template*", "*.model", "README.md"]
INIT = "/workspace/sft_build/runs/final_1433"
STAGE = "/workspace/jobs/r7/push_stage"
STATUS = "/workspace/jobs/r7/SFT_STATUS.txt"


def val_numbers(log: str) -> str:
    out = []
    if log and os.path.exists(log):
        for line in open(log, errors="replace"):
            if "eval @" in line or "eval[val" in line:
                out.append("    " + re.sub(r"^\[gemma_mt\]\s*", "", line.rstrip()))
    return "\n".join(out) if out else "    (training log not found)"


def card(a, meta2: dict, cfg_text: str, steps: str, notes: str) -> str:
    tb = meta2.get("tokens_by_source", {})
    tot = sum(tb.values()) or 1
    rows = "\n".join(f"| {k} | {v:,} | {100 * v / tot:.1f} % |" for k, v in tb.items() if v)
    lic = {}
    for cls, d in (meta2.get("tokens_by_licence_class") or {}).items():
        lic[cls] = sum(d.values()) if isinstance(d, dict) else d
    lic_rows = "\n".join(f"| {k} | {v:,} |" for k, v in sorted(lic.items(), key=lambda x: -x[1]))
    keep = [l for l in cfg_text.splitlines() if re.match(r"^(learning_rate|warmup|lr_schedule|min_lr|weight_decay|grad_clip|tokens_per_step|micro_batch|grad_accum|epochs|seq_len|l2sp|adam|beta|optimizer|gradient_checkpointing)", l)]
    return f"""---
license: other
tags: [translation, african-languages, gemma, sft-stage]
---
# {a.repo.split('/')[-1]} / sft/{a.folder}: science-SFT stage model `{a.variant}`{'' if a.alpha == 1.0 else f' (interpolated, alpha = {a.alpha})'}

* Variant: **{a.variant}**; mixture directory `{a.mix}` (agent R11, D-58/D-59). Weights: theta = (1 - alpha) * theta_final_1433 + alpha * theta_sft with **alpha = {a.alpha}** (alpha 1.0 = the trained model).
* Start model: `final_1433` (= <HF_ORG_LEGACY>/gemma-3-270m-translate-rl2c @ latest, the general RL model at step 1433; also the KL reference of the later science RL run). Full fine-tune, fresh AdamW, bf16 autocast / fp32 weights.
* Steps: {steps} optimizer steps of 65,536 tokens (1 epoch over the packed mixture, 8,192-token packs). Hyper-parameters (from `{os.path.basename(a.config)}`):
```
{chr(10).join(keep)}
```
* Date: {time.strftime('%Y-%m-%d', time.gmtime())}. Part of the 2-stage plan: SFT stage (this folder) -> science + general GRPO run.

## Mixture (training tokens by component)
| component | tokens | share |
|---|---:|---:|
{rows}

Licence classes of the added rows (tokens, from R11's audit):

| class | tokens |
|---|---:|
{lic_rows}

## Validation (training log; val = replay validation over 77 languages, val_sci = AfriScience-MT dev)
{val_numbers(a.log)}

## Guardrail results / notes
{notes or '(not yet evaluated against the guardrails when this folder was first pushed; see the repository history for an updated README)'}

## Licence
{a.licence}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True); ap.add_argument("--mix", required=True); ap.add_argument("--repo", required=True); ap.add_argument("--folder", required=True)
    ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--run", required=True); ap.add_argument("--log", default=None); ap.add_argument("--config", required=True)
    ap.add_argument("--notes", default=None); ap.add_argument("--licence", default=None); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--readme-only", action="store_true", help="only (re)upload README.md of the folder")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    meta = json.load(open(f"/workspace/jobs/r11/{a.mix}/meta.json"))
    meta2 = meta.get("mixture_v2", {})
    cfg_text = open(a.config).read()
    steps = str(meta["splits"]["train"]["n_tokens"] // 65536) if "splits" in meta else "?"
    m = re.search(r"(\d+) total steps", open(a.log, errors="replace").read()) if a.log and os.path.exists(a.log) else None
    steps = m.group(1) if m else steps
    nc = meta2.get("tokens_by_source", {}).get("r10_nc", 0) > 0 or "ibom" in a.variant.lower() or "nc" in a.variant.lower().split("_")
    acs = a.variant.startswith("B_v2") or "code_switch" in str(meta2.get("tokens_by_source", {})) and meta2["tokens_by_source"].get("r10_code_switch", 0) > 0
    a.licence = a.licence or (
        "**Treat this model as NON-COMMERCIAL.** Its training mixture contains CC-BY-NC data (" + ("IbomMT train; " if nc else "") + "see the licence-class table), CC-BY-SA data (OLDI/SALT/Wikimedia where listed), the owner's African-Code-Switch subset"
        + " and the base SFT data of final_1433 (JW/Bible/news-like sources, see D-21). The model-level licence of the Gemma base also applies." if (nc or acs) else
        "Mixture = AfriScience-MT (Apache-2.0) + the original SFT replay windows of final_1433 (provenance D-21: JW/Bible/news-like sources) + RL-archive/long-document distillation rows (outputs of our own model) + R10 commercial-OK extras where listed. "
        "Not audited for commercial use as a whole; the Gemma base-model terms apply.")
    notes = open(a.notes).read() if a.notes else ""
    tmp = os.path.join(STAGE, a.folder)
    os.makedirs(STAGE, exist_ok=True)
    free = shutil.disk_usage(STAGE).free / 2**30
    if not a.readme_only and free < 2.6:
        print(f"only {free:.1f} GiB free: refusing to stage a 1.1 GB model", file=sys.stderr); return 3
    shutil.rmtree(tmp, ignore_errors=True)
    if a.readme_only:
        os.makedirs(tmp, exist_ok=True)
    else:
        from interpolate_weights import interpolate_to_dir

        interpolate_to_dir(INIT, os.path.join(a.run, "last", "trainer_state.pt"), a.alpha, tmp)
    open(os.path.join(tmp, "README.md"), "w").write(card(a, meta2, cfg_text, steps, notes))
    bad = [f for f in os.listdir(tmp) if f.endswith(".pt") or f.startswith("optimizer")]
    assert not bad, bad
    print("staged:", {f: os.path.getsize(os.path.join(tmp, f)) for f in sorted(os.listdir(tmp))})
    if a.dry_run:
        return 0
    from huggingface_hub import HfApi

    from gemma_mt.hub.push_checkpoint import write_token

    tok = write_token()
    if not tok:
        print("no write token", file=sys.stderr); return 4
    api = HfApi(token=tok)
    api.create_repo(a.repo, private=True, exist_ok=True)
    pats = ["README.md"] if a.readme_only else ALLOW
    info = api.upload_folder(folder_path=tmp, repo_id=a.repo, repo_type="model", path_in_repo=f"sft/{a.folder}", allow_patterns=pats,
                             commit_message=f"SFT stage: {a.variant} alpha {a.alpha} ({'README only' if a.readme_only else 'model files'})")
    sha = getattr(info, "oid", None) or str(info)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} PUSHED repo={a.repo} branch=main folder=sft/{a.folder} variant={a.variant} alpha={a.alpha} {'README-only ' if a.readme_only else ''}commit={sha}"
    open(STATUS, "a").write(line + "\n")
    print(line)
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
