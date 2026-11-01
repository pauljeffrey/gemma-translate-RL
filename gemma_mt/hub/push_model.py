# SPDX-License-Identifier: AGPL-3.0-or-later
"""Push a trained checkpoint (and its generated model card) to the Hugging Face Hub.

    python -m gemma_mt.hub.push_model --model-dir runs/sft/best --stage sft            # DRY RUN: prints, uploads nothing
    python -m gemma_mt.hub.push_model --model-dir runs/rl/best  --stage rl --yes       # actually pushes (private repo)

Safety: nothing is uploaded unless `--yes` is given; repositories are PRIVATE unless `--public`; the token is
read from the environment (HF_TOKEN, falling back to HF_API_KEY); optimizer/trainer state is never uploaded; the
checkpoint's generation_config must list <end_of_turn> as a stop token or the push is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from gemma_mt.hub.model_card import DEFAULT_BASE, build_model_card

DEFAULT_NAMESPACE = os.environ['HF_ORG_LEGACY']
DEFAULT_REPOS = {"sft": f"{DEFAULT_NAMESPACE}/gemma3-270m-translate",
                 "rl": f"{DEFAULT_NAMESPACE}/gemma3-270m-en2african-rl"}
EOT_ID = 106
ALLOW = ["*.safetensors", "config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json",
         "added_tokens.json", "chat_template*", "*.model", "README.md"]


def check_checkpoint(model_dir: str) -> List[str]:
    """Problems that must block a push (empty list = fine)."""
    problems = []
    if not os.path.isdir(model_dir):
        return [f"{model_dir} is not a directory"]
    names = os.listdir(model_dir)
    if not any(n.endswith(".safetensors") for n in names):
        problems.append("no .safetensors weights found")
    if "config.json" not in names:
        problems.append("config.json missing")
    gc = os.path.join(model_dir, "generation_config.json")
    if not os.path.exists(gc):
        problems.append("generation_config.json missing (generation would not stop on <end_of_turn>)")
    else:
        with open(gc) as f:
            eos = json.load(f).get("eos_token_id")
        ids = eos if isinstance(eos, list) else [eos]
        if EOT_ID not in ids:
            problems.append(f"generation_config.eos_token_id={eos} does not contain <end_of_turn> ({EOT_ID})")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--stage", choices=["sft", "rl"], required=True)
    ap.add_argument("--repo-id", default=None, help=f"default: {DEFAULT_REPOS}")
    ap.add_argument("--base-model", default=DEFAULT_BASE)
    ap.add_argument("--license", default="gemma")
    ap.add_argument("--training-data", default=None, help="text for the 'Training' section of the card")
    ap.add_argument("--languages", nargs="*", default=None,
                    help="tags the model was trained on (default: every language in the table except unsupported ones)")
    ap.add_argument("--results-md", default=None, help="Markdown file (e.g. from make_report) to embed in the card")
    ap.add_argument("--public", action="store_true", help="create/push a PUBLIC repository (default: private)")
    ap.add_argument("--yes", action="store_true", help="really upload (without it nothing leaves this machine)")
    a = ap.parse_args(argv)

    repo_id = a.repo_id or DEFAULT_REPOS[a.stage]
    results_md = open(a.results_md, encoding="utf-8").read() if a.results_md else None
    card = build_model_card(repo_id, stage=a.stage, base_model=a.base_model, license_id=a.license,
                            training_data=a.training_data, results_markdown=results_md,
                            languages=a.languages)
    problems = check_checkpoint(a.model_dir)
    print(f"repo: {repo_id} ({'public' if a.public else 'private'})  model dir: {a.model_dir}")
    print(f"files: {sorted(os.listdir(a.model_dir)) if os.path.isdir(a.model_dir) else '-'}")
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    if problems:
        return 3
    if not a.yes:
        print("\nDRY RUN (no --yes): nothing uploaded. Model card preview (first 40 lines):\n")
        print("\n".join(card.splitlines()[:40]))
        return 2
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if not token:
        print("set HF_TOKEN in the environment", file=sys.stderr)
        return 4
    from huggingface_hub import HfApi

    with open(os.path.join(a.model_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=not a.public, exist_ok=True)
    api.upload_folder(folder_path=a.model_dir, repo_id=repo_id, repo_type="model", allow_patterns=ALLOW,
                      commit_message=f"{a.stage} checkpoint")
    print(f"uploaded to https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
