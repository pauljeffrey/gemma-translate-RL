# SPDX-License-Identifier: AGPL-3.0-or-later
"""Push the CURRENT training checkpoint to a Hub model repo from inside the training loop (owner 2026-10-14: "push to hub at
every interval so we don't lose work; update the hub with checkpoint iter").

Every call is one commit on the repo's main branch:

    <repo>/                    weights + config + tokenizer + generation_config + README.md   (the latest step)
    <repo>/best/               the best checkpoint by validation loss (only rewritten when it improves)
    <repo>/training_state.json step, tokens, validation losses (overall, macro, per language), sampler settings, time
    tag  step-<N>              so every pushed iteration can be loaded with  from_pretrained(repo, revision="step-<N>")

Safety (same rules as hub.push_model): nothing is uploaded unless `hub_repo_id` is configured AND a WRITE token is in the
environment (HF_WRITE_TOKEN, falling back to HF_TOKEN, HF_API_KEY); the repo is created PRIVATE; optimizer state is never
uploaded; a failed upload is logged and retried at the next interval, it never stops the training. The upload runs in a
background thread so the GPU keeps training.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional

from gemma_mt.hub.model_card import DEFAULT_BASE, build_model_card

ALLOW = ["*.safetensors", "config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json",
         "added_tokens.json", "chat_template*", "*.model", "README.md", "training_state.json"]


TOKEN_FILES = ("/workspace/sft_build/.hf_write_token", os.path.expanduser("~/.hf_write_token"), "/workspace/sft_build/env.sh")


def _token_from_files() -> Optional[str]:
    """A run started before the owner added the token keeps its old environment, so look for it again at push time:
    a one-line token file, or `export HF_WRITE_TOKEN=...` in env.sh. The value is never printed."""
    import re

    for path in TOKEN_FILES:
        try:
            txt = open(path, encoding="utf-8").read()
        except OSError:
            continue
        m = re.search(r"^\s*(?:export\s+)?HF_WRITE_TOKEN\s*=\s*[\"']?([A-Za-z0-9_\-]{20,})", txt, re.M)
        if m:
            return m.group(1)
        if not path.endswith("env.sh") and re.fullmatch(r"\s*[A-Za-z0-9_\-]{20,}\s*", txt):
            return txt.strip()
    return None


def write_token() -> Optional[str]:
    return (os.environ.get("HF_WRITE_TOKEN") or _token_from_files() or os.environ.get("HF_TOKEN")
            or os.environ.get("HF_API_KEY"))


class CheckpointPusher:
    def __init__(self, repo_id: Optional[str], *, private: bool = True, stage: str = "sft", base_model: str = DEFAULT_BASE,
                 directions=("en2x",), api: Any = None, log=print, branch: Optional[str] = None, tag_prefix: Optional[str] = "step",
                 squash: bool = False, prune_tags: bool = False):
        """`branch`: push to this branch of the repo instead of main (e.g. "rl" next to the SFT model on main).
        `tag_prefix`: every push creates the tag `<prefix>-<step>`; None = no tags (a branch that is overwritten all the time, e.g. RL `latest`).
        Storage limits (owner 2026-10-19: the Hub quota is small): `prune_tags` deletes every OLDER tag `<prefix>-N` after a push (only the newest is kept);
        `squash` squashes the history of the branch into one commit after a push (`HfApi.super_squash_history`), so superseded weights are no longer referenced and
        the repo holds one copy per branch (the new tag is created after the squash, on the squashed head). Both are best effort: a failure is logged, the push stays."""
        self.repo_id, self.private, self.stage, self.base_model = repo_id, private, stage, base_model
        self.branch, self.tag_prefix = branch, tag_prefix
        self.squash, self.prune_tags = bool(squash), bool(prune_tags)
        self.directions, self.log = tuple(directions), log
        self._api = api
        self._injected_api = api is not None
        self._token_used: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None
        self.pushed = 0

    @property
    def enabled(self) -> bool:
        return bool(self.repo_id) and (self._api is not None or bool(write_token()))

    def _client(self):
        if not self._injected_api:                       # 2026-10-20: re-read the token on every push, so a rotated token (new .hf_write_token) is picked up without a restart
            tok = write_token()
            if self._api is None or tok != self._token_used:
                from huggingface_hub import HfApi

                self._api, self._token_used = HfApi(token=tok), tok
        return self._api

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def push(self, model_dir: str, step: int, state: Dict[str, Any], *, as_best: bool = False, results_md: Optional[str] = None,
             block: bool = False, cleanup: bool = False) -> bool:
        """`model_dir` must already hold the saved checkpoint (save_model). Returns False when pushing is disabled."""
        if not self.repo_id:
            return False
        if not self.enabled:
            self.log(f"[gemma_mt] hub push skipped at step {step}: no write token in HF_WRITE_TOKEN / HF_TOKEN")
            return False
        self.wait()                                              # one upload at a time; the previous one is normally long done
        if os.environ.get("HUB_NO_MODEL_CARD") == "1":           # owner 2026-10-30: no model card until everything is finished
            readme = os.path.join(model_dir, "README.md")
            if os.path.exists(readme):
                os.remove(readme)
        else:
            card = build_model_card(self.repo_id, stage=self.stage, base_model=self.base_model, results_markdown=results_md,
                                    directions=self.directions)
            with open(os.path.join(model_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(card)
        with open(os.path.join(model_dir, "training_state.json"), "w", encoding="utf-8") as f:
            json.dump({"step": step, "pushed_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), **state}, f, indent=2,
                      default=float)
        args = (model_dir, step, as_best, cleanup)
        if block:
            self._upload(*args)
        else:
            self._thread = threading.Thread(target=self._upload, args=args, daemon=True)
            self._thread.start()
        return True

    def _prune_old_tags(self, api, keep: Optional[str]) -> None:
        """Delete every tag `<tag_prefix>-<something>` except `keep` (older tags would keep the superseded weights alive)."""
        try:
            refs = api.list_repo_refs(self.repo_id, repo_type="model")
            for t in getattr(refs, "tags", []) or []:
                name = getattr(t, "name", t)
                if name.startswith(f"{self.tag_prefix}-") and name != keep:
                    api.delete_tag(self.repo_id, tag=name, repo_type="model")
        except Exception as e:
            self.log(f"[gemma_mt] hub tag pruning failed: {e}")

    def _upload(self, model_dir: str, step: int, as_best: bool, cleanup: bool = False) -> None:
        try:
            api = self._client()
            api.create_repo(self.repo_id, repo_type="model", private=self.private, exist_ok=True)
            if self.private:                            # 2026-10-29 (owner: all models private): create_repo(exist_ok=True) leaves an EXISTING public repo public, so say it explicitly
                try:
                    api.update_repo_settings(self.repo_id, private=True, repo_type="model")
                except Exception as e:                  # a client without that call (tests) or a token without the right to change settings
                    self.log(f"[gemma_mt] could not confirm that {self.repo_id} is private: {type(e).__name__}: {e}")
            rev = {}
            if self.branch:
                api.create_branch(self.repo_id, branch=self.branch, repo_type="model", exist_ok=True)
                rev = {"revision": self.branch}
            msg = f"{self.stage} step {step}"
            api.upload_folder(folder_path=model_dir, repo_id=self.repo_id, repo_type="model", allow_patterns=ALLOW,
                              commit_message=msg, **rev)
            if as_best:
                api.upload_folder(folder_path=model_dir, repo_id=self.repo_id, repo_type="model", path_in_repo="best",
                                  allow_patterns=ALLOW, commit_message=f"{msg} (best validation loss)", **rev)
            tag = f"{self.tag_prefix}-{step}" if self.tag_prefix else None
            if self.prune_tags and self.tag_prefix:
                self._prune_old_tags(api, keep=tag)
            if self.squash:
                try:
                    api.super_squash_history(repo_id=self.repo_id, branch=self.branch or "main", repo_type="model")
                except Exception as e:                           # the weights are up; only the history stays long
                    self.log(f"[gemma_mt] hub squash of {self.repo_id}@{self.branch or 'main'} failed: {e}")
            if tag:
                try:
                    api.create_tag(self.repo_id, tag=tag, repo_type="model", exist_ok=True, **rev)    # after the squash: the tag points at the squashed head
                except Exception as e:                           # a tag is a convenience, the weights are already up
                    self.log(f"[gemma_mt] hub tag {tag} failed: {e}")
            self.pushed += 1
            self.last_error = None
            if cleanup:                                          # only the uploaded staging copy; kept on failure so it can be retried
                import shutil

                shutil.rmtree(model_dir, ignore_errors=True)
            self.log(f"[gemma_mt] pushed step {step} to {self.repo_id}" + (f"@{self.branch}" if self.branch else "") + (" (+best)" if as_best else ""))
        except Exception as e:                                   # never stop the training for a network problem
            self.last_error = f"{type(e).__name__}: {e}"
            self.log(f"[gemma_mt] hub push FAILED at step {step}: {self.last_error} (will retry at the next interval)")
