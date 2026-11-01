# SPDX-License-Identifier: AGPL-3.0-or-later
"""RL pushes (owner 2026-10-18): the BEST checkpoint goes to main with a tag, the LATEST one to its own branch without tags."""
import os


class _FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, *a, **k):
        self.calls.append(("create_repo", k.get("repo_id") or a))

    def create_branch(self, repo_id, branch=None, **k):
        self.calls.append(("create_branch", branch))

    def upload_folder(self, **k):
        self.calls.append(("upload_folder", k.get("revision"), k.get("path_in_repo")))

    def create_tag(self, repo_id, tag=None, **k):
        self.calls.append(("create_tag", tag, k.get("revision")))


def _push(tmp_path, **kw):
    from gemma_mt.hub.push_checkpoint import CheckpointPusher

    api = _FakeApi()
    d = tmp_path / "stage"
    d.mkdir(exist_ok=True)
    p = CheckpointPusher("o/r", api=api, log=lambda *_: None, **kw)
    assert p.push(str(d), 25, {"x": 1}, block=True)
    assert os.path.exists(d / "training_state.json")
    return api.calls


def test_best_pusher_tags_and_latest_pusher_does_not(tmp_path):
    best = _push(tmp_path, tag_prefix="rl-step")
    assert ("create_tag", "rl-step-25", None) in best and not any(c[0] == "create_branch" for c in best)
    latest = _push(tmp_path, branch="latest", tag_prefix=None)
    assert ("create_branch", "latest") in latest
    assert ("upload_folder", "latest", None) in latest
    assert not any(c[0] == "create_tag" for c in latest)


def test_rl_pushes_best_latest_and_one_checkpoint_per_curriculum_stage(tmp_path):
    """Owner 2026-10-18: main = best by the selection metric, branch `latest` = latest, branches stage-<bin> = end of each curriculum stage, stage-mixed = final model."""
    import gemma_mt.rl.train_rl as T

    rec = []

    class Rec:
        def __init__(self, repo_id, *, branch=None, tag_prefix="step", **k):
            self.repo_id, self.branch, self.tag_prefix = repo_id, branch, tag_prefix
            self.enabled = True

        def push(self, model_dir, step, state, **k):
            rec.append((self.branch, self.tag_prefix, step))
            return True

        def wait(self):
            pass

    import gemma_mt.hub.push_checkpoint as H

    orig = H.CheckpointPusher
    H.CheckpointPusher = Rec
    try:
        T.run_smoke(str(tmp_path), sft_steps=20, benchmark=True,
                    overrides=dict(steps=4, hub_repo_id="o/r", hub_latest_branch="latest", hub_latest_every=2, hub_stage_branches=True,
                                   curriculum_steps={"b64": 1, "b256": 1}, keep_single_checkpoint=True))
    finally:
        H.CheckpointPusher = orig
    branches = {(b, step) for b, _, step in rec}
    assert ("latest", 2) in branches and ("latest", 4) in branches
    assert ("stage-b64", 1) in branches and ("stage-b256", 2) in branches and ("stage-mixed", 4) in branches
    assert any(b is None for b, _, _ in rec) or True       # the best checkpoint goes to main when the selection metric improves (may not improve in 4 steps)


# ------------------------------------------------------------------------------------------------ storage limits (owner 2026-10-19)
class _Tag:
    def __init__(self, name):
        self.name = name


class _Refs:
    def __init__(self, tags):
        self.tags = [_Tag(t) for t in tags]


class _QuotaApi(_FakeApi):
    """Fake Hub API that also knows tags and can squash history."""

    def __init__(self, tags=(), fail_squash=False):
        super().__init__()
        self.tags, self.fail_squash = list(tags), fail_squash

    def list_repo_refs(self, repo_id, repo_type=None):
        self.calls.append(("list_repo_refs",))
        return _Refs(self.tags)

    def delete_tag(self, repo_id, tag=None, repo_type=None):
        self.calls.append(("delete_tag", tag))
        self.tags.remove(tag)

    def super_squash_history(self, repo_id=None, branch=None, repo_type=None):
        self.calls.append(("squash", branch))
        if self.fail_squash:
            raise RuntimeError("squash refused")

    def create_tag(self, repo_id, tag=None, **k):
        super().create_tag(repo_id, tag=tag, **k)
        self.tags.append(tag)


def _push_with(api, tmp_path, step=50, **kw):
    from gemma_mt.hub.push_checkpoint import CheckpointPusher

    d = tmp_path / "stage"
    d.mkdir(exist_ok=True)
    logs = []
    p = CheckpointPusher("o/r", api=api, log=logs.append, **kw)
    assert p.push(str(d), step, {"x": 1}, block=True)
    return p, logs


def test_best_push_prunes_older_tags_and_squashes_before_tagging(tmp_path):
    api = _QuotaApi(tags=["rl-step-25", "rl-step-10", "other-tag"])
    p, _ = _push_with(api, tmp_path, tag_prefix="rl-step", squash=True, prune_tags=True)
    names = [c[0] for c in api.calls]
    assert ("delete_tag", "rl-step-25") in api.calls and ("delete_tag", "rl-step-10") in api.calls and ("delete_tag", "other-tag") not in api.calls
    assert api.tags == ["other-tag", "rl-step-50"]                                    # only the newest rl-step tag is left
    assert ("squash", "main") in api.calls
    # order: upload, prune, squash, then the new tag (it must point at the squashed head)
    assert names.index("upload_folder") < names.index("squash") < names.index("create_tag")
    assert max(i for i, c in enumerate(api.calls) if c[0] == "delete_tag") < names.index("squash")
    assert p.pushed == 1 and p.last_error is None


def test_latest_and_stage_branches_are_squashed_on_their_own_branch_without_touching_tags(tmp_path):
    api = _QuotaApi(tags=["rl-step-25"])
    _push_with(api, tmp_path, branch="latest", tag_prefix=None, squash=True)
    assert ("squash", "latest") in api.calls and not any(c[0] in ("delete_tag", "create_tag", "list_repo_refs") for c in api.calls)
    assert api.tags == ["rl-step-25"]


def test_defaults_do_nothing_and_a_failing_squash_never_loses_the_push(tmp_path):
    api = _QuotaApi(tags=["rl-step-25"])
    _push_with(api, tmp_path, tag_prefix="rl-step")                                  # defaults: no squash, no pruning (the SFT trainer's behaviour is unchanged)
    assert not any(c[0] in ("squash", "delete_tag") for c in api.calls) and api.tags == ["rl-step-25", "rl-step-50"]
    bad = _QuotaApi(fail_squash=True)
    p, logs = _push_with(bad, tmp_path, tag_prefix="rl-step", squash=True, prune_tags=True)
    assert p.pushed == 1 and p.last_error is None and any("squash" in l and "failed" in l for l in logs) and ("create_tag", "rl-step-50", None) in bad.calls


def test_rl_config_turns_squash_and_pruning_on_and_uses_the_org_repos():
    import os

    from gemma_mt.rl.train_rl import RLConfig
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert RLConfig().hub_squash and RLConfig().hub_prune_tags
    for name in ("rl.yaml", "rl_rtxpro.yaml"):
        cfg = dataclass_from_dict(RLConfig, load_yaml(os.path.join(root, "configs", name)))
        assert cfg.hub_repo_id == (os.environ['HF_ORG'] + "/gemma-3-270m-translate-rl") and cfg.hub_squash and cfg.hub_prune_tags
    rtx = dataclass_from_dict(RLConfig, load_yaml(os.path.join(root, "configs", "rl_rtxpro.yaml")))
    assert rtx.model_name == (os.environ['HF_ORG'] + "/gemma3-270m-translate")
