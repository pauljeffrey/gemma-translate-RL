# SPDX-License-Identifier: AGPL-3.0-or-later
"""configs/clean_rerun_sft/*.yaml: strict load, the pre-registered invariants, and a CPU dry run of every config on a tiny random model.

The dry run changes ONLY what cannot exist on a laptop (model, tokenizer, data, schedule file, output folder, Hub push off, bf16 autocast off, and for S2 the warm-up length /
first new row to fit the tiny vocabulary); every other field - optimizer, lr, warm-up fraction, clip, batch shape, precision settings, stopping - is the YAML's own."""
import glob
import hashlib
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("yaml")

from gemma_mt.data.packed_dataset import PackedDataset, balanced_schedule
from gemma_mt.data.prepare_sft import PrepareConfig, prepare
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
from gemma_mt.smoke import synthetic_rows, write_jsonl
from gemma_mt.train_sft import SFTConfig, load_schedule, pack_lang_sha256, packed_structure_sha256, train, validate_attention_config
from gemma_mt.utils import dataclass_from_dict, load_yaml

CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "clean_rerun_sft")
FILES = sorted(glob.glob(os.path.join(CFG_DIR, "*.yaml")))
NAMES = [os.path.basename(f)[:-5] for f in FILES]
EXPECTED = {"a0", "s1", "s2", "g0", "a2_base", "a2_up6", "a2_up12", "a2_1b", "a0_control", "p1", "x1", "sanity_adamw", "sanity_muon"}
SHARED = ("learning_rate", "min_lr_ratio", "warmup_steps", "warmup_ratio", "weight_decay", "beta1", "beta2", "eps", "grad_clip", "optimizer", "muon_momentum", "muon_ns_steps",
          "micro_batch_size", "grad_accum_steps", "param_dtype", "autocast_dtype", "seed", "attn_implementation", "loss_impl", "gradient_checkpointing", "val_metric")
TOK = FakeGemmaTokenizer()


def cfg_of(name):
    return load_yaml(os.path.join(CFG_DIR, name + ".yaml"))


def test_all_expected_configs_exist():
    assert set(NAMES) == EXPECTED


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_strict_load_and_invariants(name):
    d = cfg_of(name)
    cfg = dataclass_from_dict(SFTConfig, d)                       # strict: an unknown key is an error
    validate_attention_config(cfg)
    assert cfg.hub_repo_id and cfg.hub_repo_id.startswith((os.environ['HF_ORG'] + "/")) and cfg.hub_private is True and "hub_private: true" in open(os.path.join(CFG_DIR, name + ".yaml")).read()
    assert cfg.early_stopping_patience == 0, "early stopping is inert"
    assert cfg.schedule_file and cfg.max_steps == 0 and cfg.schedule_require_sidecar is True
    assert cfg.language_alpha is None and cfg.unimax_budget_tokens == 0
    assert cfg.micro_batch_size * cfg.grad_accum_steps == 32, "32 packs of 4,096 tokens = the width of the schedule files"
    assert cfg.optimizer == ("adamw" if name == "sanity_adamw" else "muon")
    assert cfg.out_dir == f"runs/{name}" and cfg.run_name == name
    assert cfg.resume and cfg.hub_push_best_only is False
    assert cfg.hub_push_every % cfg.eval_every == 0, "a push can only happen at an evaluation"
    assert (cfg.embed_only_steps > 0) == (name == "s2")
    assert (cfg.init_from is not None) == (name == "a0_control")


def test_repos_are_distinct_and_hyperparameters_are_shared_where_the_preregistration_says():
    cfgs = {n: cfg_of(n) for n in EXPECTED}
    repos = [c["hub_repo_id"] for c in cfgs.values()]
    assert len(set(repos)) == len(repos)
    arms = ("a0", "s1", "s2", "g0", "a2_base", "a2_up6", "a2_up12", "a2_1b", "sanity_muon", "a0_control", "x1")
    for k in SHARED:
        assert len({json.dumps(cfgs[a][k]) for a in arms}) == 1, k
    assert cfgs["sanity_adamw"]["learning_rate"] == cfgs["sanity_muon"]["learning_rate"]
    assert {k: v for k, v in cfgs["sanity_adamw"].items() if k not in ("optimizer", "hub_repo_id", "out_dir", "run_name")} == \
           {k: v for k, v in cfgs["sanity_muon"].items() if k not in ("optimizer", "hub_repo_id", "out_dir", "run_name")}, "the sanity pair differs only in the optimizer"
    skip = ("model_name", "tokenizer_name", "hub_repo_id", "out_dir", "run_name")
    for arm in ("a2_up6", "a2_up12", "a2_1b"):
        assert {k: v for k, v in cfgs["a2_base"].items() if k not in skip} == {k: v for k, v in cfgs[arm].items() if k not in skip}, f"a2_base and {arm} differ only in the model"
        assert cfgs[arm]["schedule_file"] == cfgs["a2_base"]["schedule_file"] == "sched/A2/schedule.npy", "one shared subset file"
    assert cfgs["a2_1b"]["model_name"] == "google/gemma-3-1b-it" and cfgs["a2_1b"]["tokenizer_name"] == "google/gemma-3-270m-it"
    assert cfgs["p1"]["learning_rate"] < cfgs["a0"]["learning_rate"]
    assert cfgs["s2"]["embed_only_min_row"] == 256001 and 0 < cfgs["s2"]["embed_only_steps"] <= 200


# ------------------------------------------------------------------------------------------------ CPU dry run of every config
@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    d = tmp_path_factory.mktemp("cfg_toy")
    raw = write_jsonl(str(d / "raw.jsonl"), synthetic_rows(900, isos=("yor", "hau", "ibo", "amh", "fon", "ewe"), seed=5))
    prepare(PrepareConfig(input=raw, out_dir=str(d / "packed"), seq_len=128, val_fraction=0.2, seed=1, pack_by_language=True), tokenizer=TOK)
    ds = PackedDataset(str(d / "packed"), "train")
    sched, _ = balanced_schedule(ds.pack_languages(), 2, 32, seed=1)                  # 2 steps of 32 packs
    sched = np.ascontiguousarray(sched, dtype=np.int64)
    np.save(d / "schedule.npy", sched)
    json.dump({"schedule_sha256": hashlib.sha256(sched.tobytes()).hexdigest(), "steps": 2, "per_step": 32, "n_packs": len(ds),
               "pack_lang_sha256": pack_lang_sha256(ds.pack_languages()), "packed_structure_sha256": packed_structure_sha256(ds)}, open(d / "schedule.json", "w"))
    assert len(ds) >= 32
    return d


def _dry(name, toy, tmp_path, **extra):
    d = cfg_of(name)
    d.update(model_name="tiny-random", tokenizer_name=None, data_dir=str(toy / "packed"), schedule_file=str(toy / "schedule.npy"), out_dir=str(tmp_path / name),
             hub_repo_id=None, autocast_dtype="none", device="cpu", init_from=None, log_every=1)
    if name == "s2":
        d.update(embed_only_steps=1, embed_only_min_row=480)                           # the tiny vocabulary has 512 rows
    d.update(extra)
    return train(dataclass_from_dict(SFTConfig, d), tokenizer=TOK)


@pytest.mark.parametrize("name", sorted(EXPECTED - {"a0_control"}))
def test_config_runs_two_steps_on_cpu(name, toy, tmp_path):
    out = _dry(name, toy, tmp_path)
    assert out["steps"] == out["total_steps"] == 2 and out["schedule"]["steps"] == 2
    rec = [json.loads(l) for l in open(tmp_path / name / "metrics.jsonl")]
    assert all(np.isfinite(r["train_loss"]) for r in rec if "train_loss" in r)
    man = json.load(open(tmp_path / name / "schedule_manifest.json"))
    assert man["steps"] == 2 and man["draws"] == 64 and "packed_structure_sha256" in man["checks_passed"]


def test_a0_control_continues_from_an_a0_checkpoint(toy, tmp_path):
    _dry("a0", toy, tmp_path)
    ck = str(tmp_path / "a0" / "last" / "trainer_state.pt")
    assert os.path.exists(ck)
    out = _dry("a0_control", toy, tmp_path, init_from=ck)
    st = torch.load(tmp_path / "a0_control" / "last" / "trainer_state.pt", weights_only=False)["state"]
    assert out["steps"] == 4 and st["run_start"] == 2 and out["total_steps"] == 4, "the step counter continues from A0; the new phase has its own 2-step schedule"


@pytest.mark.parametrize("arm,k", [("a2_up6", 6), ("a2_up12", 12)])
def test_up_arms_document_the_post_norm_gain_in_their_build_command(arm, k):
    txt = open(os.path.join(CFG_DIR, arm + ".yaml")).read()
    assert f"--n-new {k} --post-norm-gain 0.01" in txt
