# SPDX-License-Identifier: AGPL-3.0-or-later
"""Growth 1 (2026-11-01): staged unfreezing in the SFT trainer (freeze_regex, train_embedding_rows / train_embedding_tags), the freeze pattern of stage A, the tag-row initialisation
and the stage configs / schedule splitter. CPU, tiny random Gemma3 (6 layers), toy packed set; nothing here needs a GPU."""
import hashlib
import importlib.util
import json
import os
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gemma_mt.data.packed_dataset import PackedDataset, balanced_schedule
from gemma_mt.data.prepare_sft import PrepareConfig, prepare
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
from gemma_mt.growth import freeze_regex_new_layers_only, init_tag_rows, new_layer_indices, tag_token_ids
from gemma_mt.languages import ALL_ISOS
from gemma_mt.smoke import synthetic_rows, write_jsonl
from gemma_mt.train_sft import SFTConfig, apply_freeze, packed_structure_sha256, pack_lang_sha256, train

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ISOS = ("yor", "hau", "ibo", "amh", "fon", "ewe")
PER_STEP = 4                                                   # micro_batch_size 2 x grad_accum_steps 2
TOK = FakeGemmaTokenizer()
NEW = [1, 4]                                                   # "new layers" of the tiny 6-layer model
TAG_ROWS = [300, 301, 302]


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    d = tmp_path_factory.mktemp("growth_toy")
    raw = write_jsonl(str(d / "raw.jsonl"), synthetic_rows(600, isos=ISOS, seed=5))
    prepare(PrepareConfig(input=raw, out_dir=str(d / "packed"), seq_len=128, val_fraction=0.2, seed=1, pack_by_language=True), tokenizer=TOK)
    return str(d / "packed")


def _cfg(tmp_path, packed_dir, out, **kw):
    base = dict(model_name="tiny-random", data_dir=packed_dir, out_dir=str(tmp_path / out), autocast_dtype="none",
                micro_batch_size=2, grad_accum_steps=2, epochs=50, max_steps=0, learning_rate=2e-3, warmup_steps=2,
                eval_every=0, save_every=100, log_every=1, early_stopping_patience=0, device="cpu", seed=3, weight_decay=0.1)
    base.update(kw)
    return SFTConfig(**base)


def _schedule(tmp_path, name, ds, steps, seed=1):
    sched, _ = balanced_schedule(ds.pack_languages(), steps, PER_STEP, seed=seed)
    d = tmp_path / name
    d.mkdir()
    sched = np.ascontiguousarray(sched, dtype=np.int64)
    np.save(d / "schedule.npy", sched)
    json.dump({"schedule_sha256": hashlib.sha256(sched.tobytes()).hexdigest(), "steps": int(sched.shape[0]), "per_step": int(sched.shape[1]), "n_packs": len(ds),
               "pack_lang_sha256": pack_lang_sha256(ds.pack_languages()), "packed_structure_sha256": packed_structure_sha256(ds)}, open(d / "schedule.json", "w"))
    return str(d / "schedule.npy")


def _init_state():
    from gemma_mt.modeling import build_tiny_model
    return {k: v.detach().clone() for k, v in build_tiny_model(attn_implementation="sdpa", dtype=torch.float32).state_dict().items()}


def _model(tmp_path, out):
    return torch.load(tmp_path / out / "last" / "trainer_state.pt", weights_only=False)["model"]


def _layer_of(k):
    m = re.match(r"model\.layers\.(\d+)\.", k)
    return int(m.group(1)) if m else None


def _stage_a(tmp_path, packed, optimizer, name="a"):
    ds = PackedDataset(packed, "train")
    f = _schedule(tmp_path, "sa_" + name, ds, 3)
    return train(_cfg(tmp_path, packed, name, schedule_file=f, optimizer=optimizer, freeze_regex=freeze_regex_new_layers_only(NEW), train_embedding_rows=TAG_ROWS), tokenizer=TOK)


# ------------------------------------------------------------------------------------------------ the freeze pattern
def test_freeze_regex_new_layers_only_matches_exactly_what_it_should():
    rx = re.compile(freeze_regex_new_layers_only([2, 6, 31]))
    frozen = ["model.layers.0.self_attn.q_proj.weight", "model.layers.3.mlp.down_proj.weight", "model.layers.20.input_layernorm.weight",
              "model.layers.12.self_attn.o_proj.weight", "model.norm.weight", "lm_head.weight"]
    live = ["model.layers.2.self_attn.q_proj.weight", "model.layers.6.mlp.down_proj.weight", "model.layers.31.post_feedforward_layernorm.weight",
            "model.embed_tokens.weight"]
    assert all(rx.search(n) for n in frozen), [n for n in frozen if not rx.search(n)]
    assert not any(rx.search(n) for n in live), [n for n in live if rx.search(n)]
    assert not rx.search("model.layers.2.mlp.up_proj.weight") and rx.search("model.layers.22.mlp.up_proj.weight"), "2 must not match 22"


def test_apply_freeze_counts_and_default_is_a_noop():
    from gemma_mt.modeling import build_tiny_model
    m = build_tiny_model(attn_implementation="sdpa")
    total = sum(p.numel() for p in m.parameters())
    assert apply_freeze(m, None) == {"frozen": 0, "trainable": total}
    r = apply_freeze(m, freeze_regex_new_layers_only(NEW))
    assert r["frozen"] + r["trainable"] == total and 0 < r["trainable"] < total
    for n, p in m.named_parameters():
        assert p.requires_grad == (_layer_of(n) in NEW or "embed_tokens" in n), n


# ------------------------------------------------------------------------------------------------ stage A: new layers + tag rows only
@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_stage_a_moves_only_the_new_layers_and_the_tag_rows(tmp_path, packed, optimizer):
    if optimizer == "muon" and not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon not available")
    out = _stage_a(tmp_path, packed, optimizer)
    assert out["steps"] == 3
    init, now = _init_state(), _model(tmp_path, "a")
    for k in init:
        if k == "model.embed_tokens.weight" or k == "lm_head.weight":
            e0, e1 = init["model.embed_tokens.weight"], now["model.embed_tokens.weight"]
            other = torch.ones(e0.shape[0], dtype=torch.bool); other[TAG_ROWS] = False
            assert torch.equal(e0[other], e1[other]), "an embedding row outside the tag rows moved (weight decay / momentum / gradient leak)"
            assert not torch.equal(e0[TAG_ROWS], e1[TAG_ROWS]), "the tag rows must move (the output softmax gives every row a gradient)"
        elif _layer_of(k) in NEW:
            assert not torch.equal(init[k], now[k]), f"new layer parameter {k} did not move"
        else:
            assert torch.equal(init[k], now[k]), f"frozen parameter {k} moved"


def test_stage_b_continues_from_stage_a_and_trains_everything(tmp_path, packed):
    _stage_a(tmp_path, packed, "adamw")
    ds = PackedDataset(packed, "train")
    f = _schedule(tmp_path, "sb", ds, 3, seed=2)
    a_state = _model(tmp_path, "a")
    out = train(_cfg(tmp_path, packed, "b", schedule_file=f, init_from=str(tmp_path / "a" / "last" / "trainer_state.pt")), tokenizer=TOK)
    assert out["steps"] == 6, "the step counter continues, the phase has its own length"
    b_state = _model(tmp_path, "b")
    moved = [k for k in a_state if not torch.equal(a_state[k], b_state[k])]
    assert {_layer_of(k) for k in moved if _layer_of(k) is not None} == set(range(6)), "stage B must train every layer"
    assert "model.norm.weight" in moved and "model.embed_tokens.weight" in moved


def test_default_config_is_unchanged_every_parameter_trains(tmp_path, packed):
    ds = PackedDataset(packed, "train")
    f = _schedule(tmp_path, "sd", ds, 2)
    train(_cfg(tmp_path, packed, "d", schedule_file=f), tokenizer=TOK)
    init, now = _init_state(), _model(tmp_path, "d")
    assert all(not torch.equal(init[k], now[k]) for k in init if k != "lm_head.weight"), [k for k in init if torch.equal(init[k], now[k])]


def test_resume_of_a_stage_a_run_is_exact(tmp_path, packed):
    ds = PackedDataset(packed, "train")
    f = _schedule(tmp_path, "sr", ds, 4)
    kw = dict(schedule_file=f, freeze_regex=freeze_regex_new_layers_only(NEW), train_embedding_rows=TAG_ROWS, save_every=1)
    train(_cfg(tmp_path, packed, "full", **kw), tokenizer=TOK)
    train(_cfg(tmp_path, packed, "split", stop_after_steps=2, **kw), tokenizer=TOK)
    train(_cfg(tmp_path, packed, "split", **kw), tokenizer=TOK)
    a, b = _model(tmp_path, "full"), _model(tmp_path, "split")
    assert max((a[k].float() - b[k].float()).abs().max().item() for k in a) == 0.0


def test_bad_freeze_settings_are_refused_before_training(tmp_path, packed):
    ds = PackedDataset(packed, "train")
    f = _schedule(tmp_path, "se", ds, 1)
    with pytest.raises(ValueError, match="freezes every parameter"):
        train(_cfg(tmp_path, packed, "e1", schedule_file=f, freeze_regex=r"."), tokenizer=TOK)
    with pytest.raises(ValueError, match="embedding matrix is frozen"):
        train(_cfg(tmp_path, packed, "e2", schedule_file=f, freeze_regex=r"embed_tokens", train_embedding_rows=[300]), tokenizer=TOK)
    with pytest.raises(ValueError, match="ids must lie"):
        train(_cfg(tmp_path, packed, "e3", schedule_file=f, train_embedding_rows=[10 ** 6]), tokenizer=TOK)


def test_train_embedding_tags_selects_the_125_tag_rows(tmp_path, packed, monkeypatch):
    import gemma_mt.growth as growth
    ds = PackedDataset(packed, "train")
    f = _schedule(tmp_path, "st", ds, 2)
    table = {iso: 300 + i for i, iso in enumerate(ALL_ISOS)}                   # the byte-level fake tokenizer has no far-end <unusedK> slots: the id table is injected
    monkeypatch.setattr(growth, "tag_token_ids", lambda tok: table)
    ids = sorted(table.values())
    assert len(ids) == len(ALL_ISOS) == 125
    train(_cfg(tmp_path, packed, "t", schedule_file=f, freeze_regex=r"^(?!model\.embed_tokens\.)", train_embedding_tags=True), tokenizer=TOK)
    init, now = _init_state(), _model(tmp_path, "t")
    e0, e1 = init["model.embed_tokens.weight"], now["model.embed_tokens.weight"]
    other = torch.ones(e0.shape[0], dtype=torch.bool); other[ids] = False
    assert torch.equal(e0[other], e1[other]) and not torch.equal(e0[ids], e1[ids])
    assert all(torch.equal(init[k], now[k]) for k in init if "embed_tokens" not in k and k != "lm_head.weight")


# ------------------------------------------------------------------------------------------------ tag-row initialisation
class _TagTok:
    """Minimal tokenizer: the tag token of table row i (gemma_mt.tags.tag_token: <unused0..98> and, from row 99 on, <unused6241> downward) has id 300+i, plain text = the code
    points of the characters modulo 200."""
    unk_token_id = 3

    def __init__(self):
        from gemma_mt.tags import tag_token
        self.ids = {tag_token(iso): 300 + i for i, iso in enumerate(ALL_ISOS)}

    def convert_tokens_to_ids(self, t):
        return self.ids.get(t, self.unk_token_id)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 200 for c in text]}


def test_tag_tokens_follow_the_lead_mapping_and_are_distinct():
    from gemma_mt.tags import tag_token
    assert tag_token(ALL_ISOS[0]) == "<unused0>" and tag_token(ALL_ISOS[98]) == "<unused98>"
    assert tag_token(ALL_ISOS[99]) == "<unused6241>" and tag_token(ALL_ISOS[124]) == "<unused6216>"
    ids = tag_token_ids(_TagTok())
    assert len(ids) == len(set(ids.values())) == 125 and ids[ALL_ISOS[99]] == 399


def test_init_tag_rows_sets_the_mean_of_the_old_subtokens_and_nothing_else(tmp_path):
    from safetensors.torch import load_file, save_file
    from gemma_mt.modeling import build_tiny_model
    sd = {k: v.detach().clone().contiguous() for k, v in build_tiny_model(attn_implementation="sdpa", dtype=torch.float32).state_dict().items() if k != "lm_head.weight"}
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    save_file(sd, str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text("{}")
    tab = init_tag_rows(str(src), str(out), tokenizer=_TagTok())
    new = load_file(str(out / "model.safetensors"))
    assert (out / "config.json").exists() and len(tab) == 125
    for iso, t in tab.items():
        want = sd["model.embed_tokens.weight"][t["old_subtoken_ids"]].mean(0)
        assert torch.allclose(new["model.embed_tokens.weight"][t["token_id"]], want, atol=1e-6), iso
    ids = [t["token_id"] for t in tab.values()]
    other = torch.ones(512, dtype=torch.bool); other[ids] = False
    assert torch.equal(new["model.embed_tokens.weight"][other], sd["model.embed_tokens.weight"][other])
    assert all(torch.equal(new[k], sd[k]) for k in sd if k != "model.embed_tokens.weight")


# ------------------------------------------------------------------------------------------------ the real growth-1 layout and the stage configs
def _load_upcycle():
    spec = importlib.util.spec_from_file_location("upcycle_layers", os.path.join(ROOT, "scripts", "upcycle_layers.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_growth1_layout_is_39_layers_with_pattern_13_and_the_stage_a_regex_matches_the_config():
    up = _load_upcycle()
    types = ["sliding_attention"] * 24
    for g in (7, 15, 23):
        types[g] = "full_attention"                                        # A0 = up6: 24 layers, pattern 8
    after = up.plan_positions(24, 15, types)                               # the evenly spaced formula would land on global layer 7 -> spread inside the three runs of 7
    assert after == [0, 2, 3, 4, 6, 8, 10, 11, 12, 14, 16, 18, 19, 20, 22]
    lm = up.layer_map(24, after)
    new_types = [types[s] if not new else "sliding_attention" for s, new in lm]
    assert len(new_types) == 39 and up.sliding_pattern(new_types) == 13 and [i for i, t in enumerate(new_types) if t == "full_attention"] == [12, 25, 38]
    info = {"layers": [{"new_index": i, "new": bool(new)} for i, (s, new) in enumerate(lm)]}
    idx = new_layer_indices(info)
    assert idx == [1, 4, 6, 8, 11, 14, 17, 19, 21, 24, 27, 30, 32, 34, 37] and len(idx) == 15
    cfg_a = open(os.path.join(ROOT, "configs", "growth1", "g1a.yaml")).read()
    m = re.search(r"^freeze_regex:\s*'(.*)'\s*(?:#.*)?$", cfg_a, flags=re.M)
    assert m and m.group(1) == freeze_regex_new_layers_only(idx), "configs/growth1/g1a.yaml freeze_regex must match the 15 new layers"
    assert 301539968 + 15 * 5573632 == 385144448                           # A0 + 15 layers (parameters_per_layer of the real upcycle.json)


def test_stage_configs_load_and_differ_only_where_intended():
    from gemma_mt.utils import dataclass_from_dict, load_yaml
    a = dataclass_from_dict(SFTConfig, load_yaml(os.path.join(ROOT, "configs", "growth1", "g1a.yaml")))
    b = dataclass_from_dict(SFTConfig, load_yaml(os.path.join(ROOT, "configs", "growth1", "g1b.yaml")))
    assert a.train_embedding_tags and a.freeze_regex and not b.freeze_regex and not b.train_embedding_tags and not a.init_from and b.init_from == "runs/g1a/last/trainer_state.pt"
    for c in (a, b):
        assert c.optimizer == "muon" and c.param_dtype == "float32" and c.autocast_dtype == "bfloat16" and c.save_every == 20 and c.hub_push_every == 500
        assert c.micro_batch_size == 8 and c.grad_accum_steps == 4 and c.schedule_require_sidecar
        assert c.keep_single_checkpoint and c.keep_prev_checkpoint and c.resume
    assert (a.data_dir, a.schedule_file) == ("packed_g0", "sched_g/A/schedule.npy") and (b.data_dir, b.schedule_file) == ("packed_g0", "sched_g/B/schedule.npy")
    assert a.model_name == b.model_name == "ckpt/gemma3-up6-g15-tags" and a.hub_repo_id != b.hub_repo_id
    assert b.warmup_ratio == pytest.approx(0.03)


# ------------------------------------------------------------------------------------------------ the schedule splitter (data_pipeline/sft_clean/fresh/split_schedule.py)
def _load_splitter(monkeypatch):
    path = os.path.abspath(os.path.join(ROOT, "data_pipeline", "sft_clean", "fresh", "split_schedule.py"))
    if not os.path.exists(path) or not os.path.exists(os.path.join(os.path.dirname(path), "..", "chain", "sampler.py")):
        pytest.skip("data_pipeline/sft_clean/{fresh,chain} are not next to gemma_mt")
    monkeypatch.setenv("GEMMA_MT_PKG", os.path.abspath(ROOT))
    spec = importlib.util.spec_from_file_location("split_schedule", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_split_schedule_cuts_one_schedule_into_two_phases_the_trainer_accepts(tmp_path, packed, monkeypatch):
    mod = _load_splitter(monkeypatch)
    ds = PackedDataset(packed, "train")
    parent = _schedule(tmp_path, "parent", ds, 6)
    mod.main(["--sched", os.path.dirname(parent), "--out", str(tmp_path / "sched_g"), "--stage-a-steps", "2", "--packed", packed])
    full = np.load(parent)
    a, b = np.load(tmp_path / "sched_g" / "G1A" / "schedule.npy"), np.load(tmp_path / "sched_g" / "G1B" / "schedule.npy")
    assert a.shape == (2, PER_STEP) and b.shape == (4, PER_STEP) and np.array_equal(np.concatenate([a, b]), full)
    for name, arr in (("G1A", a), ("G1B", b)):
        sc = json.load(open(tmp_path / "sched_g" / name / "schedule.json"))
        assert sc["schedule_sha256"] == hashlib.sha256(np.ascontiguousarray(arr, dtype=np.int64).tobytes()).hexdigest() and sc["steps"] == arr.shape[0] and sc["split_from"]["parent_steps"] == 6
        assert sc["packed_structure_sha256"] == packed_structure_sha256(ds) and sc["pack_lang_sha256"] == pack_lang_sha256(ds.pack_languages())
    # the trainer takes both phases with its sidecar checks ON (schedule_require_sidecar defaults to True) and stage B continues stage A
    out_a = train(_cfg(tmp_path, packed, "ra", schedule_file=str(tmp_path / "sched_g" / "G1A" / "schedule.npy"),
                       freeze_regex=freeze_regex_new_layers_only(NEW), train_embedding_rows=TAG_ROWS), tokenizer=TOK)
    out_b = train(_cfg(tmp_path, packed, "rb", schedule_file=str(tmp_path / "sched_g" / "G1B" / "schedule.npy"),
                       init_from=str(tmp_path / "ra" / "last" / "trainer_state.pt")), tokenizer=TOK)
    assert out_a["steps"] == 2 and out_b["steps"] == 6
    with pytest.raises(SystemExit):                                              # a stage boundary outside the schedule is refused
        mod.main(["--sched", os.path.dirname(parent), "--out", str(tmp_path / "x"), "--stage-a-steps", "6", "--packed", packed])
