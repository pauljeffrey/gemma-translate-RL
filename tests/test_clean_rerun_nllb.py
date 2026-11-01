# SPDX-License-Identifier: AGPL-3.0-or-later
"""2026-10-29 evening: generator support for the NLLB pseudo-reference stages, the early-stop numbers of the stage yamls, and the replay of the prompts a stage draws."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, SCRIPTS)


@pytest.fixture()
def gen():
    import clean_rerun_make_configs as g

    saved = (list(g.NLLB_STAGES), g.NLLB_LANG_FILE, g.OPTIMIZER)
    yield g
    g.NLLB_STAGES[:] = saved[0]
    g.NLLB_LANG_FILE, g.OPTIMIZER = saved[1], saved[2]


def _yaml(gen, stage, spec=False):
    import yaml

    return yaml.safe_load(gen.render(stage, "specialist" if spec else "a0", gen.paths("/workspace/clean_rerun")))


def test_default_nllb_stages_and_language_file(gen):
    d = json.load(open(gen.NLLB_LANG_FILE))
    assert d["provisional"] is True and d["languages"] == ["kea", "kbp", "nus", "bam", "kab"]
    on = {st: _yaml(gen, st)["reward"] for st in gen.STAGES}
    for st in ("preflight", "A", "B1", "mixed", "final"):
        assert on[st]["nllb_signal"] is True and on[st]["nllb_languages"] == d["languages"], st
    for st in ("B2", "B3"):
        assert on[st]["nllb_signal"] is False and on[st]["nllb_languages"] == [], st
    assert on["final"]["nllb_modes"] == ["consensus", "blend"] and on["A"]["nllb_pseudoref_dir"].endswith("pseudorefs_reward_only_nc_v1")
    assert on["A"]["nllb_weight_default"] == 0.35 and on["A"]["nllb_weights"] == {"kon": 0.0}


def test_stage_list_and_language_file_are_options(gen, tmp_path):
    f = tmp_path / "nl.json"
    f.write_text(json.dumps({"provisional": False, "languages": ["kea", "nus", "yor", "swh"], "weights": {"nus": 0.2}}))
    gen.NLLB_LANG_FILE = str(f)
    gen.NLLB_STAGES[:] = ["final"]
    assert _yaml(gen, "A")["reward"]["nllb_signal"] is False and _yaml(gen, "preflight")["reward"]["nllb_signal"] is False
    r = _yaml(gen, "final")["reward"]
    assert r["nllb_signal"] is True and r["nllb_languages"] == ["kea", "nus", "yor", "swh"] and r["nllb_weights"] == {"kon": 0.0, "nus": 0.2}
    rs = _yaml(gen, "final", spec=True)["reward"]                         # the specialist arm keeps its own 20 languages only (swh is not one of them)
    assert rs["nllb_languages"] == ["kea", "nus", "yor"]
    txt = gen.render("final", "a0", gen.paths("/x"))
    assert "PROVISIONAL" not in txt.split("nllb_languages")[1].split("\n")[0]
    gen.NLLB_STAGES[:] = []
    assert all(_yaml(gen, st)["reward"]["nllb_signal"] is False for st in gen.STAGES)


def test_cli_refuses_long_bins_without_the_override(gen, tmp_path, capsys):
    with pytest.raises(SystemExit):
        gen.main(["--out", str(tmp_path), "--nllb-stages", "A,B2"])
    with pytest.raises(SystemExit):
        gen.main(["--out", str(tmp_path), "--nllb-stages", "A,nope"])
    assert gen.main(["--out", str(tmp_path), "--nllb-stages", "A,B1"]) == 0
    assert "nllb_signal: true" in open(tmp_path / "rl_A_b64.yaml").read() and "nllb_signal: false" in open(tmp_path / "rl_C_mixed.yaml").read()
    assert gen.main(["--out", str(tmp_path), "--nllb-stages", "A,B2", "--nllb-allow-long"]) == 0
    assert "nllb_signal: true" in open(tmp_path / "rl_B2_b512.yaml").read()


def test_early_stop_values_of_the_evening_decision_and_extension_blocks(gen):
    want = {"A": (150, 150, 400, 100), "B1": (80, 80, 150, 40), "B2": (50, 30, 80, 20), "B3": (40, 30, 60, 15), "mixed": (40, 30, 60, 15), "final": (80, 60, 150, 40)}
    for st, (pat, mn, cap, ext) in want.items():
        y = _yaml(gen, st)
        assert (y["early_stop_patience"], y["early_stop_min_steps"], y["steps"], y["extend_steps"]) == (pat, mn, cap, ext), st
        assert y["early_stop"] is True and y["extend_if_improving"] is True and y["extend_max_frac"] == 0.5 and y["optimizer"] == "muon" and y["eval_vllm_only"] is True
    assert _yaml(gen, "final")["planned_winddown_last"] == 30 and _yaml(gen, "preflight")["early_stop"] is False
    y = _yaml(gen, "A", spec=True)                                       # specialist: scaled by 0.4367 with floors
    assert (y["steps"], y["early_stop_patience"], y["early_stop_min_steps"]) == (175, 65, 65)
    for st in gen.STAGES:
        for sp in (False, True):
            y = _yaml(gen, st, sp)
            assert y["hub_private"] is True and (y["hub_repo_id"] is None) == (st == "preflight") and (y["hub_repo_id"] or (os.environ['HF_ORG'] + "/")).startswith((os.environ['HF_ORG'] + "/"))
            from gemma_mt.rl.train_rl import RLConfig, validate_config
            from gemma_mt.utils import dataclass_from_dict

            validate_config(dataclass_from_dict(RLConfig, y))


# ------------------------------------------------------------------------------------------------------------------ draws
def _pool(tmp_path):
    from gemma_mt.smoke import ENGLISH, write_jsonl

    rows = []
    for l in ("kea", "kbp", "yor"):
        for b, tok in (("b64", 12), ("b256", 100), ("b512", 300)):
            for i in range(40):
                rows.append({"english": f"{ENGLISH[i % len(ENGLISH)]} {l} {b} {i}. Second sentence {i}.", "iso": l, "language": "", "src_afr_tokens": tok, "source_dataset": "smoke", "length_bin": b})
    return write_jsonl(str(tmp_path / "pool.jsonl"), rows)


def _cfg(tmp_path, **kw):
    from gemma_mt.rl.train_rl import RLConfig

    base = dict(prompts_path=_pool(tmp_path), max_src_tokens=2048, length_bins=True, curriculum_steps={"b64": 8}, steps=8, eval_prompts=6, eval_min_per_bin=1, prompts_per_step_by_bin={"b64": 6, "mixed": 6},
                early_stop=True, early_stop_patience=30, early_stop_min_steps=30, extend_if_improving=True, extend_steps=4, run_controls=True)
    base.update(kw)
    return RLConfig(**base)


def test_draws_replay_matches_the_trainer_schedule_restricts_languages_bins_and_covers_the_extension(tmp_path):
    from clean_rerun_nllb_draws import draws_of
    from gemma_mt.rl.data import load_rl_prompts
    from gemma_mt.rl.train_rl import _subset, build_prompt_schedule, split_eval_prompts

    cfg = _cfg(tmp_path)
    prompts = load_rl_prompts(cfg.prompts_path, max_src_tokens=cfg.max_src_tokens)
    tr, _ = split_eval_prompts(prompts, 6, cfg.seed, cfg.eval_min_per_bin)
    train = _subset(prompts, tr)
    sch = build_prompt_schedule(cfg, train)
    assert sch.bin_of_step(11) == "b64"                                   # the +50 % extension keeps the bin
    base = draws_of(train, sch, cfg, ["kea", "kbp"], 8, "b256")
    ext = draws_of(train, sch, cfg, ["kea", "kbp"], 12, "b256")
    assert set(base) == {"kea", "kbp"} and all(base[l] <= ext[l] for l in base) and sum(len(v) for v in ext.values()) > sum(len(v) for v in base.values())
    for l, idx in ext.items():
        assert idx and all(train.iso[i] == l and train.length_bin[i] == "b64" for i in idx)         # only the stage's bin is drawn
    # the draws of the first 8 steps are exactly what the trainer's loop would take
    manual = set()
    for step in range(8):
        manual |= {i for i in sch.indices(step, cfg.prompts_per_step_by_bin["b64"]) if train.iso[i] in ("kea", "kbp")}
    assert manual == set().union(*base.values())
    none = draws_of(train, sch, cfg, ["kea"], 8, None)
    assert none["kea"] == base["kea"]


def test_max_bin_drops_long_windows_in_a_mixed_stage(tmp_path):
    from clean_rerun_nllb_draws import draws_of
    from gemma_mt.rl.data import load_rl_prompts
    from gemma_mt.rl.train_rl import build_prompt_schedule

    cfg = _cfg(tmp_path, curriculum_steps={"b64": 0}, steps=30, extend_if_improving=False, early_stop=False, prompts_per_step_by_bin={"b64": 6, "mixed": 9})
    prompts = load_rl_prompts(cfg.prompts_path, max_src_tokens=cfg.max_src_tokens)
    sch = build_prompt_schedule(cfg, prompts)
    allb = draws_of(prompts, sch, cfg, ["kea"], 30, "b512")
    low = draws_of(prompts, sch, cfg, ["kea"], 30, "b256")
    assert {prompts.length_bin[i] for i in allb["kea"]} == {"b64", "b256", "b512"}
    assert {prompts.length_bin[i] for i in low["kea"]} == {"b64", "b256"} and low["kea"] < allb["kea"]


def test_stage_draws_end_to_end_from_a_yaml_and_the_cli_summary(tmp_path):
    pytest.importorskip("pyarrow")
    import yaml

    import clean_rerun_make_configs as g
    from clean_rerun_nllb_draws import main

    cfgdir = tmp_path / "cfg"
    cfgdir.mkdir()
    pool = _pool(tmp_path)
    import dataclasses

    for st, S in g.STAGES.items():
        y = _yaml(g, st)
        if st == "A":
            y.update(prompts_path=pool, max_src_tokens=2048, steps=8, curriculum_steps={"b64": 8}, eval_prompts=6, eval_min_per_bin=1, prompts_per_step_by_bin={"b64": 6, "mixed": 6},
                     early_stop_patience=30, early_stop_min_steps=30, extend_steps=4, plateau_stop=False, planned_winddown_last=0)
        (cfgdir / S["file"]).write_text(yaml.safe_dump(y))
    nl = tmp_path / "nl.json"
    nl.write_text(json.dumps({"provisional": False, "languages": ["kea", "kbp"]}))
    out = tmp_path / "out"
    # only stage A points at the tiny pool: the other yamls have nllb_signal on too, so give them the pool as well
    for st, S in g.STAGES.items():
        y = yaml.safe_load((cfgdir / S["file"]).read_text())
        if st != "A":
            y.update(prompts_path=pool, max_src_tokens=2048, steps=8, curriculum_steps={"b64": 0}, eval_prompts=6, eval_min_per_bin=1, prompts_per_step_by_bin={"b64": 6, "mixed": 6}, domain_weights={},
                     domain_bin_weights={}, language_floor_steps=0, early_stop_patience=30, early_stop_min_steps=30, extend_steps=4, plateau_stop=False, planned_winddown_last=0, language_weights={})
            (cfgdir / S["file"]).write_text(yaml.safe_dump(y))
    assert main(["--configs", str(cfgdir), "--out", str(out), "--languages-file", str(nl)]) == 0
    summ = json.load(open(out / "draws_summary.json"))
    assert summ["languages"] == ["kea", "kbp"] and summ["unique_prompts_total"] > 0 and summ["sentences_total"] >= summ["unique_prompts_total"]
    import pyarrow.parquet as pq

    t = pq.read_table(out / "draws_pool" / "kea.parquet").to_pylist()
    assert t and all("kea" in r["english"] and "b512" not in r["english"] for r in t)               # b512 windows are beyond --max-bin b256
    assert "--pools" in summ["commands"]["precompute"] and "--max-batch-tokens 16384" in summ["commands"]["precompute"]


def test_owner_priority_languages_join_the_weak_draw_class_in_every_yaml(gen):
    assert gen.OWNER_PRIORITY == ["efi", "tiv", "bin", "ish"] and not set(gen.OWNER_PRIORITY) & set(gen.WEAK_LIST)
    for st in gen.STAGES:
        lw = _yaml(gen, st)["language_weights"]
        assert set(lw) == set(gen.WEAK_LIST) | set(gen.OWNER_PRIORITY) and len(lw) == 29 and set(lw.values()) == {2.5}, st
        sp = _yaml(gen, st, spec=True)["language_weights"]
        assert set(gen.OWNER_PRIORITY) <= set(sp) and set(gen.SPECIALIST_LANGS) <= set(sp) and set(sp.values()) == {2.5}, st
    assert "OWNER-PRIORITY" in gen.render("A", "a0", gen.paths("/x"))
