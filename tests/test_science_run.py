# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-52 (science run): the domain mixture of PromptSchedule, the `domain` prompt column, the science-anchor reward, the resume selection reset and the aux dev sets."""
import os
import collections
import json

import numpy as np
import pytest

from gemma_mt.rl import sci_anchor
from gemma_mt.rl.data import MIXED, PromptSchedule, load_rl_prompts
from gemma_mt.rl.reward import RewardConfig, Sample
from gemma_mt.smoke import write_jsonl
from test_reward import ConstScorer, ids, make


def _pool(per_cell=40, langs=("yor", "hau", "zul"), bins=("b64", "b256", "b512", "b1024", "b2048"), science_bins=("b256", "b512", "b1024", "b2048")):
    iso, lb, dom = [], [], []
    for d in ("science", "general"):
        for b in (science_bins if d == "science" else bins):
            for l in langs:
                for _ in range(per_cell):
                    iso.append(l); lb.append(b); dom.append(d)
    return iso, lb, dom


def _sched(**kw):
    iso, lb, dom = _pool()
    base = dict(alpha=0.0, seed=7, length_bin=lb, curriculum={"b256": 3, "b512": 3}, domain=dom, domain_weights={"science": 0.7, "general": 0.3})
    base.update(kw)
    return PromptSchedule(iso, **base), iso, lb, dom


def test_domain_share_and_science_bin_weights_in_the_mixed_stage():
    s, iso, lb, dom = _sched(domain_bin_weights={"science": {"b256": 0.1, "b512": 0.2, "b1024": 0.3, "b2048": 0.4}})
    assert s.bin_of_step(6) == MIXED and s.bin_of_step(0) == "b256"
    picks = [i for step in range(6, 106) for i in s.indices(step, 40)]
    share = collections.Counter(dom[i] for i in picks)
    assert abs(share["science"] / len(picks) - 0.7) < 0.03
    sci_bins = collections.Counter(lb[i] for i in picks if dom[i] == "science")
    n = sum(sci_bins.values())
    assert abs(sci_bins["b2048"] / n - 0.4) < 0.04 and abs(sci_bins["b256"] / n - 0.1) < 0.03 and "b64" not in sci_bins
    gen_bins = collections.Counter(lb[i] for i in picks if dom[i] == "general")
    assert set(gen_bins) == {"b64", "b256", "b512", "b1024", "b2048"}            # default: uniform over the bins the domain has
    m = sum(gen_bins.values())
    assert all(abs(v / m - 0.2) < 0.05 for v in gen_bins.values())


def test_single_bin_stage_keeps_the_bin_and_the_mixture_and_is_deterministic():
    s, iso, lb, dom = _sched()
    p0 = s.indices(1, 200)
    assert {lb[i] for i in p0} == {"b256"} and 0.6 < sum(dom[i] == "science" for i in p0) / 200 < 0.8
    assert s.indices(9, 30) == s.indices(9, 30) and s.indices(9, 30) != s.indices(10, 30)
    a, b = s.indices(9, 20, rank=0, world=2), s.indices(9, 20, rank=1, world=2)
    assert sorted(a + b) == sorted(s.indices(9, 40))


def test_language_weights_apply_inside_the_domain_pool():
    s, iso, lb, dom = _sched(language_weights={"zul": 4.0})
    picks = [i for step in range(6, 60) for i in s.indices(step, 40)]
    c = collections.Counter(iso[i] for i in picks)
    assert c["zul"] / len(picks) == pytest.approx(4 / 6, abs=0.04)


def test_off_by_default_stream_is_untouched():
    iso, lb, dom = _pool()
    a = PromptSchedule(iso, alpha=0.0, seed=7, length_bin=lb, curriculum={"b256": 3, "b512": 3})
    b = PromptSchedule(iso, alpha=0.0, seed=7, length_bin=lb, curriculum={"b256": 3, "b512": 3}, domain=dom)      # labels without weights = no mixture
    assert all(a.indices(st, 25) == b.indices(st, 25) for st in range(0, 12))


def test_domain_errors():
    iso, lb, dom = _pool()
    with pytest.raises(ValueError):
        PromptSchedule(iso, length_bin=lb, domain_weights={"science": 1.0})                  # no labels
    with pytest.raises(ValueError):
        PromptSchedule(iso, length_bin=lb, domain=dom, domain_weights={"history": 1.0})      # nobody has that label
    with pytest.raises(ValueError):
        PromptSchedule(iso, length_bin=lb, domain=dom, domain_weights={"science": 1.0}, domain_bin_weights={"science": {"b4096": 1.0}})


def test_loader_reads_the_domain_column_and_older_files_have_none(tmp_path):
    rows = [{"english": "a sentence", "iso": "yor", "src_afr_tokens": 10, "source_dataset": "x", "domain": "science"},
            {"english": "another one", "iso": "hau", "src_afr_tokens": 10, "source_dataset": "x", "domain": "general"}]
    pr = load_rl_prompts(write_jsonl(str(tmp_path / "p.jsonl"), rows))
    assert pr.domain == ["science", "general"] and pr.domain_of(0) == "science"
    old = load_rl_prompts(write_jsonl(str(tmp_path / "q.jsonl"), [{k: v for k, v in r.items() if k != "domain"} for r in rows]))
    assert old.domain == [] and old.domain_of(1) == ""


# ------------------------------------------------------------------------------------------------------------------ science anchor
SRC = "Serum zinc fell by 12.5% (95% CI 3.1-4.2) in 240 children given 5 mg of ZnSO4 daily, and the WHO guideline was met in 1998."
GOOD = "Zinki ya seramu ya ragu da 12.5% (95% CI 3.1-4.2) a cikin yara 240 da aka ba 5 mg na ZnSO4 kullum, an cika jagorar WHO a 1998."
BAD = "Zinki ya seramu ya ragu sosai a cikin yara da yawa da aka ba maganin zinki kullum, an cika jagora a shekarun baya."


def test_features_and_deficit():
    f_good, f_bad = sci_anchor.features(SRC, GOOD), sci_anchor.features(SRC, BAD)
    assert f_good["num_recall"] == 1.0 and f_good["pair_recall"] == 1.0 and f_good["protected_recall"] == 1.0
    assert f_bad["num_recall"] == 0.0 and f_bad["protected_recall"] == 0.0
    th = sci_anchor.DEFAULT_THRESHOLDS
    assert sci_anchor.deficit(f_good, th) == 0.0
    assert sci_anchor.deficit(f_bad, th) > 0.5
    assert sci_anchor.deficit({k: None for k in th}, th) == 0.0                        # nothing to judge -> no penalty
    invented = sci_anchor.features(SRC, GOOD + " Kuma 9999 da 8888 aka lura.")
    assert invented["num_precision"] < 1.0 and invented["num_recall"] == 1.0
    assert sci_anchor.features("The cell divides.", "Kwayar halitta tana rarrabuwa.")["num_recall"] is None


def test_protected_tokens_are_not_judged_for_geez_script_outputs():
    src = "WHO recommends zinc in Malawi and Ethiopia, 240 children."
    amh = "የዓለም ጤና ድርጅት በማላዊ እና በኢትዮጵያ ዚንክ ይመክራል፣ 240 ልጆች።"
    f = sci_anchor.features(src, amh)
    assert f["protected_recall"] is None and f["num_recall"] == 1.0
    assert sci_anchor.features(src, "WHO ta ba da shawarar zinc a Malawi, yara 240.")["protected_recall"] == 1.0


def test_calibration_roundtrip(tmp_path):
    rows = [("hau", SRC, GOOD)] * 80 + [("hau", SRC, BAD)] * 5 + [("yor", SRC, GOOD)] * 40
    blob = sci_anchor.build_calibration(rows, min_n=50, min_n_language=60)
    assert blob["default"]["num_recall"]["p10"] == 1.0 and blob["languages"]["hau"]["num_recall"]["n"] == 85 and blob["languages"]["yor"] == {}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(blob))
    sa = sci_anchor.ScienceAnchor(str(p))
    assert sa.thresholds("yor")["num_recall"] == 1.0 and sa.deficit("hau", SRC, GOOD) == 0.0 and sa.deficit("hau", SRC, BAD) > 0.5


def _cal(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"hau": {"anchors": [0.46, 0.55, 0.65], "usable": True, "reliability": 0.9}}))
    return str(p)


def test_reward_penalises_dropped_numbers_only_for_the_science_domain_and_only_when_on(tmp_path):
    cal = _cal(tmp_path)
    off = make(RewardConfig(calibration_path=cal, science_anchor_penalty=0.0), scorer=ConstScorer(0.9))
    on = make(RewardConfig(calibration_path=cal, science_anchor_penalty=1.0), scorer=ConstScorer(0.9))
    assert off.sci_anchor is None and on.sci_anchor is not None

    def r(rw, text, domain):
        return rw([Sample(ids("hau " + text), "hau", SRC, domain=domain)])

    base = r(off, BAD, "science").rewards[0]
    assert r(off, GOOD, "science").rewards[0] == pytest.approx(r(off, BAD, "science").rewards[0])      # QE (constant) cannot tell them apart
    g, b = r(on, GOOD, "science"), r(on, BAD, "science")
    assert g.rewards[0] == pytest.approx(base), "a faithful translation is untouched (penalty only, never a bonus)"
    assert b.rewards[0] < base - 0.5 and b.rewards[0] >= on.penalty - 1e-9
    assert b.science_deficit[0] > 0.5 and g.science_deficit[0] == 0.0
    assert r(on, BAD, "general").rewards[0] == pytest.approx(base) and r(on, BAD, "general").science_deficit[0] is None   # replay prompts untouched
    assert r(on, BAD, "").rewards[0] == pytest.approx(base)
    assert b.stats["sci_anchor/penalised_rate"] == 1.0 and b.stats["sci_anchor/deficit_mean"] > 0.5


def test_default_reward_config_is_off():
    c = RewardConfig()
    assert c.science_anchor_penalty == 0.0 and c.science_anchor_domains == ["science"] and c.science_anchor_path is None


# ------------------------------------------------------------------------------------------------------------------ end to end (tiny CPU model)
def test_cpu_smoke_science_mixture_aux_sets_and_selection_reset_on_resume(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.rl.train_rl import RLConfig, run_smoke, train_rl
    from gemma_mt.smoke import ENGLISH, fake_translation
    from gemma_mt.utils import dataclass_from_dict

    langs = ("yor", "hau", "ibo")
    rows = []
    for d in ("science", "general"):
        for b, tok in (("b64", 12), ("b256", 100)):
            for i in range(24):
                rows.append({"english": f"{ENGLISH[i % len(ENGLISH)]} {d} {b} {i} measured 25 mg", "iso": langs[i % 3], "language": "", "src_afr_tokens": tok,
                             "source_dataset": "smoke", "length_bin": b, "domain": d})
    p = write_jsonl(str(tmp_path / "p.jsonl"), rows)

    def dev(name, n):
        r = [{"english": ENGLISH[i % len(ENGLISH)], "reference": fake_translation(ENGLISH[i % len(ENGLISH)], langs[i % 3]), "iso": langs[i % 3], "benchmark": name, "id": str(i)} for i in range(n)]
        return write_jsonl(str(tmp_path / f"{name}.jsonl"), r)

    aux = dev("aux", 12)
    summ = run_smoke(workdir=str(tmp_path / "w"), sft_steps=20, benchmark=True, overrides=dict(
        prompts_path=p, max_src_tokens=2048, length_bins=True, curriculum_steps={"b64": 1}, steps=4, eval_prompts=6, eval_every=2, save_every=2,
        prompts_per_step_by_bin={"b64": 3, "mixed": 4}, max_new_tokens_by_bin={"b64": 24, "mixed": 32},
        domain_weights={"science": 0.7, "general": 0.3}, domain_bin_weights={"science": {"b64": 0.4, "b256": 0.6}},
        benchmark_aux=[{"name": "aux", "path": aux, "per_language": 3, "max_new_tokens": 16}],
        reward={"max_comet_tokens": 510, "science_anchor_penalty": 0.5}))
    assert summ["steps"] == 4
    recs = [json.loads(l) for l in open(tmp_path / "w" / "rl" / "metrics.jsonl") if l.strip()]
    ev = [r for r in recs if "eval/reward" in r]
    assert ev and all("bench_aux/aux/chrf_macro" in r and "bench/chrf_macro" in r for r in ev)
    assert any("eval/reward_domain/science" in r or "eval/reward_domain/general" in r for r in ev)
    assert any(r.get("baseline") == 1.0 and "bench_aux/aux/chrf_macro" in r for r in recs), "the aux sets are scored at the start"
    assert any("reward/sci_anchor/judged_rate" in r for r in recs)

    # resume the same state with ANOTHER selection benchmark: the resumed policy is scored first, best_select restarts from it, and a plain restart (same benchmark) does not do it again
    cfgd = json.load(open(tmp_path / "w" / "rl" / "config.json"))["config"]
    cfgd.update(steps=6, resume=True, benchmark_eval_path=dev("bench2", 12), resume_reset_selection=True, save_every=2, eval_every=2)
    train_rl(dataclass_from_dict(RLConfig, cfgd), tokenizer=FakeGemmaTokenizer(), scorer=FakeScorer())
    recs2 = [json.loads(l) for l in open(tmp_path / "w" / "rl" / "metrics.jsonl") if l.strip()]
    base = [r for r in recs2 if r.get("baseline") == 1.0 and r["step"] == 4]
    assert base, "the resumed policy must be scored as the new baseline at the resumed step"
    n_base = len([r for r in recs2 if r.get("baseline") == 1.0])
    cfgd.update(steps=8)
    train_rl(dataclass_from_dict(RLConfig, cfgd), tokenizer=FakeGemmaTokenizer(), scorer=FakeScorer())
    recs3 = [json.loads(l) for l in open(tmp_path / "w" / "rl" / "metrics.jsonl") if l.strip()]
    assert len([r for r in recs3 if r.get("baseline") == 1.0]) == n_base, "a restart on the same benchmark keeps best_select"


def test_validate_config_rejects_bad_science_options():
    pytest.importorskip("torch")
    from gemma_mt.rl.train_rl import RLConfig, validate_config

    with pytest.raises(ValueError):
        validate_config(RLConfig(domain_weights={"science": 1.0}, length_bins=False))
    with pytest.raises(ValueError):
        validate_config(RLConfig(length_bins=True, domain_bin_weights={"science": {"b9": 1.0}}))
    with pytest.raises(ValueError):
        validate_config(RLConfig(benchmark_aux=[{"name": "x"}]))
    validate_config(RLConfig(length_bins=True, domain_weights={"science": 0.7, "general": 0.3}, domain_bin_weights={"science": {"b2048": 1.0}},
                             benchmark_aux=[{"name": "x", "path": "p"}]))


def test_rl_science_yaml_loads_strictly_and_differs_from_the_live_config_only_where_intended():
    import os

    pytest.importorskip("yaml")
    pytest.importorskip("torch")
    from gemma_mt.rl.reward import RewardConfig
    from gemma_mt.rl.train_rl import RLConfig, validate_config
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sci, live = load_yaml(os.path.join(root, "configs", "rl_science.yaml")), load_yaml(os.path.join(root, "configs", "rl_rtxpro_phase2c_mixed.yaml"))
    cfg = dataclass_from_dict(RLConfig, sci)
    validate_config(cfg)
    rc = dataclass_from_dict(RewardConfig, cfg.reward)
    assert cfg.steps == 1433 + 500 and sum(cfg.curriculum_steps.values()) == 1284 < 1433, "the resumed steps are all in the mixed stage"
    assert cfg.domain_weights == {"science": 0.7, "general": 0.3} and cfg.length_bins and cfg.resume and cfg.resume_reset_selection
    assert cfg.hub_repo_id == (os.environ['HF_ORG_LEGACY'] + "/gemma-science-translate") and cfg.hub_private and cfg.hub_latest_branch == "latest" and cfg.hub_latest_every == 25
    assert cfg.out_dir.endswith("runs/sci") and cfg.out_dir != live["out_dir"] and cfg.archive_path.startswith(cfg.out_dir)
    assert cfg.model_name.endswith("final_1433") and cfg.select_by == "benchmark_chrf"
    assert rc.science_anchor_penalty == 0.5 and rc.science_anchor_domains == ["science"]
    same = {k for k in live if k not in ("reward",) and live[k] == sci.get(k)}
    for k in ("learning_rate", "kl_coef", "kl_estimator", "kl_guard_threshold", "group_size", "temperature", "max_new_tokens_by_bin", "prompts_per_step_by_bin", "curriculum_steps", "vllm_max_model_len"):
        assert k in same, f"{k} must stay as in the live 2c config"
    for k, v in live["reward"].items():
        assert sci["reward"][k] == v, f"reward.{k} changed against the live config"
    assert {k for k in sci["reward"] if k not in live["reward"] and not k.startswith("anchors_")} == {"science_anchor_penalty", "science_anchor_path", "science_anchor_domains"}
    assert rc.anchors_enabled and rc.anchors_modes == ["consensus", "blend"] and rc.anchors_off_languages == [] and rc.anchors_lexicon_weight == 0 and rc.anchors_lid_prob_weight == 0   # D-56 (constraint penalties)
    assert not RewardConfig().anchors_enabled                                                                                                                   # ... off in the dataclass: the general runs are untouched
    # the general run is untouched: neither the 2c yaml nor the dataclass default switches anything of this on
    assert "domain_weights" not in live and RLConfig().domain_weights == {} and RLConfig().benchmark_aux == [] and RLConfig().resume_reset_selection is False
