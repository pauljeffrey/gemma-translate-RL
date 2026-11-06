# SPDX-License-Identifier: AGPL-3.0-or-later
"""The 385M RL configs are exactly what the generator writes, keep the plan's numbers, point at the 385M assets, keep the new reward flags off; shaping_clean_v2 = v1 + 8 languages."""
import json
import os
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "configs", "clean_rerun_385m")
NEW8 = ["guw", "lue", "tll", "zne", "lub", "crs", "mfe", "arz"]
F2_FILE = "rl_F2_long.yaml"                                                                                                   # OPTIONAL stage (owner 2026-11-02), not part of the default plan
FILES = ["rl_00_preflight.yaml", "rl_A_b64.yaml", "rl_B1_b256.yaml", "rl_C_mixed.yaml", "rl_F_final.yaml"]          # b256 CAP (owner 2026-11-02): no B2 (b512) / B3 (b1024)


def test_committed_yamls_equal_the_generator_output():
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "clean_rerun_385m_make_configs.py"), "--out", td], check=True, capture_output=True)
        for f in FILES + [F2_FILE]:
            generated = open(os.path.join(td, f)).read()
            for var in ("HF_ORG_LEGACY", "HF_ORG"):      # the committed YAML keeps the ${VAR} placeholders, the generator writes the environment's values
                generated = generated.replace(os.environ[var], "${%s}" % var)
            assert generated == open(os.path.join(CFG, f)).read(), f


def test_overrides_and_plan_numbers():
    yaml = pytest.importorskip("yaml")
    for f in FILES:
        d = yaml.safe_load(open(os.path.join(CFG, f)))
        old = yaml.safe_load(open(os.path.join(ROOT, "configs", "clean_rerun", f)))
        assert d["tokenizer_name"].endswith("assets/tokenizer_gemma3_tags") and "385m" in d["out_dir"]
        assert d["reward"]["calibration_path"].endswith("shaping_clean_v2.json")
        assert d["save_every"] == 20 and (d["hub_repo_id"] is None or d["hub_repo_id"].startswith((os.environ['HF_ORG'] + "/gemma-0.4B-")))
        for k in ("steps", "group_size", "kl_coef", "learning_rate", "kl_guard_threshold", "language_weights"):
            assert d[k] == old[k], (f, k)
        # b256 CAP: only b64 / b256 / mixed rows, the numbers of the kept bins are the plan's, source cap 256, engine length 4096
        assert d["prompts_per_step_by_bin"] == {b: old["prompts_per_step_by_bin"][b] for b in ("b64", "b256", "mixed")}, f
        assert d["max_new_tokens_by_bin"] == {"b64": 384, "b256": 1280, "mixed": 1280} and d["max_src_tokens"] == 256 and d["vllm_max_model_len"] == 4096, f
        assert not ({"b512", "b1024", "b2048"} & set(d["curriculum_steps"])), f
        for k in ("consensus", "qe_ensemble", "anchors_enabled", "lid_enabled", "nllb_signal"):
            assert d["reward"][k] == old["reward"][k], (f, k)
        assert d["reward"]["longdoc_v2"] is True and d["reward"]["longdoc_min_src_words"] == 1 and d["reward"]["longdoc_compare_v1"] is True, f    # longdoc_v2 on EVERY source (owner 2026-11-02)
        assert not any(k.startswith(("confidence", "fluency")) for k in d["reward"]), "new flags must stay off (absent) in the yamls"
        assert (d["prompts_path"].endswith("clean_v3_385m")) == (f != "rl_F_final.yaml")


def test_the_removed_stages_are_gone_and_the_chain_skips_them():
    assert not os.path.exists(os.path.join(CFG, "rl_B2_b512.yaml")) and not os.path.exists(os.path.join(CFG, "rl_B3_b1024.yaml"))
    mixed = open(os.path.join(CFG, "rl_C_mixed.yaml")).read()
    assert "runs_385m/B1/final" in mixed.split("model_name:")[1].split("\n")[0], "mixed must start from B1's final/"
    fin = open(os.path.join(CFG, "rl_F_final.yaml")).read()
    assert "science: {b64: 0.3333333333, b256: 0.6666666667}" in fin and "general: {b64: 0.5, b256: 0.5}" in fin and "afrisci_dev_doc" not in fin
    launcher = open(os.path.join(ROOT, "scripts", "launch_clean_rerun.sh")).read()
    assert 'STAGES=(A B1 mixed)' in launcher and 'mixed) echo B1' in launcher
    assert 'STAGE_PLAN="${STAGE_PLAN:-b256}"' in open(os.path.join(ROOT, "scripts", "launch_clean_rerun_385m.sh")).read()


def test_calibration_v2_is_v1_plus_the_eight_languages():
    d = os.path.join(ROOT, "calibration", "clean_2026-10-28")
    v1 = json.load(open(os.path.join(d, "shaping_clean_v1.json"))); v2 = json.load(open(os.path.join(d, "shaping_clean_v2.json")))
    for k in v1:
        if k != "_meta":
            assert json.dumps(v1[k], sort_keys=True) == json.dumps(v2[k], sort_keys=True), k
    assert [k for k in v2 if k in NEW8] == NEW8 and not [k for k in NEW8 if k in v1]
    for k in NEW8:
        assert v2[k]["content"]["mode"] == "consensus" and v2[k]["reliability"] == 0.5 and v2[k]["lid"]["labels"]
    assert v2["_meta"]["v2_additions"]["languages"] == NEW8


# ------------------------------------------------------------------------------------------------ optional stage F2 (final_long)
def _launcher(env, *args):
    e = dict(os.environ, **env)
    return subprocess.run(["bash", os.path.join(ROOT, "scripts", "launch_clean_rerun.sh"), *args], capture_output=True, text=True, env=e, timeout=120)


def test_f2_is_not_a_default_stage_and_its_settings_are_the_owners():
    yaml = pytest.importorskip("yaml")
    assert len(FILES) == 5 and F2_FILE not in FILES
    launcher = open(os.path.join(ROOT, "scripts", "launch_clean_rerun.sh")).read()
    assert "STAGES=(A B1 mixed)" in launcher and "STAGES=(A B1 mixed F2)" not in launcher and "STAGES=(A B1 B2 B3 mixed F2)" not in launcher      # F2 is never in the default chain
    d = yaml.safe_load(open(os.path.join(CFG, F2_FILE)))
    fin = yaml.safe_load(open(os.path.join(CFG, "rl_F_final.yaml")))
    assert d["steps"] == 70 and d["model_name"].endswith("runs_385m/final/final") and d["out_dir"].endswith("runs_385m/F2") and d["hub_repo_id"].endswith("gemma-0.4B-science-rl") and d["hub_branch"] == "stage-f2-long"
    assert d["domain_weights"] == {"science": 1.0} and d["domain_bin_weights"] == {"science": {"b1024": 0.6, "b2048": 0.4}}
    assert d["max_src_tokens"] == 2048 and d["vllm_max_model_len"] == 10240 and d["max_new_tokens_by_bin"]["b1024"] == 3584 and d["max_new_tokens_by_bin"]["b2048"] == 7168 == d["max_new_tokens_by_bin"]["mixed"]
    assert d["prompts_per_step_by_bin"] == {"b1024": 42, "b2048": 21, "mixed": 28} and d["prompts_path"].endswith("prompts/clean_sci_v2_long")
    assert d["gradient_checkpointing"] is True and d["vllm_gpu_memory_utilization"] == 0.3 and d["archive"] is False
    for k in ("learning_rate", "kl_coef", "kl_guard_threshold", "group_size", "language_weights", "language_floor_steps", "language_floor_prompts", "greedy_in_group"):
        assert d[k] == fin[k], k
    diff = {k for k in set(d["reward"]) | set(fin["reward"]) if d["reward"].get(k) != fin["reward"].get(k)}
    assert diff == {"nllb_signal", "nllb_languages"} and d["reward"]["nllb_signal"] is False        # every other reward rule is the final run's (longdoc_v2 on, same calibration)
    assert d["reward"]["longdoc_v2"] is True and d["reward"]["longdoc_calibration_path"] == fin["reward"]["longdoc_calibration_path"]
    assert [x["name"] for x in d["benchmark_aux"]][0] == "afrisci_dev_doc" and d["benchmark_aux"][0]["max_new_tokens"] == 7168


def test_f2_validates_with_the_real_config_classes():
    pytest.importorskip("yaml")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "clean_rerun_tools.py"), "validate", "--plan", "b256_385m", "--configs", CFG, "--stub-heavy", "--no-generated-check", "--extra-stages", "F2"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert "F2 " in r.stdout and "665,920" in r.stdout          # the five default stages still total 665,920 completions; F2 is listed but not counted


def test_launcher_runs_f2_only_when_asked_for(tmp_path):
    base = {"ROOT": str(tmp_path), "TREE": ROOT, "CFGDIR": CFG, "RUNS": str(tmp_path / "runs"), "HUB_PREFIX": ""}
    for env in ({}, {"STAGE_PLAN": "b256"}, {"STAGE_PLAN": "full"}):
        r = _launcher({**base, **env}, "run", "F2")
        assert r.returncode == 2 and "OPTIONAL and OFF" in r.stdout, (env, r.stdout, r.stderr)
    r = _launcher({**base, "STAGE_PLAN": "full", "EXTRA_STAGES": "F2"}, "run", "F2")
    assert r.returncode == 2 and "needs STAGE_PLAN=b256" in r.stdout
    r = _launcher({**base, "STAGE_PLAN": "b256+long"}, "run", "F2")                      # asked for, but the b2048 science pool was not prepared
    assert r.returncode == 3 and "zz_b2048" in r.stdout, r.stdout
    pool = tmp_path / "prompts" / "clean_sci_v2_long" / "zz_b2048"; pool.mkdir(parents=True); (pool / "yor.parquet").write_bytes(b"x")
    r = _launcher({**base, "EXTRA_STAGES": "F2", "STAGE_PLAN": "b256"}, "run", "F2")      # pool present: it gets as far as the manifest pin (none here), i.e. it is a real stage
    assert r.returncode == 6 and "no manifest" in (r.stdout + r.stderr), r.stdout
    r = _launcher({**base, "STAGE_PLAN": "b256", "EXTRA_STAGES": "F3"}, "status")
    assert r.returncode == 2


def _preds(path, spec):
    """spec: {iso: [(ref, hyp, finished), ...]} -> predictions_dev.json in the format of scripts/eval_afriscience.py"""
    json.dump({"document": {iso: [{"paper": f"p{i}", "hyp": h, "finished": f, "ref": r} for i, (r, h, f) in enumerate(rows)] for iso, rows in spec.items()}}, open(path, "w"))


def _paper(i, iso):
    w = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "mu"]
    return " ".join(w[(i * 5 + j * 3 + len(iso)) % len(w)] + str((i * 7 + j) % 11) for j in range(60))


def _spec(degrade=lambda i, iso: 0.0, unfinished=lambda i, iso: False, loops=lambda i, iso: False):
    out = {}
    for iso in ("amh", "hau", "lug", "nso", "yor", "zul"):
        rows = []
        for i in range(25):
            ref = _paper(i, iso); words = ref.split(); keep = max(3, int(len(words) * (1 - degrade(i, iso))))
            hyp = " ".join(words[:keep])
            if loops(i, iso):
                hyp += " " + " ".join(["stuck"] * 40)
            rows.append((ref, hyp, not unfinished(i, iso)))
        out[iso] = rows
    return out


def _tool(*args):
    pytest.importorskip("sacrebleu"); pytest.importorskip("numpy")
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "clean_rerun_tools.py"), *args], capture_output=True, text=True)


def test_f2_precheck_skips_only_when_whole_papers_already_finish(tmp_path):
    good, bad, looped = tmp_path / "good.json", tmp_path / "bad.json", tmp_path / "loop.json"
    _preds(good, _spec())
    _preds(bad, _spec(unfinished=lambda i, iso: i < 2))                       # 12 of 150 = 8 % unfinished
    _preds(looped, _spec(loops=lambda i, iso: i == 0 and iso in ("amh", "hau", "lug", "nso")))     # 4 of 150 = 2.7 % loops
    r = _tool("f2-precheck", "--preds", str(good)); assert r.returncode == 10 and '"verdict": "SKIP"' in r.stdout, r.stdout + r.stderr
    r = _tool("f2-precheck", "--preds", str(bad)); assert r.returncode == 0 and '"verdict": "RUN"' in r.stdout
    r = _tool("f2-precheck", "--preds", str(looped)); assert r.returncode == 0 and '"verdict": "RUN"' in r.stdout
    short = tmp_path / "short.json"; _preds(short, {k: v[:5] for k, v in _spec().items()})
    assert _tool("f2-precheck", "--preds", str(short)).returncode == 1       # fewer than 150 papers: INVALID


def test_f2_decide_keeps_f2_only_on_the_owners_two_criteria(tmp_path):
    f, same, better, fin3, fin1 = (tmp_path / n for n in ("f.json", "same.json", "better.json", "fin3.json", "fin1.json"))
    deg = lambda i, iso: 0.3 if i % 2 == 0 else 0.0                              # F loses 30 % of the words of every second paper
    unf = lambda i, iso: i == 0 or (iso in ("amh", "hau", "lug", "nso", "yor") and i == 1)      # 11 unfinished papers in F (the same papers keep their text in the variants)
    _preds(f, _spec(degrade=deg, unfinished=unf))
    _preds(same, _spec(degrade=deg, unfinished=unf))                              # identical -> no improvement
    _preds(better, _spec(degrade=lambda i, iso: 0.0, unfinished=unf))             # chrF up on 75 papers, finished unchanged -> criterion A
    _preds(fin3, _spec(degrade=deg, unfinished=lambda i, iso: i == 0 and iso in ("amh", "hau", "lug", "nso", "yor")))   # F2: 6 more papers finished (+4 points), same text -> criterion B
    _preds(fin1, _spec(degrade=deg, unfinished=lambda i, iso: i == 0 or (iso in ("amh", "hau", "yor") and i == 1)))      # F2: 2 more papers finished (+1.3 points) -> neither
    out = {}
    for name, p in (("same", same), ("better", better), ("fin3", fin3), ("fin1", fin1)):
        r = _tool("f2-decide", "--base", str(f), "--new", str(p), "--bootstrap", "2000", "--out", str(tmp_path / f"{name}_out.json"))
        out[name] = (r.returncode, json.load(open(tmp_path / f"{name}_out.json")) if os.path.exists(tmp_path / f"{name}_out.json") else r.stdout + r.stderr)
    assert out["same"][0] == 10 and out["same"][1]["verdict"] == "DISCARD_F2"
    assert out["better"][0] == 0 and out["better"][1]["criterion_A_chrf_ci_above_0"] and out["better"][1]["chrf_diff_ci95"][0] > 0
    assert out["fin3"][0] == 0 and out["fin3"][1]["criterion_B_finished_plus_2_points_chrf_not_lower"] and not out["fin3"][1]["criterion_A_chrf_ci_above_0"]
    assert out["fin1"][0] == 10 and out["fin1"][1]["verdict"] == "DISCARD_F2"
    # a finished-rate gain with a LOWER mean chrF is not enough
    worse = tmp_path / "worse.json"; _preds(worse, _spec(degrade=lambda i, iso: 0.6 if i % 2 == 0 else 0.0, unfinished=lambda i, iso: False))
    r = _tool("f2-decide", "--base", str(f), "--new", str(worse), "--bootstrap", "2000")
    assert r.returncode == 10 and '"criterion_B_finished_plus_2_points_chrf_not_lower": false' in r.stdout


def test_dynamic_sampling_and_reference_term_settings():
    """Owner 2026-11-03: dynamic sampling ON in A / B1 / mixed / final (OFF in the preflight and in the long-document F2), the reference term (reward.reference_weight 0.35) in every stage but the preflight,
    the reference domain (15 % of the draws) in the translation stages; the preflight keeps its 70 / 30 general / news mixture."""
    yaml = pytest.importorskip("yaml")
    got = {}
    for f in FILES + [F2_FILE]:
        got[f] = yaml.safe_load(open(os.path.join(CFG, f)))
    assert got["rl_00_preflight.yaml"].get("dynamic_sampling", False) is False and "reference_weight" not in got["rl_00_preflight.yaml"]["reward"]
    assert not any(k in got["rl_00_preflight.yaml"] for k in ("adv_std_floor", "guide_fraction", "log_group_stats"))
    assert got["rl_00_preflight.yaml"]["domain_weights"] == {"general": 0.7, "news": 0.3}
    for f in ("rl_A_b64.yaml", "rl_B1_b256.yaml", "rl_C_mixed.yaml", "rl_F_final.yaml"):
        d = got[f]
        assert d["dynamic_sampling"] is True and d["dynamic_sampling_oversample"] == 1.1 and d["dynamic_sampling_max_refills"] == 2 and d["dynamic_sampling_min_std"] == 0.01, f
        assert d["reward"]["reference_weight"] == 0.35, f
        assert d["adv_std_floor"] == 0.02 and d["log_group_stats"] is True, f                                                                              # advantage std floor + per-prompt group statistics (owner 2026-11-03)
        assert d["guide_fraction"] == 0.25 and d["guide_beams"] == 4 and d["guide_adv_share_cap"] == 0.25 and d.get("hard_fail_mask", "off") == "off", f      # beam-guided member ON, capped
    for f in ("rl_A_b64.yaml", "rl_B1_b256.yaml", "rl_C_mixed.yaml"):
        w = got[f]["domain_weights"]
        assert abs(sum(w.values()) - 1.0) < 1e-9 and w["ref"] == 0.15 and abs(w["news"] / (w["general"] + w["news"]) - 0.3) < 1e-3, (f, w)
    assert got["rl_F_final.yaml"]["domain_weights"] == {"science": 0.7, "general": 0.3}
    assert got[F2_FILE]["dynamic_sampling"] is False and got[F2_FILE]["guide_fraction"] == 0.0 and got[F2_FILE]["early_stop_dev"] is False
    # MBR guide for the weak list (+ orm lug) and the dev early stop: B1 / mixed / final only (stage A had run), benchmark_eval_every 25
    weak = set(got["rl_B1_b256.yaml"]["language_weights"]) | {"orm", "lug"}
    assert "orm" not in got["rl_B1_b256.yaml"]["language_weights"] and {"ewe", "tir"} <= set(got["rl_B1_b256.yaml"]["language_weights"]) | {"ewe", "tir"}
    for f in ("rl_B1_b256.yaml", "rl_C_mixed.yaml", "rl_F_final.yaml"):
        d = got[f]
        assert set(d["guide_mode_by_language"]) == weak and set(d["guide_mode_by_language"].values()) == {"mbr"} and len(weak) == 31, f
        assert d["guide_mbr_extra_samples"] == 16 and d["guide_mbr_temperature"] == 0.7 and d["guide_mbr_top_p"] == 0.95 and d["guide_mbr_max_overhead_frac"] == 0.20, f
        assert d["early_stop_dev"] is True and d["early_stop_dev_k"] == 3 and d["early_stop_dev_eps"] == 0.15 and d["early_stop_dev_min_steps"] == 75 and d["early_stop_dev_min_finished"] == 0.98, f
        assert d["early_stop_dev_winddown"] == 10 and d["benchmark_eval_every"] == 25 and d["early_stop_dev_aux"] == "weak_dev" and d["benchmark_eval_path"], f
    for f in ("rl_B1_b256.yaml", "rl_C_mixed.yaml", "rl_F_final.yaml"):
        assert got[f]["save_dev_predictions"] is True, f
    for f in ("rl_C_mixed.yaml", "rl_F_final.yaml"):                                        # patch 3: the held-out AfriCOMET monitor and what hangs on it (mixed / final only)
        d = got[f]
        assert d["heldout_comet"] is True and d["heldout_comet_model"] == "masakhane/africomet-stl-1.1" and d["heldout_comet_keep_loaded"] is False and d["early_stop_dev_require_comet"] is True, f
        assert d["hacking_alarm"] is True and d["hacking_alarm_k"] == 3 and d["hacking_alarm_eps_chrf"] == 0.3 and d["hacking_alarm_eps_comet"] == 0.005 and d["early_stop_dev_eps_comet"] == 0.004, f
        assert d["lang_stall_log"] is True and d["scorer"]["python"] and d["scorer"]["model_id"] == "masakhane/africomet-qe-stl-1.1" and d["scorer"]["ensemble_model_id"] == "McGill-NLP/ssa-comet-mtl", f
    assert not any(k in got["rl_B1_b256.yaml"] for k in ("heldout_comet", "hacking_alarm", "lang_stall_log", "early_stop_dev_require_comet")) and "save_dev_predictions" not in got["rl_A_b64.yaml"]
    assert not any(got[F2_FILE][k] for k in ("heldout_comet", "hacking_alarm", "lang_stall_log", "early_stop_dev_require_comet", "save_dev_predictions"))
    assert "guide_mode_by_language" not in got["rl_A_b64.yaml"] and "early_stop_dev" not in got["rl_A_b64.yaml"] and got["rl_A_b64.yaml"]["benchmark_eval_every"] == 50      # stage A had run: unchanged


def test_features_option_of_the_generator_switches_the_additions_on_one_by_one():
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "clean_rerun_385m_make_configs.py"), "--out", td, "--features", "ds,adv,log"], check=True, capture_output=True)
        yaml = pytest.importorskip("yaml")
        d = yaml.safe_load(open(os.path.join(td, "rl_A_b64.yaml")))
        assert d["dynamic_sampling"] is True and d["adv_std_floor"] == 0.02 and d["log_group_stats"] is True
        assert "guide_fraction" not in d and "reference_weight" not in d["reward"] and d["domain_weights"] == {"general": 0.7, "news": 0.3}
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "clean_rerun_385m_make_configs.py"), "--out", td, "--features", "ds,adv,log,guide"], check=True, capture_output=True)
        d = yaml.safe_load(open(os.path.join(td, "rl_A_b64.yaml")))
        assert d["guide_fraction"] == 0.25 and "ref" not in d["domain_weights"]
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "clean_rerun_385m_make_configs.py"), "--out", td, "--features", "bogus"], capture_output=True)
        assert r.returncode != 0
