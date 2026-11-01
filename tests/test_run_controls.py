# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-56: run controls (WINDDOWN / LR_SCALE files), monitor sets with `every`, and the CPU smoke with every new feature on."""
import json
import os

import pytest

from gemma_mt.rl.run_controls import DEFAULT_WINDDOWN, RunControls, parse_count


def test_parse_count():
    assert parse_count("50") == 50 and parse_count("  30\n") == 30 and parse_count("") == DEFAULT_WINDDOWN
    assert parse_count("soon") == DEFAULT_WINDDOWN and parse_count("0") == DEFAULT_WINDDOWN and parse_count("-5") == DEFAULT_WINDDOWN
    assert parse_count("20 steps") == 20 and parse_count("12.0") == 12


def test_no_files_means_no_change(tmp_path):
    c = RunControls(str(tmp_path), 100, log=lambda m: None)
    s = c.update(0)
    assert s.lr_mult == 1.0 and s.end_step == 100 and not s.winddown
    assert c.finish(100) is None


def test_lr_scale_file_is_read_every_step_and_clamped(tmp_path):
    msgs = []
    c = RunControls(str(tmp_path), 100, log=msgs.append)
    assert c.update(0).lr_mult == 1.0
    (tmp_path / "LR_SCALE").write_text("0.5\n")
    assert c.update(1).lr_mult == 0.5 and any("0.5" in m for m in msgs)
    (tmp_path / "LR_SCALE").write_text("garbage")
    assert c.update(2).lr_mult == 1.0 and any("unreadable" in m for m in msgs)
    (tmp_path / "LR_SCALE").write_text("100")
    assert c.update(3).lr_mult == 4.0
    (tmp_path / "LR_SCALE").write_text("-1")
    assert c.update(4).lr_mult == 0.0
    os.remove(tmp_path / "LR_SCALE")
    assert c.update(5).lr_mult == 1.0


def test_winddown_linear_to_zero_then_ends_and_is_archived(tmp_path):
    c = RunControls(str(tmp_path), 1000, log=lambda m: None)
    for s in range(7):
        assert c.update(s).end_step == 1000
    (tmp_path / "WINDDOWN").write_text("10")
    got = [c.update(7 + k) for k in range(10)]
    assert got[0].winddown and got[0].end_step == 17 and got[0].lr_mult == pytest.approx(1.0)
    assert [round(g.lr_mult, 3) for g in got] == [round(1 - k / 10, 3) for k in range(10)]
    assert got[-1].lr_mult == pytest.approx(0.1) and all(a.lr_mult > b.lr_mult for a, b in zip(got, got[1:]))
    (tmp_path / "LR_SCALE").write_text("0.5")                         # multiplies with the wind-down
    assert c.update(10).lr_mult == pytest.approx(0.5 * (1 - 3 / 10))
    rec = c.finish(17)
    assert rec == {"started_at_step": 7, "steps": 10, "finished_at_step": 17, "completed": True}
    assert not (tmp_path / "WINDDOWN").exists() and (tmp_path / "WINDDOWN.done").exists() and not (tmp_path / "WINDDOWN.state").exists()
    assert RunControls(str(tmp_path), 1000, log=lambda m: None).update(17).end_step == 1000      # restarting the run does not wind it down again


def test_winddown_is_capped_by_the_configured_steps_and_empty_file_means_default(tmp_path):
    c = RunControls(str(tmp_path), 60, log=lambda m: None)
    (tmp_path / "WINDDOWN").write_text("")
    s = c.update(30)
    assert s.end_step == 60 and s.winddown                                        # 30 + 50 > 60: the run's own end comes first
    assert s.lr_mult == pytest.approx(1.0) and c.update(45).lr_mult == pytest.approx(1 - 15 / 50)


def test_restart_in_the_middle_continues_and_removal_cancels(tmp_path):
    (tmp_path / "WINDDOWN").write_text("20")
    c1 = RunControls(str(tmp_path), 1000, log=lambda m: None)
    assert c1.update(100).end_step == 120
    c2 = RunControls(str(tmp_path), 1000, log=lambda m: None)                     # a new process (resume) at step 110
    s = c2.update(110)
    assert s.end_step == 120 and s.lr_mult == pytest.approx(0.5)
    os.remove(tmp_path / "WINDDOWN")
    s = c2.update(111)
    assert not s.winddown and s.end_step == 1000 and s.lr_mult == 1.0 and not (tmp_path / "WINDDOWN.state").exists()
    assert c2.finish(111) is None
    (tmp_path / "WINDDOWN").write_text("5")                                       # a later wind-down starts afresh
    assert c2.update(200).end_step == 205


def test_winddown_unfinished_is_not_archived(tmp_path):
    (tmp_path / "WINDDOWN").write_text("40")
    c = RunControls(str(tmp_path), 1000, log=lambda m: None)
    c.update(10)
    rec = c.finish(25)                                                              # the run died / was stopped before the end of the wind-down
    assert rec["completed"] is False and (tmp_path / "WINDDOWN").exists()


# ------------------------------------------------------------------------------------------------ the CPU smoke with everything on
def _cal():
    thr = {"3": [0.0, 0.05, 0.05, 0.10, 0.10, 0.10], "4": [0.0, 0.02, 0.03, 0.05, 0.05, 0.05], "5": [0.0, 0.0, 0.01, 0.03, 0.03, 0.03]}
    cap = {"3": 0.3, "4": 0.3, "5": 0.3}
    return {"_meta": {"weak_languages": ["yor", "hau"]}, "languages": {},
            "pooled": {"entity": {"num": {"t": 1.0, "min_n": 1, "enabled": True}}, "repeat": {"thr": thr, "cap": cap, "enabled": True}, "boiler": {"thr": 0.03, "cap": 0.4, "enabled": True}}}


def test_cpu_smoke_with_winddown_lr_scale_anchors_and_monitor_sets(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from gemma_mt.rl.train_rl import run_smoke
    from gemma_mt.smoke import ENGLISH, fake_translation, write_jsonl

    work = tmp_path / "w"
    (work / "rl").mkdir(parents=True)
    (work / "rl" / "WINDDOWN").write_text("5")
    (work / "rl" / "LR_SCALE").write_text("0.5")
    cal = tmp_path / "anchors.json"
    cal.write_text(json.dumps(_cal()))
    langs = ["yor", "hau", "ibo"]
    dev = [{"english": ENGLISH[i % len(ENGLISH)], "reference": fake_translation(ENGLISH[i % len(ENGLISH)], langs[i % 3]), "iso": langs[i % 3], "benchmark": "aux", "id": str(i)} for i in range(18)]
    aux = write_jsonl(str(tmp_path / "aux.jsonl"), dev)
    summ = run_smoke(workdir=str(work), sft_steps=20, benchmark=True, overrides=dict(
        steps=12, eval_every=2, save_every=0, warmup_steps=1, learning_rate=1e-4, benchmark_eval_every=2,
        benchmark_aux=[{"name": "small", "path": aux, "per_language": 3, "max_new_tokens": 16},
                       {"name": "big", "path": aux, "per_language": 2, "max_new_tokens": 16, "every": 4}],
        reward={"max_comet_tokens": 510, "consensus": True, "chunking": "sentences", "content_modes": {"yor": "blend", "hau": "consensus"},
                "anchors_enabled": True, "anchors_calibration_path": str(cal), "anchors_buffer_min_outputs": 8, "anchors_buffer_min_prompts": 3, "anchors_lexicon_weight": 0.0}))
    assert summ["steps"] == 5, summ                                               # 12 configured, wind-down of 5 steps from step 0
    assert summ["winddown"] == {"started_at_step": 0, "steps": 5, "finished_at_step": 5, "completed": True}
    rows = [json.loads(l) for l in open(work / "rl" / "metrics.jsonl") if l.strip()]
    steps = [r for r in rows if "lr" in r]
    lrs = [r["lr"] for r in steps]
    assert lrs == pytest.approx([0.5e-4 * (1 - k / 5) for k in range(5)], rel=1e-6)
    assert all("reward/anchors/applied_rate" in r and "reward/anchors/entity/judged_rate" in r and "reward/anchors/boiler/ready_rate" in r for r in steps)
    assert all(r["winddown_factor"] == pytest.approx(1 - k / 5) and r["lr_file_scale"] == 0.5 for k, r in enumerate(steps))
    evs = [r for r in rows if "bench_aux/small/chrf_macro" in r and "eval/reward" in r]
    assert len(evs) == 3                                                           # steps 2, 4 and the last one (5)
    assert sum("bench_aux/big/chrf_macro" in r for r in evs) == 2                  # `every: 4`: step 4 and the last step only
    assert all("bench_aux/small/finished/yor" in r and "bench_aux/small/seconds" in r and "bench_aux/seconds_total" in r for r in evs)
    assert os.path.exists(work / "rl" / "WINDDOWN.done") and not os.path.exists(work / "rl" / "WINDDOWN")
    assert os.path.exists(work / "rl" / "summary.json") and os.path.exists(work / "rl" / "last" / "trainer_state.pt")
