# SPDX-License-Identifier: AGPL-3.0-or-later
"""scripts/dev_comet_retro.py: scores saved dev predictions offline and replays the monitors."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "dev_comet_retro.py")


def make_run(tmp_path, steps, qe):
    run = tmp_path / "run"
    (run / "dev_predictions").mkdir(parents=True)
    for k, step in enumerate(steps):
        rows = []
        for iso in ("yor", "hau", "xxx"):
            for i in range(5):
                ref = f"awon agbe si mu ewebe tutu wa {i}"
                hyp = ref if k == 0 else ref[: max(8, len(ref) - 6 * k)]          # the translations get worse with the evaluation index
                rows.append({"set": "bench", "iso": iso, "id": str(i), "src": f"source sentence {i} " + "w" * (3 * k), "hyp": hyp, "ref": ref, "finished": True, "step": step})
        for i in range(4):
            rows.append({"set": "weak_dev", "iso": "yor", "id": str(i), "src": f"s {i}", "hyp": "y" * (6 - k), "ref": "y" * 6, "finished": True, "step": step})
        with open(run / "dev_predictions" / f"step_{step}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    with open(run / "metrics.jsonl", "w") as f:
        for step, q in zip(steps, qe):
            f.write(json.dumps({"step": step, "eval/qe_mean_scored": q, "eval/greedy_finished_rate": 0.99, "bench/finished_rate": 0.99}) + "\n")
    return run


def test_retro_scores_replays_and_reports(tmp_path):
    run = make_run(tmp_path, [25, 50, 75, 100, 125], [-0.6, -0.5, -0.4, -0.3, -0.2])
    r = subprocess.run([sys.executable, SCRIPT, str(run), "--fake", "--k", "3", "--alarm-eps-chrf", "0.1", "--alarm-eps-comet", "0.0001"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    out = json.load(open(run / "dev_comet_retro.json"))
    assert [e["step"] for e in out["table"]] == [25, 50, 75, 100, 125]
    b0, b4 = out["table"][0]["bench"], out["table"][-1]["bench"]
    assert set(b0["chrf"]) == {"yor", "hau", "xxx"} and set(b0["comet"]) == {"yor", "hau"}          # "xxx" is not an AfriCOMET language: chrF++ only
    assert b4["chrf_macro"] < b0["chrf_macro"] and out["table"][0]["weak_dev"]["comet_macro"] >= 0
    assert set(out["replay"]) == {"dev_patience (patch 2, chrF++ only)", "combined stop (patch 3, + AfriCOMET flat)", "reward-hacking alarm"}
    assert out["replay"]["reward-hacking alarm"] is not None          # QE rises while chrF++ and the (fake) AfriCOMET fall: the alarm replays
    assert os.path.exists(run / "dev_comet_retro_scores.json") and "reward-hacking alarm: would have stopped at step" in r.stdout


def test_retro_says_so_when_predictions_were_not_saved(tmp_path):
    (tmp_path / "A").mkdir()
    r = subprocess.run([sys.executable, SCRIPT, str(tmp_path / "A"), "--fake"], capture_output=True, text=True)
    assert r.returncode == 1 and "did not save its dev predictions" in r.stdout
