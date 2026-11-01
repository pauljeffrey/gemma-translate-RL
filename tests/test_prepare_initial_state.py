# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-59: scripts/prepare_initial_state.py builds the starting trainer state of the final run (start-model weights + the 2c Adam moments, step 0) and train_rl resumes from it."""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_initial_state", os.path.join(HERE, "..", "scripts", "prepare_initial_state.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_prepared_state_carries_adam_moments_resets_step_and_the_trainer_resumes_from_it(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.rl.train_rl import RLConfig, run_smoke, train_rl
    from gemma_mt.utils import dataclass_from_dict

    work = tmp_path / "w"
    run_smoke(workdir=str(work), sft_steps=20, benchmark=True)                      # a "2c" run: trainer_state.pt with Adam moments after 4 steps
    old = work / "rl" / "last" / "trainer_state.pt"
    assert old.exists()
    ck_old = torch.load(old, map_location="cpu", weights_only=False)
    steps_old = {int(s["step"]) for s in ck_old["optimizer"]["state"].values()}
    assert steps_old == {4} and ck_old["state"]["step"] == 4
    from gemma_mt.modeling import load_model, save_model

    base = load_model(str(work / "sft" / "final"), dtype=torch.float32, device="cpu")
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p_ in base.parameters():
            p_.add_(1e-3 * torch.randn(p_.shape, generator=g))                       # an "SFT-stage" model: close to, but not equal to, the 2c policy
    save_model(base, None, str(work / "start_model"))
    start = str(work / "start_model")                                                # stands in for final_1433 / the SFT-stage model

    P = _load_script()
    state, info = P.build_state(start, str(old))
    assert state["state"]["step"] == 0 and state["state"]["best_select"] == -float("inf") and state["state"]["bench_path"] is None
    assert info["adam_state_tensors"] == len(ck_old["optimizer"]["state"]) and info["rel_l2_start_vs_2c_policy"] > 0          # start model != the 2c policy
    for k in ("model", "optimizer", "rng"):
        assert k in state
    # a different optimizer layout / parameter set is refused
    bad = tmp_path / "bad.pt"
    ck_bad = dict(ck_old)
    ck_bad["model"] = {k: v for k, v in list(ck_old["model"].items())[:-1]}
    torch.save(ck_bad, bad)
    with pytest.raises(SystemExit):
        P.build_state(start, str(bad))

    # CLI: writes last/trainer_state.pt + PREPARED.json, verifies it, refuses to overwrite
    out_dir = tmp_path / "final"
    assert P.main.__code__ is not None
    import sys

    argv = ["prepare_initial_state.py", "--start-model", start, "--optimizer-from", str(old), "--out-dir", str(out_dir), "--verify"]
    old_argv, sys.argv = sys.argv, argv
    try:
        assert P.main() == 0
        assert P.main() == 2                                                         # exists, no --force
    finally:
        sys.argv = old_argv
    prep = json.load(open(out_dir / "last" / "PREPARED.json"))
    assert prep["verify"]["loads_into_model_and_adamw"] and prep["verify"]["weights_equal_start_model"] and prep["verify"]["state"]["step"] == 0

    # the trainer resumes from it: step 0, the benchmark baseline is scored (bench_path None != benchmark_eval_path), moments continue (step counts 4 -> 7)
    cfgd = json.load(open(work / "rl" / "config.json"))["config"]
    cfgd.update(out_dir=str(out_dir), steps=3, resume=True, resume_reset_selection=True, model_name=start, save_every=1, eval_every=0)
    summ = train_rl(dataclass_from_dict(RLConfig, cfgd), tokenizer=FakeGemmaTokenizer(), scorer=FakeScorer())
    assert summ["steps"] == 3
    recs = [json.loads(l) for l in open(out_dir / "metrics.jsonl") if l.strip()]
    assert any(r.get("baseline") == 1.0 and r["step"] == 0 for r in recs)
    ck_new = torch.load(out_dir / "last" / "trainer_state.pt", map_location="cpu", weights_only=False)
    assert ck_new["state"]["step"] == 3
    assert {int(s["step"]) for s in ck_new["optimizer"]["state"].values()} == {7}
