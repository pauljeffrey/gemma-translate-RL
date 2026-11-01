#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPU smoke test of the vLLM rollout sampler (DECISIONS.md D-28). Needs a CUDA GPU, vLLM (in this interpreter for
--mode inprocess, in --vllm-python for --mode worker) and torch + transformers in THIS interpreter.

  # worker mode: trainer env = venv_gemma, engine env = venv_vllm
  PYTHONPATH=/workspace/sft_build/gemma_mt /workspace/venv_gemma/bin/python scripts/vllm_smoke.py \
      --model /path/to/checkpoint --mode worker --vllm-python /workspace/venv_vllm/bin/python \
      --prompts /workspace/sft_build/gemma/rl_prompts_v2/yor.parquet --out smoke.json
  # in-process mode: ONE interpreter that has both (venv_vllm + pandas/pyarrow from a --target dir)
  PYTHONPATH=/workspace/sft_build/gemma_mt:/workspace/sft_build/pylibs_vllm /workspace/venv_vllm/bin/python scripts/vllm_smoke.py \
      --model /path/to/checkpoint --mode inprocess ...

Sections (--only a,b,..; default all): sync stop dist reward loop throughput failure
  sync        two consecutive weight syncs of deliberately perturbed weights: vLLM greedy ids / log-probs vs HF for the SAME weights
  stop        stop ids (<end_of_turn>, <eos>, one extra), truncation (finish_reason length), post_stop_tokens == 0, ids returned verbatim
  dist        first-token distribution at T=1 / top_p=1 / top_k=0 vs the HF softmax (no hidden truncation)
  reward      rollout_and_reward / generate_translations of the trainer through the engine (real reward code, fake QE scorer), vs HF
  loop        8 x (new weights -> sync -> G=8 x 20 prompts at b64): steady-state times, memory growth, level-2 sleep with --sleep
  throughput  G=8 samples/prompt at the per-bin max_new_tokens, vLLM then HF on the same prompts, sync time, peak GPU memory
  failure     engine killed -> SamplerError; start-up that cannot fit -> SamplerError; prompt too long -> SamplerError
Memory: the GPU may be shared. `--gpu-gib-budget` is only documented, the actual limits are --kv-cache-gib / --gpu-util (engine) and
--hf-mem-fraction (HF phase, enforced with torch.cuda.set_per_process_memory_fraction).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List

BIN_NEW_TOKENS = {"b64": 384, "b256": 1280, "b512": 2304, "b1024": 3584}      # configs/rl.yaml max_new_tokens_by_bin


class MemPoller:
    """Peak GPU memory of this process and its children according to nvidia-smi (the GPU may be shared with foreign jobs)."""

    def __init__(self, pids):
        self.pids, self.peak, self.peak_per_pid, self._stop = set(pids), 0, {}, threading.Event()
        self.last: Dict[int, int] = {}
        self.th = threading.Thread(target=self._run, daemon=True)

    def add(self, pid):
        self.pids.add(pid)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=10).stdout
                tot = 0
                cur: Dict[int, int] = {}
                for line in out.strip().splitlines():
                    pid, mem = [x.strip() for x in line.split(",")]
                    if int(pid) in self.pids:
                        m = int(mem)
                        tot += m
                        cur[int(pid)] = m
                        self.peak_per_pid[int(pid)] = max(self.peak_per_pid.get(int(pid), 0), m)
                self.peak = max(self.peak, tot)
                self.last = cur
            except Exception:
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.th.join(5)

    def report(self):
        return {"peak_total_mib": self.peak, "peak_per_pid_mib": self.peak_per_pid}


def perturb(policy, w0, seed: int, rel: float):
    """policy <- W0 + noise (NOT cumulative): every parameter gets gaussian noise of `rel` x its own std. seed < 0: restore W0."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(max(seed, 0))
    with torch.no_grad():
        for n, p in policy.named_parameters():
            p.copy_(w0[n])
            if seed >= 0:
                noise = torch.randn(p.shape, generator=g).to(p.device, p.dtype)
                p.add_(noise * (rel * p.float().std().item()))


def load_prompts(path: str, tok, per_bin: Dict[str, int], seed: int = 0):
    import pandas as pd

    from gemma_mt import format as fmt

    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_json(path, lines=True)
    out = {}
    for b, n in per_bin.items():
        sub = df[df["length_bin"] == b].sample(n=n, random_state=seed)
        out[b] = [fmt.build_prompt(r.english, r.iso, tok) for r in sub.itertuples()]
    return out


def hf_logprobs(policy, prompts, comps, autocast_ctx, pad_id):
    """Teacher-forced log p(completion tokens) under the HF policy, exactly the function the trainer's loss uses."""
    import torch

    from gemma_mt.losses import completion_logprobs
    from gemma_mt.rl.sampling import assemble_batch

    b = assemble_batch(prompts, comps, pad_id, "cuda")
    policy.eval()
    with torch.no_grad(), autocast_ctx():
        lp = completion_logprobs(policy, b["input_ids"], b["attention_mask"], b["prompt_width"], b["completion_mask"], 1.0)
    return lp.cpu()


def _lp_stats(v_lists, h_mat):
    import torch

    d = torch.cat([(torch.tensor(v) - h_mat[r, : len(v)]).abs() for r, v in enumerate(v_lists)])
    return {"mean": d.mean().item(), "p99": d.quantile(0.99).item(), "max": d.max().item()}


def section_sync(args, sampler, engine, policy, w0, tok, prompts16, stop_ids, pad_id, autocast_ctx):
    """Weight sync correctness. For each of two consecutive syncs of DIFFERENT perturbed weights the engine's greedy ids and log-probs
    are compared with HF scoring the SAME weights in three precisions: fp32 (exact), fp32 weights under bf16 autocast (what the trainer's
    loss uses) and pure bf16 weights (what the engine computes with). A broken sync (stale / partial / mis-mapped weights) shows up as a
    large gap to ALL three; ordinary bf16 noise shows up as a small gap to bf16 and a somewhat larger one to fp32."""
    import copy

    import torch

    from gemma_mt.rl.sampling import generate_completions

    res: Dict[str, Any] = {"perturb_rel": args.perturb_rel, "syncs": []}
    prev_out = None
    for k, seed in enumerate((101, 202)):
        perturb(policy, w0, seed, args.perturb_rel)
        dt = sampler.sync(policy, version=k + 1)
        roll = sampler.generate(prompts16, group_size=1, max_new_tokens=48, temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids,
                                pad_id=pad_id, greedy=True, return_logprobs=True)
        hf = generate_completions(policy, prompts16, group_size=1, max_new_tokens=48, temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids,
                                  pad_id=pad_id, device="cuda", greedy=True, autocast=autocast_ctx())
        none = lambda: torch.autocast(device_type="cuda", enabled=False)       # noqa: E731
        lp_fp32 = hf_logprobs(policy, roll.prompts, roll.completions, none, pad_id)
        lp_ac = hf_logprobs(policy, roll.prompts, roll.completions, autocast_ctx, pad_id)
        bf = copy.deepcopy(policy).to(torch.bfloat16)
        lp_bf = hf_logprobs(bf, roll.prompts, roll.completions, none, pad_id)
        del bf
        # is the engine's token the HF argmax? (teacher forced on the engine's ids; fraction of positions)
        eq = sum(1 for a, b in zip(roll.completions, hf.completions) if a == b)
        div = [next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b))) for a, b in zip(roll.completions, hf.completions)]
        rec = {"sync_seconds": dt, "sync_info": getattr(sampler, "last_sync_info", None), "n_tokens": sum(len(c) for c in roll.completions),
               "vllm_vs_hf_fp32": _lp_stats(roll.sampler_logp, lp_fp32), "vllm_vs_hf_autocast_bf16": _lp_stats(roll.sampler_logp, lp_ac),
               "vllm_vs_hf_pure_bf16": _lp_stats(roll.sampler_logp, lp_bf),
               "hf_autocast_vs_hf_fp32(reference noise)": _lp_stats([[0.0] * len(c) for c in roll.completions], (lp_ac - lp_fp32)),
               "greedy_identical_sequences_vs_hf_trainer_path": f"{eq}/{len(prompts16)}", "first_divergence_index_median": sorted(div)[len(div) // 2],
               "vllm_first_completion_ids": roll.completions[0][:12]}
        if prev_out is not None:
            rec["completions_changed_vs_previous_sync"] = sum(1 for a, b in zip(roll.completions, prev_out) if a != b)
        prev_out = roll.completions
        res["syncs"].append(rec)
        print(f"[sync {k + 1}] {dt:.2f}s  |dlogp| vs fp32 {rec['vllm_vs_hf_fp32']}  vs autocast {rec['vllm_vs_hf_autocast_bf16']}  vs pure bf16 {rec['vllm_vs_hf_pure_bf16']}  "
              f"identical {rec['greedy_identical_sequences_vs_hf_trainer_path']}  changed {rec.get('completions_changed_vs_previous_sync')}", flush=True)
    # same prompts again after a third, different sync: a stale prefix cache would reproduce the OLD continuation
    perturb(policy, w0, 303, args.perturb_rel)
    sampler.sync(policy, version=3)
    roll3 = sampler.generate(prompts16, group_size=1, max_new_tokens=48, temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids,
                             pad_id=pad_id, greedy=True, return_logprobs=True)
    lp3 = hf_logprobs(policy, roll3.prompts, roll3.completions, autocast_ctx, pad_id)
    res["third_sync_same_prompts_again"] = {"vllm_vs_hf_autocast_bf16": _lp_stats(roll3.sampler_logp, lp3),
                                            "completions_changed_vs_sync2": sum(1 for a, b in zip(roll3.completions, prev_out) if a != b)}
    # strong perturbation: the model is nearly garbage; documents how far bf16 and fp32 drift apart when the logits are huge
    perturb(policy, w0, 404, 5 * args.perturb_rel)
    sampler.sync(policy, version=4)
    roll4 = sampler.generate(prompts16, group_size=1, max_new_tokens=48, temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids,
                             pad_id=pad_id, greedy=True, return_logprobs=True)
    none = lambda: torch.autocast(device_type="cuda", enabled=False)           # noqa: E731
    bf = copy.deepcopy(policy).to(torch.bfloat16)
    res["strong_perturbation(5x)"] = {"vllm_vs_hf_pure_bf16": _lp_stats(roll4.sampler_logp, hf_logprobs(bf, roll4.prompts, roll4.completions, none, pad_id)),
                                      "vllm_vs_hf_fp32": _lp_stats(roll4.sampler_logp, hf_logprobs(policy, roll4.prompts, roll4.completions, none, pad_id))}
    del bf
    print("[sync 3/4]", json.dumps({k: res[k] for k in ("third_sync_same_prompts_again", "strong_perturbation(5x)")}), flush=True)
    perturb(policy, w0, -1, 0.0)                       # back to W0 for the other sections
    sampler.sync(policy, version=5)
    return res


def section_stop(args, sampler, policy, tok, prompts, stop_ids, pad_id, autocast_ctx):
    from collections import Counter

    res: Dict[str, Any] = {}
    # 1. normal sampling, generous budget: finished rows end in exactly one stop id; tiny budget: unfinished rows are max_new_tokens long
    for budget in (6, 64):
        roll = sampler.generate(prompts, group_size=8, max_new_tokens=budget, temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids,
                                pad_id=pad_id, key=7)
        ok_fin = all((c[-1] in stop_ids) and not any(t in stop_ids for t in c[:-1]) for c, f in zip(roll.completions, roll.finished) if f)
        ok_unf = all(len(c) == budget and not any(t in stop_ids for t in c) for c, f in zip(roll.completions, roll.finished) if not f)
        res[f"budget_{budget}"] = {"n": len(roll.completions), "finished": sum(roll.finished), "all_finished_end_in_one_stop_id": ok_fin,
                                   "all_unfinished_have_budget_tokens_and_no_stop": ok_unf, "post_stop_tokens_sum": sum(roll.post_stop_tokens),
                                   "max_len": max(len(c) for c in roll.completions), "which_stop": dict(Counter(c[-1] for c, f in zip(roll.completions, roll.finished) if f))}
        assert ok_fin and ok_unf and sum(roll.post_stop_tokens) == 0 and max(len(c) for c in roll.completions) <= budget, res[f"budget_{budget}"]
    # 2. an extra stop id (the most frequent non-stop token of a sample) must also end a completion and be INCLUDED
    roll = sampler.generate(prompts, group_size=2, max_new_tokens=32, temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids, pad_id=pad_id, key=8)
    extra = Counter(t for c in roll.completions for t in c if t not in stop_ids).most_common(1)[0][0]
    ids2 = list(stop_ids) + [extra]
    roll2 = sampler.generate(prompts, group_size=2, max_new_tokens=32, temperature=1.0, top_p=1.0, top_k=0, stop_ids=ids2, pad_id=pad_id, key=8)
    n_extra = sum(1 for c, f in zip(roll2.completions, roll2.finished) if f and c[-1] == extra)
    assert all((c[-1] in ids2) and not any(t in ids2 for t in c[:-1]) for c, f in zip(roll2.completions, roll2.finished) if f)
    res["extra_stop_id"] = {"id": extra, "completions_ended_by_it": n_extra, "all_ok": True}
    # 3. same seed key -> same ids (deterministic seeding per step), another key -> different
    a = sampler.generate(prompts[:4], group_size=8, max_new_tokens=32, temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids, pad_id=pad_id, key=11)
    b = sampler.generate(prompts[:4], group_size=8, max_new_tokens=32, temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids, pad_id=pad_id, key=11)
    c = sampler.generate(prompts[:4], group_size=8, max_new_tokens=32, temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids, pad_id=pad_id, key=12)
    n_same_in_group = sum(1 for g in range(4) for i in range(1, 8) if a.completions[g * 8 + i] == a.completions[g * 8])
    res["seeding"] = {"same_key_identical": a.completions == b.completions, "other_key_differs": a.completions != c.completions,
                      "duplicate_samples_inside_groups(of 28 pairs vs first)": n_same_in_group}
    print("[stop]", json.dumps(res), flush=True)
    return res


def section_dist(args, sampler, policy, tok, prompts, stop_ids, pad_id, autocast_ctx):
    """First-token distribution of N samples vs (a) vLLM's OWN reported probabilities (z-scores ~ N(0,1) when the sampler draws what it
    reports, no hidden top-k / top-p) and (b) HF's softmax in fp32 / autocast / pure bf16 (their distance is bf16 logit noise).
    Case 2 uses T = 2 so that a lot of mass sits beyond the 64 most likely tokens: a hidden top_k=64 would be obvious there."""
    import copy

    import torch

    N = args.dist_samples
    none = lambda: torch.autocast(device_type="cuda", enabled=False)           # noqa: E731
    bf = copy.deepcopy(policy).to(torch.bfloat16)
    res = []
    for pi, temp in ((0, 1.0), (1, 1.0), (2, 2.0)):
        p = prompts[pi]
        roll = sampler.generate([p], group_size=N, max_new_tokens=1, temperature=temp, top_p=1.0, top_k=0, stop_ids=stop_ids, pad_id=pad_id,
                                key=100 + pi, return_logprobs=True)
        first = torch.tensor([c[0] for c in roll.completions])
        own_lp = {}
        for c, l in zip(roll.completions, roll.sampler_logp):
            own_lp.setdefault(c[0], []).append(l[0])
        own_p = torch.zeros(262144)
        for t, ls in own_lp.items():
            own_p[t] = math.exp(sum(ls) / len(ls))
        counts = torch.bincount(first, minlength=262144).float()

        def hf_probs(model, ctx):
            model.eval()
            with torch.no_grad(), ctx():
                out = model(input_ids=torch.tensor([p], device="cuda"), use_cache=False)
            return torch.softmax(out.logits[0, -1].float() / temp, -1).cpu()

        ref = {"fp32": hf_probs(policy, none), "autocast": hf_probs(policy, autocast_ctx), "pure_bf16": hf_probs(bf, none)}
        top = torch.topk(ref["fp32"], 64)
        tail_mass = 1.0 - top.values.sum().item()
        tail_frac = float((~torch.isin(first, top.indices)).float().mean())
        seen = (counts > 0) & (N * own_p >= 30)          # normal approximation only where >= 30 draws are expected
        z_own = ((counts[seen] - N * own_p[seen]) / (N * own_p[seen] * (1 - own_p[seen])).sqrt().clamp(min=1e-9)).abs()
        if temp != 1.0 or not seen.any():                 # reported log-probs are the raw (T = 1) ones
            z_own = torch.zeros(1)
        rec = {"prompt": pi, "temperature": temp, "n": N, "distinct_tokens_sampled": int((counts > 0).sum()),
               "mass_beyond_hf_top64": tail_mass, "vllm_fraction_beyond_hf_top64": tail_frac,
               "tail_sigma": math.sqrt(max(tail_mass * (1 - tail_mass) / N, 1e-12)),
               "max_abs_z_vs_vllm_own_probs(tokens with >=30 expected draws, T=1 only)": float(z_own.max()), "n_tokens_in_z": int(seen.sum()),
               "tv_empirical_vs": {k: 0.5 * (counts / N - v).abs().sum().item() for k, v in ref.items()},
               "tv_vllm_reported_vs": {k: 0.5 * (own_p - v).abs().sum().item() for k, v in ref.items()},
               "tv_fp32_vs_pure_bf16(logit noise)": 0.5 * (ref["fp32"] - ref["pure_bf16"]).abs().sum().item()}
        res.append(rec)
        print("[dist]", json.dumps(rec), flush=True)
    del bf
    return res


def section_reward(args, sampler, policy, tok, stop_ids, pad_id, autocast_ctx):
    """The trainer's own functions with the engine behind them: rollout_and_reward (real tokenizer, real reward code, fake QE scorer),
    HF vs vLLM status mix, and the greedy dev-benchmark path (generate_translations with sampler=)."""
    from collections import Counter

    import pandas as pd

    from gemma_mt.eval.eval_translation import generate_translations
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.rl.data import RLPrompts
    from gemma_mt.rl.reward import RewardConfig, TranslationReward
    from gemma_mt.rl.train_rl import RLConfig, rollout_and_reward
    from gemma_mt.utils import dataclass_from_dict

    df = pd.read_parquet(args.prompts) if args.prompts.endswith(".parquet") else pd.read_json(args.prompts, lines=True)
    sub = df[df["length_bin"] == "b64"].sample(n=16, random_state=3)
    prompts = RLPrompts(list(sub.english), list(sub.iso), ["smoke"] * 16, list(sub.src_afr_tokens), {}, 0, list(sub.length_bin))
    cfg = RLConfig(group_size=8, max_new_tokens=384, temperature=1.0)
    reward = TranslationReward(dataclass_from_dict(RewardConfig, {"max_comet_tokens": 510}), FakeScorer(), tok, stop_ids, pad_id=pad_id)
    res: Dict[str, Any] = {}
    for name, smp in (("vllm", sampler), ("hf", None)):
        t0 = time.time()
        roll, samples, out = rollout_and_reward(policy, tok, reward, prompts, list(range(8)), cfg, stop_ids, pad_id, "cuda", autocast_ctx(),
                                                sampler=smp, key=5)
        res[name] = {"seconds": time.time() - t0, "n": len(roll.completions), "finished_rate": sum(roll.finished) / len(roll.finished),
                     "status": dict(Counter(out.status)), "reward_mean": sum(out.rewards) / len(out.rewards),
                     "post_stop_tokens_sum": sum(roll.post_stop_tokens), "mean_len": sum(len(c) for c in roll.completions) / len(roll.completions),
                     "rewards_finite": all(r == r for r in out.rewards)}
        print(f"[reward {name}]", json.dumps(res[name]), flush=True)
    rows = [{"english": e, "iso": i, "reference": e, "benchmark": "smoke", "id": str(k)} for k, (e, i) in enumerate(zip(prompts.english, prompts.iso))]
    v = generate_translations(None, tok, rows, stop_ids=stop_ids, pad_id=pad_id, device="cuda", max_new_tokens=128, sampler=sampler)
    h = generate_translations(policy, tok, rows, stop_ids=stop_ids, pad_id=pad_id, device="cuda", max_new_tokens=128, autocast=autocast_ctx())
    res["greedy_dev_path"] = {"rows": len(rows), "identical_hyp": sum(a["hyp"] == b["hyp"] for a, b in zip(v, h)),
                              "finished_vllm": sum(a["finished"] for a in v), "finished_hf": sum(b["finished"] for b in h),
                              "example_vllm": v[0]["raw"][:120], "example_hf": h[0]["raw"][:120]}
    print("[greedy dev path]", json.dumps(res["greedy_dev_path"]), flush=True)
    return res


def section_loop(args, sampler, policy, w0, tok, mem, stop_ids, pad_id, steady_steps: int):
    """The training cadence without training: `steady_steps` x (new weights -> sync -> G=8 x 20 prompts at b64), to see steady-state times and
    whether the engine's / the process's memory grows. With --sleep the engine sleeps after every rollout and wakes up at the next sync."""
    prompts = load_prompts(args.prompts, tok, {"b64": 20}, seed=5)["b64"]
    rows = []
    for k in range(steady_steps):
        perturb(policy, w0, 500 + k, args.perturb_rel)
        t0 = time.time()
        ts = sampler.sync(policy, version=700 + k)
        t_sync = time.time() - t0
        t0 = time.time()
        roll = sampler.generate(prompts, group_size=8, max_new_tokens=BIN_NEW_TOKENS["b64"], temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids,
                                pad_id=pad_id, key=900 + k)
        t_gen = time.time() - t0
        mem_awake = dict(mem.last)
        t0 = time.time()
        sampler.sleep()
        t_sleep = time.time() - t0
        time.sleep(1.2)
        rows.append({"step": k, "sync_s": t_sync, "gen_s": t_gen, "tokens": sum(len(c) for c in roll.completions), "sleep_s": t_sleep,
                     "mem_mib_awake": mem_awake, "mem_mib_after_sleep": dict(mem.last)})
    perturb(policy, w0, -1, 0.0)
    print("[loop]", json.dumps([{k: v for k, v in r.items() if k != "tokens"} for r in rows]), flush=True)
    return rows


def section_throughput(args, sampler, engine, policy, tok, mem, stop_ids, pad_id, autocast_ctx, hf_phase: bool):
    import torch

    from gemma_mt.rl.vllm_sampler import HFSampler

    spec = args.hf_bins if hf_phase else args.bins
    per_bin = {kv: int(kv.split(":")[1]) for kv in spec.split(",")}            # key "b256:10" -> 10 prompts of bin b256
    prompts = {k: load_prompts(args.prompts, tok, {k.split(":")[0]: n})[k.split(":")[0]] for k, n in per_bin.items()}
    res: Dict[str, Any] = {"bins": per_bin, "vllm": {}, "hf": {}}
    if not hf_phase:
        # warm-up (Triton / CUDA graph capture are one-off costs), reported separately
        t0 = time.time()
        sampler.generate(next(iter(prompts.values()))[:2], group_size=8, max_new_tokens=16, temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids, pad_id=pad_id, key=0)
        res["vllm_first_call_seconds"] = time.time() - t0
        for b, ps in prompts.items():
            policy_v = 10 + list(per_bin).index(b)
            ts = sampler.sync(policy, version=1000 + policy_v)      # a real sync before every step, as in training
            t0 = time.time()
            roll = sampler.generate(ps, group_size=8, max_new_tokens=BIN_NEW_TOKENS[b.split(":")[0]], temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids,
                                    pad_id=pad_id, key=policy_v)
            dt = time.time() - t0
            ntok = sum(len(c) for c in roll.completions)
            res["vllm"][b] = {"prompts": len(ps), "completions": len(roll.completions), "tokens": ntok, "mean_len": ntok / len(roll.completions),
                              "max_len": max(len(c) for c in roll.completions), "finished_rate": sum(roll.finished) / len(roll.finished),
                              "gen_seconds": dt, "tokens_per_s": ntok / dt, "sync_seconds": ts, "step_seconds_sync_plus_gen": ts + dt}
            print(f"[throughput vllm {b}] {len(roll.completions)} samples {ntok} tok in {dt:.1f}s = {ntok / dt:.0f} tok/s  mean len {ntok / len(roll.completions):.0f} "
                  f"max {max(len(c) for c in roll.completions)} fin {sum(roll.finished) / len(roll.finished):.2f}  sync {ts:.2f}s", flush=True)
    else:
        torch.cuda.reset_peak_memory_stats()
        hf = HFSampler(policy, "cuda", autocast_ctx)
        for b, ps in prompts.items():
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                roll = hf.generate(ps, group_size=8, max_new_tokens=BIN_NEW_TOKENS[b.split(":")[0]], temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids, pad_id=pad_id)
            except torch.cuda.OutOfMemoryError as e:
                res["hf"][b] = {"prompts": len(ps), "error": "CUDA OOM within the memory budget: " + str(e)[:120]}
                print(f"[throughput hf {b}] OOM", flush=True)
                torch.cuda.empty_cache()
                continue
            torch.cuda.synchronize()
            dt = time.time() - t0
            ntok = sum(len(c) for c in roll.completions)
            res["hf"][b] = {"prompts": len(ps), "completions": len(roll.completions), "tokens": ntok, "mean_len": ntok / len(roll.completions),
                            "max_len": max(len(c) for c in roll.completions), "finished_rate": sum(roll.finished) / len(roll.finished),
                            "gen_seconds": dt, "tokens_per_s": ntok / dt, "peak_torch_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
            print(f"[throughput hf {b}] {len(roll.completions)} samples {ntok} tok in {dt:.1f}s = {ntok / dt:.0f} tok/s  max len "
                  f"{max(len(c) for c in roll.completions)}  peak torch {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB", flush=True)
    return res


def section_failure(args, ec, stop_ids, pad_id, prompts, make_engine):
    from gemma_mt.rl.vllm_engine import EngineConfig, SamplerError, SamplingSpec, VLLMEngine, WorkerEngine
    from gemma_mt.rl.vllm_sampler import VLLMSampler

    res: Dict[str, Any] = {}
    # (1) a prompt that does not fit max_model_len fails loudly (never silently truncated)
    eng = make_engine(ec)
    sam = VLLMSampler(eng, seed=0)
    sam._version = 0
    try:
        sam.generate([[2] * 50], group_size=2, max_new_tokens=ec.max_model_len, temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids, pad_id=pad_id)
        res["too_long"] = "NO ERROR (bad)"
    except SamplerError as e:
        res["too_long"] = "SamplerError: " + str(e)[:140]
    # (2) kill the engine process: the next call raises, quickly (worker mode only: in-process the trainer itself would be gone)
    if isinstance(eng, WorkerEngine):
        os.kill(eng.proc.pid, 9)
        t0 = time.time()
        try:
            sam.generate(prompts[:2], group_size=2, max_new_tokens=8, temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop_ids, pad_id=pad_id)
            res["engine_killed"] = "NO ERROR (bad)"
        except SamplerError as e:
            res["engine_killed"] = f"SamplerError after {time.time() - t0:.1f}s: " + str(e).splitlines()[0][:140]
    eng.close()
    # (3) an engine that cannot fit (KV cache of 0.01 GiB for max_model_len 8192) fails at start-up with the reason
    bad = EngineConfig(**{**ec.__dict__, "kv_cache_gib": 0.005, "enforce_eager": True})
    t0 = time.time()
    try:
        e2 = make_engine(bad)
        e2.close()
        res["startup_cannot_fit"] = "NO ERROR (bad)"
    except SamplerError as e:
        res["startup_cannot_fit"] = f"SamplerError after {time.time() - t0:.0f}s: " + str(e).replace("\n", " | ")[:260]
    print("[failure]", json.dumps(res, indent=1), flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint dir (config + tokenizer + weights) or Hub id")
    ap.add_argument("--engine-model", default=None, help="checkpoint dir the ENGINE loads its config from (default: --model); e.g. a dir with a "
                    "config.json written by the trainer's transformers version")
    ap.add_argument("--mode", choices=["inprocess", "worker"], default="worker")
    ap.add_argument("--vllm-python", default=None, help="interpreter with vLLM for --mode worker")
    ap.add_argument("--prompts", required=True, help="rl prompts parquet/jsonl with english, iso, length_bin")
    ap.add_argument("--out", default="vllm_smoke.json")
    ap.add_argument("--only", default="sync,stop,dist,reward,throughput,failure")
    ap.add_argument("--bins", default="b64:20,b256:20,b256:10,b512:20,b512:8,b1024:20,b1024:4",
                    help="bin:prompts per optimizer step for the vLLM throughput phase (the same bin may appear with several counts)")
    ap.add_argument("--hf-bins", default="b64:20,b256:10,b512:8,b1024:4", help="same for the HF phase (limited by the memory cap)")
    ap.add_argument("--gpu-util", type=float, default=0.09)
    ap.add_argument("--kv-cache-gib", type=float, default=2.0)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--attention-backend", default=None, help="vLLM attention backend (default: vLLM's choice = FLASH_ATTN on sm_120)")
    ap.add_argument("--sleep", action="store_true", help="enable_sleep_mode (level-2 sleep between rollouts is tested in the throughput section)")
    ap.add_argument("--hf-mem-fraction", type=float, default=0.09, help="cap for this process during the HF phase")
    ap.add_argument("--dist-samples", type=int, default=4096)
    ap.add_argument("--loop-steps", type=int, default=8)
    ap.add_argument("--perturb-rel", type=float, default=0.01, help="noise std of the test perturbations, relative to each parameter's std")
    ap.add_argument("--sync-dir", default="/dev/shm")
    ap.add_argument("--no-hf-throughput", action="store_true")
    args = ap.parse_args(argv)
    only = set(args.only.split(","))

    import torch
    from transformers import AutoTokenizer

    from gemma_mt import format as fmt
    from gemma_mt.modeling import load_model
    from gemma_mt.rl.vllm_engine import EngineConfig, VLLMEngine, WorkerEngine
    from gemma_mt.rl.vllm_sampler import VLLMSampler

    tok = AutoTokenizer.from_pretrained(args.model)
    fmt.assert_template_matches_reference(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    policy = load_model(args.model, attn_implementation="sdpa", dtype=torch.float32, device="cuda")
    stop = fmt.resolve_stop_tokens(tok, policy.generation_config)
    fmt.ensure_generation_config(policy.generation_config, stop, pad_id=pad_id)
    stop_ids = stop.ids
    policy.config.use_cache = False
    w0 = {n: p.detach().clone() for n, p in policy.named_parameters()}

    def autocast_ctx():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    ec = EngineConfig(model=args.engine_model or args.model, gpu_memory_utilization=args.gpu_util, kv_cache_gib=args.kv_cache_gib, max_model_len=args.max_model_len,
                      max_num_seqs=args.max_num_seqs, enforce_eager=args.enforce_eager, enable_sleep_mode=args.sleep, seed=0,
                      attention_backend=args.attention_backend)

    def make_engine(cfg):
        if args.mode == "worker":
            return WorkerEngine(cfg, python=args.vllm_python, sync_dir=args.sync_dir, log_path=os.path.join(os.path.dirname(os.path.abspath(args.out)), "vllm_worker.log"))
        return VLLMEngine(cfg)

    report: Dict[str, Any] = {"mode": args.mode, "model": args.model, "stop_ids": stop_ids, "torch": torch.__version__,
                              "engine_config": ec.__dict__, "gpu": torch.cuda.get_device_name(0)}
    with MemPoller([os.getpid()]) as mem:
        t0 = time.time()
        engine = make_engine(ec)
        report["engine_start_seconds"] = time.time() - t0
        report["engine_info"] = engine.info()
        if args.mode == "worker":
            mem.add(engine.proc.pid)
        print(f"[engine] started in {report['engine_start_seconds']:.0f}s: {report['engine_info']}", flush=True)
        sampler = VLLMSampler(engine, seed=1234, sleep_between=args.sleep)
        prompts_by_bin = load_prompts(args.prompts, tok, {"b64": 16})
        prompts16 = prompts_by_bin["b64"]
        try:
            if "sync" in only:
                report["sync"] = section_sync(args, sampler, engine, policy, w0, tok, prompts16, stop_ids, pad_id, autocast_ctx)
            else:
                sampler.sync(policy, 0)
            if "stop" in only:
                report["stop"] = section_stop(args, sampler, policy, tok, prompts16, stop_ids, pad_id, autocast_ctx)
            if "dist" in only:
                report["dist"] = section_dist(args, sampler, policy, tok, prompts16, stop_ids, pad_id, autocast_ctx)
            if "reward" in only:
                report["reward"] = section_reward(args, sampler, policy, tok, stop_ids, pad_id, autocast_ctx)
            if "loop" in only:
                report["loop"] = section_loop(args, sampler, policy, w0, tok, mem, stop_ids, pad_id, args.loop_steps)
            if "throughput" in only:
                report["throughput_vllm"] = section_throughput(args, sampler, engine, policy, tok, mem, stop_ids, pad_id, autocast_ctx, hf_phase=False)
                if args.sleep:
                    t0 = time.time()
                    sampler.sleep()
                    report["sleep_seconds"] = time.time() - t0
                    time.sleep(1.5)
                    report["after_sleep_info"] = engine.info()
                    t0 = time.time()
                    engine.wake_up()
                    report["wake_seconds"] = time.time() - t0
                    sampler.invalidate()
                    report["sync_after_wake_seconds"] = sampler.sync(policy, 5000)
            report["memory_vllm_phase"] = mem.report()
            if "failure" in only:
                engine.close()
                engine = None
                time.sleep(2)
                report["failure"] = section_failure(args, ec, stop_ids, pad_id, prompts16, make_engine)
        finally:
            if engine is not None:
                engine.close()
        if "throughput" in only and not args.no_hf_throughput:
            time.sleep(3)
            torch.cuda.empty_cache()
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(args.hf_mem_fraction)
            mem.peak = 0
            mem.peak_per_pid = {}
            report["throughput_hf"] = section_throughput(args, None, None, policy, tok, mem, stop_ids, pad_id, autocast_ctx, hf_phase=True)
            report["memory_hf_phase"] = mem.report()
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("wrote", args.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
