#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AB1 cost model (planning calculator, no GPU): GPU hours and dollars of the ablation = weak SFT + two RL arms + the six FLORES+ evaluations, at central / high-cost assumptions.

    python scripts/ab1_cost.py [--rate 1.19] [--sft-steps 3500] [--b64 100 --b256 400] [--budget 20]

Measured inputs (labelled [M]) and assumptions ([A]):
  [M] RL step time of phase 2c on this card, b256, 48 prompts x 16 = 768 completions: median 29.1-31.7 s over steps 1-396 (box_snapshot_2026-10-21/runs_metrics/rl2c.tgz, t_step).
  [A] b64 step (64 prompts x 16 = 1,024 completions of ~70 tokens): no measurement exists; 15 s central (about half a b256 step: 2.7x fewer generated tokens, 2.5x fewer QE segments,
      the per-step fixed cost - weight sync 0.4 s, LID + consensus ~2 s, optimizer - stays), 20 s high.
  [M] SFT step: the last 2,138 steps of the real run took 5,557 s incl. start-up, evals and Hub pushes = 2.60 s/step (varlen attention + fused loss, micro 16 x accum 2, card alone).
  [A] weak arm slower by 10% (central) / 25% (high): a weak policy loops more, vLLM decodes until the longest sample (cap 1,280 tokens at b256).
  [A] FLORES+ evaluation of one model (45 languages x ~1,000 sentences, 768 tokens, vLLM 0.1 of the card): 12 min; six of them, all run next to a trainer (+20% on that trainer) except
      the last one (W500), which is exposed.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=1.19)
    ap.add_argument("--sft-steps", type=int, default=3500)
    ap.add_argument("--b64", type=int, default=100)
    ap.add_argument("--b256", type=int, default=300)
    ap.add_argument("--budget", type=float, default=20.0)
    ap.add_argument("--margin", type=float, default=0.15)
    a = ap.parse_args()
    cases = {"central": dict(sft=2.6, b64=15, b256=30, weak=1.10, ev=12, dev=2.5, start=10, handoff=0.25, exposed=1),
             "high": dict(sft=3.3, b64=20, b256=38, weak=1.25, ev=15, dev=3.5, start=15, handoff=0.5, exposed=1)}
    print(f"rate ${a.rate}/h, budget ${a.budget} (+{int(a.margin * 100)}% margin => plan <= ${a.budget / (1 + a.margin):.2f})")
    for name, c in cases.items():
        sft = (a.sft_steps * c["sft"] + 7 * 60 + 300) / 3600                       # 7 validation passes, start-up, convert
        n_dev = a.b64 // 50 + a.b256 // 50 + 1
        def arm(factor):
            steps = (a.b64 * c["b64"] + a.b256 * c["b256"]) * factor / 3600
            return steps + (c["start"] + n_dev * c["dev"]) / 60 + 0.05
        f, w = arm(1.0), arm(c["weak"])
        ev_hidden = 5 * c["ev"] / 60 * 0.20                                          # five evals next to a trainer slow it by 20%
        ev_exposed = c["exposed"] * c["ev"] / 60
        total = sft + f + w + ev_hidden + ev_exposed + c["handoff"]
        print(f"\n[{name}]  SFT-weak {sft:.2f} h | arm F {f:.2f} h | arm W {w:.2f} h | evals {ev_hidden + ev_exposed:.2f} h | hand-offs {c['handoff']:.2f} h  =>  {total:.2f} h = ${total * a.rate:.2f}"
              f"  (x{1 + a.margin:.2f} = ${total * a.rate * (1 + a.margin):.2f}; budget left ${a.budget - total * a.rate:.2f})")
        steps = a.b64 * c["b64"] / 3600
        print(f"    arm F detail: b64 {a.b64} x {c['b64']} s = {steps:.2f} h, b256 {a.b256} x {c['b256']} s = {a.b256 * c['b256'] / 3600:.2f} h, start-up {c['start']} min, {n_dev} dev evals x {c['dev']} min")


if __name__ == "__main__":
    main()
