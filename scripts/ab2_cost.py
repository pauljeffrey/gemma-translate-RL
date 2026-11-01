#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""AB2 cost model (planning calculator, no GPU): GPU hours and dollars of the cheap ablation, central / high, at 1.19 USD/h, with the owner's hard cap of 9 USD INCLUDING a 15% margin.

    python scripts/ab2_cost.py [--rate 1.19] [--budget 9] [--margin 0.15] [--steps 49] [--sft-steps 3500] [--s1-n N]

IN the total: the weak SFT (3,500 steps + 7 validation passes + start-up + conversion), two RL arms (start-up, step-0 dev evaluation, 49 b64 steps, dev evaluations every 10 steps,
snapshot), S1 (N chosen so that its job equals one RL arm, rule of DECISION_RULE.txt) and S1P, the eight evaluations' GPU time (F0 W0 F_RL W_RL W_S1 W_S1P = 6 systems: generation for
yor kin orm lin x FLORES+ / NTREX / MAFAND / SMOL + AfriCOMET / SSA-COMET), hand-offs between jobs. Evaluations that run NEXT to a trainer cost the trainer +20% instead of their own time.
OUT of the total: CPU work (prompt pool, S1 data build, analysis; the box is paid for by the hour either way, but nothing here waits for it), the wait for the final run (the clock starts when it ends),
the 5 min of downloading the presigned packed set (CPU/network, overlaps the weak SFT), Hub pushes (none), any extension to 98 steps (+~0.3 h per arm, see the last line), a re-run after a crash.
Inputs: [M] measured, [A] assumed.
  [M] RL step of phase 2c, b256, 768 completions: median 29.1-31.7 s (rl2c_metrics.jsonl); b64 step (1,024 completions of ~75 tokens): never measured on this card [A]: 15 s central (~half a b256 step: 2.6x fewer generated
      tokens, the per-step fixed cost - weight sync 0.4 s, LID + consensus ~2 s, optimizer - stays), 20 s high.
  [M] SFT step 2.60 s (last 2,138 steps of the real run, card alone); [A] 3.3 s high (shared card).
  [A] weak arm slower by 10% / 25% (a weak policy loops more: vLLM decodes up to the 384-token cap).
  [A] RL start-up (vLLM engine, COMET worker, GlotLID self-test, step-0 dev evaluation) 8 / 12 min (AB1 assumed 10 / 15 incl. the first dev evaluation); dev evaluation (1,200 clean dev sentences, 256 tokens, + 128 reward-eval prompts) 0.5 / 1.0 min x 6.
  [M] neural scoring throughput 458 pairs/s for SSA-COMET-QE (E1 log); [A] 15k sentences x (AfriCOMET ref + QE, SSA ref (3 views) + QE) = 3.5 / 6 min; generation (vLLM 0.1 of the card) 3 / 5 min incl. engine start.
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=1.19)
    ap.add_argument("--budget", type=float, default=9.0)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--steps", type=int, default=49)
    ap.add_argument("--sft-steps", type=int, default=3500)
    ap.add_argument("--s1-n", type=int, default=0, help="0 = chosen so that the S1 job equals one RL arm (scripts/ab2_set_n.py rule)")
    a = ap.parse_args()
    cases = {"central": dict(sft=2.6, b64=15, weak=1.10, start=8, dev=0.5, ev_gen=3, ev_neu=3.5, handoff=0.25, sftfix=5 + 7 * 1.0 + 0.5, s1fix=2 + 3 * 1.0 + 0.5),
             "high": dict(sft=3.3, b64=20, weak=1.25, start=12, dev=1.0, ev_gen=5, ev_neu=6, handoff=0.5, sftfix=8 + 7 * 1.5 + 1.0, s1fix=3 + 3 * 1.5 + 1.0)}
    print(f"rate ${a.rate}/h, cap ${a.budget:.2f} INCLUDING a {int(a.margin * 100)}% margin => plan <= ${a.budget / (1 + a.margin):.2f} ({a.budget / (1 + a.margin) / a.rate:.2f} h)")
    for name, c in cases.items():
        n_dev = a.steps // 10 + 2
        def arm(factor):
            return (c["start"] + a.steps * c["b64"] * factor / 60 + n_dev * c["dev"] + 1.0) / 60            # + snapshot / last/ write
        f, w = arm(1.0), arm(c["weak"])
        sft = (a.sft_steps * c["sft"] + c["sftfix"] * 60) / 3600
        n_s1 = a.s1_n or int((f * 3600 - c["s1fix"] * 60) // c["sft"])
        s1 = (n_s1 * c["sft"] + c["s1fix"] * 60) / 3600
        s1p = (19 * c["sft"] * 0.5 + (c["s1fix"] - 1.0) * 60) / 3600                    # ~19 steps of 8 packs (a quarter of a 32-pack step)
        ev_one = (c["ev_gen"] + c["ev_neu"]) / 60
        exposed = 2 * ev_one                                                              # W_S1P and W_S1 are evaluated after the last trainer (the others next to a trainer)
        hidden = 4 * ev_one * 0.20
        total = sft + f + w + s1 + s1p + exposed + hidden + c["handoff"]
        print(f"\n[{name}]  weak SFT {sft:.2f} h | arm F {f:.2f} h | arm W {w:.2f} h | S1 ({n_s1} steps) {s1:.2f} h | S1P {s1p:.2f} h | evals {exposed + hidden:.2f} h | hand-offs {c['handoff']:.2f} h"
              f"  =>  {total:.2f} h = ${total * a.rate:.2f}   (x{1 + a.margin:.2f} = ${total * a.rate * (1 + a.margin):.2f}; cap ${a.budget:.2f}: {'OK' if total * a.rate * (1 + a.margin) <= a.budget else 'OVER'})")
        ext = (a.steps * c["b64"] / 3600) * (1 + c["weak"]) + (a.steps * c["sft"] / 3600) * 0.0
        print(f"    optional extension of both RL arms to {2 * a.steps} steps: +{ext:.2f} h RL + S1 recomputed (+{a.steps * c['b64'] * (1 + c['weak']) / 3600:.2f} h) = ${ext * a.rate * 2:.2f} more")


if __name__ == "__main__":
    main()
