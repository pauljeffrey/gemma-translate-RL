#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-61: would a stuffed / padded output WIN a GRPO group in the consensus / blend languages, with and without the v2 penalties?

For every held-out FLORES+ dev prompt of a language, a group of G=8 is built from the REAL policy samples of agent A5 (`policy_samples.jsonl`: 8 samples at T=1 of the rl2c checkpoint per prompt) in which k
members are replaced by an attack text (identical members for the deterministic attacks, independent draws for `stuff`), one by the human reference. Reward of a member = consensus value
(mean chrF++ against the other members mapped linearly to [-2, 2.25], rl/consensus.py) minus the penalties (copy cap, func, pad, repeat, length, word ratio, lexical, lexicon), floored at -2; hard gates
(is_copy_of_source, is_repetitive, is_self_repetitive, length_ratio_bad) give -2. Reported: mean reward of the attackers, of the honest policy members and of the human reference, and the share of
groups in which the attackers' group-normalised advantage (mean_std, as grpo.group_advantages) is POSITIVE (= the attack would be reinforced). Penalty weights: func 0.3, pad 1.0, copy scale 2.0.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_anchors_v2 as E  # noqa: E402
from gemma_mt.rl import anchors_attacks as AT  # noqa: E402
from gemma_mt.rl import anchors_v2 as V  # noqa: E402
from gemma_mt.rl import consensus as C  # noqa: E402
from gemma_mt.rl import guards  # noqa: E402
from gemma_mt.rl import textstats as T  # noqa: E402

WSETS = {"func0.3_pad1": {"func": 0.3, "pad": 1.0, "repeat": 1.0}, "func1_pad1": {"func": 1.0, "pad": 1.0, "repeat": 1.0}, "func2_pad2": {"func": 2.0, "pad": 2.0, "repeat": 1.0},
         "gate0.9_func0.3_pad1": {"func": 0.3, "pad": 1.0, "repeat": 1.0, "gate": 0.9}}


def member_reward(cons_chrf, d, lexw, W):
    if d["gate_copy"] or d["gate_loop"] or d["gate_len"]:
        return -2.0
    r = -2.0 + 4.25 * cons_chrf / 100.0
    pen = W["func"] * (d.get("func") or 0) + W["pad"] * (d.get("pad") or 0) + W["repeat"] * (d.get("repeat") or 0) + (d.get("len") or 0) + (d.get("wordratio") or 0) + (d.get("lexical") or 0)
    pen += lexw * (d.get("lexicon") or 0)
    r -= pen
    if d.get("copy"):
        r = min(r, -2.0 * d["copy"])
    return max(-2.0, r)


def adv(rewards):
    r = np.asarray(rewards, float)
    s = r.std()
    return np.zeros_like(r) if s < 1e-6 else (r - r.mean()) / (s + 1e-4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal", required=True)
    ap.add_argument("--human", required=True)
    ap.add_argument("--funcwords", required=True)
    ap.add_argument("--bigrams", default=None)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--lexicon-dir", required=True)
    ap.add_argument("--shaping", required=True)
    ap.add_argument("--lex-cal", required=True)
    ap.add_argument("--langs", default="fon,sag,umb,kab,kam,mos,kmb,kbp,kea,vmw,dyu,nus")
    ap.add_argument("--max-prompts", type=int, default=40)
    ap.add_argument("--docs", type=int, default=0, help="glue this many consecutive held-out prompts (and their samples, member by member) into one document-level prompt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cal = json.load(open(a.cal))
    fwd = json.load(open(a.funcwords))
    lex_cal = json.load(open(a.lex_cal))["languages"]
    for iso, blk in lex_cal.items():
        if "lexicon" in blk:
            cal["languages"].setdefault(iso, {})["lexicon"] = blk["lexicon"]
    A = V.Anchors2(cal, detectors=("entity", "repeat", "boiler", "lexicon"), lexicon_dir=a.lexicon_dir, bigram_path=a.bigrams)
    from gemma_mt.rl import lexicon as L
    from gemma_mt.rl.reward_shaping import ContentShaper

    shaper = ContentShaper.from_file(a.shaping, reliability_from="same_source", mode="calibrated", preset="repo", neg_scale=2.0)
    cfgw = {"length": 1.0, "wordratio": 0.5, "lexical": 1.0}
    pol = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(a.policy, encoding="utf-8"):
        r = json.loads(l)
        pol[r["iso"]][r["id"]].append(r)
    res = {}
    rng = random.Random(0)
    for iso in a.langs.split(","):
        if iso not in cal["languages"] or iso not in pol or iso not in fwd:
            continue
        lex = L.load_lexicon(iso, a.lexicon_dir)
        lex = lex if lex is not None and lex.validated else None
        lexw = min(0.25, float(lex.meta.get("weight", 0))) if lex is not None else 0.0
        fb = cal["languages"][iso].get("func") or {}
        ent = fwd[iso]
        fw = fb.get("fw") or [t[0] for t in ent["top"][:80]]
        prompts = [(pid, rows) for pid, rows in sorted(pol[iso].items()) if E.fold_of(iso, rows[0]["src"]) == 1 and len(rows) >= 8][: a.max_prompts]
        if a.docs:
            glued = []
            for w in range(0, len(prompts) - a.docs + 1, a.docs):
                win = prompts[w: w + a.docs]
                rows = [{"src": " ".join(p[1][0]["src"] for p in win), "ref": " ".join(p[1][0]["ref"] for p in win),
                         "hyp": " ".join(sorted(p[1], key=lambda r: r["k"])[m]["hyp"] for p in win)} for m in range(8)]
                glued.append((win[0][0], rows))
            prompts = glued
        units = [{"src": rows[0]["src"], "ref": rows[0]["ref"]} for _, rows in prompts]
        others = [rows[0]["ref"] for _, rows in prompts]
        atk = AT.make_attacks(units, ent["top"], fw, others, random.Random(5), lex=lex, fw_share=0.25)
        atk_names = ["stuff30", "pad", "pad_fit", "stuff_fw"] + (["gloss", "gloss_pad", "gloss_fw"] if lex is not None else [])
        R = {}
        for name in atk_names + ["human_only"]:
            for k in (2, 4, 6):
                if name == "human_only" and k != 2:
                    continue
                rows_out = []
                for j, (pid, rows) in enumerate(prompts):
                    hon = [r["mt"] if "mt" in r else r.get("hyp", "") for r in rows[:8]]
                    src, ref = rows[0]["src"], rows[0]["ref"]
                    if name == "human_only":
                        texts = hon[:7] + [ref]
                        labels = ["honest"] * 7 + ["human"]
                    else:
                        t_att = atk[name][j]
                        texts = [t_att] * k + [ref] + hon[: 7 - k]
                        labels = ["attack"] * k + ["human"] + ["honest"] * (7 - k)
                    profs = [C.Profile(t) for t in texts]
                    chrf = C.group_consensus(profs, 3)
                    ds = [E.detect_all(A, shaper, iso, src, t, lex_cal, a.lexicon_dir, cfgw) for t in texts]
                    rw_raw = [-2.0 + 4.25 * c / 100.0 if not (d["gate_copy"] or d["gate_loop"] or d["gate_len"]) else -2.0 for c, d in zip(chrf, ds)]
                    rw_pen = {}
                    for ws, W in WSETS.items():
                        if "gate" in W:                                     # D-61 anchors_stuff_gate: saturated func / pad deficit = hard failure, out of the consensus pool
                            gt = [((d.get("func") or 0) >= W["gate"]) or ((d.get("pad") or 0) >= W["gate"]) for d in ds]
                            ch2 = C.group_consensus([None if g else p for g, p in zip(gt, profs)], 3)
                            rw_pen[ws] = [-2.0 if g else member_reward(c if c is not None else 0.0, d, lexw, W) for g, c, d in zip(gt, ch2, ds)]
                        else:
                            rw_pen[ws] = [member_reward(c, d, lexw, W) for c, d in zip(chrf, ds)]
                    rows_out.append((labels, rw_raw, rw_pen))
                if name == "human_only":
                    R["honest_baseline"] = {ws: {"human": float(np.mean([ro[2][ws][7] for ro in rows_out])), "honest": float(np.mean([np.mean(ro[2][ws][:7]) for ro in rows_out]))} for ws in WSETS}
                    continue
                def val(ro, which, ws):
                    return ro[1] if which == "raw" else ro[2][ws]
                def agg(idx, which, ws=None):
                    return float(np.mean([np.mean([val(ro, which, ws)[i] for i, l in enumerate(ro[0]) if l == idx]) for ro in rows_out]))
                def pos(which, ws=None):
                    return float(np.mean([np.mean([adv(val(ro, which, ws))[i] for i, l in enumerate(ro[0]) if l == "attack"]) > 0 for ro in rows_out]))
                ent = {"attack_raw": agg("attack", "raw"), "honest_raw": agg("honest", "raw"), "human_raw": agg("human", "raw"), "attack_adv_pos_raw": pos("raw")}
                for ws in WSETS:
                    ent[ws] = {"attack": agg("attack", "pen", ws), "honest": agg("honest", "pen", ws), "human": agg("human", "pen", ws), "adv_pos": pos("pen", ws)}
                R[f"{name}@{k}"] = ent
        res[iso] = R
        print(iso, {k: (round(v["attack_raw"], 2), v["attack_adv_pos_raw"], round(v["func0.3_pad1"]["attack"], 2), v["func0.3_pad1"]["adv_pos"], round(v["func1_pad1"]["attack"], 2), v["func1_pad1"]["adv_pos"], round(v["func1_pad1"]["human"], 2), round(v["func1_pad1"]["honest"], 2)) for k, v in R.items() if "@" in k and k.endswith("@4")}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
