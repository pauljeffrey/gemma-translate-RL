#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does the QE-free consensus signal (D-38) rank a model's own samples the way a human reference does? (D-43)

For N hold-out sources of one language: G completions per source (HF `generate`, temperature 1.0, top_p 1, top_k 0 = the RL sampling), the REAL reward path
(`TranslationReward`, content mode forced to `consensus`, no scorer: any QE call raises) gives every completion its consensus chrF++ (mean chrF++ against the
valid siblings of the group, rl/consensus.py), and sacrebleu chrF++ against the human reference gives the quality. Reported (GRPO only ever compares samples of
the SAME prompt, so the within-prompt number is the one the decision uses):

  within_prompt   mean / median over the groups of the Spearman rho(consensus, reference chrF++) among the group's valid samples (constant groups skipped)
  centered        Spearman of the group-demeaned values over all samples (what group-standardised advantages see)
  overall         Spearman over all valid samples pooled

    CUDA_VISIBLE_DEVICES= python scripts/validate_consensus.py --model DIR --holdout refs/efi_holdout.jsonl --iso efi --n 150 --out val_efi.json
    --greedy-only  : only the greedy translation of each source and its chrF++ against the reference (baseline; not used for any decision)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

from gemma_mt.rl import calibrate_refs as C


class NoScorer:
    def __call__(self, *a, **k):
        raise RuntimeError("QE scorer called: a consensus-mode language must never reach it")

    def token_lengths(self, texts):
        return [len(t.split()) for t in texts]


def analyse(groups):
    """groups: list of dicts {cons: [..], chrf: [..]} (valid samples only)."""
    rhos, cx, cy, ax, ay = [], [], [], [], []
    for g in groups:
        if len(g["cons"]) < 3:
            continue
        r = C.spearman(g["cons"], g["chrf"])
        if r is not None:
            rhos.append(r)
        mc, mr = statistics.mean(g["cons"]), statistics.mean(g["chrf"])
        cx += [c - mc for c in g["cons"]]
        cy += [c - mr for c in g["chrf"]]
        ax += g["cons"]
        ay += g["chrf"]
    se = (statistics.pstdev(rhos) / len(rhos) ** 0.5) if len(rhos) > 1 else None
    return {"groups_used": len(rhos), "within_prompt_mean": statistics.mean(rhos) if rhos else None, "within_prompt_median": statistics.median(rhos) if rhos else None,
            "within_prompt_se": se, "share_positive": (sum(r > 0 for r in rhos) / len(rhos)) if rhos else None,
            "centered": C.spearman(cx, cy), "overall": C.spearman(ax, ay), "samples": len(ax)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--holdout", required=True, help="jsonl of reference pairs (english, reference): the HOLD-OUT half, never the calibration half")
    ap.add_argument("--iso", required=True)
    ap.add_argument("--calibration", default=None, help="calibration file for the reward (lid / length / diacritic blocks); consensus mode is forced for --iso")
    ap.add_argument("--reward-yaml", default=None, help="a rl yaml whose `reward:` block is used (default: library defaults + lid from --lid-model)")
    ap.add_argument("--lid-model", default="/workspace/sft_build/hf/hub/models--cis-lmu--glotlid/snapshots/85cd6716494360367b75f642b5bc78667605d0b4/model_v3.bin")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--min-words", type=int, default=5)
    ap.add_argument("--max-words", type=int, default=45)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--batch-prompts", type=int, default=4)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: this process handles every n-th selected source")
    ap.add_argument("--greedy-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import torch
    from transformers import AutoTokenizer
    from sacrebleu.metrics import CHRF

    from gemma_mt import format as fmt
    from gemma_mt.modeling import load_model
    from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
    from gemma_mt.rl.sampling import generate_completions
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    torch.set_num_threads(a.threads)
    rows = [json.loads(l) for l in open(a.holdout, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if a.min_words <= len(r["english"].split()) <= a.max_words]
    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.n]
    si, sn = map(int, a.shard.split("/"))
    rows = rows[si::sn]

    tok = AutoTokenizer.from_pretrained(a.model)
    model = load_model(a.model, dtype=torch.float32, device="cpu")
    stop = fmt.resolve_stop_tokens(tok, model.generation_config)
    fmt.ensure_generation_config(model.generation_config, stop, pad_id=tok.pad_token_id)
    chrf = CHRF(word_order=2)

    rcfg = dict((load_yaml(a.reward_yaml).get("reward") or {})) if a.reward_yaml else {}
    rcfg.update({"consensus": True, "lid_enabled": True, "lid_model_path": a.lid_model, "content_modes": {a.iso: "consensus"}, "calibration_path": a.calibration})
    rcfg.setdefault("chunking", "sentences")
    reward = TranslationReward(dataclass_from_dict(RewardConfig, rcfg), NoScorer(), tok, stop.ids, pad_id=tok.pad_token_id)
    assert reward.content_spec(a.iso) == ("consensus", 1.0), reward.content_spec(a.iso)


    if a.greedy_only:
        res = []
        t0 = time.time()
        for s in range(0, len(rows), 8):
            b = rows[s: s + 8]
            roll = generate_completions(model, [fmt.build_prompt(r["english"], a.iso, tok) for r in b], group_size=1, max_new_tokens=a.max_new_tokens,
                                        temperature=1.0, top_p=1.0, top_k=0, stop_ids=stop.ids, pad_id=tok.pad_token_id, device="cpu", greedy=True)
            out = reward([Sample(c, a.iso, r["english"], 0) for r, c in zip(b, roll.completions)], group_size=1)
            for i, r in enumerate(b):
                txt = out.texts[i]
                res.append({"english": r["english"], "hyp": txt, "status": out.status[i], "reference": r["reference"],
                            "chrf": chrf.sentence_score(txt, [r["reference"]]).score if txt else 0.0, "english_seen": r.get("english_seen")})
            print(f"greedy {len(res)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
        json.dump({"iso": a.iso, "model": a.model, "n": len(res), "mean_chrf": statistics.mean(x["chrf"] for x in res),
                   "status_counts": {k: sum(1 for x in res if x["status"] == k) for k in sorted({x["status"] for x in res})}, "rows": res},
                  open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("mean chrF++", statistics.mean(x["chrf"] for x in res))
        return 0

    out_rows, groups = [], []
    t0 = time.time()
    for s in range(0, len(rows), a.batch_prompts):
        b = rows[s: s + a.batch_prompts]
        prompts = [fmt.build_prompt(r["english"], a.iso, tok) for r in b]
        torch.manual_seed(a.seed * 100003 + s)
        roll = generate_completions(model, prompts, group_size=a.group, max_new_tokens=a.max_new_tokens, temperature=1.0, top_p=1.0, top_k=0,
                                    stop_ids=stop.ids, pad_id=tok.pad_token_id, device="cpu")
        samples = [Sample(roll.completions[i], a.iso, b[i // a.group]["english"], 0) for i in range(len(roll.completions))]
        out = reward(samples, group_size=a.group)
        for gi, r in enumerate(b):
            g = {"cons": [], "chrf": []}
            for k in range(a.group):
                i = gi * a.group + k
                ok = out.status[i] == "ok" and out.consensus_chrf[i] is not None
                ref_chrf = chrf.sentence_score(out.texts[i], [r["reference"]]).score if out.texts[i] else None
                out_rows.append({"english": r["english"], "reference": r["reference"], "k": k, "status": out.status[i], "text": out.texts[i], "consensus": out.consensus_chrf[i],
                                 "reward": out.rewards[i], "ref_chrf": ref_chrf, "english_seen": r.get("english_seen"), "source": r.get("source")})
                if ok and ref_chrf is not None:
                    g["cons"].append(out.consensus_chrf[i])
                    g["chrf"].append(ref_chrf)
            groups.append(g)
        print(f"{min(s + a.batch_prompts, len(rows))}/{len(rows)} sources  {time.time() - t0:.0f}s", flush=True)
    res = analyse(groups)
    res.update({"iso": a.iso, "model": a.model, "sources": len(rows), "group": a.group, "shard": a.shard, "seconds": time.time() - t0,
                "status_counts": {k: sum(1 for r in out_rows if r["status"] == k) for k in sorted({r["status"] for r in out_rows})}})
    json.dump({"summary": res, "groups": groups, "rows": out_rows}, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
