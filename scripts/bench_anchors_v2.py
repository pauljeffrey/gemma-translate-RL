#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""D-61: seconds per reward call (CPU, stub scorer) with the anchors off / v1 / v1 + v2, on a realistic training batch: 84 prompts x 16 samples, human documents of 100-450 words,
30 languages, consensus on (samples are the human reference with 5 % of the words dropped at random, so groups are not flat). The v2 time is the NET cost of the new detectors."""
import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemma_mt.fake_tokenizer import FakeGemmaTokenizer, SPECIALS
from gemma_mt.rl.comet import FakeScorer
from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
from gemma_mt.rl.reward_shaping import ContentShaper

ap = argparse.ArgumentParser()
ap.add_argument("--human", required=True)
ap.add_argument("--cal-v2", required=True)
ap.add_argument("--shaping", required=True)
ap.add_argument("--lexicon-dir", required=True)
ap.add_argument("--bigrams", default=None)
ap.add_argument("--prompts", type=int, default=84)
ap.add_argument("--g", type=int, default=16)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--min-words", type=int, default=100)
ap.add_argument("--max-words", type=int, default=450)
a = ap.parse_args()
TOK = FakeGemmaTokenizer()
EOT = SPECIALS["<end_of_turn>"]
rng = random.Random(0)
docs = {}
for l in open(a.human):
    r = json.loads(l)
    if r["set"] == "smol_doc" and a.min_words <= len(r["src"].split()) <= a.max_words:
        docs.setdefault(r["iso"], []).append(r)
langs = sorted(docs)[:30]
cal = json.load(open(a.cal_v2))
cons = {i: {"mode": "consensus"} for i in langs[::2]}
cons.update({i: {"mode": "blend", "beta": 0.5} for i in langs[1::2]})
prompts = []
while len(prompts) < a.prompts:
    iso = langs[len(prompts) % len(langs)]
    prompts.append(rng.choice(docs[iso]))
batch = []
for p in prompts:
    w = p["ref"].split()
    for _ in range(a.g):
        t = " ".join(x for x in w if rng.random() > 0.05)
        batch.append(Sample(TOK.encode(p["iso"] + " " + t, add_special_tokens=False) + [EOT], p["iso"], p["src"]))
print(f"[bench] {len(batch)} samples, {len(langs)} languages, mean source words {sum(len(p['src'].split()) for p in prompts) / len(prompts):.0f}")


def mk(**kw):
    sh = ContentShaper.from_file(a.shaping, reliability_from="same_source", mode="calibrated", preset="repo", neg_scale=2.0)
    sh.content = dict(cons)
    base = dict(consensus=True, chunking="sentences", omission_detector=True, length_penalty=1.0, word_ratio_penalty=0.5, lexical_penalty=1.0, diacritic_penalty=1.0)
    base.update(kw)
    return TranslationReward(RewardConfig(**base), FakeScorer(), TOK, [EOT], shaper=sh)


v1 = dict(anchors_enabled=True, anchors_calibration_path=a.cal_v2, anchors_entity_weight=0.5, anchors_repeat_weight=1.0, anchors_boiler_weight=1.0, anchors_lexicon_weight=1.0,
          anchors_lexicon_per_language=True, anchors_lexicon_dir=a.lexicon_dir)
v2 = dict(v1, anchors_v2_enabled=True, anchors_entity_mode="soft", anchors_copy_scale=2.0, anchors_func_weight=0.3, anchors_pad_weight=1.0, anchors_bigram_path=a.bigrams)
res = {}
for name, kw in (("anchors off", {}), ("anchors v1", v1), ("anchors v1 + v2", v2)):
    rw = mk(**kw)
    rw(batch, group_size=a.g)                       # warm-up (caches, imports)
    ts = []
    for _ in range(a.reps):
        t0 = time.time()
        out = rw(batch, group_size=a.g)
        ts.append(time.time() - t0)
    res[name] = (statistics.mean(ts), min(ts), max(ts))
    print(f"[bench] {name}: {statistics.mean(ts):.1f} s (min {min(ts):.1f}, max {max(ts):.1f}); anchors applied {out.stats.get('anchors/applied_rate', 0):.2f}, penalty_mean {out.stats.get('anchors/penalty_mean', 0):.3f}", flush=True)
print(f"[bench] v2 net cost {res['anchors v1 + v2'][0] - res['anchors v1'][0]:+.1f} s; v1 net cost {res['anchors v1'][0] - res['anchors off'][0]:+.1f} s")
