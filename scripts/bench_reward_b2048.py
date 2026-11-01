#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU cost of the reward code path for LONG sources (D-45: the b2048 bin). Nothing is trained, no GPU, no QE model (stub scorer).

    CUDA_VISIBLE_DEVICES= python scripts/bench_reward_b2048.py --config configs/rl_rtxpro_phase2c_b2048.yaml [--windows FILE.parquet] [--tokenizer PATH]

Builds batches of `prompts x group_size` samples of a given bin: the English source is a real b2048 window (--windows; else a synthetic text of the same size),
every completion is a pseudo translation (word-for-word substitution + 8 % random word swaps per sample, so the group agrees like real siblings), 35 % of the
completions have a different number of sentences than the source (the Gale-Church alignment path, 35 % in the b512 stage). The reward is the production
`TranslationReward` with the config's `reward` block (GlotLID off, QE = FakeScorer). Reports seconds per step for the bin's prompts-per-step and the
per-component stats (`consensus/seconds`, ...), next to a b1024-sized batch of the same code so the ratio is readable on a loaded machine.
"""
from __future__ import annotations

import argparse
import random
import sys
import time


def pseudo_words(rng, n=4000):
    return ["".join(rng.choice("abdefghiklmnoprstuwyáàéèọẹṣ") for _ in range(rng.randint(2, 9))) for _ in range(n)]


def translate(sentences, vocab, rng, swap=0.08, mismatch=False):
    out = []
    for s in sentences:
        ws = []
        for w in s.split():
            if rng.random() < swap:
                ws.append(rng.choice(vocab))
            else:
                ws.append(vocab[hash(w.lower().strip(".,;:!?\"'()")) % len(vocab)] + (w[-1] if w[-1] in ".,;:!?" else ""))
        t = " ".join(ws)
        out.append(t[:1].upper() + t[1:])                       # a capitalised next sentence: a full stop before a lower-case word is not a boundary (segments.py)
    if mismatch and len(out) > 3:
        k = rng.choice([1, 2, 3])
        out = out[:-k] if rng.random() < 0.5 else out + out[:k]
    return " ".join(out)


def synthetic_source(rng, vocab, n_sentences):
    return " ".join(" ".join(rng.choice(vocab) for _ in range(rng.randint(8, 26))).capitalize() + "." for _ in range(n_sentences))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--windows", default=None)
    ap.add_argument("--tokenizer", default=None, help="Gemma tokenizer dir (default: the byte-level FakeGemmaTokenizer)")
    ap.add_argument("--sentences", type=int, default=48, help="sentences per synthetic b2048 source")
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args(argv)

    from gemma_mt import format as fmt
    from gemma_mt.fake_tokenizer import FakeGemmaTokenizer
    from gemma_mt.rl.comet import FakeScorer
    from gemma_mt.rl.reward import RewardConfig, Sample, TranslationReward
    from gemma_mt.rl.segments import split_sentences
    from gemma_mt.utils import dataclass_from_dict, load_yaml

    cfg = load_yaml(a.config)
    rcfg = dataclass_from_dict(RewardConfig, cfg["reward"])
    rcfg.lid_enabled = False
    if a.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        stop = fmt.resolve_stop_tokens(tok, None).ids
    else:
        tok = FakeGemmaTokenizer()
        stop = [tok.convert_tokens_to_ids("<end_of_turn>")]
    rw = TranslationReward(rcfg, FakeScorer(), tok, stop, lid=None)
    rng = random.Random(0)
    vocab = pseudo_words(rng)
    G = cfg.get("group_size", 16)
    langs = ["yor", "kbp", "efi", "hau", "amh", "nus"]

    def batch(n_prompts, n_sent, sources=None):
        samples = []
        for p in range(n_prompts):
            iso = langs[p % len(langs)]
            src = sources[p % len(sources)] if sources else synthetic_source(rng, vocab, n_sent)
            sents = split_sentences(src)
            for g in range(G):
                text = translate(sents, vocab, rng, mismatch=rng.random() < 0.35)
                ids = tok.encode(f"{iso} {text}", add_special_tokens=False) + stop[:1]
                samples.append(Sample(ids, iso, src))
        return samples

    sources = None
    if a.windows:
        import pyarrow.parquet as pq

        sources = pq.read_table(a.windows, columns=["english"]).column("english").to_pylist()[:200]
        print(f"{len(sources)} real windows from {a.windows}; sentences per window: {sorted(len(split_sentences(s)) for s in sources)[::25]}", flush=True)
    for name, n_prompts, n_sent, src in (("b1024-like", cfg["prompts_per_step_by_bin"]["b1024"], 24, None),
                                         ("b2048", cfg["prompts_per_step_by_bin"]["b2048"], a.sentences, sources),
                                         ("b2048 worst (110 sentences)", cfg["prompts_per_step_by_bin"]["b2048"], 110, None)):
        for _ in range(a.repeat):
            samples = batch(n_prompts, n_sent, src)
            toks = sum(len(s.token_ids) for s in samples) / len(samples)
            t = time.time()
            out = rw(samples, group_size=G)
            dt = time.time() - t
            st = {k: round(v, 2) for k, v in out.stats.items() if k.endswith("/seconds") or k in ("chunk/segments_per_sample", "chunk/count_mismatch_rate", "consensus/groups")}
            fails = {s: out.status.count(s) for s in set(out.status)}
            print(f"{name}: {n_prompts} prompts x {G} = {len(samples)} samples, mean completion {toks:.0f} tokens: reward() {dt:.1f}s; {st}; status {fails}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
