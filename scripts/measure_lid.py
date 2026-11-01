#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure GlotLID top-1 recall on HUMAN text of every RL language and decide the per-language guard (D-37). CPU only.

    python scripts/measure_lid.py --dev flores_dev_all_v2.jsonl --dev flores_dev_extra.jsonl --sft-targets sft_targets.jsonl --out lid_measure.json

Human text per language, in this order of preference: FLORES+ DEV references (997 sentences; `--dev`, rows with iso / reference) and, for languages without a
dev set, a random sample of the SFT training targets (`--sft-targets`: rows iso / text; real corpora, 30-500 characters, up to 1,200 per language;
their recall is a proxy: the domain is not the RL prompts' and some rows are noisy, which can only LOWER the measured recall). The dev sets are never used
for anything but this measurement and the calibration.

Per language the script records
  recall_strict   top-1 label is one of the language's own GlotLID labels (languages.py `glotlid`)
  confusions      the most frequent other top-1 labels
  labels          the accepted set: own labels plus the SIBLING labels (rl/lid.py SIBLING_GROUPS: zul/xho/ssw/nbl/nde, sot/tsn/nso, kin/run, swh/swc, bam/dyu ...)
                  that absorb >= `--min-confusion` (3%) of the language's human sentences. Nothing outside a sibling group is ever added (English never becomes acceptable)
  recall          recall with that set
  enabled         recall >= `--target` on >= `--min-n` sentences (otherwise the guard is OFF for the language: reported, not applied)
  min_chars       smallest prefix length in (15, 30, 60, 120) at which the recall on text prefixes of that length is still >= `--target`
  short_recall    recall on prefixes of those lengths (first words up to the length), same label set
  cross_accept    fraction of OTHER languages' human sentences whose top-1 label is in this language's accepted set (the price of the sibling labels)
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import unicodedata

from gemma_mt.languages import ALL_ISOS, LANGUAGES, SUPPORTED_RL
from gemma_mt.rl.lid import GlotLIDClassifier, siblings_of

SHORT_LENGTHS = (15, 30, 60, 120)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prefix(text: str, n: int) -> str:
    """The first words of `text` up to the first space at or after position n (so the string has n .. n+word characters)."""
    if len(text) <= n:
        return text
    cut = text.find(" ", n)
    return text if cut < 0 else text[:cut]


def choose_labels(own, top1, min_conf, sibling_pool):
    """own labels + the SIBLING labels (rl/lid.py SIBLING_GROUPS) that absorb >= `min_conf` of the human sentences. Never anything else."""
    n = max(len(top1), 1)
    cnt = collections.Counter(top1)
    cur = set(own)
    extras = [lab for lab, c in cnt.most_common() if lab in sibling_pool and lab not in cur and c / n >= min_conf]
    cur |= set(extras)
    return sorted(cur), extras, sum(c for l, c in cnt.items() if l in cur) / n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="append", default=[], help="jsonl with iso, reference (FLORES+ DEV only)")
    ap.add_argument("--sft-targets", default=None, help="jsonl with iso, text (SFT training targets; used where no dev set exists)")
    ap.add_argument("--model", default=None, help="local model_v3.bin (default: cis-lmu/glotlid from the Hub cache)")
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--min-confusion", type=float, default=0.03)
    ap.add_argument("--min-n", type=int, default=300)
    ap.add_argument("--max-per-language", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    dev, sft = collections.defaultdict(list), collections.defaultdict(list)
    for p in a.dev:
        for r in read_jsonl(p):
            if r.get("reference") and r.get("benchmark", "").find("mafand") < 0:
                dev[r["iso"]].append(unicodedata.normalize("NFC", r["reference"]))
    if a.sft_targets:
        for r in read_jsonl(a.sft_targets):
            sft[r["iso"]].append(unicodedata.normalize("NFC", r["text"]))

    lid = GlotLIDClassifier(path=a.model)
    t0 = time.time()
    model_labels = set(lid.labels())
    print(f"GlotLID loaded: {len(model_labels)} labels ({time.time() - t0:.0f}s)", flush=True)

    langs = [l for l in (a.languages or SUPPORTED_RL) if l in LANGUAGES]
    texts, source, top1, top3 = {}, {}, {}, {}
    for iso in langs:
        src = "flores_dev" if len(dev.get(iso, [])) >= a.min_n else ("sft_targets" if sft.get(iso) else None)
        if src is None:
            if dev.get(iso):
                src = "flores_dev"
            else:
                continue
        ts = (dev[iso] if src == "flores_dev" else sft[iso])[: a.max_per_language]
        texts[iso], source[iso] = ts, src
        res = lid.top(ts, 3)
        top1[iso] = [r[0][0] for r in res]
        top3[iso] = [[l for l, _ in r] for r in res]
    print(f"top-1 computed for {len(texts)} languages ({time.time() - t0:.0f}s)", flush=True)

    out = {"_meta": {"model": a.model or "cis-lmu/glotlid model_v3.bin", "target_recall": a.target, "min_confusion": a.min_confusion,
                     "min_n": a.min_n, "short_lengths": list(SHORT_LENGTHS), "n_model_labels": len(model_labels)}}
    sets = {}
    for iso in texts:
        own = [l for l in LANGUAGES[iso].glotlid]
        has = [l for l in own if l in model_labels]
        n = len(texts[iso])
        strict = sum(l in set(own) for l in top1[iso]) / max(n, 1)
        labels, extras, rec = choose_labels(has, top1[iso], a.min_confusion, set(siblings_of(own)))
        conf = collections.Counter(l for l in top1[iso] if l not in set(has)).most_common(6)
        sets[iso] = set(labels)
        e = {"source": source[iso], "n": n, "own_labels": own, "own_labels_in_model": has, "recall_strict": strict, "labels": labels, "extra_labels": extras,
             "recall": rec, "top_confusions": [[l, c / n] for l, c in conf], "enabled": bool(labels) and rec >= a.target and n >= a.min_n}
        if source[iso] == "sft_targets" or (iso in sft and source[iso] == "flores_dev"):
            ts2 = sft.get(iso, [])[: a.max_per_language]
            if ts2 and source[iso] == "flores_dev":
                r2 = lid.top(ts2, 1)
                e["recall_sft_targets"] = sum(r[0][0] in sets[iso] for r in r2) / len(ts2)
        # short strings
        sr, minc = {}, None
        for L in SHORT_LENGTHS:
            pre = [prefix(t, L) for t in texts[iso] if len(t) >= L]
            if len(pre) >= 50:
                r = lid.top(pre, 1)
                sr[str(L)] = sum(x[0][0] in sets[iso] for x in r) / len(pre)
                if minc is None and sr[str(L)] >= a.target:
                    minc = L
        e["short_recall"] = sr
        e["min_chars"] = minc if minc is not None else SHORT_LENGTHS[-1]
        out[iso] = e
    # price of the sibling labels: how often another language's human text is accepted
    for iso, s in sets.items():
        others = [(j, l) for j in top1 if j != iso for l in top1[j]]
        acc = collections.Counter(j for j, l in others if l in s)
        tot = collections.Counter(j for j, _ in others)
        out[iso]["cross_accept"] = sum(acc.values()) / max(len(others), 1)
        out[iso]["cross_accept_by"] = {j: acc[j] / tot[j] for j in sorted(acc, key=lambda j: -acc[j] / tot[j])[:5] if acc[j] / tot[j] >= 0.02}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    en = [k for k, v in out.items() if k != "_meta" and v["enabled"]]
    print(f"{len(en)} of {len(out) - 1} languages enabled ({time.time() - t0:.0f}s) -> {a.out}")
    for iso in sorted(out):
        if iso == "_meta":
            continue
        e = out[iso]
        sr = e["short_recall"]
        print(f"{iso} {e['source'][:3]} n={e['n']:4d} strict={e['recall_strict']:.3f} set={e['recall']:.3f} {'ON ' if e['enabled'] else 'off'} min={e['min_chars']:3d} "
              f"short15/30/60/120={'/'.join(f'{sr[k]:.2f}' for k in sorted(sr, key=int))} xacc={e['cross_accept']:.3f} +{','.join(e['extra_labels'])} "
              f"conf={','.join(f'{l}:{f:.3f}' for l, f in e['top_confusions'][:2])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
