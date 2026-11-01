#!/usr/bin/env python3
"""W1 (2026-10-26): dev contamination flags for the 100-sentence sample, from the archived scan of 2026-10-18
(research_archive_2026-10-23/box/other/sft_build/bench/contamination_dev.json; scanner gemma_mt.eval.contamination over gemma/windows_sft, 11,442,172 windows).

The scan covers FLORES+ dev for 17 languages only (afr ewe fon hau ibo kin nya orm sna som sot swh tsn wol xho yor zul; key '<iso>:<iso>-<FLORES id>')
plus MAFAND dev (pcm twi). The English-side flag is identical for every language (same sentences); the pair flag exists only for those 17.
Writes pair_flags.json ({'<iso>:flores_dev-<id>': bool}) and contamination_dev_sample.json (counts). The SFT windows are not on disk or on the Hub: no new scan is possible here.
"""
import json, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "reports", "data", "weak_selection_2026-10-26")
d = json.load(open(os.path.join(ROOT, "research_archive_2026-10-23/box/other/sft_build/bench/contamination_dev.json")))
ids = json.load(open(os.path.join(OUT, "sample_ids.json")))["flores_dev_ids"]
fl = [l for l in d["meta"]["by_language"] if l not in ("pcm", "twi") and sum(1 for k in d["pair"] if k.startswith(l + ":" + l + "-")) >= 997]
assert len({d["exact"][f"{l}:{l}-{i}"] for l in fl for i in range(997) if False} | {0}) == 1
flags = {}
for l in fl:
    for i in ids:
        flags[f"{l}:flores_dev-{i}"] = bool(d["pair"][f"{l}:{l}-{i}"])
eng = {i: d["exact"][f"{fl[0]}:{fl[0]}-{i}"] for i in range(997)}
assert all(d["exact"][f"{l}:{l}-{i}"] == eng[i] for l in fl for i in range(997)), "English-side flag differs between languages"
summary = dict(source="contamination_dev.json (scan 2026-10-18, training_text gemma/windows_sft)", flores_dev_sentences=997, english_seen_all_dev=sum(eng.values()),
               sample_n=len(ids), english_seen_in_sample=sum(eng[i] for i in ids), languages_with_pair_flag=fl,
               pair_seen_in_sample={l: sum(flags[f"{l}:flores_dev-{i}"] for i in ids) for l in fl},
               pair_seen_all_997={l: sum(d["pair"][f"{l}:{l}-{i}"] for i in range(997)) for l in fl})
json.dump(flags, open(os.path.join(OUT, "pair_flags.json"), "w"))
json.dump(summary, open(os.path.join(OUT, "contamination_dev_sample.json"), "w"), indent=1)
print(json.dumps(summary)[:900])
