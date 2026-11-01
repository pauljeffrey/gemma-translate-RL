"""Run 2 build (no API calls): FLORES+ devtest, SFT-final (step 27,638) vs RL-final (GRPO step 1,433), eight languages with pilot-validated demonstrations.

    python judge_j2_run2_build.py --e1 <archive box/jobs/e1> --e2 <archive box/jobs/e2> --pilot-items <scratch j1/work/items.jsonl> --scratch <dir> --out <repo out dir>

Sample: per language min(200, pool) FLORES+ (Hausa has only 170 `no_pair` sentences, so n=170 there; all others 200)
FLORES+ devtest sentence ids drawn with random.Random("2026-flores-<lang>") from the ids that (a) have a stored output for BOTH
systems (e1/sys_{sft,rl}.jsonl.gz, 768-token greedy generation of the paper's evaluation), (b) have a non-empty human reference and (c) are `no_pair`
(the reference pair was not in the SFT text; the paper's headline subset, and the subset on which the stored AfriCOMET / SSA-COMET scores exist).
Prompt = Figure 8 (reference-based, score only), five demonstrations of the pilot (demo_ids.json) for that language, human FLORES+ reference.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import random
from pathlib import Path

from judge_pilot_common import SEED, build_mte_prompt

LANGS = "amh hau kin yor swh ibo som xho".split()
LP = {l: f"eng-{l}" for l in LANGS}
N = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", required=True)
    ap.add_argument("--e2", required=True)
    ap.add_argument("--pilot-items", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, scratch = Path(a.out), Path(a.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    demos = {}
    for l in open(a.pilot_items, encoding="utf-8"):
        it = json.loads(l)
        demos.setdefault(it["lp"], it["demos"])
    rows = {}
    for sysn in ("sft", "rl"):
        for l in gzip.open(Path(a.e1) / f"sys_{sysn}.jsonl.gz", "rt", encoding="utf-8"):
            r = json.loads(l)
            if r["b"] == "flores_plus" and r["iso"] in LANGS:
                rows[(sysn, r["iso"], r["id"])] = r
    comet = {s: pickle.load(open(Path(a.e2) / "scores" / f"{s}.comet.pkl", "rb")) for s in ("sft", "rl")}
    ssa = {s: pickle.load(open(Path(a.e2) / "scores" / f"{s}.ssa.pkl", "rb")) for s in ("sft", "rl")}
    items, sample_rows = [], []
    summary = {}
    for lang in LANGS:
        ids = sorted({k[2] for k in rows if k[1] == lang and k[0] == "sft"} & {k[2] for k in rows if k[1] == lang and k[0] == "rl"}, key=lambda x: int(x))
        pool = [i for i in ids if rows[("sft", lang, i)]["ref"].strip() and not rows[("sft", lang, i)]["pair_seen"] and not rows[("rl", lang, i)]["pair_seen"]]
        assert all(rows[("sft", lang, i)]["ref"] == rows[("rl", lang, i)]["ref"] and rows[("sft", lang, i)]["en"] == rows[("rl", lang, i)]["en"] for i in pool)
        pick = sorted(random.Random(f"{SEED}-flores-{lang}").sample(pool, min(N, len(pool))), key=lambda x: int(x))
        summary[lang] = dict(both_systems=len(ids), no_pair_pool=len(pool), sampled=len(pick), clean_in_sample=sum(1 for i in pick if not rows[("sft", lang, i)]["contaminated"]))
        for i in pick:
            sr = {}
            for s in ("sft", "rl"):
                r = rows[(s, lang, i)]
                hyp = r["hyp"].split("</think>")[-1].strip()
                items.append(dict(id=f"flores:{lang}:{s}:{i}", lang=lang, system=s, sid=i,
                                  prompt=build_mte_prompt(demos[LP[lang]], r["en"], hyp, r["ref"])))
                sr[s] = dict(chrf=r["chrf"], afri=comet[s].get(lang, {}).get(i), ssa=ssa[s].get(lang, {}).get(i), empty=r["empty"], fin=r["fin"], loop=r["loop"], lid=r["lid"])
            r = rows[("sft", lang, i)]
            sample_rows.append(dict(lang=lang, sid=i, contaminated=r["contaminated"], pair_seen=r["pair_seen"],
                                    **{f"{s}_{k}": v for s in ("sft", "rl") for k, v in sr[s].items()}))
    assert len(items) == sum(v['sampled'] for v in summary.values()) * 2
    with open(scratch / "items_run2.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with gzip.open(out / "prompts_run2_flores.jsonl.gz", "wt", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(dict(id=it["id"], prompt=it["prompt"]), ensure_ascii=False) + "\n")
    with open(out / "run2_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
        w.writeheader()
        w.writerows(sample_rows)
    json.dump(summary, open(out / "run2_sample_summary.json", "w"), indent=1)
    (out / "prompt_example_run2_flores_hau_rl.txt").write_text(next(it for it in items if it["lang"] == "hau" and it["system"] == "rl")["prompt"], encoding="utf-8")
    print(json.dumps(summary, indent=0))


if __name__ == "__main__":
    main()
