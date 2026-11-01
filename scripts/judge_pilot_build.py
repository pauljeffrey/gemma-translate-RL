"""Build the J1 judge-pilot sample and prompts (no API calls).

SSA-MTE (McGill-NLP/SSA-MTE, CC-BY-SA-4.0) eng-amh/yor/kin/hau test segments (50 each, quantile-stratified on the human z-score,
seed 2026) with 5-shot demonstrations from the SAME pair's TRAIN split (paper protocol); AfriMTE-WMT2024 (masakhane, CC-BY-4.0)
eng-swh/ibo/som/xho (25 each, same stratification). AfriMTE has no train split and no raw DA (only z-scores), so its five
demonstrations come from the same pair's remaining AfriMTE rows (sources not in the sample), z min-max scaled to 0-100 (deviation).

    python judge_pilot_build.py --pool-dir <dir with ssamte_rows.jsonl, afrimte_rows.jsonl, ssamte_scores/, afrimte_scores/> \
        --train-dir <dir with ssamte/train/*.json> --scratch <dir for prompts with text> --out <repo csv dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from judge_pilot_common import SEED, build_mte_prompt, load_jsonl, pick_five_demos

SSA_PAIRS = {"eng-amh": "Amharic", "eng-yor": "Yoruba", "eng-kin": "Kinyarwanda", "eng-hau": "Hausa"}
AFRI_PAIRS = ["eng-swh", "eng-ibo", "eng-som", "eng-xho"]
COMET = ["ssa_mtl_ref", "africomet_stl11_ref", "africomet_mtl_ref", "ssa_mtl_qe", "ssa_qe"]
N_PROBE = 12  # cheap first batch used to measure thinking-token usage before sizing the main batch


def stratified(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n rows, n/5 from each of 5 quantile strata of the human z-score."""
    order = sorted(rows, key=lambda r: r["z"])
    k = n // 5
    out = []
    for s in range(5):
        lo, hi = round(s * len(order) / 5), round((s + 1) * len(order) / 5)
        out += rng.sample(order[lo:hi], k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pool, scratch, out = Path(a.pool_dir), Path(a.scratch), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    ssa = load_jsonl(pool / "ssamte_rows.jsonl")
    afri = load_jsonl(pool / "afrimte_rows.jsonl")
    for name, rows, sub in (("ssa", ssa, "ssamte_scores"), ("afri", afri, "afrimte_scores")):
        for c in COMET:
            v = json.load(open(pool / sub / f"{c}.json"))
            assert len(v) == len(rows), (name, c, len(v), len(rows))
            for r, x in zip(rows, v):
                r[c] = x
        for g, r in enumerate(rows):
            r["g"] = g  # global row index, aligned with the earlier agent's score files

    items, demos_log = [], {}
    # ---- SSA-MTE
    for lp, lang in SSA_PAIRS.items():
        rng = random.Random(f"{SEED}-{lp}")
        train = load_jsonl(Path(a.train_dir) / "train" / f"{lang}.json")
        for k, t in enumerate(train):
            t["row"], t["raw"] = k, float(t["score"])
        lo, hi = min(t["raw"] for t in train), max(t["raw"] for t in train)
        d = pick_five_demos(train, "raw", lo, hi, rng)
        demos = [{"src": x["src_sent"], "mt": x["tgt_sent"], "ref": x["ref"], "score": round(x["raw"])} for x in d]
        demos_log[lp] = {"source": f"SSA-MTE train/{lang}.json", "train_rows": [x["row"] for x in d], "scores": [x["score"] for x in demos], "range": [lo, hi]}
        sample = stratified([r for r in ssa if r["lp"] == lp], 50, rng)
        for r in sample:
            items.append(dict(set="ssamte", lp=lp, g=r["g"], i=r["i"], src=r["src"], mt=r["mt"], ref=r["ref"], z=r["z"], da=r["da"], demos=demos, **{c: r[c] for c in COMET}))
    # ---- AfriMTE
    for lp in AFRI_PAIRS:
        rng = random.Random(f"{SEED}-{lp}")
        rows = [r for r in afri if r["lp"] == lp]
        sample = stratified(rows, 25, rng)
        used_src = {r["src"] for r in sample}
        rest = [dict(r) for r in rows if r["src"] not in used_src]
        z = np.array([r["z"] for r in rest])
        lo, hi = float(z.min()), float(z.max())
        for r in rest:
            r["scaled"] = 100.0 * (r["z"] - lo) / (hi - lo)
        d = pick_five_demos(rest, "scaled", 0.0, 100.0, rng)
        demos = [{"src": x["src"], "mt": x["mt"], "ref": x["ref"], "score": round(x["scaled"])} for x in d]
        demos_log[lp] = {"source": "AfriMTE-WMT2024 same pair, sources outside the sample, z min-max to 0-100", "train_rows": [x["g"] for x in d], "scores": [x["score"] for x in demos], "range": [lo, hi]}
        for r in sample:
            items.append(dict(set="afrimte", lp=lp, g=r["g"], i=r["i"], src=r["src"], mt=r["mt"], ref=r["ref"], z=r["z"], da=r["da"], demos=demos, **{c: r[c] for c in COMET}))

    # ---- probe split: 2 per SSA pair + 1 per AfriMTE pair (same for both judges); the rest is the main batch
    prng = random.Random(f"{SEED}-probe")
    probe = set()
    for lp, k in [(l, 2) for l in SSA_PAIRS] + [(l, 1) for l in AFRI_PAIRS]:
        idx = [n for n, it in enumerate(items) if it["lp"] == lp]
        probe |= set(prng.sample(idx, k))
    assert len(probe) == N_PROBE
    for n, it in enumerate(items):
        it["id"] = f"{it['set']}:{it['lp']}:{it['g']}"
        it["batch"] = "probe" if n in probe else "main"
        it["prompt"] = build_mte_prompt(it["demos"], it["src"], it["mt"], it["ref"])
    assert len({it["id"] for it in items}) == len(items) == 300

    with open(scratch / "items.jsonl", "w", encoding="utf-8") as f:  # text stays in the scratchpad
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(out / "sample_ids.csv", "w", newline="") as f:  # ids and scores only, no text
        w = csv.writer(f)
        w.writerow(["id", "set", "lp", "row_in_pool", "batch", "human_z", "human_da_raw"] + COMET)
        for it in items:
            w.writerow([it["id"], it["set"], it["lp"], it["g"], it["batch"], it["z"], it["da"]] + [it[c] for c in COMET])
    json.dump(demos_log, open(out / "demo_ids.json", "w"), indent=1)
    (out / "prompt_example_pair_eng-yor.txt").write_text(  # one full example prompt; the source data is CC-BY-SA, shown for format only
        next(it for it in items if it["lp"] == "eng-yor")["prompt"], encoding="utf-8")
    print("items", len(items), "probe", len(probe))
    for lp, d in demos_log.items():
        print(lp, d["scores"], d["source"][:40])


if __name__ == "__main__":
    main()
