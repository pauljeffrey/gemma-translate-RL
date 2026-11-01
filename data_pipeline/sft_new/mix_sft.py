"""Build a training mix from per-dataset bins (sft_new.tokenize_bins) according to a recipe. Copies token ids only; nothing is re-tokenized.

    python -m sft_new.mix_sft --bins /workspace/sft_build/bins --final /workspace/sft_build/final --recipe sft_new/mix_recipes/default.json \
        --out /workspace/sft_build/mix [--dry-run]

  * every dataset (one converter) is assigned to the FIRST recipe group whose `match` regex hits its converter name
  * a group gets `share` x budget_tokens (what a group cannot absorb is passed to the others), split across languages by UniMax: languages are visited from the smallest, each gets min(equal share of what is
    left, its tokens x max_epochs); datasets inside a language are drawn proportionally to their size
  * "language" of a row = the non-English side of a pair (translation), else its source language
  * rows are drawn without replacement per epoch (reshuffled each epoch), then ALL selected rows are shuffled globally
  * validation: up to `validation_rows_per_dataset` rows from every dataset's .val bin
Outputs: sft_training.bin, sft_training.rows.npy, sft_validation.bin, sft_validation.rows.npy, mix_report.json (tokens per group / language / dataset, epochs).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROW_DT = np.dtype([("off", "<i8"), ("len", "<i4"), ("src", "u1"), ("tgt", "u1"), ("task", "u1"), ("flags", "u1")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rec = json.loads(a.recipe.read_text())
    rng = np.random.default_rng(rec.get("seed", 0))
    vocab = json.loads((a.bins / "vocab.json").read_text())
    inv_lang = {v: k for k, v in vocab["lang"].items()}
    ds_meta = json.loads((a.final / "datasets.json").read_text())
    eng = vocab["lang"].get("eng", 254)
    groups = [{**g, "rx": re.compile(g["match"]), "rows": []} for g in rec["groups"]]
    pools = {}                                                                   # tag -> (rows array, group index)
    for tag, meta in ds_meta.items():
        rp = a.bins / f"{tag}.rows.npy"
        if not rp.exists():
            continue
        rows = np.load(rp)
        conv = meta["converter"]
        gi = next(i for i, g in enumerate(groups) if g["rx"].fullmatch(conv))
        pools[tag] = rows
        key_lang = np.where(rows["src"] != eng, rows["src"], rows["tgt"])        # the non-English side of a pair, else the source language
        groups[gi]["rows"].append((tag, rows, key_lang))
    report = {"budget_tokens": rec["budget_tokens"], "groups": {}}
    selection = []                                                               # (tag, row index array)
    # ---- group level: share x budget, water-filled -- what a group cannot absorb (its tokens x max_epochs) goes to the groups that still can
    cap = {}
    prep = {}
    for g in groups:
        by_lang: dict = {}
        for tag, rows, kl in g["rows"]:
            for l in np.unique(kl):
                m = kl == l
                by_lang.setdefault(int(l), []).append((tag, np.flatnonzero(m), int(rows["len"][m].sum())))
        avail = {l: sum(x[2] for x in v) for l, v in by_lang.items()}
        prep[g["name"]] = (by_lang, avail)
        cap[g["name"]] = sum(avail.values()) * g["max_epochs"]
    alloc = {g["name"]: 0 for g in groups}
    pending = {g["name"]: g["share"] * rec["budget_tokens"] for g in groups}
    left_total = rec["budget_tokens"]
    for _ in range(10):
        room = {n: cap[n] - alloc[n] for n in alloc}
        active = [g for g in groups if room[g["name"]] > 1]
        if not active or left_total < 1:
            break
        wsum = sum(g["share"] for g in active)
        used = 0
        for g in active:
            give = min(room[g["name"]], left_total * g["share"] / wsum)
            alloc[g["name"]] += give
            used += give
        left_total -= used
    for g in groups:
        budget = int(alloc[g["name"]])
        by_lang, avail = prep[g["name"]]
        weights = g.get("language_weights", {})
        inv_to_code = {k: v for k, v in vocab["lang"].items()}
        w = {l: float(weights.get(inv_lang.get(l, ""), 1.0)) for l in avail}
        left, take, order = budget, {}, sorted(avail, key=lambda l: avail[l] / w[l])
        wrem = sum(w.values())
        for l in order:                                                         # UniMax with per-language weights
            share = left * w[l] / max(wrem, 1e-9)
            take[l] = int(min(share, avail[l] * g["max_epochs"]))
            left -= take[l]
            wrem -= w[l]
        grep = {"budget": budget, "share_budget": int(g["share"] * rec["budget_tokens"]), "available_tokens": sum(avail.values()), "languages": {}}
        for l, goal in take.items():
            parts = by_lang[l]                                              # [(tag, row indices, tokens)]
            got, epochs = 0, 0
            while got < goal and epochs < g["max_epochs"] + 1:
                epochs += 1
                remaining = goal - got
                tot_all = sum(p[2] for p in parts)
                for tag, idx, tot in parts:
                    want = min(tot, int(round(remaining * tot / max(tot_all, 1))))
                    if want <= 0:
                        continue
                    lens = pools[tag]["len"][idx].astype(np.int64)
                    perm = rng.permutation(len(idx))
                    n = max(1, int(np.searchsorted(np.cumsum(lens[perm]), want, side="right")))
                    selection.append((tag, idx[perm[:n]]))
                    got += int(lens[perm[:n]].sum())
            grep["languages"][inv_lang.get(l, str(l))] = {"goal": goal, "tokens": got, "available": avail[l], "epochs": round(got / max(avail[l], 1), 2)}
        report["groups"][g["name"]] = grep
    total = sum(int(pools[t]["len"][s].sum()) for t, s in selection)
    report["selected_tokens"] = total
    if a.dry_run:
        print(json.dumps(report, indent=1))
        return
    a.out.mkdir(parents=True, exist_ok=True)
    flat = [(t, int(i)) for t, s in selection for i in s]
    perm = rng.permutation(len(flat))
    mm = {}
    rows_out, off = [], 0
    with open(a.out / "sft_training.bin", "wb") as fo:
        for j in perm:
            tag, i = flat[j]
            if tag not in mm:
                mm[tag] = np.memmap(a.bins / f"{tag}.bin", dtype=np.uint16, mode="r")
            r = pools[tag][i]
            fo.write(np.asarray(mm[tag][r["off"]: r["off"] + r["len"]]).tobytes())
            rows_out.append((off, r["len"], r["src"], r["tgt"], r["task"], r["flags"]))
            off += int(r["len"])
    np.save(a.out / "sft_training.rows.npy", np.array(rows_out, dtype=ROW_DT))
    vrows, voff = [], 0
    with open(a.out / "sft_validation.bin", "wb") as fv:
        for tag in pools:
            vp = a.bins / f"{tag}.val.rows.npy"
            if not vp.exists():
                continue
            vr = np.load(vp)[: rec.get("validation_rows_per_dataset", 2000)]
            if not len(vr):
                continue
            m = np.memmap(a.bins / f"{tag}.val.bin", dtype=np.uint16, mode="r")
            for r in vr:
                fv.write(np.asarray(m[r["off"]: r["off"] + r["len"]]).tobytes())
                vrows.append((voff, r["len"], r["src"], r["tgt"], r["task"], r["flags"]))
                voff += int(r["len"])
    np.save(a.out / "sft_validation.rows.npy", np.array(vrows, dtype=ROW_DT))
    report["validation_tokens"] = voff
    (a.out / "mix_report.json").write_text(json.dumps(report, indent=1))
    print(f"mix: {total / 1e9:.2f}B training tokens in {len(rows_out):,} rows, {voff / 1e6:.1f}M validation tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
