#!/usr/bin/env python3
"""Global exact dedupe + final shards + the statistics of the clean SFT rebuild.

Input  out/meta/*.parquet (one line per ok() pair of every unit, kept or dropped, no text), out/kept/*.parquet (decontaminated rows), out/ledger.jsonl.
Dedupe key = xxh64(english.casefold, iso, target.casefold) (gemma_sft_pairs.py key); first occurrence in ledger order wins. The filter decision depends only on the text,
so a duplicate key has the same decision everywhere and "before filter" (unique ok() pairs) and "after filter" are measured on the same unique set.

Output  out/final/pairs-NNNN.parquet   english, target, iso, source_dataset, licence, flipped      (2M rows per shard)
        out/stats/*.csv, out/stats/summary.json
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RULES = ("R1_exact", "R2_contiguous", "R3_8gram")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-rows", type=int, default=2_000_000)
    ap.add_argument("--chars-per-token", default=None, help="json {iso: chars_per_token}; default 3.4 for every language (estimate)")
    ap.add_argument("--min-rows", type=int, default=2000)
    a = ap.parse_args()
    out = a.out
    units = [json.loads(l)["unit"] for l in open(f"{out}/ledger.jsonl") if l.strip()]
    uid = lambda u: u.replace("/", "_").replace(":", "_")
    metas, kept_files = [], []
    for u in units:
        mp_, kp = f"{out}/meta/{uid(u)}.parquet", f"{out}/kept/{uid(u)}.parquet"
        if os.path.exists(mp_):
            metas.append((u, pq.read_table(mp_)))
        if os.path.exists(kp):
            kept_files.append(kp)
    M = pa.concat_tables([t for _, t in metas])
    unit_col = np.concatenate([np.full(t.num_rows, i, np.int32) for i, (_, t) in enumerate(metas)])
    key = M["key"].to_numpy()
    quar = M["quar"].to_numpy().astype(bool)
    drop = M["drop"].to_numpy()
    tlen = M["tlen"].to_numpy().astype(np.int64)
    iso = pc.dictionary_encode(M["iso"]).combine_chunks()
    iso_names = iso.dictionary.to_pylist()
    iso_i = iso.indices.to_numpy()
    src = pc.dictionary_encode(M["src"]).combine_chunks()
    src_names = src.dictionary.to_pylist()
    src_i = src.indices.to_numpy()
    lic = pc.dictionary_encode(M["lic"]).combine_chunks()
    lic_names = lic.dictionary.to_pylist()
    lic_i = lic.indices.to_numpy()
    n = len(key)
    print("meta rows", n, "units", len(metas), flush=True)

    # first occurrence among MAIN rows
    main_idx = np.nonzero(~quar)[0]
    _, first_pos = np.unique(key[main_idx], return_index=True)
    first = np.zeros(n, bool)
    first[main_idx[first_pos]] = True
    hit = (drop & 7) != 0
    uniq = first                                  # unique ok() pairs (before filter)
    kept_u = uniq & ~hit
    # quarantine uniqueness (own key space)
    q_idx = np.nonzero(quar)[0]
    q_first = np.zeros(n, bool)
    if len(q_idx):
        _, qp = np.unique(key[q_idx], return_index=True)
        q_first[q_idx[qp]] = True

    # ---- final shards: kept rows appear in the same order as meta rows with drop==0 and quar==0
    sel_main_kept = (~quar) & (~hit)
    flags = first[sel_main_kept]                 # for each kept row in order: is it the first occurrence
    os.makedirs(f"{out}/final", exist_ok=True)
    for f in glob.glob(f"{out}/final/*.parquet"):
        os.remove(f)
    pos = 0
    shard = 0
    buf = []
    nbuf = 0
    total_final = 0
    cols = ("english", "target", "iso", "source_dataset", "licence", "flipped")
    for kp in kept_files:
        t = pq.read_table(kp, columns=list(cols))
        m = flags[pos:pos + t.num_rows]
        pos += t.num_rows
        t = t.filter(pa.array(m))
        t = t.append_column("decon_flag", pa.array([""] * t.num_rows, pa.string()))     # no weaker near-match review implemented: every hit is dropped, the flag is always empty
        if t.num_rows:
            buf.append(t); nbuf += t.num_rows
        if nbuf >= a.shard_rows:
            pq.write_table(pa.concat_tables(buf), f"{out}/final/pairs-{shard:04d}.parquet", compression="zstd")
            total_final += nbuf; shard += 1; buf = []; nbuf = 0
    if nbuf:
        pq.write_table(pa.concat_tables(buf), f"{out}/final/pairs-{shard:04d}.parquet", compression="zstd")
        total_final += nbuf
    assert pos == int(sel_main_kept.sum()), (pos, int(sel_main_kept.sum()))
    assert total_final == int(kept_u.sum()), (total_final, int(kept_u.sum()))
    print("final unique kept pairs", total_final, flush=True)

    os.makedirs(f"{out}/stats", exist_ok=True)
    cpt = json.load(open(a.chars_per_token)) if a.chars_per_token else {}
    cpt_default = 3.4

    def toks(chars, iso_name):
        return chars / cpt.get(iso_name, cpt_default)

    # ---- per language
    L = collections.defaultdict(lambda: collections.Counter())
    for ii, name in enumerate(iso_names):
        sel = iso_i == ii
        b = sel & uniq
        k = sel & kept_u
        c = L[name]
        c["pairs_before"] = int(b.sum()); c["pairs_after"] = int(k.sum())
        c["tchars_before"] = int(tlen[b].sum()); c["tchars_after"] = int(tlen[k].sum())
        h = b & hit
        c["r1"] = int((h & ((drop & 1) != 0)).sum())
        c["r2"] = int((h & ((drop & 2) != 0)).sum())
        c["r3"] = int((h & ((drop & 4) != 0)).sum())
        c["only_r3"] = int((h & ((drop & 3) == 0)).sum())
        c["hit_en"] = int((h & ((drop & 8) != 0)).sum())
        c["hit_tg"] = int((h & ((drop & 16) != 0)).sum())
        c["hit_both"] = int((h & ((drop & 8) != 0) & ((drop & 16) != 0)).sum())
    with open(f"{out}/stats/per_language.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iso", "pairs_before", "pairs_after", "removed", "removed_pct", "tchars_before", "tchars_after", "est_tokens_before", "est_tokens_after", "removed_R1", "removed_R2",
                    "removed_R3_any", "removed_only_R3_8gram", "hit_english_side", "hit_target_side", "hit_both_sides", "below_min_rows"])
        for name, c in sorted(L.items(), key=lambda kv: -kv[1]["pairs_before"]):
            rem = c["pairs_before"] - c["pairs_after"]
            w.writerow([name, c["pairs_before"], c["pairs_after"], rem, round(100 * rem / max(1, c["pairs_before"]), 2), c["tchars_before"], c["tchars_after"],
                        int(toks(c["tchars_before"], name)), int(toks(c["tchars_after"], name)), c["r1"], c["r2"], c["r3"], c["only_r3"], c["hit_en"], c["hit_tg"], c["hit_both"],
                        int(c["pairs_after"] < a.min_rows)])

    # ---- per source and licence
    S = collections.defaultdict(collections.Counter)
    for si, name in enumerate(src_names):
        sel = src_i == si
        b, k = sel & uniq, sel & kept_u
        base = name.split(":")[0]
        c = S[base]
        c["before"] += int(b.sum()); c["after"] += int(k.sum())
        c["tchars_after"] += int(tlen[k].sum())
        li = np.unique(lic_i[sel]) if sel.any() else []
        for l in li:
            S[base]["lic:" + lic_names[l]] += 1
    with open(f"{out}/stats/per_source.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_repo", "licence_classes", "pairs_before", "pairs_after", "removed", "removed_pct"])
        for name, c in sorted(S.items(), key=lambda kv: -kv[1]["before"]):
            rem = c["before"] - c["after"]
            w.writerow([name, "+".join(k[4:] for k in c if k.startswith("lic:")), c["before"], c["after"], rem, round(100 * rem / max(1, c["before"]), 2)])

    # language x source
    with open(f"{out}/stats/per_language_source.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iso", "source_repo", "pairs_before", "pairs_after"])
        comb = collections.Counter(); comb_k = collections.Counter()
        for ii in np.unique(iso_i):
            sel = iso_i == ii
            for si in np.unique(src_i[sel & uniq]):
                s2 = sel & (src_i == si)
                comb[(iso_names[ii], src_names[si].split(":")[0])] += int((s2 & uniq).sum())
                comb_k[(iso_names[ii], src_names[si].split(":")[0])] += int((s2 & kept_u).sum())
        for (i_, s_), v in sorted(comb.items(), key=lambda kv: -kv[1]):
            w.writerow([i_, s_, v, comb_k[(i_, s_)]])

    # licence tables (main output only; kept pairs, estimated tokens)
    LT = collections.defaultdict(float)
    for ii, name in enumerate(iso_names):
        for li, ln in enumerate(lic_names):
            sel = (iso_i == ii) & (lic_i == li) & kept_u
            if sel.any():
                LT[(name, ln)] = (int(sel.sum()), toks(int(tlen[sel].sum()), name))
    with open(f"{out}/stats/licence_by_language.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iso", "licence_class", "pairs_after", "est_target_tokens"])
        for (name, ln), (p_, t_) in sorted(LT.items()):
            w.writerow([name, ln, p_, int(t_)])
    with open(f"{out}/stats/licence_by_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["licence_class", "pairs_after", "est_target_tokens", "source_repos"])
        for li, ln in enumerate(lic_names):
            sel = (lic_i == li) & kept_u
            repos = sorted({src_names[s].split(":")[0] for s in np.unique(src_i[sel])}) if sel.any() else []
            w.writerow([ln, int(sel.sum()), int(sum(toks(int(tlen[sel & (iso_i == ii)].sum()), nm) for ii, nm in enumerate(iso_names))), "; ".join(repos)])

    # quarantine (separate from the main output)
    qrows = []
    for kp in sorted(glob.glob(f"{out}/quarantine_nc/*.parquet")):
        t = pq.read_table(kp, columns=["iso", "source_dataset", "licence", "reason"])
        c = collections.Counter(zip(t["iso"].to_pylist(), [s.split(":")[0] for s in t["source_dataset"].to_pylist()], t["licence"].to_pylist(), t["reason"].to_pylist()))
        qrows += [(k, v) for k, v in c.items()]
    qag = collections.Counter()
    for (i_, s_, l_, r_), v in qrows:
        qag[(i_, s_, l_, r_)] += v
    with open(f"{out}/stats/quarantine_nc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iso", "source_repo", "licence", "reason", "pairs_kept_in_quarantine_folder"])
        for k, v in sorted(qag.items(), key=lambda kv: -kv[1]):
            w.writerow([*k, v])

    # rule / set tables from the ledger
    RS = collections.Counter(); FR = collections.Counter(); ST = collections.Counter(); DEL = {}
    for l in open(f"{out}/ledger.jsonl"):
        r = json.loads(l)
        for k, v in r["rule_set_side"].items():
            RS[k] += v
        for k, v in r["stats"].items():
            ST[k] += v
        DEL[r["unit"]] = r["stats"]
    with open(f"{out}/stats/removed_by_rule_set_side.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rule", "protected_set", "side", "rows_hit (before global dedupe; a row can hit several sets/rules)", "scope"])
        for k, v in sorted(RS.items(), key=lambda kv: -kv[1]):
            q = k.startswith("Q_")
            r_, s_, sd = (k[2:] if q else k).split("|")
            w.writerow([r_, s_, sd, v, "quarantine_nc" if q else "main"])
    summ = {"units": len(units), "meta_rows": int(n), "unique_ok_pairs_main": int(uniq.sum()), "unique_kept_pairs_main": int(kept_u.sum()),
            "removed_main": int((uniq & hit).sum()), "removed_pct": round(100 * float((uniq & hit).sum()) / max(1, int(uniq.sum())), 2),
            "unique_quarantine_pairs": int(q_first.sum()), "quarantine_removed_by_filter": int((q_first & hit).sum()),
            "duplicates_dropped_main": int((~quar).sum() - uniq.sum()), "stats_totals": dict(ST), "chars_per_token_default": cpt_default,
            "languages": len(iso_names), "languages_below_min_rows": sorted(nm for nm, c in L.items() if c["pairs_after"] < a.min_rows)}
    json.dump(summ, open(f"{out}/stats/summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
