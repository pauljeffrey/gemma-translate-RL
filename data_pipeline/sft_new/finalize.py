"""Turn the per-step shards of a build into DEDUPLICATED per-dataset shards (one folder per build step) and write the inventory.

    python -m sft_new.finalize --out /workspace/sft_build/out --final /workspace/sft_build/final [--shard-rows 100000] [--seed 0]

  1. steps = every finished step (stats/<tag>.json) that is not HELD / EXCLUDED (ext_policy); steps are visited in combine.plan() order
  2. a row is kept when its content hash `id` has not been seen in an earlier step (first occurrence wins, so the sources we vouch for come first);
     every row therefore lives in exactly ONE dataset folder
  3. inside a step the kept rows are shuffled (hash buckets, then a shuffle per bucket), so "the first K rows" of a dataset is a random sample
  4. final/<tag>/part-XXXXX.parquet, final/datasets.json (tag -> converter, rows, licence flag) and final/inventory.json:
     rows per source language x task, per target language x task, per source dataset, per language pair; the held steps with reasons

The tokenizer stage (sft_new.tokenize_bins) turns every final/<tag>/ into a .bin + row index; the mixer (sft_new.mix_sft) builds the training mix from them.
Nothing is pushed anywhere.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sft_new.textutil import nfc_table  # noqa: E402


def _bucket_count(rows: int, shard_rows: int) -> int:
    return max(1, -(-rows // (shard_rows * 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--shard-rows", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--add", action="store_true",
                    help="incremental: keep the folders already in --final, process only steps that are new or were rebuilt after the last finalize, "
                         "deduplicate them against every kept row, and refresh datasets.json / inventory.json (stale folders are deleted first)")
    ap.add_argument("--exclude-step", action="append", default=[], help="converter name to leave out (repeatable), on top of ext_policy")
    ap.add_argument("--drop-list", action="append", type=Path, default=[], help="drop_list.json / drop_list_extra.json of the QE audits (repeatable): their (source dataset | pair) groups are removed")
    a = ap.parse_args()
    from sft_new import combine as K
    from sft_new.ext_policy import is_held

    stats = {}
    for f in sorted((a.out / "stats").glob("*.json")):
        try:
            j = json.loads(f.read_text())
        except ValueError:
            continue
        if j.get("kept", 0) > 0 and (a.out / "data" / f.stem).exists():
            stats[f.stem] = j
    order = {n: i for i, n in enumerate(K.plan(False))}
    steps, held = [], {}
    for tag, j in stats.items():
        name = j["converter"]
        why = is_held(name) or ("excluded on the command line" if name in a.exclude_step else None)
        if why:
            held[name] = {"rows": j["kept"], "reason": why}
            continue
        steps.append((order.get(name, 10 ** 6), name, tag))
    steps.sort()
    existing: dict = {}
    stale: list = []
    if a.add:
        dj = a.final / "datasets.json"
        existing = json.loads(dj.read_text()) if dj.exists() else {}
        t_ref = dj.stat().st_mtime if dj.exists() else 0
        todo = []
        for item in steps:
            tag = item[2]
            if tag in existing and (a.out / "stats" / f"{tag}.json").stat().st_mtime <= t_ref:
                continue
            if tag in existing:
                stale.append(tag)
            todo.append(item)
        for tag in stale:                                                      # rebuilt after the last finalize: drop the old folder, it is re-added below
            shutil.rmtree(a.final / tag, ignore_errors=True)
            existing.pop(tag, None)
        for tag in [t for t in existing if t not in {x[2] for x in steps}]:     # became held / excluded since
            shutil.rmtree(a.final / tag, ignore_errors=True)
            existing.pop(tag, None)
        steps = todo
        print(f"incremental: {len(existing)} folders kept, {len(steps)} to add ({len(stale)} rebuilt)", file=sys.stderr, flush=True)
    if a.max_steps:
        steps = steps[: a.max_steps]
    if not a.add:
        shutil.rmtree(a.final, ignore_errors=True)
    a.final.mkdir(parents=True, exist_ok=True)
    work = a.final.with_name(a.final.name + "_work")
    shutil.rmtree(work, ignore_errors=True)
    seen: set = set()
    drop_groups = set()
    for p in a.drop_list:
        if p.exists():
            drop_groups |= {d["group"] for d in json.loads(p.read_text())}
    dropped_qe = collections.Counter()
    dropped_notools = collections.Counter()
    by_src, by_tgt = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    by_ds, by_pair = collections.Counter(), collections.Counter()
    datasets: dict = {}
    total_in = total_kept = 0
    nfc_fixed = 0                                                                 # string cells changed by the NFC pass
    rng = random.Random(a.seed)
    if a.add:
        for tag, meta in existing.items():
            datasets[tag] = meta
            for shard in sorted((a.final / tag).glob("part-*.parquet")):
                t = pq.read_table(shard, columns=["id", "task", "source_lang", "targ_lang", "source_dataset"])
                seen.update(t.column("id").to_pylist())
                total_kept += t.num_rows
                for x, s_, g, d in zip(*(t.column(c).to_pylist() for c in ("task", "source_lang", "targ_lang", "source_dataset"))):
                    by_src[s_][x] += 1
                    by_tgt[g][x] += 1
                    by_ds[d] += 1
                    by_pair[f"{s_}>{g}"] += 1
        total_in = total_kept
        print(f"existing: {total_kept:,} rows, {len(seen):,} ids", file=sys.stderr, flush=True)
    for _, name, tag in steps:
        shards = sorted((a.out / "data" / tag).glob("part-*.parquet"))
        n_est = stats[tag]["kept"]
        nb = _bucket_count(n_est, a.shard_rows)
        bdir = work / tag
        bdir.mkdir(parents=True)
        kept = 0
        wrote = collections.Counter()
        for shard in shards:
            t = pq.read_table(shard)
            t, n_nfc = nfc_table(t)                                              # one canonical Unicode form in every bin (see textutil.nfc_table)
            nfc_fixed += n_nfc
            ids = t.column("id").to_pylist()
            total_in += len(ids)
            keep = []
            if drop_groups:
                tk, sd, sl, tl = (t.column(c).to_pylist() for c in ("task", "source_dataset", "source_lang", "targ_lang"))
            msgs, tools = t.column("messages").to_pylist(), t.column("tools").to_pylist()
            for i, h in enumerate(ids):
                if not tools[i] and '"tool_calls"' in msgs[i]:                 # a tool-calling row must list the functions it calls (TOOL_CALL_FORMAT.md section 3)
                    dropped_notools[name] += 1
                    continue
                if drop_groups and tk[i] == "translation" and f"{sd[i]}|{sl[i]}>{tl[i]}" in drop_groups:
                    dropped_qe[f"{sd[i]}|{sl[i]}>{tl[i]}"] += 1
                    continue
                if h not in seen:
                    seen.add(h)
                    keep.append(i)
            if not keep:
                continue
            t = t.take(pa.array(keep))
            kept += t.num_rows
            for x, s, g, d in zip(*(t.column(c).to_pylist() for c in ("task", "source_lang", "targ_lang", "source_dataset"))):
                by_src[s][x] += 1
                by_tgt[g][x] += 1
                by_ds[d] += 1
                by_pair[f"{s}>{g}"] += 1
            if nb == 1:
                wrote[0] += 1
                pq.write_table(t, bdir / f"b000-{wrote[0]:05d}.parquet", compression="zstd")
                continue
            buckets: dict = collections.defaultdict(list)
            for i, h in enumerate(t.column("id").to_pylist()):
                buckets[int(h, 16) % nb].append(i)
            for b, idx in buckets.items():
                wrote[b] += 1
                pq.write_table(t.take(pa.array(idx)), bdir / f"b{b:03d}-{wrote[b]:05d}.parquet", compression="zstd")
        outdir = a.final / tag
        outdir.mkdir()
        n_out = rows = 0
        order_b = list(range(nb))
        rng.shuffle(order_b)
        pend = []
        def flush(final=False):
            nonlocal pend, n_out, rows
            if not pend:
                return
            t = pa.concat_tables(pend)
            perm = list(range(t.num_rows))
            rng.shuffle(perm)
            t = t.take(pa.array(perm))
            lo = 0
            while t.num_rows - lo >= a.shard_rows or (final and lo < t.num_rows):
                pq.write_table(t.slice(lo, a.shard_rows), outdir / f"part-{n_out:05d}.parquet", compression="zstd")
                n_out += 1
                rows += min(a.shard_rows, t.num_rows - lo)
                lo += a.shard_rows
            pend = [t.slice(lo)] if lo < t.num_rows else []
        for b in order_b:
            for p in sorted(bdir.glob(f"b{b:03d}-*.parquet")):
                pend.append(pq.read_table(p))
            if sum(x.num_rows for x in pend) >= a.shard_rows * 2:
                flush()
        flush(final=True)
        shutil.rmtree(bdir, ignore_errors=True)
        datasets[tag] = {"converter": name, "rows": rows, "rows_before_dedupe": n_est, "shards": n_out}
        total_kept += rows
        print(f"{name}: {rows:,} of {n_est:,}  (running total {total_kept:,})", file=sys.stderr, flush=True)
    shutil.rmtree(work, ignore_errors=True)
    (a.final / "datasets.json").write_text(json.dumps(datasets, indent=1))
    print(f"NFC pass: {nfc_fixed:,} string cells normalised", file=sys.stderr)
    inv = {"nfc_cells_fixed": nfc_fixed, "rows_in": total_in, "rows_final": total_kept, "duplicates_removed": total_in - total_kept, "datasets": len(datasets),
           "by_source_language_task": {k: dict(v.most_common()) for k, v in sorted(by_src.items(), key=lambda kv: -sum(kv[1].values()))},
           "by_target_language_task": {k: dict(v.most_common()) for k, v in sorted(by_tgt.items(), key=lambda kv: -sum(kv[1].values()))},
           "by_source_dataset": dict(by_ds.most_common()), "by_pair": dict(by_pair.most_common(500)),
           "held_out_of_the_mix": held, "dropped_by_qe_audit": dict(dropped_qe.most_common()),
           "dropped_tool_rows_without_tools": dict(dropped_notools)}
    (a.final / "inventory.json").write_text(json.dumps(inv, indent=1, ensure_ascii=False))
    print(json.dumps({k: inv[k] for k in ("rows_in", "rows_final", "duplicates_removed", "datasets")}), file=sys.stderr)


if __name__ == "__main__":
    main()
