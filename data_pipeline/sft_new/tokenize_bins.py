"""Tokenize every dataset folder of `finalize` into its own .bin + row index (one bin per dataset; language, task and flags are per-row metadata).

    python -m sft_new.tokenize_bins --final /workspace/sft_build/final --bins /workspace/sft_build/bins [--workers 4] [--max-tokens 32768] [--only TAG ...]

For every final/<tag>/ it writes
    bins/<tag>.bin / <tag>.rows.npy         training rows, uint16 ids, whole conversations back to back
    bins/<tag>.val.bin / <tag>.val.rows.npy  1% of the rows (hash of the row id), kept out of every training mix
    bins/<tag>.json                          done marker + token / row counts per language and task
rows.npy is a structured array: off int64, len int32, src u1, tgt u1, task u1, flags u1 (see FLAGS); the code tables are in bins/vocab.json.
Rows are rendered from `messages` (+ `tools`) with the canonical chat template (sft_new/render.py), never from the stored `text`. Each row already starts with
<s> and ends with the turn's </s>, so rows are concatenated with no separator. Rows longer than --max-tokens are dropped and counted.
The tokenizer is read from the Hub (<HF_ORG>/sabiyarn-0.5B) with tokenizers.Tokenizer.from_file; its commit hash is recorded in the manifest.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _k, _v in (("RAYON_NUM_THREADS", "1"), ("TOKENIZERS_PARALLELISM", "false"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1")):
    os.environ.setdefault(_k, _v)

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROW_DT = np.dtype([("off", "<i8"), ("len", "<i4"), ("src", "u1"), ("tgt", "u1"), ("task", "u1"), ("flags", "u1")])
FLAGS = {"tools": 1, "think": 2, "tool_call": 4, "unsure": 8, "multi_turn": 16}
REPO = (os.environ['HF_ORG'] + "/sabiyarn-0.5B")
_TOK = None


def load_tok(path: str):
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(path)
    return _TOK


def work(job):
    tag, final, bins, tok_path, vocab, max_tokens, val_mod, shards, part = job
    from sft_new.render import render_json
    tok = load_tok(tok_path)
    name = tag if part is None else f"{tag}.p{part:02d}"
    done = bins / f"{name}.json"
    if done.exists() or (bins / f"{tag}.json").exists():
        return tag, json.loads(done.read_text())["rows"] if done.exists() else 0, "skipped"
    L, T = vocab["lang"], vocab["task"]
    asst = tok.token_to_id("<|assistant|>")
    files = {k: open(bins / f"{name}{k}.bin", "wb") for k in ("", ".val")}
    idx = {"": [], ".val": []}
    off = {"": 0, ".val": 0}
    stat = {"rows": 0, "val_rows": 0, "tokens": 0, "val_tokens": 0, "dropped_long": 0, "dropped_no_assistant": 0, "by_lang": {}, "by_task": {}}
    for shard in shards:
        t = pq.read_table(shard, columns=["id", "task", "source_lang", "targ_lang", "messages", "tools"]).to_pydict()
        texts = [render_json(m, False, tl or None) for m, tl in zip(t["messages"], t["tools"])]
        encs = tok.encode_batch(texts, add_special_tokens=False)
        for i, e in enumerate(encs):
            ids = e.ids
            if len(ids) > max_tokens:
                stat["dropped_long"] += 1
                continue
            if asst not in ids:
                stat["dropped_no_assistant"] += 1
                continue
            flags = 0
            tx = texts[i]
            flags |= FLAGS["tools"] if t["tools"][i] else 0
            flags |= FLAGS["think"] if "<think>" in tx else 0
            flags |= FLAGS["tool_call"] if "<tool_call>" in tx else 0
            flags |= FLAGS["unsure"] if "unsure, but most likely" in tx else 0
            flags |= FLAGS["multi_turn"] if ids.count(asst) > 1 else 0
            part = ".val" if int(t["id"][i], 16) % val_mod == 0 else ""
            arr = np.asarray(ids, dtype=np.uint16)
            files[part].write(arr.tobytes())
            idx[part].append((off[part], len(ids), L.get(t["source_lang"][i], 255), L.get(t["targ_lang"][i], 255), T.get(t["task"][i], 255), flags))
            off[part] += len(ids)
            k = "val_" if part else ""
            stat[k + "rows"] += 1
            stat[k + "tokens"] += len(ids)
            if not part:
                lang = t["source_lang"][i] if t["source_lang"][i] != "eng" else t["targ_lang"][i]
                stat["by_lang"][lang] = stat["by_lang"].get(lang, 0) + len(ids)
                stat["by_task"][t["task"][i]] = stat["by_task"].get(t["task"][i], 0) + len(ids)
    for f in files.values():
        f.close()
    for k, rows in idx.items():
        np.save(bins / f"{name}{k}.rows.npy", np.array(rows, dtype=ROW_DT))
    done.write_text(json.dumps(stat))
    return tag, stat["rows"], "ok"


def merge_parts(tag: str, bins: Path, n_parts: int):
    """Concatenate <tag>.pNN.* into <tag>.* (bin copy in 64 MB blocks, row offsets shifted), sum the statistics, remove the parts."""
    total = {"rows": 0, "val_rows": 0, "tokens": 0, "val_tokens": 0, "dropped_long": 0, "dropped_no_assistant": 0, "by_lang": {}, "by_task": {}}
    for k in ("", ".val"):
        rows, shift = [], 0
        with open(bins / f"{tag}{k}.bin", "wb") as out:
            for i in range(n_parts):
                nm = f"{tag}.p{i:02d}"
                r = np.load(bins / f"{nm}{k}.rows.npy")
                r = r.copy()
                r["off"] += shift
                rows.append(r)
                with open(bins / f"{nm}{k}.bin", "rb") as fh:
                    while True:
                        blk = fh.read(1 << 26)
                        if not blk:
                            break
                        out.write(blk)
                shift += os.path.getsize(bins / f"{nm}{k}.bin") // 2
        np.save(bins / f"{tag}{k}.rows.npy", np.concatenate(rows) if rows else np.zeros(0, ROW_DT))
    for i in range(n_parts):
        nm = f"{tag}.p{i:02d}"
        st = json.loads((bins / f"{nm}.json").read_text())
        for key in ("rows", "val_rows", "tokens", "val_tokens", "dropped_long", "dropped_no_assistant"):
            total[key] += st[key]
        for key in ("by_lang", "by_task"):
            for kk, vv in st[key].items():
                total[key][kk] = total[key].get(kk, 0) + vv
        for ext in (".bin", ".val.bin", ".rows.npy", ".val.rows.npy", ".json"):
            (bins / f"{nm}{ext}").unlink(missing_ok=True)
    (bins / f"{tag}.json").write_text(json.dumps(total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--bins", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--val-mod", type=int, default=100)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--split-shards", type=int, default=6, help="datasets with more shards than this are tokenized as several parallel jobs and merged")
    ap.add_argument("--tokenizer-json", type=Path, default=None)
    a = ap.parse_args()
    if a.workers > 6:
        sys.exit("refusing more than 6 workers on the shared instance")
    a.bins.mkdir(parents=True, exist_ok=True)
    inv = json.loads((a.final / "inventory.json").read_text())
    langs = sorted({k for k in inv["by_source_language_task"]} | {k for k in inv["by_target_language_task"]})
    tasks = sorted({t for d in inv["by_source_language_task"].values() for t in d})
    # codes are STABLE across incremental runs: keep what vocab.json already assigns and append new languages / tasks (uint8 codes, bins of earlier runs stay valid)
    prev = json.loads((a.bins / "vocab.json").read_text()) if (a.bins / "vocab.json").exists() else {"lang": {}, "task": {}}
    lang_codes, task_codes = dict(prev["lang"]), dict(prev["task"])
    for l in langs:
        lang_codes.setdefault(l, len(lang_codes))
    for t in tasks:
        task_codes.setdefault(t, len(task_codes))
    if len(lang_codes) > 254 or len(task_codes) > 254:
        sys.exit("language / task code table no longer fits uint8")
    vocab = {"lang": lang_codes, "task": task_codes, "flags": FLAGS, "row_dtype": str(ROW_DT)}
    (a.bins / "vocab.json").write_text(json.dumps(vocab, indent=1))
    rev = None
    if a.tokenizer_json:
        tok_path = str(a.tokenizer_json)
    else:
        from huggingface_hub import HfApi, hf_hub_download
        token = os.environ.get("HF_TOKEN")
        rev = HfApi(token=token).model_info(REPO).sha
        tok_path = hf_hub_download(REPO, "tokenizer.json", token=token, revision=rev)
    (a.bins / "tokenizer.json.rev").write_text(str(rev or tok_path))
    tags = [p.name for p in sorted(a.final.iterdir()) if p.is_dir()]
    if a.only:
        tags = [t for t in tags if t in a.only]
    jobs, parts_of = [], {}
    for t in tags:
        shards = sorted((a.final / t).glob("part-*.parquet"))
        if (a.bins / f"{t}.json").exists() or not shards:
            continue
        if len(shards) > a.split_shards:                                    # a big dataset becomes several jobs that are merged afterwards
            groups = [shards[i:i + a.split_shards] for i in range(0, len(shards), a.split_shards)]
            parts_of[t] = len(groups)
            jobs += [(t, a.final, a.bins, tok_path, vocab, a.max_tokens, a.val_mod, g, k) for k, g in enumerate(groups)]
        else:
            jobs.append((t, a.final, a.bins, tok_path, vocab, a.max_tokens, a.val_mod, shards, None))
    jobs.sort(key=lambda j: -sum(p.stat().st_size for p in j[7]))             # biggest first
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for tag, rows, state in pool.imap_unordered(work, jobs):
            print(f"{tag}: {rows:,} rows {state} ({(time.time() - t0) / 60:.1f} min)", file=sys.stderr, flush=True)
    for tag, n in parts_of.items():
        if all((a.bins / f"{tag}.p{i:02d}.json").exists() for i in range(n)):
            merge_parts(tag, a.bins, n)
            print(f"{tag}: merged {n} parts", file=sys.stderr, flush=True)
    man = {}
    for p in sorted(a.bins.glob("*.json")):
        if p.name in ("vocab.json", "manifest.json"):
            continue
        man[p.stem] = json.loads(p.read_text())
    (a.bins / "manifest.json").write_text(json.dumps({"tokenizer": REPO, "revision": rev, "max_tokens": a.max_tokens, "datasets": man}, indent=1))
    print(f"{len(man)} datasets, {sum(v['tokens'] for v in man.values()) / 1e9:.2f}B training tokens", file=sys.stderr)


if __name__ == "__main__":
    main()
