"""Export the SFT data as the Hugging Face dataset that is EXACTLY what is in the bin files (owner 2026-10-12: "it should match exactly what is in our bin
files, tagged with language, target language, tasks, as we have filtered, and nothing else").

    # 1. verify only (writes nothing): every exported row, its tags and its token count must equal the bins
    python -m sft_new.export_hf --final final_v2 --bins bins_v2 --out /workspace/sft_build/hf_export --verify-only
    # 2. write parquet shards + dataset card
    python -m sft_new.export_hf --final final_v2 --bins bins_v2 --out /workspace/sft_build/hf_export
    # 3. upload (only after the owner has confirmed repo name and visibility)
    python -m sft_new.export_hf --out /workspace/sft_build/hf_export --push OWNER/NAME --private

What "exactly" means here:
  * rows   = the rows of tokenize_bins' kept set: a row of final_v2/<dataset>/part-*.parquet is exported iff tokenize_bins wrote it (it drops rows whose
             rendered text is longer than --max-tokens or has no <|assistant|>).  Datasets with no drops are the identity; for the others the drop
             decision is recomputed with the same tokenizer and rendering.  Per dataset the number of exported train / validation rows must equal
             rows.npy.
  * split  = tokenize_bins' rule (int(id, 16) % val_mod == 0 -> validation), the same rows as <dataset>.val.bin.
  * tags   = source_lang, targ_lang, task as written in rows.npy (the vocab.json codes are decoded back to names and compared row by row).
  * text   = the string that was tokenized (sft_new.render.render_json of messages + tools), not a re-rendering by anything else; a sample of rows per
             dataset is re-tokenized and compared with the token ids stored in the bin.
Columns: id, split, task, source_lang, targ_lang, source_dataset, text, messages, tools, flags, metadata.  Nothing else is added or kept
(instruction / input / response / output_structure of final_v2 are derivable from messages and are left out).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
FLAGS = {"tools": 1, "think": 2, "tool_call": 4, "unsure": 8, "multi_turn": 16}
COLS = ["id", "split", "task", "source_lang", "targ_lang", "source_dataset", "text", "messages", "tools", "flags", "metadata"]
SHARD_ROWS = 100_000


def _tok(path: str | None):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    p = path or hf_hub_download((os.environ['HF_ORG'] + "/sabiyarn-0.5B"), "tokenizer.json", token=os.environ.get("HF_TOKEN"))
    return Tokenizer.from_file(p)


def kept_rows(shard: Path, tok, asst: int, max_tokens: int, val_mod: int, drop_free: bool):
    """Yield (row_dict) for the rows tokenize_bins kept in this shard, in order, with `split` and `flags` set."""
    from sft_new.render import render_json
    t = pq.read_table(shard, columns=["id", "task", "source_lang", "targ_lang", "source_dataset", "messages", "tools", "metadata"]).to_pydict()
    n = len(t["id"])
    texts = [render_json(m, False, tl or None) for m, tl in zip(t["messages"], t["tools"])]
    if drop_free:
        lens = None
    else:
        lens = [(len(e.ids), asst in e.ids, e.ids.count(asst)) for e in tok.encode_batch(texts, add_special_tokens=False)]
    for i in range(n):
        if lens is not None:
            ln, has_asst, _ = lens[i]
            if ln > max_tokens or not has_asst:
                continue
        tx = texts[i]
        flags = (FLAGS["tools"] if t["tools"][i] else 0) | (FLAGS["think"] if "<think>" in tx else 0) | (FLAGS["tool_call"] if "<tool_call>" in tx else 0) \
            | (FLAGS["unsure"] if "unsure, but most likely" in tx else 0) | (FLAGS["multi_turn"] if tx.count("<|assistant|>") > 1 else 0)
        yield {"id": t["id"][i], "split": "validation" if int(t["id"][i], 16) % val_mod == 0 else "train", "task": t["task"][i],
               "source_lang": t["source_lang"][i], "targ_lang": t["targ_lang"][i], "source_dataset": t["source_dataset"][i], "text": tx,
               "messages": t["messages"][i], "tools": t["tools"][i] or "", "flags": flags, "metadata": t["metadata"][i] or ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", type=Path)
    ap.add_argument("--bins", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-mod", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--tokenizer-json", default=None)
    ap.add_argument("--sample", type=int, default=2000, help="rows per dataset re-tokenized and compared with the bin")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--shard", type=int, default=0, help="process only the datasets whose crc32(tag) %% --nshards == --shard (run several processes)")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--card", action="store_true", help="after all shards: merge export_report_s*.json into export_report.json and write README.md")
    ap.add_argument("--push", default=None, help="OWNER/NAME of the Hub dataset repo (needs --private or --public)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--public", action="store_true")
    a = ap.parse_args()
    if a.push:
        return push(a)
    if a.card:
        import glob
        rows, problems = {"train": 0, "validation": 0}, []
        for f in sorted(glob.glob(str(a.out / "export_report_s*.json"))):
            r = json.loads(Path(f).read_text())
            rows["train"] += r["rows"]["train"]
            rows["validation"] += r["rows"]["validation"]
            problems += r["problems"]
        (a.out / "README.md").write_text(card(a, rows, problems))
        (a.out / "export_report.json").write_text(json.dumps({"rows": rows, "problems": problems}, indent=1))
        print(json.dumps({"rows": rows, "n_problems": len(problems)}))
        return 1 if problems else 0
    vocab = json.loads((a.bins / "vocab.json").read_text())
    inv_l = {v: k for k, v in vocab["lang"].items()}
    inv_t = {v: k for k, v in vocab["task"].items()}
    tok = _tok(a.tokenizer_json)
    asst = tok.token_to_id("<|assistant|>")
    ds = json.loads((a.final / "datasets.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    problems, totals = [], {"train": 0, "validation": 0}
    writers = {"train": [0, None, []], "validation": [0, None, []]}              # shard counter, buffer, written files

    def flush(split, force=False):
        buf = writers[split][2]
        while len(buf) >= SHARD_ROWS or (force and buf):
            chunk, writers[split][2] = buf[:SHARD_ROWS], buf[SHARD_ROWS:]
            buf = writers[split][2]
            d = a.out / "data"
            d.mkdir(exist_ok=True)
            path = d / (f"{split}-s{a.shard:02d}-{writers[split][0]:05d}.parquet" if a.nshards > 1 else f"{split}-{writers[split][0]:05d}.parquet")
            pq.write_table(pa.table({c: [r[c] for r in chunk] for c in COLS}), path, compression="zstd")
            writers[split][0] += 1

    rng = random.Random(0)
    import zlib
    for tag in sorted(ds):
        if a.only and not any(o in tag for o in a.only):
            continue
        if a.nshards > 1 and zlib.crc32(tag.encode()) % a.nshards != a.shard:
            continue
        meta = json.loads((a.bins / f"{tag}.json").read_text()) if (a.bins / f"{tag}.json").exists() else None
        if meta is None:
            if ds[tag]["rows"]:
                problems.append(f"{tag}: not tokenized yet")
            continue
        drop_free = meta["dropped_long"] + meta["dropped_no_assistant"] == 0
        rows = {"train": [], "validation": []}
        for shard in sorted((a.final / tag).glob("part-*.parquet")):
            for r in kept_rows(shard, tok, asst, a.max_tokens, a.val_mod, drop_free):
                rows[r["split"]].append(r)
        for split, k in (("train", ""), ("validation", ".val")):
            rn = np.load(a.bins / f"{tag}{k}.rows.npy")
            got = rows[split]
            if len(got) != len(rn):
                problems.append(f"{tag} {split}: exported {len(got)} rows != bins {len(rn)}")
                continue
            if len(rn):
                src = [inv_l.get(int(x), "?") for x in rn["src"]]
                tgt = [inv_l.get(int(x), "?") for x in rn["tgt"]]
                tsk = [inv_t.get(int(x), "?") for x in rn["task"]]
                bad = sum(1 for r, s, t_, k_ in zip(got, src, tgt, tsk) if (r["source_lang"], r["targ_lang"], r["task"]) != (s, t_, k_))
                badf = sum(1 for r, f in zip(got, rn["flags"]) if r["flags"] != int(f))
                if bad or badf:
                    problems.append(f"{tag} {split}: {bad} rows differ in language / task tags, {badf} in flags")
                # token-level check on a sample: re-tokenize the exported text and compare with the ids stored in the bin
                bin_ = np.memmap(a.bins / f"{tag}{k}.bin", dtype=np.uint16, mode="r")
                for j in rng.sample(range(len(got)), min(a.sample, len(got))):
                    ids = tok.encode(got[j]["text"], add_special_tokens=False).ids
                    off, ln = int(rn["off"][j]), int(rn["len"][j])
                    if ln != len(ids) or not np.array_equal(bin_[off:off + ln], np.asarray(ids, dtype=np.uint16)):
                        problems.append(f"{tag} {split}: row {j} tokens differ from the bin")
                        break
            totals[split] += len(got)
            if not a.verify_only:
                writers[split][2].extend({c: r[c] for c in COLS} for r in got)
                flush(split)
        print(f"{tag}: train {len(rows['train']):,} validation {len(rows['validation']):,}", flush=True)
    if not a.verify_only:
        for s in writers:
            flush(s, force=True)
        if a.nshards == 1:
            (a.out / "README.md").write_text(card(a, totals, problems))
            (a.out / "export_report.json").write_text(json.dumps({"rows": totals, "problems": problems}, indent=1))
    if a.nshards > 1:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / f"export_report_s{a.shard:02d}.json").write_text(json.dumps({"rows": totals, "problems": problems}, indent=1))
    print(json.dumps({"rows": totals, "problems": problems[:20], "n_problems": len(problems)}))
    return 1 if problems else 0


def card(a, totals, problems) -> str:
    return "\n".join([
        "# SabiYarn SFT (October 2026 build)", "",
        f"{totals['train']:,} training and {totals['validation']:,} validation conversations, exactly the rows of the tokenized SFT bins.", "",
        "| column | meaning |", "|---|---|",
        "| `id` | content hash (global de-duplication key) |", "| `split` | `train` / `validation` (1% by id hash, as in the bins) |",
        "| `task` | task label |", "| `source_lang`, `targ_lang` | language tags of the source and target side (our tokenizer's tags) |",
        "| `source_dataset` | the dataset the row came from |", "| `text` | the rendered training string (chat template v2) |",
        "| `messages`, `tools` | the conversation as JSON (OpenAI-shaped) and the function definitions of tool-using rows |",
        "| `flags` | bit set: 1 tools, 2 think, 4 tool_call, 8 unsure language, 16 multi-turn |", "| `metadata` | per-row provenance / licence notes (JSON) |", "",
        "Licences are mixed: rows are used only from sources that are commercial-OK or have no declared licence; non-commercial sources were skipped.",
        "See `source_dataset` and `metadata` for provenance. Verification: " + ("no differences against the bins." if not problems else f"{len(problems)} problems, see export_report.json."), ""])


def push(a):
    from huggingface_hub import HfApi
    if not (a.private ^ a.public):
        sys.exit("pass exactly one of --private / --public")
    tok = os.environ.get("HF_WRITE_TOKEN")
    if not tok:
        sys.exit("HF_WRITE_TOKEN is not set")
    rep = json.loads((a.out / "export_report.json").read_text())
    if rep["problems"]:
        sys.exit(f"export has {len(rep['problems'])} verification problems; not pushing")
    api = HfApi(token=tok)
    api.create_repo(a.push, repo_type="dataset", private=a.private, exist_ok=True)
    api.upload_folder(folder_path=str(a.out), repo_id=a.push, repo_type="dataset", commit_message="SFT build (matches the tokenized bins)",
                      ignore_patterns=["export_report.json"], allow_patterns=["data/*", "README.md"])
    print("pushed", a.push)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
