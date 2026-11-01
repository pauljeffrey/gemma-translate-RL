"""AfriCOMET-QE filter for michsethowusu/twi-english-parallel-synthetic-50m (or any other two-column parquet translation repo), saved as INDICES.

    python -m sft_new.twi50m_filter --out DIR [--chunk 200000] [--batch 192] [--gpu-fraction 0.3] [--max-chunks N]
    python -m sft_new.twi50m_filter --repo michsethowusu/english-setswana_sentence-pairs_mt560 --cols tsn,eng --out DIR     # --cols NON_ENGLISH,ENGLISH

Row index = position in the dataset when its 12 parquet shards are read in file-name order (train-00000 ... train-00011), rows in file order, starting
at 0 (`shards.json` lists the shards and their global offsets). Columns: twi_sentence (source), english_parahrase (the English side).

Outputs in DIR:
  scores_float16.npy          one float16 QE score per row; NaN for rows that were not scored (see status)
  status_uint8.npy            0 scored | 1 empty/null | 2 identical sides | 3 too short or too long | 4 length ratio | 5 exact duplicate pair
                              | 6 mostly non-letters
  keep_qe_ge_0.46.npy         int64 indices with status 0 and score >= 0.46 (AfriCOMET-QE failure line: below it a translation is gibberish/omission)
  keep_qe_ge_0.60.npy         int64 indices with status 0 and score >= 0.60 (about the median of human references for English-centric pairs)
  part/chunk_XXXXX.npz        per-chunk results (resumable); progress.json; README.json
The job is resumable and a finished chunk is usable immediately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

REPO = "michsethowusu/twi-english-parallel-synthetic-50m"
COLS = ("twi_sentence", "english_parahrase")
PLACEHOLDER = {"", "none", "null", "nan", "n/a", "!!null", "null!!", "#value!"}
STATUS = {0: "scored", 1: "empty_or_null", 2: "identical_sides", 3: "too_short_or_long", 4: "length_ratio", 5: "duplicate_pair", 6: "mostly_non_letters"}


def norm(s) -> str:
    if not isinstance(s, str):
        return ""
    t = unicodedata.normalize("NFC", s).strip()
    return "" if t.lower() in PLACEHOLDER else " ".join(t.split())


def prefilter(src: str, tgt: str) -> int:
    if not src or not tgt:
        return 1
    if src.casefold() == tgt.casefold():
        return 2
    a, b = len(src), len(tgt)
    if a < 2 or b < 2 or a > 600 or b > 600:
        return 3
    if min(a, b) >= 12 and max(a, b) / min(a, b) > 4.0:
        return 4
    for t in (src, tgt):
        letters = sum(c.isalpha() for c in t)
        if letters < 0.5 * len(t):
            return 6
    return 0


def main():
    global REPO, COLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--cols", default=",".join(COLS), help="NON_ENGLISH_COLUMN,ENGLISH_COLUMN")
    ap.add_argument("--src-dir", type=Path, default=None, help="where the parquet shards are kept (default OUT/../twi50m_src)")
    ap.add_argument("--chunk", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--gpu-fraction", type=float, default=0.3)
    ap.add_argument("--max-chunks", type=int, default=0)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    import torch
    from huggingface_hub import HfApi, hf_hub_download

    REPO, COLS = a.repo, tuple(a.cols.split(","))
    token = os.environ.get("HF_TOKEN")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "part").mkdir(exist_ok=True)
    src_dir = a.src_dir or (a.out.parent / "twi50m_src" if a.repo == "michsethowusu/twi-english-parallel-synthetic-50m" else a.out.parent / (a.out.name + "_src"))
    src_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in HfApi(token=token).list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
    local = []
    for f in files:
        local.append(Path(hf_hub_download(REPO, f, repo_type="dataset", token=token, local_dir=str(src_dir))))
    offsets, total = [], 0
    for p in local:
        n = pq.ParquetFile(p).metadata.num_rows
        offsets.append({"file": p.name, "rows": n, "offset": total})
        total += n
    (a.out / "shards.json").write_text(json.dumps({"repo": REPO, "total_rows": total, "shards": offsets}, indent=1))
    print(f"{len(local)} shards, {total:,} rows", flush=True)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.gpu_fraction)
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model("masakhane/africomet-qe-stl-1.1"))
    model.eval()

    seen: set[bytes] = set()          # exact (casefolded) pair duplicates; 47.9M x 8 bytes
    n_chunks = (total + a.chunk - 1) // a.chunk
    done_rows, t0, scored_total = 0, time.time(), 0
    # stream the shards in order, cutting chunks of `a.chunk` rows that may span shard boundaries
    buf_src: list[str] = []
    buf_tgt: list[str] = []
    chunk_id = 0

    def flush(chunk_id: int, base: int, srcs: list[str], tgts: list[str]) -> int:
        part = a.out / "part" / f"chunk_{chunk_id:05d}.npz"
        n = len(srcs)
        status = np.zeros(n, np.uint8)
        todo = []
        for i, (s, t) in enumerate(zip(srcs, tgts)):
            st = prefilter(s, t)
            if st == 0:
                h = hashlib.blake2b((s.casefold() + "\x1f" + t.casefold()).encode(), digest_size=8).digest()
                if h in seen:
                    st = 5
                else:
                    seen.add(h)
            status[i] = st
            if st == 0:
                todo.append(i)
        scores = np.full(n, np.nan, np.float32)
        if part.exists():                                    # resumed run: the dedupe set above was rebuilt, the scores are kept
            return int(np.load(part)["status"].size)
        for lo in range(0, len(todo), 40_000):
            sel = todo[lo:lo + 40_000]
            data = [{"src": tgts[i], "mt": srcs[i]} for i in sel]            # source = English, mt = Twi (the direction AfriCOMET-QE judges)
            for attempt in range(6):
                try:
                    out = model.predict(data, batch_size=a.batch, gpus=1, progress_bar=False, num_workers=2)
                    break
                except torch.cuda.OutOfMemoryError as e:         # the GPU is shared with another job: wait and retry
                    print(f"CUDA out of memory (attempt {attempt + 1}); batch {a.batch}; {str(e)[:260]}", flush=True)
                    torch.cuda.empty_cache()
                    time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError("GPU out of memory six times in a row")
            sc = np.asarray(out.scores if hasattr(out, "scores") else out, np.float32)
            scores[np.asarray(sel)[: len(sc)]] = sc
        np.savez_compressed(part, status=status, scores=scores.astype(np.float16), base=np.int64(base))
        return n

    for pth in local:
        pf = pq.ParquetFile(pth)
        for batch in pf.iter_batches(batch_size=50_000, columns=list(COLS)):
            d = batch.to_pydict()
            buf_src += [norm(x) for x in d[COLS[0]]]
            buf_tgt += [norm(x) for x in d[COLS[1]]]
            while len(buf_src) >= a.chunk:
                n = flush(chunk_id, done_rows, buf_src[:a.chunk], buf_tgt[:a.chunk])
                buf_src, buf_tgt = buf_src[a.chunk:], buf_tgt[a.chunk:]
                done_rows += n
                chunk_id += 1
                el = time.time() - t0
                (a.out / "progress.json").write_text(json.dumps({"chunks_done": chunk_id, "chunks_total": n_chunks, "rows_done": done_rows, "rows_total": total,
                                                                 "elapsed_s": round(el), "rows_per_s": round(done_rows / max(el, 1))}))
                print(f"chunk {chunk_id}/{n_chunks} rows {done_rows:,} elapsed {el / 60:.1f} min", flush=True)
                if a.max_chunks and chunk_id >= a.max_chunks:
                    return
    if buf_src:
        n = flush(chunk_id, done_rows, buf_src, buf_tgt)
        done_rows += n
        chunk_id += 1
    # ---- assemble the global arrays
    scores = np.full(total, np.nan, np.float16)
    status = np.zeros(total, np.uint8)
    for i in range(chunk_id):
        z = np.load(a.out / "part" / f"chunk_{i:05d}.npz")
        b = int(z["base"])
        status[b:b + z["status"].size] = z["status"]
        scores[b:b + z["scores"].size] = z["scores"]
    np.save(a.out / "scores_float16.npy", scores)
    np.save(a.out / "status_uint8.npy", status)
    ok = status == 0
    keep = {}
    for thr in (0.46, 0.60):
        idx = np.flatnonzero(ok & (scores.astype(np.float32) >= thr)).astype(np.int64)
        np.save(a.out / f"keep_qe_ge_{thr:.2f}.npy", idx)
        keep[f"{thr:.2f}"] = int(idx.size)
    readme = {"dataset": REPO, "rows": total, "index": "global row position over shards sorted by name, file row order (see shards.json)",
              "model": "masakhane/africomet-qe-stl-1.1 (reference-free; src = English side, mt = the other language)", "status_codes": STATUS,
              "status_counts": {STATUS[k]: int((status == k).sum()) for k in STATUS}, "kept": keep,
              "score_quantiles_scored": {f"p{q}": float(np.nanpercentile(scores[ok].astype(np.float32), q)) for q in (5, 10, 25, 50, 75, 90)} if ok.any() else {},
              "note": "QE falls with sentence length and is a sentence-level metric; thresholds are a starting point, scores_float16.npy lets you pick your own."}
    (a.out / "README.json").write_text(json.dumps(readme, indent=1))
    print(json.dumps(readme), flush=True)


if __name__ == "__main__":
    main()
