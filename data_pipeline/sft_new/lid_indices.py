"""GlotLID audit of the DECLARED language of both columns of a parquet translation repo, saved as INDICES (companion of twi50m_filter.py).

    python -m sft_new.lid_indices --repo michsethowusu/twi-english-parallel-synthetic-50m --cols twi_sentence,english_parahrase --declared twi,eng \
        --src-dir /workspace/sft_build/out/twi50m_src --shards /workspace/sft_build/out/twi50m_filter/shards.json \
        --keep /workspace/sft_build/out/twi50m_filter/keep_qe_ge_0.46.npy --out /workspace/sft_build/out/twi50m_filter --procs 6

Row index = global row position over the shards in `shards.json` (the same index twi50m_filter.py uses). A row is `mismatch` on a side when GlotLID is
confident (top-1 probability >= 0.55) that the text is a different language than declared (sft_new/langaudit.py explains the rule).

Outputs in --out:
  lid_status_uint8.npy      per row: 0 both sides fine | 1 first (non-English) column mismatched | 2 second column mismatched | 3 both | 255 not audited
  keep_<keepname>_lid_ok.npy  the --keep indices whose status is 0  (e.g. keep_qe_ge_0.46_lid_ok.npy)
  lid_audit.json            counts, and the most common wrong languages per side
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TAU = 0.55
BATCH = 2000


def _clean(s) -> str:
    return " ".join(s.split())[:1000] if isinstance(s, str) else ""


def _work(job):
    path, offset, cols, declared, keep_local, tau = job
    import pyarrow.parquet as pq
    from africomet_translation import lid
    from sft_new.langaudit import accepted
    model = lid._model().f
    acc = [accepted(d) for d in declared]
    tbl = pq.read_table(path, columns=list(cols))
    n = tbl.num_rows
    sel = np.arange(n) if keep_local is None else keep_local
    texts = [tbl.column(c).to_pylist() for c in cols]
    status = np.zeros(len(sel), np.uint8)
    wrong = [collections.Counter(), collections.Counter()]
    for side in (0, 1):
        for lo in range(0, len(sel), BATCH):
            ids = sel[lo:lo + BATCH]
            batch = [_clean(texts[side][i]) for i in ids]
            raw = model.multilinePredict(batch, 1, 0.0, "replace")
            if isinstance(raw, tuple) and len(raw) == 2:
                labels, probs = raw
            else:
                labels = [[x[1] for x in r] for r in raw]
                probs = [[x[0] for x in r] for r in raw]
            for j, (lb, pr) in enumerate(zip(labels, probs)):
                if not lb or not batch[j]:
                    continue
                iso = lb[0].replace("__label__", "").split("_")[0]
                if iso not in acc[side] and float(pr[0]) >= tau:
                    status[lo + j] |= 1 << side
                    wrong[side][lb[0].replace("__label__", "")] += 1
    return offset, np.asarray(sel, np.int64) + offset, status, [dict(w.most_common(40)) for w in wrong]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cols", required=True)
    ap.add_argument("--declared", required=True)
    ap.add_argument("--src-dir", type=Path, required=True)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--keep", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--tau", type=float, default=TAU)
    a = ap.parse_args()
    cols, declared = a.cols.split(","), a.declared.split(",")
    meta = json.loads(a.shards.read_text())
    total = meta["total_rows"]
    keep = np.sort(np.load(a.keep)) if a.keep else None
    jobs = []
    for sh in meta["shards"]:
        lo, hi = sh["offset"], sh["offset"] + sh["rows"]
        local = None
        if keep is not None:
            i, j = np.searchsorted(keep, [lo, hi])
            local = keep[i:j] - lo
            if local.size == 0:
                continue
        jobs.append((str(a.src_dir / sh["file"]), lo, cols, declared, local, a.tau))
    status = np.full(total, 255, np.uint8)
    wrong = [collections.Counter(), collections.Counter()]
    with mp.get_context("spawn").Pool(a.procs) as pool:
        for off, idx, st, w in pool.imap_unordered(_work, jobs):
            status[idx] = st
            for side in (0, 1):
                wrong[side].update(w[side])
            print(f"shard offset {off:,}: {int((st == 0).sum()):,} ok of {st.size:,}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "lid_status_uint8.npy", status)
    summary = {"repo": a.repo, "columns": cols, "declared": declared, "tau": a.tau, "rows": total,
               "audited": int((status != 255).sum()), "ok": int((status == 0).sum()),
               "mismatch_first_col": int(((status != 255) & ((status & 1) > 0)).sum()),
               "mismatch_second_col": int(((status != 255) & ((status & 2) > 0)).sum()),
               "most_common_wrong_labels_first_col": dict(wrong[0].most_common(25)), "most_common_wrong_labels_second_col": dict(wrong[1].most_common(25))}
    if keep is not None:
        ok = keep[status[keep] == 0]
        name = a.keep.stem.replace("keep_", "")
        np.save(a.out / f"keep_{name}_lid_ok.npy", ok)
        summary["kept_after_lid"] = int(ok.size)
        summary["keep_file"] = f"keep_{name}_lid_ok.npy"
    (a.out / "lid_audit.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
