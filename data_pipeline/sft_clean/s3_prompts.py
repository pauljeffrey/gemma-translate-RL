#!/usr/bin/env python3
"""Clean RL prompt pools (agent S3, 2026-10-28).

Takes the old pools (rl_prompts_v3: sft_build/gemma/rl_prompts_v3/<iso>.parquet; rl_prompts_sci_v2: jobs/r7/assets2/rl_prompts_sci_v2/<iso>.parquet; both in the Hub backup
<HF_ORG_LEGACY>/sabiyarn-box-final-backup), filters the ENGLISH text of every prompt against the protected index (decon.py: R1 exact sentence, R2 contained sentence,
R3 shared word 8-gram; both normal forms; strict rule) and writes the clean pools in the format gemma_mt.rl.data.load_rl_prompts reads (one parquet per language with the
original columns, all length bins in one file) plus the same rows split per length bin. The b2048 bin is dropped by design (counted, not scanned). Provenance per kept row:
prov_pool, prov_file, prov_row (row number in the old file), prov_en_xxh64, prov_index, flag_ssa_mte_train (flag only, never removes).

The strict mode does NOT attach the FLORES+ dev train-half exemption: prompts do not need the train half, so it is removed as well; the removals that exist only because of the
train half are counted separately (exempt_only). One file at a time: download -> check -> write -> (batched) upload -> delete.

    HF_TOKEN=... taskset -c 24,26 nice -n 19 python s3_prompts.py --pool sci_v2 --index protected/protected_index.npz --manifest protected/protected_manifest.json \
        --aux aux/ssa_mte_train/ssa_mte_train_index.npz --work /workspace/clean_rl/work --out /workspace/clean_rl/out --upload
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import xxhash  # noqa: E402

import decon as D  # noqa: E402
import s3_common as C  # noqa: E402

SRC_REPO = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-box-final-backup")
OUT_REPO = (os.environ['HF_ORG_LEGACY'] + "/sabiyarn-clean-rl-prompts-2026-10-28")
POOLS = {
    "v3": dict(prefix="sft_build/gemma/rl_prompts_v3/", out="clean_v3"),
    "sci_v2": dict(prefix="jobs/r7/assets2/rl_prompts_sci_v2/", out="clean_sci_v2"),
}
KEEP_BINS = ("b64", "b256", "b512", "b1024")
DROP_BINS = ("b2048",)


def h64(s: str) -> int:
    return xxhash.xxh64_intdigest((s or "").encode("utf-8", "surrogatepass"))


class Uploader:
    """Batches commit operations and uploads them in a background thread (at most one commit in flight) so that the CPU workers never wait for the network."""

    def __init__(self, api, repo, enabled, delete_after=False, flush_bytes=500 << 20, flush_n=60):
        self.api, self.repo, self.enabled, self.delete_after = api, repo, enabled, delete_after
        self.ops, self.bytes, self.flush_bytes, self.flush_n = [], 0, flush_bytes, flush_n
        self.local = []
        self.ex = ThreadPoolExecutor(1)
        self.fut = None
        self.committed = []

    def add(self, local, repo_path):
        from huggingface_hub import CommitOperationAdd
        self.ops.append(CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local))
        self.local.append(local)
        self.bytes += os.path.getsize(local)
        if self.bytes >= self.flush_bytes or len(self.ops) >= self.flush_n:
            self.flush()

    def _commit(self, ops, local, msg):
        for attempt in range(8):
            try:
                self.api.create_commit(self.repo, ops, repo_type="dataset", commit_message=f"{msg} ({len(ops)} files)")
                break
            except Exception as e:  # noqa: BLE001
                print("upload retry", attempt, type(e).__name__, str(e)[:160], flush=True)
                time.sleep(20 * (attempt + 1))
        else:
            raise RuntimeError("upload failed")
        self.committed.extend(o.path_in_repo for o in ops)
        if self.delete_after:
            for p in local:
                try:
                    os.remove(p)
                except OSError:
                    pass
        print(C.now(), "committed", len(ops), "files", flush=True)

    def flush(self, msg="S3 clean RL prompts", wait=False):
        if self.ops and self.enabled:
            if self.fut is not None:
                self.fut.result()          # at most one commit in flight
            self.fut = self.ex.submit(self._commit, self.ops, self.local, msg)
        self.ops, self.local, self.bytes = [], [], 0
        if wait and self.fut is not None:
            self.fut.result()
        return self.committed


def fieldgroup(f: str) -> str:
    return (f or "").split(":")[0] if f else ""


def process_file(path, pool, iso, repo_file, checker, ix, aux, memo_clean, memo_dirty, outdir, index_version, limit=0):
    t = pq.read_table(path)
    if limit:
        t = t.slice(0, limit)
    n = t.num_rows
    cols = t.column_names
    eng = t.column("english").to_pylist()
    bins = t.column("length_bin").to_pylist()
    src = t.column("source_dataset").to_pylist() if "source_dataset" in cols else [""] * n
    dom = t.column("domain").to_pylist() if "domain" in cols else [None] * n
    fld = t.column("field").to_pylist() if "field" in cols else [None] * n
    hashes = [h64(x) for x in eng]
    need = {}
    for i in range(n):
        if bins[i] in DROP_BINS:
            continue
        h = hashes[i]
        if h in memo_clean or h in memo_dirty or h in need:
            continue
        need[h] = eng[i] or ""
    t0 = time.time()
    hs = list(need.keys())
    res = checker.run([need[h] for h in hs]) if hs else []
    for h, r in zip(hs, res):
        if r[0] | r[1] | r[2] | r[4] | r[5] | r[6]:
            memo_dirty[h] = r
        else:
            memo_clean.add(h)
    dt = time.time() - t0
    keep = np.zeros(n, dtype=bool)
    st = dict(pool=pool, iso=iso, file=repo_file, rows_in=n, unique_english=len(set(hashes)), unique_checked_now=len(hs), check_seconds=round(dt, 1),
              rows_b2048_dropped=0, rows_scanned=0, rows_removed=0, rows_kept=0, rows_removed_exempt_only=0, kept_flag_ssa_mte_train=0, removed_flag_ssa_mte_train=0)
    cells = collections.defaultdict(lambda: [0, 0])       # "bin|source|domain|fieldgroup" -> [before, after]
    by_rule_set = collections.Counter()                  # "rule|set" -> rows
    by_set_any = collections.Counter()                   # "set" -> rows with any rule
    by_set_sole = collections.Counter()                  # "set" -> rows whose hits are in this set only
    by_set_bin = collections.Counter()                   # "bin|set" -> rows
    by_dom_set = collections.Counter()                   # "source|domain|set" -> rows
    exempt_sets = collections.Counter()                  # sets of the rows removed only because of the FLORES+ dev train half
    rem_rows = []
    flags = np.zeros(n, dtype=bool)
    for i in range(n):
        b = bins[i]
        d = dom[i] if dom[i] else "general"
        key = f"{b}|{src[i]}|{d}|{fieldgroup(fld[i])}"
        if b in DROP_BINS:
            st["rows_b2048_dropped"] += 1
            cells[key][0] += 1
            continue
        st["rows_scanned"] += 1
        cells[key][0] += 1
        h = hashes[i]
        r = memo_dirty.get(h)
        if r is None:
            keep[i] = True
            st["rows_kept"] += 1
            cells[key][1] += 1
            continue
        removed = bool(r[0] | r[1] | r[2])
        flagged = bool(r[4] | r[5] | r[6])
        flags[i] = flagged
        if not removed:
            keep[i] = True
            st["rows_kept"] += 1
            st["kept_flag_ssa_mte_train"] += int(flagged)
            cells[key][1] += 1
            continue
        st["rows_removed"] += 1
        st["removed_flag_ssa_mte_train"] += int(flagged)
        allm = r[0] | r[1] | r[2]
        names = ix.names(allm)
        for rule, m in zip(C.RULES, r[:3]):
            for nm in ix.names(m):
                by_rule_set[f"{rule}|{nm}"] += 1
        for nm in names:
            by_set_any[nm] += 1
            by_set_bin[f"{b}|{nm}"] += 1
            by_dom_set[f"{src[i]}|{d}|{nm}"] += 1
        if len(names) == 1:
            by_set_sole[names[0]] += 1
        ex_only = bool(r[3])
        if ex_only:
            st["rows_removed_exempt_only"] += 1
            for nm in names:
                exempt_sets[nm] += 1
        rem_rows.append((i, hashes[i], b, src[i], d, ";".join(names), ";".join(f"{rule}" for rule, m in zip(C.RULES, r[:3]) if m), ex_only))
    st["cells"] = {k: v for k, v in cells.items()}
    st["by_rule_set"], st["by_set_any"], st["by_set_sole"] = dict(by_rule_set), dict(by_set_any), dict(by_set_sole)
    st["by_set_bin"], st["by_dom_set"], st["exempt_only_sets"] = dict(by_set_bin), dict(by_dom_set), dict(exempt_sets)
    # ---- outputs
    kt = t.filter(pa.array(keep))
    kidx = np.nonzero(keep)[0]
    prov = {
        "prov_pool": pa.array([pool] * len(kidx), pa.string()),
        "prov_file": pa.array([repo_file] * len(kidx), pa.string()),
        "prov_row": pa.array(kidx.astype(np.int64), pa.int64()),
        "prov_en_xxh64": pa.array([format(hashes[i], "016x") for i in kidx], pa.string()),
        "prov_index": pa.array([index_version] * len(kidx), pa.string()),
        "flag_ssa_mte_train": pa.array(flags[kidx], pa.bool_()),
    }
    for k, v in prov.items():
        kt = kt.append_column(k, v)
    outs = []
    os.makedirs(os.path.join(outdir, POOLS[pool]["out"]), exist_ok=True)
    p_all = os.path.join(outdir, POOLS[pool]["out"], f"{iso}.parquet")
    pq.write_table(kt, p_all, compression="zstd")
    outs.append((p_all, f"{POOLS[pool]['out']}/{iso}.parquet"))
    kb = kt.column("length_bin").to_pylist()
    st["kept_by_bin"] = collections.Counter(kb)
    st["kept_by_bin"] = dict(st["kept_by_bin"])
    for b in KEEP_BINS:
        sel = [j for j, x in enumerate(kb) if x == b]
        if not sel:
            continue
        d_ = os.path.join(outdir, "by_bin", pool, b)
        os.makedirs(d_, exist_ok=True)
        pb = os.path.join(d_, f"{iso}.parquet")
        pq.write_table(kt.take(pa.array(sel)), pb, compression="zstd")
        outs.append((pb, f"by_bin/{pool}/{b}/{iso}.parquet"))
    # removal log: provenance of every removed prompt, no text
    os.makedirs(os.path.join(outdir, "removed_log"), exist_ok=True)
    rl = os.path.join(outdir, "removed_log", f"{pool}__{iso}.parquet")
    cols_ = list(zip(*rem_rows)) if rem_rows else [[], [], [], [], [], [], [], []]
    pq.write_table(pa.table({"prov_row": pa.array(cols_[0], pa.int64()), "prov_en_xxh64": pa.array([format(x, "016x") for x in cols_[1]], pa.string()),
                             "length_bin": pa.array(cols_[2], pa.string()), "source_dataset": pa.array(cols_[3], pa.string()), "domain": pa.array(cols_[4], pa.string()),
                             "protected_sets": pa.array(cols_[5], pa.string()), "rules": pa.array(cols_[6], pa.string()), "removed_only_by_flores_dev_train_half": pa.array(cols_[7], pa.bool_())}),
                   rl, compression="zstd")
    outs.append((rl, f"removed_log/{pool}__{iso}.parquet"))
    st["schema_in"] = [f"{f.name}:{f.type}" for f in t.schema]
    return st, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, choices=list(POOLS))
    ap.add_argument("--index", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--aux", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit-rows", type=int, default=0)
    ap.add_argument("--delete-after-upload", action="store_true", help="default: keep the local clean files until s3_verify.py has run")
    ap.add_argument("--manifest-csv", default=None, help="MANIFEST.csv of the backup (sha256 per file) to verify the downloads")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ["HF_TOKEN"]
    api = HfApi(token=tok)
    pool = a.pool
    cfg = POOLS[pool]
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "stats"), exist_ok=True)
    os.makedirs(a.work, exist_ok=True)
    sha_ref = {}
    if a.manifest_csv:
        for row in csv.DictReader(open(a.manifest_csv, encoding="utf-8")):
            sha_ref[row["path"]] = (row["sha256"], int(row["size"]))
    files = sorted(f.path for f in api.list_repo_tree(SRC_REPO, repo_type="dataset", path_in_repo=cfg["prefix"].rstrip("/")) if f.path.endswith(".parquet") and "/" not in f.path[len(cfg["prefix"]):])
    isos = [os.path.basename(f)[:-8] for f in files]
    if a.only:
        files = [f for f, i in zip(files, isos) if i in a.only]
        isos = [i for i in isos if i in a.only]
    print(C.now(), pool, len(files), "files", flush=True)
    ix, ex, man = C.load_strict(a.index, a.manifest)
    index_version = man["version"]
    aux = D.Index.load(a.aux) if a.aux else None
    print(C.now(), "index", index_version, man["index_file_sha256"][:16], "aux", bool(aux), flush=True)
    checker = C.Checker(ix, ex, aux, nproc=a.nproc)
    up = Uploader(api, OUT_REPO, a.upload, delete_after=a.delete_after_upload)
    if a.upload:
        api.create_repo(OUT_REPO, repo_type="dataset", private=True, exist_ok=True)
    memo_clean, memo_dirty = set(), {}
    pool_ex = ThreadPoolExecutor(1)

    def dl(f):
        p = hf_hub_download(SRC_REPO, f, repo_type="dataset", token=tok, local_dir=a.work)
        sz = os.path.getsize(p)
        sha = None
        if f in sha_ref:
            hh = hashlib.sha256()
            with open(p, "rb") as fh:
                for ch in iter(lambda: fh.read(1 << 24), b""):
                    hh.update(ch)
            sha = hh.hexdigest()
            if sha != sha_ref[f][0]:
                raise RuntimeError(f"sha256 mismatch for {f}")
        return p, sz, sha

    fut = {}
    todo = []
    for f, i in zip(files, isos):
        sp = os.path.join(a.out, "stats", f"{pool}__{i}.json")
        if os.path.exists(sp) and not a.limit_rows:
            continue
        todo.append((f, i))
    if todo:
        fut[todo[0][0]] = pool_ex.submit(dl, todo[0][0])
    tot = collections.Counter()
    for k, (f, i) in enumerate(todo):
        if k + 1 < len(todo):
            fut[todo[k + 1][0]] = pool_ex.submit(dl, todo[k + 1][0])
        p, sz, sha = fut.pop(f).result()
        t0 = time.time()
        st, outs = process_file(p, pool, i, f, checker, ix, aux, memo_clean, memo_dirty, a.out, index_version, a.limit_rows)
        st.update(input_bytes=sz, input_sha256=sha, input_sha256_verified=bool(sha), index_version=index_version, seconds=round(time.time() - t0, 1))
        os.remove(p)
        json.dump(st, open(os.path.join(a.out, "stats", f"{pool}__{i}.json") if not a.limit_rows else os.path.join(a.out, "stats", f"TEST_{pool}__{i}.json"), "w"))
        for lp, rp in outs:
            up.add(lp, rp)
        tot.update(rows_in=st["rows_in"], scanned=st["rows_scanned"], removed=st["rows_removed"], kept=st["rows_kept"])
        print(C.now(), f"{k + 1}/{len(todo)} {pool}/{i}: in {st['rows_in']} scanned {st['rows_scanned']} removed {st['rows_removed']} "
              f"({100 * st['rows_removed'] / max(1, st['rows_scanned']):.2f}%) kept {st['rows_kept']} uniq-checked {st['unique_checked_now']} {st['check_seconds']}s total {st['seconds']}s", flush=True)
    up.flush(wait=True)
    checker.close()
    print(C.now(), "finished", dict(tot), flush=True)


if __name__ == "__main__":
    main()
