#!/usr/bin/env python3
"""STEP (a): AfriCOMET-QE-STL-1.1 (masakhane/africomet-qe-stl-1.1, reference-free) over the clean pairs. Resumable per chunk; prints pairs/s and an ETA.

    python qe_score.py --src data data_devhalf specialist --out scores [--half] [--batch 64] [--chunk 100000] [--workers 4] [--part 0/1]
    python qe_score.py --src data --out scores --stub          # CPU test: STUB scores, written to scores_STUB/, can never be mistaken for real ones

Output: <out>/<dir>__<shard>-<chunk>.parquet with columns key (uint64, = key_of(english, iso, target) as in out/kept), qe (float32), scorer (string: the model id, or
`STUB-NOT-REAL`). A pair that is in several directories (specialist repeats main rows) is scored once per directory it appears in; gate.py/windows.py match by key.
A chunk file that exists is skipped (resume). Real scoring needs torch + unbabel-comet; the old script's fp16 setting (--half) agreed with fp32 to pearson 0.99998.
Pairs whose English or target is empty are skipped (the old script did the same).

Resume across devices (CPU job first, GPU later): every score row carries key (the stable pair id), qe, scorer (model id), model_sha, device, dtype, ts. `--skip-scored DIR...` skips every key that
already has a score in any of these directories (shards written by a CPU or a GPU run are interchangeable: gate.py takes several --scores directories and the first score per key wins).
`--device cpu` forces fp32 on CPU (CUDA hidden); `--device cuda` (default when available) with `--half` is the GPU path. `--manifest/--progress` give per-stage progress for the CPU queue.
"""
import argparse, collections, json, os, sys, time
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import key_of, shard_files, shard_id, cap_thresholds, in_cap

MODEL = "masakhane/africomet-qe-stl-1.1"


def stub_scores(keys):
    """deterministic pseudo-scores in [0.05, 0.95] from the key: NOT a quality estimate, only for pipeline tests"""
    return np.array([0.05 + 0.9 * ((k * 0x9E3779B97F4A7C15) % (1 << 64)) / (1 << 64) for k in keys], np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--part", default="0/1", help="i/N: this process handles shard files with index %% N == i (run several processes on one GPU)")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (default: all visible cores)")
    ap.add_argument("--skip-scored", nargs="*", default=[], help="score directories whose keys are NOT scored again")
    ap.add_argument("--manifest", default=None, help="queue manifest json {shard: {stage, n}} (qe_cpu_queue.py)")
    ap.add_argument("--max-load", type=float, default=0.0, help="back off (sleep) while the 1-minute load average is above this (shared box; 0 = off)")
    ap.add_argument("--progress", default=None, help="progress json rewritten after every chunk")
    ap.add_argument("--stub", action="store_true", help="NO model: pseudo scores, output dir gets the suffix _STUB")
    ap.add_argument("--exclude-nc", action="store_true", help="do not score rows of licence class non_commercial (they are removed from training: Amendment 4)")
    ap.add_argument("--nc-allow", nargs="*", default=[], help="NC sources that stay (owner exceptions), repo[=tag]: scored although --exclude-nc")
    ap.add_argument("--max-pairs", type=int, default=0, help="smoke test: stop after this many pairs in total")
    ap.add_argument("--iso", nargs="*", default=None, help="only these languages")
    ap.add_argument("--skip-keys-of", nargs="*", default=[], help="do not score pairs whose key already occurs in these directories (specialist/ repeats the main rows of its languages: score data/ once)")
    ap.add_argument("--cap-per-lang", type=int, default=0, help="score only a deterministic, MONOTONE sample of about N pairs per language: pair kept when (key & 0xFFFFFFFF) < ceil(N * 2^32 / pairs_of_language); languages with <= N pairs in full; 0 = no cap")
    ap.add_argument("--cap-counts", default=None, help="cap_counts.json written ONCE by qe_cap.py count over all source dirs (the same file for every run, so that a larger N scores a superset); default: count the --src dirs now")
    ap.add_argument("--no-fast", action="store_true", help="use comet's model.predict for every batch (the old path) instead of qe_fast.FastQE (separate length-sorted source / translation passes, token-budget batches, English-source cache)")
    ap.add_argument("--fast-token-budget", type=int, default=24576); ap.add_argument("--fast-cache-gb", type=float, default=8.0, help="CPU memory for the cache of English-source embeddings (fp16, 2 KB per sentence)")
    ap.add_argument("--fast-min-pearson", type=float, default=0.9999, help="the first chunk is also scored by model.predict (up to 384 pairs) and compared; below this Pearson the run falls back to model.predict")
    ap.add_argument("--budget-pairs-per-s", type=float, default=0.0, help="print an ETA also for this assumed throughput")
    a = ap.parse_args(argv)
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    out = a.out.rstrip("/") + ("_STUB" if a.stub and not a.out.rstrip("/").endswith("_STUB") else "")
    os.makedirs(out, exist_ok=True)
    files = shard_files(a.src)
    pi, pn = (int(x) for x in a.part.split("/"))
    files = files[pi::pn]
    allowset = set(x.partition("=")[0] for x in a.nc_allow)
    skip = None
    if a.skip_keys_of:
        ks = []
        for f in shard_files(a.skip_keys_of):
            t = pq.read_table(f, columns=[c for c in ("key", "english", "target", "iso") if c in pq.ParquetFile(f).schema_arrow.names]).to_pydict()
            ks.append(np.array(t["key"], np.uint64) if "key" in t else np.array([key_of(e, i, g) for e, i, g in zip(t["english"], t["iso"], t["target"])], np.uint64))
        skip = np.sort(np.concatenate(ks)); print(f"{len(skip):,} keys of {a.skip_keys_of} will be skipped", flush=True)
    capthr = None
    if a.cap_per_lang > 0:
        if a.cap_counts:
            counts = json.load(open(a.cap_counts))["counts"]
        else:
            from common import cap_counts
            counts = cap_counts(a.src, a.exclude_nc, allowset); print("WARNING: no --cap-counts file: counted the --src directories of THIS run (a later run with other --src would draw a different sample)", flush=True)
        capthr = cap_thresholds(counts, a.cap_per_lang)
        print(f"cap {a.cap_per_lang:,} per language: {sum(1 for c in counts.values() if c > a.cap_per_lang)} languages capped, about {sum(min(c, a.cap_per_lang) for c in counts.values()):,} of {sum(counts.values()):,} pairs in the cap", flush=True)
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    if capthr is not None:                                       # progress / ETA base: rows x the share of pairs inside the cap
        total = int(total * sum(min(c, a.cap_per_lang) for c in counts.values()) / max(1, sum(counts.values())))
    print(f"{len(files)} shard files, {total:,} pairs to score ({'STUB' if a.stub else MODEL})", flush=True)
    model = None
    scored = set()
    if a.skip_scored:
        import pyarrow.dataset as pads
        for d in a.skip_scored:
            if os.path.isdir(d) and os.listdir(d):
                scored.update(pads.dataset(d, format="parquet").to_table(columns=["key"])["key"].to_pylist())
        print(f"{len(scored):,} already scored keys will be skipped", flush=True)
    manifest = json.load(open(a.manifest)) if a.manifest else {}
    stage_done, stage_total = collections.Counter(), collections.Counter()
    for v in manifest.values():
        stage_total[v["stage"]] += v["n"]
    recent = []
    model_sha = ""
    if not a.stub:
        import torch
        if a.threads:
            torch.set_num_threads(a.threads)
        from comet import download_model, load_from_checkpoint
        model = load_from_checkpoint(download_model(MODEL))
        model.eval()
        use_cuda = torch.cuda.is_available() and a.device != "cpu"
        if a.half and use_cuda:
            model.half()
        try:
            from huggingface_hub import HfApi
            model_sha = HfApi().model_info(MODEL).sha
        except Exception:
            model_sha = "unknown"
    dev = "stub" if a.stub else ("cuda" if (a.device != "cpu" and __import__("torch").cuda.is_available()) else "cpu")
    dtype = "stub" if a.stub else ("fp16" if (a.half and dev == "cuda") else "fp32")
    scorer = "STUB-NOT-REAL" if a.stub else MODEL
    fq = None
    if not a.stub and not a.no_fast:
        from qe_fast import FastQE
        fq_dev = "cuda" if dev == "cuda" else "cpu"
        model.to(fq_dev)
        fq = FastQE(model, fq_dev, dtype == "fp16", token_budget=a.fast_token_budget, cache_gb=a.fast_cache_gb)
        print(f"fast scoring on {fq_dev} ({dtype}): token budget {a.fast_token_budget}, source cache {a.fast_cache_gb} GB; first chunk is cross-checked against model.predict (Pearson >= {a.fast_min_pearson})", flush=True)
    fast_state = {"checked": False}
    run_tag = format(os.getpid(), "x") + time.strftime("%m%d%H%M%S")
    def comet_ref(srcs, mts):
        data = [{"src": e, "mt": t} for e, t in zip(srcs, mts)]
        res = model.predict(data, batch_size=a.batch, gpus=(a.gpus if dev == "cuda" else 0), progress_bar=False, num_workers=a.workers)
        if not a.stub:
            model.to("cuda" if dev == "cuda" else "cpu")                   # Lightning moves the module back to the CPU after predict
        return np.asarray(res.scores if hasattr(res, "scores") else res, np.float32)

    t0, n_done, n_skip = time.time(), 0, 0
    for f in files:
        sid = shard_id(f)
        pf = pq.ParquetFile(f)
        pend = {"english": [], "target": [], "iso": [], "key": []}
        ci = 0

        def write(buf, ci):
            nonlocal n_done, fq, t0
            path = f"{out}/{sid}-{ci:04d}-{run_tag}.parquet"          # unique per process start: after a restart (--skip-scored drops the finished keys) chunk numbers begin at 0 again and must NOT collide with the files of the earlier run
            if a.stub:
                sc = stub_scores(buf["key"])
            else:
                sc = None
                if fq is not None:
                    sc = fq.score(buf["english"], buf["target"])
                    if not fast_state["checked"]:                          # one-time proof on the first chunk: same scores as comet's own predict
                        tc = time.time(); k = min(384, len(sc)); ref = comet_ref(buf["english"][:k], buf["target"][:k]); t0 += time.time() - tc          # the one-time check is not counted in the rate / elapsed_s
                        corr = float(np.corrcoef(ref, sc[:k])[0, 1]) if k > 2 else 1.0
                        fast_state["checked"] = True
                        json.dump({"pearson_fast_vs_predict": corr, "n": k, "max_abs_diff": float(np.abs(ref - sc[:k]).max()), "min_pearson": a.fast_min_pearson, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, open(f"{out.rstrip(chr(47))}_fast_check.json", "w"))
                        print(f"fast-vs-predict check on {k} pairs: Pearson {corr:.6f}, max |diff| {float(np.abs(ref - sc[:k]).max()):.5f}", flush=True)
                        if corr < a.fast_min_pearson:
                            print(f"!!! fast path below Pearson {a.fast_min_pearson}: FALLING BACK to model.predict for this and all later chunks", flush=True)
                            fq = None; sc = None
                if sc is None:
                    sc = comet_ref(buf["english"], buf["target"])
                assert len(sc) == len(buf["english"])
            tbl = pa.table({"key": pa.array(buf["key"], pa.uint64()), "qe": pa.array(sc, pa.float32()), "scorer": [scorer] * len(sc), "model_sha": [model_sha] * len(sc), "device": [dev] * len(sc),
                            "dtype": [dtype] * len(sc), "ts": [time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())] * len(sc)})
            pq.write_table(tbl, path + ".tmp", compression="zstd")
            os.replace(path + ".tmp", path)
            n_done += len(sc)
            el = time.time() - t0
            rate = n_done / max(el, 1e-9)
            eta = (total - n_done) / max(rate, 1e-9) / 3600
            extra = f"  at the assumed {a.budget_pairs_per_s:.0f}/s: {(total - n_done) / a.budget_pairs_per_s / 3600:.1f} h" if a.budget_pairs_per_s else ""
            if a.progress:
                stage = manifest.get(os.path.basename(f), {}).get("stage", "all")
                stage_done[stage] += len(sc)
                recent.append((time.time(), len(sc)))
                recent[:] = recent[-12:]
                rr = (sum(n for _, n in recent[1:]) / max(recent[-1][0] - recent[0][0], 1e-9)) if len(recent) > 1 else rate
                json.dump({"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": dev, "dtype": dtype, "model_sha": model_sha, "pairs_done_this_run": n_done, "pairs_per_s_run_average": round(rate, 3),
                           "pairs_per_s_recent": round(rr, 3), "elapsed_h": round(el / 3600, 3), "stage_done": dict(stage_done), "stage_total": dict(stage_total), "last_shard": os.path.basename(f)}, open(a.progress + ".tmp", "w"), indent=1)
                os.replace(a.progress + ".tmp", a.progress)
            print(f"{sid} chunk {ci}: scored {n_done:,}/{total:,}  {rate:.1f} pairs/s  elapsed {el / 60:.1f} min  ETA {eta:.2f} h{extra}", flush=True)

        pcols = [c for c in pf.schema_arrow.names if c in ("english", "target", "iso", "key", "licence", "source_dataset")]
        for rb in pf.iter_batches(batch_size=50_000, columns=pcols):
            while a.max_load and os.getloadavg()[0] > a.max_load:
                print(f"load {os.getloadavg()[0]:.1f} > {a.max_load}: backing off 120 s", flush=True); time.sleep(120)
            d = rb.to_pydict()
            keys = d["key"] if "key" in d else None
            licc = d.get("licence")
            for i in range(len(d["english"])):
                en, tg, iso = d["english"][i], d["target"][i], d["iso"][i]
                if not en or not tg or (a.iso and iso not in a.iso):
                    n_skip += 1
                    continue
                if a.exclude_nc and licc is not None and licc[i] == "non_commercial" and (d.get("source_dataset") or [""] * len(licc))[i].split(":")[0] not in allowset:
                    n_skip += 1
                    continue
                kk = keys[i] if keys else key_of(en, iso, tg)
                if capthr is not None and not in_cap(kk, iso, capthr):
                    n_skip += 1
                    continue
                if kk in scored:
                    n_skip += 1
                    continue
                if skip is not None:
                    j = np.searchsorted(skip, np.uint64(kk))
                    if j < len(skip) and skip[j] == kk:
                        n_skip += 1
                        continue
                pend["english"].append(en); pend["target"].append(tg); pend["iso"].append(iso); pend["key"].append(kk)
                if len(pend["english"]) >= a.chunk or (a.max_pairs and n_done + len(pend["english"]) >= a.max_pairs):
                    write(pend, ci); ci += 1
                    pend = {k: [] for k in pend}
                    if a.max_pairs and n_done >= a.max_pairs:
                        print("max-pairs reached"); return
        if pend["english"]:
            write(pend, ci)
    print(f"DONE {n_done:,} pairs scored, {n_skip:,} skipped, {(time.time() - t0) / 60:.1f} min, {n_done / max(time.time() - t0, 1e-9):.1f} pairs/s elapsed_s={time.time() - t0:.2f} (model load excluded)", flush=True)
    if fq is not None:
        print("fast scoring:", fq.summary(), flush=True)


if __name__ == "__main__":
    main()
