#!/usr/bin/env python3
"""Aya-collection ALIGNED rows as instruction data (owner decision 2026-10-31, decision 4): from the African splits of CohereLabs/aya_collection_language_split keep only the rows that sit in a
fully aligned 100k-row block (row k of the split has the same (dataset_name, id) as row k of the English split; see intake2_aya.py), use the African side only (inputs -> user, targets ->
assistant). The parallel en->X pairs stay out (hub/extra2/aya_aligned_QUARANTINE_nllb). The translated subsets are NLLB-3.3B output: rows are flagged synthetic=true and licence_raw says so.
Same cleaning / strict protected filter / dedupe / shard format as intake2_instr.py. At most --cap rows per language (deterministic hash sample of the aligned rows).

    python intake2_aya_instr.py [--cap 100000] [--workers 12]   -> /workspace/work/instr/aya_collection_aligned/part-NNNN.parquet
"""
import argparse, collections, json, multiprocessing as mp, os, shutil, sys, time
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import xxhash
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import intake2_instr as M
import intake2_aya as A

LANGS = {"afrikaans": "afr", "amharic": "amh", "central_kanuri": "knc", "hausa": "hau", "igbo": "ibo", "northern_sotho": "nso", "plateau_malagasy": "mlg", "shona": "sna", "somali": "som",
         "southern_sotho": "sot", "swahili": "swh", "xhosa": "xho"}
NAME = "aya_collection_aligned"
EN_KEYS = None


def h(ds, i):
    return np.array([xxhash.xxh64_intdigest(f"{d}\x01{k}".encode()) for d, k in zip(ds, i)], np.uint64)


def lang_job(args):
    lang, iso, cap = args
    try:
        files = A.train_files(lang)
        ds, ids = A.read_keys(files, 4_000_000)
        n = min(len(ds), len(EN_KEYS))
        eq = h(ds[:n], ids[:n]) == EN_KEYS[:n]
        full = np.zeros(n, bool)
        for s in range(0, n, A.BLOCK):
            blk = eq[s:s + A.BLOCK]
            if len(blk) == A.BLOCK and blk.mean() >= 0.999:
                full[s:s + A.BLOCK] = True
        pos = np.nonzero(full & eq)[0]
        if len(pos) > cap:                                         # deterministic hash sample of the aligned rows
            hv = np.array([xxhash.xxh64_intdigest(f"{lang}:{p}".encode()) for p in pos], np.uint64)
            pos = np.sort(pos[np.argsort(hv, kind="stable")[:cap]])
        want = set(pos.tolist())
        src = M.SOURCES[NAME]
        stats, recs, pending, base = collections.Counter(), [], [], 0
        stats["aligned_rows_in_full_blocks"] = int((full & eq).sum()); stats["selected"] = len(pos)
        wdir = os.path.join(M.TMP, f"{NAME}_{lang}"); os.makedirs(wdir, exist_ok=True)

        def flush():
            if not pending:
                return
            texts = [t for r in pending for t in r["_texts"]]
            hits = M.protected_hits(_IX, texts); k = 0
            for r in pending:
                m = len(r["_texts"]); hh = next((x for x in hits[k:k + m] if x), None); k += m
                if hh:
                    stats["reject_protected_" + hh] += 1; continue
                del r["_texts"]; recs.append(r)
            pending.clear()
        seen = set()
        for p in files:
            local = M.fetch(A.REPO, p, wdir)
            nrows = pq.ParquetFile(local).metadata.num_rows
            if base > max(want, default=-1):
                os.remove(local); base += nrows; continue
            for i, r in enumerate(M.iter_file_rows(local, cols=["inputs", "targets", "dataset_name", "id"])):
                g = base + i
                if g not in want:
                    continue
                rec, why = M.convert_row(src, dict(r, language_iso=iso), f"{lang}/{os.path.basename(p)}", g)
                if rec is None:
                    stats["reject_" + why] += 1; continue
                if rec["dkey"] in seen:
                    stats["reject_dup_in_file"] += 1; continue
                seen.add(rec["dkey"]); pending.append(rec)
                if len(pending) >= M.BATCH:
                    flush()
            os.remove(local); base += nrows
        flush()
        od = os.path.join(M.CAND, NAME); os.makedirs(od, exist_ok=True)
        if recs:
            pq.write_table(pa.Table.from_pylist(recs, schema=M.SCHEMA), os.path.join(od, f"{lang}.parquet"), compression="zstd")
        stats["kept_candidates"] = len(recs)
        shutil.rmtree(wdir, ignore_errors=True)
        return lang, dict(stats), None
    except Exception as e:
        import traceback
        return lang, {}, traceback.format_exc()[-600:]


_IX = None


def main():
    global EN_KEYS, _IX
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", type=int, default=100_000); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    M.add(NAME, A.REPO, lambda paths: [], M.S_qa("inputs", "targets"), lambda p, r: r["language_iso"], "commercial_ok",
          "apache-2.0 (Aya collection card). Translated subsets are NLLB-3.3B machine translations; only rows of fully aligned 100k-blocks (same dataset_name+id as the English split) are used; accepted by the owner 2026-10-31",
          synthetic=True, task="instruction", note="African side of the aligned blocks only; hash-sampled to --cap rows per language", order=5)
    _IX = M.load_index()
    en_files = A.train_files("english")[:3]
    ds, ids = A.read_keys(en_files, 4_000_000)
    EN_KEYS = h(ds, ids)
    print("english keys", len(EN_KEYS), flush=True)
    shutil.rmtree(os.path.join(M.CAND, NAME), ignore_errors=True)
    t0 = time.time(); tot = {}
    with mp.get_context("fork").Pool(min(a.workers, len(LANGS))) as pool:
        for lang, st, err in pool.imap_unordered(lang_job, [(l, i, a.cap) for l, i in LANGS.items()]):
            print(lang, err or st, flush=True); tot[lang] = err or st
            M.progress(f"{NAME}: {lang} {st.get('kept_candidates', 'ERR')}")
    mst, kept, capped = M.merge_source(NAME)
    shutil.rmtree(os.path.join(M.CAND, NAME), ignore_errors=True)
    json.dump(dict(source=NAME, repo=A.REPO, seconds=round(time.time() - t0), per_language=tot, merge=mst, kept_by_lang=kept, licence="commercial_ok", synthetic=True, cap=a.cap),
              open(os.path.join(M.STATE, NAME + ".report.json"), "w"), indent=1)
    open(os.path.join(M.STATE, NAME + ".done"), "w").write(time.strftime("%F %T"))
    M.progress(f"{NAME}: merged kept={mst.get('kept', 0)} langs={len(kept)}")
    print(mst, kept)


if __name__ == "__main__":
    main()
