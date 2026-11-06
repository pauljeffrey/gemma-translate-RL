"""E-07(c): overlap of the AfriScience-MT TRAIN pairs (clean train, 34,007 rows, 6 languages) with the clean A0 translation pair text that is reachable from the laptop
(<HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28: data/pairs-*.parquet 23.4 M pairs + specialist 5.1 M pairs + data_devhalf; a PRE-gate superset of A0's translation pairs).
Rules, normalisation and hash scheme are those of the contamination audit (gemma_mt.eval.contamination of the run's tree: NFKC, casefold, [\\W_]+ -> ' ';
exact = sentence or contiguous token run, sentence = whole sentence, near = >= 50% of the 8-grams, pair = exact and the reference in the other side of the same pair;
8-gram sets via hash(tuple) with PYTHONHASHSEED=0).  Both sides of every pair are scanned (flipped pairs), as in CONT section 2.
Usage:  PYTHONHASHSEED=0 python c1_scan_afriscience_vs_a0_pairs.py DL_DIR OUT_DIR [NPROC]
Index 1 = AfriScience English (source) with the target as reference; index 2 = the AfriScience African text (target) as 'English' with the source as reference (target-side exposure)."""
import os, sys, json, glob, time, collections, multiprocessing as mp
import pyarrow.parquet as pq
ROOT = "/path/to/repo"
sys.path.insert(0, f"{ROOT}/research_archive_2026-11-03/clean_rerun_misc/tree/gemma_mt")
from gemma_mt.eval import contamination as C   # stdlib-only module of the audited tree
assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"
DL = sys.argv[1]; OUT = sys.argv[2]; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 6
SET = sys.argv[4] if len(sys.argv) > 4 else "a0"       # "a0" = Hub clean pair data (data + specialist + devhalf); "stageB" = databox_backup/hub/{extra, extra2k, extra2_hold}
TAG = "" if SET == "a0" else "_" + SET
os.makedirs(OUT, exist_ok=True)

def load_rows():
    t = pq.read_table(f"{DL}/afriscience_clean_train/afriscience_clean_train.parquet", columns=["paper_id", "sentence_id", "target_lang", "source", "target"]).to_pylist()
    rows1, rows2 = [], []
    for k, r in enumerate(t):
        rid = f"{r['paper_id']}_{r['sentence_id']}_{k}"
        rows1.append(dict(english=r["source"], reference=r["target"], iso=r["target_lang"], id=rid))
        rows2.append(dict(english=r["target"], reference=r["source"], iso=r["target_lang"], id=rid))
    # positive control: 500 pairs of data_devhalf (FLORES+ dev TRAIN half, in the A0 pair data on purpose); iso label "ctl-<iso>" keeps them out of the AfriScience rows
    c = pq.ParquetFile(f"{DL}/data_devhalf/pairs-0000.parquet").read_row_group(0, columns=["english", "target", "iso"]).slice(0, 500).to_pylist()
    for k, r in enumerate(c):
        rows1.append(dict(english=r["english"], reference=r["target"], iso=r["iso"], id=f"ctl{k}"))
        rows2.append(dict(english=r["target"], reference=r["english"], iso=r["iso"], id=f"ctl{k}"))
    return t, rows1, rows2
G = {}
def init():
    t, rows1, rows2 = load_rows()
    G["i1"] = C.BenchIndex(rows1); G["i2"] = C.BenchIndex(rows2)

def work(task):
    f, wid, W = task
    pf = pq.ParquetFile(f); res1, res2 = C.ScanResult(), C.ScanResult(); n = 0
    for bi, b in enumerate(pf.iter_batches(batch_size=100_000, columns=["english", "target", "iso"])):
        if bi % W != wid: continue
        d = b.to_pydict(); n += len(d["english"])
        w = list(zip(d["english"], d["target"], d["iso"]))
        wf = [(t, e, i) for e, t, i in w]
        for res, idx in ((res1, G["i1"]), (res2, G["i2"])):
            res.merge(C.scan_windows(idx, w)); res.merge(C.scan_windows(idx, wf))
    return os.path.relpath(f, ROOT) if f.startswith(ROOT) else os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f), n, res1, res2

if __name__ == "__main__":
    t0 = time.time()
    if SET == "a0":
        files = sorted(glob.glob(f"{DL}/data/pairs-*.parquet")) + sorted(glob.glob(f"{DL}/specialist/pairs-*.parquet")) + sorted(glob.glob(f"{DL}/data_devhalf/pairs-*.parquet"))
    else:
        H = f"{ROOT}/research_archive_2026-11-03/databox_backup/hub"
        files = [f for f in sorted(glob.glob(f"{H}/extra/pairs-*.parquet")) + sorted(glob.glob(f"{H}/extra2k/**/pairs-*.parquet", recursive=True)) + sorted(glob.glob(f"{H}/extra2_hold/**/pairs-*.parquet", recursive=True)) if pq.ParquetFile(f).metadata.num_rows > 0]
    tasks = [(f, w, NP) for f in files for w in range(NP)]
    t, rows1, rows2 = load_rows(); i1 = C.BenchIndex(rows1); i2 = C.BenchIndex(rows2)
    R1, R2 = C.ScanResult(), C.ScanResult(); nrows = collections.Counter()
    with mp.get_context("spawn").Pool(NP, initializer=init) as pool:
        for name, n, r1, r2 in pool.imap_unordered(work, tasks):
            R1.merge(r1); R2.merge(r2); nrows[name] += n
            print(f"[{time.time()-t0:.0f}s] {name} +{n}", flush=True)
    # flags per row
    fl1 = C.flags_for_rows(rows1, i1, R1); fl2 = C.flags_for_rows(rows2, i2, R2)
    # any shared 8-gram (strict R3) for each index
    def any8(idx, R, rows):
        ov = {}
        for i, s in enumerate(idx.texts):
            gs = set(C._grams(s.split(), idx.n)); ov[i] = sum(1 for g in gs if g in R.grams)
        return {C.row_key(r["iso"], r["id"]): bool((idx.ids.get(C.normalise(r["english"])) is not None) and ov[idx.ids[C.normalise(r["english"])]] > 0) for r in rows}
    a1 = any8(i1, R1, rows1); a2 = any8(i2, R2, rows2)
    per = collections.defaultdict(collections.Counter)
    ctl = collections.Counter()
    for r in rows1:
        k = C.row_key(r["iso"], r["id"])
        if str(r["id"]).startswith("ctl"):
            ctl["n"] += 1; ctl["en_exact"] += int(fl1["exact"][k]); ctl["en_pair"] += int(fl1["pair"][k]); ctl["ref_exact"] += int(fl2["exact"][k]); continue
        toks = len(C.normalise(r["english"]).split()); toks2 = len(C.normalise(r["reference"]).split()); d = per[r["iso"]]
        d["n"] += 1
        for nm, fl, a, tk in (("en", fl1, a1, toks), ("ref", fl2, a2, toks2)):
            for f in ("exact", "sentence", "near", "pair"):
                v = fl[f][k]; d[f"{nm}_{f}"] += int(v); d[f"{nm}_{f}_ge3tok"] += int(v and tk >= 3)
            d[f"{nm}_any8"] += int(a[k])
    tot = collections.Counter()
    for iso, d in per.items(): tot.update(d)
    rows_out = [dict(iso=iso, **d) for iso, d in sorted(per.items())] + [dict(iso="TOTAL", **tot)]
    import csv
    keys = list(rows_out[-1].keys())
    with open(f"{OUT}/c_afriscience_train_vs_a0_pairs_by_language{TAG}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k, 0) for k in keys}) for r in rows_out]
    # list of exposed rows (ids/lengths only, no text)
    with open(f"{OUT}/c_afriscience_train_exposed_rows{TAG}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "row_id", "n_tok_en", "n_tok_ref", "en_exact", "en_near", "en_pair", "ref_exact", "ref_near", "en_any8", "ref_any8"])
        for r in rows1:
            k = C.row_key(r["iso"], r["id"])
            if str(r["id"]).startswith("ctl"): continue
            if fl1["exact"][k] or fl1["near"][k] or fl2["exact"][k] or fl2["near"][k] or a1[k] or a2[k]:
                w.writerow([r["iso"], r["id"], len(C.normalise(r["english"]).split()), len(C.normalise(r["reference"]).split()), int(fl1["exact"][k]), int(fl1["near"][k]), int(fl1["pair"][k]), int(fl2["exact"][k]), int(fl2["near"][k]), int(a1[k]), int(a2[k])])
    meta = dict(positive_control_flores_dev_trainhalf=dict(ctl), pair_rows_scanned=dict(nrows), pair_rows_total=sum(nrows.values()), windows_scanned_both_orientations=sum(nrows.values()) * 2, afriscience_rows=len(t),
                distinct_en=len(i1.texts), distinct_ref=len(i2.texts), seconds=round(time.time() - t0), source=((os.environ['HF_ORG_LEGACY'] + "/sabiyarn-clean-sft-2026-10-28 data/ + specialist/ + data_devhalf/") if SET == "a0" else "databox_backup/hub/extra + extra2k (incl. nllb/) + extra2_hold"))
    json.dump(meta, open(f"{OUT}/c_scan_meta{TAG}.json", "w"), indent=1)
    print(json.dumps(meta, indent=1)); print(rows_out[-1])
