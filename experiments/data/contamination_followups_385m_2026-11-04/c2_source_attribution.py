"""E-07(c) follow-up: which source_dataset of the A0 pair data carries the AfriScience-MT train sentences that c1 found (exact normalised sentence, either column)?
Reads c_afriscience_train_exposed_rows.csv to choose the rows; re-reads the AfriScience train parquet for their text; scans the pair parquets (source_dataset column) with a set lookup.
Usage: python c2_source_attribution.py DL_DIR OUT_DIR NPROC   (no hash seed needed: exact normalised-string membership)"""
import os, sys, csv, glob, json, re, unicodedata, collections, multiprocessing as mp
import pyarrow.parquet as pq
DL, OUT, NP = sys.argv[1], sys.argv[2], int(sys.argv[3])
SPLIT = re.compile(r"(?<=[.!?])\s+")
NON = re.compile(r"[\W_]+", re.UNICODE)
def norm(t): return NON.sub(" ", unicodedata.normalize("NFKC", t or "").casefold()).strip()
def targets():
    rows = {r["iso"] + "_" + r["row_id"]: r for r in csv.DictReader(open(f"{OUT}/c_afriscience_train_exposed_rows.csv"))}
    t = pq.read_table(f"{DL}/afriscience_clean_train/afriscience_clean_train.parquet", columns=["paper_id", "sentence_id", "target_lang", "source", "target"]).to_pylist()
    want = {}
    for k, r in enumerate(t):
        rid = f"{r['paper_id']}_{r['sentence_id']}_{k}"; key = r["target_lang"] + "_" + rid
        x = rows.get(key)
        if not x: continue
        if x["ref_exact"] == "1": want.setdefault(norm(r["target"]), []).append(("ref", r["target_lang"], rid))
        if x["en_exact"] == "1": want.setdefault(norm(r["source"]), []).append(("en", r["target_lang"], rid))
    return want
W = None
def init():
    global W; W = targets()
def work(task):
    f, wid, NPR = task; out = collections.Counter(); ex = []
    pf = pq.ParquetFile(f)
    for bi, b in enumerate(pf.iter_batches(batch_size=100_000, columns=["english", "target", "iso", "source_dataset"] if "specialist" in f or "data/" in f else ["english", "target", "iso"])):
        if bi % NPR != wid: continue
        d = b.to_pydict(); sd = d.get("source_dataset") or [""] * len(d["english"])
        for e, t, i, s in zip(d["english"], d["target"], d["iso"], sd):
            for col, x in (("en_col", e), ("tgt_col", t)):
                if x and len(x) < 20000:
                    found = set()
                    for sent in SPLIT.split(x):
                        n = norm(sent)
                        if n in W: found.add(n)
                    n = norm(x)
                    if n in W: found.add(n)
                    for n in found:
                        for kind, lang, rid in W[n]:
                            out[(os.path.basename(os.path.dirname(f)), s, i, col, kind, lang)] += 1
    return out
if __name__ == "__main__":
    files = sorted(glob.glob(f"{DL}/data/pairs-*.parquet")) + sorted(glob.glob(f"{DL}/specialist/pairs-*.parquet"))
    tasks = [(f, w, NP) for f in files for w in range(NP)]
    tot = collections.Counter()
    with mp.get_context("spawn").Pool(NP, initializer=init) as pool:
        for o in pool.imap_unordered(work, tasks): tot.update(o)
    with open(f"{OUT}/c_ref_exact_source_attribution.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["pair_dir", "source_dataset", "pair_iso", "column", "afriscience_side", "afriscience_lang", "n_pair_rows"])
        for k, v in sorted(tot.items(), key=lambda x: -x[1]): w.writerow(list(k) + [v])
    print("distinct (source,iso,col) combos:", len(tot), "total pair rows:", sum(tot.values()))
    by = collections.Counter()
    for (d, s, i, col, kind, lang), v in tot.items(): by[(kind, s)] += v
    for k, v in by.most_common(15): print(k, v)
