"""E-07(a): fuzzy (paraphrase-tolerant) near-duplicate scan of the benchmark rows against the SFT text that is REACHABLE from the laptop.
Benchmarks: the eight audited benchmarks (FLORES+ devtest, NTREX, MAFAND, SmolSent, SALT, GlotStoryBook, IbomMT, FLORES-MOZ); rows (English + reference + language) are taken from the stored
predictions of the 385M SFT run (research_archive_2026-11-03/eval_385m/out_m385/predictions.jsonl).
Training text scanned (both sides of every pair, every message of instruction chats):
   pairs: <HF_ORG_LEGACY>/sabiyarn-clean-sft-2026-10-28 data/ (A0 main pairs) + specialist/ ; databox_backup/hub/{extra, extra2k, extra2_hold}; instruction chats: databox_backup/instr/*/*.parquet (messages)
Method (stricter than the audit's exact / 8-gram rules, which it does not replace): normalise as the audit (NFKC, casefold, [\\W_]+ -> ' '); candidates = training text sharing >= 25% of a benchmark string's word
trigrams (strings of >= 4 tokens; trigrams occurring in > 40 benchmark strings are ignored); classify each candidate: run = the normalised benchmark string occurs as a contiguous token run in the text (the audit's exact rule);
near = character 5-gram Jaccard >= 0.5 (not a run); fuzzy = >= 80% of the benchmark string's word trigrams occur in the text (not a run, Jaccard < 0.5).
Applied to the benchmark English (kind E) and to the benchmark reference text (kind R, target-side exposure).
Usage:  PYTHONHASHSEED=0 python a1_fuzzy_scan.py OUT_DIR PAIRS_DL_DIR NPROC"""
import os, sys, json, glob, re, time, unicodedata, collections, multiprocessing as mp
import pyarrow.parquet as pq
assert os.environ.get("PYTHONHASHSEED") == "0"
ROOT = "/path/to/repo"; ARC = f"{ROOT}/research_archive_2026-11-03"
OUT, DL, NP = sys.argv[1], sys.argv[2], int(sys.argv[3]); os.makedirs(OUT, exist_ok=True)
NON = re.compile(r"[\W_]+", re.UNICODE)
def norm(t): return NON.sub(" ", unicodedata.normalize("NFKC", t or "").casefold()).strip()
def c5(s):
    s = " " + s + " "
    return {s[i:i + 5] for i in range(len(s) - 4)}
G = {}
def build():
    rows = {}
    for l in open(f"{ARC}/eval_385m/out_m385/predictions.jsonl"):
        d = json.loads(l)
        k = d["benchmark"]
        rows.setdefault(("E", k, norm(d["english"])), set()).add(d["iso"])
        rows.setdefault(("R", k, norm(d["reference"])), set()).add(d["iso"])
    strings = []; tri = collections.defaultdict(list)
    for (kind, bench, s), isos in rows.items():
        toks = s.split()
        if len(toks) < 4: continue
        i = len(strings); strings.append((kind, bench, s, len(toks), sorted(isos)[:3], None))
        for g in {hash(tuple(toks[j:j + 3])) for j in range(len(toks) - 2)}: tri[g].append(i)
    tri = {g: v for g, v in tri.items() if len(v) <= 40}
    ntri = [len({hash(tuple(s[2].split()[j:j + 3])) for j in range(s[3] - 2)}) for s in strings]
    return strings, tri, ntri
def init():
    G["s"], G["t"], G["n"] = build()
    G["tg"] = lambda i: list({hash(tuple(G["s"][i][2].split()[j:j + 3])) for j in range(G["s"][i][3] - 2)})
def scan_text(text, src, hits, seen):
    s = norm(text)
    toks = s.split()
    if len(toks) < 4: return
    if s in seen: return
    seen.add(s)
    cnt = collections.Counter()
    t = G["t"]
    for j in range(len(toks) - 2):
        v = t.get(hash((toks[j], toks[j + 1], toks[j + 2])))
        if v:
            for i in v: cnt[i] += 1
    if not cnt: return
    tset = None; tc = None
    for i, c in cnt.items():
        if c < 0.25 * G["n"][i]: continue
        bs = G["s"][i][2]
        if " " + bs + " " in " " + s + " ":
            hits.append((i, src, 1.0, 1.0, len(toks), "run")); continue                    # contiguous token run = the audit's exact rule
        if tset is None: tset = {hash((toks[j], toks[j + 1], toks[j + 2])) for j in range(len(toks) - 2)}
        bt = G["tg"](i)
        T = sum(1 for g in bt if g in tset) / len(bt)
        if tc is None: tc = c5(s)
        b = c5(bs); inter = len(b & tc); jac = inter / (len(b) + len(tc) - inter)
        if jac >= 0.5: hits.append((i, src, round(jac, 3), round(T, 3), len(toks), "near"))
        elif T >= 0.8: hits.append((i, src, round(jac, 3), round(T, 3), len(toks), "fuzzy"))
def work(task):
    kind, f, wid, W = task
    hits = []; seen = set(); n = 0
    pf = pq.ParquetFile(f)
    if kind == "pairs":
        for bi, b in enumerate(pf.iter_batches(batch_size=100_000, columns=["english", "target"])):
            if bi % W != wid: continue
            d = b.to_pydict(); n += len(d["english"])
            for e, t in zip(d["english"], d["target"]):
                scan_text(e, "pair_en_col", hits, seen); scan_text(t, "pair_tgt_col", hits, seen)
    else:
        for bi, b in enumerate(pf.iter_batches(batch_size=20_000, columns=["messages"])):
            if bi % W != wid: continue
            for m in b.column("messages").to_pylist():
                n += 1
                try: msgs = json.loads(m)
                except Exception: continue
                for x in msgs:
                    c = x.get("content") if isinstance(x, dict) else None
                    if isinstance(c, str): scan_text(c, "instr", hits, seen)
    return kind, os.path.relpath(f, ROOT) if f.startswith(ROOT) else f.split("dl_a0pairs/")[-1], n, hits
if __name__ == "__main__":
    t0 = time.time()
    pairs = sorted(glob.glob(f"{DL}/data/pairs-*.parquet")) + sorted(glob.glob(f"{DL}/specialist/pairs-*.parquet"))
    pairs += sorted(glob.glob(f"{ARC}/databox_backup/hub/extra/pairs-*.parquet")) + sorted(glob.glob(f"{ARC}/databox_backup/hub/extra2k/**/pairs-*.parquet", recursive=True)) + sorted(glob.glob(f"{ARC}/databox_backup/hub/extra2_hold/**/pairs-*.parquet", recursive=True))
    instr = sorted(glob.glob(f"{ARC}/databox_backup/instr/*/*.parquet"))
    tasks = [("pairs", f, w, NP) for f in pairs if pq.ParquetFile(f).metadata.num_rows > 0 for w in range(NP)] + [("instr", f, w, NP) for f in instr for w in range(NP)]
    strings, _, _ = build()
    allhits = []; nrows = collections.Counter()
    with mp.get_context("spawn").Pool(NP, initializer=init) as pool:
        for kind, name, n, hits in pool.imap_unordered(work, tasks, chunksize=1):
            nrows[(kind, name.split("/")[-2] if "/" in name else name)] += n; allhits.extend((name,) + h for h in hits)
    rank = {"run": 3, "near": 2, "fuzzy": 1}
    best = {}
    for name, i, src, jac, tcov, ntok, how in allhits:
        if i not in best or (rank[how], jac, tcov) > (rank[best[i][5]], best[i][3], best[i][4]): best[i] = (name, src, ntok, jac, tcov, how)
    out_rows = []
    for i, (name, src, ntok, jac, tcov, how) in best.items():
        kind, bench, st, nt, isos, _ = strings[i]
        out_rows.append(dict(sid=i, kind=kind, benchmark=bench, n_tok=nt, isos=",".join(isos), how=how, best_jaccard=jac, best_trigram_cover=tcov, where=f"{name}:{src}", hit_text_tokens=ntok))
    import csv
    keys = ["sid", "kind", "benchmark", "n_tok", "isos", "how", "best_jaccard", "best_trigram_cover", "where", "hit_text_tokens"]
    with open(f"{OUT}/a_fuzzy_hits_per_benchmark_string.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in sorted(out_rows, key=lambda r: (r["kind"], r["benchmark"], -rank[r["how"]], -r["best_jaccard"]))]
    ndist = collections.Counter(); tot = collections.Counter()
    for (kind, bench, st, nt, isos, _) in strings: ndist[(kind, bench)] += 1
    for r in out_rows:
        k = (r["kind"], r["benchmark"]); tot[(k, "any")] += 1; tot[(k, r["how"])] += 1; tot[(k, "ge8tok_any")] += int(r["n_tok"] >= 8); tot[(k, "ge8tok_" + r["how"])] += int(r["n_tok"] >= 8)
    summ = [dict(kind=k[0], benchmark=k[1], distinct_strings_ge4tok=n, any=tot[(k, "any")], contiguous_run=tot[(k, "run")], near_jaccard_ge_0_5=tot[(k, "near")], fuzzy_trigram_cover_ge_0_8=tot[(k, "fuzzy")],
                 ge8tok_any=tot[(k, "ge8tok_any")], ge8tok_run=tot[(k, "ge8tok_run")], ge8tok_near=tot[(k, "ge8tok_near")], ge8tok_fuzzy=tot[(k, "ge8tok_fuzzy")]) for k, n in sorted(ndist.items())]
    with open(f"{OUT}/a_fuzzy_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); [w.writerow(r) for r in summ]
    cov = collections.Counter()
    for (kind, name), n in nrows.items(): cov[kind] += n
    json.dump(dict(rows_scanned=dict(cov), files=len(pairs) + len(instr), seconds=round(time.time() - t0), benchmark_strings_ge4tok=len(strings)), open(f"{OUT}/a_fuzzy_meta.json", "w"), indent=1)
    print(json.dumps(dict(rows_scanned=dict(cov), seconds=round(time.time() - t0))), flush=True)
    for r in summ: print(r)
