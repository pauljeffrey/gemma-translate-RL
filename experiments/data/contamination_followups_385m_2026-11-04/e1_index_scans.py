"""E-07 (b)(e) index scans with the repo's own checker (data_pipeline/sft_clean/s3_common.Checker = decon.py rules R1 exact sentence, R2 contained sentence, R3 shared word 8-gram,
two normal forms, protected-2026-10-28.3, WITHOUT the FLORES+ dev train-half exemption, as scan_long_pool.py).
 task "long":  every row of the long-document RL pool copied from the A100 (research_archive_2026-11-03/long_rl_pool_2026-11-04/long_rl): flagged distinct texts by directory / bin / source; LibreTexts counts.
 task "seven": the 7 unfiltered new-language files (140,000 rows): reproduce the flagged-window list of hit_texts.json (2,279 distinct windows / 2,619 rows).
 task "drawn": every prompt stage A drew (28,120: replayed schedule steps 1-40 + group_stats steps 41-400): distinct texts flagged by the index.
Usage: python e1_index_scans.py TASK[,TASK..] OUT_DIR [NPROC]"""
import os, sys, json, glob, collections, time, importlib.util
import pyarrow.parquet as pq
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = "/path/to/repo"; ARC = f"{ROOT}/research_archive_2026-11-03"
sys.path.insert(0, f"{ROOT}/data_pipeline/sft_clean")
import s3_common as C
tasks = sys.argv[1].split(","); OUT = sys.argv[2]; NP = int(sys.argv[3]) if len(sys.argv) > 3 else 4
IDX = f"{ARC}/databox_backup/work/_hub_protected/protected_index/protected_index.npz"; MAN = f"{ARC}/databox_backup/work/_hub_protected/protected_index/protected_manifest.json"
t0 = time.time(); ix, ex, man = C.load_strict(IDX, MAN)
print(C.now(), "index", man["version"], man["index_file_sha256"][:16], f"{time.time()-t0:.0f}s", flush=True)
chk = C.Checker(ix, None, None, nproc=NP)
def run(texts):
    res = {}; uniq = list(dict.fromkeys(texts)); step = 2000
    for i in range(0, len(uniq), step):
        for t, r in zip(uniq[i:i + step], chk.run(uniq[i:i + step])): res[t] = r
        print(C.now(), f"{min(len(uniq), i+step)}/{len(uniq)} distinct texts, flagged so far {sum(1 for r in res.values() if r[0]|r[1]|r[2])} ({time.time()-t0:.0f}s)", flush=True)
    return res
def hit(r): return bool(r[0] | r[1] | r[2])
summary = {}
if "long" in tasks:
    rows = []
    for f in sorted(glob.glob(f"{ARC}/long_rl_pool_2026-11-04/long_rl/*/*.parquet")):
        d = os.path.basename(os.path.dirname(f)); cols = [c for c in ("english", "iso", "length_bin", "source_dataset", "domain") if c in pq.ParquetFile(f).schema_arrow.names]
        t = pq.read_table(f, columns=cols).to_pydict(); n = len(t["english"])
        for i in range(n): rows.append((d, t["iso"][i] if "iso" in t else "", t["length_bin"][i] if "length_bin" in t else "", t["source_dataset"][i] if "source_dataset" in t else "", t["english"][i]))
    print("long pool rows", len(rows), flush=True)
    res = run([r[4] for r in rows])
    by = collections.defaultdict(lambda: [0, 0]); bysrc = collections.defaultdict(lambda: [0, 0]); sets = collections.Counter(); rules = collections.Counter()
    for d, iso, lb, src, en in rows:
        h = hit(res[en]); by[(d, lb)][0] += 1; by[(d, lb)][1] += int(h); bysrc[(d, src)][0] += 1; bysrc[(d, src)][1] += int(h)
    for en, r in res.items():
        if hit(r):
            for nm in ix.names(r[0] | r[1] | r[2]): sets[nm] += 1
            for rule, m in zip(C.RULES, r[:3]):
                if m: rules[rule] += 1
    import csv
    with open(f"{OUT}/e_long_pool_rescan_by_dir_bin.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dir", "length_bin", "rows", "rows_flagged"]); [w.writerow([k[0], k[1], v[0], v[1]]) for k, v in sorted(by.items())]
    with open(f"{OUT}/e_long_pool_by_dir_source.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dir", "source_dataset", "rows", "rows_flagged"]); [w.writerow([k[0], k[1], v[0], v[1]]) for k, v in sorted(by.items() if False else bysrc.items())]
    summary["long"] = dict(rows=len(rows), distinct_texts=len(res), distinct_flagged=sum(1 for r in res.values() if hit(r)), rows_flagged=sum(v[1] for v in by.values()), flagged_distinct_by_set=dict(sets.most_common()), flagged_distinct_by_rule=dict(rules),
                           libretexts_rows={d: sum(v[0] for (dd, s), v in bysrc.items() if dd == d and "libretexts" in s.lower()) for d in sorted({k[0] for k in bysrc})},
                           index=man["version"])
if "seven" in tasks:
    flagged_ref = set(json.load(open(f"{ARC}/clean_rerun_misc/hit_texts.json")))
    rows = []
    for iso in ("tll", "mfe", "guw", "lub", "crs", "zne", "lue"):
        t = pq.read_table(f"{ARC}/clean_rerun_misc/prompts/unfiltered_new7/{iso}.parquet", columns=["english", "length_bin"]).to_pydict()
        rows += [(iso, b, e) for e, b in zip(t["english"], t["length_bin"])]
    res = run([r[2] for r in rows]); mine = {t for t, r in res.items() if hit(r)}
    nrows_hit = sum(1 for _, _, e in rows if e in mine)
    summary["seven"] = dict(rows=len(rows), distinct_texts=len(res), distinct_flagged_now=len(mine), rows_flagged_now=nrows_hit, distinct_in_hit_texts_json=len(flagged_ref), overlap=len(mine & flagged_ref),
                            only_now=len(mine - flagged_ref), only_json=len(flagged_ref - mine), by_bin_rows_flagged=dict(collections.Counter(b for _, b, e in rows if e in mine)))
if "drawn" in tasks:
    spec = importlib.util.spec_from_file_location("b2", f"{HERE}/b2_replay_stageA_prompts.py"); b2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(b2)
    P41 = b2.build_pool(True); P1 = b2.build_pool(False)
    sch1, _ = b2.cfg_schedule(P1, with_ref=False)
    gs = [json.loads(l) for l in open(f"{ARC}/runs_385m/A/group_stats.jsonl")]
    drawn_old = []   # (step, kind, old index)
    for s in range(1, 41):
        for p in sch1.indices(s - 1, 64): drawn_old.append((s, "no_ref", P1["old_of_new"][p]))
    for x in gs: drawn_old.append((x["step"], "with_ref", P41["old_of_new"][x["prompt"]]))
    def english_for(P, kindname, wanted):
        out = {}
        for (kind, iso0), (st, en_, key) in P["slices"].items():
            ws = sorted(o for o in wanted if st <= o < en_)
            if not ws: continue
            rp = P["rowpos"][(kind, iso0)]
            if kind == "own7": col = pq.read_table(f"{ARC}/clean_rerun_misc/prompts/unfiltered_new7/{iso0}.parquet", columns=["english"]).column("english").to_pylist()
            else: col = pq.read_table(f"{b2.SPD}/{kind}__{iso0}.parquet").column("english").to_pylist()
            for o in ws: out[o] = col[rp[o - st]].strip()
        return out
    e1 = english_for(P1, "no_ref", {o for s, k, o in drawn_old if k == "no_ref"}); e2 = english_for(P41, "with_ref", {o for s, k, o in drawn_old if k == "with_ref"})
    texts = []; meta = []
    for s, k, o in drawn_old:
        t = (e1 if k == "no_ref" else e2)[o]; texts.append(t); meta.append((s, k, o))
    res = run(texts)
    fl = [(s, k, o) for (s, k, o), t in zip(meta, texts) if hit(res[t])]
    seven_iso = {"tll", "mfe", "guw", "lub", "crs", "zne", "lue"}
    # language of each drawn prompt
    iso_of = lambda s, k, o: (P1 if k == "no_ref" else P41)["iso"][(P1 if k == "no_ref" else P41)["new_of_old"][o]]
    summary["drawn"] = dict(prompts_drawn=len(texts), distinct_texts=len(res), distinct_flagged=sum(1 for r in res.values() if hit(r)), drawn_flagged=len(fl),
                            drawn_flagged_by_iso=dict(collections.Counter(iso_of(*x) for x in fl)), drawn_flagged_steps=[(s, k) for s, k, o in fl][:50])
json.dump(summary, open(f"{OUT}/e_index_scans_{'_'.join(tasks)}.json", "w"), indent=1)
print(json.dumps(summary, indent=1)[:4000])
chk.close()
