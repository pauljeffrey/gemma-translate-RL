#!/usr/bin/env python3
"""NLLB mined bitext (OPUS NLLB v1, English <-> African) into chain-input shards, with a per-language cap (owner 2026-10-31: use the mined bitext, ODC-BY per the allenai/nllb card, with special
attribution; low- and mid-resource languages first, high-resource ones later on request).

    python nllb_intake.py --out hub/extra2/nllb --index work/_hub_protected/protected_index/protected_index.npz --manifest work/_hub_protected/protected_index/protected_manifest.json \
        --cap 500000 --skip afr xho zul --workers 14 --tmp work/_nllb_dl

Per language: stream the whole OPUS zip (moses, en-xx.txt.zip), clean with the chain rules (extra_sources.ok), then keep a MONOTONE hash sample of about --cap pairs: a pair is kept when
(key & 0xFFFFFFFF) < ceil(cap * 2^32 / pairs_in_corpus); raising --cap later keeps a superset (only the new keys need QE scoring). Dedupe inside the language, strict decontamination against the
protected index (both sides), shards of 500k rows. The raw zip is deleted after the language is done. Stats per language in <out>/stats/.
"""
import argparse, collections, csv, io, json, math, multiprocessing as mp, os, sys, time, unicodedata, urllib.request, zipfile
import pyarrow as pa, pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import key_of
from extra_sources import ok, norm_ws, Filter, SCHEMA_COLS
from nllb_probe import LANG

SRC = "nllb-mined:OPUS-NLLB-v1"
LIC_RAW = "ODC-BY (allenai/nllb card); bound by the terms of the original web sources; NLLB mined bitext, Meta AI"


def work(job):
    code, url, total, cap, index, manifest, out, tmp = job
    iso = LANG[code]; t0 = time.time(); st = collections.Counter(); removed = collections.Counter()
    os.makedirs(tmp, exist_ok=True); zp = os.path.join(tmp, f"{code}.zip")
    for t in range(4):
        try:
            urllib.request.urlretrieve(url, zp); break
        except Exception:
            time.sleep(10 * (t + 1))
    else:
        return code, iso, {"error": "download"}, {}
    thr = (1 << 32) if total <= cap else math.ceil(cap * (1 << 32) / total)
    filt = Filter(index, manifest); rows = []; seen = set(); buf = []; shard = 0
    os.makedirs(os.path.join(out, iso), exist_ok=True)

    def flush():
        nonlocal buf, shard, rows
        if not buf: return
        res = filt.check(buf)
        for r, h in zip(buf, res):
            if h: st["removed_protected"] += 1; removed[h] += 1
            else: rows.append(r); st["kept"] += 1
        buf = []
        if len(rows) >= 500_000: write_shard()

    def write_shard():
        nonlocal rows, shard
        if not rows: return
        t = {c: [r.get(c) for r in rows] for c in SCHEMA_COLS}
        pq.write_table(pa.table({**t, "key": pa.array(t["key"], pa.uint64()), "aux": pa.array(t["aux"], pa.float32())}), os.path.join(out, iso, f"pairs-{shard:04d}.parquet"), compression="zstd")
        shard += 1; rows = []

    with zipfile.ZipFile(zp) as z:
        names = z.namelist(); n_en = [n for n in names if n.endswith(".en")][0]; n_tg = [n for n in names if n.endswith("." + code)][0]
        with z.open(n_en) as fe, z.open(n_tg) as ft:
            for le, lt in zip(io.TextIOWrapper(fe, encoding="utf-8", errors="replace"), io.TextIOWrapper(ft, encoding="utf-8", errors="replace")):
                st["raw"] += 1
                en, tg = norm_ws(unicodedata.normalize("NFC", le)), norm_ws(unicodedata.normalize("NFC", lt))
                if not ok(en, tg): st["dropped_clean"] += 1; continue
                k = key_of(en, iso, tg)
                if (k & 0xFFFFFFFF) >= thr: continue
                if k in seen: st["dropped_duplicate"] += 1; continue
                seen.add(k); st["sampled"] += 1
                buf.append(dict(english=en, target=tg, iso=iso, source_dataset=SRC, licence="commercial_ok", licence_raw=LIC_RAW, kind="mined", source_family="nllb_mined", flipped=False, key=k, aux=float("nan")))
                if len(buf) >= 5000: flush()
    flush(); write_shard(); os.remove(zp)
    st["seconds"] = round(time.time() - t0)
    return code, iso, dict(st), dict(removed)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--index", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--cap", type=int, default=500_000); ap.add_argument("--skip", nargs="*", default=["afr", "xho", "zul"]); ap.add_argument("--langs", nargs="*"); ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--tmp", default="work/_nllb_dl")
    a = ap.parse_args()
    api = json.load(urllib.request.urlopen("https://opus.nlpl.eu/opusapi/?source=en&corpus=NLLB&preprocessing=moses&version=v1", timeout=120))["corpora"]
    jobs = []
    for c in api:
        o = c["target"] if c["source"] == "en" else c["source"]
        if o in LANG and LANG[o] not in a.skip and (not a.langs or o in a.langs or LANG[o] in a.langs):
            jobs.append((o, c["url"], int(c["alignment_pairs"]), a.cap, a.index, a.manifest, a.out, a.tmp))
    jobs.sort(key=lambda j: -j[2])                      # biggest downloads first
    print(len(jobs), "languages, cap", a.cap, flush=True)
    os.makedirs(os.path.join(a.out, "stats"), exist_ok=True); allst = {}; allrm = collections.Counter()
    with mp.Pool(a.workers) as pool:
        for code, iso, st, rm in pool.imap_unordered(work, jobs):
            allst[iso] = st; print(code, iso, st, flush=True)
            for h, n in rm.items(): allrm[(iso, h)] += n
            json.dump({"cap": a.cap, "source": SRC, "licence_raw": LIC_RAW, "languages": allst}, open(os.path.join(a.out, "stats", "summary.json"), "w"), indent=1)
    with open(os.path.join(a.out, "stats", "removed_by_protected_set.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iso", "first_rule_and_side", "rows"])
        for (iso, h), n in sorted(allrm.items(), key=lambda kv: -kv[1]): w.writerow([iso, h, n])
    print("done", sum(s.get("kept", 0) for s in allst.values()), "pairs kept")


if __name__ == "__main__":
    main()
