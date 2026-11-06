#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""News-style English RL prompt pool, RELEASE 2 (2026-11-03): BBC (release 1 pool) + English Wikinews + Global Voices + Wikivoyage, decontaminated.

Reuses the window logic of build_rl_prompts_news.py (release 1: BBC). New in release 2:
  * three more sources (Hub: izumi-lab/wikinews-en-20230728, alvations/globalvoices-en-es English side, darkpredator/stealth-wikivoyage-en prose paragraphs);
  * an ARTICLE-LEVEL guard on top of the window-level rule: the cleaned full text of every article (all five sources, BBC included) is checked against the protected index
    (same rules R1/R2/R3, strict, no dev-half exemption); an article with ANY hit loses ALL its windows. Reason: FLORES was cut from Wikinews / Wikijunior / Wikivoyage articles and
    MAFAND / NTREX are news; windows next to a benchmark sentence come from the same document.
  * a per-language draw that mixes the sources with fixed shares (shortfalls of a source are filled from BBC).

    python scripts/build_rl_prompts_news_r2.py artflag  --src SRC_R2 --v1-src SRC_V1 --work W2 --index I --manifest M --nproc 6
    python scripts/build_rl_prompts_news_r2.py extract  --src SRC_R2 --work W2
    python scripts/build_rl_prompts_news_r2.py decon    --work W2 --index I --manifest M --nproc 6
    python scripts/build_rl_prompts_news_r2.py draw     --work W2 --v1-work W1 --out OUT
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sqlite3
import sys
import time
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_rl_prompts_news as B  # noqa: E402  (release 1: window logic, constants)

NEW_SRC_LICENCE = {
    "izumi-lab/wikinews-en-20230728": "cc-by-2.5",
    "alvations/globalvoices-en-es": "cc-by-3.0-original-no-hub-licence",
    "darkpredator/stealth-wikivoyage-en": "cc-by-sa-4.0",
}
B.SRC_LICENCE.update(NEW_SRC_LICENCE)
POOL_VERSION = "news_v2_2026-11-03"
SHARE = {"RealTimeData/bbc_news_alltime": 0.30, "permutans/c4-bbc-news": 0.25, "izumi-lab/wikinews-en-20230728": 0.25,
         "alvations/globalvoices-en-es": 0.15, "darkpredator/stealth-wikivoyage-en": 0.05}      # SetFit/bbc-news (22 articles) only appears as filler
NS_DROP = {"Category", "User", "Talk", "Portal", "Template", "Help", "File", "MediaWiki", "Wikinews", "Special", "Draft", "Module", "Wikipedia", "Forum"}
CUT = re.compile(r"(?im)^\s*(have an opinion on this story|this page is archived|sources?\s*:?\s*$|related news|related articles|external links|share this|got a correction|"
                 r"add a comment|comments?\s*$|retrieved from|from wikinews|this article (is|was)|this is a (developing|breaking))")


def now():
    return time.strftime("%H:%M:%S")


# ----------------------------------------------------------------------------------------------------------------------- readers
def read_wikinews(src):
    f = glob.glob(os.path.join(src, "izumi-lab__wikinews-en-20230728", "data", "*.parquet"))[0]
    t = pq.read_table(f, columns=["text", "title"])
    for i, (txt, title) in enumerate(zip(t.column("text").to_pylist(), t.column("title").to_pylist())):
        if ":" in (title or "") and title.split(":", 1)[0] in NS_DROP:
            continue
        m = CUT.search(txt or "")
        txt = (txt[:m.start()] if m else txt) or ""
        yield "izumi-lab/wikinews-en-20230728", "data/train.parquet", i, txt


def read_gv(src):
    f = glob.glob(os.path.join(src, "alvations__globalvoices-en-es", "data", "*.parquet"))[0]
    rows = pq.read_table(f, columns=["en"]).column("en").to_pylist()
    start, cur = 0, []
    for i, s in enumerate(rows):
        s = (s or "").strip()
        if s.endswith("· Global Voices"):                         # the title row opens a new document
            if cur:
                yield "alvations/globalvoices-en-es", "data/train.parquet", start, "\n".join(cur)
            start, cur = i, []
            continue
        cur.append(s)
    if cur:
        yield "alvations/globalvoices-en-es", "data/train.parquet", start, "\n".join(cur)


def read_voyage(src):
    c = sqlite3.connect(os.path.join(src, "darkpredator__stealth-wikivoyage-en", "index.sqlite"))
    cur_a, parts = None, []
    for aid, heading, text in c.execute("select article_id, heading, text from section order by article_id, number"):
        if aid != cur_a:
            if parts:
                yield "darkpredator/stealth-wikivoyage-en", "index.sqlite", cur_a, "\n".join(parts)
            cur_a, parts = aid, []
        for line in (text or "").split("\n"):
            line = line.strip()
            if not line or line.startswith(("-", "•", "*", "☏")) or "☏" in line or "(updated" in line:
                continue
            parts.append(line)
    if parts:
        yield "darkpredator/stealth-wikivoyage-en", "index.sqlite", cur_a, "\n".join(parts)


NEW_READERS = {"wikinews": read_wikinews, "gv": read_gv, "voyage": read_voyage}


def all_new(src):
    for r in NEW_READERS.values():
        yield from r(src)


# ----------------------------------------------------------------------------------------------------------------------- stages
def cmd_artflag(a):
    """Check the cleaned full text of every article of every source; write the flagged article keys."""
    sc = os.path.abspath(os.path.join(HERE, "..", "data_pipeline", "sft_clean"))
    sys.path.insert(0, sc)
    import s3_common as C
    ix, ex, man = C.load_strict(a.index, a.manifest)
    print(now(), "index", man["version"], man["index_file_sha256"][:16], flush=True)
    chk = C.Checker(ix, None, None, nproc=a.nproc)
    os.makedirs(a.work, exist_ok=True)
    flagged = collections.defaultdict(list)
    checked = collections.Counter()
    by_set = collections.Counter()
    by_src_set = collections.defaultdict(collections.Counter)
    n = 0

    def run(batch):
        res = chk.run([b[3] for b in batch])
        for (s, f, r, _t), t in zip(batch, res):
            checked[s] += 1
            m = t[0] | t[1] | t[2]
            if m:
                flagged[s].append([f, r])
                for nm in ix.names(m):
                    by_set[nm] += 1
                    by_src_set[s][nm] += 1

    batch = []

    def feed(gen):
        nonlocal batch, n
        for s, f, r, text in gen:
            paras = B.clean_paragraphs(text)
            if not paras:
                continue
            batch.append((s, f, r, "\n".join(paras)))
            n += 1
            if len(batch) >= 3000:
                run(batch)
                batch = []
                if n % 30000 < 3000:
                    print(now(), n, "articles checked,", sum(len(v) for v in flagged.values()), "flagged", flush=True)

    feed(all_new(a.src))
    feed(B.iter_articles(a.v1_src))
    if batch:
        run(batch)
    chk.close()
    out = dict(index_version=man["version"], index_sha256=man["index_file_sha256"], articles_checked=dict(checked),
               articles_flagged={k: len(v) for k, v in flagged.items()}, flagged_by_set=dict(by_set.most_common()),
               flagged_by_source_set={k: dict(v.most_common()) for k, v in by_src_set.items()}, keys=flagged)
    json.dump(out, open(os.path.join(a.work, "article_flags.json"), "w"))
    print(now(), json.dumps({k: v for k, v in out.items() if k != "keys"}, indent=1), flush=True)


def cmd_extract(a):
    flags = json.load(open(os.path.join(a.work, "article_flags.json")))["keys"]
    bad = {s: {(f, int(r)) for f, r in v} for s, v in flags.items()}

    def it(_src):
        for s, f, r, text in all_new(a.src):
            if (f, r) in bad.get(s, ()):
                continue
            yield s, f, r, text

    B.iter_articles = it
    ns = argparse.Namespace(src=a.src, work=a.work, seed=a.seed, boiler_min=a.boiler_min, limit=0)
    B.cmd_extract(ns)
    os.replace(os.path.join(a.work, "pool_raw.parquet"), os.path.join(a.work, "pool_raw_new.parquet"))
    os.replace(os.path.join(a.work, "extract_stats.json"), os.path.join(a.work, "extract_stats_new.json"))


def cmd_decon(a):
    os.replace(os.path.join(a.work, "pool_raw_new.parquet"), os.path.join(a.work, "pool_raw.parquet"))
    B.cmd_decon(argparse.Namespace(work=a.work, index=a.index, manifest=a.manifest, nproc=a.nproc))
    os.replace(os.path.join(a.work, "pool_clean.parquet"), os.path.join(a.work, "pool_clean_new.parquet"))
    os.replace(os.path.join(a.work, "decon_stats.json"), os.path.join(a.work, "decon_stats_new.json"))
    os.replace(os.path.join(a.work, "pool_raw.parquet"), os.path.join(a.work, "pool_raw_new.parquet"))


def cmd_draw(a):
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
    from gemma_mt.languages import LANGUAGES, SUPPORTED_RL
    isos = sorted(i for i in SUPPORTED_RL if i not in set(a.exclude))
    flags = json.load(open(os.path.join(a.work, "article_flags.json")))["keys"]
    bad = {s: {(f, int(r)) for f, r in v} for s, v in flags.items()}
    v1 = pq.read_table(os.path.join(a.v1_work, "pool_clean.parquet"))
    keep = np.array([(f, r) not in bad.get(s, ()) for s, f, r in zip(v1.column("source_dataset").to_pylist(), v1.column("prov_file").to_pylist(), v1.column("prov_row").to_pylist())])
    removed_v1 = int((~keep).sum())
    v1 = v1.filter(pa.array(keep))
    new = pq.read_table(os.path.join(a.work, "pool_clean_new.parquet"))
    cols = [c for c in v1.column_names if c in new.column_names]
    t = pa.concat_tables([v1.select(cols), new.select(cols)])
    # identical windows (case-folded) across sources
    eng = t.column("english").to_pylist()
    seen, ok = set(), []
    for i, e in enumerate(eng):
        k = e.casefold()
        if k in seen:
            continue
        seen.add(k)
        ok.append(i)
    t = t.take(pa.array(ok))
    print(now(), "pool", t.num_rows, "(v1 windows removed by the article guard:", removed_v1, ", cross-source duplicates dropped:", len(eng) - len(ok), ")", flush=True)
    lb = np.asarray(t.column("length_bin").to_pylist())
    sd = np.asarray(t.column("source_dataset").to_pylist())
    per_bin = dict(zip(B.BINS, a.per_bin))
    avail = {b: {s: np.nonzero((lb == b) & (sd == s))[0] for s in sorted(set(sd))} for b in B.BINS}
    pool_stats = {b: {s: int(len(v)) for s, v in avail[b].items()} for b in B.BINS}
    print(json.dumps(pool_stats, indent=1), flush=True)
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.out_full, exist_ok=True)
    base = {c: t.column(c) for c in ("english", "src_afr_tokens", "length_bin", "n_sentences", "source_dataset", "licence", "prov_file", "prov_row", "prov_en_xxh64", "prov_index")}
    use = collections.Counter()
    comp = collections.defaultdict(collections.Counter)
    for iso in isos:
        rng = np.random.default_rng([a.seed, zlib.crc32(iso.encode())])
        picks_by_bin = {}
        for b in B.BINS:
            need = per_bin[b]
            quota = {s: int(round(need * SHARE.get(s, 0.0))) for s in avail[b]}
            quota[max(SHARE, key=SHARE.get)] += need - sum(quota.values())
            take = {s: min(quota[s], len(avail[b][s])) for s in avail[b]}
            short = need - sum(take.values())
            order = [s for s in sorted(avail[b], key=lambda s: -len(avail[b][s]) + take[s])]
            for s in order:                                          # fill shortfalls from the sources with the most unused windows
                if short <= 0:
                    break
                extra = min(short, len(avail[b][s]) - take[s])
                take[s] += extra
                short -= extra
            assert short == 0, (iso, b, "pool too small")
            ps = [np.sort(rng.choice(avail[b][s], size=take[s], replace=False)) for s in avail[b] if take[s]]
            picks_by_bin[b] = np.concatenate(ps)
            for s in avail[b]:
                comp[b][s] += take[s]
        for out_dir, bins in ((a.out, ("b64", "b256")), (a.out_full, ("b64", "b256", "b512", "b1024"))):
            picks = np.concatenate([picks_by_bin[b] for b in bins])
            picks = picks[rng.permutation(len(picks))] if out_dir == a.out else picks[np.random.default_rng([a.seed, 1, zlib.crc32(iso.encode())]).permutation(len(picks))]
            sub = pa.table({c: col.take(pa.array(picks)) for c, col in base.items()})
            k = len(picks)
            sub = sub.append_column("iso", pa.array([iso] * k, pa.string())).append_column("language", pa.array([LANGUAGES[iso].name] * k, pa.string()))
            sub = sub.append_column("domain", pa.array(["news"] * k, pa.string())).append_column("prov_pool", pa.array([POOL_VERSION] * k, pa.string()))
            sub = sub.append_column("flag_ssa_mte_train", pa.array([False] * k, pa.bool_()))
            pq.write_table(sub.select(list(B.COLS)), os.path.join(out_dir, f"{iso}.parquet"), compression="zstd")
        for p in np.concatenate(list(picks_by_bin.values())):
            use[int(p)] += 1
    dist = {b: int(sum(1 for i in range(t.num_rows) if use[i] > 0 and lb[i] == b)) for b in B.BINS}
    st = dict(pool_version=POOL_VERSION, languages=len(isos), per_bin=per_bin, share=SHARE, windows_available=pool_stats, v1_windows_removed_by_article_guard=removed_v1,
              cross_source_duplicates_dropped=len(eng) - len(ok), distinct_windows_used=dist, rows_drawn_by_bin_source={b: dict(v) for b, v in comp.items()}, seed=a.seed)
    json.dump(st, open(os.path.join(a.work, "draw_stats.json"), "w"), indent=1)
    print(now(), json.dumps(st, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("artflag"); f.add_argument("--src", required=True); f.add_argument("--v1-src", required=True); f.add_argument("--work", required=True)
    f.add_argument("--index", required=True); f.add_argument("--manifest", required=True); f.add_argument("--nproc", type=int, default=4)
    e = sp.add_parser("extract"); e.add_argument("--src", required=True); e.add_argument("--work", required=True); e.add_argument("--seed", type=int, default=1234); e.add_argument("--boiler-min", type=int, default=8)
    d = sp.add_parser("decon"); d.add_argument("--work", required=True); d.add_argument("--index", required=True); d.add_argument("--manifest", required=True); d.add_argument("--nproc", type=int, default=4)
    w = sp.add_parser("draw"); w.add_argument("--work", required=True); w.add_argument("--v1-work", required=True); w.add_argument("--out", required=True); w.add_argument("--out-full", required=True)
    w.add_argument("--seed", type=int, default=4321); w.add_argument("--per-bin", type=int, nargs=4, default=[6000, 6000, 5000, 5000]); w.add_argument("--exclude", nargs="*", default=["arz"])
    a = ap.parse_args()
    {"artflag": cmd_artflag, "extract": cmd_extract, "decon": cmd_decon, "draw": cmd_draw}[a.cmd](a)


if __name__ == "__main__":
    main()
