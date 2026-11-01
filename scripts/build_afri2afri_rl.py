#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inter-African RL prompt builder (D-46): monolingual SOURCE windows + target tag, in the `rl_prompts_v2` format family. CPU only.

    python scripts/build_afri2afri_rl.py pool-docs  --shards /workspace/tokdata/shards --out OUT --langs amh hau ibo pcm tir yor
    python scripts/build_afri2afri_rl.py pool-pairs --sft OUT_SFT --pairs /workspace/sft_build/gemma/pairs_final --out OUT
    python scripts/build_afri2afri_rl.py assemble   --sft OUT_SFT --out OUT --bench ... --calibration ...

pool-docs   windows of consecutive whole sentences from real monolingual text shards (fineweb-2, ODC-By); used where such shards exist
pool-pairs  windows from the African-side texts of the en->X pairs that are NOT in the inter-African SFT selection (neither the text nor the English sentence
            of any selected group): held out from the SFT by construction; every one of the 77 languages has some
assemble    leak filter (benchmarks, the SFT texts), GlotLID guard, de-duplication, then the direction plan: for every (source language, bin) the windows are
            spread over the target languages with quotas proportional to the SFT allocation of the pair (UniMax) and written as one parquet per TARGET language.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import Dict, Iterator, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from gemma_mt.afri2afri import rl_prompts as R, windows as W      # noqa: E402
from gemma_mt.languages import ALL_ISOS, LANGUAGES                 # noqa: E402

SFT77 = tuple(i for i in ALL_ISOS if i not in ("nde", "fat", "knc", "gaa", "men", "luy", "tzm", "ful"))
_TK = {}


def afro_tokenizer(name: str = "Davlan/afro-xlmr-large-76L"):
    if "tk" not in _TK:
        from transformers import AutoTokenizer

        _TK["tk"] = AutoTokenizer.from_pretrained(name)
    return _TK["tk"]


def count_afro(texts: List[str]) -> List[int]:
    return [len(x) for x in afro_tokenizer()(texts, add_special_tokens=True, truncation=False)["input_ids"]]


def iter_line_blocks(path: str, lines_per_block: int, max_lines: int) -> Iterator[str]:
    """Blocks of consecutive lines of a .txt.zst shard (the shards keep paragraphs one per line and have no reliable document separator)."""
    p = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
    buf, n = [], 0
    try:
        for raw in p.stdout:
            n += 1
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.strip():
                buf.append(line)
            if len(buf) >= lines_per_block:
                yield "\n".join(buf)
                buf = []
            if n >= max_lines:
                break
        if buf:
            yield "\n".join(buf)
    finally:
        p.kill()


def _doc_worker(args):
    lang, path, seed, lines_per_block, max_lines, source = args
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    rows: Dict[str, list] = {k: [] for k in ("source", "src_afr_tokens", "length_bin", "n_sentences")}
    for block in iter_line_blocks(path, lines_per_block, max_lines):
        sents = R.clean_doc(block)
        if len(sents) < 3:
            continue
        for w in R.windows_from_doc(sents, rng, count_afro, chars_per_token=4.0):
            for k in rows:
                rows[k].append(w[k])
    return lang, source, os.path.basename(path), rows


def cmd_pool_docs(a) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from multiprocessing import Pool

    os.makedirs(os.path.join(a.out, "pool_docs"), exist_ok=True)
    jobs = []
    for lang in a.langs:
        files = sorted(glob.glob(os.path.join(a.shards, lang, "*fineweb-2*.txt.zst")))[: a.max_files]
        for i, f in enumerate(files):
            jobs.append((lang, f, a.seed + 7 * i + hash(lang) % 1000, a.lines_per_block, a.max_lines, "HuggingFaceFW/fineweb-2:" + lang))
    t0 = time.time()
    acc: Dict[str, Dict[str, list]] = {}
    with Pool(a.workers) as pool:
        for lang, source, fname, rows in pool.imap_unordered(_doc_worker, jobs):
            d = acc.setdefault(lang, {"source": [], "src_afr_tokens": [], "length_bin": [], "n_sentences": [], "source_dataset": []})
            for k in rows:
                d[k].extend(rows[k])
            d["source_dataset"].extend([source] * len(rows["source"]))
            print(f"{lang} {fname}: {len(rows['source']):,} windows {time.time() - t0:.0f}s", flush=True)
    for lang, d in acc.items():
        d["licence"] = ["odc-by"] * len(d["source"])
        pq.write_table(pa.table(d), os.path.join(a.out, "pool_docs", f"{lang}.parquet"), compression="zstd")
    return 0



# ----------------------------------------------------------------------------------------------------------------------- pool-web
WEB_LICENCE = {"fineweb-2": ("HuggingFaceFW/fineweb-2", "odc-by"), "glotcc": ("cis-lmu/GlotCC-V1", "cc0"), "hplt2": ("HPLT/HPLT2.0_cleaned", "cc0"),
               "wikipedia": ("wikimedia/wikipedia", "share_alike"), "c4": ("allenai/c4", "odc-by")}
WEB_ORDER = ("fineweb-2", "glotcc", "hplt2", "wikipedia", "c4")


def _iter_web_docs(src: str, files, token: str):
    """Yield document texts of one (language, source) by STREAMING the remote files (HfFileSystem range reads: no full download)."""
    import gzip
    import json as _json

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    for path, _size in files:
        full = f"datasets/{ {'fineweb-2': 'HuggingFaceFW/fineweb-2', 'glotcc': 'cis-lmu/GlotCC-V1', 'hplt2': 'HPLT/HPLT2.0_cleaned', 'wikipedia': 'wikimedia/wikipedia', 'c4': 'allenai/c4'}[src] }/{path}"
        with fs.open(full, "rb") as f:
            if path.endswith(".parquet"):
                pf = pq.ParquetFile(f)
                col = "text" if "text" in pf.schema_arrow.names else "content"
                for batch in pf.iter_batches(batch_size=500, columns=[col]):
                    for t in batch.column(0).to_pylist():
                        yield t
            else:
                with gzip.open(f, "rt", encoding="utf-8") as g:
                    for line in g:
                        try:
                            yield _json.loads(line)["text"]
                        except Exception:
                            continue


def _web_worker(args):
    lang, src, code, files, max_docs, target_per_bin, seed, out_dir = args
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = os.path.join(out_dir, f"{lang}__{src}.parquet")
    if os.path.exists(out):
        return lang, src, -1, 0
    rng = np.random.default_rng(seed + ALL_ISOS.index(lang))
    rows: Dict[str, list] = {k: [] for k in ("source", "src_afr_tokens", "length_bin", "n_sentences")}
    counts = {b: 0 for b in R.BINS}
    n_docs = 0
    try:
        for text in _iter_web_docs(src, files, os.environ.get("HF_TOKEN")):
            n_docs += 1
            if not text or len(text) < 400:
                continue
            sents = R.clean_doc(text[:60000])
            if len(sents) < 3:
                continue
            for w in R.windows_from_doc(sents, rng, count_afro, chars_per_token=4.0):
                for k in rows:
                    rows[k].append(w[k])
                counts[w["length_bin"]] += 1
            if n_docs >= max_docs or min(counts.values()) >= target_per_bin:
                break
    except Exception as e:                       # a broken remote file must not lose what was read so far
        print(f"{lang} {src}: stopped after {n_docs} docs: {type(e).__name__} {str(e)[:100]}", flush=True)
    repo, lic = WEB_LICENCE[src]
    n = len(rows["source"])
    rows["source_dataset"] = [f"{repo}:{code}"] * n
    rows["licence"] = [lic] * n
    os.makedirs(out_dir, exist_ok=True)
    pq.write_table(pa.table(rows) if n else pa.table({"source": pa.array([], pa.string())}), out, compression="zstd")
    return lang, src, n, n_docs


def cmd_pool_web(a) -> int:
    from multiprocessing import Pool

    plan = json.load(open(a.sources))
    langs = a.langs or [l for l in SFT77 if l in plan]
    jobs = []
    for lang in langs:
        have = plan.get(lang, {})
        for src in WEB_ORDER:
            if src not in have or src not in a.use:
                continue
            files = have[src]["files"]
            if src == "c4":
                files = files[:2]
            elif src == "hplt2":
                files = files[:2]
            jobs.append((lang, src, have[src]["code"], files, a.max_docs, a.target_per_bin, a.seed, os.path.join(a.out, "pool_web")))
    print(len(jobs), "(language, source) jobs", flush=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for k, (lang, src, n, nd) in enumerate(pool.imap_unordered(_web_worker, jobs)):
            print(f"{k + 1}/{len(jobs)} {lang} {src}: {n} windows from {nd} docs {time.time() - t0:.0f}s", flush=True)
    return 0


# ----------------------------------------------------------------------------------------------------------------------- pool-pairs
def _pairs_worker(args):
    lang, shard_rows, tok_name, seed = args       # shard_rows: list of (shard_path, [rows], ds_name, lic)
    import pyarrow as pa
    import pyarrow.parquet as pq

    texts, srcs, lics = [], [], []
    for path, rows, ds, lic in shard_rows:
        t = pq.read_table(path, columns=["target"]).take(pa.array(rows, pa.int64()))["target"].to_pylist()
        texts.extend(t)
        srcs.extend([ds] * len(t))
        lics.extend([lic] * len(t))
    from gemma_mt.afri2afri.text import norm_text

    texts = [norm_text(x) for x in texts]
    toks = count_afro(texts) if texts else []
    return lang, texts, [x - 2 for x in toks], srcs, lics


def cmd_pool_pairs(a) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from multiprocessing import Pool

    sys.path.insert(0, HERE)
    import build_afri2afri_sft as B

    arr, names, _ = B.load_index(a.sft)
    files = sorted(glob.glob(os.path.join(a.pairs, "pairs-*.parquet")))
    sel_keys = set()
    sel_text = set()
    from gemma_mt.afri2afri.text import hash64, pivot_key

    for split in ("train", "val"):
        t = pq.read_table(os.path.join(a.sft, f"pairs_{split}.parquet"), columns=["group_key", "src", "tgt", "origin"]).to_pydict()
        for g, s, tg, o in zip(t["group_key"], t["src"], t["tgt"], t["origin"]):
            if o == "pivot_en":
                sel_keys.add(int(g, 16))
            sel_text.add(hash64(pivot_key(s)))
            sel_text.add(hash64(pivot_key(tg)))
    print("SFT groups", len(sel_keys), "SFT texts", len(sel_text), flush=True)
    ex_keys = np.fromiter(sel_keys, dtype=np.uint64, count=len(sel_keys))
    ex_text = np.fromiter(sel_text, dtype=np.uint64, count=len(sel_text))
    elig = (arr["key"] != 0) & (arr["iso"] != 255) & ~np.isin(arr["key"], ex_keys) & ~np.isin(arr["tkey"], ex_text)
    # drop the lowest-QE third of each language, rows from synthetic sources, and very short texts
    tier = np.array([B.prov.tier_of(x) for x in names], np.uint8)[arr["ds"]]
    elig &= (tier >= 1) & (arr["tlen"] >= 20)
    rng = np.random.default_rng(a.seed)
    jobs = []
    lic_names = B.ix.LICENCE_CLASSES
    for iso in SFT77:
        i = ALL_ISOS.index(iso)
        idx = np.flatnonzero(elig & (arr["iso"] == i))
        if len(idx) == 0:
            continue
        thr = np.quantile(arr["qe"][idx], 0.33)
        idx = idx[arr["qe"][idx] >= thr]
        if len(idx) > a.per_language:
            idx = rng.choice(idx, a.per_language, replace=False)
        idx = idx[np.lexsort((arr["row"][idx], arr["shard"][idx], arr["ds"][idx]))]       # file order inside a source: consecutive rows make the pseudo-paragraphs
        groups = []
        for (si, ds), g in _runs(arr, idx):
            groups.append((files[int(si)], [int(r) for r in arr["row"][g]], names[int(ds)], lic_names[int(arr["lic"][g[0]])]))
        jobs.append((iso, groups, None, a.seed))
    print("languages with held-out rows", len(jobs), flush=True)
    os.makedirs(os.path.join(a.out, "pool_pairs"), exist_ok=True)
    t0 = time.time()
    with Pool(a.workers) as pool:
        for lang, texts, toks, srcs, lics in pool.imap_unordered(_pairs_worker, jobs):
            if not texts:
                continue
            rng2 = np.random.default_rng(a.seed + ALL_ISOS.index(lang))
            used = {b: 0 for b in W.CYCLE}
            out = {"source": [], "src_afr_tokens": [], "length_bin": [], "n_sentences": [], "source_dataset": [], "licence": []}
            # runs of the same source dataset
            start = 0
            while start < len(texts):
                end = start
                while end < len(texts) and srcs[end] == srcs[start] and lics[end] == lics[start]:
                    end += 1
                ws = W.build_windows(texts[start:end], texts[start:end], toks[start:end], toks[start:end], rng2, used)
                for w in ws:
                    n = w["src_tokens"]
                    out["source"].append(w["src"]); out["src_afr_tokens"].append(n); out["length_bin"].append(w["bin"]); out["n_sentences"].append(w["n_pairs"])
                    out["source_dataset"].append(srcs[start]); out["licence"].append(lics[start])
                start = end
            pq.write_table(pa.table(out), os.path.join(a.out, "pool_pairs", f"{lang}.parquet"), compression="zstd")
            print(lang, len(texts), "->", len(out["source"]), "windows", f"{time.time() - t0:.0f}s", flush=True)
    return 0


def _runs(arr, idx):
    """Split a sorted index array into runs of equal (shard, dataset)."""
    sh, ds = arr["shard"][idx], arr["ds"][idx]
    cut = np.flatnonzero((sh[1:] != sh[:-1]) | (ds[1:] != ds[:-1])) + 1
    for part in np.split(np.arange(len(idx)), cut):
        if len(part):
            yield (sh[part[0]], ds[part[0]]), idx[part]


# ----------------------------------------------------------------------------------------------------------------------- assemble
def cmd_assemble(a) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    sys.path.insert(0, HERE)
    import build_afri2afri_sft as B
    from gemma_mt.afri2afri import filters as F
    from gemma_mt.afri2afri.text import hash64, pivot_key, shingles
    from gemma_mt.rl.lid import GlotLIDClassifier

    t0 = time.time()
    cal = json.load(open(a.calibration))
    pol = F.lid_policy(cal, fallback={k: v.glotlid for k, v in LANGUAGES.items()})
    leak = B.build_leak(a.bench, a.extra_bench)
    sel_text = set()
    for split in ("train", "val"):
        t = pq.read_table(os.path.join(a.sft, f"pairs_{split}.parquet"), columns=["src", "tgt"]).to_pydict()
        for s, tg in zip(t["src"], t["tgt"]):
            sel_text.add(hash64(pivot_key(s))); sel_text.add(hash64(pivot_key(tg)))
    lid = None if a.glotlid == "fake" else GlotLIDClassifier(path=a.glotlid, cache_size=1)           # "fake": every text gets the first accepted label (tests)
    pair_alloc = {k: v["alloc"] for k, v in json.load(open(os.path.join(a.sft, "selection_pairs.json"))).items()}
    wts = R.direction_weights(pair_alloc, list(SFT77))
    rng = np.random.default_rng(a.seed)
    report: Dict = {"per_source": {}, "dropped": {}}
    rows_by_target: Dict[str, Dict[str, list]] = {t: {c: [] for c in R.RL_X2X_COLUMNS} for t in SFT77}
    drop = report["dropped"]
    for src in SFT77:
        parts = []
        for kind in ("pool_docs", "pool_pairs"):
            p = os.path.join(a.out, kind, f"{src}.parquet")
            if os.path.exists(p):
                parts.append((kind, pq.read_table(p).to_pylist()))
        for p in sorted(glob.glob(os.path.join(a.out, "pool_web", f"{src}__*.parquet"))):
            t = pq.read_table(p)
            if t.num_rows:
                parts.append(("pool_web", t.to_pylist()))
        parts.sort(key=lambda kv: 1 if kv[0] == "pool_pairs" else 0)           # real documents first: the cap below keeps them in preference to pseudo-paragraphs
        if not parts:
            report["per_source"][src] = {"windows": 0}
            continue
        keep: Dict[str, list] = {b: [] for b in R.BINS}
        seen = set()
        target = int(a.cap_per_bin * 1.6)                  # candidates per bin to collect before the LID guard and the cap (documents first, then pseudo-paragraphs)
        raw = {}
        for kind, rows in parts:
            cnt = {b: 0 for b in R.BINS}
            for r in rows:
                cnt[r["length_bin"]] += 1
            raw[kind] = cnt
        report.setdefault("pool_raw", {})[src] = {k: sum(v.values()) for k, v in raw.items()}
        for kind, rows in parts:
            order = rng.permutation(len(rows))
            for oi in order:
                r = rows[int(oi)]
                b = r["length_bin"]
                if len(keep[b]) >= target:
                    continue
                s = r["source"]
                h = hash64(pivot_key(s))
                if h in seen:
                    drop["duplicate"] = drop.get("duplicate", 0) + 1
                    continue
                seen.add(h)
                if not (F.basic_ok(s, max_chars=12000) and F.script_ok(s, src)):
                    drop["basic_or_script"] = drop.get("basic_or_script", 0) + 1
                    continue
                if leak.is_leak(s):
                    drop["benchmark_leak"] = drop.get("benchmark_leak", 0) + 1
                    continue
                if kind != "pool_pairs" and any(hash64(pivot_key(x)) in sel_text for x in R.split_sentences(s)):
                    drop["in_sft_texts"] = drop.get("in_sft_texts", 0) + 1
                    continue
                keep[b].append((kind, r))
        report.setdefault("pool_after_filters", {})[src] = {b: len(v) for b, v in keep.items()}
        report.setdefault("pool_real_documents", {})[src] = {b: sum(1 for x in v if x[0] != "pool_pairs") for b, v in keep.items()}
        # GlotLID guard on what is left (cap per bin first)
        accepted, entry = {}, pol.get(src)
        for b, lst in keep.items():
            docs_ = [x for x in lst if x[0] != "pool_pairs"]
            pairs_ = [x for x in lst if x[0] == "pool_pairs"]
            rng.shuffle(docs_)
            rng.shuffle(pairs_)
            lst[:] = docs_ + pairs_
            lst = lst[: int(a.cap_per_bin * 1.3)]
            if entry and entry[2]:
                labs = ([[(sorted(entry[0])[0], 1.0)]] * len(lst)) if lid is None else lid._predict_many([r["source"][:1000] for _, r in lst], 1)
                ok = []
                for (kind, r), lab in zip(lst, labs):
                    if len(r["source"]) < entry[1] or lab[0][0] in entry[0]:
                        ok.append((kind, r))
                    else:
                        drop["lid_bad"] = drop.get("lid_bad", 0) + 1
                lst = ok
            accepted[b] = lst[: a.cap_per_bin]
        report["per_source"][src] = {b: len(v) for b, v in accepted.items()}
        targets = [t for t in SFT77 if t != src]
        for b, lst in accepted.items():
            if not lst:
                continue
            quota = {t: int(round(a.quota * wts[(src, t)])) for t in targets}
            assign = R.assign_targets(len(lst), targets, quota, rng, max_reuse=a.max_reuse)
            for t, idxs in assign.items():
                d = rows_by_target[t]
                for k in idxs:
                    kind, r = lst[int(k)]
                    d["source"].append(r["source"]); d["src_iso"].append(src); d["iso"].append(t); d["language"].append(LANGUAGES[t].name)
                    d["src_language"].append(LANGUAGES[src].name); d["src_afr_tokens"].append(int(r["src_afr_tokens"])); d["length_bin"].append(r["length_bin"])
                    d["n_sentences"].append(int(r["n_sentences"])); d["source_dataset"].append(r["source_dataset"]); d["licence"].append(r["licence"])
                    d["src_qe_valid"].append(src in QE_VALID); d["tgt_qe_valid"].append(t in QE_VALID)
        print(src, report["per_source"][src], f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.join(a.out, "rl_prompts_x2x"), exist_ok=True)
    per_target = {}
    cnt_dir: Dict[str, int] = {}
    for t, d in rows_by_target.items():
        if not d["source"]:
            continue
        order = rng.permutation(len(d["source"]))
        tbl = {c: [d[c][i] for i in order] for c in R.RL_X2X_COLUMNS}
        pq.write_table(pa.table(tbl), os.path.join(a.out, "rl_prompts_x2x", f"{t}.parquet"), compression="zstd")
        per_target[t] = len(order)
        for s in d["src_iso"]:
            cnt_dir[f"{s}>{t}"] = cnt_dir.get(f"{s}>{t}", 0) + 1
    report["rows_per_target"] = per_target
    report["directions_with_prompts"] = len(cnt_dir)
    report["rows_per_direction_summary"] = {"min": min(cnt_dir.values()), "median": int(np.median(list(cnt_dir.values()))), "max": max(cnt_dir.values())} if cnt_dir else {}
    json.dump(cnt_dir, open(os.path.join(a.out, "rl_directions.json"), "w"))
    json.dump(report, open(os.path.join(a.out, "rl_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "per_source"}, indent=1))
    return 0


QE_VALID = set(l.iso for l in LANGUAGES.values() if l.africomet_status == "verified" and l.africomet_code)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pool-docs")
    p.add_argument("--shards", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--langs", nargs="+", default=["amh", "hau", "ibo", "pcm", "tir", "yor"])
    p.add_argument("--max-files", type=int, default=3)
    p.add_argument("--lines-per-block", type=int, default=60)
    p.add_argument("--max-lines", type=int, default=400_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_pool_docs)
    p = sp.add_parser("pool-web")
    p.add_argument("--sources", required=True, help="json written by afri2afri_scout_sources.py")
    p.add_argument("--out", required=True)
    p.add_argument("--langs", nargs="*", default=None)
    p.add_argument("--use", nargs="+", default=list(WEB_ORDER))
    p.add_argument("--max-docs", type=int, default=60000)
    p.add_argument("--target-per-bin", type=int, default=3000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_pool_web)
    p = sp.add_parser("pool-pairs")
    p.add_argument("--sft", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-language", type=int, default=60_000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_pool_pairs)
    p = sp.add_parser("assemble")
    p.add_argument("--sft", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bench", required=True)
    p.add_argument("--extra-bench", nargs="*", default=[])
    p.add_argument("--calibration", required=True)
    p.add_argument("--glotlid", required=True)
    p.add_argument("--cap-per-bin", type=int, default=1500)
    p.add_argument("--quota", type=float, default=24.0, help="prompts per direction per bin before the SFT-allocation weight")
    p.add_argument("--max-reuse", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_assemble)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
