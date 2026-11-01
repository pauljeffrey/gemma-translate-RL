#!/usr/bin/env python3
"""Aya collection aligned en->X pairs (intake2). ONLY run after intake2_aya.py has shown alignment. Output goes to a QUARANTINE directory (common.assert_clean_src refuses it as a chain input) because the translated
subsets of the Aya collection are NLLB-3.3B machine translations (owner rule: no NLLB-derived data); the lead decides whether to rename it.

Pairs (chain pair schema of extra_sources.py): english, target, iso, source_dataset, licence, licence_raw, kind, source_family, flipped, key, aux; key = common.key_of(english, iso, target).
Join: rows of the X split and of the English split with the same (dataset_name, sub_dataset_name, id, template_id) are translations of each other (inputs<->inputs and targets<->targets are emitted as separate pairs).
Only rows whose dataset_name ends with "(T)" (the translated datasets) are used; templated / human datasets of the collection are not translations of an English row.

    python intake2_aya_emit.py --langs hausa,igbo,... --max-per-lang 1000000
"""
import argparse, collections, glob, json, os, shutil, sys, time
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq
import xxhash
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import key_of
from huggingface_hub import HfApi, hf_hub_download

REPO = "CohereLabs/aya_collection_language_split"
OUT = "/workspace/work/hub/extra2/aya_aligned_QUARANTINE_nllb"
TMP = "/workspace/work/instr/_tmp/aya_emit"
ISO = {"afrikaans": "afr", "amharic": "amh", "hausa": "hau", "igbo": "ibo", "northern_sotho": "nso", "plateau_malagasy": "mlg", "shona": "sna", "somali": "som", "southern_sotho": "sot", "swahili": "swh", "xhosa": "xho", "twi": "twi", "wolof": "wol"}
SCHEMA = pa.schema([("english", pa.string()), ("target", pa.string()), ("iso", pa.string()), ("source_dataset", pa.string()), ("licence", pa.string()), ("licence_raw", pa.string()), ("kind", pa.string()),
                    ("source_family", pa.string()), ("flipped", pa.bool_()), ("key", pa.uint64()), ("aux", pa.float64())])
api = HfApi(token=os.environ.get("HF_TOKEN"))


def files(lang):
    return sorted(f.path for f in api.list_repo_tree(REPO, path_in_repo=lang, repo_type="dataset", recursive=True) if hasattr(f, "size") and "/train-" in f.path)


def jkey(r):
    return xxhash.xxh64_intdigest(f"{r['dataset_name']}\x01{r['sub_dataset_name']}\x01{r['id']}\x01{r['template_id']}".encode())


def stream(path):
    os.makedirs(TMP, exist_ok=True)
    local = hf_hub_download(REPO, path, repo_type="dataset", local_dir=TMP, token=os.environ.get("HF_TOKEN"))
    try:
        pf = pq.ParquetFile(local)
        for rb in pf.iter_batches(batch_size=20000, columns=["id", "inputs", "targets", "dataset_name", "sub_dataset_name", "template_id"]):
            for r in rb.to_pylist():
                if r["dataset_name"] and r["dataset_name"].endswith("(T)"):
                    yield r
    finally:
        os.remove(local)


_IX = None


def _decon(chunk):
    """strict protected filter (R1/R2/R3 on both sides): returns the keep mask of a chunk of rows"""
    ra = _IX.check_texts([r["english"] for r in chunk]); rb = _IX.check_texts([r["target"] for r in chunk])
    return [not (any(x[:3]) or any(y[:3])) for x, y in zip(ra, rb)]


def main():
    global _IX
    import multiprocessing as mp
    sys.path.insert(0, "/workspace/work/code/sft_clean")
    import protected_index_loader as PL
    _IX, _ = PL.load("/workspace/work/work/_hub_protected/protected_index/protected_index.npz", "/workspace/work/work/_hub_protected/protected_index/protected_manifest.json")
    ap = argparse.ArgumentParser(); ap.add_argument("--langs", default=",".join(ISO)); ap.add_argument("--max-per-lang", type=int, default=400_000)
    a = ap.parse_args()
    langs = a.langs.split(",")
    xrows = {}
    for lang in langs:                                                   # pass 1: X side (kept in memory as key -> (inputs, targets))
        d = {}
        for p in files(lang):
            for r in stream(p):
                if len(d) >= a.max_per_lang:
                    break
                d[jkey(r)] = (r["inputs"], r["targets"], r["dataset_name"])
        xrows[lang] = d
        print(lang, "X rows", len(d), flush=True)
    wanted = set()
    for d in xrows.values():
        wanted |= d.keys()
    en = {}
    for p in files("english"):                                           # pass 2: English side, only the keys that some X split needs
        for r in stream(p):
            k = jkey(r)
            if k in wanted:
                en[k] = (r["inputs"], r["targets"])
        print("english", p, "matched so far", len(en), flush=True)
        if len(en) >= len(wanted):
            break
    shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
    summary = {}
    for lang in langs:
        iso, rows, seen = ISO[lang], [], set()
        for k, (xi, xt, dn) in xrows[lang].items():
            if k not in en:
                continue
            for e, x, part in ((en[k][0], xi, "inputs"), (en[k][1], xt, "targets")):
                e, x = (e or "").strip(), (x or "").strip()
                if len(e) < 3 or len(x) < 3 or e.casefold() == x.casefold():
                    continue
                kk = key_of(e, iso, x)
                if kk in seen:
                    continue
                seen.add(kk)
                rows.append(dict(english=e, target=x, iso=iso, source_dataset="CohereLabs/aya_collection_language_split:" + lang, licence="unspecified", licence_raw="apache-2.0 card; translated subsets are NLLB-3.3B output (QUARANTINED: owner rule no NLLB-derived)",
                                 kind="aya_translated_" + part, source_family="aya-collection-nllb", flipped=False, key=kk, aux=float("nan")))
        chunks = [rows[i:i + 2000] for i in range(0, len(rows), 2000)]
        with mp.get_context("fork").Pool(16) as pool:
            masks = pool.map(_decon, chunks)
        kept = [r for ch, m in zip(chunks, masks) for r, ok_ in zip(ch, m) if ok_]
        print(iso, "pairs before decon", len(rows), "after", len(kept), flush=True)
        rows = kept
        summary[iso] = len(rows)
        for i, s in enumerate(range(0, len(rows), 200_000)):
            pq.write_table(pa.Table.from_pylist(rows[s:s + 200_000], schema=SCHEMA), f"{OUT}/pairs-{iso}-{i:04d}.parquet", compression="zstd")
        print(iso, len(rows), flush=True)
    json.dump(summary, open(OUT + "/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
