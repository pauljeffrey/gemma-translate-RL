#!/usr/bin/env python3
"""Aya collection alignment check (intake2). Question: is row k of an African-language split of CohereLabs/aya_collection_language_split the translation of row k of the English split
(same dataset_name / id at the same position)? Only the `id`, `dataset_name` and `sub_dataset_name` columns of the first train files are read (range reads, nothing is stored), the share of
equal (dataset_name, id) pairs is reported per 100k-row block, and 200 aligned rows per split are compared on content (digit sets, length ratio, short samples are printed for a human check).

    python intake2_aya.py        -> /workspace/work/instr/_work/aya_alignment.json (+ stdout summary)
"""
import json, os, re, sys, random
import numpy as np
from huggingface_hub import HfApi, HfFileSystem
import pyarrow.parquet as pq

REPO = "CohereLabs/aya_collection_language_split"
LANGS = {"afrikaans": "afr", "amharic": "amh", "bemba": "bem", "central_kanuri": "knc", "fon": "fon", "hausa": "hau", "igbo": "ibo", "kinyarwanda": "kin", "northern_sotho": "nso", "nyanja": "nya",
         "plateau_malagasy": "mlg", "shona": "sna", "somali": "som", "southern_sotho": "sot", "swahili": "swh", "tamasheq": "tmh", "twi": "twi", "wolof": "wol", "xhosa": "xho"}
BLOCK = 100_000
tok = os.environ.get("HF_TOKEN")
fs, api = HfFileSystem(token=tok), HfApi(token=tok)


def train_files(lang):
    return sorted(f.path for f in api.list_repo_tree(REPO, path_in_repo=lang, repo_type="dataset", recursive=True) if hasattr(f, "size") and "/train-" in f.path)


def read_keys(files, limit):
    """(dataset_name, id) per row of the first `limit` rows of the concatenated files"""
    ds, ids, n = [], [], 0
    for p in files:
        with fs.open(f"datasets/{REPO}/{p}", "rb") as f:
            t = pq.ParquetFile(f).read(columns=["dataset_name", "id"])
        ds += t.column(0).to_pylist(); ids += t.column(1).to_pylist()
        n += t.num_rows
        if n >= limit:
            break
    return ds[:limit], ids[:limit]


def read_rows(files, positions):
    """full rows (inputs, targets, dataset_name, id) at the given global positions"""
    out, base = {}, 0
    want = sorted(positions)
    for p in files:
        with fs.open(f"datasets/{REPO}/{p}", "rb") as f:
            pf = pq.ParquetFile(f)
            cnt = pf.metadata.num_rows
            local = [w - base for w in want if base <= w < base + cnt]
            if local:
                acc, rg_of = 0, {}
                starts = []
                for rg in range(pf.num_row_groups):
                    m = pf.metadata.row_group(rg).num_rows
                    starts.append((acc, acc + m, rg)); acc += m
                cache = {}
                for w in local:
                    for a, b, rg in starts:
                        if a <= w < b:
                            if rg not in cache:
                                cache[rg] = pf.read_row_group(rg, columns=["inputs", "targets", "dataset_name", "id"]).to_pylist()
                            out[w + base] = cache[rg][w - a]
                            break
        base += cnt
    return out


def digits(s):
    return sorted(re.findall(r"\d+", s or ""))


def main():
    en_files = train_files("english")[:3]
    en_ds, en_id = read_keys(en_files, 4_000_000)
    print("english rows read", len(en_ds), flush=True)
    res = {"english_rows_compared": len(en_ds)}
    rnd = random.Random(0)
    for lang, iso in LANGS.items():
        files = train_files(lang)
        if not files:
            res[lang] = {"error": "no train file"}; continue
        ds, ids = read_keys(files, 4_000_000)
        n = min(len(ds), len(en_ds))
        eq = np.array([(ds[k] == en_ds[k] and ids[k] == en_id[k]) for k in range(n)], bool)
        blocks = [round(float(eq[s:s + BLOCK].mean()), 3) for s in range(0, n, BLOCK)]
        # content check on 200 aligned positions spread over the aligned region
        pos = np.nonzero(eq)[0]
        sample = sorted(rnd.sample(list(pos), min(200, len(pos)))) if len(pos) else []
        rows_en = read_rows(en_files, sample) if sample else {}
        rows_x = read_rows(files, sample) if sample else {}
        dig_ok = dig_n = 0; ratios = []; shown = []
        for k in sample:
            a, b = rows_en[k], rows_x[k]
            da, db = digits(a["inputs"]), digits(b["inputs"])
            if da:
                dig_n += 1; dig_ok += int(da == db)
            ratios.append(len(b["inputs"]) / max(1, len(a["inputs"])))
            if len(shown) < 3:
                shown.append({"pos": int(k), "dataset": a["dataset_name"], "en": a["inputs"][:110], lang: b["inputs"][:110]})
        res[lang] = {"iso": iso, "rows_in_first_files": len(ds), "n_compared": n, "aligned_share_overall": round(float(eq.mean()), 3) if n else 0, "aligned_share_per_100k_block": blocks,
                     "content_digits_equal": f"{dig_ok}/{dig_n}", "median_len_ratio_x_over_en": round(float(np.median(ratios)), 2) if ratios else None, "aligned_datasets": sorted({rows_en[k]['dataset_name'] for k in sample})[:12], "examples": shown}
        print(lang, json.dumps({k: v for k, v in res[lang].items() if k != "examples"})[:700], flush=True)
    json.dump(res, open("/workspace/work/instr/_work/aya_alignment_v2.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
