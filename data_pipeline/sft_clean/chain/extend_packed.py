#!/usr/bin/env python3
"""Append the TRAIN packs of a second packed dataset to a copy of an existing one. Pack indices of the base dataset stay valid, so a schedule already
partly trained on the base can continue on the extended set (sampler.py stitch). Same tokenizer / iso table / seq_len are required; the base val split is kept.

    python extend_packed.py --base work/packed --add work/packed_efi --out work/packed_ext
"""
import argparse, json, os, shutil
import numpy as np

NAMES = ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id", "direction", "pack_start")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--add", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--inplace", action="store_true", help="append the added packs INTO --base (out must equal base): no copy of the base tokens (disk-tight boxes)")
    a = ap.parse_args()
    if a.inplace: assert os.path.abspath(a.out) == os.path.abspath(a.base), "--inplace needs --out == --base"
    mb, ma = (json.load(open(os.path.join(d, "meta.json"))) for d in (a.base, a.add))
    for k in ("seq_len", "dtype", "format_version"):
        assert mb[k] == ma[k], f"{k} differs between the datasets"
    # the iso table only ever grows at its END (gemma_mt.languages): one table must be a prefix of the other, language ids of the shorter one stay valid
    short, long_ = sorted((mb["iso_table"], ma["iso_table"]), key=len)
    assert long_[:len(short)] == short, "iso_table differs between the datasets (not a prefix)"
    assert mb["tokenizer"]["fingerprint"] == ma["tokenizer"]["fingerprint"], "tokenizer differs"
    os.makedirs(os.path.join(a.out, "train"), exist_ok=True)
    if not os.path.exists(os.path.join(a.out, "val")):
        shutil.copytree(os.path.join(a.base, "val"), os.path.join(a.out, "val"))
    B = {n: np.load(os.path.join(a.base, "train", n + ".npy")) for n in NAMES}; A = {n: np.load(os.path.join(a.add, "train", n + ".npy")) for n in NAMES}
    tb, ta = os.path.join(a.base, "train", "tokens.u32.bin"), os.path.join(a.add, "train", "tokens.u32.bin")
    n_tok_base, n_samples_base, n_packs_base = os.path.getsize(tb) // 4, len(B["sample_len"]), len(B["pack_start"]) - 1
    out_tok = os.path.join(a.out, "train", "tokens.u32.bin")
    if not a.inplace: shutil.copyfile(tb, out_tok)
    with open(out_tok, "ab") as f, open(ta, "rb") as g:
        shutil.copyfileobj(g, f, 1 << 24)
    new = {n: np.concatenate([B[n], A[n]]) for n in NAMES if n not in ("sample_offset", "pack_start")}
    new["sample_offset"] = np.concatenate([B["sample_offset"], A["sample_offset"] + np.uint64(n_tok_base)]).astype(np.uint64)
    new["pack_start"] = np.concatenate([B["pack_start"], A["pack_start"][1:] + np.uint64(n_samples_base)]).astype(np.uint64)
    for n, v in new.items():                                           # atomic replace (new inode): a process that has the old arrays memory-mapped keeps a consistent view
        tmp = os.path.join(a.out, "train", n + ".tmp.npy"); np.save(tmp, v.astype(B[n].dtype)); os.replace(tmp, os.path.join(a.out, "train", n + ".npy"))
    assert os.path.getsize(out_tok) // 4 == n_tok_base + os.path.getsize(ta) // 4
    assert len(new["pack_start"]) - 1 == n_packs_base + len(A["pack_start"]) - 1 and int(new["pack_start"][-1]) == len(new["sample_len"])
    meta = dict(mb); meta["iso_table"] = long_; meta["extension"] = {"base": os.path.abspath(a.base), "added": os.path.abspath(a.add), "base_train_packs": n_packs_base, "added_train_packs": len(A["pack_start"]) - 1,
                                          "note": "train packs appended; base pack indices unchanged; val split = base val"}
    mt = os.path.join(a.out, "meta.json.tmp"); json.dump(meta, open(mt, "w"), indent=1); os.replace(mt, os.path.join(a.out, "meta.json"))
    print(f"extended: {n_packs_base} + {len(A['pack_start']) - 1} train packs, {n_samples_base} + {len(A['sample_len'])} samples, {n_tok_base} + {os.path.getsize(ta) // 4} tokens")


if __name__ == "__main__":
    main()
