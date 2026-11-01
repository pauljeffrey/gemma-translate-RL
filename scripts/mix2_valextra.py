#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build OUT/val_extra from the validation windows R10 packed with its extra data (<=100 samples per language and group), for the groups used in the mixture.
    python scripts/mix2_valextra.py --out /workspace/jobs/r11/mix_v2B_v2 --groups commercial_ok,share_alike,owner_code_switch"""
import argparse, os, shutil, sys, collections
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mix2_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--groups", required=True)
ap.add_argument("--r10", default="/workspace/jobs/r10/sft_extra_v1"); ap.add_argument("--per-language", type=int, default=100); ap.add_argument("--seq-len", type=int, default=8192); ap.add_argument("--only", default="", help="group=iso,iso;group2=iso : keep only these languages of that group"); a = ap.parse_args()
only = {x.split("=")[0]: set(x.split("=")[1].split(",")) for x in a.only.split(";") if x}
from gemma_mt.languages import ALL_ISOS
B = C.load_script("build_sft_science_mix")
rng = np.random.default_rng(0)
items = []
for g in a.groups.split(","):
    d = os.path.join(a.r10, g, "val")
    if not os.path.exists(os.path.join(d, "iso_id.npy")):
        continue
    iso = np.load(os.path.join(d, "iso_id.npy")); ln = np.load(os.path.join(d, "sample_len.npy")); off = np.load(os.path.join(d, "sample_offset.npy"))
    pl = np.load(os.path.join(d, "prompt_len.npy")); tl = np.load(os.path.join(d, "target_len.npy"))
    tok = np.memmap(os.path.join(d, "tokens.u32.bin"), dtype=np.uint32, mode="r")
    for l in np.unique(iso):
        if g in only and ALL_ISOS[int(l)] not in only[g]:
            continue
        idx = np.flatnonzero(iso == l); idx = rng.permutation(idx)[:a.per_language]
        for i in idx:
            items.append((SimpleNamespace(input_ids=np.asarray(tok[int(off[i]):int(off[i]) + int(ln[i])]), prompt_len=int(pl[i]), target_len=int(tl[i])), int(l)))
tmp = os.path.join(a.out, "_val_extra_tmp")
B.write_split(items, tmp, a.seq_len, 5, "val")
shutil.rmtree(os.path.join(a.out, "val_extra"), ignore_errors=True)
shutil.move(os.path.join(tmp, "val"), os.path.join(a.out, "val_extra")); shutil.rmtree(tmp, ignore_errors=True)
print("val_extra:", len(items), "samples from", a.groups)
