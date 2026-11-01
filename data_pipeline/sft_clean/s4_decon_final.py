#!/usr/bin/env python3
"""Final protected-index check of the science-stage mixture (agent C2, 2026-10-28). The number to report is HITS = 0.

    python s4_decon_final.py --packed MIX --index protected_index.npz --manifest protected_manifest.json [--out MIX/decon_final.json] [--controls-jsonl FILE] [--jsonl rows.jsonl ...]

* decodes EVERY sample of the packed training split (`MIX/train`: the actual training tokens, not the input jsonl) into its English prompt text and its target text, and runs the strict
  protected-index check (data_pipeline/sft_clean/decon.py, rules R1 exact sentence / R2 contained sentence / R3 shared word 8-gram, both normal forms) on both; NO exemption is attached
  (the FLORES+ dev train-half exemption of protected-2026-10-28.3 is not used). Hits are counted per component (train/source.npy) and per protected set. Exit code 1 if any hit.
* --jsonl: additionally checks the rows of component files (english + target).
* --controls-jsonl: rows whose `english` MUST be flagged (e.g. FLORES+ dev train-half windows): the checker is alive only if all of them hit; reported as `controls_caught`.
The mixture's validation splits (val/, val_sci/) are monitors; val_sci is protected AfriScience-MT dev text by design and is NOT part of this check (it is never trained on).
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s4_common as K  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--packed", required=True)
ap.add_argument("--index", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--out", default="")
ap.add_argument("--jsonl", nargs="*", default=[])
ap.add_argument("--controls-jsonl", default="")
ap.add_argument("--nproc", type=int, default=2)
ap.add_argument("--model", default="", help="tokenizer (default: the one recorded in meta.json)")
a = ap.parse_args()
K.gemma_pkg()
from gemma_mt.languages import ALL_ISOS

meta = json.load(open(os.path.join(a.packed, "meta.json")))
tok = K.tokenizer(a.model or meta["tokenizer"]["requested"])
d = os.path.join(a.packed, "train")
off, ln, pl, tl, iso = (np.load(os.path.join(d, n + ".npy")) for n in ("sample_offset", "sample_len", "prompt_len", "target_len", "iso_id"))
src = np.load(os.path.join(d, "source.npy")) if os.path.exists(os.path.join(d, "source.npy")) else np.zeros(len(ln), np.uint8)
tk = np.memmap(os.path.join(d, "tokens.u32.bin"), dtype=np.uint32, mode="r")
P = K.Protected(a.index, a.manifest, a.nproc)
K.log(f"protected index {P.version}; decoding {len(ln)} training samples")
hits_by_comp, hits_by_set, bad_decode = collections.Counter(), collections.Counter(), 0
n_checked, n_hit_samples, B = 0, 0, 4000
examples = []
for s in range(0, len(ln), B):
    ens, tgs, idx = [], [], []
    for i in range(s, min(s + B, len(ln))):
        ids = np.asarray(tk[int(off[i]):int(off[i]) + int(ln[i])])
        en = K.english_of_prompt(tok, ids, int(pl[i]))
        tg = K.target_of_sample(tok, ids, int(pl[i]), int(tl[i]), ALL_ISOS[int(iso[i])])
        if en is None or tg is None:
            bad_decode += 1
            continue
        ens.append(en); tgs.append(tg); idx.append(i)
    h = P.names(ens + tgs)
    for k, i in enumerate(idx):
        hh = h[k] + h[len(idx) + k]
        n_checked += 1
        if hh:
            n_hit_samples += 1
            hits_by_comp[K.COMPONENTS[int(src[i])]] += 1
            for nm in set(hh):
                hits_by_set[nm] += 1
            if len(examples) < 5:
                examples.append({"sample": i, "component": K.COMPONENTS[int(src[i])], "iso": ALL_ISOS[int(iso[i])], "sets": sorted(set(hh)), "english_xxh64": __import__("xxhash").xxh64_hexdigest(ens[k].encode())})
rows_hits = {}
for f in a.jsonl:
    rr = K.read_jsonl(f)
    hh = P.names([r["english"] for r in rr] + [r["target"] for r in rr])
    rows_hits[os.path.basename(f)] = {"rows": len(rr), "hit_rows": sum(1 for k in range(len(rr)) if hh[k] or hh[len(rr) + k])}
controls = None
if a.controls_jsonl:
    cr = [r["english"] for r in K.read_jsonl(a.controls_jsonl)]
    ch = P.names(cr)
    controls = {"n": len(cr), "caught": sum(1 for x in ch if x)}
P.close()
out = {"index_version": P.version, "index_file_sha256": P.manifest.get("index_file_sha256"), "mode": "strict, no FLORES+ dev train-half exemption", "packed_dir": os.path.abspath(a.packed),
       "samples_checked": n_checked, "undecodable_samples": bad_decode, "HITS_samples": n_hit_samples, "hits_by_component": dict(hits_by_comp), "hits_by_protected_set": dict(hits_by_set),
       "example_hits_no_text": examples, "jsonl_checks": rows_hits, "controls_caught": controls}
if a.out:
    json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(out, indent=1))
sys.exit(1 if (n_hit_samples or bad_decode) else 0)
