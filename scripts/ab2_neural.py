# SPDX-License-Identifier: AGPL-3.0-or-later
"""E1 pairwise benchmark comparison, step 2 (GPU, box): neural MT metrics per sentence for several systems on the same benchmark rows.

    python pairwise_neural.py --sys sft=pairs_sft.jsonl.gz --sys rl=pairs_rl.jsonl.gz --out neural --cap 400 \
        --metrics afri_ref,afri_qe,ssa_ref,ssa_qe [--gpu-gib 5]

Metrics (all on NFC text, English source, hypothesis with the language tag already stripped):
  afri_ref   AfriCOMET reference-based  masakhane/africomet-stl-1.1            (src, mt, ref)
  afri_qe    AfriCOMET-QE                masakhane/africomet-qe-stl-1.1         (src, mt)         = the RL reward model
  ssa_ref    SSA-COMET-MTL with reference McGill-NLP/ssa-comet-mtl, mean of the 3 views (mt/src, mt/ref, mt/src/ref); ONE joint [mt, ...] sequence <= 512 positions
  ssa_qe     SSA-COMET-MTL reference-free (src, mt); pairs over the joint budget are cut into pieces (rl/joint_budget.py) and averaged
Only languages AfriCOMET covers (languages.qe_covered, 54) are scored. Rows: every `clean` row (English sentence not in the SFT text) + a deterministic
hash-ordered sample of the other rows up to --cap rows per (benchmark, language). Identical (src, mt, ref) triples of different systems are scored once.

Output: <out>/<metric>.jsonl, one line per scored chunk {"chunk": "bench|iso", "scores": {"<id>": {"<system>": value}}}; resumable (done chunks are skipped);
<out>/rows.json = the selected keys. The GPU share is capped with torch.cuda.set_per_process_memory_fraction (--gpu-gib) and OOM halves the batch and retries.
Run with venv_gemma, PYTHONPATH=/workspace/sft_build/pylibs_tf4:/workspace/sft_build/pylibs:/workspace/sft_build/gemma_mt_2c, HF_HOME=/workspace/sft_build/hf HF_HUB_OFFLINE=1.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
import time
import unicodedata
from collections import defaultdict

sys.path.insert(0, "/workspace/sft_build/pylibs")

METRICS = {
    "afri_ref": ("masakhane/africomet-stl-1.1", True),
    "afri_qe": ("masakhane/africomet-qe-stl-1.1", False),
    "ssa_ref": ("McGill-NLP/ssa-comet-mtl", True),
    "ssa_qe": ("McGill-NLP/ssa-comet-mtl", False),
}


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def load_system(path):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[(r["b"], r["iso"], r["id"])] = r
    return out


def select_keys(systems, cap, seed):
    from gemma_mt.languages import LANGUAGES

    keys = None
    for s in systems.values():
        keys = set(s) if keys is None else keys & set(s)
    base = next(iter(systems.values()))
    by_unit = defaultdict(list)
    for k in keys:
        lg = LANGUAGES.get(k[1])
        if lg is None or not lg.qe_covered:
            continue
        by_unit[(k[0], k[1])].append(k)
    sel = []
    for unit, ks in sorted(by_unit.items()):
        ks.sort(key=lambda k: hashlib.sha1(f"{seed}|{k[0]}|{k[1]}|{k[2]}".encode()).hexdigest())
        clean = [k for k in ks if base[k].get("contaminated") is False]
        cs = set(clean)
        rest = [k for k in ks if k not in cs]
        n_rest = max(0, cap - len(clean)) if cap > 0 else len(rest)
        sel += sorted(clean + rest[:n_rest])
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", action="append", required=True, help="name=path/to/surface.jsonl.gz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=400, help="rows per (benchmark, language) besides all clean rows; 0 = all")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--metrics", default="afri_ref,afri_qe,ssa_ref,ssa_qe")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gpu-gib", type=float, default=5.0)
    a = ap.parse_args()
    import torch

    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(0.95, a.gpu_gib / total))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ab2_comet import CometScorer

    os.makedirs(a.out, exist_ok=True)
    systems = {}
    for spec in a.sys:
        n, p = spec.split("=", 1)
        systems[n] = load_system(p)
    sel = select_keys(systems, a.cap, a.seed)
    chunks = defaultdict(list)
    for k in sel:
        chunks[f"{k[0]}|{k[1]}"].append(k)
    print(len(sel), "rows in", len(chunks), "chunks", flush=True)
    with open(os.path.join(a.out, "rows.json"), "w") as f:
        json.dump({"cap": a.cap, "seed": a.seed, "systems": list(systems), "n": len(sel)}, f)

    scorer = None
    loaded = None
    for metric in a.metrics.split(","):
        model_id, ref_based = METRICS[metric]
        path = os.path.join(a.out, f"{metric}.jsonl")
        done = set()
        if os.path.isfile(path):
            for line in open(path, encoding="utf-8"):
                try:
                    done.add(json.loads(line)["chunk"])
                except Exception:
                    pass
        todo = [c for c in sorted(chunks) if c not in done]
        print(f"[{metric}] {len(done)} chunks done, {len(todo)} to do", flush=True)
        if not todo:
            continue
        if loaded != model_id:
            scorer = None
            torch.cuda.empty_cache()
            mid = model_id
            if model_id == "masakhane/africomet-stl-1.1":
                mid = glob.glob("/workspace/sft_build/hf/hub/models--masakhane--africomet-stl-1.1/snapshots/*/checkpoints/model.ckpt")[0]
            scorer = CometScorer(mid, batch_size=a.batch, reference_based=ref_based, max_batch_chars=a.batch * 700)
            loaded = model_id
        scorer.reference_based = ref_based
        t0 = time.time()
        nsc = 0
        with open(path, "a", encoding="utf-8") as out:
            for c in todo:
                ks = chunks[c]
                trip = {}
                for sname, rows in systems.items():
                    for k in ks:
                        r = rows[k]
                        trip.setdefault((nfc(r["en"]), nfc(r["hyp"]), nfc(r["ref"])), None)
                uniq = list(trip)
                while True:
                    try:
                        sc = scorer([u[0] for u in uniq], [u[1] for u in uniq], [u[2] for u in uniq] if ref_based else None)
                        break
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if scorer.batch_size <= 1:
                            raise
                        scorer.batch_size = max(1, scorer.batch_size // 2)
                        scorer.max_batch_chars = scorer.batch_size * 700
                        print(f"OOM: batch -> {scorer.batch_size}", flush=True)
                        time.sleep(5)
                look = dict(zip(uniq, sc))
                res = {}
                for k in ks:
                    res[k[2]] = {s: look[(nfc(rows_[k]["en"]), nfc(rows_[k]["hyp"]), nfc(rows_[k]["ref"]))] for s, rows_ in systems.items()}
                out.write(json.dumps({"chunk": c, "scores": res}, ensure_ascii=False) + "\n")
                out.flush()
                nsc += len(uniq)
                print(f"[{metric}] {c}: {len(uniq)} unique pairs; total {nsc} in {time.time() - t0:.0f}s ({nsc / max(1, time.time() - t0):.0f}/s)", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
