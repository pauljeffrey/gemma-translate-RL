#!/usr/bin/env python3
"""Reusable loader / filter for the versioned protected-set hash index (built by fetch_protected.py, 2026-10-28).

The index holds ONLY hashes (xxh64 of normalised sentences and of word 8-grams) plus protected-set bitmasks; no benchmark text. Use it to filter any text that
must not contain evaluation data: RL prompts, the science replay mix, reward-calibration sets, new SFT sources.

    from protected_index_loader import load, check_rows
    ix, manifest = load("protected_index.npz", "protected_manifest.json")          # verifies the sha256 of the file against the manifest, refuses on mismatch
    hits = ix.check_texts(["some text", ...])                                       # per text (r1_mask, r2_mask, r3_mask, near_ge50); 0 = clean
    ix.names(mask)                                                                  # protected sets behind a bitmask

    python protected_index_loader.py --seal  --index I.npz --manifest M.json          # writes version + file sha256 into the manifest (done once after the build)
    python protected_index_loader.py --verify --index I.npz --manifest M.json
    python protected_index_loader.py --index I.npz --manifest M.json --jsonl in.jsonl --fields text [--out clean.jsonl] [--report r.json]
        # drops a row when ANY field has ANY hit (strict rule: exact sentence, contained sentence, or a shared word 8-gram); the report counts removed rows per rule and set

Semantics = decon.py = gemma_mt/scripts/contam_audit_12_decontaminate.py (rules R1 exact, R2 contiguous, R3 8-gram; both normal forms).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decon as D  # noqa: E402

VERSION = "protected-2026-10-28.1"      # v.2 manifests carry their own version string (protected-2026-10-28.2) and the FLORES+ dev train-half exemption


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""):
            h.update(ch)
    return h.hexdigest()


def seal(index: str, manifest: str) -> dict:
    m = json.load(open(manifest))
    m["version"] = m.get("version") or VERSION          # devhalf.py writes protected-2026-10-28.N into the manifest
    m["index_file"] = os.path.basename(index)
    m["index_file_sha256"] = sha256_file(index)
    ix = D.Index.load(index)
    m["digest"] = ix.digest()
    json.dump(m, open(manifest, "w"), indent=1, ensure_ascii=False)
    return m


def load(index: str, manifest: str | None = None, verify: bool = True):
    m = json.load(open(manifest)) if manifest else None
    if verify and m is not None:
        want = m.get("index_file_sha256")
        if not want:
            raise RuntimeError("manifest is not sealed (no index_file_sha256); run --seal once")
        got = sha256_file(index)
        if got != want:
            raise RuntimeError(f"index file sha256 mismatch: {got} != {want}")
    ix = D.Index.load(index)
    dh = (m or {}).get("devhalf")
    if dh and verify is not False:
        ep = os.path.join(os.path.dirname(index), dh["exemption_file"])
        if sha256_file(ep) != dh["exemption_file_sha256"]:
            raise RuntimeError("exemption file sha256 mismatch")
        ix.exempt = D.Index.load(ep)
    return ix, m


def check_rows(ix, rows, fields, batch=5000):
    """Yield (row, hit_mask_by_rule) for every row; hit_mask_by_rule = (r1, r2, r3) OR-ed over the fields."""
    buf = []
    for r in rows:
        buf.append(r)
        if len(buf) >= batch:
            yield from _flush(ix, buf, fields)
            buf = []
    if buf:
        yield from _flush(ix, buf, fields)


def _flush(ix, buf, fields):
    res = [ix.check_texts([str(r.get(f) or "") for r in buf]) for f in fields]
    for i, r in enumerate(buf):
        m = [0, 0, 0]
        for rr in res:
            for k in range(3):
                m[k] |= rr[i][k]
        yield r, tuple(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--jsonl")
    ap.add_argument("--fields", nargs="*", default=["text"])
    ap.add_argument("--out")
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.seal:
        m = seal(a.index, a.manifest)
        print("sealed", m["version"], m["index_file_sha256"])
        return
    ix, m = load(a.index, a.manifest, verify=bool(a.manifest))
    if a.verify:
        print("ok", m["version"], m["index_file_sha256"])
        return
    rep = collections.Counter()
    kept = removed = 0
    fo = open(a.out, "w", encoding="utf-8") if a.out else None
    rules = ("R1_exact", "R2_contiguous", "R3_8gram")
    for r, m3 in check_rows(ix, (json.loads(l) for l in open(a.jsonl, encoding="utf-8") if l.strip()), a.fields):
        if any(m3):
            removed += 1
            for rule, mask in zip(rules, m3):
                for nm in ix.names(mask):
                    rep[f"{rule}|{nm}"] += 1
        else:
            kept += 1
            if fo:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    out = {"index_version": (m or {}).get("version"), "kept": kept, "removed": removed, "by_rule_and_set": dict(rep)}
    if a.report:
        json.dump(out, open(a.report, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
