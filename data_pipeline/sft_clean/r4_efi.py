#!/usr/bin/env python3
"""RE-ENABLED 2026-10-29 (evening): IbomMT is CC-BY-NC-4.0, used under an EXPLICIT OWNER EXCEPTION; every output derived from it is tagged nc_exception: ibom-mt (see r4_assemble.py).

r4: Efik (efi) and Ibibio (ibb) consensus-entry statistics from IbomMT TRAIN (CC-BY-NC-4.0, owner accepted, rows are tagged non_commercial), D-43 rule on clean data.

    python r4_efi.py --kept /workspace/work/out/kept --code <repo gemma_mt/> --out clean/efi_ibb_blocks.json --work clean/ibom_halves

IbomMT TRAIN pairs that survived the D2 protected-index filter (out/kept/ibom_train_{efi,ibb}.parquet) are split by D-43's rule sha1("calib-split-v1" + English) < 0.5 into a
CALIBRATION half (this script: len_ratio, diacritics, lexical, omission, GlotLID recall for efi) and a HOLD-OUT half (kept untouched for the model-dependent validation of the consensus
weight: scripts/validate_consensus.py on the new policy, WAITS-FOR-MODEL). IbomMT test and dev, SmolSent, SmolDoc are NOT used. Output: numbers only; the halves are written to --work
(on the box only).
"""
import argparse, hashlib, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kept", required=True); ap.add_argument("--code", required=True); ap.add_argument("--out", required=True); ap.add_argument("--work", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.code)
    import pyarrow.parquet as pq
    from gemma_mt.rl import calibrate_refs as C
    os.makedirs(a.work, exist_ok=True)
    res = {}
    for iso in ("efi", "ibb"):
        t = pq.read_table(os.path.join(a.kept, f"ibom_train_{iso}.parquet")).to_pydict()
        assert set(t["source_dataset"]) == {"howard-nlp/ibom-mt"}
        rows, seen = [], set()
        for e, g, k in zip(t["english"], t["target"], t["key"]):
            e, g = C.nfc(e.strip()), C.nfc(g.strip())
            if e and g and e not in seen:
                seen.add(e); rows.append({"english": e, "reference": g, "source": "ibom_mt_train", "doc": "", "idx": len(rows), "iso": iso, "key": str(k)})
        cal, hold = C.split_rows(rows)
        for name, rs in (("calibration", cal), ("holdout", hold)):
            with open(os.path.join(a.work, f"{iso}_{name}.jsonl"), "w", encoding="utf-8") as f:
                for r in rs:
                    f.write(json.dumps(dict(r, split=name), ensure_ascii=False) + "\n")
        st = C.reference_stats(cal)
        entry = {"n_pairs_total": len(rows), "n_calibration": len(cal), "n_holdout": len(hold), "split_rule": 'sha1("calib-split-v1" + English) < 0.5 (D-43)', "stats": st,
                 "calibration_half_sha256": hashlib.sha256(json.dumps(sorted(r["key"] for r in cal)).encode()).hexdigest(),
                 "holdout_half_sha256": hashlib.sha256(json.dumps(sorted(r["key"] for r in hold)).encode()).hexdigest()}
        if iso == "efi":
            from gemma_mt.rl.lid import GlotLIDClassifier
            lid = GlotLIDClassifier()
            top = lid.top([r["reference"] for r in cal], 1)
            lab = [x[0][0] if x else "" for x in top]
            entry["lid_efi_Latn_top1_recall"] = sum(1 for x in lab if str(x).startswith("efi")) / max(len(lab), 1)
        res[iso] = entry
        print(iso, {k: v for k, v in entry.items() if k != "stats"}, {k: (v if k in ("n",) else "...") for k, v in st.items()})
    res["_meta"] = {"licence": "CC-BY-NC-4.0 (IbomMT), owner decision 2026-10-28, tagged non_commercial", "weight_provisional": 0.5, "weight_rule": "D-43: within-prompt Spearman(consensus, reference chrF++) >= 0.25 -> 0.5; needs the new policy (WAITS-FOR-MODEL)"}
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
