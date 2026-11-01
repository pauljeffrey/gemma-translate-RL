#!/usr/bin/env python3
"""r4: constraint anchors v1 refit on clean T2 human text. Wraps scripts/build_anchors_calibration.py (unchanged code) and replaces only its three data loaders:
    C.flores_dev  -> H_cal v2 pairs (set "flores_dev", split label asserted)
    C.local_sets  -> SSA-MTE TRAIN references (12 languages, CC-BY-SA) and IbomMT TRAIN calibration half (efi, ibb; CC-BY-NC, OWNER EXCEPTION, tagged nc_exception later); smol_sent / ibom test / glot_story / salt / flores_moz / ntrex are NOT read
    C.documents   -> clean AfriScience-MT TRAIN papers (whole-paper English/reference, 6 languages) for the long length buckets of the repetition detector (no SmolDoc)
Text statistics only (entities, repetition, boilerplate, LID probability): the SSA-MTE / IbomMT references are NOT used with a learned metric here.

    python r4_anchors_v1.py --code <repo gemma_mt/> --cal refs/cal.jsonl --split heldout_split_cal_sel_v2.json --kept /workspace/work/out/kept --afs afriscience_clean_train.parquet --shaping clean/shaping_clean_v1.json --lid-model <model_v3.bin> --out clean/anchors_clean_v1.json --dump-human clean/human_units_clean.jsonl
"""
import argparse, collections, json, os, sys, unicodedata


def main():
    ap = argparse.ArgumentParser()
    for k in ("code", "cal", "split", "kept", "shaping", "out", "dump_human"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    ap.add_argument("--lid-model", default=None)
    ap.add_argument("--ibom-halves", default=None, help="dir with efi_calibration.jsonl / ibb_calibration.jsonl (r4_efi.py); owner exception for IbomMT")
    ap.add_argument("--afs", default=None, help="clean AfriScience train parquet (documents for the repetition floor); optional")
    a = ap.parse_args()
    sys.path.insert(0, a.code); sys.path.insert(0, os.path.join(a.code, "scripts")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pyarrow.parquet as pq
    from r4_qe_score import load_rows
    import anchors_common as C
    nfc = lambda s: " ".join(unicodedata.normalize("NFC", s).split())
    rows = load_rows(a.cal, "calibration", a.split)

    def flores_dev(isos, cache_dir=None, log=print):
        out = collections.defaultdict(list)
        for r in rows:
            if r["iso"] in isos:
                out[r["iso"]].append({"src": nfc(r["english"]), "ref": nfc(r["reference"]), "set": "flores_dev", "id": str(r["id"])})
        return out

    def local_sets(isos, test_sets, sets=(), log=print):
        out = collections.defaultdict(list)
        t = pq.read_table(os.path.join(a.kept, "ssa_mte_train.parquet"), columns=["english", "target", "iso"]).to_pydict()
        for k, (e, g, i) in enumerate(zip(t["english"], t["target"], t["iso"])):
            if i in isos and e.strip() and g.strip():
                out[i].append({"src": nfc(e), "ref": nfc(g), "set": "ssa_mte_train", "id": str(k)})
        for iso in ("efi", "ibb"):
            p = os.path.join(a.ibom_halves, f"{iso}_calibration.jsonl") if a.ibom_halves else ""
            if iso in isos and os.path.exists(p):
                for k, l in enumerate(open(p, encoding="utf-8")):
                    r = json.loads(l)
                    assert r["split"] == "calibration"
                    out[iso].append({"src": nfc(r["english"]), "ref": nfc(r["reference"]), "set": "ibom_mt_train", "id": str(k)})
        return out

    def documents(isos, test_sets, doc_files=(), log=print):
        out = collections.defaultdict(list)
        if not a.afs:
            return out
        t = pq.read_table(a.afs, columns=["paper_id", "sentence_id", "target_lang", "source", "target"]).to_pydict()
        papers = collections.defaultdict(list)
        for i in range(len(t["source"])):
            papers[(t["target_lang"][i], t["paper_id"][i])].append((int(t["sentence_id"][i]), nfc(t["source"][i]), nfc(t["target"][i])))
        for (iso, pid), v in sorted(papers.items()):
            if iso in isos:
                v.sort()
                out[iso].append({"src": " ".join(x[1] for x in v), "ref": " ".join(x[2] for x in v), "set": "afriscience_train_doc", "id": str(pid)})
        return out

    C.flores_dev, C.local_sets, C.documents = flores_dev, local_sets, documents
    import build_anchors_calibration as B
    B.C = C
    argv = ["--shaping", a.shaping, "--out", a.out, "--test-sets", "/nonexistent-clean-run", "--dump-human", a.dump_human]
    if a.lid_model:
        argv += ["--lid-model", a.lid_model]
    return B.main(argv)


if __name__ == "__main__":
    sys.exit(main())
