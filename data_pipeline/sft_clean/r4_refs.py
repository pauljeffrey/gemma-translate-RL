#!/usr/bin/env python3
"""r4: write the human-reference files of H_cal (role T2, calibration) and H_sel (role S, selection) for every RL language that has a FLORES+ dev file.

    HF_TOKEN=... python r4_refs.py --split heldout_split_cal_sel_v2.json --langs-dir /workspace/calib_r4/code --out /workspace/calib_r4/refs

Writes <out>/cal.jsonl and <out>/sel.jsonl (+ counts.json = numbers only). Rows: id, iso, english, reference, split ('calibration' | 'selection'), benchmark 'flores_plus_dev',
split_sha256 (of the id list). THIS IS BENCHMARK TEXT: it stays on the box and is never pushed, committed or quoted; counts.json is the only publishable output.
Every consumer (r4_calibrate_*.py) calls r4_common.load_rows(path, want_split) which refuses a file whose split label is not 'calibration' (T2 scripts) or 'selection' (S scripts).
"""
import argparse, collections, hashlib, json, os, sys


def sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--langs-dir", required=True, help="directory that contains the gemma_mt package (repo gemma_mt/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default="/workspace/calib_r4/work/flores_dl")
    ap.add_argument("--flores-dir", default=None, help="local copy of openlanguagedata/flores_plus (contains dev/<code>.jsonl); default: download with HF_TOKEN")
    a = ap.parse_args()
    sys.path.insert(0, a.langs_dir)                       # the repo package (gemma_mt.languages): same language table and FLORES codes as the old calibration
    import gemma_mt.languages as GL
    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    sp = json.load(open(a.split))
    ids = {"calibration": set(sp["h_cal"]), "selection": set(sp["h_sel"])}
    shas = {"calibration": sp["h_cal_sha256"], "selection": sp["h_sel_sha256"]}
    if a.flores_dir:
        have = {f[:-6] for f in os.listdir(os.path.join(a.flores_dir, "dev")) if f.endswith(".jsonl")}
    else:
        have = {s.rfilename[4:-6] for s in HfApi(token=tok).dataset_info("openlanguagedata/flores_plus").siblings if s.rfilename.startswith("dev/")}

    def load_one(code):
        p = os.path.join(a.flores_dir, "dev", f"{code}.jsonl") if a.flores_dir else hf_hub_download("openlanguagedata/flores_plus", f"dev/{code}.jsonl", repo_type="dataset", token=tok, local_dir=a.tmp)
        d = {}
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if (r.get("text") or "").strip():
                    d[int(r["id"])] = r["text"].strip()
        return d

    def load(code):
        if code == "twi_Latn":                            # as build_flores_dev_all / calibrate_qe_ensemble: the two Twi varieties alternate by id parity
            va, vb = load_one("twi_Latn_akua1239"), load_one("twi_Latn_asan1239")
            return {i: (va, vb)[i % 2].get(i) or (vb, va)[i % 2].get(i) for i in sorted(set(va) | set(vb)) if (va.get(i) or vb.get(i))}
        return load_one(code)

    eng = load("eng_Latn")
    os.makedirs(a.out, exist_ok=True)
    fh = {k: open(os.path.join(a.out, {"calibration": "cal", "selection": "sel"}[k] + ".jsonl"), "w", encoding="utf-8") for k in ids}
    counts = {k: collections.Counter() for k in ids}
    missing = []
    for iso in GL.SUPPORTED_RL:
        L = GL.LANGUAGES[iso]
        code = L.flores or L.flores_moz
        if not code or (code not in have and not (code == "twi_Latn" and "twi_Latn_akua1239" in have)):
            continue
        t = load(code)
        for k in ids:
            for i in sorted(ids[k]):
                if i in t and i in eng:
                    fh[k].write(json.dumps({"id": i, "iso": iso, "english": eng[i], "reference": t[i], "split": k, "benchmark": "flores_plus_dev", "split_sha256": shas[k]}, ensure_ascii=False) + "\n")
                    counts[k][iso] += 1
            if counts[k][iso] < len(ids[k]):
                missing.append((k, iso, len(ids[k]) - counts[k][iso]))
    for f in fh.values():
        f.close()
    out = {"split_file_sha256": hashlib.sha256(open(a.split, "rb").read()).hexdigest(), "n_ids": {k: len(v) for k, v in ids.items()},
           "n_languages": {k: len(c) for k, c in counts.items()}, "per_language": {k: dict(sorted(c.items())) for k, c in counts.items()}, "languages_with_missing_ids": missing}
    json.dump(out, open(os.path.join(a.out, "counts.json"), "w"), indent=1)
    print({k: (v if k != "per_language" else "...") for k, v in out.items()})


if __name__ == "__main__":
    main()
