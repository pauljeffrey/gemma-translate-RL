#!/usr/bin/env python3
"""H_cal / H_sel split of the held-out FLORES+ dev ids and the human-reference pairs of H_cal (needed by gate rule CAL).

    python hcal.py split --devhalf-split out/stats/devhalf_split.json --out protected_index/heldout_split_cal_sel.json
    python hcal.py refs  --split protected_index/heldout_split_cal_sel.json --out refs_hcal [--isos amh yor ...]      # needs HF_TOKEN; writes BENCHMARK TEXT locally: never push this folder

NOTE 2026-10-29: the SPLIT USED BY THE PROJECT is now heldout_split_cal_sel_v2.json (agent R4, article level, H_cal 261 / H_sel 296, Amendment 3); pass it to `refs --split`. The `split`
subcommand below makes the superseded v1 (282/275, next sha256 bit) and is kept only for reproducibility.

Rule v1 (pre-registration section 4: "H is split again by sha256 parity into H_cal and H_sel"; the exact rule was not in the repo, so it is fixed here and written to the file): the held-out
ids already are the ids with odd sha256(str(id)) plus the 100 forced ids, so the same bit cannot be used again; the NEXT bit is used:
    id in H_cal  if (int(sha256(str(id)).hexdigest(), 16) >> 1) & 1 == 0, else in H_sel.
"""
import argparse, hashlib, json, os, sys


def h(i):
    return int(hashlib.sha256(str(i).encode()).hexdigest(), 16)


def sha(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def split(a):
    d = json.load(open(a.devhalf_split))
    held = sorted(d["heldout_ids"])
    cal = [i for i in held if (h(i) >> 1) & 1 == 0]
    sel = [i for i in held if (h(i) >> 1) & 1 == 1]
    out = {"rule": "H_cal if (int(sha256(str(id)).hexdigest(),16) >> 1) & 1 == 0 else H_sel; applied to the held-out ids of protected-2026-10-28.x devhalf split",
           "heldout_ids_sha256": sha(held), "h_cal": cal, "h_sel": sel, "n_cal": len(cal), "n_sel": len(sel), "h_cal_sha256": sha(cal), "h_sel_sha256": sha(sel),
           "forced_in_cal": sum(1 for i in d["forced_ids"] if i in set(cal)), "forced_in_sel": sum(1 for i in d["forced_ids"] if i in set(sel))}
    assert not set(cal) & set(sel) and len(cal) + len(sel) == len(held)
    json.dump(out, open(a.out, "w"))
    print({k: v for k, v in out.items() if k not in ("h_cal", "h_sel")})


def refs(a):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import pyarrow as pa, pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, HfApi
    try:
        from gemma_mt import languages as GL                  # PYTHONPATH=<code dir containing gemma_mt/> (run_chain.sh sets it)
    except ImportError:
        try:
            import gemma_languages as GL                        # the copy used on box 2
        except ImportError:
            sys.path.insert(0, "/path/to/repo/gemma_mt/gemma_mt"); import languages as GL
    from common import key_of
    sp = json.load(open(a.split)); cal = set(sp["h_cal"])
    tok = os.environ["HF_TOKEN"]
    api = HfApi(token=tok)
    have = {s.rfilename[4:-6] for s in api.dataset_info("openlanguagedata/flores_plus").siblings if s.rfilename.startswith("dev/")}
    def load(code):
        p = hf_hub_download("openlanguagedata/flores_plus", f"dev/{code}.jsonl", repo_type="dataset", token=tok, local_dir=a.tmp)
        d = {}
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if (r.get("text") or "").strip():
                    d[int(r["id"])] = r["text"].strip()
        os.remove(p); return d
    eng = load("eng_Latn")
    rows = {"english": [], "target": [], "iso": [], "key": []}
    for iso, L in GL.LANGUAGES.items():
        code = L.flores or L.flores_moz
        if not code or code not in have or (a.isos and iso not in a.isos):
            continue
        t = load(code)
        for i in sorted(cal):
            if i in t and i in eng:
                rows["english"].append(eng[i]); rows["target"].append(t[i]); rows["iso"].append(iso); rows["key"].append(key_of(eng[i], iso, t[i]))
    os.makedirs(a.out, exist_ok=True)
    pq.write_table(pa.table({"english": rows["english"], "target": rows["target"], "iso": rows["iso"], "key": pa.array(rows["key"], pa.uint64())}), a.out + "/pairs-0000.parquet", compression="zstd")
    import collections
    print(len(rows["iso"]), "human-reference pairs of H_cal,", len(set(rows["iso"])), "languages", dict(collections.Counter(rows["iso"]).most_common(3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split"); s.add_argument("--devhalf-split", required=True); s.add_argument("--out", required=True)
    r = sub.add_parser("refs"); r.add_argument("--split", required=True); r.add_argument("--out", required=True); r.add_argument("--tmp", default="/tmp/hcal_dl"); r.add_argument("--isos", nargs="*")
    a = ap.parse_args()
    (split if a.cmd == "split" else refs)(a)
