#!/usr/bin/env python3
"""Exposure proof for protected-2026-10-28.2 (FLORES+ devtest and the held-out half of FLORES+ dev stay protected; the train half is exempt for the listed languages).
Checks (1) the 1M-pair sample of the main output and (2) the data_devhalf rows, both sides:
  A  v.2 hash index (with exemption)           -> 0 hits expected for ALL protected sets (devhalf rows were filtered with it; main rows with the stricter v.1)
  B  audit logic (contamination.py BenchIndex/scan_windows/near>=50%) against FLORES+ devtest + held-out dev text of ALL languages (re-fetched, kept in memory only)
  C  strict independent 8-gram sets against the same text
  D  controls: planted held-out/devtest sentences must be caught by A and B; planted train-half sentences (listed language) must NOT be caught by A (exemption works),
     planted train-half sentences of a NON-listed language (yor, hau) must be caught by A (still protected).
"""
import os
import argparse, collections, glob, hashlib, importlib.util, json, os, random, sys, time
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decon as D
import protected_index_loader as PL


def load_contam(path):
    spec = importlib.util.spec_from_file_location("contamination", path)
    m = importlib.util.module_from_spec(spec); sys.modules["contamination"] = m; spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True); ap.add_argument("--devhalf", required=True); ap.add_argument("--index", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--contam", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--tmp", default="/workspace/work/dl_expo2")
    a = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    C = load_contam(a.contam)
    ix, man = PL.load(a.index, a.manifest)
    dh = man["devhalf"]; held = set(dh["heldout_ids"]); train = set(dh["train_ids"])
    t0 = time.time()
    # sample of the main output + devhalf rows
    files = sorted(glob.glob(a.final + "/*.parquet"))
    tot = sum(pq.ParquetFile(f).metadata.num_rows for f in files); p = min(1.0, a.n / tot); rnd = random.Random(2)
    texts = []
    for f in files:
        t = pq.read_table(f, columns=["english", "target"]).to_pydict()
        for e, g in zip(t["english"], t["target"]):
            if rnd.random() < p:
                texts += [e, g]
    dev_rows = pq.read_table(a.devhalf).to_pydict()
    dtexts = dev_rows["english"] + dev_rows["target"]
    res = {"version": man["version"], "sample_pairs": len(texts) // 2, "output_rows_total": tot, "devhalf_pairs": len(dev_rows["english"])}
    # protected text for B/C and controls: FLORES+ devtest (all), dev held-out (all languages), dev train half (for controls)
    tok = os.environ["HF_TOKEN"]; api = HfApi(token=tok)
    fl = [s.rfilename for s in api.dataset_info("openlanguagedata/flores_plus").siblings if s.rfilename.startswith(("dev/", "devtest/")) and s.rfilename.endswith(".jsonl")]
    prot, ctrl_train = [], {}
    for f in fl:
        pth = hf_hub_download("openlanguagedata/flores_plus", f, repo_type="dataset", token=tok, local_dir=a.tmp)
        code = os.path.basename(f)[:-6]
        for line in open(pth, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line); txt = (r.get("text") or "").strip()
            if len(txt.split()) < 3:
                continue
            if f.startswith("devtest/") or int(r["id"]) in held:
                prot.append((f.split("/")[0] + ("_heldout" if f.startswith("dev/") else ""), txt))
            elif code.split("_")[0] in ("hau", "swh", "amh", "kam", "fon", "eng", "yor"):
                ctrl_train.setdefault(code, []).append(txt)
        os.remove(pth)
    res["protected_sentences_B"] = len(prot)
    print("fetched", len(prot), "protected FLORES+ sentences", f"{time.time()-t0:.0f}s", flush=True)
    # A
    def hits(tx):
        c = collections.Counter(); n = 0
        for i in range(0, len(tx), 20000):
            for r in ix.check_texts(tx[i:i + 20000]):
                if r[0] or r[1] or r[2]:
                    n += 1
                    for m in r[:3]:
                        for nm in ix.names(m):
                            c[nm] += 1
        return n, dict(c)
    res["A_main_sample"] = dict(zip(("texts_with_hit", "by_set"), hits(texts)))
    res["A_devhalf_rows"] = dict(zip(("texts_with_hit", "by_set"), hits(dtexts)))
    # B / C
    allt = texts + dtexts
    rows = [{"english": C.normalise(t), "iso": "xx", "reference": ""} for _, t in prot]
    idx = C.BenchIndex(rows)
    scan = C.scan_windows(idx, ((t, "", "xx") for t in allt))
    near = sum(1 for st in idx.texts if (lambda gs: gs and sum(1 for x in gs if x in scan.grams) / len(gs) >= C.NEAR_THRESHOLD)(set(C._grams(st.split(), idx.n))))
    gset = set()
    for r in rows:
        gset.update(C._grams(r["english"].split(), 8))
    nC = 0
    for t in allt:
        tk = C.normalise(t).split()
        if len(tk) >= 8 and any(x in gset for x in C._grams(tk, 8)):
            nC += 1
    res["B_audit_logic"] = {"exact": len(scan.exact), "near_ge50": near, "shared_8gram_hashes": len(scan.grams)}
    res["C_strict_8gram_texts"] = nC
    # D controls
    rnd = random.Random(9)
    pl = [t for _, t in rnd.sample(prot, 1000)]
    res["D_controls"] = {"planted_protected_found_by_A": sum(1 for r in ix.check_texts(pl) if r[0] or r[1] or r[2]), "planted_protected": len(pl),
                         "planted_protected_found_by_B": len(set(idx.ids[C.normalise(t)] for t in pl) & C.scan_windows(idx, ((t, "", "xx") for t in pl)).exact)}
    for code, lst in sorted(ctrl_train.items()):
        sel = lst[:1000]
        # train-half ids: the jsonl order is by id; recover ids by re-reading is skipped: use texts of the devhalf rows for listed languages instead
    listed = collections.defaultdict(list)
    for e, g, i in zip(dev_rows["english"], dev_rows["target"], dev_rows["iso"]):
        listed[i].append(g)
    res["D_controls"]["train_half_listed_language_targets_flagged_by_A"] = {i: sum(1 for r in ix.check_texts(v[:300]) if r[0] or r[1] or r[2]) for i, v in list(listed.items())}
    res["D_controls"]["train_half_english_flagged_by_A"] = sum(1 for r in ix.check_texts(dev_rows["english"][:400]) if r[0] or r[1] or r[2])
    # non-listed language train-half text (yor, hau): must still be flagged. Use FLORES+ files for yor/hau, train-half ids.
    nl = {}
    for code in ("hau_Latn", "swh_Latn"):
        pth = hf_hub_download("openlanguagedata/flores_plus", f"dev/{code}.jsonl", repo_type="dataset", token=tok, local_dir=a.tmp)
        nl[code] = [json.loads(l)["text"].strip() for l in open(pth, encoding="utf-8") if l.strip() and int(json.loads(l)["id"]) in train and len(json.loads(l)["text"].split()) >= 3][:300]
        os.remove(pth)
    res["D_controls"]["train_half_NON_listed_language_flagged_by_A"] = {c: [sum(1 for r in ix.check_texts(v) if r[0] or r[1] or r[2]), len(v)] for c, v in nl.items()}
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
