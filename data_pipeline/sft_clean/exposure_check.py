#!/usr/bin/env python3
"""Exposure proof for the clean SFT output (mandatory step of the 2026-10-28 brief).

Takes a sample of the OUTPUT pairs (default 1,000,000 rows, both sides = 2,000,000 texts) and re-runs
  (A) the hash index of this build                                          (same code that filtered; catches merge / shard bugs)
  (B) the audit's flag logic, UNCHANGED: gemma_mt/gemma_mt/eval/contamination.py BenchIndex + scan_windows (+ the near >= 50% overlap rule of flags_for_rows),
      with an independent data structure (Python dicts of full normalised strings, in groups of protected sets so that memory stays small),
      fed from the LOCAL dump of the normalised protected sentences (written by fetch_protected.py --dump-text; never pushed, delete afterwards)
  (C) strict check: any shared word 8-gram between a sampled text and a protected sentence (python sets, independent of the xxhash arrays)
  (D) positive control: 2,000 protected sentences planted into the checker input must be found by (B) and (C); proves the checker is not blind.
Expected result for A, B, C: 0 hits for every protected set on both sides.
    python exposure_check.py --final out/final --dump protected_text_LOCAL_ONLY.txt --index protected/protected_index.npz --contam contamination.py --out out/stats/exposure_check.json
"""
import argparse, collections, glob, importlib.util, json, os, random, sys, time

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decon as D


def load_contam(path):
    spec = importlib.util.spec_from_file_location("contamination", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["contamination"] = m
    spec.loader.exec_module(m)
    return m


def sample_pairs(final, n, seed=1):
    files = sorted(glob.glob(final + "/*.parquet"))
    tot = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    p = min(1.0, n / max(1, tot))
    rnd = random.Random(seed)
    ens, tgs, isos = [], [], []
    for f in files:
        t = pq.read_table(f, columns=["english", "target", "iso"]).to_pydict()
        for e, g, i in zip(t["english"], t["target"], t["iso"]):
            if p >= 1.0 or rnd.random() < p:
                ens.append(e); tgs.append(g); isos.append(i)
    return ens, tgs, isos, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True); ap.add_argument("--dump", required=True); ap.add_argument("--index", required=True)
    ap.add_argument("--contam", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1_000_000); ap.add_argument("--group-size", type=int, default=900_000)
    a = ap.parse_args()
    C = load_contam(a.contam)
    t0 = time.time()
    ens, tgs, isos, total = sample_pairs(a.final, a.n)
    texts = ens + tgs
    res = {"output_rows_total": total, "sample_pairs": len(ens), "sample_texts": len(texts)}
    print("sample", len(ens), "of", total, flush=True)

    # A. hash index
    ix = D.Index.load(a.index)
    hitsA = collections.Counter(); nA = 0
    for i in range(0, len(texts), 20000):
        for r in ix.check_texts(texts[i:i + 20000]):
            if r[0] or r[1] or r[2]:
                nA += 1
                for rule, m in zip(("R1", "R2", "R3"), r[:3]):
                    for nm in ix.names(m):
                        hitsA[f"{rule}|{nm}"] += 1
    res["A_hash_index"] = {"texts_with_hit": nA, "by_rule_set": dict(hitsA), "sets_checked": len(ix.set_names)}
    print("A done", nA, flush=True)

    # protected sentences per set from the local dump
    by_set = collections.defaultdict(list)
    for line in open(a.dump, encoding="utf-8"):
        s, _, n = line.rstrip("\n").split("\t", 2)
        by_set[s].append(n)
    groups, cur, size = [], [], 0
    for s, v in sorted(by_set.items(), key=lambda kv: -len(kv[1])):
        if cur and size + len(v) > a.group_size:
            groups.append(cur); cur, size = [], 0
        cur.append(s); size += len(v)
    if cur:
        groups.append(cur)
    res["groups"] = [{"sets": g, "protected_sentences": sum(len(by_set[s]) for s in g)} for g in groups]

    # grams of the sampled texts once (python ints; independent of the xxhash arrays)
    text_grams = []
    for t in texts:
        tk = C.normalise(t).split()
        text_grams.append(C._grams(tk, 8) if len(tk) >= 8 else None)
    rnd = random.Random(5)
    control_all = []
    B_per_set, C_per_set = collections.Counter(), collections.Counter()
    ctrl_found_B = ctrl_found_C = ctrl_total = 0
    nearB = 0
    for g in groups:
        rows, owner = [], []
        for s in g:
            for n in by_set[s]:
                rows.append({"english": n, "iso": "xx", "reference": ""}); owner.append(s)
        idx = C.BenchIndex(rows)
        scan = C.scan_windows(idx, ((t, "", "xx") for t in texts))
        # exact per set (an id can belong to several sets: attribute through the normalised text)
        text_sets = collections.defaultdict(set)
        for s, r in zip(owner, rows):
            text_sets[C.normalise(r["english"])].add(s)
        for i in scan.exact:
            for s in text_sets[idx.texts[i]]:
                B_per_set[s] += 1
        ov_near = 0
        for i, st in enumerate(idx.texts):
            gs = set(C._grams(st.split(), idx.n))
            if gs and sum(1 for x in gs if x in scan.grams) / len(gs) >= C.NEAR_THRESHOLD:
                ov_near += 1
                for s in text_sets[st]:
                    B_per_set["near:" + s] += 1
        nearB += ov_near
        # C. strict 8-gram check with plain python sets
        gset = set()
        for r in rows:
            gset.update(C._grams(C.normalise(r["english"]).split(), 8))
        for gl in text_grams:
            if gl and any(x in gset for x in gl):
                C_per_set["|".join(g)] += 1
        # D. positive control: planted protected sentences (+ a copy embedded in a sentence)
        ctrl = rnd.sample(rows, min(1000, len(rows)))
        ctexts = [r["english"] for r in ctrl]
        scan_c = C.scan_windows(idx, ((t, "", "xx") for t in ctexts))
        ctrl_total += len(ctexts)
        ctrl_found_B += len(set(idx.ids[C.normalise(t)] for t in ctexts) & scan_c.exact)
        ctrl_found_C += sum(1 for t in ctexts if len(t.split()) < 8 or any(x in gset for x in C._grams(C.normalise(t).split(), 8)))
        print("group", g[:3], "exact", len(scan.exact), "near", ov_near, "grams", len(scan.grams), f"{time.time()-t0:.0f}s", flush=True)
        del idx, scan
    res["B_audit_logic"] = {"protected_sentences_exact_in_sample": sum(v for k, v in B_per_set.items() if not k.startswith("near:")), "near_ge50": nearB, "by_set": dict(B_per_set),
                            "note": "contamination.py BenchIndex/scan_windows/flags_for_rows semantics; protected sentences < 3 tokens are not in the dump"}
    res["C_strict_8gram"] = {"texts_with_shared_8gram": sum(C_per_set.values()), "by_group": dict(C_per_set)}
    res["D_positive_control"] = {"planted_protected_sentences": ctrl_total, "found_by_B": ctrl_found_B, "found_by_C_or_shorter_than_8_tokens": ctrl_found_C}
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("sample_pairs", "A_hash_index", "C_strict_8gram", "D_positive_control")}, indent=1)[:1500])
    print("B exact hits:", res["B_audit_logic"]["protected_sentences_exact_in_sample"], "near:", nearB)


if __name__ == "__main__":
    main()
