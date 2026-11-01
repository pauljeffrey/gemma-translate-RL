#!/usr/bin/env python3
"""r4: QE scores of the H_cal human references and of the four corruptions (mismatch / copy / half / append), per model, resumable, any device.

    # CPU (box 2, 2 cores, ~1-3 pairs/s):   CUDA_VISIBLE_DEVICES="" taskset -c 8-9 nice -n 19 python r4_qe_score.py --refs refs/cal.jsonl --split heldout_split_cal_sel_v2.json --work qe_work \
    #                                         --model africomet --phase A
    # GPU (any, ~300 pairs/s on a 5070):    python r4_qe_score.py ... --model africomet --phase ALL --batch 128 --device cuda

Same corruption semantics as scripts/calibrate_qe_ensemble.py (make_sets: shuffle by Random(f"b1-{iso}"), mismatch = the reference of the next shuffled sentence, copy = the English source,
half = first half of the words, append = reference + next shuffled reference). Phases: A = human + mismatch (+ copy, scored ONCE for all languages: the copy of an English sentence is
the same pair in every language) = what t_fail / t_mid / t_great / AUC / reliability need; B = half + append (reliability_same_source, the ensemble report); ALL = A + B.
Output: <work>/<model>/<iso>.<set>.json = {"iso", "set", "model", "n", "ids": [...], "scores": [...]} (ids in the shuffled order), and <work>/<model>/COPY.json for the copy set.
Refuses a refs file whose rows are not split == 'calibration' with ids inside H_cal (v2).
"""
import argparse, hashlib, json, os, sys, time, unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {"africomet": "masakhane/africomet-qe-stl-1.1", "ssa_mtl": "McGill-NLP/ssa-comet-mtl"}
# priority: the languages the reward matters most for first (specialist set G+W, then the rest of the weak list, then the strong languages)
PRIORITY = "amh tir fon dyu kbp nus mos kam bam umb kmb kab wol ewe luo tum yor hau swh kin ibo afr bem kik lin lua lug mlg nso nya orm run sag sna som sot ssw tsn tso twi xho zul".split()


def load_rows(path, want_split, split_file):
    sp = json.load(open(split_file))
    ok = set(sp["h_cal"] if want_split == "calibration" else sp["h_sel"])
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    bad = [r for r in rows if r.get("split") != want_split or r["id"] not in ok]
    if bad:
        raise SystemExit(f"REFUSED: {len(bad)} rows of {path} are not split == {want_split!r} / not in the {want_split} id list")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--phase", choices=["A", "B", "ALL"], default="A")
    ap.add_argument("--languages", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--code", default=os.path.join(HERE, "..", "code"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--part", default="0/1", help="i/N: this process scores the languages with index %% N == i (several processes on one big GPU); the copy set is scored by part 0 only")
    a = ap.parse_args()
    sys.path.insert(0, a.code); sys.path.insert(0, os.path.join(a.code, "scripts"))
    import torch
    torch.set_num_threads(a.threads)
    from calibrate_qe_ensemble import make_sets
    from gemma_mt.rl.comet import CometScorer

    rows = load_rows(a.refs, "calibration", a.split)
    by = {}
    for r in rows:
        by.setdefault(r["iso"], []).append((unicodedata.normalize("NFC", r["english"]), unicodedata.normalize("NFC", r["reference"]), r["id"]))
    from gemma_mt.languages import QE_LANGUAGES
    langs = [l for l in PRIORITY if l in by and l in QE_LANGUAGES] + sorted(l for l in by if l in QE_LANGUAGES and l not in PRIORITY)
    if a.languages:
        langs = [l for l in langs if l in a.languages]
    pi, pn = (int(x) for x in a.part.split("/"))
    langs = langs[pi::pn]
    out = os.path.join(a.work, a.model); os.makedirs(out, exist_ok=True)
    sets_wanted = {"A": ("human", "mismatch"), "B": ("half", "append"), "ALL": ("human", "mismatch", "half", "append")}[a.phase]
    todo = [(l, s) for l in langs for s in sets_wanted if not os.path.exists(os.path.join(out, f"{l}.{s}.json"))]
    need_copy = pi == 0 and a.phase in ("A", "ALL") and not os.path.exists(os.path.join(out, "COPY.json"))
    print(f"{len(todo)} (language, set) jobs, copy={need_copy}, model {MODELS[a.model]}", flush=True)
    if not todo and not need_copy:
        return
    sc = CometScorer(MODELS[a.model], batch_size=a.batch, device=a.device)
    if need_copy:
        eng = {}
        for l in by:
            for e, _, i in by[l]:
                eng[i] = e
        ids = sorted(eng); t0 = time.time()
        s = sc([eng[i] for i in ids], [eng[i] for i in ids])
        json.dump({"set": "copy", "model": MODELS[a.model], "n": len(ids), "ids": ids, "scores": [round(float(x), 6) for x in s]}, open(os.path.join(out, "COPY.json"), "w"))
        print(f"copy {len(ids)} pairs {time.time() - t0:.0f}s", flush=True)
    cache = {}
    for l, s in todo:
        if l not in cache:
            pairs = [(e, t) for e, t, _ in by[l]]
            idl = {(e, t): i for e, t, i in by[l]}
            ms = make_sets(pairs, l, 997)
            cache[l] = (ms, [idl[(e, t)] for e, t in zip(ms["src"], ms["ref"])])
        ms, ids = cache[l]
        t0 = time.time()
        sc_ = sc(ms["src"], ms["sets"][s])
        json.dump({"iso": l, "set": s, "model": MODELS[a.model], "n": len(ids), "ids": ids, "scores": [round(float(x), 6) for x in sc_]}, open(os.path.join(out, f"{l}.{s}.json.tmp"), "w"))
        os.replace(os.path.join(out, f"{l}.{s}.json.tmp"), os.path.join(out, f"{l}.{s}.json"))
        print(f"{a.model} {l:4} {s:8} n={len(ids)} {time.time() - t0:5.0f}s  {len(ids) / (time.time() - t0):.2f} pairs/s", flush=True)


if __name__ == "__main__":
    main()
