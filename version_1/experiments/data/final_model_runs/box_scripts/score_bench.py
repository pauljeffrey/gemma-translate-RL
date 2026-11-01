"""a1: per (system, benchmark, language, subset) metrics from the pipeline's predictions.jsonl: chrF++ (word_order 2, NFC), spBLEU (flores200), finished / prefix_ok / empty
rates and the GlotLID right-language rate (rl/lid.py). Subsets: all, clean (English sentence not in SFT text), no_pair (reference pair not seen)."""
import csv, glob, json, os, sys, unicodedata
from collections import defaultdict
from multiprocessing import Pool
sys.path.insert(0, "/workspace/sft_build/pylibs")
from gemma_mt.eval import metrics as M
from gemma_mt.rl import lid as L
from gemma_mt.languages import LANGUAGES

OUT = "/workspace/jobs/a1"
SUBS = {"all": lambda r: True, "clean": lambda r: r.get("contaminated") is False, "no_pair": lambda r: r.get("pair_seen") is False}

def one(args):
    key, rows = args
    system, bench, iso, sub = key
    hyps = [r["hyp"] for r in rows]; refs = [r["reference"] for r in rows]
    n = len(rows)
    rec = {"system": system, "benchmark": bench, "iso": iso, "subset": sub, "n": n,
           "chrf": M.chrf_pp(hyps, refs), "spbleu": M.spbleu(hyps, refs),
           "finished": sum(bool(r.get("finished", True)) for r in rows) / n,
           "prefix_ok": sum(r.get("prefix_status") == "ok" for r in rows) / n,
           "empty": sum(1 for h in hyps if not h.strip()) / n}
    lv = [r["_lid"] for r in rows if r["_lid"] is not None]
    rec["lid_n"] = len(lv)
    rec["lid_ok"] = (sum(v[0] for v in lv) / len(lv)) if lv else ""
    rec["lid_ok_sib"] = (sum(v[1] for v in lv) / len(lv)) if lv else ""
    return rec

def main():
    systems = {"rl": OUT + "/out_rl/predictions.jsonl", "sft": OUT + "/out_sft/predictions.jsonl", "sft320": OUT + "/out_sft320/predictions.jsonl", "rl320": OUT + "/out_rl320/predictions.jsonl"}
    only = sys.argv[1:] or list(systems)
    lidbin = glob.glob("/workspace/sft_build/hf/hub/models--cis-lmu--glotlid/snapshots/*/model_v3.bin")[0]
    clf = L.GlotLIDClassifier(path=lidbin); L.selftest(clf)
    known = set(clf.labels())
    acc = {}
    for iso, lg in LANGUAGES.items():
        own = tuple(x for x in lg.glotlid if x in known)
        acc[iso] = (set(own), set(own) | {x for x in L.siblings_of(own) if x in known}) if own else None
    allrecs = []
    for s in only:
        rows = [json.loads(l) for l in open(systems[s], encoding="utf-8")]
        print(s, len(rows), "rows", flush=True)
        # LID once per row (cache by text)
        texts = [r["hyp"] for r in rows]
        top = clf(texts)
        for r, t in zip(rows, top):
            a = acc.get(r["iso"])
            r["_lid"] = None if (a is None or not r["hyp"].strip()) else (t in a[0], t in a[1])
            if a is not None and not r["hyp"].strip(): r["_lid"] = (False, False)
        groups = defaultdict(list)
        for r in rows:
            for sub, f in SUBS.items():
                if r["benchmark"] in ("gatitos", "smol_doc"): continue
                v = f(r)
                if v: groups[(s, r["benchmark"], r["iso"], sub)].append(r)
        with Pool(4) as p:
            allrecs += p.map(one, sorted(groups.items()), chunksize=4)
    fields = ["system", "benchmark", "iso", "subset", "n", "chrf", "spbleu", "finished", "prefix_ok", "empty", "lid_n", "lid_ok", "lid_ok_sib"]
    path = OUT + "/bench_metrics_%s.csv" % "_".join(only)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in allrecs: w.writerow(r)
    print("wrote", path, len(allrecs))
main()
