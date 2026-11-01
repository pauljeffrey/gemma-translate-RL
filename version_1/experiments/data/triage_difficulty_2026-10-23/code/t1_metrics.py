"""T1: per-chunk quality table. For every unit (benchmark, language, k consecutive sentences) and model: chrF++ of the JOINED hypothesis against the JOINED reference
(sacrebleu CHRF(word_order=2), NFC, sentence level, joined with a single space), mean of the existing per-sentence AfriCOMET (masakhane/africomet-stl-1.1) and SSA-COMET-MTL scores
(only when every sentence of the chunk has a score), flags of our models, and agreement chrF++ (model hypothesis as hypothesis, NLLB-3.3B beam-4 output as pseudo-reference).
usage: t1_metrics.py OUT_DIR [benchmark ...]   (2 worker processes, run under nice)"""
import sys, os, gzip, json, pickle, unicodedata, collections, csv, time
sys.path[:0] = ["/workspace/sft_build/pylibs"]
import numpy as np
from sacrebleu.metrics import CHRF
from multiprocessing import Pool
OUT = sys.argv[1]; BENCH = sys.argv[2:] or ["flores_plus", "ntrex", "mafand", "smol_sent", "glot_story", "salt"]
os.makedirs(OUT, exist_ok=True)
nfc = lambda s: unicodedata.normalize("NFC", s)
E2 = "/workspace/jobs/e2/scores/"
FLORES_MODELS = ["rl", "sft", "a2w", "live250", "nllb3b3", "nllb600m", "tpsy08", "tpsy2b"]
OTHER_MODELS = ["rl", "sft", "nllb3b3"]
AGREE_WITH = "nllb3b3"
AGREE_OF = ["rl", "sft", "a2w", "live250", "nllb600m", "tpsy08"]
chrf = CHRF(word_order=2)

# ---------------- load data -------------------------------------------------------------------------------------------------
t0 = time.time()
H = collections.defaultdict(dict)        # (b,iso) -> model -> {id: hyp}
REF = collections.defaultdict(dict); PS = collections.defaultdict(dict); EN = collections.defaultdict(dict)
FL = collections.defaultdict(dict)       # (b,iso) -> model -> id -> (fin, loop, lid, rep3, empty)
COMET = collections.defaultdict(lambda: collections.defaultdict(dict)); SSA = collections.defaultdict(lambda: collections.defaultdict(dict))
def e1_load(model, fn):
    for l in gzip.open(fn, "rt"):
        r = json.loads(l); key = (r["b"], r["iso"])
        if r["b"] not in BENCH: continue
        H[key].setdefault(model, {})[r["id"]] = nfc(r["hyp"])
        REF[key][r["id"]] = nfc(r["ref"]); PS[key][r["id"]] = bool(r["pair_seen"]); EN[key][r["id"]] = r["en"]
        FL[key].setdefault(model, {})[r["id"]] = (r["fin"], r["loop"], r["lid"], r["rep3"], r["empty"])
if any(b != "flores_plus" for b in BENCH):
    e1_load("sft", "/workspace/jobs/e1/sys_sft.jsonl.gz"); e1_load("rl", "/workspace/jobs/e1/sys_rl.jsonl.gz")
    for fn in ("afri_ref", "ssa_ref"):
        for l in open(f"/workspace/jobs/e1/neural/{fn}.jsonl"):
            r = json.loads(l); b, iso = r["chunk"].split("|")
            if b not in BENCH: continue
            for i, d in r["scores"].items():
                for m, v in d.items():
                    (COMET if fn == "afri_ref" else SSA)[(b, iso)][m][i] = v
    # NLLB-3.3B on ntrex / mafand / salt / smol_sent (R14, beam 4), joined on (language, English text)
    nl = {}
    for fn in ("nllb3b3_beam4_ntrex.jsonl", "nllb3b3_beam4_mafand+salt+smol_sent.jsonl"):
        for l in open("/workspace/jobs/r14/out/" + fn):
            r = json.loads(l); nl.setdefault((r["iso"], r["english"]), nfc(r["hyp"]))
    for key, en in EN.items():
        if key[0] == "glot_story": continue
        got = {i: nl[(key[1], e)] for i, e in en.items() if (key[1], e) in nl}
        if got: H[key]["nllb3b3"] = got
if "flores_plus" in BENCH:
    refs = {}
    for name in FLORES_MODELS:
        S = pickle.load(open(E2 + name + ".pkl", "rb"))
        for iso, d in S.items():
            key = ("flores_plus", iso)
            H[key][name] = {i: nfc(h) for i, h in zip(d["ids"], d["hyp"])}
            if name == "rl":
                REF[key] = {i: nfc(r) for i, r in zip(d["ids"], d["ref"])}; PS[key] = {i: bool(p) for i, p in zip(d["ids"], d["pair_seen"])}
                EN[key] = dict(zip(d["ids"], d["src"]))
        for kind, store in (("comet", COMET), ("ssa", SSA)):
            p = E2 + f"{name}.{kind}.pkl"
            if os.path.exists(p):
                for iso, dd in pickle.load(open(p, "rb")).items(): store[("flores_plus", iso)][name] = dict(dd)
        del S
    # flags for flores come from e1 (sft, rl only)
    for model, fn in (("sft", "sys_sft.jsonl.gz"), ("rl", "sys_rl.jsonl.gz")):
        for l in gzip.open("/workspace/jobs/e1/" + fn, "rt"):
            r = json.loads(l)
            if r["b"] == "flores_plus": FL[("flores_plus", r["iso"])].setdefault(model, {})[r["id"]] = (r["fin"], r["loop"], r["lid"], r["rep3"], r["empty"])
extra = "/workspace/jobs/t1/comet_extra"
if os.path.isdir(extra):
    for fn in os.listdir(extra):
        if not fn.endswith(".jsonl"): continue
        for l in open(os.path.join(extra, fn)):
            r = json.loads(l); store = COMET if r["metric"] == "afri" else SSA
            store[(r["b"], r["iso"])][r["model"]][r["id"]] = r["score"]
print("loaded", len(H), "groups in", round(time.time() - t0), "s", flush=True)

units = collections.defaultdict(list)
for l in gzip.open("/workspace/jobs/t1/units.jsonl.gz", "rt"):
    u = json.loads(l)
    if u["b"] in BENCH: units[(u["b"], u["iso"])].append(u)

def work(key):
    b, iso = key
    models = FLORES_MODELS if b == "flores_plus" else OTHER_MODELS
    rows = []
    hy, ref, ps = H[key], REF[key], PS[key]
    for u in units[key]:
        ids = u["ids"]
        r = {"b": b, "iso": iso, "k": u["k"], "uid": u["uid"], "ekey": u["ekey"], "n_sent": len(ids), "n_pair_seen": int(sum(ps[i] for i in ids))}
        rtxt = " ".join(ref[i] for i in ids); r["ref_chars"] = len(rtxt)
        joined = {}
        for m in models:
            hm = hy.get(m)
            if hm is None or any(i not in hm for i in ids): continue
            h = " ".join(hm[i] for i in ids); joined[m] = h
            r["chrf_" + m] = chrf.sentence_score(h, [rtxt]).score if rtxt else np.nan
            r["hchars_" + m] = len(h)
            for nm, store in (("comet", COMET[key]), ("ssa", SSA[key])):
                sm = store.get(m)
                if sm is not None and all(i in sm for i in ids): r[nm + "_" + m] = float(np.mean([sm[i] for i in ids]))
        for m in ("sft", "rl"):
            f = FL[key].get(m)
            if f is not None and all(i in f for i in ids):
                fs = [f[i] for i in ids]
                r["fin_" + m] = int(all(x[0] for x in fs)); r["loop_" + m] = int(any(x[1] for x in fs)); r["lid_" + m] = int(all(x[2] for x in fs))
                r["rep3_" + m] = int(any(x[3] for x in fs)); r["empty_" + m] = int(any(x[4] for x in fs))
        if AGREE_WITH in joined:
            for m in AGREE_OF:
                if m in joined: r["agree_" + m] = chrf.sentence_score(joined[m], [joined[AGREE_WITH]]).score
        rows.append(r)
    return key, rows

if __name__ == "__main__":
    keys = sorted(units)
    allrows = collections.defaultdict(list)
    with Pool(2) as p:
        for n, (key, rows) in enumerate(p.imap_unordered(work, keys)):
            allrows[key[0]] += rows
            if n % 10 == 0: print(n, len(keys), key, len(rows), round(time.time() - t0), "s", flush=True)
    import pandas as pd
    for b, rows in allrows.items():
        df = pd.DataFrame(rows).sort_values(["iso", "k", "uid"]); df.to_csv(f"{OUT}/chunks_{b}.csv.gz", index=False, float_format="%.5g")
        print(b, df.shape)
