"""R14 stage 1: NLLB-600M vs our final model, per language, chrF++ and spBLEU with paired bootstrap (sentence resampling).
Usage: score_stage1.py OUT_DIR   (reads /workspace/jobs/r14/out/*.jsonl, ours/predictions.jsonl)  -> CSVs in OUT_DIR
Segment-level sufficient statistics (sacrebleu) are summed over bootstrap resamples (multinomial weights), so 1,000 resamples are cheap."""
import csv, glob, json, os, sys, unicodedata
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sacrebleu.metrics import CHRF, BLEU
from nllb_langmap import EXACT, NLLB_MAP
from gemma_mt import languages as L

R = "/workspace/jobs/r14"
OUTD = sys.argv[1] if len(sys.argv) > 1 else R + "/out/tables"
os.makedirs(OUTD, exist_ok=True)
NB = 1000
CHRF_M, BLEU_M = CHRF(word_order=2), BLEU(tokenize="flores200")
nfc = lambda s: unicodedata.normalize("NFC", s)
BENCH_ORDER = ["flores_plus", "ntrex", "mafand", "smol_sent", "salt"]
SUBS = {"all": lambda r: True, "no_pair": lambda r: r["pair_seen"] is False, "clean": lambda r: r["contaminated"] is False}

FAMILY = {}
for fam, isos in {
    "Bantu": "swh swc kin run lug zul xho nbl nde ssw sna nya sot tsn nso ven tso lin tum bem umb kmb kik kam lua kqn lun loz kua her yao ndc vmw tsc seh nyn luy kon dua bas".split(),
    "Atlantic-Congo (non-Bantu)": "yor ibo efi ibb ish bin iso oke urh twi fat ewe fon gaa tiv bbj kbp mos dag sag wol ful".split(),
    "Mande": "bam dyu men".split(), "Afroasiatic": "hau amh tir som orm sid wal kab tzm shi".split(),
    "Nilo-Saharan": "luo ach nus knc".split(), "Creole / Pidgin": "pcm wes kri kea".split(), "Other (Indo-European, Austronesian)": "afr mlg".split()}.items():
    for i in isos: FAMILY[i] = fam
def group_of(iso):
    l = L.LANGUAGES[iso]
    if iso in L.QE_VALIDATED: return "qe_validated"
    if l.rl_consensus: return "consensus_only"
    if l.rl_eligible: return "blend"
    return "sft_only"

MODELS = {"n600": "nllb", "n33": "nllb3b3"}
def load_nllb(prefix, cfg):
    d = {}
    for p in glob.glob(f"{R}/out/{prefix}_{cfg}_*.jsonl"):
        for l in open(p):
            x = json.loads(l); d[(x["iso"], x["english"])] = x["hyp"]
    return d

rows = [json.loads(l) for l in open(R + "/out/bench_rows.jsonl")]
ours = {}
pp = R + "/ours/predictions.jsonl"
if os.path.exists(pp):
    for l in open(pp):
        x = json.loads(l); ours[(x["benchmark"], x["iso"], str(x["id"]))] = x
print("rows", len(rows), "ours", len(ours), flush=True)
cfgs = {f"{m}_{c}": load_nllb(pre, c) for m, pre in MODELS.items() for c in ["beam4", "greedy", "beam4nr4"]}
print({c: len(v) for c, v in cfgs.items()}, flush=True)

def seg_stats(metric, hyps, refs):
    return np.array(metric._extract_corpus_statistics([nfc(h) for h in hyps], [[nfc(r) for r in refs]]), dtype=np.float64)
def score(metric, st):
    return metric._compute_score_from_stats(st).score

rng = np.random.default_rng(14)
out_rows, boot = [], {}
by = defaultdict(list)
for r in rows: by[(r["benchmark"], r["iso"])].append(r)
ONLY = os.environ.get("BENCHES")
for (b, iso), rs in sorted(by.items()):
    if ONLY and b not in ONLY.split(","): continue
    refs = [r["reference"] for r in rs]
    systems = {}
    for c, d in cfgs.items():
        h = [d.get((iso, r["english"])) for r in rs]
        if any(x is None for x in h): continue
        systems["nllb_" + c] = h
    if b != "flores_plus_dev":
        o = [ours.get((b, iso, str(r["id"]))) for r in rs]
        if all(x is not None for x in o): systems["ours"] = [x["hyp"] for x in o]
    if not systems: continue
    st = {k: (seg_stats(CHRF_M, h, refs), seg_stats(BLEU_M, h, refs)) for k, h in systems.items()}
    for sub, f in SUBS.items():
        idx = np.array([i for i, r in enumerate(rs) if f(r)])
        if len(idx) < 20 or (b == "flores_plus_dev" and sub != "all"): continue
        rec = {"benchmark": b, "iso": iso, "subset": sub, "n": len(idx)}
        for k, (sc, sb) in st.items():
            rec[k + "_chrf"] = score(CHRF_M, sc[idx].sum(0)); rec[k + "_spbleu"] = score(BLEU_M, sb[idx].sum(0))
        W = rng.multinomial(len(idx), np.ones(len(idx)) / len(idx), size=NB).astype(np.float64)
        bs = {}
        for k, (sc, sb) in st.items():
            A, Bm = W @ sc[idx], W @ sb[idx]
            bs[k] = (np.array([score(CHRF_M, a) for a in A]), np.array([score(BLEU_M, a) for a in Bm]))
        for k in st:
            if k == "ours": continue
            if "ours" in st:
                for mi, m in enumerate(["chrf", "spbleu"]):
                    d = bs[k][mi] - bs["ours"][mi]
                    rec[f"{k}_d_{m}"] = rec[f"{k}_{m}"] - rec[f"ours_{m}"]
                    rec[f"{k}_d_{m}_lo"], rec[f"{k}_d_{m}_hi"] = np.percentile(d, [2.5, 97.5])
                    boot[(b, iso, sub, k, m)] = d
        out_rows.append(rec)
    print(b, iso, len(out_rows), flush=True)

for r in out_rows:
    r["language"] = L.LANGUAGES[r["iso"]].name; r["group"] = group_of(r["iso"]); r["family"] = FAMILY.get(r["iso"], "?"); r["nllb_code"] = EXACT[r["iso"]]
fields = sorted({k for r in out_rows for k in r}, key=lambda k: (k not in ("iso", "language", "group", "family", "nllb_code", "benchmark", "subset", "n"), k))
with open(f"{OUTD}/nllb600m_per_language_benchmark.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in out_rows: w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

# macro bootstrap per (benchmark, subset, set)
macro = []
sets = {"all": lambda iso: True}
for g in ["qe_validated", "blend", "consensus_only", "sft_only"]: sets[g] = (lambda g: lambda iso: group_of(iso) == g)(g)
for fam in sorted(set(FAMILY.values())): sets["fam:" + fam] = (lambda fam: lambda iso: FAMILY.get(iso) == fam)(fam)
for (cfgk, m) in [(f"nllb_{a}_{b}", mm) for a in MODELS for b in ["beam4", "greedy", "beam4nr4"] for mm in ["chrf", "spbleu"]]:
    for b in BENCH_ORDER:
        for sub in SUBS:
            for sname, sf in sets.items():
                isos = [r["iso"] for r in out_rows if r["benchmark"] == b and r["subset"] == sub and sf(r["iso"]) and (b, r["iso"], sub, cfgk, m) in boot]
                if not isos: continue
                D = np.mean([boot[(b, i, sub, cfgk, m)] for i in isos], axis=0)
                nl = np.mean([r[f"{cfgk}_{m}"] for r in out_rows if r["benchmark"] == b and r["subset"] == sub and r["iso"] in isos])
                ou = np.mean([r[f"ours_{m}"] for r in out_rows if r["benchmark"] == b and r["subset"] == sub and r["iso"] in isos])
                macro.append({"benchmark": b, "subset": sub, "set": sname, "system": cfgk, "metric": m, "n_langs": len(isos), "nllb_macro": round(nl, 2), "ours_macro": round(ou, 2),
                              "delta": round(nl - ou, 2), "lo": round(float(np.percentile(D, 2.5)), 2), "hi": round(float(np.percentile(D, 97.5)), 2)})
with open(f"{OUTD}/nllb600m_macro.csv", "w", newline="") as f:
    if macro:
        w = csv.DictWriter(f, fieldnames=list(macro[0])); w.writeheader(); w.writerows(macro)
print("wrote tables", len(out_rows), len(macro))
