#!/usr/bin/env python
"""Per-sentence npz (ev_stats.py layout + bstats for spBLEU + iloop = the independent detector of E-08) from an e02_decode.py json (+ e02_comet.py json).  CPU only (venv_comet python, PYTHONPATH pylibs_none).
    python e02_mknpz.py DECODE.json COMET.json|none OUTDIR PREFIX [--flores DIR]    -> OUTDIR/PREFIX_<system>.npz for every system in the json
Independent detector: thresholds of reports/data/degeneration_audit_independent_2026-11-04/thresholds.json (z < -0.2536 lzma residual against a per-language log-log fit on the FLORES+ devtest references,
repeated char 8-gram share > 0.2345, repeated word 3-gram share > 0.1429), exactly the features of compute_features.py."""
import importlib.util, json, lzma, os, sys, unicodedata
import numpy as np
from sacrebleu.metrics import BLEU, CHRF

GUARDS = "/workspace/clean_rerun/tree/gemma_mt/gemma_mt/rl/guards.py"
spec = importlib.util.spec_from_file_location("guards_standalone", GUARDS); guards = importlib.util.module_from_spec(spec); spec.loader.exec_module(guards)
THR = {"z": -0.2535888440397546, "c8": 0.23451316190228919, "w3": 0.1428571428571429}
_LZ = [{"id": lzma.FILTER_LZMA2, "preset": 9, "dict_size": 1 << 17}]   # dict_size 128 KB: identical size for texts < 128 KB, 30x faster set-up than the 64 MB default of preset 9
nfc = lambda s: unicodedata.normalize("NFC", s)


def rep_share(items):
    return 0.0 if not items else 1.0 - len(set(items)) / len(items)


def feats(t):
    t = nfc(t or ""); b = t.encode("utf-8"); nb = len(b)
    lz = len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=_LZ)) if nb else 0
    s = " ".join(t.lower().split()); w = s.split()
    return nb, lz, rep_share([s[i:i + 8] for i in range(len(s) - 7)]), rep_share([tuple(w[i:i + 3]) for i in range(len(w) - 2)])


def main():
    dec, comet_p, outdir, prefix = sys.argv[1:5]
    flores = sys.argv[sys.argv.index("--flores") + 1] if "--flores" in sys.argv else "/workspace/eval_385m/bench/test_sets/flores_plus/devtest"
    d = json.load(open(dec)); items = d["items"]; os.makedirs(outdir, exist_ok=True)
    cm = json.load(open(comet_p)) if comet_p != "none" else {}
    sys.path.insert(0, os.environ.get("GEMMA_MT_TREE", "/workspace/eval_385m/gemma_mt_tree"))
    from gemma_mt import languages as L
    files = set(os.listdir(flores))
    def code(iso):
        c = getattr(L.LANGUAGES[iso], "flores", None)
        if c and f"{c}.jsonl" in files: return c
        return [f[:-6] for f in files if f.startswith(iso + "_")][0]
    # lzma log-log fit per language on ALL devtest references (the E-08 calibration is per benchmark-language cell on the references)
    coef = {}; allb, allc = [], []
    for iso in sorted(set(i[0] for i in items)):
        refs = [json.loads(l)["text"] for l in open(os.path.join(flores, code(iso) + ".jsonl"), encoding="utf-8")]
        fe = [feats(r) for r in refs]; fe = [f for f in fe if f[0] >= 20]
        lb = np.log([f[0] for f in fe]); lc = np.log([max(f[1], 1) for f in fe]); allb += list(lb); allc += list(lc)
        coef[iso] = np.polyfit(lb, lc, 1) if len(fe) >= 100 and lb.std() > 0 else None
    pooled = np.polyfit(allb, allc, 1)
    chrf = CHRF(word_order=2); bleu = BLEU(tokenize="flores200")
    refs_all = [nfc(i[2]) for i in items]
    for sysn, hs in d["hyp"].items():
        hyps = [nfc(h["hyp"]) for h in hs]
        stats = np.array(chrf._extract_corpus_statistics(hyps, [refs_all]), dtype=np.int32)
        bstats = np.array(bleu._extract_corpus_statistics(hyps, [refs_all]), dtype=np.int32)
        loop = np.array([bool(guards.is_repetitive(h["hyp"]) or guards.is_self_repetitive(h["hyp"])) for h in hs])
        iloop = np.zeros(len(hs), bool)
        for k, h in enumerate(hs):
            nb, lz, c8, w3 = feats(h["hyp"]); z = np.nan
            if nb >= 20:
                c = coef[items[k][0]]; c = pooled if c is None else c
                z = np.log(max(lz, 1)) - (c[0] * np.log(nb) + c[1])
            iloop[k] = bool((np.isfinite(z) and z < THR["z"]) or c8 > THR["c8"] or w3 > THR["w3"])
        cv = np.full(len(hs), np.nan)
        if sysn in cm: cv = np.array(cm[sysn]["comet"], dtype=np.float64)
        np.savez_compressed(os.path.join(outdir, f"{prefix}_{sysn}.npz"), system=f"{prefix}_{sysn}", bench=np.array(["flores_plus"] * len(items)), iso=np.array([i[0] for i in items]),
                            rid=np.array([str(i[3]) for i in items]), stats=stats, tstats=stats.copy(), bstats=bstats, finished=np.array([h["finished"] for h in hs]), loop=loop, iloop=iloop, comet=cv,
                            hyp_chars=np.array([len(h["hyp"]) for h in hs], dtype=np.int32), ref_chars=np.array([len(i[2]) for i in items], dtype=np.int32), ntok=np.array([h.get("ntok", 0) for h in hs], dtype=np.int32))
        print(prefix, sysn, len(hs), "finished", np.mean([h["finished"] for h in hs]).round(4), "loop", loop.mean().round(4), "iloop", iloop.mean().round(4), "chrF++ corpus", round(chrf._compute_f_score(stats.sum(0)), 2), flush=True)


if __name__ == "__main__":
    main()
